// src/metal/post/ssr.rs
//
// Screen-space reflections: the reflection targets the SSR and ray-traced
// resolves write, the roughness-aware blur + composite that blends them over
// the scene, and where the resolve's inputs come from this frame. The resolve
// itself -- its pipeline and its draw -- is written once in
// `concinnity_core::render::post::ssr` and reaches Metal through
// `MtlPostDevice`.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::gfx::render_types;
use concinnity_core::gfx::ssr::SsrSettings;
use concinnity_core::render::post::ssr::{SsrInputs, SsrPass};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice as _, MTLLoadAction, MTLPixelFormat, MTLRenderPipelineState, MTLTexture,
    MTLTextureUsage,
};

use crate::metal::context::MtlContext;
use crate::metal::descriptors::TextureDesc;
use crate::metal::encode::RenderEncode;
use crate::metal::post::fullscreen::{
    FullscreenBlend, FullscreenPass, PassTimer, build_slang_fullscreen_pipeline,
    set_fragment_sampler_range,
};
use crate::metal::post::post_device::MtlPostPipeline;
use crate::metal::slang_builtins::{REFLECTION_BLUR, REFLECTION_COMPOSITE};

// The shared resolve, holding Metal's own pipeline handle.
pub(crate) type MtlSsrPass = SsrPass<MtlPostPipeline>;

// All screen-space-reflection feature state grouped into one unit: the
// resolved tunables, the reflection targets, and the pipelines that fill them.
// `targets` is `Some` when SSR, SSGI, *or* RT reflections are on (they share
// the G-buffer pre-pass output and RT reuses `targets.reflection`); `settings`
// and `resolve` are `Some` only when SSR itself is on.
pub(crate) struct SsrState {
    pub settings: Option<SsrSettings>,
    pub targets: Option<SsrTargets>,
    pub resolve: Option<MtlSsrPass>,
    // Roughness-aware blur + composite of the reflection target over the scene.
    // Shared by the SSR and RT-reflection resolves (both write the reflection
    // target, then run this). Built whenever the reflection targets exist.
    pub composite_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // First half of the composite: the roughness blur, run at reduced resolution
    // into `SsrTargets::blur`. Built alongside `composite_pipeline`.
    pub blur_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // Per-axis divisor the reflection blur target is sized by, resolved from the
    // world's `reflection_blur_resolution`. Held so a resize / live rebuild
    // recreates the blur target at the same reduced resolution.
    pub blur_scale: u32,
}

// Pipelines

// Build the reflection composite pipeline: the full-resolution second pass that
// lerps the sharp reflection against the upsampled half-res blur by roughness
// and composites it over the scene, writing the `RGBA16Float` scene output the
// SSR / RT resolve used to write directly. Shared by both reflection paths.
pub(crate) fn build_reflection_composite_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_slang_fullscreen_pipeline(
        device,
        &REFLECTION_COMPOSITE,
        MTLPixelFormat::RGBA16Float,
        FullscreenBlend::Replace,
        hot_reload,
    )
}

// Build the reflection blur pipeline: the reduced-resolution first pass that
// weight-averages the reflection target over the roughness cone into the blur
// target the composite then upsamples. The expensive multi-tap blur runs here.
pub(crate) fn build_reflection_blur_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_slang_fullscreen_pipeline(
        device,
        &REFLECTION_BLUR,
        MTLPixelFormat::RGBA16Float,
        FullscreenBlend::Replace,
        hot_reload,
    )
}

// Targets

// The targets the reflection resolves and their composite write. The
// view-space normal / linear depth / roughness they read come from the unified
// G-buffer pre-pass (`metal/post/gbuffer.rs`). Single-sample, full render
// resolution except the blur; created when a reflection path is enabled and
// rebuilt with the HDR targets on resize.
pub(crate) struct SsrTargets {
    // Reflection target (`RGBA16Float`): the SSR / RT resolve writes reflected
    // radiance in `.rgb` and the Fresnel/gloss composite weight in `.a` here,
    // and the reflection composite blurs + composites it into `output`.
    pub reflection: Retained<ProtocolObject<dyn MTLTexture>>,
    // Scene with reflections composited in. Becomes the scene color the TAA /
    // bloom / composite passes consume when SSR or RT reflections are on.
    pub output: Retained<ProtocolObject<dyn MTLTexture>>,
    // Reduced-resolution roughness blur of `reflection` (the blur pass writes it,
    // the composite pass upsamples it). Sized at render / REFLECTION_BLUR_SCALE.
    pub blur: Retained<ProtocolObject<dyn MTLTexture>>,
}

// Create or recreate the reflection + resolve-output targets at `width`x`height`,
// plus the reduced-resolution blur target. `blur_scale` is the per-axis
// render-resolution divisor for the roughness blur pass (resolved from the
// world's `reflection_blur_resolution`): the blur is low-frequency (a widening
// glossy cone), so running it reduced and bilinear-upsampling in the composite
// is visually free; mirrors stay sharp because the composite lerps in the
// FULL-RES reflection for low roughness (see reflection_composite.metal).
pub(crate) fn create_ssr_targets(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    width: u32,
    height: u32,
    blur_scale: u32,
) -> Result<SsrTargets, String> {
    let blur_scale = blur_scale.max(1);
    let make_at = |w: usize, h: usize| -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
        let desc = TextureDesc {
            format: MTLPixelFormat::RGBA16Float,
            width: w,
            height: h,
            usage: MTLTextureUsage(MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0),
            ..Default::default()
        }
        .build();
        device.newTextureWithDescriptor(&desc)
    };
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;
    let bw = (width / blur_scale).max(1) as usize;
    let bh = (height / blur_scale).max(1) as usize;
    let reflection = make_at(w, h).ok_or("failed to create reflection texture")?;
    let output = make_at(w, h).ok_or("failed to create SSR output texture")?;
    let blur = make_at(bw, bh).ok_or("failed to create reflection blur texture")?;
    Ok(SsrTargets {
        reflection,
        output,
        blur,
    })
}

// Encoders

impl MtlContext {
    // Encode the SSR resolve into the reflection target, then blur and composite
    // it over `hdr_resolve` into the targets' `output`. Runs after the main pass;
    // only called when SSR is on.
    pub(in crate::metal) fn encode_ssr_resolve(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        ssr_params: &render_types::SsrParams,
    ) -> Result<u32, String> {
        // The pre-pass channels are pool-owned, so they are fetched here rather
        // than cached: a pool rebuild repacks every slot.
        let (Some(targets), Some(resolve), Some(normal_depth), Some(roughness)) = (
            &self.ssr.targets,
            &self.ssr.resolve,
            self.gbuffer_normal_depth(),
            self.gbuffer_roughness(),
        ) else {
            return Ok(0);
        };
        resolve.encode(
            &self.post_device(),
            cmd_buf,
            SsrInputs {
                target: targets.reflection.as_ref(),
                scene: self.hdr_targets.hdr_resolve.as_ref(),
                normal_depth,
                roughness,
                // Always valid: a gray fallback when no EnvironmentMap is bound,
                // which `SsrParams.prefilter_mip_count == 0` tells the shader to
                // ignore.
                prefilter: self.env_map.prefilter.as_ref(),
            },
            ssr_params,
        )?;
        self.encode_reflection_composite(cmd_buf)?;
        Ok(0)
    }

    // Blur the reflection target by surface roughness and composite it over
    // `hdr_resolve` into `ssr_targets.output`. Shared by the SSR and
    // RT-reflection resolves: both write the reflection target first, then call
    // this. A no-op (leaves `output` untouched) when the composite pipeline or
    // G-buffer is absent, which only happens when no reflection path is active.
    pub(in crate::metal) fn encode_reflection_composite(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
    ) -> Result<(), String> {
        let (targets, composite_ps, blur_ps, gb_normal_depth, gb_roughness) = match (
            &self.ssr.targets,
            &self.ssr.composite_pipeline,
            &self.ssr.blur_pipeline,
            self.gbuffer_normal_depth(),
            self.gbuffer_roughness(),
        ) {
            (Some(t), Some(cp), Some(bp), Some(n), Some(r)) => (t, cp, bp, n, r),
            _ => return Ok(()),
        };
        // Pass 1: the roughness blur, at reduced resolution into `blur`. Times the
        // span start; the composite below times its end (both under one slot).
        self.fullscreen_pass(
            cmd_buf,
            FullscreenPass {
                target: targets.blur.as_ref(),
                load: MTLLoadAction::DontCare,
                timer: PassTimer::First(crate::metal::pass_timing::PassId::ReflectionComposite),
                pipeline: blur_ps,
                label: "reflection blur",
            },
            |enc| {
                enc.set_fragment_texture(targets.reflection.as_ref(), 0);
                enc.set_fragment_texture(gb_roughness, 1);
                set_fragment_sampler_range(enc, &self.post_sampler, 0, 2);
            },
        )?;
        // Pass 2: lerp the sharp full-res reflection against the upsampled blur by
        // roughness, then composite over the scene into `output`.
        self.fullscreen_pass(
            cmd_buf,
            FullscreenPass {
                target: targets.output.as_ref(),
                load: MTLLoadAction::DontCare,
                timer: PassTimer::Last(crate::metal::pass_timing::PassId::ReflectionComposite),
                pipeline: composite_ps,
                label: "reflection composite",
            },
            |enc| {
                enc.set_fragment_texture(targets.reflection.as_ref(), 0);
                enc.set_fragment_texture(self.hdr_targets.hdr_resolve.as_ref(), 1);
                enc.set_fragment_texture(gb_normal_depth, 2);
                enc.set_fragment_texture(gb_roughness, 3);
                enc.set_fragment_texture(targets.blur.as_ref(), 4);
                set_fragment_sampler_range(enc, &self.post_sampler, 0, 5);
            },
        )?;
        Ok(())
    }
}

//! Screen-space reflections: the reflection target the SSR and ray-traced
//! resolves write, and where the resolve's inputs come from this frame. The
//! resolve itself -- its pipeline and its draw -- is written once in
//! `concinnity_core::render::post::ssr` and reaches Metal through
//! `MtlPostDevice`; the composite that blends the target over the scene is in
//! `reflection_composite.rs`.
#![deny(unsafe_op_in_unsafe_fn)]

use crate::metal::error::allocation_failed;
use concinnity_core::gfx::render_types;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::post::ssr::settings::SsrSettings;
use concinnity_core::render::post::ssr::{SsrInputs, SsrPass};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice as _, MTLPixelFormat, MTLTexture, MTLTextureUsage};

use crate::metal::context::MtlContext;
use crate::metal::descriptors::TextureDesc;
use crate::metal::post::post_device::MtlPostPipeline;
use crate::metal::post::reflection_composite::MtlReflectionCompositePass;

// The shared resolve, holding Metal's own pipeline handle.
pub(crate) type MtlSsrPass = SsrPass<MtlPostPipeline>;

// All screen-space-reflection feature state grouped into one unit: the
// resolved tunables, the reflection target, the resolve that fills it, and the
// composite that blends it over the scene. `reflection` is `Some` when SSR,
// SSGI, *or* RT reflections are on (RT writes it too); `settings` and `resolve`
// are `Some` only when SSR itself is on, and `composite` when SSR or RT is.
pub(crate) struct SsrState {
    pub settings: Option<SsrSettings>,
    // Reflection target (`RGBA16Float`): the SSR / RT resolve writes reflected
    // radiance in `.rgb` and the Fresnel/gloss composite weight in `.a` here.
    // Sized at render / `trace_scale`.
    pub reflection: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    pub resolve: Option<MtlSsrPass>,
    pub composite: Option<MtlReflectionCompositePass>,
    // Per-axis render-resolution divisor of the reflection target: the
    // ray-traced trace resolution when RT reflections run, else 1. Held so a
    // resize recreates it at the same reduced resolution.
    pub trace_scale: u32,
}

// Create or recreate the reflection target at `width`x`height` divided by
// `trace_scale`. A reduced target is upsampled depth- and normal-aware by the
// composite.
pub(crate) fn create_reflection_target(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    width: u32,
    height: u32,
    trace_scale: u32,
) -> RenderResult<Retained<ProtocolObject<dyn MTLTexture>>> {
    let s = trace_scale.max(1);
    let desc = TextureDesc {
        format: MTLPixelFormat::RGBA16Float,
        width: (width / s).max(1) as usize,
        height: (height / s).max(1) as usize,
        usage: MTLTextureUsage(MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0),
        ..Default::default()
    }
    .build();
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| allocation_failed("reflection texture"))
}

impl MtlContext {
    // Encode the SSR resolve into the reflection target, then blur and composite
    // it over `hdr_resolve` into the composite's output. Runs after the main
    // pass; only called when SSR is on.
    pub(in crate::metal) fn encode_ssr_resolve(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        ssr_params: &render_types::SsrParams,
    ) -> RenderResult<u32> {
        // The pre-pass channels are pool-owned, so they are fetched here rather
        // than cached: a pool rebuild repacks every slot.
        let (Some(reflection), Some(resolve), Some(normal_depth), Some(roughness)) = (
            &self.ssr.reflection,
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
                target: reflection.as_ref(),
                scene: self.targets.hdr.hdr_resolve.as_ref(),
                normal_depth,
                roughness,
                // Always valid: a gray fallback when no EnvironmentMap is bound,
                // which `SsrParams.prefilter_mip_count == 0` tells the shader to
                // ignore.
                prefilter: self.scene.env_map.prefilter.as_ref(),
            },
            ssr_params,
        )?;
        self.encode_reflection_composite(cmd_buf)?;
        Ok(0)
    }
}

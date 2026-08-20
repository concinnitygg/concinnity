// src/metal/post/ssao.rs
//
// SSAO (GTAO): a depth + normal pre-pass, the horizon-search kernel, and a
// depth-aware blur. Pipelines, targets, and encoders live together so the
// effect is a single unit Vulkan / DirectX can mirror.
//
// When SSR is also enabled the kernel shares the SSR pre-pass G-buffer and
// SSAO skips its own pre-pass entirely; with SSR off SSAO runs its own
// pre-pass over the visible static, instanced, and skinned geometry.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice as _, MTLLoadAction, MTLPixelFormat, MTLRenderCommandEncoder as _,
    MTLRenderPipelineState, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

use crate::gfx::ssao::SsaoSettings;
use crate::metal::context::MtlContext;
use crate::metal::post::fullscreen::{
    FullscreenBlend, FullscreenPass, PassTimer, build_slang_fullscreen_pipeline,
    set_fragment_sampler_range,
};
use crate::metal::slang_shaders::SlangLib;

// All SSAO (GTAO) state grouped into one feature unit: the resolved settings,
// the kernel intermediate target, the kernel + blur pipelines, and the 1×1 white
// fallback. `settings`/`targets`/the pipelines are `Some` only when SSAO is
// enabled; `white` is always present so `MtlContext::ao_output_texture` can
// return a constant-1.0 sample when SSAO is off. The blurred occlusion the main
// pass samples now lives in the transient pool, not here.
pub(crate) struct SsaoState {
    pub settings: Option<SsaoSettings>,
    pub targets: Option<SsaoTargets>,
    pub kernel_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub blur_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub white: crate::metal::allocator::PooledTexture,
}

// Pixel format of both occlusion targets: the kernel's raw output and the
// blurred `ao_output` the transient pool backs. Single-channel visibility.
pub(crate) const SSAO_OCCLUSION_FORMAT: MTLPixelFormat = MTLPixelFormat::R8Unorm;

// Pipelines

// Build one SSAO fullscreen-triangle pipeline (the GTAO kernel or the blur).
// Both target a single-sample occlusion texture and pair with the shared
// `fullscreen_vertex`; `fragment` selects which single-source variant.
pub(crate) fn build_ssao_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    fragment: &SlangLib,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_slang_fullscreen_pipeline(
        device,
        fragment,
        SSAO_OCCLUSION_FORMAT,
        FullscreenBlend::Replace,
        hot_reload,
    )
}

// Targets

// Off-screen target for the SSAO (GTAO) kernel: the raw occlusion the blur then
// reads. The blurred occlusion the main pass samples (`ao_output`) now lives in
// the transient pool, not here, so this struct holds only the kernel
// intermediate. The view-space normal / depth the kernel + blur read come from
// the unified G-buffer pre-pass (`metal/post/gbuffer.rs`). Single-sample, render
// resolution; created when SSAO is enabled and rebuilt on resize.
pub(crate) struct SsaoTargets {
    // Raw GTAO kernel output (`R8Unorm`), before the blur.
    pub ao_raw: Retained<ProtocolObject<dyn MTLTexture>>,
}

// Create or recreate the SSAO targets at `width`x`height`.
pub(crate) fn create_ssao_targets(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    width: u32,
    height: u32,
) -> Result<SsaoTargets, String> {
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;

    let sampled = MTLTextureUsage(MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0);
    let make = |label: &str| -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
        let desc = MTLTextureDescriptor::new();
        // SAFETY: plain descriptor property setters, all values in range.
        unsafe {
            desc.setTextureType(MTLTextureType::Type2D);
            desc.setPixelFormat(SSAO_OCCLUSION_FORMAT);
            desc.setWidth(w);
            desc.setHeight(h);
            desc.setUsage(sampled);
            desc.setStorageMode(objc2_metal::MTLStorageMode::Private);
        }
        device
            .newTextureWithDescriptor(&desc)
            .ok_or_else(|| format!("failed to create SSAO {} texture", label))
    };

    let ao_raw = make("raw occlusion")?;

    Ok(SsaoTargets { ao_raw })
}

// Encoders

impl MtlContext {
    // Encode the SSAO (GTAO) passes: the horizon-search kernel that
    // integrates the GTAO visibility arc and a depth-aware blur. Runs before
    // the main pass so `shade_surface` can sample the blurred occlusion and
    // modulate its ambient term. The kernel + blur read the view-space normal +
    // depth the unified G-buffer pre-pass produced; SSAO no longer runs its own
    // geometry redraw. Only called when SSAO is enabled.
    pub(in crate::metal) fn encode_ssao(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        ssao_params: &crate::gfx::render_types::SsaoParams,
    ) -> Result<u32, String> {
        let (targets, kernel_ps, blur_ps, gbuffer) = match (
            &self.ssao.targets,
            &self.ssao.kernel_pipeline,
            &self.ssao.blur_pipeline,
            // The unified pre-pass produced the depth + normal the kernel
            // reads; SSAO shares it and runs no geometry redraw of its own.
            // Pool-owned, so it is fetched here rather than cached.
            self.gbuffer_normal_depth(),
        ) {
            (Some(t), Some(b), Some(c), Some(g)) => (t, b, c, g),
            _ => return Ok(0),
        };

        // The blurred occlusion the main pass samples is the graph's `ao_output`
        // transient, owned by the pool so it can share a heap slot with
        // `bloom_top`. The pool always holds it when SSAO is on (both gate on
        // the same setting).
        let ao_output = self
            .transient_pool
            .texture_for("ao_output")
            .ok_or("ao_output missing from transient pool")?;

        // Kernel: GTAO horizon search over the G-buffer -> raw occlusion.
        self.fullscreen_pass(
            cmd_buf,
            FullscreenPass {
                target: targets.ao_raw.as_ref(),
                load: MTLLoadAction::DontCare,
                timer: PassTimer::Whole(crate::metal::pass_timing::PassId::SsaoKernel),
                pipeline: kernel_ps,
                label: "SSAO kernel",
            },
            // SAFETY: each `setBytes` pointer is derived from a live borrow with that type's
            // `size_of` as the length, and every bound resource outlives the encoder; the indices
            // are the slots the shaders declare.
            |enc| unsafe {
                enc.setFragmentTexture_atIndex(Some(gbuffer), 0);
                set_fragment_sampler_range(enc, &self.post_sampler, 0, 1);
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(ssao_params).cast(),
                    std::mem::size_of::<crate::gfx::render_types::SsaoParams>(),
                    0,
                );
            },
        )?;

        // Blur: depth-aware smoothing of the raw occlusion -> final AO.
        self.fullscreen_pass(
            cmd_buf,
            FullscreenPass {
                target: ao_output,
                load: MTLLoadAction::DontCare,
                timer: PassTimer::Whole(crate::metal::pass_timing::PassId::SsaoBlur),
                pipeline: blur_ps,
                label: "SSAO blur",
            },
            // SAFETY: every resource bound here is owned by `self` and outlives the encoder, at the
            // buffer/texture indices the shaders declare.
            |enc| unsafe {
                enc.setFragmentTexture_atIndex(Some(targets.ao_raw.as_ref()), 0);
                enc.setFragmentTexture_atIndex(Some(gbuffer), 1);
                set_fragment_sampler_range(enc, &self.post_sampler, 0, 2);
            },
        )?;

        Ok(0)
    }
}

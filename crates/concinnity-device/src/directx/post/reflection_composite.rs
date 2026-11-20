//! DirectX's share of the reflection composite, which is where its inputs come
//! from this frame. The roughness blur and the composite -- their pipelines,
//! their targets and both draws -- are written once in
//! `concinnity_engine::render::post::reflection_composite` and reach D3D12
//! through `DxPostDevice`.
//!
//! Shared by both reflection paths: `encode_ssr_resolve` /
//! `encode_rt_reflections` each render their reflection target, then call
//! `encode_reflection_composite` with that target's SRV.

use concinnity_engine::gfx::render_types::{ReflectionDenoiseParams, ReflectionEnvParams};
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::post::device::PostExtent;
use concinnity_engine::render::post::reflection_composite::{
    DenoiseFrame, ReflectionCompositeInputs, ReflectionCompositePass,
};
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::context::DxContext;
use crate::directx::descriptor_slot::SrvSlot;
use crate::directx::post::post_device::{DxPostDevice, PostPipeline, PostTarget};

// The shared composite, holding D3D12's own pipeline and target types. Its
// output is the graph's `scene_pre_taa`, the scene-with-reflections the post
// stack consumes via `scene_srv_for_post`.
pub(in crate::directx) type DxReflectionCompositePass =
    ReflectionCompositePass<PostPipeline, PostTarget>;

// Build the composite at render resolution `width` x `height`, its blur divided
// by `blur_scale`, and with the denoiser at the trace resolution `denoise`
// divides by when a ray-traced resolve feeds it.
pub(in crate::directx) fn build_reflection_composite(
    device: &DxPostDevice<'_>,
    blur_scale: u32,
    (width, height): (u32, u32),
    denoise: Option<u32>,
) -> RenderResult<DxReflectionCompositePass> {
    let extent = PostExtent { width, height };
    let mut pass = ReflectionCompositePass::new(device, blur_scale, extent)?;
    pass.set_denoise(device, denoise, extent)?;
    Ok(pass)
}

impl DxContext {
    // Blur the reflection target by roughness and composite it over the scene.
    // Behind the RT pass the reflection first accumulates over frames. A no-op
    // without the composite or the G-buffer.
    pub(in crate::directx) fn encode_reflection_composite(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        reflection_srv: SrvSlot,
        env: &ReflectionEnvParams,
        denoise: Option<(SrvSlot, &ReflectionDenoiseParams)>,
    ) {
        let (Some(rc), Some(gbuffer)) = (&self.reflection_composite, &self.gbuffer) else {
            return;
        };
        if let Err(e) = rc.encode(
            &self.post_device(frame_idx),
            cmd,
            &ReflectionCompositeInputs {
                reflection: reflection_srv,
                scene: self.targets.hdr.srv_gpu,
                normal_depth: gbuffer.normal_depth_srv_gpu,
                roughness: gbuffer.roughness_srv_gpu,
                prefilter: self.prefilter_cube_srv_gpu(),
                env,
                denoise: denoise.map(|(hits, params)| DenoiseFrame {
                    hits,
                    velocity: gbuffer.velocity_srv_gpu,
                    params,
                }),
            },
        ) {
            tracing::error!("reflection composite: {e}");
        }
    }
}

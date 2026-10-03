//! DirectX's share of the reflection composite, which is where its inputs come
//! from this frame. The roughness blur and the composite -- their pipelines,
//! their targets and both draws -- are written once in
//! `concinnity_core::render::post::reflection_composite` and reach D3D12
//! through `DxPostDevice`.
//!
//! Shared by both reflection paths: `encode_ssr_resolve` /
//! `encode_rt_reflections` each render their reflection target, then call
//! `encode_reflection_composite` with that target's SRV.

use concinnity_core::render::error::RenderResult;
use concinnity_core::render::post::device::PostExtent;
use concinnity_core::render::post::reflection_composite::{
    ReflectionCompositeInputs, ReflectionCompositePass,
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
// by `blur_scale`.
pub(in crate::directx) fn build_reflection_composite(
    device: &DxPostDevice,
    blur_scale: u32,
    width: u32,
    height: u32,
) -> RenderResult<DxReflectionCompositePass> {
    ReflectionCompositePass::new(device, blur_scale, PostExtent { width, height })
}

impl DxContext {
    // Blur the reflection target by surface roughness and composite it over the
    // scene into the composite's output. `reflection_srv` is the SRV of the
    // resolve target the SSR / RT pass just wrote (reflected radiance + weight);
    // it rests in PIXEL_SHADER_RESOURCE after the resolve. No-op when the
    // composite or G-buffer is absent (only when no reflection path is active).
    pub(in crate::directx) fn encode_reflection_composite(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        reflection_srv: SrvSlot,
    ) {
        let (Some(rc), Some(gbuffer)) = (&self.reflection_composite, &self.gbuffer) else {
            return;
        };
        if let Err(e) = rc.encode(
            &self.post_device(frame_idx),
            cmd,
            ReflectionCompositeInputs {
                reflection: reflection_srv,
                scene: self.targets.hdr.srv_gpu,
                normal_depth: gbuffer.normal_depth_srv_gpu,
                roughness: gbuffer.roughness_srv_gpu,
            },
        ) {
            tracing::error!("reflection composite: {e}");
        }
    }
}

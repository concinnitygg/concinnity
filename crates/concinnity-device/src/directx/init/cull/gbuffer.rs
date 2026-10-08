//! The GPU-driven G-buffer pre-pass's model-history ring and the snapshot
//! kernel that fills it. Its PSOs are the shader buckets' own, built while a
//! G-buffer exists.

use concinnity_core::render::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;

use super::CullPlan;
use super::compute::ComputeCull;
use crate::directx::context::{FRAMES, align256};
use crate::directx::init::InitGpu;
use crate::directx::texture::create_uav_buffer;

pub(super) struct GbufferPass {
    pub(super) prev_model_buffers: Vec<ID3D12Resource>,
    pub(super) model_history_root_sig: Option<ID3D12RootSignature>,
    pub(super) model_history_pso: Option<ID3D12PipelineState>,
}

// The GPU-driven G-buffer pre-pass's model history: one column-major
// `float4x4` per cull record per frame, and the snapshot kernel that fills it.
// Device-local, resting as a shader resource between the dispatch that writes
// a slot and the pre-pass that reads it a frame later. Built only when the
// compute cull is active and the G-buffer is enabled.
pub(super) fn build_gbuffer_pass(
    gpu: &InitGpu<'_>,
    compute: &ComputeCull,
    plan: &CullPlan,
    gbuffer_enabled: bool,
) -> RenderResult<GbufferPass> {
    if compute.kernels.is_none() || !gbuffer_enabled {
        return Ok(GbufferPass {
            prev_model_buffers: Vec::new(),
            model_history_root_sig: None,
            model_history_pso: None,
        });
    }
    let device = gpu.hw.alloc.device();
    let info_queue = gpu.hw.info_queue.as_ref();
    let hot_reload = gpu.hot_reload;
    let (mhrs, mhpso) =
        crate::directx::post::gbuffer::build_model_history(device, info_queue, hot_reload)?;
    let prev_model_size = align256((plan.n_cull * std::mem::size_of::<[[f32; 4]; 4]>()) as u64);
    let mut prev_model_buffers: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
        prev_model_buffers.push(create_uav_buffer(
            device,
            prev_model_size,
            D3D12_RESOURCE_STATE_COMMON,
        )?);
    }
    Ok(GbufferPass {
        prev_model_buffers,
        model_history_root_sig: Some(mhrs),
        model_history_pso: Some(mhpso),
    })
}

//! The GPU-driven G-buffer pre-pass: its 3-MRT bindless pipeline with the cull
//! command signature rebuilt against its root signature, and the model-history
//! ring with the snapshot kernel that fills it.

use concinnity_core::render::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;

use super::CullPlan;
use super::compute::ComputeCull;
use crate::directx::context::{FRAMES, align256};
use crate::directx::init::InitGpu;
use crate::directx::texture::create_uav_buffer;

pub(super) struct GbufferPass {
    pub(super) root_sig: Option<ID3D12RootSignature>,
    pub(super) pso: Option<ID3D12PipelineState>,
    pub(super) cmd_sig: Option<ID3D12CommandSignature>,
    pub(super) prev_model_buffers: Vec<ID3D12Resource>,
    pub(super) model_history_root_sig: Option<ID3D12RootSignature>,
    pub(super) model_history_pso: Option<ID3D12PipelineState>,
}

// GPU-driven G-buffer pre-pass: a 3-MRT bindless pipeline whose VS reads
// model + roughness from `GpuObjectData[object_id]` + the previous frame's
// model from the model-history ring, drawn by reusing the main pass's
// per-frame indirect command buffer (NO new cull -- the camera-frustum
// cull already ran). Plus that ring (one column-major `float4x4` per cull
// record per frame) and the snapshot kernel that fills it: device-local,
// resting as a shader resource between the dispatch that writes a slot
// and the pre-pass that reads it a frame later. Built only when the compute
// cull is active and the G-buffer is enabled.
pub(super) fn build_gbuffer_pass(
    gpu: &InitGpu<'_>,
    compute: &ComputeCull,
    plan: &CullPlan,
    gbuffer_enabled: bool,
) -> RenderResult<GbufferPass> {
    if compute.pso.is_none() || !gbuffer_enabled {
        return Ok(GbufferPass {
            root_sig: None,
            pso: None,
            cmd_sig: None,
            prev_model_buffers: Vec::new(),
            model_history_root_sig: None,
            model_history_pso: None,
        });
    }
    let device = gpu.hw.alloc.device();
    let info_queue = gpu.hw.info_queue.as_ref();
    let hot_reload = gpu.hot_reload;
    let (grs, gpso, gsig) =
        crate::directx::post::gbuffer::build_gbuffer_bindless(device, info_queue, hot_reload)?;
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
        root_sig: Some(grs),
        pso: Some(gpso),
        cmd_sig: Some(gsig),
        prev_model_buffers,
        model_history_root_sig: Some(mhrs),
        model_history_pso: Some(mhpso),
    })
}

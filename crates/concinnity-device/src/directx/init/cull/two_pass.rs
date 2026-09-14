//! Two-pass occlusion: the phase-2 cull pipeline and the second
//! indirect-command buffers its disocclusion draws consume.

use concinnity_core::render::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;

use super::CullPlan;
use super::bindless::BindlessPass;
use super::compute::{ComputeCull, indirect_buffer_size};
use crate::directx::context::{FRAMES, dump_on_err};
use crate::directx::cull::{compile_cull_shader_phase2, create_cull_pso};
use crate::directx::init::InitGpu;
use crate::directx::texture::create_uav_buffer;

pub(super) struct TwoPassCull {
    pub(super) pso: Option<ID3D12PipelineState>,
    pub(super) indirect_buffers: Vec<ID3D12Resource>,
}

// Phase-2 cull PSO for two-pass occlusion (same root sig as the compute cull,
// `main_phase2` entry), plus the per-frame second indirect-command buffers the
// phase-2 cull writes and `Main2` consumes. Built only when the world opted in
// and the compute cull is active.
pub(super) fn build_two_pass_cull(
    gpu: &InitGpu<'_>,
    bindless: &BindlessPass,
    compute: &ComputeCull,
    plan: &CullPlan,
    occlusion_two_pass: bool,
) -> RenderResult<TwoPassCull> {
    let (Some(crs), true) = (compute.root_sig.as_ref(), occlusion_two_pass) else {
        return Ok(TwoPassCull {
            pso: None,
            indirect_buffers: Vec::new(),
        });
    };
    let device = gpu.hw.alloc.device();
    let cs2 = compile_cull_shader_phase2(gpu.hot_reload)?;
    let pso = dump_on_err(
        gpu.hw.info_queue.as_ref(),
        create_cull_pso(device, crs, &cs2),
    )?;
    let indirect_size = indirect_buffer_size(1 + bindless.world_pipelines.len(), plan.n_cull);
    // One per frame plus the reserved reflection-probe capture slot, matching
    // the phase-1 indirect buffers.
    let mut indirect_buffers: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES + 1);
    for _ in 0..FRAMES + 1 {
        indirect_buffers.push(create_uav_buffer(
            device,
            indirect_size,
            D3D12_RESOURCE_STATE_COMMON,
        )?);
    }
    Ok(TwoPassCull {
        pso: Some(pso),
        indirect_buffers,
    })
}

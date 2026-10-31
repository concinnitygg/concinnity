//! Two-pass occlusion: the late cull pipeline and the second
//! indirect-command buffers its disocclusion draws consume.

use concinnity_engine::render::backend::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;

use super::bindless::BindlessPass;
use super::compute::ComputeCull;
use crate::directx::context::dump_on_err;
use crate::directx::cull::{compile_cull_shader_late, create_cull_pso};
use crate::directx::init::InitGpu;
use crate::directx::texture::create_uav_buffer;

pub(super) struct TwoPassCull {
    pub(super) pso: Option<ID3D12PipelineState>,
    pub(super) indirect_buffers: Vec<ID3D12Resource>,
}

// Late cull PSO for two-pass occlusion (same root sig as the compute cull,
// `CULL_LATE`), plus the per-frame second indirect-command buffers the
// late cull writes and `MainLate` consumes. Built only when the world opted in
// and the compute cull is active.
pub(super) fn build_two_pass_cull(
    gpu: &InitGpu<'_>,
    bindless: &BindlessPass,
    compute: &ComputeCull,
    occlusion_two_pass: bool,
) -> RenderResult<TwoPassCull> {
    let (Some(crs), true) = (
        compute.kernels.as_ref().map(|k| &k.root_sig),
        occlusion_two_pass,
    ) else {
        return Ok(TwoPassCull {
            pso: None,
            indirect_buffers: Vec::new(),
        });
    };
    let device = gpu.hw.alloc.device();
    let cs_late = compile_cull_shader_late(gpu.hot_reload)?;
    let pso = dump_on_err(
        gpu.hw.info_queue.as_ref(),
        create_cull_pso(device, crs, &cs_late),
    )?;
    let indirect_size =
        crate::directx::compact::indirect_buffer_bytes(super::compute::indirect_commands(
            &compute.variants,
            bindless.world_pipelines.bucket_count(),
        ));
    // One per frame plus the reserved reflection-probe capture slot, matching
    // the early indirect buffers.
    let mut indirect_buffers: Vec<ID3D12Resource> = Vec::with_capacity(gpu.hw.frames() + 1);
    for _ in 0..=gpu.hw.frames() {
        indirect_buffers.push(create_uav_buffer(device, indirect_size)?);
    }
    Ok(TwoPassCull {
        pso: Some(pso),
        indirect_buffers,
    })
}

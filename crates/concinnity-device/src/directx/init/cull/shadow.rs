//! The GPU-driven shadow pass: a depth-only bindless pipeline with the cull
//! command signature rebuilt against its root signature, the frustum-only
//! shadow cull pipeline, and per-frame indirect buffers carrying one cull
//! region per cascade.

use concinnity_core::gfx::render_types;
use concinnity_core::render::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;

use super::CullPlan;
use super::compute::{ComputeCull, status_buffer_size};
use crate::directx::context::{FRAMES, align256, dump_on_err};
use crate::directx::cull::{
    INDIRECT_COMMAND_STRIDE, compile_cull_shader_shadow, create_cull_command_signature,
    create_cull_pso,
};
use crate::directx::init::InitGpu;
use crate::directx::init::pipelines::{
    compile_shadow_bindless_vs, create_shadow_bindless_root_signature, create_shadow_pso,
};
use crate::directx::texture::create_uav_buffer;

pub(super) struct ShadowCull {
    pub(super) bindless_root_sig: Option<ID3D12RootSignature>,
    pub(super) bindless_pso: Option<ID3D12PipelineState>,
    pub(super) cmd_sig: Option<ID3D12CommandSignature>,
    pub(super) cull_pso: Option<ID3D12PipelineState>,
    pub(super) indirect_buffers: Vec<ID3D12Resource>,
    pub(super) status_buffers: Vec<ID3D12Resource>,
}

// GPU-driven shadow pass: a depth-only bindless pipeline + the shared
// cull command signature rebuilt against its root sig (object id still
// at root param 0) + per-frame indirect buffers carrying one cull region
// per cascade (`NUM_SHADOW_CASCADES * n_cull` commands) + a scratch
// cull-status buffer the shadow cull dispatches write but never read.
// Built only when the compute cull is active and shadows are enabled.
pub(super) fn build_shadow_cull(
    gpu: &InitGpu<'_>,
    compute: &ComputeCull,
    plan: &CullPlan,
    shadow_enabled: bool,
) -> RenderResult<ShadowCull> {
    let (Some(crs), true) = (compute.root_sig.as_ref(), shadow_enabled) else {
        return Ok(ShadowCull {
            bindless_root_sig: None,
            bindless_pso: None,
            cmd_sig: None,
            cull_pso: None,
            indirect_buffers: Vec::new(),
            status_buffers: Vec::new(),
        });
    };
    let device = gpu.hw.alloc.device();
    let info_queue = gpu.hw.info_queue.as_ref();
    let hot_reload = gpu.hot_reload;
    let svs = compile_shadow_bindless_vs(hot_reload)?;
    let sbrs = dump_on_err(info_queue, create_shadow_bindless_root_signature(device))?;
    // Reuse the depth-only shadow PSO builder (no pixel shader, 0 RTVs,
    // D32 DSV, slope-scaled depth bias, main vertex layout).
    let sbpso = dump_on_err(info_queue, create_shadow_pso(device, &sbrs, &svs))?;
    let sbsig = dump_on_err(info_queue, create_cull_command_signature(device, &sbrs))?;
    // Frustum-only shadow cull kernel (`main_shadow`), shares the cull root sig.
    let scs = compile_cull_shader_shadow(hot_reload)?;
    let cull_pso = dump_on_err(info_queue, create_cull_pso(device, crs, &scs))?;
    let cascades = render_types::NUM_SHADOW_CASCADES as u64;
    let shadow_indirect_size =
        align256(cascades * (plan.n_cull as u64) * INDIRECT_COMMAND_STRIDE as u64);
    let status_size = status_buffer_size(plan.n_cull);
    let mut indirect_buffers: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES);
    let mut status_buffers: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
        indirect_buffers.push(create_uav_buffer(
            device,
            shadow_indirect_size,
            D3D12_RESOURCE_STATE_COMMON,
        )?);
        status_buffers.push(create_uav_buffer(
            device,
            status_size,
            D3D12_RESOURCE_STATE_COMMON,
        )?);
    }
    Ok(ShadowCull {
        bindless_root_sig: Some(sbrs),
        bindless_pso: Some(sbpso),
        cmd_sig: Some(sbsig),
        cull_pso: Some(cull_pso),
        indirect_buffers,
        status_buffers,
    })
}

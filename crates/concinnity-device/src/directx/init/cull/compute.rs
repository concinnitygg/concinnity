//! The compute cull: its pipeline and command signature, the per-frame
//! draw-args, indirect-command and cull-status buffers it reads and writes, and
//! the Hi-Z pyramid its occlusion test samples.

use concinnity_core::gfx::render_types;
use concinnity_core::render::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;

use super::CullPlan;
use super::bindless::BindlessPass;
use crate::directx::allocator::PooledBuffer;
use crate::directx::context::{DxDescriptors, DxTargets, FRAMES, align256, dump_on_err};
use crate::directx::cull::{
    INDIRECT_COMMAND_STRIDE, compile_cull_shader, create_cull_command_signature, create_cull_pso,
    create_cull_root_signature,
};
use crate::directx::hiz::{HiZDeviceCtx, HiZResources, HiZTarget};
use crate::directx::init::{HIZ_MAX_MIPS, InitGpu};
use crate::directx::texture::{create_buffer, create_uav_buffer};

pub(super) struct ComputeCull {
    pub(super) root_sig: Option<ID3D12RootSignature>,
    pub(super) pso: Option<ID3D12PipelineState>,
    pub(super) command_signature: Option<ID3D12CommandSignature>,
    pub(super) draw_args_buffers: Vec<PooledBuffer>,
    pub(super) draw_args_ptrs: Vec<*mut u8>,
    pub(super) indirect_buffers: Vec<ID3D12Resource>,
    pub(super) status_buffers: Vec<ID3D12Resource>,
    pub(super) hiz: Option<HiZResources>,
}

pub(super) struct ComputeInputs<'a> {
    pub(super) bindless: &'a BindlessPass,
    pub(super) plan: &'a CullPlan,
    pub(super) descriptors: &'a DxDescriptors,
    pub(super) targets: &'a DxTargets,
}

// Default-heap indirect-command buffer size (UAV target for the cull kernel;
// ExecuteIndirect source for the bindless static pass). One `n_cull`-command
// region per shader bucket: the cull kernel writes every record's slot in each
// region and the main pass issues one `ExecuteIndirect` per region under that
// bucket's pipeline.
pub(super) fn indirect_buffer_size(bucket_count: usize, n_cull: usize) -> u64 {
    align256((bucket_count * n_cull) as u64 * INDIRECT_COMMAND_STRIDE as u64)
}

// Per-object cull-status buffer size (one u32 each).
pub(super) fn status_buffer_size(n_cull: usize) -> u64 {
    align256((n_cull as u64) * std::mem::size_of::<u32>() as u64)
}

// Compute cull: cull compute pipeline + per-frame draw-args /
// indirect-command buffers. Built under the same condition as the object
// buffers: the world has anything to drive.
pub(super) fn build_compute_cull(
    gpu: &InitGpu<'_>,
    inputs: ComputeInputs<'_>,
) -> RenderResult<ComputeCull> {
    let ComputeInputs {
        bindless,
        plan,
        descriptors,
        targets,
    } = inputs;
    let device = gpu.hw.alloc.device();
    let info_queue = gpu.hw.info_queue.as_ref();
    let n_cull = plan.n_cull;
    if n_cull == 0 {
        return Ok(ComputeCull {
            root_sig: None,
            pso: None,
            command_signature: None,
            draw_args_buffers: Vec::new(),
            draw_args_ptrs: Vec::new(),
            indirect_buffers: Vec::new(),
            status_buffers: Vec::new(),
            hiz: None,
        });
    }
    let cs = compile_cull_shader(gpu.hot_reload)?;
    let crs = dump_on_err(info_queue, create_cull_root_signature(device))?;
    let cps = dump_on_err(info_queue, create_cull_pso(device, &crs, &cs))?;
    let csig = dump_on_err(
        info_queue,
        create_cull_command_signature(device, &bindless.root_sig),
    )?;

    let draw_args_size =
        align256((n_cull * std::mem::size_of::<render_types::GpuDrawArgs>()) as u64);
    let indirect_size = indirect_buffer_size(1 + bindless.world_pipelines.len(), n_cull);
    // Always allocated when the cull path is active (matches Metal); resting
    // state `UAV` so it binds as a root UAV with no transition.
    let status_size = status_buffer_size(n_cull);
    let mut draw_args_buffers: Vec<PooledBuffer> = Vec::with_capacity(FRAMES + 1);
    let mut draw_args_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES + 1);
    let mut indirect_buffers: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES + 1);
    let mut status_buffers: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES + 1);
    // `FRAMES + 1`: the extra slot (index `FRAMES`) is the reserved
    // reflection-probe capture slot (see the object-buffer loop). The
    // bake culls each cube face into `indirect_cmd_buffers[FRAMES]` reading
    // `draw_args_buffer_resources[FRAMES]`, a slot the frame never overwrites.
    for _ in 0..FRAMES + 1 {
        let da = create_buffer(
            &gpu.hw.alloc,
            draw_args_size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
        // local that receives the mapping.
        unsafe { da.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| format!("map draw args buffer: {e}"))?;
        draw_args_ptrs.push(ptr as *mut u8);
        draw_args_buffers.push(da);

        // Created in COMMON (D3D12 always makes committed buffers in COMMON
        // regardless of the requested state); the cull pass transitions them
        // to UNORDERED_ACCESS / INDIRECT_ARGUMENT as it writes + executes them.
        indirect_buffers.push(create_uav_buffer(
            device,
            indirect_size,
            D3D12_RESOURCE_STATE_COMMON,
        )?);
        status_buffers.push(create_uav_buffer(
            device,
            status_size,
            D3D12_RESOURCE_STATE_COMMON,
        )?);
    }

    let hiz = build_hiz(gpu, descriptors, targets)?;
    Ok(ComputeCull {
        root_sig: Some(crs),
        pso: Some(cps),
        command_signature: Some(csig),
        draw_args_buffers,
        draw_args_ptrs,
        indirect_buffers,
        status_buffers,
        hiz: Some(hiz),
    })
}

// Hi-Z pyramid. Built under the same condition as the cull pipeline. The
// resource owns its descriptors at the reserved Hi-Z heap slots; when the
// gating condition fails the slots stay empty and the cull kernel's
// `hiz_enabled` flag stays zero so it never samples them. The init kernel
// reads the main-depth SRV the render targets wrote.
fn build_hiz(
    gpu: &InitGpu<'_>,
    descriptors: &DxDescriptors,
    targets: &DxTargets,
) -> RenderResult<HiZResources> {
    let layout = &descriptors.layout;
    let mut mip_uav_cpus: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> = Vec::with_capacity(HIZ_MAX_MIPS);
    let mut mip_uav_gpus: Vec<D3D12_GPU_DESCRIPTOR_HANDLE> = Vec::with_capacity(HIZ_MAX_MIPS);
    for i in 0..HIZ_MAX_MIPS {
        mip_uav_cpus.push(descriptors.slot_cpu(layout.hiz_uav_base_slot + i));
        mip_uav_gpus.push(descriptors.slot_gpu(layout.hiz_uav_base_slot + i));
    }
    let hiz = HiZResources::new(
        HiZDeviceCtx {
            device: &gpu.hw.device,
            info_queue: gpu.hw.info_queue.as_ref(),
            hot_reload: gpu.hot_reload,
        },
        HiZTarget {
            width: targets.extent.render_width,
            height: targets.extent.render_height,
            srv_cpu: descriptors.slot_cpu(layout.hiz_srv_slot),
            srv_gpu: descriptors.slot_gpu(layout.hiz_srv_slot),
            depth_srv_gpu: targets.main_depth_srv_gpu,
            mip_uav_cpus,
            mip_uav_gpus,
        },
    )?;
    Ok(hiz)
}

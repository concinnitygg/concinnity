//! Scene data: the probe-set, view, light and shadow constant-buffer rings, the
//! static local-light buffer, and the clustered light binning.

use concinnity_core::gfx::render_types::{GpuLight, LightUniforms, ShadowUniforms};
use concinnity_core::render::csm;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::uniforms::{ProbeSet, ViewUniforms};
use windows::Win32::Graphics::Direct3D12::*;

use super::InitGpu;
use crate::directx::allocator::PooledBuffer;
use crate::directx::context::{DxUniforms, FRAMES, align256, dump_on_err};
use crate::directx::draw::{upload_light_uniforms, upload_static_records};
use crate::directx::error::map_hresult;
use crate::directx::light_cull::{self as lc, LightCullState};

pub(super) fn build_uniforms(
    gpu: &InitGpu<'_>,
    light_uniforms: LightUniforms,
    local_lights: &[GpuLight],
) -> RenderResult<DxUniforms> {
    let hw = gpu.hw;
    // ProbeSet constant buffers: a `FRAMES` ring the main pass binds at root
    // param [11] (written per frame from `probe.set`), plus a static count-0 CBV
    // the asynchronous capture binds so a probe face samples the sky, not other
    // probes (and never reads the live ring while `record_frame` rewrites it).
    let probe_set_size = align256(std::mem::size_of::<ProbeSet>() as u64);
    let mut probe_set_cbvs: Vec<PooledBuffer> = Vec::with_capacity(FRAMES);
    let mut probe_set_cbv_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
        let buf = hw.alloc.alloc_buffer(
            probe_set_size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
        // local that receives the mapping.
        unsafe { buf.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| map_hresult(e.code(), "map probe set cbv"))?;
        // Initialize to the empty set (count 0) until the first frame writes it.
        let empty = ProbeSet::EMPTY;
        // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and
        // the source is a separate allocation, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &empty as *const ProbeSet as *const u8,
                ptr as *mut u8,
                std::mem::size_of::<ProbeSet>(),
            );
        }
        probe_set_cbv_ptrs.push(ptr as *mut u8);
        probe_set_cbvs.push(buf);
    }
    let probe_set_empty_cbv = {
        let buf = hw.alloc.alloc_buffer(
            probe_set_size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
        // local that receives the mapping.
        unsafe { buf.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| map_hresult(e.code(), "map probe empty cbv"))?;
        let empty = ProbeSet::EMPTY;
        // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and
        // the source is a separate allocation, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &empty as *const ProbeSet as *const u8,
                ptr as *mut u8,
                std::mem::size_of::<ProbeSet>(),
            );
            buf.Unmap(0, None);
        }
        buf
    };

    // Constant buffers
    let view_ubo_size = align256(std::mem::size_of::<ViewUniforms>() as u64);
    let light_ubo_size = align256(std::mem::size_of::<LightUniforms>() as u64);
    let shadow_ubo_size = align256(std::mem::size_of::<ShadowUniforms>() as u64);

    let mut view_ubo_resources = Vec::with_capacity(FRAMES);
    let mut view_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
        let buf = hw.alloc.alloc_buffer(
            view_ubo_size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
        // local that receives the mapping.
        unsafe { buf.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| map_hresult(e.code(), "map view ubo"))?;
        view_ubo_ptrs.push(ptr as *mut u8);
        view_ubo_resources.push(buf);
    }

    // Per-frame-in-flight light CBV ring, persistently mapped. One slot per
    // frame so a live directional-light or ambient change is a CPU write
    // rather than a queue drain: a turning sky rewrites the set every frame,
    // and a single shared buffer would stall on every one of them.
    let mut light_ubo_resources = Vec::with_capacity(FRAMES);
    let mut light_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
        let buf = hw.alloc.alloc_buffer(
            light_ubo_size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
        // local that receives the mapping.
        unsafe { buf.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| map_hresult(e.code(), "map light ubo"))?;
        light_ubo_ptrs.push(ptr as *mut u8);
        light_ubo_resources.push(buf);
    }

    // Triple-buffer the shadow UBO since cascade VPs are recomputed each
    // frame from the camera. Persistently mapped.
    let mut shadow_ubo_resources = Vec::with_capacity(FRAMES);
    let mut shadow_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
        let buf = hw.alloc.alloc_buffer(
            shadow_ubo_size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
        // local that receives the mapping.
        unsafe { buf.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| map_hresult(e.code(), "map shadow ubo"))?;
        shadow_ubo_ptrs.push(ptr as *mut u8);
        shadow_ubo_resources.push(buf);
    }

    let shadow_uniforms = csm::empty_shadow_uniforms();
    // Seed every frame's shadow UBO with the empty uniforms; per-frame
    // compute_shadow_uniforms in record_frame overwrites them.
    for ptr in &shadow_ubo_ptrs {
        // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and
        // the source is a separate allocation, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &shadow_uniforms as *const ShadowUniforms as *const u8,
                *ptr,
                std::mem::size_of::<ShadowUniforms>(),
            );
        }
    }
    for ptr in &light_ubo_ptrs {
        upload_light_uniforms(*ptr, &light_uniforms);
    }

    // Per-scene local-light storage buffer: a single UPLOAD resource filled
    // once from `local_lights` and never rewritten per frame. An empty scene
    // still allocates a one-element placeholder; the shader's
    // `num_local_lights == 0` guard keeps it from being read.
    let local_light_buffer = {
        let size = align256((local_lights.len().max(1) * std::mem::size_of::<GpuLight>()) as u64);
        let buf = hw.alloc.alloc_buffer(
            size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        if !local_lights.is_empty() {
            upload_static_records(&buf, local_lights, "local-light")?;
        }
        buf
    };

    Ok(DxUniforms {
        view_ubo_resources,
        view_ubo_ptrs,
        light_ubo_resources,
        light_ubo_ptrs,
        local_light_buffer,
        light_uniforms,
        light_dirty: std::cell::Cell::new(concinnity_core::render::frame_dirty::FrameDirty::new(
            FRAMES,
        )),
        shadow_ubo_resources,
        shadow_ubo_ptrs,
        probe_set_cbvs,
        probe_set_cbv_ptrs,
        probe_set_empty_cbv,
    })
}

// Clustered light binning. The per-cluster list + `ClusterParams` buffers
// are always allocated (the forward shaders reference them
// unconditionally, guarded by `use_clusters`); the compute pipeline is
// built only when the world has local lights to bin, which is also what
// gates the `LightCull` graph node.
pub(super) fn build_light_cull(
    gpu: &InitGpu<'_>,
    local_lights: &[GpuLight],
) -> RenderResult<LightCullState> {
    let hw = gpu.hw;
    let cluster_buffer = lc::build_cluster_light_buffer(&hw.device)?;
    let (params_resources, params_ptrs) = lc::build_cluster_params_buffers(&hw.alloc, FRAMES)?;
    let (root_sig, pso) = if local_lights.is_empty() {
        (None, None)
    } else {
        let cs = lc::compile_light_cull_shader(gpu.hot_reload)?;
        let rs = dump_on_err(
            hw.info_queue.as_ref(),
            lc::create_light_cull_root_signature(&hw.device),
        )?;
        let pso = dump_on_err(
            hw.info_queue.as_ref(),
            lc::create_light_cull_pso(&hw.device, &rs, &cs),
        )?;
        (Some(rs), Some(pso))
    };
    Ok(LightCullState {
        root_sig,
        pso,
        cluster_buffer,
        params_resources,
        params_ptrs,
    })
}

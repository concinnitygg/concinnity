//! Clustered binning compute pass. Once per frame, before the Main pass, bins
//! the scene's local lights (the `GpuLight` buffer the forward pass reads) into
//! per-cluster light lists and the reflection probes' influence boxes into
//! per-cluster probe masks, over a screen-tiled, exponential-depth froxel grid.
//! The forward, SSR and transparent passes then shade from only a fragment's
//! cluster's lights and blend only its cluster's probes. Mirrors
//! src/metal/light_cull.rs.

use concinnity_core::gfx::render_types::{CLUSTER_COUNT, CLUSTER_LIST_LEN, ClusterParams};
use concinnity_core::render::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::com;
use crate::directx::builtin_shaders;
use crate::directx::builtin_shaders::CompileProgram;
use crate::directx::context::DxContext;
use crate::directx::error::map_hresult;
use crate::directx::pso::compute_pso;
use crate::directx::root_sig::{RootSig, Visibility};
use crate::directx::texture::create_uav_buffer;

// Byte stride between the two `ClusterParams` slots in a frame's constant
// buffer. Root CBVs must be 256-byte aligned, so each slot is padded up.
// Slot 0 is the live (clustered) params the main camera uses; slot 1 is the
// `use_clusters = 0` copy the planar / probe re-renders bind.
const CLUSTER_PARAMS_SLOT_STRIDE: u64 = 256;
// Slot index of the clustered / unclustered `ClusterParams` copy.
const CLUSTER_SLOT_CLUSTERED: u64 = 0;
const CLUSTER_SLOT_UNCLUSTERED: u64 = 1;

// A light-cluster grid: the `ClusterParams` CBV a fragment places itself in the
// grid with and the per-cluster light lists and probe masks binned for it.
#[derive(Clone, Copy)]
pub(in crate::directx) struct ClusterGrid<'a> {
    pub params_gva: u64,
    pub lists: &'a ID3D12Resource,
}

// Clustered-lighting GPU state: the binning compute pipeline, the per-cluster
// list buffer it writes / the forward pass reads, and the per-frame
// `ClusterParams` constant buffers. All of it always exists (the forward shaders
// reference the buffers unconditionally, guarded by `use_clusters`); the kernel
// runs only on frames with a light or a probe to bin.
pub(in crate::directx) struct LightCullState {
    pub root_sig: ID3D12RootSignature,
    pub pso: ID3D12PipelineState,
    // Per-cluster light lists and probe masks: CLUSTER_LIST_LEN u32, every cluster's
    // light list and then every cluster's probe mask (see `cluster_types.hlsl`).
    // Rests in `PIXEL_SHADER_RESOURCE`; the dispatch flips it to UAV and back.
    pub cluster_buffer: ID3D12Resource,
    // Per-frame `ClusterParams` upload buffers, two 256-byte slots each.
    pub params_resources: Vec<PooledBuffer>,
    pub params_ptrs: Vec<*mut u8>,
}

impl LightCullState {
    // Unmap the persistent `ClusterParams` mappings. Called from `DxContext::drop`.
    pub(in crate::directx) fn unmap(&self) {
        for res in &self.params_resources {
            // SAFETY: the resource is live and this code mapped it, and nothing keeps the mapping
            // past this call.
            unsafe { res.Unmap(0, None) };
        }
    }
}

// Compile the clustered light-binning compute kernel to DXIL.
pub(in crate::directx) fn compile_light_cull_shader(hot_reload: bool) -> RenderResult<Vec<u8>> {
    builtin_shaders::LIGHT_CULL.compile(hot_reload)
}

// Root signature for the light-cull kernel: the `ClusterParams` CBV, the
// per-scene `GpuLight` SRV, the per-cluster list UAV and the frame's probe
// records SRV.
pub(in crate::directx) fn create_light_cull_root_signature(
    device: &ID3D12Device,
) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        // [0] Root CBV b0: ClusterParams
        .cbv(0, Visibility::All)
        // [1] Root SRV t0: StructuredBuffer<GpuLight>
        .srv(0, Visibility::All)
        // [2] Root UAV u0: RWStructuredBuffer<uint> cluster_list
        .uav(0, Visibility::All)
        // [3] Root SRV t1: StructuredBuffer<ProbeUniforms> probe_records
        .srv(1, Visibility::All)
        .build(device, "light cull root sig")
}

// Compute pipeline state for the light-cull kernel.
pub(in crate::directx) fn create_light_cull_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    cs: &[u8],
) -> RenderResult<ID3D12PipelineState> {
    compute_pso(device, root_sig, cs, "light cull")
}

// Allocate a per-cluster list buffer. Created in `COMMON` (D3D12
// creates every buffer there regardless of the requested state); the light-cull
// pass transitions it to UNORDERED_ACCESS to write and back to
// PIXEL_SHADER_RESOURCE for the forward pass, matching how the GPU-cull pass
// cycles its indirect buffers.
pub(in crate::directx) fn build_cluster_light_buffer(
    device: &ID3D12Device,
) -> RenderResult<ID3D12Resource> {
    let len = CLUSTER_LIST_LEN as u64 * std::mem::size_of::<u32>() as u64;
    create_uav_buffer(device, len, D3D12_RESOURCE_STATE_COMMON)
}

// Allocate + persistently map the per-frame `ClusterParams` constant buffers
// (two 256-byte-aligned slots each). Slot 1 is written once here with
// `use_clusters = 0`; the planar / probe re-renders bind it so they fall back
// to iterating every local light, and nothing else in it is read.
pub(in crate::directx) fn build_cluster_params_buffers(
    alloc: &DeviceAllocator,
    frames: usize,
) -> RenderResult<(Vec<PooledBuffer>, Vec<*mut u8>)> {
    let size = CLUSTER_PARAMS_SLOT_STRIDE * 2;
    let mut resources = Vec::with_capacity(frames);
    let mut ptrs = Vec::with_capacity(frames);
    for _ in 0..frames {
        let res = alloc.alloc_buffer(
            size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local
        // that receives the mapping.
        unsafe { res.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| map_hresult(e.code(), "map cluster params buffer"))?;
        let ptr = ptr as *mut u8;
        // Slot 1: the `use_clusters = 0` copy. Static for the context's life.
        let unclustered = ClusterParams::ZERO;
        // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the
        // source is a separate allocation, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &unclustered as *const ClusterParams as *const u8,
                ptr.add((CLUSTER_SLOT_UNCLUSTERED * CLUSTER_PARAMS_SLOT_STRIDE) as usize),
                std::mem::size_of::<ClusterParams>(),
            );
        }
        resources.push(res);
        ptrs.push(ptr);
    }
    Ok((resources, ptrs))
}

impl DxContext {
    // GPU virtual address of this frame's `ClusterParams` CBV. `clustered`
    // picks the live params (main camera) or the `use_clusters = 0` copy the
    // planar / probe re-renders bind.
    pub(in crate::directx) fn cluster_params_gva(&self, frame_idx: usize, clustered: bool) -> u64 {
        let slot = if clustered {
            CLUSTER_SLOT_CLUSTERED
        } else {
            CLUSTER_SLOT_UNCLUSTERED
        };
        let base = com::gpu_va(&self.light_cull.params_resources[frame_idx]);
        base + slot * CLUSTER_PARAMS_SLOT_STRIDE
    }

    // GPU virtual address of the per-cluster list buffer (root SRV).
    pub(in crate::directx) fn cluster_list_gva(&self) -> u64 {
        com::gpu_va(&self.light_cull.cluster_buffer)
    }

    // Write this frame's live `ClusterParams` into slot 0. Slot 1 (the
    // `use_clusters = 0` copy) was filled at init and is never rewritten.
    pub(in crate::directx) fn write_cluster_params(
        &self,
        frame_idx: usize,
        params: &ClusterParams,
    ) {
        let dst = self.light_cull.params_ptrs[frame_idx];
        // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the
        // source is a separate allocation, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                params as *const ClusterParams as *const u8,
                dst.add((CLUSTER_SLOT_CLUSTERED * CLUSTER_PARAMS_SLOT_STRIDE) as usize),
                std::mem::size_of::<ClusterParams>(),
            );
        }
    }

    // The main camera's cluster grid: this frame's live params and the lists
    // the frame's `LightCull` node bins.
    pub(in crate::directx) fn main_cluster_grid(&self, frame_idx: usize) -> ClusterGrid<'_> {
        ClusterGrid {
            params_gva: self.cluster_params_gva(frame_idx, true),
            lists: &self.light_cull.cluster_buffer,
        }
    }

    // Dispatch the clustered binning pass for `grid`. One thread per cluster;
    // the kernel builds the cluster's world-space AABB and tests each local
    // light's sphere and each probe's influence box against it, writing the
    // surviving indices into `grid.lists`, which the caller holds in
    // `UNORDERED_ACCESS` across the dispatch. The executor does that for the
    // main camera's lists around `LightCull`; each mirror render does it for
    // its own.
    pub(in crate::directx) fn encode_light_cull(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        grid: ClusterGrid<'_>,
    ) {
        let params_gva = grid.params_gva;
        let cluster_buffer = grid.lists;
        let lights_gva = com::gpu_va(&self.uniforms.local_light_buffer);
        let records_gva = self.probe.gpu.records[frame_idx].gpu_va();

        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetComputeRootSignature(&self.light_cull.root_sig);
            cmd.SetPipelineState(&self.light_cull.pso);
            cmd.SetComputeRootConstantBufferView(0, params_gva);
            cmd.SetComputeRootShaderResourceView(1, lights_gva);
            cmd.SetComputeRootUnorderedAccessView(2, com::gpu_va(cluster_buffer));
            cmd.SetComputeRootShaderResourceView(3, records_gva);
            // One thread per cluster, 64-wide threadgroups.
            cmd.Dispatch(CLUSTER_COUNT.div_ceil(64), 1, 1);
        }
    }
}

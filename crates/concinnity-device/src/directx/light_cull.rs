// src/directx/light_cull.rs
//
// Clustered light-binning compute pass. Once per frame, before the Main pass,
// bins the scene's local lights (the `GpuLight` buffer the forward pass reads)
// into per-cluster index lists over a screen-tiled, exponential-depth froxel
// grid. The forward pass then shades each fragment from only its cluster's
// lights instead of iterating every light. Mirrors src/metal/light_cull.rs.

use windows::Win32::Graphics::Direct3D12::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use crate::directx::builtins::{self, Ctx};
use crate::directx::context::DxContext;
use crate::directx::pipeline::serialize_desc_and_create;
use crate::directx::texture::{create_buffer, create_uav_buffer, transition_barrier};
use crate::gfx::render_types::{CLUSTER_COUNT, CLUSTER_LIGHT_LIST_STRIDE, ClusterParams};

// Byte stride between the two `ClusterParams` slots in a frame's constant
// buffer. Root CBVs must be 256-byte aligned, so each slot is padded up.
// Slot 0 is the live (clustered) params the main camera uses; slot 1 is the
// `use_clusters = 0` copy the planar / probe re-renders bind.
const CLUSTER_PARAMS_SLOT_STRIDE: u64 = 256;
// Slot index of the clustered / unclustered `ClusterParams` copy.
const CLUSTER_SLOT_CLUSTERED: u64 = 0;
const CLUSTER_SLOT_UNCLUSTERED: u64 = 1;

// Clustered-lighting GPU state: the binning compute pipeline, the per-cluster
// light-index buffer it writes / the forward pass reads, and the per-frame
// `ClusterParams` constant buffers. The buffers are always allocated (the
// forward shaders reference them unconditionally, guarded by `use_clusters`);
// the pipeline is built only when the world has local lights.
pub(in crate::directx) struct LightCullState {
    pub root_sig: Option<ID3D12RootSignature>,
    pub pso: Option<ID3D12PipelineState>,
    // Per-cluster light-index lists: CLUSTER_COUNT blocks of
    // CLUSTER_LIGHT_LIST_STRIDE u32 (slot 0 = count, slots 1.. = light indices).
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
            unsafe { res.Unmap(0, None) };
        }
    }
}

// Compile the clustered light-binning compute kernel to DXBC.
pub(in crate::directx) fn compile_light_cull_shader(hot_reload: bool) -> Result<Vec<u8>, String> {
    builtins::LIGHT_CULL.compile(&Ctx::plain(hot_reload))
}

// Root signature for the light-cull kernel: the `ClusterParams` CBV, the
// per-scene `GpuLight` SRV, and the per-cluster list UAV.
pub(in crate::directx) fn create_light_cull_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let params = [
        // [0] Root CBV b0: ClusterParams
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [1] Root SRV t0: StructuredBuffer<GpuLight>
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [2] Root UAV u0: RWStructuredBuffer<uint> cluster_list
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        ..Default::default()
    };
    serialize_desc_and_create(device, &desc, "light cull root sig")
}

// Compute pipeline state for the light-cull kernel.
pub(in crate::directx) fn create_light_cull_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    cs: &[u8],
) -> Result<ID3D12PipelineState, String> {
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        // Borrow the root signature without an AddRef. `pRootSignature` is a
        // `ManuallyDrop`, so a `clone()` here is never released and leaks one
        // reference per PSO creation. The caller's `&root_sig` outlives the
        // synchronous pipeline-state creation, so copying the raw pointer is sound.
        pRootSignature: unsafe { std::mem::transmute_copy(root_sig) },
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: cs.as_ptr() as _,
            BytecodeLength: cs.len(),
        },
        ..Default::default()
    };
    unsafe { crate::directx::pso_library::create_compute(device, &desc) }
        .map_err(|e| format!("create light cull PSO: {e}"))
}

// Allocate the per-cluster light-index buffer. Created in `COMMON` (D3D12
// creates every buffer there regardless of the requested state); the light-cull
// pass transitions it to UNORDERED_ACCESS to write and back to
// PIXEL_SHADER_RESOURCE for the forward pass, matching how the GPU-cull pass
// cycles its indirect buffers.
pub(in crate::directx) fn build_cluster_light_buffer(
    device: &ID3D12Device,
) -> Result<ID3D12Resource, String> {
    let len =
        (CLUSTER_COUNT * CLUSTER_LIGHT_LIST_STRIDE) as u64 * std::mem::size_of::<u32>() as u64;
    create_uav_buffer(device, len, D3D12_RESOURCE_STATE_COMMON)
}

// Allocate + persistently map the per-frame `ClusterParams` constant buffers
// (two 256-byte-aligned slots each). Slot 1 is written once here with
// `use_clusters = 0`; the planar / probe re-renders bind it so they fall back
// to iterating every local light, and nothing else in it is read.
pub(in crate::directx) fn build_cluster_params_buffers(
    alloc: &DeviceAllocator,
    frames: usize,
) -> Result<(Vec<PooledBuffer>, Vec<*mut u8>), String> {
    let size = CLUSTER_PARAMS_SLOT_STRIDE * 2;
    let mut resources = Vec::with_capacity(frames);
    let mut ptrs = Vec::with_capacity(frames);
    for _ in 0..frames {
        let res = create_buffer(
            alloc,
            size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut();
        unsafe { res.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| format!("map cluster params buffer: {e}"))?;
        let ptr = ptr as *mut u8;
        // Slot 1: the `use_clusters = 0` copy. Static for the context's life.
        let unclustered = ClusterParams::ZERO;
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
        let base = unsafe { self.light_cull.params_resources[frame_idx].GetGPUVirtualAddress() };
        base + slot * CLUSTER_PARAMS_SLOT_STRIDE
    }

    // GPU virtual address of the per-cluster light-index buffer (root SRV).
    pub(in crate::directx) fn cluster_list_gva(&self) -> u64 {
        unsafe { self.light_cull.cluster_buffer.GetGPUVirtualAddress() }
    }

    // Write this frame's live `ClusterParams` into slot 0. Slot 1 (the
    // `use_clusters = 0` copy) was filled at init and is never rewritten.
    pub(in crate::directx) fn write_cluster_params(
        &self,
        frame_idx: usize,
        params: &ClusterParams,
    ) {
        let dst = self.light_cull.params_ptrs[frame_idx];
        unsafe {
            std::ptr::copy_nonoverlapping(
                params as *const ClusterParams as *const u8,
                dst.add((CLUSTER_SLOT_CLUSTERED * CLUSTER_PARAMS_SLOT_STRIDE) as usize),
                std::mem::size_of::<ClusterParams>(),
            );
        }
    }

    // Dispatch the clustered light-binning pass. One thread per cluster; the
    // kernel builds the cluster's world-space AABB and tests each local light's
    // sphere against it, writing the surviving indices into `cluster_buffer`.
    // The executor orders this before Main, which reads the same buffer.
    pub(in crate::directx) fn encode_light_cull(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
    ) -> Result<(), String> {
        let (pso, root_sig) = match (&self.light_cull.pso, &self.light_cull.root_sig) {
            (Some(p), Some(r)) => (p, r),
            _ => return Ok(()),
        };
        let cluster_buffer = &self.light_cull.cluster_buffer;
        let params_gva = self.cluster_params_gva(frame_idx, true);
        let lights_gva = unsafe { self.uniforms.local_light_buffer.GetGPUVirtualAddress() };

        unsafe {
            cmd.ResourceBarrier(&[transition_barrier(
                cluster_buffer,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            cmd.SetComputeRootSignature(root_sig);
            cmd.SetPipelineState(pso);
            cmd.SetComputeRootConstantBufferView(0, params_gva);
            cmd.SetComputeRootShaderResourceView(1, lights_gva);
            cmd.SetComputeRootUnorderedAccessView(2, cluster_buffer.GetGPUVirtualAddress());
            // One thread per cluster, 64-wide threadgroups.
            cmd.Dispatch(CLUSTER_COUNT.div_ceil(64), 1, 1);
            cmd.ResourceBarrier(&[transition_barrier(
                cluster_buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The HLSL kernel hardcodes the list stride + per-cluster cap as `static
    // const uint`s (FXC has no cross-module constants), so they must track the
    // Rust values the CPU sizes the buffer with.
    #[test]
    fn hlsl_cluster_constants_match_render_types() {
        assert!(builtins::LIGHT_CULL.embedded.contains(&format!(
            "CLUSTER_LIGHT_LIST_STRIDE = {CLUSTER_LIGHT_LIST_STRIDE}u"
        )));
        let max_per_cluster = crate::gfx::render_types::MAX_LIGHTS_PER_CLUSTER;
        assert!(
            builtins::LIGHT_CULL
                .embedded
                .contains(&format!("MAX_LIGHTS_PER_CLUSTER = {max_per_cluster}u"))
        );
    }
}

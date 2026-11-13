//! Shadow receiver bounds on D3D12: the two dispatches that reduce the Hi-Z
//! pyramid's first level into the light-space bounds of what the camera saw in
//! each cascade's band, copied into a per-frame READBACK buffer. Frame slot `i`
//! writes buffer `i`, and `begin` reads it back once the fence has retired the
//! frame that wrote it, so a reading is never waited on. Mirrors
//! `metal/shadow_receivers.rs` and `vulkan/shadow_receivers.rs`.

use concinnity_engine::gfx::render_types::{
    ReceiverBands, ReceiverBounds, ReceiverBoundsParams, receiver_tiles,
};
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::shader::programs::abi::receivers::dx::{merge, reduce};
use windows::Win32::Graphics::Direct3D12::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::com;
use super::cull::CullState;
use super::draw::shadow::ShadowState;
use crate::directx::builtin_shaders;
use crate::directx::builtin_shaders::CompileProgram;
use crate::directx::context::DxContext;
use crate::directx::descriptor_slot::{DescriptorTables, SrvSlot};
use crate::directx::error::map_hresult;
use crate::directx::pso::compute_pso;
use crate::directx::root_constants::RootConstants;
use crate::directx::root_sig::{RootSig, Visibility};
use crate::directx::texture::{create_uav_buffer, transition_barrier, uav_barrier};

const BOUNDS_BYTES: u64 = size_of::<ReceiverBounds>() as u64;

// The reduction's pipelines, its per-tile scratch and the readback ring.
pub(in crate::directx) struct ShadowReceiversGpu {
    reduce_root_sig: ID3D12RootSignature,
    reduce_pso: ID3D12PipelineState,
    merge_root_sig: ID3D12RootSignature,
    merge_pso: ID3D12PipelineState,
    // One `ReceiverBands` per tile of the depth (DEFAULT heap, UAV).
    partials: ID3D12Resource,
    // The tiles `partials` holds room for.
    tiles: u32,
    // The merge's output (DEFAULT heap, UAV), copied into this frame's
    // readback.
    bounds: ID3D12Resource,
    // One `ReceiverBounds` per frame in flight, persistently mapped and zeroed
    // at creation, so a slot no frame wrote reads back as serial 0.
    readbacks: Vec<PooledBuffer>,
    readback_ptrs: Vec<*const ReceiverBounds>,
}

impl ShadowReceiversGpu {
    // The reduction beside `cull`'s Hi-Z pyramid and `shadow`'s cascades,
    // `None` without both.
    pub(in crate::directx) fn beside(
        alloc: &DeviceAllocator,
        cull: &CullState,
        shadow: &ShadowState,
        hot_reload: bool,
    ) -> RenderResult<Option<Self>> {
        match cull.hiz.as_ref() {
            Some(hiz) if shadow.map_sizes().cascades > 0 => {
                Self::new(alloc, [hiz.width, hiz.height], hot_reload).map(Some)
            }
            _ => Ok(None),
        }
    }

    // The reduction for a `size` depth, with a readback slot per frame in
    // flight.
    pub(in crate::directx) fn new(
        alloc: &DeviceAllocator,
        size: [u32; 2],
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = alloc.device();
        let reduce_root_sig = RootSig::new()
            .constants::<ReceiverBoundsParams>(reduce::PARAMS, Visibility::All) // [0]
            .srv_table(reduce::DEPTH, 1, Visibility::All) // [1]
            .uav(reduce::PARTIALS, Visibility::All) // [2]
            .build(device, "shadow receiver reduce root sig")?;
        let merge_root_sig = RootSig::new()
            .constants::<ReceiverBoundsParams>(merge::PARAMS, Visibility::All) // [0]
            .uav(merge::PARTIALS, Visibility::All) // [1]
            .uav(merge::BOUNDS, Visibility::All) // [2]
            .build(device, "shadow receiver merge root sig")?;
        let (reduce_pso, merge_pso) =
            create_pipelines(device, &reduce_root_sig, &merge_root_sig, hot_reload)?;

        let tiles = tile_count(size);
        let partials = create_partials(device, tiles)?;
        let bounds = create_uav_buffer(device, BOUNDS_BYTES)?;

        let frames = alloc.frames_in_flight();
        let mut readbacks = Vec::with_capacity(frames);
        let mut readback_ptrs = Vec::with_capacity(frames);
        for _ in 0..frames {
            let buf = alloc.alloc_buffer(
                BOUNDS_BYTES,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // READBACK heaps allow leaving Map active across submissions; the
            // pointer stays valid for the resource's lifetime.
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&raw mut ptr)) }
                .map_err(|e| map_hresult(e.code(), "shadow receiver readback map"))?;
            // SAFETY: the mapping covers at least `BOUNDS_BYTES` bytes, and no GPU work references
            // the buffer yet.
            unsafe { std::ptr::write_bytes(ptr.cast::<u8>(), 0, size_of::<ReceiverBounds>()) };
            readback_ptrs.push(ptr as *const ReceiverBounds);
            readbacks.push(buf);
        }

        Ok(Self {
            reduce_root_sig,
            reduce_pso,
            merge_root_sig,
            merge_pso,
            partials,
            tiles,
            bounds,
            readbacks,
            readback_ptrs,
        })
    }

    // Room for the tiles of a `size` depth, regrown only when it holds too
    // few. Called with the GPU idle.
    pub(in crate::directx) fn resize(
        &mut self,
        device: &ID3D12Device,
        size: [u32; 2],
    ) -> RenderResult<()> {
        let tiles = tile_count(size);
        if tiles > self.tiles {
            self.partials = create_partials(device, tiles)?;
            self.tiles = tiles;
        }
        Ok(())
    }

    // Recompile both kernels against the existing root signatures, for hot
    // reload.
    pub(in crate::directx) fn rebuild_pipelines(
        &self,
        device: &ID3D12Device,
        hot_reload: bool,
    ) -> RenderResult<(ID3D12PipelineState, ID3D12PipelineState)> {
        create_pipelines(
            device,
            &self.reduce_root_sig,
            &self.merge_root_sig,
            hot_reload,
        )
    }

    // Swap rebuilt pipelines in.
    pub(in crate::directx) fn swap_pipelines(
        &mut self,
        (reduce, merge): (ID3D12PipelineState, ID3D12PipelineState),
    ) {
        self.reduce_pso = reduce;
        self.merge_pso = merge;
    }

    // What frame slot `frame_idx`'s readback holds, once the fence has retired
    // its writer.
    pub(in crate::directx) fn read(&self, frame_idx: usize) -> Option<ReceiverBounds> {
        let &ptr = self.readback_ptrs.get(frame_idx)?;
        // SAFETY: `ptr` is this slot's persistent mapping of a READBACK buffer of one
        // `ReceiverBounds`; the fence wait ahead of this call retired the copy that wrote it, and
        // every bit pattern is a valid `ReceiverBounds`.
        Some(unsafe { ptr.read() })
    }

    // Record both dispatches under `params`, reading the pyramid through
    // `pyramid`, and copy the bounds into frame slot `frame_idx`'s readback.
    fn record(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        heap: &ID3D12DescriptorHeap,
        pyramid: SrvSlot,
        params: &ReceiverBoundsParams,
        frame_idx: usize,
    ) {
        let Some(readback) = self.readbacks.get(frame_idx) else {
            return;
        };
        let [groups_x, groups_y] = receiver_tiles(params.width, params.height);
        if groups_x * groups_y > self.tiles {
            return;
        }
        let partials_gva = com::gpu_va(&self.partials);
        // SAFETY: the command list is recording, and the root signature is owned by `self`.
        unsafe { cmd.SetComputeRootSignature(&self.reduce_root_sig) };
        // SAFETY: the PSO is owned by `self` and matches the root signature above.
        unsafe { cmd.SetPipelineState(&self.reduce_pso) };
        // SAFETY: the descriptor heap outlives the frame.
        unsafe { cmd.SetDescriptorHeaps(&[Some(heap.clone())]) };
        // SAFETY: parameter 0 of the reduce root signature holds `params`-sized root constants.
        unsafe { cmd.set_compute_root_constants(0, params) };
        // SAFETY: the pyramid's SRV lives in the heap set above, and the graph moved the pyramid
        // into a compute-readable state.
        unsafe { cmd.set_compute_srv_table(1, pyramid) };
        // SAFETY: `partials_gva` addresses the live partials buffer owned by `self`.
        unsafe { cmd.SetComputeRootUnorderedAccessView(2, partials_gva) };
        // SAFETY: the reduce pipeline and every root argument it reads are bound above.
        unsafe { cmd.Dispatch(groups_x, groups_y, 1) };

        let barrier = uav_barrier(&self.partials);
        // SAFETY: the command list is recording, and the partials buffer is live.
        unsafe { cmd.ResourceBarrier(&[barrier]) };

        // SAFETY: the command list is recording, and the root signature is owned by `self`.
        unsafe { cmd.SetComputeRootSignature(&self.merge_root_sig) };
        // SAFETY: the PSO is owned by `self` and matches the root signature above.
        unsafe { cmd.SetPipelineState(&self.merge_pso) };
        // SAFETY: parameter 0 of the merge root signature holds `params`-sized root constants.
        unsafe { cmd.set_compute_root_constants(0, params) };
        // SAFETY: the partials are live and their writes are ordered by the UAV barrier above.
        unsafe { cmd.SetComputeRootUnorderedAccessView(1, partials_gva) };
        // SAFETY: the bounds buffer is live and owned by `self`.
        unsafe { cmd.SetComputeRootUnorderedAccessView(2, com::gpu_va(&self.bounds)) };
        // SAFETY: the merge pipeline and every root argument it reads are bound above.
        unsafe { cmd.Dispatch(1, 1, 1) };

        // The transition out of UNORDERED_ACCESS makes the merge's write visible
        // to the copy.
        let to_copy = transition_barrier(
            &self.bounds,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        // SAFETY: the command list is recording, and the bounds buffer is live.
        unsafe { cmd.ResourceBarrier(&[to_copy]) };
        // SAFETY: the command list is recording, and both buffers are live and hold
        // `BOUNDS_BYTES` bytes.
        unsafe { cmd.CopyBufferRegion(&**readback, 0, &self.bounds, 0, BOUNDS_BYTES) };
        let to_uav = transition_barrier(
            &self.bounds,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        // SAFETY: the command list is recording, and the bounds buffer is live.
        unsafe { cmd.ResourceBarrier(&[to_uav]) };
    }
}

fn tile_count([width, height]: [u32; 2]) -> u32 {
    let [x, y] = receiver_tiles(width.max(1), height.max(1));
    x * y
}

fn create_partials(device: &ID3D12Device, tiles: u32) -> RenderResult<ID3D12Resource> {
    create_uav_buffer(
        device,
        u64::from(tiles.max(1)) * size_of::<ReceiverBands>() as u64,
    )
}

fn create_pipelines(
    device: &ID3D12Device,
    reduce_root_sig: &ID3D12RootSignature,
    merge_root_sig: &ID3D12RootSignature,
    hot_reload: bool,
) -> RenderResult<(ID3D12PipelineState, ID3D12PipelineState)> {
    let reduce = builtin_shaders::SHADOW_RECEIVERS_REDUCE.compile(hot_reload)?;
    let merge = builtin_shaders::SHADOW_RECEIVERS_MERGE.compile(hot_reload)?;
    Ok((
        compute_pso(device, reduce_root_sig, &reduce, "shadow receiver reduce")?,
        compute_pso(device, merge_root_sig, &merge, "shadow receiver merge")?,
    ))
}

impl DxContext {
    // Reduce the Hi-Z pyramid's first level by this frame's receiver
    // parameters into frame slot `frame_idx`'s readback. A no-op without the
    // reduction, the pyramid, or a measurement planned this frame.
    pub(super) fn encode_shadow_receivers(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
    ) {
        let (Some(gpu), Some(hiz), Some(params)) = (
            self.shadow_receivers.as_ref(),
            self.cull.hiz.as_ref(),
            self.state.planner.shadow.receivers.as_ref(),
        ) else {
            return;
        };
        // A frame planned for another size than the pyramid holds measures
        // nothing.
        if [hiz.width, hiz.height] != [params.width, params.height] {
            return;
        }
        gpu.record(
            cmd,
            &self.descriptors.srv_heap,
            hiz.srv_gpu,
            params,
            frame_idx,
        );
    }
}

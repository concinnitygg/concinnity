//! View samples on D3D12: the dispatch that samples the Hi-Z pyramid's first
//! level over a coarse grid of cells, copied into a per-frame READBACK buffer.
//! Frame slot `i` writes buffer `i`, and `begin` reads it back once the fence
//! has retired the frame that wrote it, so a reading is never waited on.
//! Mirrors `metal/view_samples.rs` and `vulkan/view_samples.rs`.

use concinnity_engine::gfx::render_types::{VIEW_SAMPLE_CELLS, ViewSampleCell, ViewSampleParams};
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::shader::programs::abi::view::dx::{CELLS, DEPTH, PARAMS};
use windows::Win32::Graphics::Direct3D12::*;

use super::allocator::DeviceAllocator;
use super::com;
use super::cull::CullState;
use super::readback_ring::ReadbackRing;
use crate::directx::builtin_shaders;
use crate::directx::builtin_shaders::CompileProgram;
use crate::directx::context::DxContext;
use crate::directx::descriptor_slot::{DescriptorTables, SrvSlot};
use crate::directx::pso::compute_pso;
use crate::directx::root_constants::RootConstants;
use crate::directx::root_sig::{RootSig, Visibility};
use crate::directx::texture::create_uav_buffer;

const CELLS_BYTES: u64 = (VIEW_SAMPLE_CELLS * size_of::<ViewSampleCell>()) as u64;

// The sampling's pipeline, its output and the readback ring.
pub(in crate::directx) struct ViewSamplesGpu {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    // The pass's output (DEFAULT heap, UAV), copied into this frame's
    // readback.
    cells: ID3D12Resource,
    // One grid of cells per frame in flight; a slot no frame wrote reads back
    // as serial 0.
    readbacks: ReadbackRing<ViewSampleCell>,
}

impl ViewSamplesGpu {
    // The sampling beside `cull`'s Hi-Z pyramid, `None` without one.
    pub(in crate::directx) fn beside(
        alloc: &DeviceAllocator,
        cull: &CullState,
        hot_reload: bool,
    ) -> RenderResult<Option<Self>> {
        cull.hiz
            .as_ref()
            .map(|_| Self::new(alloc, hot_reload))
            .transpose()
    }

    // The sampling with a readback slot per frame in flight.
    pub(in crate::directx) fn new(alloc: &DeviceAllocator, hot_reload: bool) -> RenderResult<Self> {
        let device = alloc.device();
        let root_sig = RootSig::new()
            .constants::<ViewSampleParams>(PARAMS, Visibility::All) // [0]
            .srv_table(DEPTH, 1, Visibility::All) // [1]
            .uav(CELLS, Visibility::All) // [2]
            .build(device, "view samples root sig")?;
        let pso = create_pipeline(device, &root_sig, hot_reload)?;
        let cells = create_uav_buffer(device, CELLS_BYTES)?;

        let readbacks = ReadbackRing::new(alloc, VIEW_SAMPLE_CELLS, "view samples readback map")?;

        Ok(Self {
            root_sig,
            pso,
            cells,
            readbacks,
        })
    }

    // Recompile the kernel against the existing root signature, for hot
    // reload.
    pub(in crate::directx) fn rebuild_pipeline(
        &self,
        device: &ID3D12Device,
        hot_reload: bool,
    ) -> RenderResult<ID3D12PipelineState> {
        create_pipeline(device, &self.root_sig, hot_reload)
    }

    // Swap a rebuilt pipeline in.
    pub(in crate::directx) fn swap_pipeline(&mut self, pso: ID3D12PipelineState) {
        self.pso = pso;
    }

    // The cells frame slot `frame_idx`'s readback holds, once the fence has
    // retired its writer.
    pub(in crate::directx) fn read(&self, frame_idx: usize) -> Option<&[ViewSampleCell]> {
        self.readbacks.read(frame_idx)
    }

    // Record the dispatch under `params`, reading the pyramid through
    // `pyramid`, and copy the cells into frame slot `frame_idx`'s readback.
    fn record(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        heap: &ID3D12DescriptorHeap,
        pyramid: SrvSlot,
        params: &ViewSampleParams,
        frame_idx: usize,
    ) {
        // SAFETY: the command list is recording, and the root signature is owned by `self`.
        unsafe { cmd.SetComputeRootSignature(&self.root_sig) };
        // SAFETY: the PSO is owned by `self` and matches the root signature above.
        unsafe { cmd.SetPipelineState(&self.pso) };
        // SAFETY: the descriptor heap outlives the frame.
        unsafe { cmd.SetDescriptorHeaps(&[Some(heap.clone())]) };
        // SAFETY: parameter 0 of the root signature holds `params`-sized root constants.
        unsafe { cmd.set_compute_root_constants(0, params) };
        // SAFETY: the pyramid's SRV lives in the heap set above, and the graph moved the pyramid
        // into a compute-readable state.
        unsafe { cmd.set_compute_srv_table(1, pyramid) };
        // SAFETY: the cells buffer is live and owned by `self`.
        unsafe { cmd.SetComputeRootUnorderedAccessView(2, com::gpu_va(&self.cells)) };
        // SAFETY: the pipeline and every root argument it reads are bound above.
        unsafe { cmd.Dispatch(params.columns, params.rows, 1) };

        let cells = params.columns as usize * params.rows as usize;
        self.readbacks.copy_from(cmd, &self.cells, cells, frame_idx);
    }
}

fn create_pipeline(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    hot_reload: bool,
) -> RenderResult<ID3D12PipelineState> {
    let program = builtin_shaders::VIEW_SAMPLES.compile(hot_reload)?;
    compute_pso(device, root_sig, &program, "view samples")
}

impl DxContext {
    // Sample the Hi-Z pyramid's first level by this frame's view parameters
    // into frame slot `frame_idx`'s readback. A no-op without the sampling,
    // the pyramid, or samples planned this frame.
    pub(super) fn encode_view_samples(&self, cmd: &ID3D12GraphicsCommandList, frame_idx: usize) {
        let (Some(gpu), Some(hiz), Some(params)) = (
            self.view_samples.as_ref(),
            self.cull.hiz.as_ref(),
            self.state.planner.view.samples.as_ref(),
        ) else {
            return;
        };
        // A frame planned for another size than the pyramid holds samples
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

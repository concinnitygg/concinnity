//! Packs a cull's kept draws into one list per stream (`cull_compact.hlsl`):
//! the scan and scatter kernels and their root signature, the group-count
//! scratch each cull slot owns, and the `ExecuteIndirect` of one list by the
//! count the scan wrote behind the commands.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::{
    COUNT_SLOT_BYTES, CULL_GROUP_RECORDS, GpuCompactParams, StreamSlots, count_slot_offset,
    group_count_words,
};
use concinnity_engine::render::backend::error::RenderResult;
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::builtin_shaders::{self, CompileProgram as _};
use crate::directx::com;
use crate::directx::context::{DxContext, align256};
use crate::directx::cull::INDIRECT_COMMAND_STRIDE;
use crate::directx::pso::compute_pso;
use crate::directx::root_constants::RootConstants;
use crate::directx::root_sig::{RootSig, Visibility};
use crate::directx::texture::uav_barrier;

// The scan and the scatter over one root signature.
pub(in crate::directx) struct DxCompactKernels {
    root_sig: ID3D12RootSignature,
    scan: ID3D12PipelineState,
    scatter: ID3D12PipelineState,
}

// [0] b0 GpuCompactParams, [1] t0 the variant block, [2] t1 the draw args,
// [3] u0 the group counts, [4] u1 the cull status, [5] u2 the commands (the
// scatter) or the list lengths behind them (the scan).
pub(in crate::directx) fn build_compact_kernels(
    device: &ID3D12Device,
    hot_reload: bool,
) -> RenderResult<DxCompactKernels> {
    use Visibility::All;
    let root_sig = RootSig::new()
        .constants::<GpuCompactParams>(0, All)
        .srv(0, All)
        .srv(1, All)
        .uav(0, All)
        .uav(1, All)
        .uav(2, All)
        .build(device, "cull compact root sig")?;
    let scan = compute_pso(
        device,
        &root_sig,
        &builtin_shaders::CULL_SCAN.compile(hot_reload)?,
        "cull scan",
    )?;
    let scatter = compute_pso(
        device,
        &root_sig,
        &builtin_shaders::CULL_SCATTER.compile(hot_reload)?,
        "cull scatter",
    )?;
    Ok(DxCompactKernels {
        root_sig,
        scan,
        scatter,
    })
}

// Bytes the commands of an indirect buffer holding `commands` take.
pub(in crate::directx) fn command_bytes(commands: usize) -> u64 {
    commands.max(1) as u64 * u64::from(INDIRECT_COMMAND_STRIDE)
}

// Bytes of an indirect buffer holding `commands`, its count slots behind them.
pub(in crate::directx) fn indirect_buffer_bytes(commands: usize) -> u64 {
    align256(count_slot_offset(command_bytes(commands)) + COUNT_SLOT_BYTES)
}

// Bytes of the group-count scratch a cull of `records` writes.
pub(in crate::directx) fn group_count_bytes(records: usize) -> u64 {
    align256(group_count_words(u32::saturating_from(records)).max(1) as u64 * 4)
}

// The lists a cull packs into: `commands` commands from `base` in `buffer`,
// their count slots behind them.
#[derive(Clone, Copy)]
pub(in crate::directx) struct DrawLists<'a> {
    pub(in crate::directx) buffer: &'a ID3D12Resource,
    pub(in crate::directx) base: u64,
    pub(in crate::directx) commands: usize,
}

impl DrawLists<'_> {
    // Byte offset of the count slots in `buffer`.
    fn counts(&self) -> u64 {
        self.base + count_slot_offset(command_bytes(self.commands))
    }
}

// A buffer a cull writes from `offset` on.
#[derive(Clone, Copy)]
pub(in crate::directx) struct UavSpan<'a> {
    pub(in crate::directx) buffer: &'a ID3D12Resource,
    pub(in crate::directx) offset: u64,
}

impl UavSpan<'_> {
    fn gpu_va(&self) -> u64 {
        com::gpu_va(self.buffer) + self.offset
    }
}

// One cull's packing as a recording runs it: what the cull read and wrote,
// the lists it packs into, and what to pack.
#[derive(Clone, Copy)]
pub(in crate::directx) struct CompactPass<'a> {
    pub(in crate::directx) variants: u64,
    pub(in crate::directx) draw_args: u64,
    pub(in crate::directx) group_counts: UavSpan<'a>,
    pub(in crate::directx) status: UavSpan<'a>,
    pub(in crate::directx) lists: DrawLists<'a>,
    pub(in crate::directx) params: GpuCompactParams,
}

impl DxContext {
    // Pack the cull just dispatched on `cmd`: scan its group counts, then
    // scatter its kept records' commands, each list in record order. The
    // lists' buffer is in UNORDERED_ACCESS; the caller moves it back for the
    // draws.
    pub(in crate::directx) fn encode_compaction(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        pass: CompactPass<'_>,
    ) {
        let Some(kernels) = self.cull.compact.as_ref() else {
            return;
        };
        let lists = com::gpu_va(pass.lists.buffer);
        let (counts, status) = (pass.group_counts.buffer, pass.status.buffer);
        // SAFETY: the command list is recording; the cull's writes to the group counts and the
        // status are ordered before the scan and scatter read them.
        unsafe { cmd.ResourceBarrier(&[uav_barrier(counts), uav_barrier(status)]) };
        // SAFETY: the compact root signature is owned by `self.cull`.
        unsafe { cmd.SetComputeRootSignature(&kernels.root_sig) };
        // SAFETY: parameter 0 holds `GpuCompactParams`-sized root constants.
        unsafe { cmd.set_compute_root_constants(0, &pass.params) };
        // SAFETY: the slot's variant block and draw args are live upload buffers.
        unsafe { cmd.SetComputeRootShaderResourceView(1, pass.variants) };
        // SAFETY: as above.
        unsafe { cmd.SetComputeRootShaderResourceView(2, pass.draw_args) };
        // SAFETY: the group counts and the status rest in UNORDERED_ACCESS.
        unsafe { cmd.SetComputeRootUnorderedAccessView(3, pass.group_counts.gpu_va()) };
        // SAFETY: as above.
        unsafe { cmd.SetComputeRootUnorderedAccessView(4, pass.status.gpu_va()) };
        // SAFETY: the lists' buffer is in UNORDERED_ACCESS, and its count slots lie inside it.
        unsafe { cmd.SetComputeRootUnorderedAccessView(5, lists + pass.lists.counts()) };
        // SAFETY: the scan PSO is owned by `self.cull` and built against the root signature.
        unsafe { cmd.SetPipelineState(&kernels.scan) };
        // SAFETY: every root argument the scan reads is bound above.
        unsafe { cmd.Dispatch(pass.params.bucket_count.clamp(1, 64) + 1, 1, 1) };
        // SAFETY: the scan's group places are ordered before the scatter reads them.
        unsafe { cmd.ResourceBarrier(&[uav_barrier(counts)]) };
        // SAFETY: the lists' commands lie inside the buffer, in UNORDERED_ACCESS.
        unsafe { cmd.SetComputeRootUnorderedAccessView(5, lists + pass.lists.base) };
        // SAFETY: the scatter PSO is owned by `self.cull` and built against the root signature.
        unsafe { cmd.SetPipelineState(&kernels.scatter) };
        let groups = pass.params.object_count.div_ceil(CULL_GROUP_RECORDS);
        // SAFETY: every root argument the scatter reads is bound above.
        unsafe { cmd.Dispatch(groups, 1, 1) };
    }
}

// Execute stream `stream`'s list of `lists`, whose commands sit at `slots`,
// by the count the scan wrote for it.
pub(in crate::directx) fn execute_stream(
    cmd: &ID3D12GraphicsCommandList,
    signature: &ID3D12CommandSignature,
    lists: DrawLists<'_>,
    stream: u32,
    slots: StreamSlots,
) -> u32 {
    if slots.capacity == 0 {
        return 0;
    }
    let offset =
        lists.base + u64::from(slots.region + slots.first) * u64::from(INDIRECT_COMMAND_STRIDE);
    let count_offset = lists.counts() + 4 * u64::from(stream);
    // SAFETY: the command list is recording with a root signature the command signature was
    // built against; the list and its count slot lie inside `lists.buffer`, which rests in
    // INDIRECT_ARGUMENT and outlives the frame.
    unsafe {
        cmd.ExecuteIndirect(
            signature,
            slots.capacity,
            lists.buffer,
            offset,
            lists.buffer,
            count_offset,
        );
    }
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_slots_follow_the_commands_aligned() {
        let commands = command_bytes(300);
        assert_eq!(commands, 300 * u64::from(INDIRECT_COMMAND_STRIDE));
        assert!(indirect_buffer_bytes(300) >= count_slot_offset(commands) + COUNT_SLOT_BYTES);
        assert_eq!(indirect_buffer_bytes(300) % 256, 0);
        assert_eq!(count_slot_offset(commands) % 4, 0);
    }

    #[test]
    fn a_buffer_holds_at_least_one_command() {
        assert_eq!(indirect_buffer_bytes(0), indirect_buffer_bytes(1));
        assert!(group_count_bytes(0) >= 4);
    }
}

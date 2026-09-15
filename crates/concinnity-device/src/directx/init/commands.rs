//! Frame commands: the start, per-pass and end command lists with their
//! allocators, the fence that paces the frame slots, and the timestamp queries.

use concinnity_core::render::error::RenderResult;
use concinnity_core::render::render_graph;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::CreateEventW;

use super::InitGpu;
use crate::directx::context::{
    DxCommands, DxFrameSync, FRAMES, TimestampState, build_timestamp_resources,
};
use crate::directx::error::map_hresult;

pub(super) fn build_commands(gpu: &InitGpu<'_>) -> RenderResult<DxCommands> {
    let hw = gpu.hw;
    // Per-frame command infrastructure
    let mut command_allocators = Vec::with_capacity(FRAMES);
    let mut command_lists: Vec<ID3D12GraphicsCommandList> = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
        let alloc: ID3D12CommandAllocator =
            // SAFETY: the create descriptor and every pointer it borrows are live for the call,
            // and the new COM object lands in a binding that owns it.
            unsafe { hw.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
                .map_err(|e| map_hresult(e.code(), "command allocator"))?;
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and
        // the new COM object lands in a binding that owns it.
        let list: ID3D12GraphicsCommandList = unsafe {
            hw.device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
        }
        .map_err(|e| map_hresult(e.code(), "command list"))?;
        // Close immediately; we re-open each frame.
        // SAFETY: the command list is live and in the recording state, which is what `Close`
        // requires.
        unsafe { list.Close() }.map_err(|e| map_hresult(e.code(), "close cmd list"))?;
        command_allocators.push(alloc);
        command_lists.push(list);
    }

    // Per-pass command allocator + cmd list pool for the parallel-
    // encoding path. Sized FRAMES * PASS_COUNT so each pass owns its
    // own allocator + cmd list per in-flight slot; workers reset
    // their own allocator + cmd list before recording, so multiple
    // workers can encode in parallel without contending. Allocators
    // are very lightweight (a few KB of CPU-side bookkeeping each);
    // a 21-pass x 3-frame pool is ~63 entries.
    let pass_pool_size = FRAMES * render_graph::PASS_COUNT;
    let mut pass_allocators: Vec<ID3D12CommandAllocator> = Vec::with_capacity(pass_pool_size);
    let mut pass_cmd_lists: Vec<ID3D12GraphicsCommandList> = Vec::with_capacity(pass_pool_size);
    for _ in 0..pass_pool_size {
        let alloc: ID3D12CommandAllocator =
            // SAFETY: the create descriptor and every pointer it borrows are live for the call,
            // and the new COM object lands in a binding that owns it.
            unsafe { hw.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
                .map_err(|e| map_hresult(e.code(), "per-pass command allocator"))?;
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and
        // the new COM object lands in a binding that owns it.
        let list: ID3D12GraphicsCommandList = unsafe {
            hw.device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
        }
        .map_err(|e| map_hresult(e.code(), "per-pass command list"))?;
        // Close immediately; we re-open per-pass each frame as needed.
        // SAFETY: the command list is live and in the recording state, which is what `Close`
        // requires.
        unsafe { list.Close() }.map_err(|e| map_hresult(e.code(), "close per-pass cmd list"))?;
        pass_allocators.push(alloc);
        pass_cmd_lists.push(list);
    }

    // End-of-frame outer cmd list pair (composite + final timestamp +
    // resolve). Submitted last so its `ResolveQueryData` reads every
    // per-pass `EndQuery` write.
    let mut end_command_allocators: Vec<ID3D12CommandAllocator> = Vec::with_capacity(FRAMES);
    let mut end_command_lists: Vec<ID3D12GraphicsCommandList> = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
        let alloc: ID3D12CommandAllocator =
            // SAFETY: the create descriptor and every pointer it borrows are live for the call,
            // and the new COM object lands in a binding that owns it.
            unsafe { hw.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
                .map_err(|e| map_hresult(e.code(), "end command allocator"))?;
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and
        // the new COM object lands in a binding that owns it.
        let list: ID3D12GraphicsCommandList = unsafe {
            hw.device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
        }
        .map_err(|e| map_hresult(e.code(), "end command list"))?;
        // SAFETY: the command list is live and in the recording state, which is what `Close`
        // requires.
        unsafe { list.Close() }.map_err(|e| map_hresult(e.code(), "close end cmd list"))?;
        end_command_allocators.push(alloc);
        end_command_lists.push(list);
    }

    Ok(DxCommands {
        command_allocators,
        command_lists,
        pass_allocators,
        pass_cmd_lists,
        end_command_allocators,
        end_command_lists,
    })
}

pub(super) fn build_frame_sync(gpu: &InitGpu<'_>) -> RenderResult<DxFrameSync> {
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
    // new COM object lands in a binding that owns it.
    let fence: ID3D12Fence = unsafe { gpu.hw.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .map_err(|e| map_hresult(e.code(), "create fence"))?;
    // SAFETY: an auto-reset, initially unsignaled event with no name and no security
    // attributes; the call borrows nothing.
    let fence_event = unsafe { CreateEventW(None, false, false, None) }
        .map_err(|e| map_hresult(e.code(), "create fence event"))?;
    Ok(DxFrameSync {
        fence,
        fence_values: vec![0u64; FRAMES],
        next_fence_value: std::cell::Cell::new(1),
        fence_event,
    })
}

// Timestamp infrastructure for the per-frame GPU time chip. Falls back
// to `None`s with frequency 0 when the queue does not support
// timestamps (every WDDM 2.0+ direct queue does, but the fallback keeps
// the rest of the overlay working on adapters that don't).
pub(super) fn build_timestamps(gpu: &InitGpu<'_>) -> TimestampState {
    let (query_heap, readback, readback_ptr, frequency) = build_timestamp_resources(&gpu.hw.alloc);
    TimestampState {
        query_heap,
        readback,
        readback_ptr,
        frequency,
    }
}

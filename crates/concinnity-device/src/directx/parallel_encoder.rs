//! Send/Sync shims for parallel per-pass command-list recording. The
//! render-graph executor in `directx/graph_exec.rs` fans non-composite
//! passes onto rayon workers; each worker resets its assigned pass's
//! allocator + cmd list, encodes its pass, and closes the cmd list. The
//! main thread then submits every closed cmd list in topological pass
//! order via `ExecuteCommandLists`. Mirrors `vulkan/parallel_encoder.rs`
//! and `metal/parallel_encoder.rs`.
//!
//! # Safety
//!
//! `DxContext` and the windows-crate COM smart pointers it stores are not
//! Send/Sync in Rust's type system. Microsoft's free-threading rules make
//! shared, read-only access to D3D12 device-derived objects (root signatures,
//! PSOs, descriptor heaps, mapped upload buffers, fence) thread-safe, and each
//! worker records into its own command list from its own allocator
//! (`pool_index` gives every `(frame, pass)` its own pair), so the
//! single-writer-per-allocator rule holds. The windows crate cannot encode that,
//! so the impls below adopt the claim at the parallel-dispatch boundary.
//!
//! `ParallelCtxRef` is built from `&DxContext` inside `execute_graph`, which
//! joins every worker (via `rayon::scope`) before the borrow returns, and
//! Composite encodes on the main thread after that join. The complete audit of
//! interior-mutable state reachable during `encode_pass_into` - the only basis
//! for the `unsafe impl` below - is:
//!   1. `diagnostics.draw_calls_accum` (`AtomicU32`) - bumped by
//!      `inc_draw_calls`; atomic, so concurrent bumps are sound.
//!   2. `skinned.deformed_primed` (`AtomicBool`) - the G-buffer pass's
//!      first-frame velocity priming gate, stored during encode; atomic.
//!   3. `state.model_history` (`RefCell`) - borrowed only on the main thread,
//!      by the draw-args record build that `record_frame` (before the fan-out)
//!      and the probe bake share; `upload_skinned` uses `get_mut`. The rebuild's
//!      prime request is taken in `record_frame` too and reaches the G-buffer
//!      pass as `GraphFrameParams::prime_model_history`. A worker must never
//!      borrow it.
//!   4. Particle `Cell` state (`particle.last_elapsed` / `particle.frame_index`
//!      / per-emitter `spawn_state`) - advanced by `prepare_particle_pass`
//!      before the fan-out, so the workers never write it.
//!   5. `Cell`s workers read but never write (`cull.prev_view_proj`,
//!      `cull.hiz_valid`, `upscale.jitter`) - set by the frame stages before
//!      `execute_graph` or after its join; concurrent reads alone do not race.
//!   6. `gbuffer.view_history` (`RefCell`) - `borrow`ed by the G-buffer pass
//!      alone during the fan-out (the borrow flag is a write, so a second pass
//!      must never borrow it there); `borrow_mut`ed after the join.
//!   7. The upscaler's `output_is_psr` (`Cell`) - read and written by the
//!      Upscale pass alone. Its history-reset latch is atomic.
//!   8. The upload rings (`lines.vertices`, `text.upload`; `RefCell<Slot>` per
//!      frame) and the device allocator (`DeviceAllocator`'s
//!      `Rc<RefCell<Inner>>`, mutated by allocating, cloning, or dropping a
//!      `PooledBuffer` / `PooledTexture`, as a ring's growth does) - main
//!      thread only. The line ring is reserved and filled by `upload_lines` in
//!      `record_frame`, which hands the Lines pass a plain `LineUpload`, and
//!      the text ring by Composite after the join; every other allocation runs
//!      in init, a rebuild, or `rt_dynamic_update`. A worker must never touch
//!      either ring or a pooled resource's lifetime.
//!
//! Re-audit this list whenever a new pass migrates onto the fan-out.

use concinnity_core::render::parallel_ctx;
use concinnity_core::render::render_graph;
use windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList;

use super::context::DxContext;

// `Send` wrapper around a closed but unsubmitted command list. Workers
// in the parallel-dispatch fan-out reset their cmd list, encode their
// pass into it, close it, then hand it back to the main thread to
// submit in topological order. D3D12 command lists are free-threaded
// for "record on one thread, submit on another" once closed; the
// windows crate just lacks an auto Send impl.
pub(super) struct SendableCmdList(pub ID3D12GraphicsCommandList);

// SAFETY: Each `SendableCmdList` is owned by exactly one worker for the
// span of encoding, then moves back to the main thread for submission.
// No two threads access the inner handle simultaneously.
unsafe impl Send for SendableCmdList {}

// A `Send + Sync` handle to a `&DxContext` borrow. Worker closures use it to
// reach `DxContext` while recording commands into their own command list. The
// wrapper itself is the shared generic shim in `gfx::parallel_ctx`; this alias
// keeps the `ParallelCtxRef<'a>` spelling at the directx call sites.
pub(super) type ParallelCtxRef<'a> = parallel_ctx::ParallelCtxRef<'a, DxContext>;

// SAFETY: see the module doc above for the complete audit of interior-mutable
// state reachable during `encode_pass_into`: atomics (draw-call accumulator,
// deformed-primed gate), state touched only on the main thread around the
// fan-out (model history, particle state, frame-stage cells), and state a
// single pass owns during the fan-out (G-buffer previous VP, upscaler cells).
// The upload rings and the device allocator stay on the main thread. D3D12
// device-derived objects are thread-safe for shared read, and each worker
// records into a distinct command list from a distinct allocator.
unsafe impl parallel_ctx::ParallelEncodeCtx for DxContext {}

// Index into the per-pass `pass_allocators` / `pass_cmd_lists` pools
// in `DxContext`. Layout: `frame_idx * PASS_COUNT + (PassId as usize)`.
pub(super) fn pool_index(frame_idx: usize, pass_id: render_graph::PassId) -> usize {
    frame_idx * render_graph::PASS_COUNT + pass_id as usize
}

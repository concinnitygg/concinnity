//! Send/Sync shims for parallel per-pass command-buffer recording. The
//! render-graph executor in `metal/graph_exec.rs` fans non-composite
//! passes onto rayon workers; each worker mints its own
//! `MTLCommandBuffer`, encodes its pass, and hands the
//! encoded-but-uncommitted buffer back through a per-pass slot. The main
//! thread then commits the slots in topological pass order, so the single
//! command queue's FIFO commit order is the GPU execution order; no
//! `MTLEvent` wait/signal pairs are involved.
//!
//! `MtlContext` and the objc2 protocol objects it stores are not Send/Sync
//! in Rust's type system: Apple's API contract makes shared, read-only
//! access to Metal resources thread-safe, but objc2 cannot encode that
//! without a hand claim. The wrappers below adopt that claim at the
//! parallel-dispatch boundary. Workers reach `&MtlContext` through
//! `ParallelCtxRef::as_ctx()` for strictly read-only encode work; the
//! lone `&mut self` mutations (`diagnostics.frame_stats.draw_calls`,
//! `particle.last_elapsed`, `particle.frame_index`, the per-emitter
//! `spawn_state`) all happen on the main thread before the fan-out.

use concinnity_engine::render::backend::parallel_ctx;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer;

use super::context::MtlContext;

// An encoded, uncommitted command buffer a worker hands back to the main thread
// to commit; objc2 just lacks the `Send` impl.
pub(super) struct SendableCmdBuf(pub Retained<ProtocolObject<dyn MTLCommandBuffer>>);

// SAFETY: Each `SendableCmdBuf` is owned by exactly one worker for the
// span of encoding, then moves back to the main thread for the commit.
// No two threads access the inner `Retained` simultaneously.
unsafe impl Send for SendableCmdBuf {}

// A `Send + Sync` handle to a `&MtlContext` borrow for the encoding workers.
//
// # Safety
//
// The fan-out in `graph_exec.rs` joins every worker before the borrow ends, and
// workers only read through `&self` encode helpers. Metal's devices, queues,
// buffers, textures and pipelines are thread-safe for shared reads.
pub(super) type ParallelCtxRef<'a> = parallel_ctx::ParallelCtxRef<'a, MtlContext>;

// SAFETY: see the type-level safety contract above. Workers reach `&MtlContext`
// for strictly read-only encode work; the lone `&mut self` mutations
// (`diagnostics.frame_stats.draw_calls`, `particle.last_elapsed`, `particle.frame_index`,
// the per-emitter `spawn_state`, the clouds' `CloudSchedule` step) all happen
// on the main thread before the fan-out. During it the cloud passes only read
// the schedule's `Cell`s, except `take_noise_build`, which the atmosphere pass
// alone writes. Apple's Metal device, queue, buffers, textures, and pipeline
// states are thread-safe for shared read. Two pieces of state are not
// thread-safe and are never touched from a worker: `state.model_history` is a
// `RefCell` reached only through `get_mut` from `&mut self` before the fan-out,
// so a worker must never `borrow`/`borrow_mut` it; and the device allocator's
// `Rc<RefCell<Inner>>`, which creating, cloning or dropping a pooled resource
// mutates, so a pass that changes a pooled resource's lifetime must stay off
// the fan-out.
unsafe impl parallel_ctx::ParallelEncodeCtx for MtlContext {}

// src/vulkan/parallel_encoder.rs
//
// `Send + Sync` handle to a `&VkContext` borrow, used by the parallel
// command-buffer recording in `graph_exec.rs`. Each non-composite render-graph
// pass records into its own per-`(frame, pass)` primary command buffer on a
// `jobs::pool()` worker; the workers reach the immutable subset of `VkContext`
// they need through this shim. Mirrors `directx/parallel_encoder.rs` and
// `metal/parallel_encoder.rs`.
//
// # Safety
//
// Construction takes `&'a VkContext`. The wrapper is only used inside the
// parallel-encoder fan-out in `graph_exec.rs`, which joins all workers (via
// `rayon::scope`) before the outer borrow returns. Workers perform strictly
// read-only field access; the `encode_*` helpers they call through
// `encode_pass_into` all take `&self`. Vulkan handles (device, pipelines,
// descriptor sets, persistently-mapped per-frame buffers) are safe for
// concurrent shared read, and each worker records into a *distinct* command
// buffer from a *distinct* command pool (`pass_command_pools` has one pool per
// `(frame, pass)` slot), so the externally-synchronized-pool rule is upheld.
//
// The contract is **read-only**. The complete audit of interior-mutable state
// reachable during `encode_pass_into` - the only basis for the `unsafe impl
// Send + Sync` below - is:
//   1. `draw_calls_accum` (`AtomicU32`) - bumped by `inc_draw_calls`; atomic,
//      so concurrent bumps are sound.
//   2. Particle `Cell` state (`particle_last_elapsed` / `particle_frame_index`
//      / per-emitter `spawn_state`) - hoisted to `prepare_particle_pass`
//      (`&mut self`, before the fan-out), so the workers never touch it.
//   3. The device allocator (`DeviceAllocator`'s `Rc<RefCell<Inner>>`),
//      reachable through `&VkContext` from every pooled resource: creating,
//      cloning, or dropping a `PooledBuffer` / `PooledImage` mutates it with
//      no synchronization. Only `encode_composite_and_text` does any of that
//      during encode (growing a text-upload ring slot), and Composite stays on
//      the main thread (not fanned out). A pass that touches a pooled
//      resource's lifetime -- not just its cached handles -- must never migrate
//      onto the fan-out.
//   4. The text-upload ring (`text_upload`'s `RefCell<Slot>` slots and their
//      map pointers) - reserved and appended to by the Composite pass alone,
//      which stays on the main thread.
//   5. `skinned.deformed_primed` (`AtomicBool`) - the G-buffer pass's
//      first-frame velocity priming gate, stored during encode; atomic, so
//      concurrent access is sound.
//   6. The owning device handle (`VkDevice`), reachable through `&VkContext`
//      from every owned pipeline / layout / render pass. It is `Arc` +
//      `Mutex`, so cloning it here and retiring a handle from here are both
//      sound. That is not a spare tyre: several passes on this fan-out
//      (`encode_main_pass`, `encode_main_pass_phase2`, the shadow, probe and RT
//      passes) do clone it, and while it was `Rc` that raced the refcount and
//      tore the device down early. An earlier version of this note argued the
//      case was safe because no pass *retires* during encode; that was the
//      wrong question, since a plain clone touches the same counter.
// Re-audit this list whenever a new pass migrates onto the fan-out.

use super::context::VkContext;

// The wrapper itself is the shared generic shim in `gfx::parallel_ctx`; this
// alias keeps the `ParallelCtxRef<'a>` spelling at the vulkan call sites.
pub(super) type ParallelCtxRef<'a> = crate::gfx::parallel_ctx::ParallelCtxRef<'a, VkContext>;

// SAFETY: see the module doc above for the complete audit of interior-mutable
// state reachable during `encode_pass_into` (atomic draw-call accumulator,
// atomic deformed-primed priming gate, particle Cell state hoisted before the
// fan-out, the RefCell device allocator and the RefCell text-upload ring
// touched only by the main-thread Composite pass). Vulkan handles are safe for
// concurrent shared read, and
// each worker records into a distinct command buffer from a distinct command
// pool.
unsafe impl crate::gfx::parallel_ctx::ParallelEncodeCtx for VkContext {}

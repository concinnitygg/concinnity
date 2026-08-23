//! Backend-agnostic render graph. Types, builder, and compile pass with
//! unit tests; the per-backend executors live alongside each backend
//! (`metal/graph_exec.rs`, `vulkan/graph_exec.rs`, `directx/graph_exec.rs`)
//! and consume the `CompiledGraph` this module produces.
//!
//! The graph deliberately stops short of allocating GPU resources; that
//! stays backend-owned. The graph's job is to:
//!
//!   * give every pass a stable identity ([`PassId`]),
//!   * track read / write declarations so the compile pass can derive
//!     pass order, transient resource lifetimes, and per-pass barriers,
//!   * surface a `CompiledGraph` the per-backend executor consumes.

#![expect(
    unused_imports,
    reason = "the submodules re-export a shared vocabulary that no single consumer names in full"
)]

mod alias;
mod builder;
mod compile;
mod frame;
mod passes;
mod transient;
mod types;
mod validate;
mod view_mask;

pub(crate) use alias::{AliasPlan, AliasSlot, plan_aliasing_for};
pub(crate) use builder::{GraphBuilder, PassBuilder, ResourceVersion};
pub(crate) use compile::GraphError;
pub use compile::{CompiledGraph, CompiledPass, CompiledResource};
pub use frame::{FOG_FROXEL_X, FOG_FROXEL_Y, FOG_FROXEL_Z, FrameGraphInputs, build_frame_graph};
pub use passes::{PASS_COUNT, PASS_NAMES, PassId};
pub use transient::{
    PoolGates, TransientSlot, TransientTexture, assert_slot_aliasing_sound, plan_pool_slots,
    planning_inputs, pooled,
};
pub(crate) use transient::{SlotConflict, plan_transient_slots, slot_conflicts};
pub use types::{
    BarrierOp, BufferUsage, ClearValue, GraphResourceClass, PassKind, PixelFormat, ReadStages,
    ResourceId, ResourceState, TextureDesc, TextureHandle, TextureUsage,
};
pub(crate) use types::{
    BufferDesc, BufferHandle, PassRange, ResourceOrigin, TextureSize, full_mip_levels,
};
#[cfg(test)]
pub(crate) use validate::barrier_coverage_gaps;
pub(crate) use validate::{BarrierGap, GapKind};
pub use validate::{barrier_coverage_gaps_for_driven, final_states};
pub use view_mask::apply_view;

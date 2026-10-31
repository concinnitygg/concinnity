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
//!   * schedule the passes across a graphics and an async-compute queue, with
//!     the cross-queue signal / wait pairs the edges imply and each read run's
//!     transition placed on whichever side orders it for both queues,
//!   * surface a `CompiledGraph` the per-backend executor consumes.

mod alias;
mod barrier_place;
mod barrier_table;
mod builder;
mod compile;
mod execute;
mod facts;
mod frame;
mod gates;
mod graph_cache;
mod passes;
mod reach;
mod schedule;
#[cfg(test)]
mod test_fixtures;
mod transient;
mod types;
mod validate;
mod view_mask;

pub use barrier_table::{AliasRows, BarrierTable};
pub(crate) use builder::GraphBuilder;
pub(crate) use compile::GraphError;
pub use compile::{CompiledGraph, CompiledPass, CompiledResource};
pub use execute::{
    ExecutedPasses, PassJob, PassRecorder, PassScratch, execute_passes, presenter_index,
};
pub use facts::{FrameContent, PlannedFacts, ResidentFeatures, ResidentUpscaler};
pub(crate) use frame::bloom_top_desc;
pub use frame::{FOG_FROXEL_X, FOG_FROXEL_Y, FOG_FROXEL_Z, FrameGraphInputs, build_frame_graph};
pub use gates::{BackendTraits, BuildStatus, FeatureGates, FrameFacts, frame_graph_inputs};
pub use graph_cache::GraphCache;
pub use passes::{
    GpuSpan, PASS_COUNT, PASS_NAMES, PassId, SPAN_COUNT, SPAN_NAMES, TIMING_COUNT, TIMING_NAMES,
    TimingId,
};
pub use schedule::{CrossQueueWait, PassQueue};
pub use transient::{
    PoolGates, TransientSlot, TransientTexture, assert_slot_aliasing_sound, plan_pool_slots, pooled,
};
pub use types::{
    BarrierOp, BufferUsage, ClearValue, GraphResourceClass, PassKind, PixelFormat, ReadStages,
    ResourceId, ResourceState, TextureDesc, TextureHandle, TextureUsage,
};
pub(crate) use types::{BufferDesc, TextureSize, full_mip_levels};
pub use validate::{
    SyncGap, SyncGapKind, assert_serial_order_honors_schedule, barrier_coverage_gaps_for_driven,
    sync_point_gaps,
};
pub use view_mask::apply_view;

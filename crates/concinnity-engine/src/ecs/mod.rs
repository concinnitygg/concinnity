//! Client-side ecs runtime: what only a renderer-bearing runtime has. The
//! system table, the host seams it lends core's systems, the load-time passes,
//! and the resources the render band parks in a world.
//!
//! The renderer-free half is concinnity-core's `ecs` and is named there: the
//! metadata, asset registry, registration macros, asset-construction API,
//! `PipelineContext`, the `System` behavior trait, the `World` that runs
//! systems over its data, and every core system's table entry.
//!
//! TO ADD A NEW COMPONENT: register it in concinnity-core's `ecs::registry`
//! (`define_components!`). TO ADD A NEW SYSTEM: implement `System` on it with a
//! `for_world` gate, write its entry (in concinnity-core's `CoreSystems` for a
//! core system, in `ecs::systems` for one of this crate's), and list it in
//! `SYSTEMS` in its run position.
//!
//! A system written outside the engine needs none of that: it is registered on
//! the world with `World::add_system`, naming the `Phase` it runs in.

#[cfg(debug_assertions)]
pub(crate) mod access_check;
#[cfg(test)]
mod consumed_columns_tests;
#[cfg(test)]
mod controller_schedule_tests;
#[cfg(test)]
mod determinism_tests;
mod host_resources;
/// How a world runs -- windowed or headless -- resolved once before it starts.
pub mod render_mode;
mod systems;

use concinnity_core::ecs::Parked;
use concinnity_core::render::backend;
use concinnity_core::render::error::RenderError;
use concinnity_core::render::feedback::FrameFeedback;
use concinnity_core::render::snapshot::RenderSnapshot;
pub use systems::SYSTEMS;
// What stays client-side is the content only a renderer-bearing runtime has:
// the resources below.

/// Why the renderer could not be built, left by GraphicsSystem's init for
/// `Runtime::start` to pick up and report.
///
/// Only a world that already resolved to a windowed run can leave one, so this
/// always means a GPU that refused. A machine with no GPU never builds the
/// system that publishes it.
pub struct RenderInitFailure(pub RenderError);

// The pipelined driver's channel pair, published (parked, so the per-step
// take never re-boxes) before the world moves to the simulation thread.
// Present exactly when frames are pipelined: GraphicsSystem's step extracts
// and sends the snapshot through it instead of submitting against a parked
// backend (which the render half owns), and applies the render half's
// feedback. Absent in serial execution.
pub(crate) type PipelinedFrames = Parked<PipelineChannels>;

pub(crate) struct PipelineChannels {
    pub(crate) snapshot_tx: std::sync::mpsc::SyncSender<RenderSnapshot>,
    pub(crate) feedback_rx: std::sync::mpsc::Receiver<FrameFeedback>,
}

// The world's live render backend, parked here between system steps.
// GraphicsSystem's init builds it and parks it; its frame step takes it out at
// the top of the step and puts it back before returning. `None` while a step has
// it taken, or once a live edit transplanted it out.
pub(crate) type ActiveRenderBackend = Parked<Box<dyn backend::RenderBackend>>;

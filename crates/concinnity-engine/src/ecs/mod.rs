//! Client-side ecs runtime: what only a renderer-bearing runtime has. The
//! system table itself, its gates, the load-time decomposition pass, and the
//! resources the render band parks in a world.
//!
//! The renderer-free half is concinnity-core's `ecs` and is named there: the
//! metadata, asset registry, registration macros, asset-construction API,
//! `PipelineContext`, the `System` behavior trait, and the `World` that runs
//! systems over its data.
//!
//! TO ADD A NEW COMPONENT: register it in concinnity-core's `ecs::registry`
//! (`define_components!`). TO ADD A NEW ENGINE SYSTEM: implement the `System`
//! behavior trait on it, write its gate in this crate's `ecs::schedule`, and add
//! one entry to the `define_systems!` table in `ecs::registry` -- the table is
//! the registry AND the schedule (table order is run order).
//!
//! A system written outside the engine needs none of that: it is registered on
//! the world with `World::add_system`, naming the `Phase` it runs in. The table
//! entries stay internal; the phases are what an outside registration anchors
//! to.
//!
//! A system concinnity-core owns is listed in ITS table too
//! (`ecs::HEADLESS_SYSTEMS`, what a world with no host runs), and the two must
//! agree on order, gate description, and the edges among the systems both know
//! about. `headless_drift_tests` is what holds them to that.

pub(crate) mod access_ids;
#[cfg(test)]
mod consumed_columns_tests;
#[cfg(test)]
mod controller_schedule_tests;
pub(crate) mod decompose;
#[cfg(test)]
mod determinism_tests;
#[cfg(test)]
mod headless_drift_tests;
mod host_resources;
mod registry;
/// How a world runs -- windowed or headless -- resolved once before it starts.
pub mod render_mode;
pub mod schedule;
mod world_queries;

use concinnity_core::ecs::MeshBoundsRecord;
use concinnity_core::ecs::Parked;
use concinnity_core::ecs::SceneGroup;
use concinnity_core::render::backend;
// The `SYSTEMS` table is written client-side, since its gates name the client's
// own system types (see `registry`); a gate builds one `BuiltSystem` per
// present entry. Everything that runs it is in concinnity-core.
pub use registry::SYSTEMS;
// What stays client-side is the content only a renderer-bearing runtime has:
// the resources below, and the queries over them in `world_queries`.
pub use world_queries::{
    animation_system_mut, gpu_profile, memory_drift, renders, state_tree, streaming_pressure,
    streaming_stats, take_render_backend,
};

/// Why the renderer could not be built, left by GraphicsSystem's init for
/// `Runtime::start` to pick up and report.
///
/// Only a world that already resolved to a windowed run can leave one, so this
/// always means a GPU that refused. A machine with no GPU never builds the
/// system that publishes it.
pub struct RenderInitFailure(pub concinnity_core::render::error::RenderError);

// The sampled window input and when it was sampled on the host clock, taken
// by the next tick's InputSystem. Serial execution deposits it beside the
// backend right after the draw (whose event pump produced it); the pipelined
// driver deposits the input the render half's go-ahead carried. A missed
// consume merges into the next deposit so no edge is lost.
#[derive(Default)]
pub(crate) struct InputMailbox {
    packet: Option<concinnity_core::input::snapshot::InputPacket>,
    sampled_us: u64,
}

impl InputMailbox {
    // Deposit a packet sampled at `sampled_us`, merging onto an unconsumed one.
    // The merged input is as fresh as its latest sample.
    pub(crate) fn deposit(
        &mut self,
        packet: concinnity_core::input::snapshot::InputPacket,
        sampled_us: u64,
    ) {
        match &mut self.packet {
            Some(pending) => pending.merge_from(packet),
            None => self.packet = Some(packet),
        }
        self.sampled_us = sampled_us;
    }

    // Take the pending packet and its sample time, if one was deposited.
    pub(crate) fn take(&mut self) -> Option<(concinnity_core::input::snapshot::InputPacket, u64)> {
        self.packet.take().map(|packet| (packet, self.sampled_us))
    }
}

// Recorded backend effects that failed at replay and need a simulation-side
// rollback (a streamed-mesh upload refused by a full region, a chunk add).
// Written by GraphicsSystem after submission; StreamingSystem drains it at the
// top of its next step.
#[derive(Default)]
pub(crate) struct RenderOpFailures(pub Vec<concinnity_core::render::ops::OpFailure>);

// The pipelined driver's channel pair, published (parked, so the per-step
// take never re-boxes) before the world moves to the simulation thread.
// Present exactly when frames are pipelined: GraphicsSystem's step extracts
// and sends the snapshot through it instead of submitting against a parked
// backend (which the render half owns), and applies the render half's
// feedback. Absent in serial execution.
pub(crate) type PipelinedFrames = Parked<PipelineChannels>;

pub(crate) struct PipelineChannels {
    pub(crate) snapshot_tx:
        std::sync::mpsc::SyncSender<concinnity_core::render::snapshot::RenderSnapshot>,
    pub(crate) feedback_rx:
        std::sync::mpsc::Receiver<concinnity_core::render::feedback::FrameFeedback>,
}

// The world's live render backend, parked here between system steps.
// GraphicsSystem's init builds it and parks it; each system that drives the
// GPU (GraphicsSystem's frame encode, InputSystem's poll) takes it out at the
// top of its step and puts it back before returning. `None` while a step has
// it taken, or once a live edit transplanted it out.
pub(crate) type ActiveRenderBackend = Parked<Box<dyn backend::RenderBackend>>;

/// The blob's baked per-scene exclusive content groups, published at blob load
/// for the streaming/residency wiring to consume at graphics init.
pub struct BlobSceneGroups(pub Vec<SceneGroup>);

/// The blob's baked per-mesh geometry summaries (AABB + counts by mesh-source
/// handle), published at blob load so graphics init can build draw records for
/// deferred scene-owned meshes without decoding their payloads.
pub struct BlobMeshBounds(pub Vec<MeshBoundsRecord>);

/// The system table. Generates the `SYSTEMS` table a world starts from; table
/// order is run order.
///
/// The three leading fields are the load-time passes around the systems: one
/// that completes the world before the gates read it, one that runs over the
/// world once the gates have built them and before their `init`, and one that
/// pre-creates the event queues a scheduled system's declared access can
/// touch.
///
/// Every system is internal: it has no declarable asset, is never parsed from a
/// world or written to a blob, and is constructed by its gate from world
/// content. Each entry maps a name to the behavior type that implements
/// `System`, the gate that builds it, a human-readable gate description, and
/// the `Phase` it runs in; the entry name doubles as the system's stable
/// display name for profiling and logging.
///
/// Entries are in phase order, and a system registered on a world with
/// `World::add_system` runs after every entry sharing its phase. That is the
/// whole of what a table entry owes an outside registration: the entries
/// themselves stay internal.
#[macro_export]
macro_rules! define_systems {
    ( complete_world: $complete_world:path,
      before_init: $before_init:path,
      prepare_events: $prepare_events:path,
      init_first: [ $( $init_first:ident ),* $(,)? ],
      $( $name:ident => $behavior:path {
            gate: $gate:path,
            present_when: $present_when:literal,
            phase: $phase:ident,
            after: [ $( $after:ident ),* $(,)? ],
            before: [ $( $before:ident ),* $(,)? ] $(,)?
        } ),* $(,)? ) => {
        /// The system table: one entry per system, in run order, plus the
        /// load-time passes that bracket them. `World::start` runs each gate
        /// against the world's content and builds the systems they return.
        pub const SYSTEMS: &::concinnity_core::ecs::SystemTable = &::concinnity_core::ecs::SystemTable {
            entries: &[
                $( ::concinnity_core::ecs::SystemEntry {
                    name: stringify!($name),
                    present_when: $present_when,
                    phase: ::concinnity_core::ecs::Phase::$phase,
                    // Boxing happens here rather than in the gates, so each
                    // gate returns its own system type and the entry's behavior
                    // path has to name it.
                    gate: {
                        fn build(
                            world: &::concinnity_core::ecs::World,
                        ) -> Option<::std::boxed::Box<dyn ::concinnity_core::ecs::System>> {
                            let built: Option<$behavior> = $gate(world);
                            built.map(|s| -> ::std::boxed::Box<dyn ::concinnity_core::ecs::System> {
                                ::std::boxed::Box::new(s)
                            })
                        }
                        build
                    },
                    after: &[ $( stringify!($after) ),* ],
                    before: &[ $( stringify!($before) ),* ],
                }, )*
            ],
            init_first: &[ $( stringify!($init_first) ),* ],
            complete_world: Some($complete_world),
            before_init: Some($before_init),
            prepare_events: Some($prepare_events),
        };
    };
}

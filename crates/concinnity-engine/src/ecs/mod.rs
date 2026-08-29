//! Client-side ecs runtime. The renderer-free metadata, asset registry,
//! registration macros, asset-construction API, `PipelineContext`, the `System`
//! behavior trait, and the `World` that runs systems over its data all live in
//! concinnity-core; this module re-exports them under the historical
//! `crate::ecs::*` paths and adds what only a renderer-bearing runtime has: the
//! system table itself, its gates, the load-time decomposition pass, and the
//! resources the render band parks in a world.
//!
//! TO ADD A NEW COMPONENT: register it in concinnity-core's `ecs::registry`
//! (`define_components!`). TO ADD A NEW SYSTEM: implement the `System` behavior
//! trait on it, write its gate in this crate's `ecs::schedule`, and add one
//! entry to the `define_systems!` table in `ecs::registry` -- the table is the
//! registry AND the schedule (table order is run order).
//!
//! A system concinnity-core owns is listed in ITS table too
//! (`ecs::HEADLESS_SYSTEMS`, what a world with no host runs), and the two must
//! agree on order, gate description, and the edges among the systems both know
//! about. `headless_drift_tests` is what holds them to that.

pub(crate) mod access_ids;
pub(crate) mod by_asset_id;
#[cfg(test)]
mod consumed_columns_tests;
pub(crate) mod decompose;
#[cfg(test)]
mod determinism_tests;
#[cfg(test)]
mod headless_drift_tests;
mod registry;
pub mod schedule;
mod world_queries;

// Renderer-free metadata, registry types, the asset-construction API, and the
// `PipelineContext`, re-exported from concinnity-core so the rest of the client
// keeps its historical `crate::ecs::*` import paths.
pub use concinnity_core::ecs::{
    Access, Arena, AudioClipHandle, BlobAssetDef, ColumnTicks, Component, ComponentAsset,
    ComponentId, ComponentMask, ComponentSlot, ComponentStorage, Entity, EventCursor, EventStore,
    Events, FontHandle, FrameContext, FrameVec, MAX_CHANGE_AGE, MaterialHandle, MeshBoundsRecord,
    MeshHandle, PayloadLocator, PipelineContext, Resources, RuntimeComponent, SceneGroup,
    ScratchStats, SkinnedMeshHandle, TextureHandle, Tick,
};

// The name interner keeps a per-thread table, so it lives in
// `concinnity_host::thread`, whose module re-exports the vocabulary's
// `AssetId` / `AssetRef` alongside.
pub use concinnity_host::thread::asset_id;

// Renderer-free per-frame protocol resources, moved to concinnity-core so the
// physics / audio subsystem crates can reach them without a renderer dependency.
// Re-exported here to keep the historical `crate::ecs::*` paths for every reader
// (engine systems and the editor's hook drive).
pub use concinnity_core::ecs::{
    CursorShape, CursorState, DesiredCursor, DropdownView, ExecutionTrace, FlyCam, FrameRateCap,
    GpuMemoryPressure, HiddenAssets, HudLayers, HudPrefs, MenuActive, MenuOverride, OpenDropdown,
    OverlayImage, OverlayImages, PickEntry, PickIndex, ScheduleMode, ScreenStack, SimTiming,
    TraceEvent, TracePath, TracePaths, TraceRequest, TraceStep, TraceVal, TransientSaves,
    ViewOverrides, WorldLines,
};

// The `SYSTEMS` table is written client-side, since its gates name the client's
// own system types (see `registry`); a gate builds one `BuiltSystem` per
// present entry. Everything that runs it is in concinnity-core.
pub use concinnity_core::ecs::{BuiltSystem, SystemEntry, SystemTable};
pub use registry::SYSTEMS;

// The world itself, its data and the systems that run over it, is
// concinnity-core's; re-exported here under its historical path. What stays
// client-side is the content only a renderer-bearing runtime has: the resources
// below, and the queries over them in `world_queries`.
pub use concinnity_core::ecs::World;
pub use world_queries::{
    gpu_profile, memory_budget, memory_drift, renders, streaming_pressure, streaming_stats,
    systems_and_render_backend, take_render_backend, thread_budget,
};

// The `System` behavior trait + its `StepResult` control signal are renderer-free
// (they name only `PipelineContext`), so they live in concinnity-core; re-export
// them under the historical `crate::ecs::*` paths for every reader (engine
// systems, the `define_systems!` table, and the editor's hook drive).
pub use concinnity_core::ecs::{Clock, StepResult, System};

/// A render backend transplanted out of a previous world, carried into a freshly
/// built world so its GraphicsSystem reuses the live GPU device + window instead
/// of constructing a new one. Published by the `cn editor` live SAVE swap between
/// building the post-edit world and starting it; GraphicsSystem `run_init` takes
/// it and calls `RenderBackend::reload_world` (reusing the window) instead of
/// `init_backend`, so a save applies without recreating the OS window. A shipped
/// runtime never publishes it; it exists only on the editor's live-update path.
pub struct PendingBackend(pub Box<dyn crate::gfx::backend::RenderBackend>);

// The frame's sampled window input, deposited beside the backend right after
// the draw (whose event pump produced it) and taken by InputSystem later the
// same tick. The pipelined driver deposits it from the render half's feedback
// instead; a missed consume merges into the next deposit so no edge is lost.
#[derive(Default)]
pub(crate) struct InputMailbox(pub Option<concinnity_core::render::input::InputPacket>);

impl InputMailbox {
    // Deposit a fresh packet, merging onto an unconsumed one.
    pub(crate) fn deposit(&mut self, packet: concinnity_core::render::input::InputPacket) {
        match &mut self.0 {
            Some(pending) => pending.merge_from(packet),
            None => self.0 = Some(packet),
        }
    }
}

// The frame's recording surfaces, taken and re-parked together by each
// recording system (the same handoff `ActiveRenderBackend` uses, so a step
// never re-boxes them into the resource map): the op queue the tick's backend
// effects accumulate into (drained into the frame snapshot by GraphicsSystem's
// extract, replayed in record order before the draw) and the slot-allocation
// authority ops name destinations from. Published by graphics init; absent in
// a world with no graphics, so recording systems no-op.
pub(crate) struct RenderQueues {
    pub ops: concinnity_core::render::ops::RenderOps,
    pub slots: crate::gfx::render_slots::RenderSlots,
}

// The active backend's capability flags, published by graphics init. The
// backend itself is parked (and on a pipelined frame, owned by the render
// thread), so a system that only needs to know what it supports reads this
// instead of reaching for it. Absent in a world with no graphics.
#[derive(Clone, Copy)]
pub(crate) struct ActiveDeviceCaps(pub concinnity_core::render::backend::DeviceCapabilities);

// The world's parked `RenderQueues` slot. `None` only while a step has it
// taken.
pub(crate) struct ActiveRenderQueues(pub Option<RenderQueues>);

impl ActiveRenderQueues {
    // Take the parked queues for the duration of one system step.
    pub(crate) fn take(resources: &mut Resources) -> Option<RenderQueues> {
        resources.get_mut::<Self>()?.0.take()
    }

    // Park the queues again at the end of the step that took them.
    pub(crate) fn put(resources: &mut Resources, queues: RenderQueues) {
        match resources.get_mut::<Self>() {
            Some(slot) => slot.0 = Some(queues),
            None => {
                resources.insert(Self(Some(queues)));
            }
        }
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
pub(crate) struct PipelinedFrames(pub Option<PipelineChannels>);

pub(crate) struct PipelineChannels {
    pub(crate) snapshot_tx:
        std::sync::mpsc::SyncSender<concinnity_core::render::snapshot::RenderSnapshot>,
    pub(crate) feedback_rx:
        std::sync::mpsc::Receiver<concinnity_core::render::feedback::FrameFeedback>,
}

impl PipelinedFrames {
    // Take the parked channels for the duration of one step.
    pub(crate) fn take(resources: &mut Resources) -> Option<PipelineChannels> {
        resources.get_mut::<Self>()?.0.take()
    }

    // Park the channels again at the end of the step that took them.
    pub(crate) fn put(resources: &mut Resources, channels: PipelineChannels) {
        match resources.get_mut::<Self>() {
            Some(slot) => slot.0 = Some(channels),
            None => {
                resources.insert(Self(Some(channels)));
            }
        }
    }
}

/// The world's live render backend, parked here between system steps.
/// GraphicsSystem's init builds it and parks it; each system that drives the
/// GPU (GraphicsSystem's frame encode, InputSystem's poll) takes it out at the
/// top of its step and puts it back before returning, so the backend and the
/// `PipelineContext` are never borrowed together. `None` while a step has it
/// taken, or once the editor's live SAVE transplanted it out.
pub struct ActiveRenderBackend(pub Option<Box<dyn crate::gfx::backend::RenderBackend>>);

impl ActiveRenderBackend {
    // Take the parked backend for the duration of one system step.
    pub(crate) fn take(
        resources: &mut Resources,
    ) -> Option<Box<dyn crate::gfx::backend::RenderBackend>> {
        resources.get_mut::<Self>()?.0.take()
    }

    // Park the backend again at the end of the step that took it.
    pub(crate) fn put(
        resources: &mut Resources,
        backend: Box<dyn crate::gfx::backend::RenderBackend>,
    ) {
        match resources.get_mut::<Self>() {
            Some(slot) => slot.0 = Some(backend),
            None => {
                resources.insert(Self(Some(backend)));
            }
        }
    }
}

// The active scene-flow bookkeeping, shared between SettingsSystem (which
// applies imperative scene jumps from `SceneCommand`) and GraphicsSystem
// (which ticks the timed advance + fades and submits the visibility changes).
// Published by GraphicsSystem's init when the world declares `Scene` assets;
// `flow` is `None` when it declared none, so both systems no-op. `epoch` is the
// shared clock both derive their `elapsed` from, set to GraphicsSystem's own
// `start_time` so fade timing matches the render clock.
pub(crate) struct ActiveSceneFlow {
    pub flow: Option<crate::gfx::scene_flow::SceneFlow>,
    pub(crate) epoch: std::time::Instant,
}

/// The blob's baked per-scene exclusive content groups, published at blob load
/// for the streaming/residency wiring to consume at graphics init.
pub struct BlobSceneGroups(pub Vec<crate::ecs::SceneGroup>);

/// The blob's baked per-mesh geometry summaries (AABB + counts by mesh-source
/// handle), published at blob load so graphics init can build draw records for
/// deferred scene-owned meshes without decoding their payloads.
pub struct BlobMeshBounds(pub Vec<MeshBoundsRecord>);

// Per-scene streamed-content load status, republished by StreamingSystem
// whenever it changes: `(scene, state, fraction of members resident)` in
// declaration order. Consumers (menus, loading screens) read, never write.
pub(crate) struct SceneResidencyStatus {
    pub scenes: Vec<(
        asset_id::AssetId,
        crate::gfx::scene_residency::SceneLoadState,
        f32,
    )>,
}

// Setting rows the engine has disabled at runtime (their keys, e.g. `show_fps`
// while "Display performance stats" is off). Published each frame by
// GraphicsSystem and read by `UiInputSystem`, which makes a matching row inert
// (no hover, no click) while its labels are grayed independently. Distinct from
// the init-time capability gating (which marks `HitRegion.disabled` before the
// regions are drained); this drives the same effect after they are drained.
#[derive(Debug, Clone, Default)]
pub(crate) struct DisabledSettingRows(pub std::collections::HashSet<String>);

// The display modes offered by the "Resolution" settings row, published once by
// GraphicsSystem at init (enumerated from the backend's display, or the static
// fallback when it cannot enumerate) and read by `UiInputSystem` to seed the
// row's dropdown list. Ordered as displayed; a pick's `SetIndex` indexes it.
#[derive(Debug, Clone, Default)]
pub(crate) struct DisplayModes(pub Vec<crate::gfx::display_mode::DisplayMode>);

/// The system table. Generates the `SYSTEMS` table a world starts from; table
/// order is run order.
///
/// The two leading fields are the load-time passes that bracket the systems:
/// one that runs over the world once the gates have built them and before
/// their `init`, and one that pre-creates the event queues a scheduled
/// system's declared access can touch.
///
/// Every system is internal: it has no declarable asset, is never parsed from a
/// world or written to a blob, and is constructed by its gate from world
/// content. Each entry maps a name to the behavior type that implements
/// `System`, the gate that builds it, and a human-readable gate description;
/// the entry name doubles as the system's stable display name for profiling and
/// logging.
#[macro_export]
macro_rules! define_systems {
    ( before_init: $before_init:path,
      prepare_events: $prepare_events:path,
      $( $name:ident => $behavior:path {
            gate: $gate:path,
            present_when: $present_when:literal,
            after: [ $( $after:ident ),* $(,)? ],
            before: [ $( $before:ident ),* $(,)? ] $(,)?
        } ),* $(,)? ) => {
        /// The system table: one entry per system, in run order, plus the
        /// load-time passes that bracket them. `World::start` runs each gate
        /// against the world's content and builds the systems they return.
        pub const SYSTEMS: &$crate::ecs::SystemTable = &$crate::ecs::SystemTable {
            entries: &[
                $( $crate::ecs::SystemEntry {
                    name: stringify!($name),
                    present_when: $present_when,
                    // Boxing happens here rather than in the gates, so each
                    // gate returns its own system type and the entry's behavior
                    // path has to name it.
                    gate: {
                        fn build(
                            world: &$crate::ecs::World,
                        ) -> Option<::std::boxed::Box<dyn $crate::ecs::System>> {
                            let built: Option<$behavior> = $gate(world);
                            built.map(|s| -> ::std::boxed::Box<dyn $crate::ecs::System> {
                                ::std::boxed::Box::new(s)
                            })
                        }
                        build
                    },
                    after: &[ $( stringify!($after) ),* ],
                    before: &[ $( stringify!($before) ),* ],
                }, )*
            ],
            before_init: Some($before_init),
            prepare_events: Some($prepare_events),
        };
    };
}

//! Client-side ecs runtime. The renderer-free metadata, asset registry,
//! registration macros, asset-construction API, `PipelineContext`, and the
//! `World`'s data half all live in concinnity-core; this module re-exports them
//! under the historical `crate::ecs::*` paths and adds the runtime behavior
//! half: the `System` behavior trait, `StepResult`, the `SystemAsset` value
//! enum (generated from `System` in `registry`), the unified `Asset` handle,
//! and the `World` that carries the constructed systems and their schedule over
//! that data.
//!
//! TO ADD A NEW COMPONENT: register it in concinnity-core's `ecs::registry`
//! (`define_components!`). TO ADD A NEW SYSTEM: implement the `System` behavior
//! trait on it, write its gate in this crate's `ecs::schedule`, and add one
//! entry to the `define_systems!` table in `ecs::registry` -- the table is the
//! registry AND the schedule (table order is run order).

pub(crate) mod access_ids;
pub(crate) mod by_asset_id;
#[cfg(test)]
mod consumed_columns_tests;
pub(crate) mod decompose;
#[cfg(test)]
mod determinism_tests;
mod registry;
pub mod schedule;
pub(crate) mod waves;

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

// The name interner keeps a per-thread table, so it lives in concinnity-cpu;
// its module re-exports the vocabulary's `AssetId` / `AssetRef` alongside.
pub use concinnity_cpu::ecs::asset_id;

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

// The `SystemAsset` value enum and the `SYSTEMS` schedule manifest are
// generated client-side from the system table (see `registry`).
pub use registry::{SYSTEMS, SystemAsset};

use crate::blob::BlobData;
use crate::gfx::profile::FrameProfile;
use crate::result::CnResult;

// The `System` behavior trait + its `StepResult` control signal are renderer-free
// (they name only `PipelineContext`), so they live in concinnity-core; re-export
// them under the historical `crate::ecs::*` paths for every reader (engine
// systems, the `define_systems!` table, and the editor's hook drive).
pub use concinnity_core::ecs::{StepResult, System};

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
pub(crate) struct InputMailbox(pub Option<concinnity_render::input::InputPacket>);

impl InputMailbox {
    // Deposit a fresh packet, merging onto an unconsumed one.
    pub(crate) fn deposit(&mut self, packet: concinnity_render::input::InputPacket) {
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
    pub ops: concinnity_render::ops::RenderOps,
    pub slots: crate::gfx::render_slots::RenderSlots,
}

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
pub(crate) struct RenderOpFailures(pub Vec<concinnity_render::ops::OpFailure>);

// The pipelined driver's channel pair, published (parked, so the per-step
// take never re-boxes) before the world moves to the simulation thread.
// Present exactly when frames are pipelined: GraphicsSystem's step extracts
// and sends the snapshot through it instead of submitting against a parked
// backend (which the render half owns), and applies the render half's
// feedback. Absent in serial execution.
pub(crate) struct PipelinedFrames(pub Option<PipelineChannels>);

pub(crate) struct PipelineChannels {
    pub(crate) snapshot_tx:
        std::sync::mpsc::SyncSender<concinnity_render::snapshot::RenderSnapshot>,
    pub(crate) feedback_rx: std::sync::mpsc::Receiver<concinnity_render::feedback::FrameFeedback>,
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

/// The system table. Generates the `SystemAsset` value enum that holds a
/// constructed system and dispatches `init` / `step`, plus the `SYSTEMS`
/// schedule manifest (`&[schedule::SystemEntry]`); table order is run order.
///
/// Every system is internal: it has no declarable asset, is never parsed from a
/// world or written to a blob, and is constructed by its gate from world
/// content. Each entry maps a variant name to the behavior type that implements
/// `System`, the gate that builds it, and a human-readable gate description;
/// the variant name doubles as the system's stable display name (`name()`) for
/// profiling and logging.
#[macro_export]
macro_rules! define_systems {
    ( $( $variant:ident => $behavior:path {
            gate: $gate:path,
            present_when: $present_when:literal,
            after: [ $( $after:ident ),* $(,)? ],
            before: [ $( $before:ident ),* $(,)? ] $(,)?
        } ),* $(,)? ) => {
        /// Variant sizes follow the behavior types; boxing them would only move
        /// the per-system state behind a pointer for no real gain here.
        // Gated on where the lint actually fires: the gap between the two
        // largest variants clears the threshold on macOS and Windows but not on
        // Linux.
        #[cfg_attr(
            any(target_os = "macos", target_os = "windows"),
            expect(
                clippy::large_enum_variant,
                reason = "boxing would only move the per-system state behind a pointer"
            )
        )]
        #[derive(Debug)]
        // One variant per registered system, named for that system.
        #[expect(missing_docs, reason = "one variant per registered system, named for that system")]
        pub enum SystemAsset {
            $( $variant($behavior), )*
        }

        impl SystemAsset {
            /// Stable display name used for profiling and logging. Every variant
            /// name is the system's canonical name.
            pub fn name(&self) -> &'static str {
                match self {
                    $( SystemAsset::$variant(_) => stringify!($variant), )*
                }
            }

            /// Run the system's `init`.
            pub fn init(&mut self, ctx: &mut PipelineContext) {
                match self {
                    $( SystemAsset::$variant(s) => s.init(ctx), )*
                }
            }

            /// Run the system's `step`.
            pub fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
                match self {
                    $( SystemAsset::$variant(s) => s.step(ctx), )*
                }
            }

            /// The system's declared data access, consulted at schedule build
            /// (after init). Defaults to exclusive via the `System` trait.
            pub fn access(&self) -> $crate::ecs::Access {
                match self {
                    $( SystemAsset::$variant(s) => $crate::ecs::System::access(s), )*
                }
            }
        }

        $( impl From<$behavior> for SystemAsset { fn from(s: $behavior) -> Self { SystemAsset::$variant(s) } } )*

        /// The schedule manifest: one entry per system, in run order. Drives
        /// `World::build_internal_systems` and `World::system_manifest`.
        pub const SYSTEMS: &[$crate::ecs::schedule::SystemEntry] = &[
            $( $crate::ecs::schedule::SystemEntry {
                name: stringify!($variant),
                present_when: $present_when,
                gate: $gate,
                after: &[ $( stringify!($after) ),* ],
                before: &[ $( stringify!($before) ),* ],
            }, )*
        ];
    };
}

/// A world: its data, the systems built to run over it, and their schedule.
///
/// The data half -- components, resources, events, compiled payloads, the frame
/// profile, and the frame scratch -- is `concinnity_core::ecs::World`, which
/// needs no operating system and can be built from anywhere the asset
/// vocabulary reaches. This adds what runs over it: the constructed
/// `SystemAsset`s, the executable schedule derived from their declared access,
/// and `start` / `step`.
pub struct World {
    data: concinnity_core::ecs::World,
    systems: Vec<SystemAsset>,
    // Set once `build_internal_systems` has run, so a second `start()` on the
    // same world does not append the internal systems twice.
    internal_systems_built: bool,
    // The executable schedule over the built systems: declared ordering edges
    // validated + conflict waves from each system's declared access. Built at
    // the end of `start()` (after init, when data-dependent declarations are
    // final) and rebuilt when a `Done` system leaves the set.
    schedule: Option<waves::ExecSchedule>,
}

// The world must stay movable to the simulation thread; a !Send member in any
// system, component, or resource breaks the pipelined driver's thread handoff.
const _: () = {
    const fn require_send<T: Send>() {}
    require_send::<World>()
};

impl std::fmt::Debug for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("World")
            .field("components", &self.data.component_count())
            .field("systems", &self.systems.len())
            .finish()
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl From<concinnity_core::ecs::World> for World {
    fn from(data: concinnity_core::ecs::World) -> Self {
        Self {
            data,
            systems: Vec::new(),
            internal_systems_built: false,
            schedule: None,
        }
    }
}

impl World {
    /// An empty world, for contexts that have no blob data (e.g. unit tests,
    /// or worlds built entirely from runtime-only assets).
    pub fn new() -> Self {
        concinnity_core::ecs::World::new().into()
    }

    /// A world backed by a compiled blob.
    pub fn from_blob(blob: BlobData) -> Self {
        concinnity_core::ecs::World::from_payloads(Box::new(blob)).into()
    }

    /// Pre-size the component columns from the blob manifest's per-type record
    /// counts, so the bulk `add` loop that follows never reallocates mid-push.
    pub fn reserve_components(&mut self, counts: &[(u8, u32)]) {
        self.data.reserve_components(counts);
    }

    /// Add a component loaded from a blob def, returning its minted entity so
    /// the loaders can index it by name. Systems are not added this way: they
    /// are internal and constructed by `build_internal_systems`.
    pub fn add(&mut self, component: ComponentAsset) -> Entity {
        self.data.add(component)
    }

    /// Add one component to the world.
    ///
    /// Only a [`RuntimeComponent`] can be added: a build-only asset is consumed
    /// by the cook and never reaches a world.
    pub fn add_component<C: RuntimeComponent>(&mut self, c: C) {
        self.data.add_component(c);
    }

    /// Remove and drop every component of type C. Used by `cn editor` to suppress
    /// the world's baked-in `DebugHud` before start, since the editor HUD's own
    /// F1 toggle replaces it.
    pub fn remove_all<C: ComponentSlot>(&mut self) {
        self.data.remove_all::<C>();
    }

    /// Whether the world holds no components.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty() && self.systems.is_empty()
    }

    // Whether this world drives the renderer. True when it declares a
    // `GraphicsConfig` (pre-`start`) or has a constructed `GraphicsSystem`
    // (post-`start`, after the config component has been drained), so callers
    // can decide on the render loop regardless of timing.
    /// Whether the world needs a renderer.
    pub fn renders(&self) -> bool {
        self.query::<crate::components::GraphicsConfig>()
            .next()
            .is_some()
            || self
                .systems
                .iter()
                .any(|s| matches!(s, SystemAsset::GraphicsSystem(_)))
    }

    /// Components across every typed column.
    pub fn component_count(&self) -> usize {
        self.data.component_count()
    }

    /// Systems built for this world.
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }

    /// Iterate every stored component of a given type. Mirrors
    /// `PipelineContext::query`; useful in tests that hold a `World` directly.
    pub fn query<C: ComponentSlot>(&self) -> std::slice::Iter<'_, C> {
        self.data.query::<C>()
    }

    /// Mutable iteration over all components of type C. Mirror of
    /// `PipelineContext::query_mut` for code holding a `World` directly rather
    /// than a per-system `PipelineContext`, namely the `DebugHook::tick`
    /// drive, which applies hot-reload skeleton-shape changes to the ECS-owned
    /// `SkeletonPose` components from outside the system step.
    pub fn query_mut<C: ComponentSlot>(&mut self) -> std::slice::IterMut<'_, C> {
        self.data.query_mut::<C>()
    }

    /// Push a runtime-produced component into the matching typed slot,
    /// returning its minted entity. Mirror of `PipelineContext::push`; used by
    /// the `DebugHook::tick` drive to insert `Prop`s added by a world.jsonl
    /// hot-reload so subsequent systems see them.
    pub fn push<C: ComponentSlot>(&mut self, c: C) -> Entity {
        self.data.push(c)
    }

    /// Borrow one entity's component, for code holding a `World` directly.
    /// Mirror of `PipelineContext::get`; the editor's gizmo drive reads the
    /// selected entity's transforms through this.
    pub fn get<C: ComponentSlot>(&self, entity: Entity) -> Option<&C> {
        self.data.get::<C>(entity)
    }

    /// Mutably borrow one entity's component. Mirror of
    /// `PipelineContext::get_mut`; the editor's gizmo drag moves the selected
    /// entity's `Transform` through this.
    pub fn get_mut<C: ComponentSlot>(&mut self, entity: Entity) -> Option<&mut C> {
        self.data.get_mut::<C>(entity)
    }

    /// Add a component to an existing entity. Mirror of
    /// `PipelineContext::insert`; the editor's billboard drive seeds a
    /// `Transform` onto non-rendering entities through this.
    pub fn insert<C: ComponentSlot>(&mut self, entity: Entity, c: C) {
        self.data.insert(entity, c);
    }

    /// Whether an entity is still live. Mirror of `PipelineContext::is_alive`;
    /// guards name-index resolves against entities despawned by the start-time
    /// drains (Window, GraphicsConfig, Scene, ...).
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.data.is_alive(entity)
    }

    /// Read-only join over two component types, for code holding a `World`
    /// directly (the decomposition round-trip tests). Mirror of
    /// `PipelineContext::join2`.
    pub fn join2<A: ComponentSlot, B: ComponentSlot>(
        &self,
    ) -> impl Iterator<Item = (Entity, &A, &B)> {
        self.data.join2::<A, B>()
    }

    /// Borrow the event queue for event type E, if any have been sent. Mirror of
    /// `PipelineContext::events`, for code holding a `World` directly (tests).
    pub fn events<E: 'static>(&self) -> Option<&Events<E>> {
        self.data.events::<E>()
    }

    /// Mutably borrow (creating if absent) the event queue for event type E.
    /// Mirror of `PipelineContext::events_mut`, for code holding a `World`
    /// directly: tests, and the editor's debug-driven command injection.
    pub fn events_mut<E: Send + 'static>(&mut self) -> &mut Events<E> {
        self.data.events_mut::<E>()
    }

    /// The world's systems, in schedule order.
    pub fn systems(&self) -> &[SystemAsset] {
        &self.systems
    }

    /// Per-pool `(resident, pending, unloaded)` streaming counts from the parked
    /// `StreamingState` (StreamingSystem drives it against the backend each
    /// frame). `None` before graphics init parks it, and from inside a system
    /// step, which takes the state out. Read by the `cn debug` server's
    /// `streaming` command and the editor's Health panel.
    pub fn streaming_stats(&self) -> Option<crate::gfx::streaming_system::StreamingStats> {
        self.resource::<crate::gfx::streaming_system::StreamingState>()
            .map(|s| s.streaming_stats())
    }

    /// Live process-RAM back-off pressure on streaming, published by
    /// StreamingSystem on its throttled RSS sample. `None` before the first
    /// sample or when no `MemoryBudget` / RSS is available (the valve is inert).
    /// Read by the `cn debug` server's `streaming` command; unused from the
    /// client itself.
    pub fn streaming_pressure(&self) -> Option<crate::gfx::streaming_system::StreamingPressure> {
        self.resource::<crate::gfx::streaming_system::StreamingPressure>()
            .copied()
    }

    /// Long-session memory drift, folded from the same throttled sample as the
    /// back-off valve. `None` until the session settles enough for a baseline,
    /// and for the same reasons `streaming_pressure` is absent.
    pub fn memory_drift(&self) -> Option<crate::app::mem_drift::MemoryDrift> {
        self.resource::<crate::app::mem_drift::MemoryDrift>()
            .copied()
    }

    /// The detected GPU's capability + memory profile, published by graphics
    /// init. `None` before init runs, and `GpuProfile::UNKNOWN` when the backend
    /// could not classify the device.
    pub fn gpu_profile(&self) -> Option<crate::gfx::backend::GpuProfile> {
        self.resource::<crate::gfx::backend::GpuProfile>().copied()
    }

    /// The process thread + memory budgets App published at start. `None` before
    /// `App::start` installs them. Read by the `cn debug` server's `budget`
    /// command; unused from the client itself.
    pub fn thread_budget(&self) -> Option<crate::app::budget::ThreadBudget> {
        self.resource::<crate::app::budget::ThreadBudget>().copied()
    }

    /// The world's memory budget, once `start` has published one.
    pub fn memory_budget(&self) -> Option<crate::app::budget::MemoryBudget> {
        self.resource::<crate::app::budget::MemoryBudget>().copied()
    }

    /// Take the live render backend out of this world's parked slot, leaving
    /// the world backend-less. The `cn editor` live SAVE swap transplants it into
    /// the rebuilt world (via a `PendingBackend` resource) so the edit applies
    /// without recreating the OS window / re-initialising the GPU device. `None`
    /// when the world never built a backend (or it was already yielded).
    ///
    pub fn take_render_backend(&mut self) -> Option<Box<dyn crate::gfx::backend::RenderBackend>> {
        self.data
            .resource_mut::<ActiveRenderBackend>()
            .and_then(|slot| slot.0.take())
    }

    /// Disjoint mutable borrows of the system list and the parked render
    /// backend, for the `cn debug` hot-reload drive: it applies backend edits
    /// through a system's init-captured bookkeeping, so it needs both at once.
    /// The backend is `None` while a step has it taken (never the case between
    /// ticks, where the drive runs) or when no backend was built.
    ///
    pub fn systems_and_render_backend(
        &mut self,
    ) -> (
        &mut [SystemAsset],
        Option<&mut (dyn crate::gfx::backend::RenderBackend + 'static)>,
    ) {
        let backend = self
            .data
            .resource_mut::<ActiveRenderBackend>()
            .and_then(|slot| slot.0.as_deref_mut());
        (&mut self.systems, backend)
    }

    /// Despawn an entity (all its components, recycling its id). Stands in for the
    /// GraphicsSystem-mediated despawn in system tests that need an entity gone
    /// before a later system step (e.g. physics-body reaping).
    #[cfg(test)]
    pub fn despawn(&mut self, entity: Entity) {
        self.data.despawn(entity);
    }

    /// Seed (or replace) a singleton resource that persists across steps. The
    /// `cn editor` drive publishes `MenuOverride` through this each frame; it also
    /// stands in for the render-block-published resources (e.g. OverlaySystem's
    /// `MenuActive`) in system tests that drive a later system directly.
    pub fn insert_resource<T: std::any::Any + Send>(&mut self, value: T) {
        self.data.insert_resource(value);
    }

    /// Borrow a published singleton resource. The App-level frame pacer reads
    /// the pacing state through this before each step; system tests use it for
    /// assertions (e.g. the `OpenDropdown` UiInputSystem publishes each step).
    pub fn resource<T: std::any::Any>(&self) -> Option<&T> {
        self.data.resource::<T>()
    }

    /// Withdraw a published singleton resource. Presence-keyed protocols (the
    /// `cn editor` drive's `TraceRequest`) turn off by removing their resource,
    /// so the reading system pays nothing beyond noticing the absence.
    pub fn remove_resource<T: std::any::Any>(&mut self) -> Option<T> {
        self.data.remove_resource::<T>()
    }

    /// Mutable view of the active systems. Mirror of `systems()`; lets the
    /// `DebugHook::tick` drive match out a `&mut GraphicsSystem` /
    /// `&mut AnimationSystem` (the same enum-match `systems()` already serves
    /// read-only) to drive hot-reload from outside the per-system step.
    pub fn systems_mut(&mut self) -> &mut [SystemAsset] {
        &mut self.systems
    }

    /// How many components of each type the world holds, one entry per
    /// populated type. Systems are internal and never counted here.
    pub fn component_census(&self) -> Vec<(u8, u32)> {
        self.data.component_census()
    }

    /// Build the world's internal systems and run their `init`.
    pub fn start(&mut self) -> Result<(), CnResult> {
        self.build_internal_systems();
        let mut ctx = self.data.context();
        // Give each loaded Prop's entity its per-instance components before
        // systems init, draining the Prop itself: the decomposed components are
        // the only path from here on.
        decompose::run(&mut ctx);
        for system in &mut self.systems {
            system.init(&mut ctx);
        }
        // Every system has inited and cached the payloads it keeps; nothing
        // reads compiled payloads at runtime. Free every blob section still
        // resident: the shipped runtime's blob 0, the audio / SDF / terrain
        // blobs the GraphicsSystem init sweep held back for their later
        // consumers, and every blob in a world with no GraphicsSystem to run
        // that sweep at all.
        let freed = self.data.release_payloads();
        if freed >= 1024 * 1024 {
            tracing::info!(
                "World: freed {} MiB of resident blob payloads after init",
                freed / (1024 * 1024)
            );
        }
        // Access declarations are final once every system has inited, so this
        // is the earliest the edges can be validated and the waves derived.
        let schedule = waves::build(&self.systems);
        // Pre-create the event queues declared systems can touch, so their
        // `events_mut` never grows the store's map mid-tick.
        if !schedule.is_empty() {
            let store = self.data.event_store();
            for i in 0..schedule.len() {
                access_ids::ensure_event_queues(store, schedule.access(i));
            }
        }
        self.schedule = Some(schedule);
        #[cfg(debug_assertions)]
        access_ids::install_hook();
        Ok(())
    }

    // Construct the internal systems implied by the world's content, in their
    // fixed run order, just before `init`. Internal systems are not declarable
    // assets: each is present only when its gating components are, and is built
    // from them by its gate in the `SYSTEMS` table (see `registry` for the
    // schedule and its ordering constraints). Runs at most once per world
    // (guarded by `internal_systems_built`) so a system whose gating components
    // survive `init` is not built twice.
    fn build_internal_systems(&mut self) {
        if self.internal_systems_built {
            return;
        }
        self.internal_systems_built = true;
        for entry in SYSTEMS {
            if let Some(system) = (entry.gate)(&*self) {
                self.systems.push(system);
            }
        }
    }

    /// The system names the gated schedule would build for this world's current
    /// content, in run order. Runs the same `SYSTEMS` gates `start()` runs, so
    /// cook / editor / CLI reporting cannot drift from the runtime; the probe
    /// constructs and discards each gated system, which is why constructors
    /// must stay cheap and side-effect-free (see `schedule`). Reflects the
    /// running binary (DebugHud gates on the dev profile) and the pre-`start`
    /// content: after `start` drains gating components, it reports the systems
    /// a rebuild of the CURRENT content would get, not the built set.
    pub fn system_manifest(&self) -> Vec<&'static str> {
        SYSTEMS
            .iter()
            .filter(|entry| (entry.gate)(self).is_some())
            .map(|entry| entry.name)
            .collect()
    }

    /// Per-frame profiling data: system CPU timings and render-backend stats
    /// from the most recently completed frame.
    pub fn profile(&self) -> &FrameProfile {
        self.data.profile()
    }

    /// Tick -- systems run in order, Done systems are removed.
    /// Returns Done when no systems remain, Stop on hard halt.
    pub fn step(&mut self) -> StepResult {
        // Dev builds sample the tracked heap around the frame and each system
        // step, so per-frame allocation churn is visible in the profile. The
        // counters are process-wide: a delta includes concurrent threads
        // (streaming workers, the pipelined render half), so per-system
        // attribution is approximate while the frame total is exact churn.
        #[cfg(debug_assertions)]
        let frame_alloc_start = concinnity_memory::alloc_count();
        // Rotate the profiler's system-timing buffers so the frame that just
        // finished becomes the readable snapshot for this frame's readers.
        self.data.profile_mut().begin_frame();
        // Advance every event queue once per frame, before systems run, so each
        // queue's two-frame retention holds for readers that run after the
        // writer.
        self.data.update_events();
        // Hand the whole frame's scratch back before anything runs.
        self.data.reset_scratch();
        let mut ctx = self.data.context();
        let mut i = 0;
        let mut removed_any = false;
        while i < self.systems.len() {
            let name = self.systems[i].name();
            let started = std::time::Instant::now();
            #[cfg(debug_assertions)]
            let alloc_start = concinnity_memory::alloc_count();
            #[cfg(debug_assertions)]
            access_ids::set_active(Some((self.systems[i].access(), name)));
            let result = self.systems[i].step(&mut ctx);
            #[cfg(debug_assertions)]
            access_ids::set_active(None);
            let micros = started.elapsed().as_micros().min(u32::MAX as u128) as u32;
            ctx.profile.record_system(name, micros);
            #[cfg(debug_assertions)]
            if let (Some(start), Some(end)) = (alloc_start, concinnity_memory::alloc_count()) {
                ctx.profile.record_system_allocs(
                    name,
                    end.saturating_sub(start).min(u32::MAX as u64) as u32,
                );
            }
            match result {
                StepResult::Stop => return StepResult::Stop,
                StepResult::Done => {
                    let removed = self.systems.remove(i);
                    removed_any = true;
                    tracing::debug!("System '{}' finished", removed.name());
                }
                StepResult::Continue => {
                    i += 1;
                }
            }
        }
        if removed_any && self.schedule.is_some() {
            self.schedule = Some(waves::build(&self.systems));
        }
        self.report_scratch_overflow();
        #[cfg(debug_assertions)]
        if let (Some(start), Some(end)) = (frame_alloc_start, concinnity_memory::alloc_count()) {
            self.data
                .profile_mut()
                .set_frame_allocs(end.saturating_sub(start).min(u32::MAX as u64) as u32);
        }
        if self.systems.is_empty() {
            StepResult::Done
        } else {
            StepResult::Continue
        }
    }

    // A frame that outgrew the scratch reserve fell back to the heap and still
    // rendered, so nothing breaks -- but a silent fallback reads as "the reserve
    // is sized right" when it is not. Reported once per frame rather than per
    // declined request, and only while the count is climbing, so a world that
    // is permanently too small does not fill the log.
    fn report_scratch_overflow(&mut self) {
        let overflows = self.data.take_scratch_overflows();
        if overflows == 0 {
            return;
        }
        let stats = self.data.scratch_stats();
        tracing::warn!(
            "frame scratch overflowed {overflows} time(s): reserve {} KiB, peak {} KiB",
            stats.capacity / 1024,
            stats.peak / 1024,
        );
    }

    /// What the frame scratch cost and whether it was big enough, for the
    /// `memory` query and the Health panel. `peak` is what sizes the reserve.
    pub fn scratch_stats(&self) -> ScratchStats {
        self.data.scratch_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A GraphicsConfig marks a rendering world. `renders()` reports it before
    // `start()` (while the component is present), the pre-start signal callers
    // use to choose the render loop. (The post-start GraphicsSystem path can't
    // be unit-tested here: its `init` builds the GPU backend.)
    #[test]
    fn graphics_config_makes_world_render() {
        let mut world = World::new();
        assert!(!world.renders());
        world.add_component(crate::components::GraphicsConfig::default());
        assert!(world.renders());
    }

    // The overlay HUD components each gate their internal system and build in
    // the fixed schedule order (StatHud, then DebugHud, then FpsCounter).
    // DebugHud is developer-only but `cfg!(debug_assertions)` holds under test.
    #[test]
    fn hud_components_spawn_in_schedule_order() {
        use crate::components::{DebugHud, FpsCounter, StatHud};

        let mut world = World::new();
        world.add_component(FpsCounter::default());
        world.add_component(StatHud::default());
        world.add_component(DebugHud::default());
        world.start().unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["StatHud", "DebugHud", "FpsCounter"]);
    }

    // The manifest reports exactly the systems `start()` builds, in the same
    // order, for a world gating several table entries. Audio is left ungated
    // so `start()` opens no device here.
    #[test]
    fn system_manifest_matches_started_systems() {
        use crate::components::{DebugHud, FpsCounter, StatHud, Story, TextInput};

        let mut world = World::new();
        world.add_component(StatHud::default());
        world.add_component(DebugHud::default());
        world.add_component(FpsCounter::default());
        world.add_component(Story::default());
        world.add_component(TextInput::default());

        let manifest = world.system_manifest();
        world.start().unwrap();
        let built: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(manifest, built);
    }

    // Manifest names come out in table order, and every name is a real table
    // entry (the manifest is a filtered view of `SYSTEMS`, nothing else).
    #[test]
    fn system_manifest_is_a_table_order_subset() {
        use crate::components::{FpsCounter, StatHud};

        let mut world = World::new();
        world.add_component(FpsCounter::default());
        world.add_component(StatHud::default());

        let table: Vec<&str> = SYSTEMS.iter().map(|e| e.name).collect();
        let manifest = world.system_manifest();
        let mut cursor = table.iter();
        for name in &manifest {
            assert!(
                cursor.any(|t| t == name),
                "'{name}' out of table order or unknown: {manifest:?}"
            );
        }
    }

    // A GraphicsConfig world gates the whole render band, and StreamingSystem
    // runs immediately before GraphicsSystem so its `CameraRelativeView` is
    // ready for that frame's submit. (Manifest-only: gating a GraphicsConfig
    // never builds a GPU, unlike `start()`.)
    #[test]
    fn streaming_runs_immediately_before_graphics() {
        let mut world = World::new();
        world.add_component(crate::components::GraphicsConfig::default());
        let manifest = world.system_manifest();
        let s = manifest
            .iter()
            .position(|n| *n == "StreamingSystem")
            .expect("StreamingSystem present for a GraphicsConfig world");
        let g = manifest
            .iter()
            .position(|n| *n == "GraphicsSystem")
            .expect("GraphicsSystem present for a GraphicsConfig world");
        assert_eq!(
            g,
            s + 1,
            "StreamingSystem is directly before GraphicsSystem: {manifest:?}"
        );
    }

    // The two camera-controller entries are mutually exclusive: the first
    // controlled camera's `follow` block picks exactly one of them.
    #[test]
    fn camera_controller_gates_are_exclusive() {
        use crate::components::{Camera3D, CameraController, FollowController};

        let mut fly_cam = Camera3D::bake(Default::default());
        fly_cam.controller = Some(CameraController::default());
        let mut fly = World::new();
        fly.add_component(fly_cam);
        assert_eq!(fly.system_manifest(), ["Camera3DSystem"]);

        let mut follow_cam = Camera3D::bake(Default::default());
        follow_cam.controller = Some(CameraController {
            follow: Some(FollowController::default()),
            ..Default::default()
        });
        let mut follow = World::new();
        follow.add_component(follow_cam);
        assert_eq!(follow.system_manifest(), ["ThirdPersonSystem"]);
    }

    // An audio-gating component is visible in the manifest without a device:
    // the gate probe constructs the system, and device acquisition waits for
    // `System::init`.
    #[test]
    fn audio_gate_probes_without_a_device() {
        let mut world = World::new();
        world.add_component(crate::components::AudioEmitter::default());
        assert_eq!(world.system_manifest(), ["AudioSystem"]);
    }

    // Every table entry carries a non-empty human-readable gate description.
    #[test]
    fn every_system_entry_documents_its_gate() {
        for entry in SYSTEMS {
            assert!(
                !entry.present_when.is_empty(),
                "{} has no present_when",
                entry.name
            );
        }
    }

    // A Story gates the StorySystem. An empty-node story pulls in no audio
    // device (build_audio needs a page/choice cue), so this stays device-free.
    #[test]
    fn story_component_spawns_story_system() {
        let mut world = World::new();
        world.add_component(crate::components::Story::default());
        world.start().unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["StorySystem"]);
    }

    // Every event queue rotates each step, whatever its type: an event sent
    // once retires after two steps, so a queue written every frame stays
    // bounded at two frames of events instead of growing for the session.
    #[test]
    fn event_queues_rotate_every_step() {
        struct Ping;
        struct Pong;

        let mut world = World::new();
        world.start().unwrap();
        for _ in 0..5 {
            world.events_mut::<Ping>().send(Ping);
            world.events_mut::<Pong>().send(Pong);
            world.step();
        }
        for len in [
            world.events::<Ping>().expect("Ping queue exists").len(),
            world.events::<Pong>().expect("Pong queue exists").len(),
        ] {
            assert!(len <= 2, "queue holds at most two frames of events: {len}");
        }
    }

    // The World Debug impl reports component and system counts rather than
    // dumping their contents.
    #[test]
    fn world_debug_impl_reports_counts() {
        use crate::components::TextLabel;

        let mut world = World::new();
        world.add_component(TextLabel::default());
        world.add_component(TextLabel::default());
        let text = format!("{world:?}");
        assert!(text.contains("World"), "{text}");
        assert!(text.contains("components: 2"), "{text}");
        assert!(text.contains("systems: 0"), "{text}");
    }

    // A fresh world holds nothing; adding a component (through either the blob
    // path or the typed one) fills it, and `start()` is what gives it systems.
    #[test]
    fn empty_world_fills_from_components_then_systems() {
        use crate::components::{FpsCounter, TextLabel};

        let mut world = World::new();
        assert!(world.is_empty());
        assert_eq!(world.component_count(), 0);
        assert_eq!(world.system_count(), 0);

        world.add(ComponentAsset::from(TextLabel::default()));
        assert!(!world.is_empty());
        assert_eq!(world.component_count(), 1);

        world.add_component(FpsCounter::default());
        world.start().unwrap();
        assert_eq!(world.system_count(), 1, "the FpsCounter gate built one");
    }

    // `push` lands a runtime-produced component in its typed slot, where the
    // matching query finds it.
    #[test]
    fn push_lands_in_the_typed_slot() {
        use crate::components::TextLabel;

        let mut world = World::new();
        world.push(TextLabel {
            content: "pushed".to_string(),
            ..Default::default()
        });

        let found: Vec<&str> = world
            .query::<TextLabel>()
            .map(|l| l.content.as_str())
            .collect();
        assert_eq!(found, ["pushed"]);
    }

    // `remove_all` drops every component of one type and leaves the others,
    // which is how the editor suppresses a world's baked-in HUD before start.
    #[test]
    fn remove_all_drops_only_the_named_type() {
        use crate::components::{DebugHud, TextLabel};

        let mut world = World::new();
        world.add_component(TextLabel::default());
        world.add_component(DebugHud::default());
        assert_eq!(world.component_count(), 2);

        world.remove_all::<DebugHud>();
        assert_eq!(world.query::<DebugHud>().count(), 0);
        assert_eq!(world.query::<TextLabel>().count(), 1, "others survive");
    }

    // The census carries one counted entry per populated component type, so the
    // debug server can report the world's makeup without reading their contents
    // and without a per-instance entry.
    #[test]
    fn component_census_counts_one_entry_per_populated_type() {
        use crate::components::TextLabel;

        let mut world = World::new();
        assert!(world.component_census().is_empty());

        world.add_component(TextLabel::default());
        world.add_component(TextLabel::default());
        world.add_component(crate::components::FpsCounter::default());

        let census = world.component_census();
        assert_eq!(census.len(), 2, "one entry per type, not per component");
        let mut counts: Vec<u32> = census.iter().map(|&(_, n)| n).collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![1, 2], "the two labels share one counted entry");
        assert_eq!(
            census.iter().map(|&(_, n)| n).sum::<u32>(),
            3,
            "every stored component is accounted for",
        );
    }

    // Despawning an entity removes every component on it: decompose drains the
    // authored Prop into per-entity components, and the despawn takes them all.
    #[test]
    fn despawn_removes_an_entitys_components() {
        use crate::components::{MeshRenderer, Prop, Transform};

        let mut world = World::new();
        world.add_component(Prop::default());
        // start() runs decompose, which gives the Prop's entity its Transform +
        // MeshRenderer.
        world.start().unwrap();
        let entity = world
            .join2::<Transform, MeshRenderer>()
            .next()
            .map(|(e, _, _)| e)
            .expect("the Prop decomposed onto one entity");

        world.despawn(entity);
        assert_eq!(world.query::<Transform>().count(), 0);
        assert_eq!(world.query::<MeshRenderer>().count(), 0);
    }

    // The frame profile is exposed for the debug server's readout and holds no
    // timings before any step has run.
    #[test]
    fn profile_is_exposed_before_any_step() {
        let world = World::new();
        assert!(world.profile().system_timings().is_empty());
    }

    // The streaming readouts are `None` until graphics init parks the state, so
    // a world that never built a backend reports nothing rather than panicking.
    #[test]
    fn streaming_readouts_are_absent_before_graphics_init() {
        let world = World::new();
        assert!(world.streaming_stats().is_none());
        assert!(world.streaming_pressure().is_none());
    }

    // A world that never built a backend has none to yield, and the disjoint
    // borrow still hands back the (empty) system list.
    #[test]
    fn render_backend_accessors_without_a_backend() {
        let mut world = World::new();
        assert!(world.take_render_backend().is_none());

        let (systems, backend) = world.systems_and_render_backend();
        assert!(systems.is_empty());
        assert!(backend.is_none());
    }
}

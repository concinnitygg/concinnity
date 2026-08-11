// src/ecs/mod.rs
//
// Client-side ecs runtime. The renderer-free metadata, asset registry,
// registration macros, asset-construction API, and `PipelineContext` all live
// in concinnity-core; this module re-exports them under the historical
// `crate::ecs::*` paths and adds the runtime behavior half: the `System`
// behavior trait, `StepResult`, the `SystemAsset` value enum (generated from
// `System` in `registry`), the unified `Asset` handle, and the `World`.
//
// TO ADD A NEW COMPONENT: register it in concinnity-core's `ecs::registry`
// (`define_components!`). TO ADD A NEW SYSTEM: implement the `System` behavior
// trait on it, write its gate in this crate's `ecs::schedule`, and add one
// entry to the `define_systems!` table in `ecs::registry` -- the table is the
// registry AND the schedule (table order is run order).

pub(crate) mod by_asset_id;
pub(crate) mod decompose;
mod registry;
pub mod schedule;

// Renderer-free metadata, registry types, the asset-construction API, and the
// `PipelineContext`, re-exported from concinnity-core so the rest of the client
// keeps its historical `crate::ecs::*` import paths.
pub use concinnity_core::ecs::{
    Arena, AudioClipHandle, BlobAssetDef, ColumnTicks, Component, ComponentAsset, ComponentSlot,
    ComponentStorage, Entity, EventCursor, EventStore, Events, FontHandle, FrameContext,
    MAX_CHANGE_AGE, MaterialHandle, MeshBoundsRecord, MeshHandle, PayloadLocator, PipelineContext,
    Resources, SceneGroup, SkinnedMeshHandle, TextureHandle, Tick,
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
    OverlayImage, OverlayImages, PickEntry, PickIndex, ScreenStack, SimTiming, TraceEvent,
    TracePath, TracePaths, TraceRequest, TraceStep, TraceVal, TransientSaves, ViewOverrides,
    WorldLines,
};

// The `SystemAsset` value enum and the `SYSTEMS` schedule manifest are
// generated client-side from the system table (see `registry`).
pub use registry::{SYSTEMS, SystemAsset};

use concinnity_memory::MemTag;

use crate::blob::BlobData;
use crate::gfx::profile::FrameProfile;
use crate::result::CnResult;

// The `System` behavior trait + its `StepResult` control signal are renderer-free
// (they name only `PipelineContext`), so they live in concinnity-core; re-export
// them under the historical `crate::ecs::*` paths for every reader (engine
// systems, the `define_systems!` table, and the editor's hook drive).
pub use concinnity_core::ecs::{StepResult, System};

// A render backend transplanted out of a previous world, carried into a freshly
// built world so its GraphicsSystem reuses the live GPU device + window instead
// of constructing a new one. Published by the `cn editor` live SAVE swap between
// building the post-edit world and starting it; GraphicsSystem `run_init` takes
// it and calls `RenderBackend::reload_world` (reusing the window) instead of
// `init_backend`, so a save applies without recreating the OS window. A shipped
// runtime never publishes it; it exists only on the editor's live-update path.
pub struct PendingBackend(pub Box<dyn crate::gfx::backend::RenderBackend>);

// The world's live render backend, parked here between system steps.
// GraphicsSystem's init builds it and parks it; each system that drives the
// GPU (GraphicsSystem's frame encode, InputSystem's poll) takes it out at the
// top of its step and puts it back before returning, so the backend and the
// `PipelineContext` are never borrowed together. `None` while a step has it
// taken, or once the editor's live SAVE transplanted it out.
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
pub struct ActiveSceneFlow {
    pub flow: Option<crate::gfx::scene_flow::SceneFlow>,
    pub epoch: std::time::Instant,
}

// The blob's baked per-scene exclusive content groups, published at blob load
// for the streaming/residency wiring to consume at graphics init.
pub struct BlobSceneGroups(pub Vec<crate::ecs::SceneGroup>);

// The blob's baked per-mesh geometry summaries (AABB + counts by mesh-source
// handle), published at blob load so graphics init can build draw records for
// deferred scene-owned meshes without decoding their payloads.
pub struct BlobMeshBounds(pub Vec<MeshBoundsRecord>);

// Per-scene streamed-content load status, republished by StreamingSystem
// whenever it changes: `(scene, state, fraction of members resident)` in
// declaration order. Consumers (menus, loading screens) read, never write.
pub struct SceneResidencyStatus {
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
pub struct DisabledSettingRows(pub std::collections::HashSet<String>);

// The display modes offered by the "Resolution" settings row, published once by
// GraphicsSystem at init (enumerated from the backend's display, or the static
// fallback when it cannot enumerate) and read by `UiInputSystem` to seed the
// row's dropdown list. Ordered as displayed; a pick's `SetIndex` indexes it.
#[derive(Debug, Clone, Default)]
pub struct DisplayModes(pub Vec<crate::gfx::display_mode::DisplayMode>);

// The system table. Generates the `SystemAsset` value enum that holds a
// constructed system and dispatches `init` / `step`, plus the `SYSTEMS`
// schedule manifest (`&[schedule::SystemEntry]`); table order is run order.
//
// Every system is internal: it has no declarable asset, is never parsed from a
// world or written to a blob, and is constructed by its gate from world
// content. Each entry maps a variant name to the behavior type that implements
// `System`, the gate that builds it, and a human-readable gate description;
// the variant name doubles as the system's stable display name (`name()`) for
// profiling and logging.
#[macro_export]
macro_rules! define_systems {
    ( $( $variant:ident => $behavior:path {
            gate: $gate:path,
            present_when: $present_when:literal $(,)?
        } ),* $(,)? ) => {
        // Variant sizes follow the behavior types; boxing them would only move
        // the per-system state behind a pointer for no real gain here.
        #[allow(clippy::large_enum_variant)]
        #[derive(Debug)]
        pub enum SystemAsset {
            $( $variant($behavior), )*
        }

        impl SystemAsset {
            // Stable display name used for profiling and logging. Every variant
            // name is the system's canonical name.
            pub fn name(&self) -> &'static str {
                match self {
                    $( SystemAsset::$variant(_) => stringify!($variant), )*
                }
            }

            pub fn init(&mut self, ctx: &mut PipelineContext) {
                match self {
                    $( SystemAsset::$variant(s) => s.init(ctx), )*
                }
            }

            pub fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
                match self {
                    $( SystemAsset::$variant(s) => s.step(ctx), )*
                }
            }
        }

        $( impl From<$behavior> for SystemAsset { fn from(s: $behavior) -> Self { SystemAsset::$variant(s) } } )*

        // The schedule manifest: one entry per system, in run order. Drives
        // `World::build_internal_systems` and `World::system_manifest`.
        pub const SYSTEMS: &[$crate::ecs::schedule::SystemEntry] = &[
            $( $crate::ecs::schedule::SystemEntry {
                name: stringify!($variant),
                present_when: $present_when,
                gate: $gate,
            }, )*
        ];
    };
}

// What one frame's scratch reserve cost and whether it held. A non-zero
// `overflows` means some frame fell back to the heap, so `peak` understates
// what the frame actually wanted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScratchStats {
    pub capacity: usize,
    pub peak: usize,
    pub overflows: u64,
}

// The per-frame scratch reserve. An engine constant rather than an authored
// field: a schema field would be blob churn for a knob nobody should have to
// set, and `World::step` reports any frame that outgrows it.
//
// A frame's draw scales with the runtime requests it drains: 2,000 visibility
// requests in one frame measured 24 KiB, so this holds on the order of 87,000.
// `cn debug send '{"cmd":"memory"}'` reports the live peak against it.
const FRAME_SCRATCH_BYTES: usize = 1 << 20;

pub struct World {
    components: ComponentStorage,
    systems: Vec<SystemAsset>,
    blob: BlobData,
    profile: FrameProfile,
    // Type-keyed engine singletons (e.g. the per-frame FrameInput snapshot
    // GraphicsSystem publishes) and the event queues.
    resources: Resources,
    // Per-frame scratch, reset at the top of every `step`. Owned here because
    // `reset` needs `&mut`, which is what proves no system still holds an
    // allocation from the frame just finished.
    scratch: Arena,
    // Requests the scratch reserve could not satisfy, over the world's whole
    // life. The arena's own counter is cleared each frame once reported, so
    // this is what survives to say the reserve wants raising.
    scratch_overflows: u64,
    // Set once `build_internal_systems` has run, so a second `start()` on the
    // same world does not append the internal systems twice.
    internal_systems_built: bool,
}

impl std::fmt::Debug for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("World")
            .field("components", &self.components.len())
            .field("systems", &self.systems.len())
            .finish()
    }
}

impl World {
    pub fn new(blob: BlobData) -> Self {
        Self {
            components: ComponentStorage::default(),
            systems: Vec::new(),
            blob,
            profile: FrameProfile::default(),
            resources: Resources::new(),
            scratch: Arena::tagged(FRAME_SCRATCH_BYTES, MemTag::Scratch),
            scratch_overflows: 0,
            internal_systems_built: false,
        }
    }

    // Convenience constructor for contexts that have no blob data
    // (e.g. unit tests, or worlds built entirely from runtime-only assets).
    pub fn new_empty() -> Self {
        Self::new(BlobData::empty())
    }

    // Pre-size the component columns from the blob manifest's per-type record
    // counts, so the bulk `add` loop that follows never reallocates mid-push.
    pub fn reserve_components(&mut self, counts: &[(u8, u32)]) {
        for &(discriminant, count) in counts {
            self.components.reserve(
                concinnity_core::ecs::ComponentId::new(discriminant),
                count as usize,
            );
        }
    }

    // Add a component loaded from a blob def, returning its minted entity so
    // the loaders can index it by name. Systems are not added this way: they
    // are internal and constructed by `build_internal_systems`.
    pub fn add(&mut self, component: ComponentAsset) -> Entity {
        self.components.push(component)
    }

    pub fn add_component<C: Into<ComponentAsset>>(&mut self, c: C) {
        self.components.push(c.into());
    }

    // Remove and drop every component of type C. Used by `cn editor` to suppress
    // the world's baked-in `DebugHud` before start, since the editor HUD's own
    // F1 toggle replaces it.
    pub fn remove_all<C: ComponentSlot>(&mut self) {
        let _ = self.components.drain::<C>();
    }

    pub fn is_empty(&self) -> bool {
        self.components.is_empty() && self.systems.is_empty()
    }

    // Whether this world drives the renderer. True when it declares a
    // `GraphicsConfig` (pre-`start`) or has a constructed `GraphicsSystem`
    // (post-`start`, after the config component has been drained), so callers
    // can decide on the render loop / Metal activation regardless of timing.
    // Used only on the macOS NSApp-activation path in `app::run` (and in the
    // tests below), so it has no caller on other platforms in a non-test build.
    // Genuinely platform-conditional (unlike the dyn-dispatch dead-code blind
    // spots in the DX backend), so gate the allow on the same condition.
    #[cfg_attr(
        not(target_os = "macos"),
        allow(
            dead_code,
            reason = "used only on the macOS render-activation path in app::run, plus tests"
        )
    )]
    pub fn renders(&self) -> bool {
        self.query::<crate::assets::GraphicsConfig>()
            .next()
            .is_some()
            || self
                .systems
                .iter()
                .any(|s| matches!(s, SystemAsset::GraphicsSystem(_)))
    }

    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    pub fn system_count(&self) -> usize {
        self.systems.len()
    }

    // Iterate every stored component of a given type. Mirrors
    // `PipelineContext::query`; useful in tests that hold a `World` directly.
    pub fn query<C: ComponentSlot>(&self) -> std::slice::Iter<'_, C> {
        C::slot(&self.components).iter()
    }

    // Mutable iteration over all components of type C. Mirror of
    // `PipelineContext::query_mut` for code holding a `World` directly rather
    // than a per-system `PipelineContext`, namely the `DebugHook::tick`
    // drive, which applies hot-reload skeleton-shape changes to the ECS-owned
    // `SkeletonPose` components from outside the system step.
    pub fn query_mut<C: ComponentSlot>(&mut self) -> std::slice::IterMut<'_, C> {
        self.components.values_mut::<C>().iter_mut()
    }

    // Push a runtime-produced component into the matching typed slot,
    // returning its minted entity. Mirror of `PipelineContext::push`; used by
    // the `DebugHook::tick` drive to insert `Prop`s added by a world.jsonl
    // hot-reload so subsequent systems see them.
    pub fn push<C: ComponentSlot>(&mut self, c: C) -> Entity {
        self.components.push_typed(c)
    }

    // Borrow one entity's component, for code holding a `World` directly.
    // Mirror of `PipelineContext::get`; the editor's gizmo drive reads the
    // selected entity's transforms through this.
    pub fn get<C: ComponentSlot>(&self, entity: Entity) -> Option<&C> {
        self.components.get::<C>(entity)
    }

    // Mutably borrow one entity's component. Mirror of
    // `PipelineContext::get_mut`; the editor's gizmo drag moves the selected
    // entity's `Transform` through this.
    pub fn get_mut<C: ComponentSlot>(&mut self, entity: Entity) -> Option<&mut C> {
        self.components.get_mut::<C>(entity)
    }

    // Add a component to an existing entity. Mirror of
    // `PipelineContext::insert`; the editor's billboard drive seeds a
    // `Transform` onto non-rendering entities through this.
    pub fn insert<C: ComponentSlot>(&mut self, entity: Entity, c: C) {
        self.components.insert_typed(entity, c);
    }

    // Whether an entity is still live. Mirror of `PipelineContext::is_alive`;
    // guards name-index resolves against entities despawned by the start-time
    // drains (Window, GraphicsConfig, Scene, ...).
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.components.is_alive(entity)
    }

    // Read-only join over two component types, for code holding a `World`
    // directly (the decomposition round-trip tests). Mirror of
    // `PipelineContext::join2`.
    pub fn join2<A: ComponentSlot, B: ComponentSlot>(
        &self,
    ) -> impl Iterator<Item = (Entity, &A, &B)> {
        self.components.join2::<A, B>()
    }

    // Borrow the event queue for event type E, if any have been sent. Mirror of
    // `PipelineContext::events`, for code holding a `World` directly (tests).
    pub fn events<E: 'static>(&self) -> Option<&Events<E>> {
        self.resources.get::<EventStore>()?.get::<E>()
    }

    // Mutably borrow (creating if absent) the event queue for event type E.
    // Mirror of `PipelineContext::events_mut`, for code holding a `World`
    // directly: tests, and the editor's debug-driven command injection.
    pub fn events_mut<E: 'static>(&mut self) -> &mut Events<E> {
        if !self.resources.contains::<EventStore>() {
            self.resources.insert(EventStore::new());
        }
        self.resources
            .get_mut::<EventStore>()
            .expect("EventStore was just inserted")
            .get_mut_or_create::<E>()
    }

    pub fn systems(&self) -> &[SystemAsset] {
        &self.systems
    }

    // Per-pool `(resident, pending, unloaded)` streaming counts from the parked
    // `StreamingState` (StreamingSystem drives it against the backend each
    // frame). `None` before graphics init parks it, and from inside a system
    // step, which takes the state out. Read by the `cn debug` server's
    // `streaming` command and the editor's Health panel.
    pub fn streaming_stats(&self) -> Option<crate::gfx::streaming_system::StreamingStats> {
        self.resources
            .get::<crate::gfx::streaming_system::StreamingState>()
            .map(|s| s.streaming_stats())
    }

    // Live process-RAM back-off pressure on streaming, published by
    // StreamingSystem on its throttled RSS sample. `None` before the first
    // sample or when no `MemoryBudget` / RSS is available (the valve is inert).
    // Read by the `cn debug` server's `streaming` command; unused from the
    // client itself.
    pub fn streaming_pressure(&self) -> Option<crate::gfx::streaming_system::StreamingPressure> {
        self.resources
            .get::<crate::gfx::streaming_system::StreamingPressure>()
            .copied()
    }

    // Long-session memory drift, folded from the same throttled sample as the
    // back-off valve. `None` until the session settles enough for a baseline,
    // and for the same reasons `streaming_pressure` is absent.
    pub fn memory_drift(&self) -> Option<crate::app::mem_drift::MemoryDrift> {
        self.resources
            .get::<crate::app::mem_drift::MemoryDrift>()
            .copied()
    }

    // The detected GPU's capability + memory profile, published by graphics
    // init. `None` before init runs, and `GpuProfile::UNKNOWN` when the backend
    // could not classify the device.
    pub fn gpu_profile(&self) -> Option<crate::gfx::backend::GpuProfile> {
        self.resources
            .get::<crate::gfx::backend::GpuProfile>()
            .copied()
    }

    // The process thread + memory budgets App published at start. `None` before
    // `App::start` installs them. Read by the `cn debug` server's `budget`
    // command; unused from the client itself.
    pub fn thread_budget(&self) -> Option<crate::app::budget::ThreadBudget> {
        self.resources
            .get::<crate::app::budget::ThreadBudget>()
            .copied()
    }

    pub fn memory_budget(&self) -> Option<crate::app::budget::MemoryBudget> {
        self.resources
            .get::<crate::app::budget::MemoryBudget>()
            .copied()
    }

    // Take the live render backend out of this world's parked slot, leaving
    // the world backend-less. The `cn editor` live SAVE swap transplants it into
    // the rebuilt world (via a `PendingBackend` resource) so the edit applies
    // without recreating the OS window / re-initialising the GPU device. `None`
    // when the world never built a backend (or it was already yielded).
    //
    pub fn take_render_backend(&mut self) -> Option<Box<dyn crate::gfx::backend::RenderBackend>> {
        ActiveRenderBackend::take(&mut self.resources)
    }

    // Disjoint mutable borrows of the system list and the parked render
    // backend, for the `cn debug` hot-reload drive: it applies backend edits
    // through a system's init-captured bookkeeping, so it needs both at once.
    // The backend is `None` while a step has it taken (never the case between
    // ticks, where the drive runs) or when no backend was built.
    //
    pub fn systems_and_render_backend(
        &mut self,
    ) -> (
        &mut [SystemAsset],
        Option<&mut (dyn crate::gfx::backend::RenderBackend + 'static)>,
    ) {
        let backend = self
            .resources
            .get_mut::<ActiveRenderBackend>()
            .and_then(|slot| slot.0.as_deref_mut());
        (&mut self.systems, backend)
    }

    // Despawn an entity (all its components, recycling its id). Stands in for the
    // GraphicsSystem-mediated despawn in system tests that need an entity gone
    // before a later system step (e.g. physics-body reaping).
    #[cfg(test)]
    pub fn despawn(&mut self, entity: Entity) {
        self.components.despawn(entity);
    }

    // Seed (or replace) a singleton resource that persists across steps. The
    // `cn editor` drive publishes `MenuOverride` through this each frame; it also
    // stands in for the render-block-published resources (e.g. OverlaySystem's
    // `MenuActive`) in system tests that drive a later system directly.
    pub fn insert_resource<T: std::any::Any>(&mut self, value: T) {
        self.resources.insert(value);
    }

    // Borrow a published singleton resource. The App-level frame pacer reads
    // the pacing state through this before each step; system tests use it for
    // assertions (e.g. the `OpenDropdown` UiInputSystem publishes each step).
    pub fn resource<T: std::any::Any>(&self) -> Option<&T> {
        self.resources.get::<T>()
    }

    // Withdraw a published singleton resource. Presence-keyed protocols (the
    // `cn editor` drive's `TraceRequest`) turn off by removing their resource,
    // so the reading system pays nothing beyond noticing the absence.
    pub fn remove_resource<T: std::any::Any>(&mut self) -> Option<T> {
        self.resources.remove::<T>()
    }

    // Mutable view of the active systems. Mirror of `systems()`; lets the
    // `DebugHook::tick` drive match out a `&mut GraphicsSystem` /
    // `&mut AnimationSystem` (the same enum-match `systems()` already serves
    // read-only) to drive hot-reload from outside the per-system step.
    pub fn systems_mut(&mut self) -> &mut [SystemAsset] {
        &mut self.systems
    }

    // How many components of each type the world holds, one entry per
    // populated type. Systems are internal and never counted here.
    pub fn component_census(&self) -> Vec<(u8, u32)> {
        self.components.component_census()
    }

    pub fn start(&mut self) -> Result<(), CnResult> {
        self.build_internal_systems();
        let mut ctx = PipelineContext {
            components: &mut self.components,
            blob: &mut self.blob,
            profile: &mut self.profile,
            resources: &mut self.resources,
            frame: crate::ecs::FrameContext::new(&self.scratch),
        };
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
        let freed = self.blob.release_all_resident();
        if freed >= 1024 * 1024 {
            tracing::info!(
                "World: freed {} MiB of resident blob payloads after init",
                freed / (1024 * 1024)
            );
        }
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

    // The system names the gated schedule would build for this world's current
    // content, in run order. Runs the same `SYSTEMS` gates `start()` runs, so
    // cook / editor / CLI reporting cannot drift from the runtime; the probe
    // constructs and discards each gated system, which is why constructors
    // must stay cheap and side-effect-free (see `schedule`). Reflects the
    // running binary (DebugHud gates on the dev profile) and the pre-`start`
    // content: after `start` drains gating components, it reports the systems
    // a rebuild of the CURRENT content would get, not the built set.
    pub fn system_manifest(&self) -> Vec<&'static str> {
        SYSTEMS
            .iter()
            .filter(|entry| (entry.gate)(self).is_some())
            .map(|entry| entry.name)
            .collect()
    }

    // Per-frame profiling data: system CPU timings and render-backend stats
    // from the most recently completed frame.
    pub fn profile(&self) -> &FrameProfile {
        &self.profile
    }

    // Advance every event queue once per frame, before systems run, so each
    // queue's two-frame retention holds for readers that run after the writer.
    // The `EventStore` owns every queue `events_mut` ever created (on `World`
    // or `PipelineContext`), so no per-type rotation list exists to fall out
    // of sync.
    fn update_events(&mut self) {
        if let Some(store) = self.resources.get_mut::<EventStore>() {
            store.update_all();
        }
    }

    // Tick -- systems run in order, Done systems are removed.
    // Returns Done when no systems remain, Stop on hard halt.
    pub fn step(&mut self) -> StepResult {
        // Rotate the profiler's system-timing buffers so the frame that just
        // finished becomes the readable snapshot for this frame's readers.
        self.profile.begin_frame();
        self.update_events();
        // Hand the whole frame's scratch back before anything runs. `&mut self`
        // here is the proof that no allocation from the last frame survives.
        self.scratch.reset();
        let mut ctx = PipelineContext {
            components: &mut self.components,
            blob: &mut self.blob,
            profile: &mut self.profile,
            resources: &mut self.resources,
            frame: crate::ecs::FrameContext::new(&self.scratch),
        };
        let mut i = 0;
        while i < self.systems.len() {
            let name = self.systems[i].name();
            let started = std::time::Instant::now();
            let result = self.systems[i].step(&mut ctx);
            let micros = started.elapsed().as_micros().min(u32::MAX as u128) as u32;
            ctx.profile.record_system(name, micros);
            match result {
                StepResult::Stop => return StepResult::Stop,
                StepResult::Done => {
                    let removed = self.systems.remove(i);
                    tracing::debug!("System '{}' finished", removed.name());
                }
                StepResult::Continue => {
                    i += 1;
                }
            }
        }
        self.report_scratch_overflow();
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
        let overflows = self.scratch.overflows();
        if overflows == 0 {
            return;
        }
        self.scratch.clear_overflows();
        self.scratch_overflows = self.scratch_overflows.saturating_add(overflows as u64);
        tracing::warn!(
            "frame scratch overflowed {overflows} time(s): reserve {} KiB, peak {} KiB",
            self.scratch.capacity() / 1024,
            self.scratch.peak() / 1024,
        );
    }

    // What the frame scratch cost and whether it was big enough, for the
    // `memory` query and the Health panel. `peak` is what sizes the reserve.
    pub fn scratch_stats(&self) -> ScratchStats {
        ScratchStats {
            capacity: self.scratch.capacity(),
            peak: self.scratch.peak(),
            overflows: self.scratch_overflows,
        }
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
        let mut world = World::new_empty();
        assert!(!world.renders());
        world.add_component(crate::assets::GraphicsConfig::default());
        assert!(world.renders());
    }

    // The overlay HUD components each gate their internal system and build in
    // the fixed schedule order (StatHud, then DebugHud, then FpsCounter).
    // DebugHud is developer-only but `cfg!(debug_assertions)` holds under test.
    #[test]
    fn hud_components_spawn_in_schedule_order() {
        use crate::assets::{DebugHud, FpsCounter, StatHud};

        let mut world = World::new_empty();
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
        use crate::assets::{DebugHud, FpsCounter, StatHud, Story, TextInput};

        let mut world = World::new_empty();
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
        use crate::assets::{FpsCounter, StatHud};

        let mut world = World::new_empty();
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
        let mut world = World::new_empty();
        world.add_component(crate::assets::GraphicsConfig::default());
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
        use crate::assets::{Camera3D, CameraController, FollowController};

        let mut fly_cam = Camera3D::bake(Default::default());
        fly_cam.controller = Some(CameraController::default());
        let mut fly = World::new_empty();
        fly.add_component(fly_cam);
        assert_eq!(fly.system_manifest(), ["Camera3DSystem"]);

        let mut follow_cam = Camera3D::bake(Default::default());
        follow_cam.controller = Some(CameraController {
            follow: Some(FollowController::default()),
            ..Default::default()
        });
        let mut follow = World::new_empty();
        follow.add_component(follow_cam);
        assert_eq!(follow.system_manifest(), ["ThirdPersonSystem"]);
    }

    // An audio-gating component is visible in the manifest without a device:
    // the gate probe constructs the system, and device acquisition waits for
    // `System::init`.
    #[test]
    fn audio_gate_probes_without_a_device() {
        let mut world = World::new_empty();
        world.add_component(crate::assets::AudioEmitter::default());
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
        let mut world = World::new_empty();
        world.add_component(crate::assets::Story::default());
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

        let mut world = World::new_empty();
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
        use crate::assets::TextLabel;

        let mut world = World::new_empty();
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
        use crate::assets::{FpsCounter, TextLabel};

        let mut world = World::new_empty();
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
        use crate::assets::TextLabel;

        let mut world = World::new_empty();
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
        use crate::assets::{DebugHud, TextLabel};

        let mut world = World::new_empty();
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
        use crate::assets::TextLabel;

        let mut world = World::new_empty();
        assert!(world.component_census().is_empty());

        world.add_component(TextLabel::default());
        world.add_component(TextLabel::default());
        world.add_component(crate::assets::FpsCounter::default());

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
        use crate::assets::{MeshRenderer, Prop, Transform};

        let mut world = World::new_empty();
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
        let world = World::new_empty();
        assert!(world.profile().system_timings().is_empty());
    }

    // The streaming readouts are `None` until graphics init parks the state, so
    // a world that never built a backend reports nothing rather than panicking.
    #[test]
    fn streaming_readouts_are_absent_before_graphics_init() {
        let world = World::new_empty();
        assert!(world.streaming_stats().is_none());
        assert!(world.streaming_pressure().is_none());
    }

    // A world that never built a backend has none to yield, and the disjoint
    // borrow still hands back the (empty) system list.
    #[test]
    fn render_backend_accessors_without_a_backend() {
        let mut world = World::new_empty();
        assert!(world.take_render_backend().is_none());

        let (systems, backend) = world.systems_and_render_backend();
        assert!(systems.is_empty());
        assert!(backend.is_none());
    }
}

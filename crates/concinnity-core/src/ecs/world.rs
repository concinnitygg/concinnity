//! A world: its data, the systems built to run over it, and their schedule.
//!
//! The data half is components, resources, events, the compiled-payload store,
//! the frame profile, and the frame scratch -- exactly the five things a
//! [`PipelineContext`] borrows, owned in one place. Over it run the systems a
//! host's [`SystemTable`] gates in, in table order, under a schedule derived
//! from what each declares it touches.
//!
//! Building and running one needs no operating system: the two ties a step
//! would otherwise have are seams instead -- the [`Clock`] resource for the
//! per-system profile micros, and the debug-build access validator's hooks in
//! [`access_check`](crate::ecs::access_check).

use alloc::boxed::Box;
use alloc::vec::Vec;
use concinnity_memory::{Arena, MemTag};

use crate::ecs::waves::{self, ExecSchedule};
use crate::ecs::{
    BuiltSystem, Clock, ComponentAsset, ComponentId, ComponentSlot, ComponentStorage, Entity,
    EventStore, Events, FrameContext, NoPayloads, PayloadStore, PipelineContext, Resources,
    RuntimeComponent, StepResult, SystemEntry, SystemTable,
};
use crate::gfx::profile::FrameProfile;
use crate::result::CnResult;

// The per-frame scratch reserve. An engine constant rather than an authored
// field: a schema field would be blob churn for a knob nobody should have to
// set, and the frame loop reports any frame that outgrows it.
//
// A frame's draw scales with the runtime requests it drains: 2,000 visibility
// requests in one frame measured 24 KiB, so this holds on the order of 87,000.
const FRAME_SCRATCH_BYTES: usize = 1 << 20;

/// What one frame's scratch reserve cost and whether it held. A non-zero
/// `overflows` means some frame fell back to the heap, so `peak` understates
/// what the frame actually wanted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScratchStats {
    /// The reserve's size in bytes.
    pub capacity: usize,
    /// The most bytes any frame took from it.
    pub peak: usize,
    /// Requests the reserve declined, sending the caller to the heap.
    pub overflows: u64,
}

/// A world: its component storage, its resources, the compiled payloads it
/// loads from, and the systems that run over all three.
///
/// Constructing one and filling it with components needs no systems, so this is
/// the whole world for any caller that only builds or inspects content.
/// [`start`](World::start) is what gives it systems, from the table the caller
/// hands it.
pub struct World {
    components: ComponentStorage,
    // Compiled payloads, behind the store seam rather than a concrete type, so
    // a world names no blob file format and no filesystem.
    blob: Box<dyn PayloadStore + Send>,
    profile: FrameProfile,
    // Type-keyed engine singletons (e.g. the per-frame FrameInput snapshot
    // GraphicsSystem publishes) and the event queues.
    resources: Resources,
    // Per-frame scratch, reset at the top of every step. Owned here because
    // `reset` needs `&mut`, which is what proves no system still holds an
    // allocation from the frame just finished.
    scratch: Arena,
    // Requests the scratch reserve could not satisfy, over the world's whole
    // life. The arena's own counter is cleared each frame once reported, so
    // this is what survives to say the reserve wants raising.
    scratch_overflows: u64,
    // The systems built for this world, in table order.
    systems: Vec<BuiltSystem>,
    // The table `start` built them from, kept for the schedule rebuild a
    // finished system triggers.
    entries: &'static [SystemEntry],
    // Set once the systems have been built, so a second `start()` on the same
    // world does not append them twice.
    systems_built: bool,
    // The executable schedule over the built systems: declared ordering edges
    // validated + conflict waves from each system's declared access. Built at
    // the end of `start()` (after init, when data-dependent declarations are
    // final) and rebuilt when a `Done` system leaves the set.
    schedule: Option<ExecSchedule>,
}

// A world must stay movable to the simulation thread; a !Send member in any
// system, component, or resource breaks the pipelined driver's thread handoff.
const _: () = {
    const fn require_send<T: Send>() {}
    require_send::<World>()
};

impl core::fmt::Debug for World {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("World")
            .field("components", &self.components.len())
            .field("systems", &self.systems.len())
            .finish()
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// An empty world, for contexts that have no compiled payloads (e.g. unit
    /// tests, or worlds built entirely from runtime-only components).
    pub fn new() -> Self {
        Self::from_payloads(Box::new(NoPayloads))
    }

    /// A world backed by a compiled payload store.
    pub fn from_payloads(blob: Box<dyn PayloadStore + Send>) -> Self {
        Self {
            components: ComponentStorage::default(),
            blob,
            profile: FrameProfile::default(),
            resources: Resources::new(),
            scratch: Arena::tagged(FRAME_SCRATCH_BYTES, MemTag::Scratch),
            scratch_overflows: 0,
            systems: Vec::new(),
            entries: &[],
            systems_built: false,
            schedule: None,
        }
    }

    /// Pre-size the component columns from the blob manifest's per-type record
    /// counts, so the bulk `add` loop that follows never reallocates mid-push.
    pub fn reserve_components(&mut self, counts: &[(u8, u32)]) {
        for &(discriminant, count) in counts {
            self.components
                .reserve(ComponentId::new(discriminant), count as usize);
        }
    }

    /// Add a component loaded from a blob def, returning its minted entity so
    /// the loaders can index it by name.
    pub fn add(&mut self, component: ComponentAsset) -> Entity {
        self.components.push(component)
    }

    /// Add one component to the world.
    ///
    /// Only a [`RuntimeComponent`] can be added: a build-only asset is consumed
    /// by the cook and never reaches a world.
    pub fn add_component<C: RuntimeComponent>(&mut self, c: C) {
        self.components.push(c.into());
    }

    /// Remove and drop every component of type C.
    pub fn remove_all<C: ComponentSlot>(&mut self) {
        let _ = self.components.drain::<C>();
    }

    /// Whether the world holds neither components nor systems.
    pub fn is_empty(&self) -> bool {
        self.components.is_empty() && self.systems.is_empty()
    }

    /// Components across every typed column.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Iterate every stored component of a given type. Mirrors
    /// `PipelineContext::query`; useful in tests that hold a `World` directly.
    pub fn query<C: ComponentSlot>(&self) -> core::slice::Iter<'_, C> {
        C::slot(&self.components).iter()
    }

    /// Mutable iteration over all components of type C. Mirror of
    /// `PipelineContext::query_mut` for code holding a `World` directly rather
    /// than a per-system `PipelineContext`.
    pub fn query_mut<C: ComponentSlot>(&mut self) -> core::slice::IterMut<'_, C> {
        self.components.values_mut::<C>().iter_mut()
    }

    /// Push a runtime-produced component into the matching typed slot,
    /// returning its minted entity. Mirror of `PipelineContext::push`.
    pub fn push<C: ComponentSlot>(&mut self, c: C) -> Entity {
        self.components.push_typed(c)
    }

    /// Borrow one entity's component, for code holding a `World` directly.
    /// Mirror of `PipelineContext::get`.
    pub fn get<C: ComponentSlot>(&self, entity: Entity) -> Option<&C> {
        self.components.get::<C>(entity)
    }

    /// Mutably borrow one entity's component. Mirror of
    /// `PipelineContext::get_mut`.
    pub fn get_mut<C: ComponentSlot>(&mut self, entity: Entity) -> Option<&mut C> {
        self.components.get_mut::<C>(entity)
    }

    /// Add a component to an existing entity. Mirror of
    /// `PipelineContext::insert`.
    pub fn insert<C: ComponentSlot>(&mut self, entity: Entity, c: C) {
        self.components.insert_typed(entity, c);
    }

    /// Overwrite an existing component with a rebuilt one, keeping the entity
    /// and its other components. `false` when the entity holds no component of
    /// that type. An editing tool that rebuilds one component from changed
    /// authoring data writes it back through here rather than reloading the
    /// world around it.
    pub fn replace_component(&mut self, entity: Entity, asset: ComponentAsset) -> bool {
        self.components.replace(entity, asset)
    }

    /// Whether an entity is still live. Mirror of `PipelineContext::is_alive`;
    /// guards name-index resolves against entities despawned by the start-time
    /// drains (Window, GraphicsConfig, Scene, ...).
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.components.is_alive(entity)
    }

    /// Despawn an entity (all its components, recycling its id). Stands in for
    /// the GraphicsSystem-mediated despawn in system tests that need an entity
    /// gone before a later system step (e.g. physics-body reaping).
    pub fn despawn(&mut self, entity: Entity) {
        self.components.despawn(entity);
    }

    /// Read-only join over two component types, for code holding a `World`
    /// directly (the decomposition round-trip tests). Mirror of
    /// `PipelineContext::join2`.
    pub fn join2<A: ComponentSlot, B: ComponentSlot>(
        &self,
    ) -> impl Iterator<Item = (Entity, &A, &B)> {
        self.components.join2::<A, B>()
    }

    /// How many components of each type the world holds, one entry per
    /// populated type.
    pub fn component_census(&self) -> Vec<(u8, u32)> {
        self.components.component_census()
    }

    /// Borrow the event queue for event type E, if any have been sent. Mirror of
    /// `PipelineContext::events`, for code holding a `World` directly (tests).
    pub fn events<E: 'static>(&self) -> Option<&Events<E>> {
        self.resources.get::<EventStore>()?.get::<E>()
    }

    /// Mutably borrow (creating if absent) the event queue for event type E.
    /// Mirror of `PipelineContext::events_mut`, for code holding a `World`
    /// directly: tests, and the editor's debug-driven command injection.
    pub fn events_mut<E: Send + 'static>(&mut self) -> &mut Events<E> {
        self.event_store().get_mut_or_create::<E>()
    }

    /// Seed (or replace) a singleton resource that persists across steps.
    pub fn insert_resource<T: core::any::Any + Send>(&mut self, value: T) {
        self.resources.insert(value);
    }

    /// Borrow a published singleton resource.
    pub fn resource<T: core::any::Any>(&self) -> Option<&T> {
        self.resources.get::<T>()
    }

    /// Mutably borrow a published singleton resource.
    pub fn resource_mut<T: core::any::Any>(&mut self) -> Option<&mut T> {
        self.resources.get_mut::<T>()
    }

    /// Withdraw a published singleton resource. Presence-keyed protocols turn
    /// off by removing their resource, so the reading system pays nothing
    /// beyond noticing the absence.
    pub fn remove_resource<T: core::any::Any>(&mut self) -> Option<T> {
        self.resources.remove::<T>()
    }

    /// Per-frame profiling data: system CPU timings and render-backend stats
    /// from the most recently completed frame.
    pub fn profile(&self) -> &FrameProfile {
        &self.profile
    }

    /// Mutable view of the frame profile, for the frame loop that rotates its
    /// buffers and stamps the frame's totals around each step.
    pub fn profile_mut(&mut self) -> &mut FrameProfile {
        &mut self.profile
    }

    /// What the frame scratch cost and whether it was big enough, for the
    /// `memory` query and the Health panel. `peak` is what sizes the reserve.
    pub fn scratch_stats(&self) -> ScratchStats {
        ScratchStats {
            capacity: self.scratch.capacity(),
            peak: self.scratch.peak(),
            overflows: self.scratch_overflows,
        }
    }

    /// The systems' view of this world for one tick. The caller holds the
    /// returned context for the whole tick, so the borrow of `self` is what
    /// keeps the world's data still while systems run over it.
    pub fn context(&mut self) -> PipelineContext<'_> {
        self.systems_and_context().1
    }

    /// The `EventStore` resource, created on first use. Every queue
    /// `events_mut` ever handed out (here or on a `PipelineContext`) lives in
    /// this one resource, so no per-type rotation list can fall out of sync.
    pub fn event_store(&mut self) -> &mut EventStore {
        if !self.resources.contains::<EventStore>() {
            self.resources.insert(EventStore::new());
        }
        self.resources
            .get_mut::<EventStore>()
            .expect("EventStore was just inserted")
    }

    /// Advance every event queue once, before systems run, so each queue's
    /// two-frame retention holds for readers that run after the writer.
    pub fn update_events(&mut self) {
        if let Some(store) = self.resources.get_mut::<EventStore>() {
            store.update_all();
        }
    }

    /// Hand the whole frame's scratch back. `&mut self` is the proof that no
    /// allocation from the last frame survives.
    pub fn reset_scratch(&mut self) {
        self.scratch.reset();
    }

    /// Release every resident compiled payload, returning the bytes freed. Run
    /// once every system has inited and cached what it keeps.
    pub fn release_payloads(&mut self) -> usize {
        self.blob.release_all_resident()
    }

    /// The world's systems, in schedule order.
    pub fn systems(&self) -> &[BuiltSystem] {
        &self.systems
    }

    /// Mutable view of the active systems. Lets a caller holding the world
    /// downcast one system out of the boxed set and drive it from outside the
    /// per-system step (the `cn debug` hot-reload drive).
    pub fn systems_mut(&mut self) -> &mut [BuiltSystem] {
        &mut self.systems
    }

    /// Disjoint mutable borrows of the system list and the resource map, for a
    /// caller that drives a system against something parked in a resource (the
    /// `cn debug` hot-reload drive reaches the render backend that way).
    pub fn systems_and_resources(&mut self) -> (&mut [BuiltSystem], &mut Resources) {
        (&mut self.systems, &mut self.resources)
    }

    /// Systems built for this world.
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }

    /// The system names `table` would build for this world's current content,
    /// in run order. Runs the same gates [`start`](World::start) runs, so
    /// tooling that reports a world's schedule cannot drift from the runtime;
    /// the probe constructs and discards each gated system, which is why
    /// constructors must stay cheap and side-effect-free. After `start` has
    /// drained the gating components it reports the systems a rebuild of the
    /// CURRENT content would get, not the built set.
    pub fn system_manifest(&self, table: &SystemTable) -> Vec<&'static str> {
        table
            .entries
            .iter()
            .filter(|entry| (entry.gate)(self).is_some())
            .map(|entry| entry.name)
            .collect()
    }

    // Disjoint borrows of the system list and the tick's context over the data
    // half. Splitting the two is what lets a system step against the world it
    // lives in.
    fn systems_and_context(&mut self) -> (&mut Vec<BuiltSystem>, PipelineContext<'_>) {
        (
            &mut self.systems,
            PipelineContext {
                components: &mut self.components,
                blob: &mut *self.blob,
                profile: &mut self.profile,
                resources: &mut self.resources,
                frame: FrameContext::new(&self.scratch),
            },
        )
    }

    /// Build the systems `table` gates in for this world's content and run
    /// their `init`.
    pub fn start(&mut self, table: &SystemTable) -> Result<(), CnResult> {
        self.build_systems(table);
        let (systems, mut ctx) = self.systems_and_context();
        // The host's load-time pass, before systems init: the engine gives each
        // loaded placement its per-instance components here.
        if let Some(before_init) = table.before_init {
            before_init(&mut ctx);
        }
        for system in systems.iter_mut() {
            system.init(&mut ctx);
        }
        // Every system has inited and cached the payloads it keeps; nothing
        // reads compiled payloads at runtime. Free every blob section still
        // resident: the shipped runtime's blob 0, the audio / SDF / terrain
        // blobs the GraphicsSystem init sweep held back for their later
        // consumers, and every blob in a world with no GraphicsSystem to run
        // that sweep at all.
        let freed = self.release_payloads();
        if freed >= 1024 * 1024 {
            tracing::info!(
                "World: freed {} MiB of resident blob payloads after init",
                freed / (1024 * 1024)
            );
        }
        // Access declarations are final once every system has inited, so this
        // is the earliest the edges can be validated and the waves derived.
        let schedule = waves::build(&self.systems, self.entries);
        // Pre-create the event queues declared systems can touch, so their
        // `events_mut` never grows the store's map mid-tick.
        if let Some(prepare_events) = table.prepare_events
            && !schedule.is_empty()
        {
            for i in 0..schedule.len() {
                let access = schedule.access(i);
                prepare_events(self.event_store(), access);
            }
        }
        self.schedule = Some(schedule);
        Ok(())
    }

    // Construct the systems the table gates in, in table order, just before
    // `init`. Each entry is present only when its gating content is, and is
    // built from it by the entry's gate. Runs at most once per world (guarded
    // by `systems_built`) so a system whose gating components survive `init` is
    // not built twice.
    fn build_systems(&mut self, table: &SystemTable) {
        if self.systems_built {
            return;
        }
        self.systems_built = true;
        self.entries = table.entries;
        for entry in table.entries {
            if let Some(system) = (entry.gate)(self) {
                self.systems.push(BuiltSystem::new(entry.name, system));
            }
        }
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
        self.profile.begin_frame();
        // Advance every event queue once per frame, before systems run, so each
        // queue's two-frame retention holds for readers that run after the
        // writer.
        self.update_events();
        // Hand the whole frame's scratch back before anything runs.
        self.reset_scratch();
        // The host's monotonic clock, read once per tick. A world running
        // without one records zero micros per system.
        let clock = self.resources.get::<Clock>().map(|c| c.0);
        let (systems, mut ctx) = self.systems_and_context();
        let mut i = 0;
        let mut removed_any = false;
        while i < systems.len() {
            let name = systems[i].name();
            let started = clock.map_or(0, |now| now());
            #[cfg(debug_assertions)]
            let alloc_start = concinnity_memory::alloc_count();
            #[cfg(debug_assertions)]
            crate::ecs::access_check::set_active(Some((systems[i].access(), name)));
            let result = systems[i].step(&mut ctx);
            #[cfg(debug_assertions)]
            crate::ecs::access_check::set_active(None);
            let micros = clock.map_or(0, |now| {
                now().saturating_sub(started).min(u32::MAX as u64) as u32
            });
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
                    let removed = systems.remove(i);
                    removed_any = true;
                    tracing::debug!("System '{}' finished", removed.name());
                }
                StepResult::Continue => {
                    i += 1;
                }
            }
        }
        if removed_any && self.schedule.is_some() {
            self.schedule = Some(waves::build(&self.systems, self.entries));
        }
        self.report_scratch_overflow();
        #[cfg(debug_assertions)]
        if let (Some(start), Some(end)) = (frame_alloc_start, concinnity_memory::alloc_count()) {
            self.profile
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
        let overflows = self.take_scratch_overflows();
        if overflows == 0 {
            return;
        }
        let stats = self.scratch_stats();
        tracing::warn!(
            "frame scratch overflowed {overflows} time(s): reserve {} KiB, peak {} KiB",
            stats.capacity / 1024,
            stats.peak / 1024,
        );
    }

    /// Fold the frame's declined scratch requests into the world's running
    /// total, returning what this frame declined. A frame that outgrew the
    /// reserve fell back to the heap and still rendered, so nothing breaks --
    /// but the caller reports it, since a silent fallback reads as "the reserve
    /// is sized right" when it is not.
    pub fn take_scratch_overflows(&mut self) -> u32 {
        let overflows = self.scratch.overflows();
        if overflows > 0 {
            self.scratch.clear_overflows();
            self.scratch_overflows = self.scratch_overflows.saturating_add(overflows as u64);
        }
        overflows
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::TextLabel;

    #[test]
    fn a_new_world_is_empty() {
        let world = World::new();
        assert!(world.is_empty());
        assert_eq!(world.component_count(), 0);
    }

    #[test]
    fn components_are_queryable_after_add() {
        let mut world = World::new();
        world.add_component(TextLabel {
            content: "hello".into(),
            ..Default::default()
        });
        assert!(!world.is_empty());
        assert_eq!(world.component_count(), 1);
        assert_eq!(world.query::<TextLabel>().count(), 1);
        assert_eq!(world.query::<TextLabel>().next().unwrap().content, "hello");
    }

    #[test]
    fn reserve_components_leaves_the_world_empty() {
        let mut world = World::new();
        world.reserve_components(&[(TextLabel::DISCRIMINANT, 8)]);
        assert!(world.is_empty());
        assert_eq!(world.query::<TextLabel>().count(), 0);
    }

    #[test]
    fn a_pushed_component_is_reachable_by_its_entity() {
        let mut world = World::new();
        let entity = world.push(TextLabel {
            content: "one".into(),
            ..Default::default()
        });
        assert!(world.is_alive(entity));
        assert_eq!(world.get::<TextLabel>(entity).unwrap().content, "one");
        world.get_mut::<TextLabel>(entity).unwrap().content = "two".into();
        assert_eq!(world.get::<TextLabel>(entity).unwrap().content, "two");
        world.despawn(entity);
        assert!(!world.is_alive(entity));
    }

    #[test]
    fn remove_all_drains_one_column() {
        let mut world = World::new();
        world.add_component(TextLabel::default());
        world.add_component(TextLabel::default());
        assert_eq!(world.component_count(), 2);
        world.remove_all::<TextLabel>();
        assert!(world.is_empty());
    }

    #[test]
    fn the_census_counts_each_populated_type() {
        let mut world = World::new();
        world.add_component(TextLabel::default());
        world.add_component(TextLabel::default());
        let census = world.component_census();
        assert_eq!(census, alloc::vec![(TextLabel::DISCRIMINANT, 2)]);
    }

    #[test]
    fn resources_round_trip() {
        let mut world = World::new();
        assert!(world.resource::<u32>().is_none());
        world.insert_resource(7u32);
        assert_eq!(world.resource::<u32>(), Some(&7));
        *world.resource_mut::<u32>().unwrap() = 9;
        assert_eq!(world.remove_resource::<u32>(), Some(9));
        assert!(world.resource::<u32>().is_none());
    }

    #[test]
    fn events_are_readable_after_send() {
        let mut world = World::new();
        assert!(world.events::<u8>().is_none());
        world.events_mut::<u8>().send(3);
        assert_eq!(
            world.events::<u8>().expect("queue was just created").len(),
            1
        );
    }

    // Two frames' worth of rotation: the queue's retention must outlive one
    // update so a reader running after the writer still sees the send.
    #[test]
    fn update_events_retains_a_send_for_one_frame() {
        let mut world = World::new();
        world.events_mut::<u8>().send(3);
        world.update_events();
        assert_eq!(world.events::<u8>().unwrap().len(), 1);
        world.update_events();
        assert_eq!(world.events::<u8>().unwrap().len(), 0);
    }

    #[test]
    fn the_context_sees_the_worlds_components() {
        let mut world = World::new();
        world.add_component(TextLabel {
            content: "ctx".into(),
            ..Default::default()
        });
        let ctx = world.context();
        assert_eq!(ctx.query::<TextLabel>().next().unwrap().content, "ctx");
    }

    // The reserve is whole at rest, and a world that never allocated from it
    // has declined nothing.
    #[test]
    fn a_quiet_world_reports_no_scratch_overflow() {
        let mut world = World::new();
        assert_eq!(world.take_scratch_overflows(), 0);
        let stats = world.scratch_stats();
        assert_eq!(stats.capacity, FRAME_SCRATCH_BYTES);
        assert_eq!(stats.overflows, 0);
    }

    #[test]
    fn an_empty_payload_store_frees_nothing() {
        let mut world = World::new();
        assert_eq!(world.release_payloads(), 0);
    }
}

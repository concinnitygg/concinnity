//! `PipelineContext`: what a system sees of the world during one tick.
//!
//! Renderer-free. It exposes the typed component storage, the type-keyed
//! resource and event surfaces, the compiled-payload store, the per-frame
//! profiler, and the frame scratch. A `World` constructs one each
//! tick and hands it to every `System::step`.
//!
//! Every accessor reports what it touched to [`crate::ecs::access_check`] in
//! debug builds, so a system reaching outside its declared [`Access`] is caught
//! at the point of the read rather than as a data race later.
//!
//! [`Access`]: crate::ecs::Access

use alloc::vec::Vec;

use crate::ecs::{
    ColumnTicks, ComponentSlot, ComponentStorage, Entity, EventStore, Events, FrameContext,
    PayloadLocator, PayloadStore, Resources, Tick, access_check,
};
use crate::gfx::profile::FrameProfile;
use crate::result::CnResult;

// Debug-only touch reporters for the accessors below, so each accessor carries
// one line. Compiled out of release builds.
#[cfg(debug_assertions)]
fn note_read<C: ComponentSlot>() {
    access_check::touch(access_check::Touch::ComponentRead {
        id: C::DISCRIMINANT,
        type_name: core::any::type_name::<C>(),
    });
}

#[cfg(debug_assertions)]
fn note_write<C: ComponentSlot>() {
    access_check::touch(access_check::Touch::ComponentWrite {
        id: C::DISCRIMINANT,
        type_name: core::any::type_name::<C>(),
    });
}

#[cfg(debug_assertions)]
fn note_structural(op: &'static str) {
    access_check::touch(access_check::Touch::Structural { op });
}

#[cfg(debug_assertions)]
fn note_resource<T: 'static>(write: bool) {
    access_check::touch(access_check::Touch::Resource {
        type_id: core::any::TypeId::of::<T>(),
        type_name: core::any::type_name::<T>(),
        write,
    });
}

/// A system's view of the world for the duration of one `step`: the five things
/// it borrows, and the accessors that reach them.
pub struct PipelineContext<'a> {
    /// Per-type component storage. Systems should not access this directly;
    /// use `query`, `query_mut`, `drain`, or `push` instead.
    pub components: &'a mut ComponentStorage,
    /// Compiled-payload store. Systems use `read_payload` to fetch binary data,
    /// then call `release_blob` when done with it. A trait object so the ECS
    /// mechanism names no concrete blob store; the runtime passes `BlobData`.
    pub blob: &'a mut dyn PayloadStore,
    /// Per-frame profiling data. `World::step` records each system's CPU step
    /// time here; `GraphicsSystem` writes the backend `RenderStats` after its
    /// draw call, and `StatHud` reads it back to drive the on-screen HUD.
    pub profile: &'a mut FrameProfile,
    /// Type-keyed engine singletons (e.g. the per-frame FrameInput snapshot),
    /// accessed via `resource` / `resource_mut` / `insert_resource`. Events live
    /// here too, inside a single `EventStore` resource that owns one `Events<E>`
    /// queue per event type, reached via `events` / `events_mut`.
    pub resources: &'a mut Resources,
    /// Frame-scoped facilities: scratch that the frame loop reclaims wholesale.
    /// One field rather than several so what a frame offers can grow without
    /// touching every system and every construction site again.
    pub frame: FrameContext<'a>,
}

impl<'a> PipelineContext<'a> {
    /// Immutable iteration over all components of type C.
    pub fn query<C: ComponentSlot>(&self) -> core::slice::Iter<'_, C> {
        #[cfg(debug_assertions)]
        note_read::<C>();
        C::slot(self.components).iter()
    }

    /// Iterate all components of type C paired with their owning Entity.
    pub fn query_with_entity<C: ComponentSlot>(&self) -> impl Iterator<Item = (Entity, &C)> {
        #[cfg(debug_assertions)]
        note_read::<C>();
        C::slot(self.components).iter_with_entities()
    }

    /// Mutable iteration over all components of type C.
    pub fn query_mut<C: ComponentSlot>(&mut self) -> core::slice::IterMut<'_, C> {
        #[cfg(debug_assertions)]
        note_write::<C>();
        self.components.values_mut::<C>().iter_mut()
    }

    /// Mutable iteration over all components of type C paired with their owning
    /// Entity (the mutable counterpart of `query_with_entity`), so a system can
    /// update each component and still know which entity owns it without first
    /// materializing the entity set into a Vec.
    pub fn query_mut_with_entity<C: ComponentSlot>(
        &mut self,
    ) -> impl Iterator<Item = (Entity, &mut C)> {
        #[cfg(debug_assertions)]
        note_write::<C>();
        self.components.values_mut_with_entities::<C>()
    }

    /// The change tick of component type C's column (bumped on any insert /
    /// remove / mutable access of a C). Comparing two reads across frames detects
    /// whether any C changed without scanning the column, so a per-frame pass can
    /// skip its work when nothing touched C since it last ran.
    pub fn changed_tick<C: ComponentSlot>(&self) -> Tick {
        #[cfg(debug_assertions)]
        note_read::<C>();
        self.components.changed_tick::<C>()
    }

    /// Every tick stamp of C's column. `changed` answers "did any C move"; a
    /// pass that wants to re-examine only the components that moved also needs
    /// `bulk` (a whole-column write, after which every row must be assumed
    /// written) and `structural` (a row added or removed, after which row
    /// positions and membership have moved).
    pub fn column_ticks<C: ComponentSlot>(&self) -> ColumnTicks {
        #[cfg(debug_assertions)]
        note_read::<C>();
        self.components.column_ticks::<C>()
    }

    /// Components of type C written since `since`, paired with their owning
    /// entity: the dirty set a per-frame pass walks instead of the whole column.
    /// Reports only rows a targeted `get_mut` touched, so it is meaningful only
    /// while C's `bulk` and `structural` ticks have not moved since `since`.
    pub fn changed_rows<C: ComponentSlot>(
        &self,
        since: Tick,
    ) -> impl Iterator<Item = (Entity, &C)> {
        #[cfg(debug_assertions)]
        note_read::<C>();
        self.components.changed_rows::<C>(since)
    }

    /// Mutable slice of all components of type C. Unlike `query_mut` this
    /// exposes the backing storage as a slice, which a system can hand to the
    /// job pool for parallel per-component work.
    pub fn query_slice_mut<C: ComponentSlot>(&mut self) -> &mut [C] {
        #[cfg(debug_assertions)]
        note_write::<C>();
        self.components.values_mut::<C>()
    }

    /// Remove and return all components of type C, despawning each removed
    /// row's Entity so the indices recycle.
    pub fn drain<C: ComponentSlot>(&mut self) -> Vec<C> {
        #[cfg(debug_assertions)]
        note_structural("drain");
        self.components.drain::<C>()
    }

    /// Push a runtime-produced component into the matching typed column,
    /// minting a fresh Entity for it. Preferred over reaching into
    /// `self.components` directly.
    pub fn push<C: ComponentSlot>(&mut self, c: C) {
        #[cfg(debug_assertions)]
        note_structural("push");
        self.components.push_typed(c);
    }

    /// Add a component to an existing entity, so an entity can own more than one
    /// component. The entity must be alive and must not already have C. Allowed
    /// dead because the caller is in the client crate (the load-time Prop
    /// decomposition); core itself has no systems.
    pub fn insert<C: ComponentSlot>(&mut self, entity: Entity, c: C) {
        #[cfg(debug_assertions)]
        note_structural("insert");
        self.components.insert_typed(entity, c);
    }

    /// Remove a component from an entity, returning it if present. The entity
    /// keeps its other components. Allowed dead for the same cross-crate reason
    /// as `insert` (the client toggles the Held tag on pickup/drop).
    pub fn remove<C: ComponentSlot>(&mut self, entity: Entity) -> Option<C> {
        #[cfg(debug_assertions)]
        note_structural("remove");
        self.components.remove_typed::<C>(entity)
    }

    /// Remove an entity entirely: swap-remove its row from every component
    /// column and recycle its id (a stale handle to it then reads as dead). A
    /// no-op on an already-dead or unknown entity. Allowed dead for the same
    /// cross-crate reason as `insert` (the client despawns entities at runtime
    /// from the GraphicsSystem).
    pub fn despawn(&mut self, entity: Entity) {
        #[cfg(debug_assertions)]
        note_structural("despawn");
        self.components.despawn(entity);
    }

    /// Whether an entity is still live (not despawned, matching generation).
    /// Allowed dead for the same cross-crate reason as `insert` (the client
    /// reaps a despawned entity's physics body in PhysicsSystem).
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.components.is_alive(entity)
    }

    /// Borrow one entity's component C read-only. Allowed dead for the same
    /// cross-crate reason as `insert` (the client reads Transform / Held by
    /// entity in the physics, camera, and audio systems).
    pub fn get<C: ComponentSlot>(&self, entity: Entity) -> Option<&C> {
        #[cfg(debug_assertions)]
        note_read::<C>();
        self.components.get::<C>(entity)
    }

    /// Every entity carrying the component with this tag. Serves the queries a
    /// Behavior declares by component name, which no type parameter can express.
    pub fn entities_with_tag(&self, tag: u8) -> &[Entity] {
        #[cfg(debug_assertions)]
        access_check::touch(access_check::Touch::ComponentRead {
            id: tag,
            type_name: "<by tag>",
        });
        self.components.entities_with_tag(tag)
    }

    /// Read-only join over two component types: iterate the first type's rows
    /// and yield both refs for every entity that also has the second. Allowed
    /// dead for the same cross-crate reason as `insert`.
    pub fn join2<A: ComponentSlot, B: ComponentSlot>(
        &self,
    ) -> impl Iterator<Item = (Entity, &A, &B)> {
        #[cfg(debug_assertions)]
        {
            note_read::<A>();
            note_read::<B>();
        }
        self.components.join2::<A, B>()
    }

    /// Mutably borrow one entity's component C (a propagation pass writing a
    /// single entity's value). Allowed dead for the same cross-crate reason as
    /// `insert`.
    pub fn get_mut<C: ComponentSlot>(&mut self, entity: Entity) -> Option<&mut C> {
        #[cfg(debug_assertions)]
        note_write::<C>();
        self.components.get_mut::<C>(entity)
    }

    /// Borrow the singleton resource of type T, if present.
    pub fn resource<T: core::any::Any>(&self) -> Option<&T> {
        #[cfg(debug_assertions)]
        note_resource::<T>(false);
        self.resources.get::<T>()
    }

    /// Mutably borrow the singleton resource of type T, if present.
    pub fn resource_mut<T: core::any::Any>(&mut self) -> Option<&mut T> {
        #[cfg(debug_assertions)]
        note_resource::<T>(true);
        self.resources.get_mut::<T>()
    }

    /// Install (or replace) the singleton resource of type T, returning the
    /// previous instance if one was present.
    pub fn insert_resource<T: core::any::Any + Send>(&mut self, value: T) -> Option<T> {
        #[cfg(debug_assertions)]
        note_resource::<T>(true);
        self.resources.insert(value)
    }

    /// Withdraw the singleton resource of type T, if present.
    pub fn remove_resource<T: core::any::Any>(&mut self) -> Option<T> {
        #[cfg(debug_assertions)]
        note_resource::<T>(true);
        self.resources.remove::<T>()
    }

    /// Take the singleton resource value of type T, leaving `T::default()`
    /// parked in its slot so a take/republish cycle reuses the allocation.
    /// `None` when the type was never inserted.
    pub fn take_resource<T: core::any::Any + Send + Default>(&mut self) -> Option<T> {
        #[cfg(debug_assertions)]
        note_resource::<T>(true);
        self.resources.take::<T>()
    }

    /// Borrow the event queue for event type E, if any events of that type have
    /// been registered.
    pub fn events<E: 'static>(&self) -> Option<&Events<E>> {
        #[cfg(debug_assertions)]
        note_resource::<E>(false);
        self.resources.get::<EventStore>()?.get::<E>()
    }

    /// Mutably borrow the event queue for event type E, creating an empty one on
    /// first access so writers and readers never miss it. All queues live in the
    /// `EventStore` resource, which the frame driver rotates wholesale.
    pub fn events_mut<E: Send + 'static>(&mut self) -> &mut Events<E> {
        #[cfg(debug_assertions)]
        note_resource::<E>(true);
        if !self.resources.contains::<EventStore>() {
            self.resources.insert(EventStore::new());
        }
        self.resources
            .get_mut::<EventStore>()
            .expect("EventStore was just inserted")
            .get_mut_or_create::<E>()
    }

    /// Read the compiled payload bytes for a locator.
    ///
    /// Takes `&mut self` because an overflow blob is read from disk lazily on
    /// first access. Returns an error if the blob was released, the locator is
    /// out of range, or the on-demand load fails.
    pub fn read_payload(&mut self, locator: &PayloadLocator) -> Result<&[u8], CnResult> {
        #[cfg(debug_assertions)]
        access_check::touch(access_check::Touch::Blob { op: "read_payload" });
        self.blob.read(locator)
    }

    /// Release the in-memory payload for an entire blob once all systems
    /// that need it have finished (e.g. after GPU upload).
    ///
    /// See `PayloadStore::release` for semantics.
    pub fn release_blob(&mut self, blob_index: u32) {
        #[cfg(debug_assertions)]
        access_check::touch(access_check::Touch::Blob { op: "release_blob" });
        self.blob.release(blob_index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::{AssetKind, BlobAssetDef, ComponentAsset, ComponentTag, EventCursor};
    use alloc::vec;

    // A payload store holding nothing: every read errors, releases are no-ops.
    // Lets the ECS tests exercise the context's payload forwarding without
    // depending on the concrete `BlobData` (which lives host-side).
    struct EmptyStore;

    impl PayloadStore for EmptyStore {
        fn read(&mut self, _locator: &PayloadLocator) -> Result<&[u8], CnResult> {
            Err(CnResult::FileIo)
        }
        fn release(&mut self, _blob_index: u32) {}
        fn disk_backed(&self) -> bool {
            false
        }
    }

    // A standalone PipelineContext over empty storage, for exercising the
    // resource and event surfaces without a running world.
    fn parts() -> (
        ComponentStorage,
        EmptyStore,
        FrameProfile,
        Resources,
        concinnity_memory::Arena,
    ) {
        (
            ComponentStorage::default(),
            EmptyStore,
            FrameProfile::default(),
            Resources::new(),
            concinnity_memory::Arena::with_capacity(64 * 1024),
        )
    }

    #[test]
    fn resources_round_trip_through_context() {
        let (mut c, mut b, mut p, mut r, scratch) = parts();
        let mut ctx = PipelineContext {
            components: &mut c,
            blob: &mut b,
            profile: &mut p,
            resources: &mut r,
            frame: FrameContext::new(&scratch),
        };
        assert!(ctx.resource::<u32>().is_none());
        assert_eq!(ctx.insert_resource(7u32), None);
        assert_eq!(ctx.resource::<u32>(), Some(&7));
        *ctx.resource_mut::<u32>().unwrap() = 9;
        assert_eq!(ctx.resource::<u32>(), Some(&9));
        // Re-inserting returns the previous value.
        assert_eq!(ctx.insert_resource(1u32), Some(9));
    }

    #[test]
    fn events_round_trip_through_context() {
        let (mut c, mut b, mut p, mut r, scratch) = parts();
        let mut ctx = PipelineContext {
            components: &mut c,
            blob: &mut b,
            profile: &mut p,
            resources: &mut r,
            frame: FrameContext::new(&scratch),
        };
        assert!(ctx.events::<u32>().is_none());
        ctx.events_mut::<u32>().send(1);
        ctx.events_mut::<u32>().send(2);

        let mut cursor = EventCursor::default();
        let seen: Vec<u32> = ctx
            .events::<u32>()
            .unwrap()
            .read(&mut cursor)
            .copied()
            .collect();
        assert_eq!(seen, vec![1, 2]);
        // The same cursor sees nothing new on a second read.
        assert_eq!(ctx.events::<u32>().unwrap().read(&mut cursor).count(), 0);
    }

    // A blob record loads through `from_baked`: the bytes are the serialized
    // runtime component, name injection follows, and an unknown tag is
    // rejected.
    #[test]
    fn baked_records_load_through_from_baked() {
        use crate::components::PointLight;
        let light = PointLight {
            intensity: 3.5,
            range: 12.0,
            ..Default::default()
        };
        let baked = BlobAssetDef {
            name: None,
            kind: AssetKind::Component,
            discriminant: ComponentTag::PointLight as u8,
            args_bytes: postcard::to_allocvec(&light).unwrap(),
            payload: None,
        };
        let from_baked = ComponentAsset::from_baked(&baked).unwrap();
        let ComponentAsset::PointLight(b) = &from_baked else {
            panic!("expected PointLight");
        };
        assert_eq!(b.intensity, 3.5);
        assert_eq!(b.range, 12.0);
        // An unknown tag is rejected.
        let mut bad = baked;
        bad.discriminant = 255;
        assert_eq!(
            ComponentAsset::from_baked(&bad).unwrap_err(),
            CnResult::AssetInvalidType
        );
    }

    #[test]
    fn storage_push_dispatches_into_the_typed_column() {
        let mut storage = ComponentStorage::default();
        storage.push(crate::components::Transform::default().into());
        let census = storage.component_census();
        // Transform's tag is its position in the component list.
        assert_eq!(census, vec![(ComponentTag::Transform as u8, 1)]);
    }

    #[test]
    fn context_component_ops_cover_the_entity_lifecycle() {
        use crate::components::{GlobalTransform, Transform};
        let (mut c, mut b, mut p, mut r, scratch) = parts();
        let mut ctx = PipelineContext {
            components: &mut c,
            blob: &mut b,
            profile: &mut p,
            resources: &mut r,
            frame: FrameContext::new(&scratch),
        };

        ctx.push(Transform::default());
        let e = ctx.components.push_typed(Transform::default());
        assert!(ctx.is_alive(e));
        assert_eq!(ctx.query::<Transform>().count(), 2);
        assert_eq!(ctx.query_with_entity::<Transform>().count(), 2);

        // Mutate through each of the mutable access paths.
        for t in ctx.query_mut::<Transform>() {
            t.position[0] = 1.0;
        }
        ctx.query_slice_mut::<Transform>()[0].position[1] = 2.0;
        ctx.get_mut::<Transform>(e).unwrap().position[2] = 3.0;
        assert_eq!(ctx.get::<Transform>(e).unwrap().position, [1.0, 0.0, 3.0]);

        // A second component on the same entity, then remove it again.
        ctx.insert(e, GlobalTransform::default());
        assert_eq!(ctx.join2::<Transform, GlobalTransform>().count(), 1);
        assert!(ctx.remove::<GlobalTransform>(e).is_some());
        assert!(ctx.remove::<GlobalTransform>(e).is_none());

        // Despawn kills the entity and its remaining components.
        ctx.despawn(e);
        assert!(!ctx.is_alive(e));
        assert_eq!(ctx.query::<Transform>().count(), 1);
        assert!(ctx.get::<Transform>(e).is_none());

        // Drain empties the column and returns the survivors.
        let drained = ctx.drain::<Transform>();
        assert_eq!(drained.len(), 1);
        assert_eq!(ctx.query::<Transform>().count(), 0);
    }

    #[test]
    fn read_payload_and_release_forward_to_the_store() {
        let (mut c, mut b, mut p, mut r, scratch) = parts();
        let mut ctx = PipelineContext {
            components: &mut c,
            blob: &mut b,
            profile: &mut p,
            resources: &mut r,
            frame: FrameContext::new(&scratch),
        };
        let loc = PayloadLocator {
            blob_index: 0,
            offset: 0,
            len: 4,
        };
        // read_payload forwards the store's error verbatim.
        assert_eq!(ctx.read_payload(&loc).unwrap_err(), CnResult::FileIo);
        // release_blob forwards without panicking.
        ctx.release_blob(0);
    }
}

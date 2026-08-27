// The hand-assembled world the driver's unit tests drive: a component storage
// plus the four other pieces a `PipelineContext` borrows, with no compiled
// blob and no renderer behind it.
//
// Building the context directly is what lets a test hand a step an explicit
// `SimTiming`, which is what makes the fixed-tick assertions deterministic.

use alloc::string::String;

use crate::components::{Collider, Pickup, PropCollider, Transform};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{
    Arena, ComponentStorage, Entity, EntityByName, FrameContext, NoPayloads, PipelineContext,
    Resources,
};
use crate::gfx::profile::FrameProfile;

pub(super) struct TestWorld {
    pub(super) components: ComponentStorage,
    blob: NoPayloads,
    profile: FrameProfile,
    pub(super) resources: Resources,
    scratch: Arena,
}

impl TestWorld {
    pub(super) fn new() -> Self {
        Self {
            components: ComponentStorage::default(),
            blob: NoPayloads,
            profile: FrameProfile::default(),
            resources: Resources::new(),
            scratch: Arena::with_capacity(64 * 1024),
        }
    }

    pub(super) fn ctx(&mut self) -> PipelineContext<'_> {
        PipelineContext {
            components: &mut self.components,
            blob: &mut self.blob,
            profile: &mut self.profile,
            resources: &mut self.resources,
            frame: FrameContext::new(&self.scratch),
        }
    }

    // Push a decomposed prop (Transform + Collider, plus the Pickup tag when
    // asked) exactly as the load-time decomposition would, and register it in
    // the name index under `id` so joints can resolve it.
    pub(super) fn spawn_prop(&mut self, id: AssetId, position: [f32; 3], pickup: bool) -> Entity {
        let entity = self.components.push_typed(Transform {
            position,
            ..Default::default()
        });
        self.components.insert_typed(
            entity,
            Collider(PropCollider {
                shape: "ball".into(),
                half_extents: [0.5; 3],
                radius: 0.5,
                half_height: 0.0,
                layer: String::new(),
            }),
        );
        if pickup {
            self.components.insert_typed(entity, Pickup);
        }
        match self.resources.get_mut::<EntityByName>() {
            Some(index) => {
                index.0.insert(id, entity);
            }
            None => {
                let mut index = alloc::collections::BTreeMap::new();
                index.insert(id, entity);
                self.resources.insert(EntityByName(index));
            }
        }
        entity
    }
}

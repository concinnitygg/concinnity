// concinnity-physics/src/props.rs
//
// The prop bodies the simulation owns: one per collider-bearing entity, plus
// the two indices kept in step with them -- handle -> entity, which the
// contact publish resolves every hit through, and the tracked-entity set the
// per-tick scan for runtime spawns skips. Bodies enter through `add` and leave
// through `reap`, so the indices have exactly two places to stay honest.

use std::collections::{HashMap, HashSet};

use concinnity_core::assets::BodyDynamics;
use concinnity_core::ecs::Entity;

use crate::interp::PoseInterp;
use crate::layers::{LAYER_PROP, LAYER_WORLD, LayerTable};
use crate::{BodyHandle, ColliderShape, PhysicsWorld};

// Friction coefficient for static (non-PropBody) prop colliders.
pub(crate) const STATIC_FRICTION: f32 = 0.8;

// Links a prop entity to its body in the simulation.
#[derive(Debug)]
pub(crate) struct PropPhysics {
    // The prop's entity, used to read/write its Transform and toggle its Held tag.
    pub entity: Entity,
    pub handle: BodyHandle,
    // False for static (immovable) props.
    pub dynamic: bool,
    // Whether the prop can be picked up and carried.
    pub pickup: bool,
    // Simulated pose snapshots the render blend samples (dynamic props only).
    pub pose: PoseInterp,
}

// A collider-bearing entity's physics description, snapshotted from its
// components when its body is built.
#[derive(Debug)]
pub(crate) struct PropCollSnap {
    pub shape: ColliderShape,
    // Authored collision layer name; empty derives from the body kind.
    pub layer: String,
    pub position: [f32; 3],
    pub rotation_deg: [f32; 3],
    pub pickup: bool,
    pub dynamics: Option<BodyDynamics>,
}

// Every prop body with its handle and entity indices.
#[derive(Debug, Default)]
pub(crate) struct PropBodies {
    bodies: Vec<PropPhysics>,
    by_handle: HashMap<BodyHandle, Entity>,
    tracked: HashSet<Entity>,
    // Scratch for `sample_poses`, refilled every frame.
    sampled: Vec<(Entity, [f32; 3], [f32; 3])>,
}

impl PropBodies {
    // Add one prop body for a collider-bearing entity and record it for pose
    // write-back and reaping. An entity with BodyDynamics simulates freely on
    // the prop layer; anything else is an immovable obstacle on the world
    // layer. An authored collider layer name overrides either default.
    pub fn add(
        &mut self,
        layers: &LayerTable,
        world: &mut PhysicsWorld,
        entity: Entity,
        snap: PropCollSnap,
    ) -> BodyHandle {
        let pose = PoseInterp::new(
            snap.position,
            crate::convert::quat_from_euler_deg(snap.rotation_deg),
        );
        let layer_or = |default: &'static str| {
            if snap.layer.is_empty() {
                layers.mask(default)
            } else {
                layers.mask(&snap.layer)
            }
        };
        let dynamic = snap.dynamics.is_some();
        let handle = if let Some(dynamics) = &snap.dynamics {
            world.add_dynamic(
                &snap.shape,
                snap.position,
                snap.rotation_deg,
                crate::dynamic_params(dynamics),
                layer_or(LAYER_PROP),
            )
        } else {
            world.add_fixed(
                &snap.shape,
                snap.position,
                snap.rotation_deg,
                STATIC_FRICTION,
                layer_or(LAYER_WORLD),
            )
        };
        self.bodies.push(PropPhysics {
            entity,
            handle,
            dynamic,
            pickup: snap.pickup && dynamic,
            pose,
        });
        self.by_handle.insert(handle, entity);
        self.tracked.insert(entity);
        handle
    }

    // Drop the bodies whose entity was despawned: GraphicsSystem runs before
    // PhysicsSystem and removes the entity from the ECS, so a body left behind
    // would keep simulating - and colliding with live bodies - invisibly.
    // Returns `held` remapped onto the compacted list (None when the carried
    // prop was one of the reaped).
    pub fn reap(
        &mut self,
        world: &mut PhysicsWorld,
        held: Option<usize>,
        alive: impl Fn(Entity) -> bool,
    ) -> Option<usize> {
        let held_entity = held.and_then(|i| self.bodies.get(i)).map(|p| p.entity);
        let Self {
            bodies,
            by_handle,
            tracked,
            ..
        } = self;
        let before = bodies.len();
        bodies.retain(|prop| {
            if alive(prop.entity) {
                return true;
            }
            world.remove_body(prop.handle);
            by_handle.remove(&prop.handle);
            tracked.remove(&prop.entity);
            false
        });
        if bodies.len() == before {
            return held;
        }
        held_entity.and_then(|e| bodies.iter().position(|p| p.entity == e))
    }

    // The entity owning `handle`, or None when the body is not a prop's (the
    // terrain, a character capsule, a joint's world anchor).
    pub fn entity_of(&self, handle: BodyHandle) -> Option<Entity> {
        self.by_handle.get(&handle).copied()
    }

    // Whether the entity already owns a body, so the per-tick spawn scan can
    // skip it.
    pub fn is_tracked(&self, entity: Entity) -> bool {
        self.tracked.contains(&entity)
    }

    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    pub fn dynamic_count(&self) -> usize {
        self.bodies.iter().filter(|p| p.dynamic).count()
    }

    pub fn get(&self, index: usize) -> Option<&PropPhysics> {
        self.bodies.get(index)
    }

    pub fn iter(&self) -> impl Iterator<Item = &PropPhysics> {
        self.bodies.iter()
    }

    // Record every dynamic prop's freshly simulated pose for the render blend.
    pub fn record_tick_poses(&mut self, world: &PhysicsWorld) {
        for prop in self.bodies.iter_mut().filter(|p| p.dynamic) {
            let (pos, rot) = world.body_pose_quat(prop.handle);
            prop.pose.push(pos, rot);
        }
    }

    // Blend each dynamic prop's tick poses by the frame's alpha, decomposing
    // the rotation to Euler degrees once here at the write boundary.
    pub fn sample_poses(&mut self, alpha: f32) -> &[(Entity, [f32; 3], [f32; 3])] {
        let Self {
            bodies, sampled, ..
        } = self;
        sampled.clear();
        sampled.extend(bodies.iter().filter(|p| p.dynamic).map(|prop| {
            let (pos, rot) = prop.pose.sample(alpha);
            (prop.entity, pos, crate::convert::euler_deg_from_quat(rot))
        }));
        sampled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::assets::{PhysicsConfig, Transform};
    use concinnity_core::ecs::ComponentStorage;

    // Distinct live entities, minted the way the load-time decomposition does.
    fn mint_entities(storage: &mut ComponentStorage, count: usize) -> Vec<Entity> {
        (0..count)
            .map(|_| storage.push_typed(Transform::default()))
            .collect()
    }

    fn ball_snap(position: [f32; 3]) -> PropCollSnap {
        PropCollSnap {
            shape: ColliderShape::Ball { radius: 0.5 },
            layer: String::new(),
            position,
            rotation_deg: [0.0; 3],
            pickup: false,
            dynamics: Some(BodyDynamics {
                mass: 1.0,
                friction: 0.5,
                ..Default::default()
            }),
        }
    }

    // Three props, one body each, with the index they are looked up through.
    fn three_props() -> (PhysicsWorld, LayerTable, PropBodies, Vec<Entity>) {
        let mut storage = ComponentStorage::default();
        let entities = mint_entities(&mut storage, 3);
        let mut world = PhysicsWorld::new(crate::system::GRAVITY);
        let layers = LayerTable::new(&PhysicsConfig::default());
        let mut props = PropBodies::default();
        for (i, &entity) in entities.iter().enumerate() {
            props.add(&layers, &mut world, entity, ball_snap([i as f32, 2.0, 0.0]));
        }
        (world, layers, props, entities)
    }

    #[test]
    fn add_indexes_every_body_by_its_handle() {
        let (mut world, layers, mut props, entities) = three_props();
        assert_eq!(props.len(), 3);
        for (i, &entity) in entities.iter().enumerate() {
            let handle = props.get(i).expect("body i").handle;
            assert_eq!(props.entity_of(handle), Some(entity));
            assert!(props.is_tracked(entity));
        }

        // A body that is not a prop's resolves to no entity.
        let loose = world.add_fixed(
            &ColliderShape::Ball { radius: 0.1 },
            [0.0; 3],
            [0.0; 3],
            STATIC_FRICTION,
            layers.mask(LAYER_WORLD),
        );
        assert_eq!(props.entity_of(loose), None);

        // A body added later indexes too, without disturbing the earlier ones.
        let mut storage = ComponentStorage::default();
        let late = mint_entities(&mut storage, 1)[0];
        let handle = props.add(&layers, &mut world, late, ball_snap([9.0, 2.0, 0.0]));
        assert_eq!(props.entity_of(handle), Some(late));
        let first = props.get(0).expect("body 0").handle;
        assert_eq!(props.entity_of(first), Some(entities[0]));
    }

    #[test]
    fn reap_drops_the_index_entries_of_despawned_props() {
        let (mut world, _layers, mut props, entities) = three_props();
        let handles: Vec<BodyHandle> = (0..3)
            .map(|i| props.get(i).expect("body i").handle)
            .collect();
        let bodies_before = world.body_count();

        // The middle prop is despawned; the other two survive untouched.
        let dead = entities[1];
        let held = props.reap(&mut world, Some(2), |e| e != dead);

        assert_eq!(props.len(), 2);
        assert_eq!(world.body_count(), bodies_before - 1);
        assert_eq!(props.entity_of(handles[1]), None, "index entry dropped");
        assert!(!props.is_tracked(dead), "tracked entry dropped");
        assert_eq!(props.entity_of(handles[0]), Some(entities[0]));
        assert_eq!(props.entity_of(handles[2]), Some(entities[2]));
        assert_eq!(held, Some(1), "the carried prop followed the compaction");
    }

    #[test]
    fn reap_forgets_a_carried_prop_and_leaves_a_live_list_alone() {
        let (mut world, _layers, mut props, entities) = three_props();
        let carried = entities[0];
        assert_eq!(props.reap(&mut world, Some(0), |e| e != carried), None);
        assert_eq!(props.len(), 2);

        // Nothing dead: the list and the carried index are untouched.
        let handles: Vec<BodyHandle> = (0..2)
            .map(|i| props.get(i).expect("body i").handle)
            .collect();
        assert_eq!(props.reap(&mut world, Some(1), |_| true), Some(1));
        assert_eq!(props.len(), 2);
        for (i, &handle) in handles.iter().enumerate() {
            assert_eq!(props.entity_of(handle), Some(entities[i + 1]));
        }
    }

    #[test]
    fn sample_poses_lists_the_dynamic_props_only() {
        let mut storage = ComponentStorage::default();
        let ids = mint_entities(&mut storage, 2);
        let mut world = PhysicsWorld::new(crate::system::GRAVITY);
        let layers = LayerTable::new(&PhysicsConfig::default());
        let mut props = PropBodies::default();
        props.add(&layers, &mut world, ids[0], ball_snap([0.0, 2.0, 0.0]));
        let mut static_snap = ball_snap([3.0, 2.0, 0.0]);
        static_snap.dynamics = None;
        props.add(&layers, &mut world, ids[1], static_snap);

        assert_eq!(props.dynamic_count(), 1);
        let sampled: Vec<Entity> = props.sample_poses(1.0).iter().map(|(e, ..)| *e).collect();
        assert_eq!(sampled, vec![ids[0]], "static props are never written back");
        assert_eq!(
            props.sample_poses(1.0).len(),
            1,
            "the scratch is refilled, not appended to"
        );
    }
}

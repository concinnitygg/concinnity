// The prop bodies the simulation owns: one per collider-bearing entity, plus
// the three indices kept in step with them -- handle -> entity, which the
// contact publish resolves every hit through, the tracked-entity set the
// per-tick scan for runtime spawns skips, and the refused set that keeps a
// spawn the budget had no room for from being retried every tick. Bodies enter
// through `add` / `adopt` and leave through `reap`, so the indices have exactly
// two places to stay honest.

use alloc::string::String;
use alloc::vec::Vec;

use concinnity_physics::{
    BodyHandle, ColliderShape, Simulation, euler_deg_from_quat, quat_from_euler_deg,
};

use crate::components::BodyDynamics;
use crate::ecs::Entity;

use super::budget::DriverCapacities;
use super::convert::dynamic_params;
use super::index::{SortedMap, SortedSet};
use super::interp::PoseInterp;
use super::layers::{LAYER_PROP, LAYER_WORLD, LayerTable};

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
    by_handle: SortedMap<BodyHandle, Entity>,
    tracked: SortedSet<Entity>,
    // Entities the body budget had no room for. Kept so the per-tick spawn
    // scan skips them instead of re-refusing them every tick; an entry lives
    // until its entity is despawned. Freed headroom does not bring one back.
    refused: SortedSet<Entity>,
    // Scratch for `sample_poses`, refilled every frame.
    sampled: Vec<(Entity, [f32; 3], [f32; 3])>,
}

impl PropBodies {
    // Reserve every index up front, so nothing here allocates once the world
    // is running.
    pub(crate) fn with_capacity(caps: &DriverCapacities) -> Self {
        Self {
            bodies: Vec::with_capacity(caps.props),
            by_handle: SortedMap::with_capacity(caps.props),
            tracked: SortedSet::with_capacity(caps.props),
            refused: SortedSet::with_capacity(caps.refused),
            sampled: Vec::with_capacity(caps.sampled),
        }
    }

    // Add one prop body for a collider-bearing entity and record it for pose
    // write-back and reaping. An entity with BodyDynamics simulates freely on
    // the prop layer; anything else is an immovable obstacle on the world
    // layer. An authored collider layer name overrides either default.
    pub(crate) fn add(
        &mut self,
        layers: &LayerTable,
        world: &mut Simulation,
        entity: Entity,
        snap: PropCollSnap,
    ) -> Option<BodyHandle> {
        let pose = PoseInterp::new(snap.position, quat_from_euler_deg(snap.rotation_deg));
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
                dynamic_params(dynamics),
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
        }?;
        self.bodies.push(PropPhysics {
            entity,
            handle,
            dynamic,
            pickup: snap.pickup && dynamic,
            pose,
        });
        self.by_handle.insert(handle, entity);
        self.tracked.insert(entity);
        Some(handle)
    }

    // Add a body for an entity that appeared after init, unless the world's
    // body budget is already full. Physics reserves every body when the world
    // loads and never grows, so an over-budget spawn is refused outright: it
    // gets no body, the refusal is reported once, and the entity is remembered
    // so the next tick's scan passes over it. It is never retried, even if
    // other bodies are reaped afterwards.
    pub(crate) fn adopt(
        &mut self,
        layers: &LayerTable,
        world: &mut Simulation,
        entity: Entity,
        snap: PropCollSnap,
        body_cap: u32,
    ) {
        let full = world.body_count() as u32 >= body_cap;
        if full || self.add(layers, world, entity, snap).is_none() {
            self.refused.insert(entity);
        }
    }

    // Drop the bodies whose entity was despawned: GraphicsSystem runs before
    // PhysicsSystem and removes the entity from the ECS, so a body left behind
    // would keep simulating - and colliding with live bodies - invisibly.
    // Returns `held` remapped onto the compacted list (None when the carried
    // prop was one of the reaped).
    pub(crate) fn reap(
        &mut self,
        world: &mut Simulation,
        held: Option<usize>,
        alive: impl Fn(Entity) -> bool,
    ) -> Option<usize> {
        let held_entity = held.and_then(|i| self.bodies.get(i)).map(|p| p.entity);
        let Self {
            bodies,
            by_handle,
            tracked,
            refused,
            ..
        } = self;
        // A refused entity that is gone has nothing left to skip.
        refused.retain(|&entity| alive(entity));
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
    pub(crate) fn entity_of(&self, handle: BodyHandle) -> Option<Entity> {
        self.by_handle.get(&handle).copied()
    }

    // Whether the entity already owns a body, so the per-tick spawn scan can
    // skip it.
    pub(crate) fn is_tracked(&self, entity: Entity) -> bool {
        self.tracked.contains(&entity)
    }

    // Whether the entity was already refused a body, so the per-tick spawn
    // scan can skip it without re-reporting.
    pub(crate) fn is_refused(&self, entity: Entity) -> bool {
        self.refused.contains(&entity)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.bodies.len()
    }

    #[cfg(test)]
    pub(crate) fn dynamic_count(&self) -> usize {
        self.bodies.iter().filter(|p| p.dynamic).count()
    }

    pub(crate) fn get(&self, index: usize) -> Option<&PropPhysics> {
        self.bodies.get(index)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &PropPhysics> {
        self.bodies.iter()
    }

    // Record every dynamic prop's freshly simulated pose for the render blend.
    pub(crate) fn record_tick_poses(&mut self, world: &Simulation) {
        for prop in self.bodies.iter_mut().filter(|p| p.dynamic) {
            if let Some((pos, rot)) = world.body_pose_quat(prop.handle) {
                prop.pose.push(pos, rot);
            }
        }
    }

    // Blend each dynamic prop's tick poses by the frame's alpha, decomposing
    // the rotation to Euler degrees once here at the write boundary.
    pub(crate) fn sample_poses(&mut self, alpha: f32) -> &[(Entity, [f32; 3], [f32; 3])] {
        let Self {
            bodies, sampled, ..
        } = self;
        sampled.clear();
        sampled.extend(bodies.iter().filter(|p| p.dynamic).map(|prop| {
            let (pos, rot) = prop.pose.sample(alpha);
            (prop.entity, pos, euler_deg_from_quat(rot))
        }));
        sampled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use crate::components::{PhysicsConfig, Transform};
    use crate::ecs::ComponentStorage;

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
    fn three_props() -> (Simulation, LayerTable, PropBodies, Vec<Entity>) {
        let mut storage = ComponentStorage::default();
        let entities = mint_entities(&mut storage, 3);
        let mut world = Simulation::with_capacity(8);
        let layers = LayerTable::new(&PhysicsConfig::default());
        let mut props = PropBodies::default();
        for (i, &entity) in entities.iter().enumerate() {
            props
                .add(&layers, &mut world, entity, ball_snap([i as f32, 2.0, 0.0]))
                .expect("room in the pool");
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
        let loose = world
            .add_fixed(
                &ColliderShape::Ball { radius: 0.1 },
                [0.0; 3],
                [0.0; 3],
                STATIC_FRICTION,
                layers.mask(LAYER_WORLD),
            )
            .expect("room in the pool");
        assert_eq!(props.entity_of(loose), None);

        // A body added later indexes too, without disturbing the earlier ones.
        let mut storage = ComponentStorage::default();
        let late = mint_entities(&mut storage, 1)[0];
        let handle = props
            .add(&layers, &mut world, late, ball_snap([9.0, 2.0, 0.0]))
            .expect("room in the pool");
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

    // Three prop bodies plus a fourth, still bodiless, entity to spawn.
    fn three_props_and_a_spare() -> (Simulation, LayerTable, PropBodies, Vec<Entity>) {
        let mut storage = ComponentStorage::default();
        let entities = mint_entities(&mut storage, 4);
        let mut world = Simulation::with_capacity(8);
        let layers = LayerTable::new(&PhysicsConfig::default());
        let mut props = PropBodies::default();
        for (i, &entity) in entities.iter().take(3).enumerate() {
            props
                .add(&layers, &mut world, entity, ball_snap([i as f32, 2.0, 0.0]))
                .expect("room in the pool");
        }
        (world, layers, props, entities)
    }

    // The budget is a ceiling, not a hint: a spawn past it gets no body, is
    // reported once, and is remembered so the per-tick scan skips it.
    #[test]
    fn a_spawn_past_the_cap_is_refused_once_and_never_retried() {
        let (mut world, layers, mut props, entities) = three_props_and_a_spare();
        let bodies_before = world.body_count();
        let late = entities[3];

        let cap = bodies_before as u32;
        props.adopt(&layers, &mut world, late, ball_snap([9.0, 2.0, 0.0]), cap);
        assert_eq!(props.len(), 3, "the refused entity got no body");
        assert_eq!(world.body_count(), bodies_before, "and none was built");
        assert!(props.is_refused(late), "so the next scan skips it");
        assert!(!props.is_tracked(late));

        // Refusing again is a no-op: the entity is already remembered.
        props.adopt(&layers, &mut world, late, ball_snap([9.0, 2.0, 0.0]), cap);
        assert_eq!(world.body_count(), bodies_before);

        // Room under the cap is taken normally.
        props.adopt(
            &layers,
            &mut world,
            late,
            ball_snap([9.0, 2.0, 0.0]),
            cap + 1,
        );
        assert_eq!(props.len(), 4);
        assert_eq!(world.body_count(), bodies_before + 1);
    }

    // A refused entity that is despawned is forgotten, so the set tracks live
    // refusals rather than accumulating every spawn the world ever attempted.
    #[test]
    fn reap_forgets_refused_entities_whose_entity_is_gone() {
        let (mut world, layers, mut props, entities) = three_props_and_a_spare();
        let late = entities[3];
        let cap = world.body_count() as u32;
        props.adopt(&layers, &mut world, late, ball_snap([9.0, 2.0, 0.0]), cap);
        assert!(props.is_refused(late));

        // Everything but the refused entity is still alive: the refusal stays.
        props.reap(&mut world, None, |e| e != entities[0]);
        assert!(props.is_refused(late));

        props.reap(&mut world, None, |e| e != late);
        assert!(!props.is_refused(late), "the despawned refusal is dropped");
    }

    #[test]
    fn with_capacity_reserves_every_index_from_the_budget() {
        let caps = DriverCapacities {
            props: 16,
            refused: 4,
            sampled: 8,
            new_props: 4,
            rigs: 0,
            root_motions: 0,
            contacts: 0,
            contact_pairs: 0,
            sensor_crossings: 0,
            sensor_filters: 0,
        };
        let props = PropBodies::with_capacity(&caps);
        assert!(props.bodies.capacity() >= 16);
        assert!(props.by_handle.capacity() >= 16);
        assert!(props.tracked.capacity() >= 16);
        assert!(props.refused.capacity() >= 4);
        assert!(props.sampled.capacity() >= 8);
    }

    #[test]
    fn sample_poses_lists_the_dynamic_props_only() {
        let mut storage = ComponentStorage::default();
        let ids = mint_entities(&mut storage, 2);
        let mut world = Simulation::with_capacity(4);
        let layers = LayerTable::new(&PhysicsConfig::default());
        let mut props = PropBodies::default();
        props
            .add(&layers, &mut world, ids[0], ball_snap([0.0, 2.0, 0.0]))
            .expect("room in the pool");
        let mut static_snap = ball_snap([3.0, 2.0, 0.0]);
        static_snap.dynamics = None;
        props
            .add(&layers, &mut world, ids[1], static_snap)
            .expect("room in the pool");

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

// The prop bodies the simulation owns: one per collider-bearing entity, plus
// the three indices kept in step with them -- handle -> entity, which the
// contact publish resolves every hit through, entity -> handle, which a wake
// request resolves through and the per-tick scan for runtime spawns skips
// what it already holds by, and the refused set that keeps a
// spawn the budget had no room for from being retried every tick. Bodies enter
// through `add` / `adopt` and leave through `reap`, so the indices have exactly
// two places to stay honest.

use alloc::string::String;
use alloc::vec::Vec;
use concinnity_core::math::{SaturatingFrom, euler_yxz_deg_from_quat, quat_from_euler_yxz_deg};

use crate::physics::{BodyHandle, ColliderShape, MAX_FLUID_PIECES, Simulation};
use crate::water::stir::{StirPiece, WaterStir};

use crate::components::BodyDynamics;
use crate::ecs::Entity;
use crate::transform::quat_slerp;
use concinnity_core::math::vec3::lerp;

use super::capacity::DriverCapacities;
use super::convert::dynamic_params;
use super::float::Floater;
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
    // The pose written back to the entity's Transform last frame. A Transform
    // that differs was written by something else and is adopted.
    written: ([f32; 3], [f32; 3]),
    // Whether the Transform already holds the pose the body is resting at, so
    // writing it again would change nothing.
    rest_written: bool,
    // How the body floats, when it is buoyant.
    pub(super) float: Option<Floater>,
    // Set while something outside the physics owns the Transform: the body is
    // driven there kinematically over the frame's ticks, and its pose is not
    // written back.
    posed: Option<PosedPath>,
}

// Where a posed body goes over the ticks to come: from the pose it last
// reached to the one its owner put it in since.
#[derive(Debug, Clone, Copy)]
struct PosedPath {
    from: ([f32; 3], [f32; 4]),
    to: ([f32; 3], [f32; 4]),
}

impl PropPhysics {
    // Whether an outside owner of the Transform poses the body.
    pub(crate) fn is_posed(&self) -> bool {
        self.posed.is_some()
    }
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
    // Entity -> index into `bodies`, renumbered when a reap compacts them.
    by_entity: SortedMap<Entity, usize>,
    // The one entity whose Transform an outside owner poses, if any.
    posed_owner: Option<Entity>,
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
            by_entity: SortedMap::with_capacity(caps.props),
            posed_owner: None,
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
        snap: &PropCollSnap,
    ) -> Option<BodyHandle> {
        let pose = PoseInterp::new(snap.position, quat_from_euler_yxz_deg(snap.rotation_deg));
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
        if snap.dynamics.is_some_and(|d| d.asleep) {
            world.sleep_body(handle);
        }
        // Spread over the spot's life so bodies loaded together refresh apart.
        let stagger = concinnity_core::math::fract(self.bodies.len() as f32 * 0.618_034);
        let float = snap
            .dynamics
            .and_then(|d| d.buoyancy)
            .map(|b| Floater::new(&snap.shape, b, stagger));
        self.bodies.push(PropPhysics {
            entity,
            handle,
            dynamic,
            pickup: snap.pickup && dynamic,
            pose,
            written: (snap.position, snap.rotation_deg),
            rest_written: false,
            float,
            posed: None,
        });
        self.by_handle.insert(handle, entity);
        self.by_entity.insert(entity, self.bodies.len() - 1);
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
        snap: &PropCollSnap,
        body_cap: u32,
    ) {
        let full = u32::saturating_from(world.body_count()) >= body_cap;
        if full || self.add(layers, world, entity, snap).is_none() {
            self.refused.insert(entity);
        }
    }

    // Drop the bodies whose entity was despawned: the renderer runs before
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
            by_entity,
            refused,
            ..
        } = self;
        // A refused entity that is gone has nothing left to skip.
        refused.retain(|&entity| alive(entity));
        if self.posed_owner.is_some_and(|e| !alive(e)) {
            self.posed_owner = None;
        }
        let before = bodies.len();
        bodies.retain(|prop| {
            if alive(prop.entity) {
                return true;
            }
            world.remove_body(prop.handle);
            by_handle.remove(&prop.handle);
            false
        });
        if bodies.len() == before {
            return held;
        }
        by_entity.retain(|_, _| false);
        for (index, prop) in bodies.iter().enumerate() {
            by_entity.insert(prop.entity, index);
        }
        held_entity.and_then(|e| by_entity.get(&e).copied())
    }

    // The entity owning `handle`, or None when the body is not a prop's (the
    // terrain, a character capsule, a joint's world anchor).
    pub(crate) fn entity_of(&self, handle: BodyHandle) -> Option<Entity> {
        self.by_handle.get(&handle).copied()
    }

    // Whether the entity already owns a body, so the per-tick spawn scan can
    // skip it.
    pub(crate) fn is_tracked(&self, entity: Entity) -> bool {
        self.by_entity.get(&entity).is_some()
    }

    // The body `entity` owns, or None when it has none.
    pub(crate) fn handle_of(&self, entity: Entity) -> Option<BodyHandle> {
        self.prop_of(entity).map(|p| p.handle)
    }

    // The prop body `entity` owns.
    pub(crate) fn prop_of(&self, entity: Entity) -> Option<&PropPhysics> {
        self.bodies.get(*self.by_entity.get(&entity)?)
    }

    fn prop_of_mut(&mut self, entity: Entity) -> Option<&mut PropPhysics> {
        self.bodies.get_mut(*self.by_entity.get(&entity)?)
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

    pub(crate) fn iter(&self) -> impl Iterator<Item = &PropPhysics> + Clone {
        self.bodies.iter()
    }

    // The share of `entity`'s body under water at its last tick: 0 for one
    // that does not float or has no body.
    pub(super) fn submerged_share(&self, entity: Entity) -> f32 {
        self.prop_of(entity)
            .and_then(|p| p.float)
            .map_or(0.0, |f| f.submerged_share())
    }

    // Hand `entity`'s dynamic body to an outside owner of its Transform, which
    // it then follows kinematically, or back to the solver, launched at the
    // speed its last `tick_dt` tick moved it at. Returns whether its state
    // changed.
    pub(crate) fn set_posed(
        &mut self,
        world: &mut Simulation,
        entity: Entity,
        posed: bool,
        tick_dt: f32,
    ) -> bool {
        let Some(prop) = self.prop_of_mut(entity).filter(|p| p.dynamic) else {
            return false;
        };
        match (prop.posed, posed) {
            (None, true) => {
                let at = world
                    .body_pose_quat(prop.handle)
                    .unwrap_or_else(|| (prop.written.0, quat_from_euler_yxz_deg(prop.written.1)));
                prop.posed = Some(PosedPath { from: at, to: at });
                world.make_kinematic(prop.handle)
            }
            (Some(path), false) => {
                prop.posed = None;
                // Where its owner last put it, which the ticks may not have
                // reached yet.
                let launch = prop.pose.last_step().map(|d| d / tick_dt.max(1e-6));
                let rotation_deg = euler_yxz_deg_from_quat(path.to.1);
                world.teleport_body(prop.handle, path.to.0, rotation_deg);
                prop.pose.snap(path.to.0, path.to.1);
                prop.rest_written = false;
                world.make_dynamic(prop.handle, launch)
            }
            _ => false,
        }
    }

    // Pose `owned`'s body, the one Transform an outside owner holds, and
    // release the one posed before it.
    pub(crate) fn pose_only(
        &mut self,
        world: &mut Simulation,
        owned: Option<Entity>,
        tick_dt: f32,
    ) {
        if self.posed_owner == owned {
            return;
        }
        if let Some(previous) = self.posed_owner.take() {
            self.set_posed(world, previous, false, tick_dt);
        }
        if let Some(entity) = owned
            && self.set_posed(world, entity, true, tick_dt)
        {
            self.posed_owner = Some(entity);
        }
    }

    // Move the posed body `tick + 1` of `ticks` of the way to where its owner
    // put it, arriving there on the last tick.
    pub(crate) fn drive_posed(&mut self, world: &mut Simulation, tick: u32, ticks: u32) {
        let Some(entity) = self.posed_owner else {
            return;
        };
        let Some(prop) = self.prop_of_mut(entity) else {
            return;
        };
        let Some(path) = prop.posed.as_mut() else {
            return;
        };
        let t = (tick + 1) as f32 / ticks.max(1) as f32;
        let at = lerp(path.from.0, path.to.0, t);
        let turned = quat_slerp(path.from.1, path.to.1, t);
        world.set_kinematic_pose(prop.handle, at, turned);
        if tick + 1 >= ticks {
            path.from = path.to;
        }
    }

    // Refill `out` with every buoyant prop's stir at its last tick.
    pub(super) fn stirs_into(&self, out: &mut Vec<WaterStir>) {
        out.clear();
        for prop in &self.bodies {
            let Some(float) = &prop.float else {
                continue;
            };
            let stir = float.stir();
            let mut pieces = [StirPiece::default(); MAX_FLUID_PIECES];
            pieces[..stir.len()].copy_from_slice(stir);
            let (up, velocity) = float.motion();
            out.push(WaterStir {
                entity: prop.entity,
                up,
                velocity,
                pieces,
                count: stir.len(),
            });
        }
    }

    // Every buoyant prop's floating state with its body.
    pub(super) fn floaters_mut(&mut self) -> impl Iterator<Item = (BodyHandle, &mut Floater)> {
        self.bodies
            .iter_mut()
            .filter_map(|p| Some((p.handle, p.float.as_mut()?)))
    }

    // Adopt externally written prop transforms before the frame's ticks run: a
    // pose that differs from the one written back last frame was not ours, so
    // the body is moved there, at rest and with no blend across the jump.
    // `authored` reads the entity's current Transform, or None when it has
    // none to read.
    pub(crate) fn sync_external_poses(
        &mut self,
        world: &mut Simulation,
        authored: impl Fn(Entity) -> Option<([f32; 3], [f32; 3])>,
    ) {
        for prop in &mut self.bodies {
            let Some((position, rotation_deg)) = authored(prop.entity) else {
                continue;
            };
            if let Some(path) = &mut prop.posed {
                path.to = (position, quat_from_euler_yxz_deg(rotation_deg));
                prop.written = (position, rotation_deg);
                continue;
            }
            if (position, rotation_deg) == prop.written {
                continue;
            }
            if world.teleport_body(prop.handle, position, rotation_deg) {
                prop.pose
                    .snap(position, quat_from_euler_yxz_deg(rotation_deg));
                prop.written = (position, rotation_deg);
                prop.rest_written = false;
            }
        }
    }

    // Carry every prop's blend snapshots into another frame (the simulation
    // carries the bodies), taking the pose last written back from `written`:
    // the entity's Transform, already carried by whatever moved the frame, so
    // the next frame does not read it as an outside move.
    pub(crate) fn rebase(
        &mut self,
        rebase: &crate::planet::Rebase,
        written: impl Fn(Entity) -> Option<([f32; 3], [f32; 3])>,
    ) {
        for prop in &mut self.bodies {
            prop.pose.rebase(rebase);
            if let Some(path) = &mut prop.posed {
                for pose in [&mut path.from, &mut path.to] {
                    *pose = (rebase.apply_point(pose.0), rebase.apply_rotation(pose.1));
                }
            }
            prop.written = written(prop.entity).unwrap_or_else(|| {
                (
                    rebase.apply_point(prop.written.0),
                    rebase.apply_euler_deg(prop.written.1),
                )
            });
        }
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
    // the rotation to Euler degrees once here at the write boundary. A prop at
    // rest whose Transform already holds that pose is left out, so a settled
    // world writes nothing back.
    pub(crate) fn sample_poses(&mut self, alpha: f32) -> &[(Entity, [f32; 3], [f32; 3])] {
        let Self {
            bodies, sampled, ..
        } = self;
        sampled.clear();
        sampled.extend(
            bodies
                .iter_mut()
                .filter(|p| p.dynamic && !p.is_posed())
                .filter_map(|prop| {
                    let still = prop.pose.is_still();
                    if still && prop.rest_written {
                        return None;
                    }
                    prop.rest_written = still;
                    let (pos, rot) = prop.pose.sample(alpha);
                    let rot = euler_yxz_deg_from_quat(rot);
                    prop.written = (pos, rot);
                    Some((prop.entity, pos, rot))
                }),
        );
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
                .add(
                    &layers,
                    &mut world,
                    entity,
                    &ball_snap([i as f32, 2.0, 0.0]),
                )
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
            assert_eq!(props.handle_of(entity), Some(handle));
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
            .add(&layers, &mut world, late, &ball_snap([9.0, 2.0, 0.0]))
            .expect("room in the pool");
        assert_eq!(props.entity_of(handle), Some(late));
        let first = props.get(0).expect("body 0").handle;
        assert_eq!(props.entity_of(first), Some(entities[0]));
    }

    // A body authored asleep is built asleep; one authored awake is not.
    #[test]
    fn a_body_authored_asleep_is_built_asleep() {
        let (mut world, layers, mut props, _) = three_props();
        let mut storage = ComponentStorage::default();
        let minted = mint_entities(&mut storage, 2);
        let (sleeper, waker) = (minted[0], minted[1]);
        let mut snap = ball_snap([0.0, 8.0, 4.0]);
        if let Some(dynamics) = snap.dynamics.as_mut() {
            dynamics.asleep = true;
        }
        let asleep = props
            .add(&layers, &mut world, sleeper, &snap)
            .expect("room in the pool");
        let awake = props
            .add(&layers, &mut world, waker, &ball_snap([0.0, 8.0, 8.0]))
            .expect("room in the pool");
        assert_eq!(world.is_sleeping(asleep), Some(true));
        assert_eq!(world.is_sleeping(awake), Some(false));
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
        assert_eq!(props.handle_of(dead), None);
        assert_eq!(props.handle_of(entities[2]), Some(handles[2]));
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
                .add(
                    &layers,
                    &mut world,
                    entity,
                    &ball_snap([i as f32, 2.0, 0.0]),
                )
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

        let cap = u32::saturating_from(bodies_before);
        props.adopt(&layers, &mut world, late, &ball_snap([9.0, 2.0, 0.0]), cap);
        assert_eq!(props.len(), 3, "the refused entity got no body");
        assert_eq!(world.body_count(), bodies_before, "and none was built");
        assert!(props.is_refused(late), "so the next scan skips it");
        assert!(!props.is_tracked(late));

        // Refusing again is a no-op: the entity is already remembered.
        props.adopt(&layers, &mut world, late, &ball_snap([9.0, 2.0, 0.0]), cap);
        assert_eq!(world.body_count(), bodies_before);

        // Room under the cap is taken normally.
        props.adopt(
            &layers,
            &mut world,
            late,
            &ball_snap([9.0, 2.0, 0.0]),
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
        let cap = u32::saturating_from(world.body_count());
        props.adopt(&layers, &mut world, late, &ball_snap([9.0, 2.0, 0.0]), cap);
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
        assert!(props.by_entity.capacity() >= 16);
        assert!(props.refused.capacity() >= 4);
        assert!(props.sampled.capacity() >= 8);
    }

    // What the write-back last put on the entity is what the next sync
    // compares against, so a pose physics itself produced is not read back as
    // an external move.
    #[test]
    fn a_simulated_pose_is_not_read_back_as_an_external_move() {
        let (mut world, _layers, mut props, entities) = three_props();
        for _ in 0..30 {
            world.step(1.0 / 60.0);
            props.record_tick_poses(&world);
        }
        let written: Vec<([f32; 3], [f32; 3])> = props
            .sample_poses(1.0)
            .iter()
            .map(|&(_, pos, rot)| (pos, rot))
            .collect();
        assert!(written[0].0[1] < 2.0, "the props fell");

        let before: Vec<[f32; 3]> = props
            .iter()
            .map(|p| world.body_pose_quat(p.handle).expect("live").0)
            .collect();
        props.sync_external_poses(&mut world, |entity| {
            let i = entities.iter().position(|&e| e == entity)?;
            Some(written[i])
        });
        let after: Vec<[f32; 3]> = props
            .iter()
            .map(|p| world.body_pose_quat(p.handle).expect("live").0)
            .collect();
        assert_eq!(before, after, "no body was moved");
    }

    // A Transform that differs from the one written back moves the body there,
    // and the render blend starts from the new pose rather than sweeping to it.
    #[test]
    fn an_externally_written_transform_moves_the_body() {
        let (mut world, _layers, mut props, entities) = three_props();
        world.step(1.0 / 60.0);
        props.record_tick_poses(&world);
        props.sample_poses(1.0);

        let moved = [7.0, 9.0, -3.0];
        props.sync_external_poses(&mut world, |entity| {
            (entity == entities[1]).then_some((moved, [0.0, 45.0, 0.0]))
        });

        let handle = props.get(1).expect("body 1").handle;
        assert_eq!(world.body_pose_quat(handle).expect("live").0, moved);
        let (blended, _) = props.get(1).expect("body 1").pose.sample(0.0);
        assert_eq!(blended, moved, "the blend starts at the new pose");

        // The props either side of it stayed where the simulation had them.
        for i in [0, 2] {
            let handle = props.get(i).expect("body i").handle;
            let y = world.body_pose_quat(handle).expect("live").0[1];
            assert!(y < 2.0 && y > 1.9, "prop {i} was left alone (y = {y})");
        }
    }

    // A static prop's body follows an external write too: it is never written
    // back to, so its collider would otherwise stay where the visual left.
    #[test]
    fn an_externally_written_transform_moves_a_static_bodys_collider() {
        let mut storage = ComponentStorage::default();
        let entity = mint_entities(&mut storage, 1)[0];
        let mut world = Simulation::with_capacity(4);
        let layers = LayerTable::new(&PhysicsConfig::default());
        let mut props = PropBodies::default();
        let mut snap = ball_snap([0.0, 2.0, 0.0]);
        snap.dynamics = None;
        props
            .add(&layers, &mut world, entity, &snap)
            .expect("room in the pool");

        props.sync_external_poses(&mut world, |_| Some(([0.0, 6.0, 0.0], [0.0; 3])));
        let handle = props.get(0).expect("body 0").handle;
        assert_eq!(
            world.body_pose_quat(handle).expect("live").0,
            [0.0, 6.0, 0.0]
        );
    }

    // An entity with no Transform to read has no pose to adopt, so its body is
    // left where the simulation has it.
    #[test]
    fn a_prop_without_a_transform_is_left_alone() {
        let (mut world, _layers, mut props, _entities) = three_props();
        let before: Vec<[f32; 3]> = props
            .iter()
            .map(|p| world.body_pose_quat(p.handle).expect("live").0)
            .collect();
        props.sync_external_poses(&mut world, |_| None);
        let after: Vec<[f32; 3]> = props
            .iter()
            .map(|p| world.body_pose_quat(p.handle).expect("live").0)
            .collect();
        assert_eq!(before, after);
    }

    #[test]
    fn sample_poses_lists_the_dynamic_props_only() {
        let mut storage = ComponentStorage::default();
        let ids = mint_entities(&mut storage, 2);
        let mut world = Simulation::with_capacity(4);
        let layers = LayerTable::new(&PhysicsConfig::default());
        let mut props = PropBodies::default();
        props
            .add(&layers, &mut world, ids[0], &ball_snap([0.0, 2.0, 0.0]))
            .expect("room in the pool");
        let mut static_snap = ball_snap([3.0, 2.0, 0.0]);
        static_snap.dynamics = None;
        props
            .add(&layers, &mut world, ids[1], &static_snap)
            .expect("room in the pool");

        assert_eq!(props.dynamic_count(), 1);
        let sampled: Vec<Entity> = props.sample_poses(1.0).iter().map(|(e, ..)| *e).collect();
        assert_eq!(sampled, vec![ids[0]], "static props are never written back");
        world.step(1.0 / 60.0);
        props.record_tick_poses(&world);
        assert_eq!(
            props.sample_poses(1.0).len(),
            1,
            "the scratch is refilled, not appended to"
        );
    }

    // A body asleep on the floor is written back once at the pose it settled
    // at and then left alone, until something moves it again.
    #[test]
    fn a_resting_prop_is_written_back_once_and_then_left_alone() {
        let (mut world, _layers, mut props, entities) = three_props();
        world
            .add_fixed(
                &ColliderShape::Cuboid {
                    half_extents: [20.0, 1.0, 20.0],
                },
                [0.0, -1.0, 0.0],
                [0.0; 3],
                0.8,
                crate::physics::LayerMask::ALL,
            )
            .expect("room for a floor");
        let mut written = Vec::new();
        for _ in 0..240 {
            world.step(1.0 / 60.0);
            props.record_tick_poses(&world);
            written.push(props.sample_poses(0.5).len());
        }
        assert!(
            props
                .iter()
                .all(|p| world.is_sleeping(p.handle) == Some(true)),
            "the props settled"
        );
        assert_eq!(written[0], 3, "falling props are written every frame");
        assert_eq!(*written.last().expect("frames"), 0, "resting props are not");
        let at_rest = written
            .iter()
            .rposition(|&count| count > 0)
            .expect("a write");
        for _ in 0..2 {
            world.step(1.0 / 60.0);
            props.record_tick_poses(&world);
        }
        assert!(
            props.sample_poses(0.5).is_empty(),
            "rested since frame {at_rest}"
        );

        // A move from outside puts the prop back in the frame's writes, along
        // with the neighbors it woke by leaving.
        props.sync_external_poses(&mut world, |entity| {
            (entity == entities[1]).then_some(([1.0, 5.0, 0.0], [0.0; 3]))
        });
        world.step(1.0 / 60.0);
        props.record_tick_poses(&world);
        let sampled: Vec<Entity> = props.sample_poses(0.5).iter().map(|(e, ..)| *e).collect();
        assert!(sampled.contains(&entities[1]), "{sampled:?}");
    }
}

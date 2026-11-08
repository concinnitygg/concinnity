//! The rigid-body simulation driver. An internal system (not a declarable
//! asset): a schedule constructs one when the world declares a `PhysicsConfig`,
//! a `RigidBody`, a `PropBody`, or a `TriggerVolume`, reading the optional
//! `PhysicsConfig`, and laying every `Terrain` down as ground.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::simulation::physics::{
    AirFalloff, BodyHandle, CharacterCapsule, ColliderShape, ContactHit, GRAVITY, PhysicsBudget,
    SensorCrossing, SimConfig, Simulation,
};

use crate::components::{
    BodyDynamics, CameraTrack, Collider, Identity, PhysicsConfig, PhysicsJoint, Pickup, PropBody,
    RigidBody, RollingControl, Transform, TriggerFilter, TriggerVolume,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Entity, EventCursor, PipelineContext, StepResult, System, World};
use crate::environment::planet::{
    FrameRebases, PlanetFrame, PlanetGravity, PlanetGroundPatch, RebaseCursor,
};
use crate::environment::water::query::WaterQuery;
use crate::environment::water::stir::WaterStirs;
use crate::environment::wind::WindField;
use crate::protocol::{MenuActive, SimTiming, WorldPhysicsBudget};
use crate::simulation::camera::track::CameraTrackStatus;
use concinnity_core::math::vec3::{normalize_or, sub};

use super::capacity::DriverCapacities;
use super::contacts::{ContactBatch, ContactGate};
use super::convert::{collider_shape, joint_spec};
use super::fanout::PhysicsFanout;
use super::index::SortedMap;
use super::interp::PointInterp;
use super::layers::{LAYER_CHARACTER, LAYER_PROP, LAYER_TRIGGER, LAYER_WORLD, LayerTable};
use super::props::{PropBodies, PropCollSnap, STATIC_FRICTION};
use super::terrain::build_terrain_colliders;
use concinnity_core::fanout::{FanOut, Inline, Lent};

// Reach distance for picking up a Prop, in world units.
const PICKUP_REACH: f32 = 3.0;
// Minimum facing dot product (~60-degree cone) for a pickup.
const PICKUP_MIN_DOT: f32 = 0.5;
// Distance ahead of the camera a carried prop hovers.
const HOLD_DISTANCE: f32 = 1.8;
// Drop of a carried prop below eye level.
const HOLD_DROP: f32 = 0.35;
// Launch speed applied to a prop when it is dropped/thrown.
const THROW_SPEED: f32 = 6.0;

/// Rigid-body simulation behavior. Constructed internally by `World::start`
/// from the world's `PhysicsConfig`; never a declarable asset.
///
/// [`new`](PhysicsSystem::new) steps the simulation on the calling thread; a
/// host lends it threads through [`with_pool`](PhysicsSystem::with_pool).
#[derive(Debug)]
pub struct PhysicsSystem {
    // Camera eye Y at spawn; the flat-floor fallback derives nothing from it,
    // but it seeds a sensible fallback camera position.
    floor_y: f32,
    // The simulation, built in init() and sized from the world's budget.
    world: Option<Simulation>,
    // The player capsule, when the world has a Camera3D + RigidBody.
    player: Option<PlayerPhysics>,
    // One capsule per root-motion character rig (see `super::rig`).
    rigs: Vec<super::rig::RigPhysics>,
    // One entry per Prop that carries a collider.
    props: PropBodies,
    // Reader cursor over the `RootMotionEvent` event queue.
    root_cursor: EventCursor,
    // Reader cursor over the `WakeRequest` event queue.
    wake_cursor: EventCursor,
    // Scratch for the per-tick scan for freshly spawned collider-bearing
    // entities, refilled every step.
    new_props: Vec<(Entity, PropCollSnap)>,
    // Per-step drain scratch, reused so the event handoffs never reallocate.
    motion_scratch: Vec<crate::components::RootMotionEvent>,
    contact_scratch: Vec<ContactHit>,
    sensor_scratch: Vec<SensorCrossing>,
    // Index into `props` of the prop currently being carried.
    held: Option<usize>,
    // Sensor tag -> the TriggerVolume it senses for, with its filter. Tags are
    // the volume's AssetId, stamped into the sensor collider's user_data.
    sensor_filters: SortedMap<u64, (AssetId, TriggerFilter)>,
    // Layer-name resolution and the collide matrix from the PhysicsConfig.
    layers: LayerTable,
    // Minimum contact impulse for publishing a ContactEvent.
    contact_min_impulse: f32,
    // Strongest contact per body pair across the frame's ticks.
    contact_batch: ContactBatch,
    // Per-pair refractory so sustained contact reports once per impact.
    contact_gate: ContactGate,
    // Bodies the world authored room for beyond the ones it declares.
    spawn_headroom: u32,
    // Hard ceiling on live bodies: the simulation's whole reservation, so a
    // spawn past it is refused rather than silently declined by a full pool.
    // Resolved at init, along with the reservation itself.
    body_cap: u32,
    // Where a step's independent work runs. `Inline` until a host lends
    // its own.
    fanout: Box<dyn PhysicsFanout>,
    // How far this system has caught up with the moves of a planet world's
    // simulated frame.
    rebases: RebaseCursor,
    // The planet's ground patch body and the revision it holds.
    planet_ground: Option<(BodyHandle, u64)>,
    // Props held off the planet's ground while the patch is elsewhere.
    off_ground: super::off_ground::OffGroundBodies,
    // Gravity's strength: the engine's, or the planet's.
    gravity: f32,
    // The world's water, read by its buoyant bodies; `None` without water.
    water: Option<WaterQuery>,
    // The bodies a `RollingDrive` steers.
    drives: drive::Drives,
}

// Runtime physics state for the player camera capsule.
#[derive(Debug)]
struct PlayerPhysics {
    handle: BodyHandle,
    // The capsule each tick's move is resolved against. Its dimensions come
    // from the RigidBody at init and never change afterwards.
    shape: CharacterCapsule,
    // Camera eye Y minus capsule-center Y.
    eye_offset: f32,
    // False for a free-flying camera (no RigidBody): no gravity, no jump.
    has_gravity: bool,
    gravity_scale: f32,
    jump_height: f32,
    // Current vertical velocity (world units/second).
    vy: f32,
    // Whether the capsule rested on a surface last tick.
    grounded: bool,
    // Authoritative simulated capsule center with its render blend snapshots.
    center: PointInterp,
    // The eye position written back last frame. A Camera3D position that
    // differs was moved externally (free-fly, a teleport) and is adopted with
    // no blend across the jump.
    written_eye: Option<[f32; 3]>,
}

impl PhysicsSystem {
    // Number of rigid bodies the physics world holds. Test-only observable for
    // the body-reaping path.
    #[cfg(test)]
    fn physics_body_count(&self) -> usize {
        self.world.as_ref().map_or(0, Simulation::body_count)
    }

    // Number of colliders the physics world holds. Test-only observable for
    // the spawn/despawn leak checks.
    #[cfg(test)]
    fn physics_collider_count(&self) -> usize {
        self.world.as_ref().map_or(0, Simulation::collider_count)
    }

    /// Build the simulation from the world's `PhysicsConfig`. Bodies,
    /// colliders and the ground are added from the ECS in [`System::init`].
    pub fn new(config: &PhysicsConfig) -> Self {
        Self {
            floor_y: config.floor_y,
            world: None,
            player: None,
            rigs: Vec::new(),
            props: PropBodies::default(),
            root_cursor: EventCursor::default(),
            wake_cursor: EventCursor::default(),
            new_props: Vec::new(),
            motion_scratch: Vec::new(),
            contact_scratch: Vec::new(),
            sensor_scratch: Vec::new(),
            held: None,
            sensor_filters: SortedMap::default(),
            layers: LayerTable::new(config),
            contact_min_impulse: config.contact_min_impulse.max(0.0),
            contact_batch: ContactBatch::default(),
            contact_gate: ContactGate::default(),
            spawn_headroom: config.spawn_headroom,
            body_cap: 0,
            fanout: Box::new(Inline),
            rebases: RebaseCursor::default(),
            planet_ground: None,
            off_ground: super::off_ground::OffGroundBodies::default(),
            gravity: GRAVITY,
            water: None,
            drives: drive::Drives::default(),
        }
    }

    // Float buoyant bodies in `water`.
    #[must_use]
    fn with_water(mut self, water: Option<WaterQuery>) -> Self {
        self.water = water.filter(|w| !w.is_empty());
        self
    }

    /// Run the step's independent work on `pool` rather than on the calling
    /// thread. What a step hands out and the order its results load back in
    /// are the simulation's; a pool decides only where the work runs, so this
    /// cannot change the state a tick lands on.
    #[must_use]
    pub fn with_pool<P: FanOut>(mut self, pool: Lent<P>) -> Self {
        self.fanout = Box::new(pool);
        self
    }

    // The world's body budget: the record cook shipped, or, when no record
    // shipped, the same derivation over the loaded components with the
    // headroom its `PhysicsConfig` authored. Only a shipped record is checked
    // against the live world.
    //
    // The two headrooms can differ: a shipped one is already raised to cover
    // the spawners whose cadence bounds their population, while a directly
    // constructed World gets only what its config authored, since nothing
    // counted its spawners.
    //
    // Whichever it came from, the budget is the whole reservation: the
    // simulation is sized from it and never grows, so its cap is the ceiling
    // spawns are refused against.
    fn resolve_budget(&self, ctx: &PipelineContext<'_>) -> PhysicsBudget {
        let scan = super::capacity::scan_counts(ctx);
        let Some(record) = ctx.resource::<WorldPhysicsBudget>().map(|b| b.0) else {
            return PhysicsBudget::derive(&scan, self.spawn_headroom);
        };
        let shipped = super::capacity::budget_of(&record);
        debug_assert_eq!(
            shipped,
            PhysicsBudget::derive(&scan, record.spawn_headroom),
            "the shipped physics budget does not match the loaded world"
        );
        shipped
    }

    // Size every container the driver holds per body from the budget, once,
    // before anything is built.
    fn reserve(&mut self, budget: &PhysicsBudget) {
        let caps = DriverCapacities::derive(budget);
        self.props = PropBodies::with_capacity(&caps);
        self.off_ground = super::off_ground::OffGroundBodies::with_capacity(caps.props);
        self.rigs = Vec::with_capacity(caps.rigs);
        self.new_props = Vec::with_capacity(caps.new_props);
        self.motion_scratch = Vec::with_capacity(caps.root_motions);
        self.contact_scratch = Vec::with_capacity(caps.contacts);
        self.sensor_scratch = Vec::with_capacity(caps.sensor_crossings);
        self.contact_batch = ContactBatch::with_capacity(caps.contact_pairs);
        self.contact_gate = ContactGate::with_capacity(caps.contact_pairs);
        self.sensor_filters = SortedMap::with_capacity(caps.sensor_filters);
        self.drives = drive::Drives::with_capacity(caps.props);
    }

    // Build one body per collider-bearing entity from its per-instance
    // components (Transform + Collider + optional BodyDynamics + the Pickup
    // tag), keying `body_handles` by each entity's asset identity
    // so the joint wiring resolves.
    fn build_prop_bodies(
        &mut self,
        ctx: &PipelineContext<'_>,
        world: &mut Simulation,
        body_handles: &mut BTreeMap<AssetId, BodyHandle>,
    ) {
        for (entity, collider, transform) in ctx.join2::<Collider, Transform>() {
            let snap = prop_snap(ctx, entity, collider, transform);
            let Some(handle) = self.props.add(&self.layers, world, entity, &snap) else {
                continue;
            };
            if let Some(identity) = ctx.get::<Identity>(entity) {
                body_handles.insert(identity.id(), handle);
            }
        }
    }
}

impl PhysicsSystem {
    /// The system for a world with physics content: a `PhysicsConfig`,
    /// `RigidBody`, `PropBody`, `TriggerVolume`, or a skinned mesh that bakes a
    /// character capsule. A world with content but no config (one built
    /// directly rather than cooked) gets the flat-floor default.
    pub fn for_world(world: &World) -> Option<Self> {
        Self::config_for(world).map(|config| Self::new(&config).with_water(water_of(world)))
    }

    fn config_for(world: &World) -> Option<PhysicsConfig> {
        let needs = world.query::<PhysicsConfig>().next().is_some()
            || world.query::<RigidBody>().next().is_some()
            || world.query::<PropBody>().next().is_some()
            || world.query::<TriggerVolume>().next().is_some()
            || world
                .resource::<crate::resource::SkinnedMeshTable>()
                .is_some_and(crate::resource::SkinnedMeshTable::has_capsule);
        needs.then(|| {
            world
                .query::<PhysicsConfig>()
                .next()
                .cloned()
                .unwrap_or_default()
        })
    }

    // A camera a track drives is posed by the track alone, so it has no
    // capsule; one handed back to its controller gets its capsule again.
    fn hand_camera_over(&mut self, ctx: &PipelineContext<'_>) {
        let Some(world) = self.world.as_mut() else {
            return;
        };
        let tracked = player::camera_is_tracked(ctx);
        if tracked && let Some(player) = self.player.take() {
            world.remove_body(player.handle);
        } else if !tracked && self.player.is_none() {
            self.player = player::spawn(
                world,
                ctx,
                &mut self.floor_y,
                self.layers.mask(LAYER_CHARACTER),
            );
        }
    }

    // Catch up with a planet world: carry the bodies through every move of
    // the simulated frame this system has not yet seen, swap in a newer
    // ground patch, and aim gravity at the planet's center in this frame.
    fn follow_planet(&mut self, ctx: &mut PipelineContext<'_>) {
        let Some(world) = self.world.as_mut() else {
            return;
        };
        if let Some(rebase) = self.rebases.take(ctx.resource::<FrameRebases>()) {
            world.rebase(rebase.rotation, rebase.translation);
            self.props.rebase(&rebase, |entity| {
                ctx.get::<Transform>(entity)
                    .map(|t| (t.position, t.rotation_deg))
            });
            if let Some(player) = self.player.as_mut() {
                player.center.rebase(&rebase);
                // Carried like the camera, so a camera another system posed
                // since is still seen as moved from outside.
                player.written_eye = player.written_eye.map(|eye| rebase.apply_point(eye));
            }
            // The authored regions, whose sensor bodies the move just carried.
            for volume in ctx.query_mut::<TriggerVolume>() {
                volume.position = rebase.apply_point(volume.position);
                volume.rotation_deg = rebase.apply_euler_deg(volume.rotation_deg);
            }
            for control in ctx.query_mut::<RollingControl>() {
                control.heading = rebase.apply_vector(control.heading);
            }
            for rig in &mut self.rigs {
                let written = rig
                    .resolve(ctx)
                    .and_then(|e| ctx.get::<crate::components::CharacterRig>(e))
                    .map(|r| r.position);
                rig.rebase(&rebase, written);
            }
        }
        if let Some(gravity) = ctx.resource::<PlanetGravity>() {
            world.set_gravity_center(Some(gravity.center));
            self.gravity = gravity.strength;
        }
        world.set_air(planet_air(ctx));
        if let (Some((handle, revision)), Some(ground)) = (
            self.planet_ground.as_mut(),
            ctx.resource::<PlanetGroundPatch>(),
        ) && ground.revision != *revision
            && super::terrain::replace_patch(world, *handle, &ground.patch)
        {
            *revision = ground.revision;
        }
        // The ground is only the patch: a prop off it is held until it is
        // back on, rather than falling into the planet.
        if self.planet_ground.is_some()
            && let Some(ground) = ctx.resource::<PlanetGroundPatch>()
        {
            let carried = self.held.and_then(|i| self.props.get(i)).map(|p| p.handle);
            let bodies = self
                .props
                .iter()
                .filter(|p| p.dynamic && !p.is_posed() && Some(p.handle) != carried)
                .map(|p| (p.handle, p.float.is_some_and(|f| f.afloat())));
            self.off_ground.update(
                world,
                bodies,
                &super::off_ground::Footprint::of(&ground.patch),
                ctx.resource::<PlanetFrame>(),
            );
        }
    }
}

// The body-building view of one collider-bearing entity.
fn prop_snap(
    ctx: &PipelineContext<'_>,
    entity: Entity,
    collider: &Collider,
    transform: &Transform,
) -> PropCollSnap {
    PropCollSnap {
        shape: collider_shape(&collider.0, transform.scale),
        layer: collider.0.layer.clone(),
        position: transform.position,
        rotation_deg: transform.rotation_deg,
        pickup: ctx.get::<Pickup>(entity).is_some(),
        dynamics: ctx.get::<BodyDynamics>(entity).copied(),
    }
}

// The world's water as its buoyant bodies read it, from the same surfaces,
// wind and planet the renderer resolves; `None` without water.
fn water_of(world: &World) -> Option<WaterQuery> {
    use crate::components::{Planet, WaterSurface, Wind};
    use crate::ecs::identity::Identity;
    use crate::environment::water::split_water;
    world.query::<WaterSurface>().next()?;
    let wind = WindField::new(world.query::<Wind>().next());
    let surfaces = world
        .query_with_entity::<WaterSurface>()
        .map(|(entity, s)| (world.get::<Identity>(entity).map(|i| i.id()), s));
    let (rectangles, sea) = split_water(surfaces, world.query::<Planet>().next());
    Some(WaterQuery::new(rectangles, &wind, sea))
}

// The entity whose Transform the world's camera track owns, if any: the
// subject the track accepted, while the track is there.
fn tracked_subject(ctx: &PipelineContext<'_>) -> Option<Entity> {
    ctx.query::<CameraTrack>().next()?;
    let subject = ctx.resource::<CameraTrackStatus>()?.subject?;
    ctx.entity_of(subject)
}

// How the planet's air thins over its ground, or `None` without a planet or
// with air as thick at every height.
fn planet_air(ctx: &PipelineContext<'_>) -> Option<AirFalloff> {
    let radius = ctx.resource::<PlanetFrame>()?.shape.radius;
    let scale_height = ctx
        .query::<crate::components::Planet>()
        .next()?
        .air_scale_height;
    (scale_height.is_finite() && scale_height > 0.0).then(|| AirFalloff {
        radius: concinnity_core::math::narrow_f32(radius),
        scale_height,
    })
}

// Up at `p`: away from the pull, which is `+Y` without a planet.
fn up_at(world: &Simulation, p: [f32; 3]) -> [f32; 3] {
    match world.config().gravity_center {
        Some(center) => normalize_or(sub(p, center), 1e-6, [0.0, 1.0, 0.0]),
        None => [0.0, 1.0, 0.0],
    }
}

impl System for PhysicsSystem {
    fn init(&mut self, ctx: &mut PipelineContext<'_>) {
        // Before anything is built, and before the joint wiring below drains
        // the column the scan counts.
        let budget = self.resolve_budget(ctx);
        self.body_cap = budget.body_cap();
        self.reserve(&budget);

        // The simulation reserves the whole budget here, so nothing on the
        // step path allocates and a body past the reservation is refused
        // rather than grown into.
        let planet_gravity = ctx.resource::<PlanetGravity>().copied();
        self.gravity = planet_gravity.map_or(GRAVITY, |g| g.strength);
        let mut world = Simulation::new(
            SimConfig {
                gravity: self.gravity,
                gravity_center: planet_gravity.map(|g| g.center),
                ..SimConfig::default()
            },
            budget.body_cap() as usize,
        );
        self.rebases = RebaseCursor::at(ctx.resource::<FrameRebases>());
        // Room for a water load per buoyant body, and per spawn that might be
        // one.
        if self.water.is_some() {
            let buoyant = ctx
                .query::<BodyDynamics>()
                .filter(|d| d.buoyancy.is_some())
                .count();
            world.reserve_fluid_loads(buoyant + self.spawn_headroom as usize);
        }
        world.set_contact_min_impulse(self.contact_min_impulse, SimTiming::TICK_DT);
        // The step's per-worker scratch, reserved from the schedule this world
        // will run under. A serial schedule reserves one worker's worth, which
        // is what a simulation that is never lent threads keeps.
        world.reserve_workers(
            self.fanout
                .worker_count(crate::protocol::schedule_mode(ctx.resources)),
        );
        let world_mask = self.layers.mask(LAYER_WORLD);

        // The ground: every terrain's heightfield and the planet's patch, or a
        // flat slab when the world lays down neither.
        self.planet_ground = ctx.resource::<PlanetGroundPatch>().and_then(|ground| {
            let handle = super::terrain::add_patch(&mut world, &ground.patch, world_mask)?;
            Some((handle, ground.revision))
        });
        if build_terrain_colliders(&mut world, world_mask, ctx) == 0 && self.planet_ground.is_none()
        {
            // A large thin slab whose top face sits at Y = 0.
            world.add_fixed(
                &ColliderShape::Cuboid {
                    half_extents: [500.0, 5.0, 500.0],
                },
                [0.0, -5.0, 0.0],
                [0.0; 3],
                STATIC_FRICTION,
                world_mask,
            );
        }

        // Sensor regions: one fixed sensor body per TriggerVolume, tagged with
        // the volume's AssetId so step's crossing drain maps back to it.
        let trigger_mask = self.layers.mask(LAYER_TRIGGER);
        let volumes: Vec<(AssetId, TriggerVolume)> = ctx
            .join2::<TriggerVolume, Identity>()
            .map(|(_, volume, identity)| (identity.id(), volume.clone()))
            .collect();
        for (id, volume) in &volumes {
            let shape = collider_shape(&volume.collider, [1.0; 3]);
            let tag = u64::from(id.0);
            if world
                .add_sensor(
                    &shape,
                    volume.position,
                    volume.rotation_deg,
                    tag,
                    trigger_mask,
                )
                .is_none()
            {
                continue;
            }
            self.sensor_filters.insert(tag, (*id, volume.detects));
        }

        // Prop name -> BodyHandle, populated alongside `self.prop_bodies`.
        // Joints resolve their `body_a`/`body_b` references through this map.
        let mut body_handles: BTreeMap<AssetId, BodyHandle> = BTreeMap::new();
        self.build_prop_bodies(ctx, &mut world, &mut body_handles);

        // joints
        // Each PhysicsJoint references one or two Props by AssetId. Cross-reference
        // validation already guarantees the Prop exists; here we additionally
        // require the Prop to own a collider (and therefore a body). A PhysicsJoint
        // with body_b empty anchors body_a to a hidden static body created on
        // demand at the world-space `anchor_b`.
        let joints: Vec<PhysicsJoint> = ctx.drain::<PhysicsJoint>();
        for joint in joints {
            let Some(body_a_id) = joint.body_a else {
                continue;
            };
            let Some(handle_a) = body_handles.get(&body_a_id.id()).copied() else {
                continue;
            };
            let handle_b = if let Some(body_b_id) = joint.body_b {
                match body_handles.get(&body_b_id.id()).copied() {
                    Some(h) => h,
                    None => {
                        continue;
                    }
                }
            } else {
                // Static world anchor at anchor_b. Sub-millimeter ball so it
                // takes effectively no space in the broad phase.
                let anchor = world.add_fixed(
                    &ColliderShape::Ball { radius: 0.001 },
                    joint.anchor_b,
                    [0.0; 3],
                    0.0,
                    world_mask,
                );
                match anchor {
                    Some(handle) => handle,
                    None => continue,
                }
            };
            // When body_b is the implicit world anchor, the anchor sits at the
            // origin of that hidden body, not at the authored offset.
            let anchor_b = if joint.body_b.is_some() {
                joint.anchor_b
            } else {
                [0.0, 0.0, 0.0]
            };
            world.add_joint(
                handle_a,
                handle_b,
                joint.anchor_a,
                anchor_b,
                joint_spec(&joint),
            );
        }
        if !player::camera_is_tracked(ctx) {
            self.player = player::spawn(
                &mut world,
                ctx,
                &mut self.floor_y,
                self.layers.mask(LAYER_CHARACTER),
            );
        }

        // Kinematic capsules for the root-motion character rigs published by
        // the renderer (which ran init first this tick).
        super::rig::init_rigs(
            &mut world,
            ctx,
            self.layers.mask(LAYER_CHARACTER),
            &mut self.rigs,
        );

        // Published from the built world: the simulation's own storage is only
        // knowable once it is reserved.
        super::capacity::publish_reservation(concinnity_core::memory::ledger(), &budget, &world);
        self.world = Some(world);
    }

    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        // Freeze while a menu is open: skip the solve and the write-back so the
        // world truly pauses (and an external editor's edits to simulated
        // Transforms are not stomped by a stale blend). The simulation clock
        // holds its accumulator across the pause, so resuming costs one normal
        // frame. The flag is published by whichever system owns the menu, which
        // runs first in the table, so it reflects this same tick.
        if ctx.resource::<MenuActive>().is_some_and(|m| m.0) {
            return StepResult::Continue;
        }

        // A planet world's frame moved since the last step: carry every body
        // along, then take the ground and the pull as they are in the new frame.
        self.follow_planet(ctx);
        self.hand_camera_over(ctx);

        // Held out of `self` for the step so each phase borrows it beside the
        // system's own state; put back before returning.
        let Some(mut world) = self.world.take() else {
            return StepResult::Continue;
        };
        self.step_world(&mut world, ctx);
        self.world = Some(world);
        StepResult::Continue
    }
}

// What the floating bodies push aside after the frame's ticks, for the water's
// wake.
fn publish_stirs(ctx: &mut PipelineContext<'_>, props: &PropBodies) {
    if ctx.resource::<WaterStirs>().is_none() {
        ctx.insert_resource(WaterStirs::default());
    }
    if let Some(stirs) = ctx.resource_mut::<WaterStirs>() {
        props.stirs_into(&mut stirs.0);
    }
}

mod drive;
mod player;
mod step;

#[cfg(test)]
mod drive_tests;
#[cfg(test)]
mod tests;

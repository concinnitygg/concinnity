// concinnity-physics/src/lib.rs
//
// A thin wrapper around the Rapier rigid-body simulation. Rapier (and its
// glam-based math) is confined to this crate: callers work entirely in the
// engine's `[f32; 3]` / Euler-degree representation and only ever see the
// opaque `BodyHandle`.
//
// `PhysicsWorld` owns one Rapier simulation. `PhysicsSystem` builds it once at
// init from the world's Props / bodies and steps it every frame. The engine
// depends on this crate and constructs `PhysicsSystem` through its system
// registry; the dependency arrow is concinnity-physics <- concinnity-engine.

// The cached capsule a character move is resolved against.
mod character;
// Contact-event shaping: extraction, per-frame batching, per-pair refractory.
mod contacts;
mod convert;
// Prev/curr pose snapshots blended by the frame's accumulator alpha.
mod interp;
// Named collision layers over Rapier's 32-bit interaction groups.
mod layers;
// Raycast probe answering for animation IK and the follow camera.
mod probes;
// Prop bodies with their handle -> entity and tracked-entity indices.
mod props;
// Root-motion character rigs: one kinematic capsule per `CharacterRig`.
mod rig;
// The internal physics system that builds and steps the simulation from the
// world's bodies, driven by an optional `PhysicsConfig`.
mod system;

pub use character::CharacterShape;
pub use contacts::ContactHit;
pub use layers::{LAYER_CHARACTER, LAYER_PROP, LAYER_TRIGGER, LAYER_WORLD, LayerMask};
// The physics system the engine registry wraps, plus the shared gravity
// constant the third-person controller matches its jump takeoff against.
pub use system::{GRAVITY, PhysicsSystem};

// Asset-to-physics conversions live with the other Rapier-type conversions so
// the asset data types (Joint, Prop, PropBody) carry no dependency on the
// physics backend.
pub(crate) use convert::{collider_shape, dynamic_params, joint_spec};
use convert::{from_rotation, from_vec, to_quat, to_rotation, to_vec};
use rapier3d::control::{CharacterLength, KinematicCharacterController};
use rapier3d::math::Pose;
use rapier3d::parry::query::DefaultQueryDispatcher;
use rapier3d::parry::utils::Array2;
use rapier3d::prelude::*;

// Opaque handle to a body inside a [`PhysicsWorld`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyHandle(RigidBodyHandle);

// Opaque handle to a joint inside a [`PhysicsWorld`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JointHandle(ImpulseJointHandle);

// Constraint shape connecting two bodies.
#[derive(Debug, Clone, Copy)]
pub enum JointSpec {
    // All six degrees of freedom locked: bodies move and rotate as one rigid
    // assembly relative to their anchors.
    Fixed,
    // Hinge: rotation is allowed only around `axis` (expressed in each body's
    // local frame). `limits` clamps the hinge angle in radians; `motor`
    // drives it at a target angular velocity in radians/second.
    Revolute {
        axis: [f32; 3],
        limits: Option<[f32; 2]>,
        motor: Option<JointMotor>,
    },
    // Ball-and-socket: translation locked, all three rotational axes free.
    Spherical,
    // Slider: translation is allowed only along `axis` (in each body's local
    // frame). `limits` clamps the slide distance in world units; `motor`
    // drives it at a target linear velocity in units/second.
    Prismatic {
        axis: [f32; 3],
        limits: Option<[f32; 2]>,
        motor: Option<JointMotor>,
    },
}

// Velocity-driven motor parameters for a revolute / prismatic joint.
#[derive(Debug, Clone, Copy)]
pub struct JointMotor {
    // Target velocity (radians/second for revolute, units/second for prismatic).
    pub target_velocity: f32,
    // Maximum force the motor may apply to reach the target.
    pub max_force: f32,
}

// A collision shape, in the body's local space.
#[derive(Debug, Clone, Copy)]
pub enum ColliderShape {
    // Box with the given half-extents along x, y, z.
    Cuboid { half_extents: [f32; 3] },
    // Sphere of the given radius.
    Ball { radius: f32 },
    // Y-axis capsule: a cylinder of `2 * half_height` capped by hemispheres.
    Capsule { half_height: f32, radius: f32 },
}

// Physical parameters for a dynamic (freely simulated) body.
#[derive(Debug, Clone, Copy)]
pub struct DynamicParams {
    // Mass in kilograms. `0.0` lets Rapier derive mass from shape volume.
    pub mass: f32,
    // Coulomb friction coefficient.
    pub friction: f32,
    // Bounciness in `[0, 1]`.
    pub restitution: f32,
    // Multiplier on the world gravity for this body.
    pub gravity_scale: f32,
    // Linear velocity damping (air drag).
    pub linear_damping: f32,
}

// One character-capsule move request, resolved against the scene by
// [`PhysicsWorld::move_character`].
#[derive(Debug, Clone, Copy)]
pub struct CharacterMoveInput {
    // World-space capsule centre before the move.
    pub center: [f32; 3],
    // Desired translation for this tick.
    pub desired: [f32; 3],
    pub dt: f32,
    // The moving capsule's own body, left out of the collision query.
    pub exclude: BodyHandle,
    pub mask: LayerMask,
}

// Result of moving the character capsule for one frame.
#[derive(Debug, Clone, Copy)]
pub struct CharacterMove {
    // The translation actually applied after collision resolution.
    pub translation: [f32; 3],
    // True when the capsule is resting on a surface after the move.
    pub grounded: bool,
}

// One raycast hit: the surface point, its outward normal, and the distance
// from the ray origin.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub distance: f32,
}

// One boundary crossing of a sensor region recorded during a step: something
// began or stopped overlapping the sensor tagged `tag` (the caller's value
// from [`PhysicsWorld::add_sensor`]). `other` is the crossing body, `None`
// when it was removed from the simulation in the same step.
#[derive(Debug, Clone, Copy)]
pub struct SensorCrossing {
    pub tag: u64,
    pub other: Option<BodyHandle>,
    pub entered: bool,
}

// Collects sensor boundary crossings and contact hits from inside the Rapier
// pipeline; drained after each step by `drain_sensor_crossings` /
// `drain_contact_hits`. Handles resolve to tags and parent bodies here,
// inside the callbacks, while the collider set still contains both sides of
// the pair.
#[derive(Default)]
struct EventSink {
    crossings: std::sync::Mutex<Vec<SensorCrossing>>,
    contacts: std::sync::Mutex<Vec<ContactHit>>,
}

impl EventHandler for EventSink {
    fn handle_collision_event(
        &self,
        _bodies: &RigidBodySet,
        colliders: &ColliderSet,
        event: CollisionEvent,
        _contact_pair: Option<&ContactPair>,
    ) {
        let (h1, h2, entered) = match event {
            CollisionEvent::Started(h1, h2, _) => (h1, h2, true),
            CollisionEvent::Stopped(h1, h2, _) => (h1, h2, false),
        };
        // Either side may be one of our sensors (two overlapping sensors
        // record a crossing each); a collider already removed this step
        // resolves to None.
        let record = |sensor: ColliderHandle, other: ColliderHandle| {
            let Some(sensor) = colliders.get(sensor).filter(|c| c.is_sensor()) else {
                return;
            };
            let other = colliders
                .get(other)
                .and_then(|c| c.parent())
                .map(BodyHandle);
            self.crossings.lock().unwrap().push(SensorCrossing {
                tag: sensor.user_data as u64,
                other,
                entered,
            });
        };
        record(h1, h2);
        record(h2, h1);
    }

    // Called only for pairs whose total force passed the collider's
    // contact-force threshold (set from the world's minimum contact impulse).
    fn handle_contact_force_event(
        &self,
        dt: Real,
        _bodies: &RigidBodySet,
        colliders: &ColliderSet,
        contact_pair: &ContactPair,
        total_force_magnitude: Real,
    ) {
        if let Some(hit) = contacts::extract_hit(dt, colliders, contact_pair, total_force_magnitude)
        {
            self.contacts.lock().unwrap().push(hit);
        }
    }
}

// One Rapier rigid-body simulation.
pub struct PhysicsWorld {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    integration_parameters: IntegrationParameters,
    pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    gravity: Vector,
    character: KinematicCharacterController,
    events: EventSink,
    // Total-force threshold below which the pipeline skips contact events;
    // derived from the world's minimum contact impulse and the fixed tick.
    contact_force_threshold: f32,
}

impl std::fmt::Debug for PhysicsWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicsWorld")
            .field("bodies", &self.bodies.len())
            .field("colliders", &self.colliders.len())
            .finish()
    }
}

impl PhysicsWorld {
    // Create an empty world. `gravity` is the downward acceleration magnitude
    // in world units per second squared.
    pub fn new(gravity: f32) -> Self {
        Self {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            integration_parameters: IntegrationParameters::default(),
            pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            gravity: Vector::new(0.0, -gravity, 0.0),
            character: KinematicCharacterController::default(),
            events: EventSink::default(),
            contact_force_threshold: 60.0,
        }
    }

    // Set the minimum contact impulse for contact events, converted to the
    // total-force threshold Rapier gates on at `tick_dt`. Applies to dynamic
    // bodies added afterwards, so call before populating the world.
    pub fn set_contact_min_impulse(&mut self, min_impulse: f32, tick_dt: f32) {
        self.contact_force_threshold = min_impulse.max(0.0) / tick_dt.max(1.0e-6);
    }

    // Tune the character controller. `grounded` is true for a gravity-bound
    // character (auto-steps, snaps to the ground) and false for a free-flying
    // camera (no auto-step, no ground snap). A slope of `0` disables the
    // climb limit.
    pub fn configure_character(&mut self, max_slope_deg: f32, step_height: f32, grounded: bool) {
        if max_slope_deg > 0.0 {
            self.character.max_slope_climb_angle = max_slope_deg.to_radians();
            self.character.min_slope_slide_angle = max_slope_deg.to_radians();
        }
        self.character.autostep = if step_height > 0.0 && grounded {
            Some(rapier3d::control::CharacterAutostep {
                max_height: CharacterLength::Absolute(step_height),
                min_width: CharacterLength::Absolute(0.05),
                include_dynamic_bodies: true,
            })
        } else {
            None
        };
        if !grounded {
            self.character.snap_to_ground = None;
        }
    }

    fn collider_builder(shape: &ColliderShape) -> ColliderBuilder {
        match *shape {
            ColliderShape::Cuboid {
                half_extents: [x, y, z],
            } => ColliderBuilder::cuboid(x, y, z),
            ColliderShape::Ball { radius } => ColliderBuilder::ball(radius),
            ColliderShape::Capsule {
                half_height,
                radius,
            } => ColliderBuilder::capsule_y(half_height, radius),
        }
    }

    fn interaction_groups(mask: LayerMask) -> InteractionGroups {
        InteractionGroups::new(
            Group::from_bits_retain(mask.memberships),
            Group::from_bits_retain(mask.filter),
            InteractionTestMode::And,
        )
    }

    // Add an immovable body (terrain, walls, static props).
    pub fn add_fixed(
        &mut self,
        shape: &ColliderShape,
        pos: [f32; 3],
        euler_deg: [f32; 3],
        friction: f32,
        mask: LayerMask,
    ) -> BodyHandle {
        let body = RigidBodyBuilder::fixed()
            .pose(Pose::from_parts(to_vec(pos), to_rotation(euler_deg)))
            .build();
        let handle = self.bodies.insert(body);
        let groups = Self::interaction_groups(mask);
        let collider = Self::collider_builder(shape)
            .friction(friction)
            .collision_groups(groups)
            .solver_groups(groups)
            .build();
        self.colliders
            .insert_with_parent(collider, handle, &mut self.bodies);
        BodyHandle(handle)
    }

    // Add a freely simulated dynamic body. Dynamic bodies are the contact
    // event sources: any pair whose total force passes the world's threshold
    // records a `ContactHit`.
    pub fn add_dynamic(
        &mut self,
        shape: &ColliderShape,
        pos: [f32; 3],
        euler_deg: [f32; 3],
        params: DynamicParams,
        mask: LayerMask,
    ) -> BodyHandle {
        let mut body = RigidBodyBuilder::dynamic()
            .pose(Pose::from_parts(to_vec(pos), to_rotation(euler_deg)))
            .gravity_scale(params.gravity_scale)
            .linear_damping(params.linear_damping)
            .ccd_enabled(true);
        if params.mass > 0.0 {
            body = body.additional_mass(params.mass);
        }
        let handle = self.bodies.insert(body.build());
        let groups = Self::interaction_groups(mask);
        let collider = Self::collider_builder(shape)
            .friction(params.friction)
            .restitution(params.restitution)
            .collision_groups(groups)
            .solver_groups(groups)
            .active_events(ActiveEvents::CONTACT_FORCE_EVENTS)
            .contact_force_event_threshold(self.contact_force_threshold)
            .build();
        self.colliders
            .insert_with_parent(collider, handle, &mut self.bodies);
        BodyHandle(handle)
    }

    // Add an immovable sensor region: it detects overlap (recorded as
    // `SensorCrossing`s with the caller's `tag`) but never collides. The
    // kinematic-vs-fixed pair is enabled explicitly so the position-kinematic
    // character capsules set it off; rapier excludes fixed-vs-fixed pairs, so
    // static geometry never does.
    pub fn add_sensor(
        &mut self,
        shape: &ColliderShape,
        pos: [f32; 3],
        euler_deg: [f32; 3],
        tag: u64,
        mask: LayerMask,
    ) -> BodyHandle {
        let body = RigidBodyBuilder::fixed()
            .pose(Pose::from_parts(to_vec(pos), to_rotation(euler_deg)))
            .build();
        let handle = self.bodies.insert(body);
        let collider = Self::collider_builder(shape)
            .sensor(true)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .active_collision_types(
                ActiveCollisionTypes::default() | ActiveCollisionTypes::KINEMATIC_FIXED,
            )
            .collision_groups(Self::interaction_groups(mask))
            .user_data(tag as u128)
            .build();
        self.colliders
            .insert_with_parent(collider, handle, &mut self.bodies);
        BodyHandle(handle)
    }

    // The sensor boundary crossings recorded by the last `step`, oldest first.
    pub fn drain_sensor_crossings(&mut self) -> Vec<SensorCrossing> {
        std::mem::take(&mut *self.events.crossings.lock().unwrap())
    }

    // The contact hits recorded by the last `step`, oldest first. Only pairs
    // whose total force passed the world's contact threshold appear.
    pub fn drain_contact_hits(&mut self) -> Vec<ContactHit> {
        std::mem::take(&mut *self.events.contacts.lock().unwrap())
    }

    // Add the player character capsule as a position-kinematic body. `center`
    // is the world-space position of the capsule centre.
    pub fn add_character(
        &mut self,
        half_height: f32,
        radius: f32,
        center: [f32; 3],
        mask: LayerMask,
    ) -> BodyHandle {
        let body = RigidBodyBuilder::kinematic_position_based()
            .translation(to_vec(center))
            .build();
        let handle = self.bodies.insert(body);
        let groups = Self::interaction_groups(mask);
        let collider = ColliderBuilder::capsule_y(half_height, radius)
            .collision_groups(groups)
            .solver_groups(groups)
            .build();
        self.colliders
            .insert_with_parent(collider, handle, &mut self.bodies);
        BodyHandle(handle)
    }

    // Add a static heightfield. `heights` is a `rows * cols` row-major grid of
    // world-space Y values; `scale` is the full extent `[width, 1, depth]`.
    pub fn add_heightfield(
        &mut self,
        rows: usize,
        cols: usize,
        heights: Vec<f32>,
        scale: [f32; 3],
        pos: [f32; 3],
        mask: LayerMask,
    ) {
        let grid = Array2::new(rows, cols, heights);
        let body = RigidBodyBuilder::fixed().translation(to_vec(pos)).build();
        let handle = self.bodies.insert(body);
        let groups = Self::interaction_groups(mask);
        let collider = ColliderBuilder::heightfield(grid, to_vec(scale))
            .friction(1.0)
            .collision_groups(groups)
            .solver_groups(groups)
            .build();
        self.colliders
            .insert_with_parent(collider, handle, &mut self.bodies);
    }

    // Constrain two bodies with a joint. Anchors are in each body's local
    // frame. Returns a handle for future inspection / removal.
    pub fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        spec: JointSpec,
    ) -> JointHandle {
        let anchor1 = to_vec(anchor_a);
        let anchor2 = to_vec(anchor_b);
        let generic: GenericJoint = match spec {
            JointSpec::Fixed => {
                let mut j = FixedJoint::new();
                j.set_local_anchor1(anchor1);
                j.set_local_anchor2(anchor2);
                j.data
            }
            JointSpec::Revolute {
                axis,
                limits,
                motor,
            } => {
                let axis = normalize_axis(axis);
                let mut j = RevoluteJoint::new(to_vec(axis));
                j.set_local_anchor1(anchor1);
                j.set_local_anchor2(anchor2);
                if let Some([min, max]) = limits {
                    j.set_limits([min, max]);
                }
                if let Some(m) = motor {
                    j.set_motor_velocity(m.target_velocity, 1.0);
                    j.set_motor_max_force(m.max_force);
                }
                j.data
            }
            JointSpec::Spherical => {
                let mut j = SphericalJoint::new();
                j.set_local_anchor1(anchor1);
                j.set_local_anchor2(anchor2);
                j.data
            }
            JointSpec::Prismatic {
                axis,
                limits,
                motor,
            } => {
                let axis = normalize_axis(axis);
                let mut j = PrismaticJoint::new(to_vec(axis));
                j.set_local_anchor1(anchor1);
                j.set_local_anchor2(anchor2);
                if let Some([min, max]) = limits {
                    j.set_limits([min, max]);
                }
                if let Some(m) = motor {
                    j.set_motor_velocity(m.target_velocity, 1.0);
                    j.set_motor_max_force(m.max_force);
                }
                j.data
            }
        };
        let handle = self
            .impulse_joints
            .insert(body_a.0, body_b.0, generic, true);
        JointHandle(handle)
    }

    // Advance the simulation by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        self.integration_parameters.dt = dt;
        self.pipeline.step(
            self.gravity,
            &self.integration_parameters,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &(),
            &self.events,
        );
    }

    // Resolve a desired move of a character capsule against the world without
    // mutating it. Apply the result with [`Self::set_kinematic_translation`].
    // `shape` is the mover's capsule, owned by the caller so the fixed tick
    // reuses one per character instead of building a fresh one per call.
    // `input.exclude` is the moving capsule's own body (from
    // [`Self::add_character`]), left out of the query so the character does
    // not collide with itself; other characters' capsules stay solid to it.
    // Sensors never block the move.
    pub fn move_character(
        &self,
        shape: &CharacterShape,
        input: &CharacterMoveInput,
    ) -> CharacterMove {
        let dispatcher = DefaultQueryDispatcher;
        let filter = QueryFilter::default()
            .exclude_rigid_body(input.exclude.0)
            .exclude_sensors()
            .groups(Self::interaction_groups(input.mask));
        let query =
            self.broad_phase
                .as_query_pipeline(&dispatcher, &self.bodies, &self.colliders, filter);
        let movement = self.character.move_shape(
            input.dt.max(1.0e-4),
            &query,
            &**shape.shape(),
            &Pose::from_translation(to_vec(input.center)),
            to_vec(input.desired),
            |_collision| {},
        );
        CharacterMove {
            translation: from_vec(movement.translation),
            grounded: movement.grounded,
        }
    }

    // Cast a ray into the scene, returning the nearest hit within
    // `max_dist`. `dir` need not be unit length (a zero direction misses).
    // `exclude` leaves one body out of the query -- pass the probing
    // character's own capsule so a ray from inside it reaches the world.
    // `mask` restricts the hit set to compatible layers; sensors never hit.
    pub fn raycast(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        max_dist: f32,
        exclude: Option<BodyHandle>,
        mask: LayerMask,
    ) -> Option<RayHit> {
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if len < 1.0e-6 || max_dist <= 0.0 {
            return None;
        }
        let unit = [dir[0] / len, dir[1] / len, dir[2] / len];
        let dispatcher = DefaultQueryDispatcher;
        let mut filter = QueryFilter::default()
            .exclude_sensors()
            .groups(Self::interaction_groups(mask));
        if let Some(handle) = exclude {
            filter = filter.exclude_rigid_body(handle.0);
        }
        let query =
            self.broad_phase
                .as_query_pipeline(&dispatcher, &self.bodies, &self.colliders, filter);
        let ray = Ray::new(to_vec(origin), to_vec(unit));
        let (_, hit) = query.cast_ray_and_get_normal(&ray, max_dist, true)?;
        Some(RayHit {
            point: [
                origin[0] + unit[0] * hit.time_of_impact,
                origin[1] + unit[1] * hit.time_of_impact,
                origin[2] + unit[2] * hit.time_of_impact,
            ],
            normal: [hit.normal.x, hit.normal.y, hit.normal.z],
            distance: hit.time_of_impact,
        })
    }

    // Set the next-frame target position of a kinematic body.
    pub fn set_kinematic_translation(&mut self, handle: BodyHandle, pos: [f32; 3]) {
        if let Some(body) = self.bodies.get_mut(handle.0) {
            body.set_next_kinematic_translation(to_vec(pos));
        }
    }

    // Switch a body to position-kinematic control (used while a prop is held).
    pub fn make_kinematic(&mut self, handle: BodyHandle) {
        if let Some(body) = self.bodies.get_mut(handle.0) {
            body.set_body_type(RigidBodyType::KinematicPositionBased, true);
        }
    }

    // Switch a body back to dynamic simulation and give it a launch velocity.
    pub fn make_dynamic(&mut self, handle: BodyHandle, linear_velocity: [f32; 3]) {
        if let Some(body) = self.bodies.get_mut(handle.0) {
            body.set_body_type(RigidBodyType::Dynamic, true);
            body.set_linvel(to_vec(linear_velocity), true);
        }
    }

    // Remove a body and its colliders from the world (used when its owning
    // entity is despawned). Rapier purges any joints incident on the body as
    // part of the removal, so no separate joint cleanup is needed.
    pub fn remove_body(&mut self, handle: BodyHandle) {
        self.bodies.remove(
            handle.0,
            &mut self.islands,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            true,
        );
    }

    // Number of rigid bodies currently in the world (player, props, anchors).
    // Test-only observable for the body-reaping path.
    #[cfg(test)]
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    // Number of colliders currently in the world. Test-only observable for
    // the spawn/despawn leak checks.
    #[cfg(test)]
    pub fn collider_count(&self) -> usize {
        self.colliders.len()
    }

    // Read a body's current world-space position and Euler rotation.
    pub fn body_pose(&self, handle: BodyHandle) -> ([f32; 3], [f32; 3]) {
        match self.bodies.get(handle.0) {
            Some(body) => (
                from_vec(body.translation()),
                from_rotation(*body.rotation()),
            ),
            None => ([0.0; 3], [0.0; 3]),
        }
    }

    // Read a body's current world-space position and `[x, y, z, w]` rotation
    // quaternion, for pose interpolation (which blends in quaternion space).
    pub fn body_pose_quat(&self, handle: BodyHandle) -> ([f32; 3], [f32; 4]) {
        match self.bodies.get(handle.0) {
            Some(body) => (from_vec(body.translation()), to_quat(*body.rotation())),
            None => ([0.0; 3], [0.0, 0.0, 0.0, 1.0]),
        }
    }
}

// Normalize a non-zero axis; fall back to +Y for a degenerate (zero) input so
// the joint stays valid instead of crashing inside Rapier.
fn normalize_axis(axis: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = axis;
    let len = (x * x + y * y + z * z).sqrt();
    if len > 1.0e-6 {
        [x / len, y / len, z / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f32 = 20.0;

    fn ground(world: &mut PhysicsWorld, top_y: f32) {
        world.add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [50.0, 5.0, 50.0],
            },
            [0.0, top_y - 5.0, 0.0],
            [0.0; 3],
            0.8,
            LayerMask::ALL,
        );
    }

    #[test]
    fn dynamic_body_falls_under_gravity() {
        let mut world = PhysicsWorld::new(G);
        let body = world.add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 10.0, 0.0],
            [0.0; 3],
            DynamicParams {
                mass: 1.0,
                friction: 0.5,
                restitution: 0.0,
                gravity_scale: 1.0,
                linear_damping: 0.0,
            },
            LayerMask::ALL,
        );
        for _ in 0..30 {
            world.step(1.0 / 60.0);
        }
        let (pos, _) = world.body_pose(body);
        assert!(pos[1] < 10.0, "body should have fallen, y = {}", pos[1]);
    }

    #[test]
    fn dynamic_body_rests_on_floor() {
        let mut world = PhysicsWorld::new(G);
        ground(&mut world, 0.0);
        let body = world.add_dynamic(
            &ColliderShape::Cuboid {
                half_extents: [0.5, 0.5, 0.5],
            },
            [0.0, 6.0, 0.0],
            [0.0; 3],
            DynamicParams {
                mass: 1.0,
                friction: 0.5,
                restitution: 0.0,
                gravity_scale: 1.0,
                linear_damping: 0.0,
            },
            LayerMask::ALL,
        );
        for _ in 0..240 {
            world.step(1.0 / 60.0);
        }
        let (pos, _) = world.body_pose(body);
        // Half-extent 0.5 means the centre rests at y ~= 0.5.
        assert!(
            (pos[1] - 0.5).abs() < 0.1,
            "box should rest on floor, y = {}",
            pos[1]
        );
    }

    #[test]
    fn character_is_blocked_by_a_wall() {
        let mut world = PhysicsWorld::new(G);
        ground(&mut world, 0.0);
        // A wall just ahead of the capsule on +X.
        world.add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [0.25, 2.0, 4.0],
            },
            [1.5, 2.0, 0.0],
            [0.0; 3],
            0.5,
            LayerMask::ALL,
        );
        let capsule = world.add_character(0.6, 0.3, [0.0, 1.0, 0.0], LayerMask::ALL);
        // The broad-phase BVH the movement query reads is built by step().
        world.step(1.0 / 60.0);
        let moved = world.move_character(
            &CharacterShape::capsule(0.6, 0.3),
            &CharacterMoveInput {
                center: [0.0, 1.0, 0.0],
                desired: [5.0, 0.0, 0.0],
                dt: 1.0 / 60.0,
                exclude: capsule,
                mask: LayerMask::ALL,
            },
        );
        // The wall stands at x = 1.25 (1.5 - 0.25); a 0.3-radius capsule from
        // x = 0 cannot advance the full 5 units into it.
        assert!(
            moved.translation[0] < 1.0,
            "wall should block the capsule, dx = {}",
            moved.translation[0]
        );
    }

    #[test]
    fn character_lands_on_ground() {
        let mut world = PhysicsWorld::new(G);
        ground(&mut world, 0.0);
        // Capsule centre at 1.5: with half-height 0.6 + radius 0.3 the bottom
        // sits 0.6 above the floor, so a 10-unit drop should be arrested.
        let capsule = world.add_character(0.6, 0.3, [0.0, 1.5, 0.0], LayerMask::ALL);
        world.step(1.0 / 60.0);
        let moved = world.move_character(
            &CharacterShape::capsule(0.6, 0.3),
            &CharacterMoveInput {
                center: [0.0, 1.5, 0.0],
                desired: [0.0, -10.0, 0.0],
                dt: 1.0 / 60.0,
                exclude: capsule,
                mask: LayerMask::ALL,
            },
        );
        assert!(
            moved.translation[1] > -1.0 && moved.grounded,
            "fall should be arrested by the floor, dy = {}, grounded = {}",
            moved.translation[1],
            moved.grounded,
        );
    }

    #[test]
    fn raycast_hits_the_ground_with_point_and_normal() {
        let mut world = PhysicsWorld::new(G);
        ground(&mut world, 0.0);
        world.step(1.0 / 60.0);
        let hit = world
            .raycast(
                [0.5, 2.0, -0.25],
                [0.0, -1.0, 0.0],
                5.0,
                None,
                LayerMask::ALL,
            )
            .expect("ray hits the floor");
        assert!((hit.point[1] - 0.0).abs() < 1e-3, "{:?}", hit.point);
        assert!((hit.point[0] - 0.5).abs() < 1e-4);
        assert!((hit.distance - 2.0).abs() < 1e-3);
        assert!(hit.normal[1] > 0.99, "upward normal: {:?}", hit.normal);
        // Out of range or pointing away: no hit.
        assert!(
            world
                .raycast([0.0, 2.0, 0.0], [0.0, -1.0, 0.0], 1.0, None, LayerMask::ALL)
                .is_none()
        );
        assert!(
            world
                .raycast(
                    [0.0, 2.0, 0.0],
                    [0.0, 1.0, 0.0],
                    100.0,
                    None,
                    LayerMask::ALL
                )
                .is_none()
        );
    }

    #[test]
    fn raycast_excluded_body_is_transparent() {
        let mut world = PhysicsWorld::new(G);
        ground(&mut world, 0.0);
        // A capsule standing between the ray origin and the floor.
        let capsule = world.add_character(0.5, 0.3, [0.0, 0.8, 0.0], LayerMask::ALL);
        world.step(1.0 / 60.0);
        let through = world
            .raycast(
                [0.0, 3.0, 0.0],
                [0.0, -1.0, 0.0],
                5.0,
                Some(capsule),
                LayerMask::ALL,
            )
            .expect("ray passes through the excluded capsule");
        assert!((through.point[1] - 0.0).abs() < 1e-3, "{:?}", through.point);
        let blocked = world
            .raycast([0.0, 3.0, 0.0], [0.0, -1.0, 0.0], 5.0, None, LayerMask::ALL)
            .expect("ray hits the capsule");
        assert!(blocked.point[1] > 1.0, "{:?}", blocked.point);
    }

    // Test helper: a tiny zero-volume static body acting as a world anchor.
    fn world_anchor(world: &mut PhysicsWorld, pos: [f32; 3]) -> BodyHandle {
        world.add_fixed(
            &ColliderShape::Ball { radius: 0.01 },
            pos,
            [0.0; 3],
            0.0,
            LayerMask::ALL,
        )
    }

    fn free_ball(world: &mut PhysicsWorld, pos: [f32; 3]) -> BodyHandle {
        world.add_dynamic(
            &ColliderShape::Ball { radius: 0.2 },
            pos,
            [0.0; 3],
            DynamicParams {
                mass: 1.0,
                friction: 0.5,
                restitution: 0.0,
                gravity_scale: 1.0,
                linear_damping: 0.0,
            },
            LayerMask::ALL,
        )
    }

    #[test]
    fn revolute_pendulum_swings_under_gravity() {
        // A ball hanging 2 m below a fixed anchor. The revolute hinge is on the
        // +Z axis, so under -Y gravity the ball is free to swing in the X-Y
        // plane. After a few seconds it should have arced away from straight
        // below the anchor.
        let mut world = PhysicsWorld::new(G);
        let anchor = world_anchor(&mut world, [0.0, 5.0, 0.0]);
        // Spawn the ball offset so the pendulum starts above horizontal: a
        // straight-down hang is an unstable equilibrium that may not move.
        let ball = free_ball(&mut world, [2.0, 3.0, 0.0]);
        world.add_joint(
            anchor,
            ball,
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            JointSpec::Revolute {
                axis: [0.0, 0.0, 1.0],
                limits: None,
                motor: None,
            },
        );

        let mut max_speed = 0.0_f32;
        for _ in 0..240 {
            world.step(1.0 / 60.0);
            // Without a velocity getter we infer motion by re-querying position;
            // grab the max horizontal swing from the rest pose.
            let (pos, _) = world.body_pose(ball);
            let dx = pos[0];
            let speed_proxy = dx.abs();
            if speed_proxy > max_speed {
                max_speed = speed_proxy;
            }
        }
        let (pos, _) = world.body_pose(ball);
        // The bob's distance from the anchor should still be ~2 m (the
        // revolute joint preserves the 2 m offset).
        let dx = pos[0];
        let dy = pos[1] - 5.0;
        let length = (dx * dx + dy * dy).sqrt();
        assert!(
            (length - 2.0).abs() < 0.2,
            "pendulum length drifted: {length} (expected ~2.0)"
        );
        assert!(
            max_speed > 0.5,
            "pendulum should have swung; max |dx| = {max_speed}"
        );
    }

    #[test]
    fn fixed_joint_keeps_bodies_attached() {
        // Two dynamic bodies welded together by a fixed joint fall together;
        // their relative offset is preserved.
        let mut world = PhysicsWorld::new(G);
        ground(&mut world, -10.0);
        let a = free_ball(&mut world, [0.0, 5.0, 0.0]);
        let b = free_ball(&mut world, [1.0, 5.0, 0.0]);
        world.add_joint(a, b, [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], JointSpec::Fixed);
        for _ in 0..120 {
            world.step(1.0 / 60.0);
        }
        let (pa, _) = world.body_pose(a);
        let (pb, _) = world.body_pose(b);
        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let dz = pb[2] - pa[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            (dist - 1.0).abs() < 0.05,
            "fixed joint should preserve 1 m offset, got {dist}"
        );
    }

    #[test]
    fn prismatic_joint_limits_clamp_slide() {
        // A dynamic body anchored to a fixed point with a Y-axis prismatic
        // joint clamped to [-0.5, 0.5] should fall at most 0.5 m below the
        // anchor before the limit catches it.
        let mut world = PhysicsWorld::new(G);
        let anchor = world_anchor(&mut world, [0.0, 5.0, 0.0]);
        let ball = free_ball(&mut world, [0.0, 5.0, 0.0]);
        world.add_joint(
            anchor,
            ball,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            JointSpec::Prismatic {
                axis: [0.0, 1.0, 0.0],
                limits: Some([-0.5, 0.5]),
                motor: None,
            },
        );
        for _ in 0..240 {
            world.step(1.0 / 60.0);
        }
        let (pos, _) = world.body_pose(ball);
        assert!(
            pos[1] >= 5.0 - 0.5 - 0.05,
            "prismatic limit should arrest fall; y = {}",
            pos[1]
        );
        // Bodies should only slide along Y: X/Z drift should be near zero.
        assert!(pos[0].abs() < 0.05, "ball drifted on X: {}", pos[0]);
        assert!(pos[2].abs() < 0.05, "ball drifted on Z: {}", pos[2]);
    }

    #[test]
    fn sensor_records_a_dynamic_body_passing_through() {
        let mut world = PhysicsWorld::new(G);
        world.add_sensor(
            &ColliderShape::Cuboid {
                half_extents: [1.0, 1.0, 1.0],
            },
            [0.0, 2.0, 0.0],
            [0.0; 3],
            7,
            LayerMask::ALL,
        );
        // A ball dropped from above the sensor falls straight through it (no
        // floor), so the run records one enter and one exit.
        let ball = free_ball(&mut world, [0.0, 6.0, 0.0]);
        let mut entered = false;
        let mut exited = false;
        for _ in 0..300 {
            world.step(1.0 / 60.0);
            for crossing in world.drain_sensor_crossings() {
                assert_eq!(crossing.tag, 7);
                assert_eq!(crossing.other, Some(ball));
                if crossing.entered {
                    entered = true;
                } else {
                    assert!(entered, "exit before enter");
                    exited = true;
                }
            }
        }
        assert!(entered && exited, "entered={entered} exited={exited}");
    }

    #[test]
    fn sensor_detects_the_kinematic_character_capsule() {
        let mut world = PhysicsWorld::new(G);
        world.add_sensor(
            &ColliderShape::Cuboid {
                half_extents: [1.0, 1.0, 1.0],
            },
            [5.0, 1.0, 0.0],
            [0.0; 3],
            9,
            LayerMask::ALL,
        );
        let capsule = world.add_character(0.6, 0.3, [0.0, 1.0, 0.0], LayerMask::ALL);
        world.step(1.0 / 60.0);
        let _ = world.drain_sensor_crossings();

        world.set_kinematic_translation(capsule, [5.0, 1.0, 0.0]);
        let mut entered = false;
        for _ in 0..5 {
            world.step(1.0 / 60.0);
            for crossing in world.drain_sensor_crossings() {
                if crossing.tag == 9 && crossing.entered {
                    assert_eq!(crossing.other, Some(capsule));
                    entered = true;
                }
            }
        }
        assert!(entered, "kinematic-vs-fixed sensor pair should report");

        world.set_kinematic_translation(capsule, [0.0, 1.0, 0.0]);
        let mut exited = false;
        for _ in 0..5 {
            world.step(1.0 / 60.0);
            exited |= world
                .drain_sensor_crossings()
                .iter()
                .any(|c| c.tag == 9 && !c.entered);
        }
        assert!(exited, "leaving the region should report an exit");
    }

    #[test]
    fn sensor_blocks_nothing() {
        let mut world = PhysicsWorld::new(G);
        ground(&mut world, 0.0);
        // A sensor spanning the resting spot: the box must fall through it and
        // settle on the floor as if it were not there.
        world.add_sensor(
            &ColliderShape::Cuboid {
                half_extents: [2.0, 2.0, 2.0],
            },
            [0.0, 2.0, 0.0],
            [0.0; 3],
            3,
            LayerMask::ALL,
        );
        let body = world.add_dynamic(
            &ColliderShape::Cuboid {
                half_extents: [0.5, 0.5, 0.5],
            },
            [0.0, 6.0, 0.0],
            [0.0; 3],
            DynamicParams {
                mass: 1.0,
                friction: 0.5,
                restitution: 0.0,
                gravity_scale: 1.0,
                linear_damping: 0.0,
            },
            LayerMask::ALL,
        );
        for _ in 0..240 {
            world.step(1.0 / 60.0);
        }
        let (pos, _) = world.body_pose(body);
        assert!(
            (pos[1] - 0.5).abs() < 0.1,
            "box should rest on the floor through the sensor, y = {}",
            pos[1]
        );
    }

    #[test]
    fn bodies_in_non_interacting_layers_pass_through_each_other() {
        // Ground on bit 0; two balls on bits 1 and 2 whose filters exclude
        // each other but keep the ground. Dropped down the same column, they
        // must overlap freely and both settle on the floor.
        let mut world = PhysicsWorld::new(G);
        world.add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [50.0, 5.0, 50.0],
            },
            [0.0, -5.0, 0.0],
            [0.0; 3],
            0.8,
            LayerMask {
                memberships: 0b001,
                filter: 0b111,
            },
        );
        let params = DynamicParams {
            mass: 1.0,
            friction: 0.5,
            restitution: 0.0,
            gravity_scale: 1.0,
            linear_damping: 0.0,
        };
        let lower = world.add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 0.5, 0.0],
            [0.0; 3],
            params,
            LayerMask {
                memberships: 0b010,
                filter: 0b001,
            },
        );
        let upper = world.add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 4.0, 0.0],
            [0.0; 3],
            params,
            LayerMask {
                memberships: 0b100,
                filter: 0b001,
            },
        );
        for _ in 0..300 {
            world.step(1.0 / 60.0);
        }
        let (lower_pos, _) = world.body_pose(lower);
        let (upper_pos, _) = world.body_pose(upper);
        assert!(
            (lower_pos[1] - 0.5).abs() < 0.1,
            "lower ball rests on the floor, y = {}",
            lower_pos[1]
        );
        assert!(
            (upper_pos[1] - 0.5).abs() < 0.1,
            "upper ball falls through the lower one onto the floor, y = {}",
            upper_pos[1]
        );
    }

    #[test]
    fn interacting_layers_still_stack() {
        // Same drop, but the balls' filters include each other: the upper one
        // stacks on the lower instead of passing through.
        let mut world = PhysicsWorld::new(G);
        ground(&mut world, 0.0);
        let params = DynamicParams {
            mass: 1.0,
            friction: 0.5,
            restitution: 0.0,
            gravity_scale: 1.0,
            linear_damping: 0.0,
        };
        world.add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 0.5, 0.0],
            [0.0; 3],
            params,
            LayerMask::ALL,
        );
        let upper = world.add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 4.0, 0.0],
            [0.0; 3],
            params,
            LayerMask::ALL,
        );
        for _ in 0..300 {
            world.step(1.0 / 60.0);
        }
        let (upper_pos, _) = world.body_pose(upper);
        assert!(
            upper_pos[1] > 1.2,
            "upper ball stacks on the lower one, y = {}",
            upper_pos[1]
        );
    }

    #[test]
    fn raycast_honors_the_layer_filter() {
        // Ground on bit 0, a box on bit 1 hanging over it. A ray filtered to
        // bit 0 passes through the box and reports the floor; an unfiltered
        // ray reports the box.
        let mut world = PhysicsWorld::new(G);
        world.add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [50.0, 5.0, 50.0],
            },
            [0.0, -5.0, 0.0],
            [0.0; 3],
            0.8,
            LayerMask {
                memberships: 0b01,
                filter: 0b11,
            },
        );
        world.add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [1.0, 0.25, 1.0],
            },
            [0.0, 1.0, 0.0],
            [0.0; 3],
            0.8,
            LayerMask {
                memberships: 0b10,
                filter: 0b11,
            },
        );
        world.step(1.0 / 60.0);
        let floor_only = LayerMask {
            memberships: u32::MAX,
            filter: 0b01,
        };
        let through = world
            .raycast([0.0, 3.0, 0.0], [0.0, -1.0, 0.0], 5.0, None, floor_only)
            .expect("filtered ray reaches the floor");
        assert!((through.point[1] - 0.0).abs() < 1e-3, "{:?}", through.point);
        let blocked = world
            .raycast([0.0, 3.0, 0.0], [0.0, -1.0, 0.0], 5.0, None, LayerMask::ALL)
            .expect("unfiltered ray hits the box");
        assert!(
            (blocked.point[1] - 1.25).abs() < 1e-3,
            "{:?}",
            blocked.point
        );
    }

    #[test]
    fn raycast_never_hits_sensors() {
        let mut world = PhysicsWorld::new(G);
        ground(&mut world, 0.0);
        world.add_sensor(
            &ColliderShape::Cuboid {
                half_extents: [1.0, 1.0, 1.0],
            },
            [0.0, 2.0, 0.0],
            [0.0; 3],
            5,
            LayerMask::ALL,
        );
        world.step(1.0 / 60.0);
        let hit = world
            .raycast(
                [0.0, 5.0, 0.0],
                [0.0, -1.0, 0.0],
                10.0,
                None,
                LayerMask::ALL,
            )
            .expect("ray reaches the floor");
        assert!(
            (hit.point[1] - 0.0).abs() < 1e-3,
            "sensor volume must be transparent to rays, hit {:?}",
            hit.point
        );
    }

    #[test]
    fn contact_hit_records_an_impact_but_not_resting_contact() {
        let mut world = PhysicsWorld::new(G);
        // Impulse threshold 1.0 at the 60 Hz tick: a 1 kg ball landing at
        // ~14 m/s is far above it, resting weight (20 N) far below.
        world.set_contact_min_impulse(1.0, 1.0 / 60.0);
        ground(&mut world, 0.0);
        let ball = world.add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 5.5, 0.0],
            [0.0; 3],
            DynamicParams {
                mass: 1.0,
                friction: 0.5,
                restitution: 0.0,
                gravity_scale: 1.0,
                linear_damping: 0.0,
            },
            LayerMask::ALL,
        );
        let mut impact: Option<ContactHit> = None;
        for _ in 0..120 {
            world.step(1.0 / 60.0);
            for hit in world.drain_contact_hits() {
                if impact.is_none() {
                    impact = Some(hit);
                }
            }
        }
        let impact = impact.expect("the landing records a contact hit");
        assert!(
            impact.a == ball || impact.b == ball,
            "the hit names the ball"
        );
        // Falling ~5 m under g = 20 arrives at ~14 m/s; the stop impulse for
        // 1 kg is ~14 mass-velocity units (solver spread makes it inexact).
        assert!(
            impact.impulse > 3.0 && impact.impulse < 50.0,
            "impulse {} out of the plausible landing range",
            impact.impulse
        );
        assert!(
            impact.normal[1].abs() > 0.9,
            "floor contact normal is vertical: {:?}",
            impact.normal
        );
        assert!(
            impact.point[1].abs() < 0.2,
            "contact point sits on the floor"
        );

        // Settled: a long rest window must stay silent.
        for _ in 0..300 {
            world.step(1.0 / 60.0);
            assert!(
                world.drain_contact_hits().is_empty(),
                "resting contact must not record hits"
            );
        }
    }

    #[test]
    fn normalize_axis_handles_degenerate_input() {
        assert_eq!(normalize_axis([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0]);
        let n = normalize_axis([3.0, 0.0, 4.0]);
        assert!((n[0] - 0.6).abs() < 1.0e-6 && (n[2] - 0.8).abs() < 1.0e-6);
        // Zero input falls back to +Y.
        assert_eq!(normalize_axis([0.0, 0.0, 0.0]), [0.0, 1.0, 0.0]);
    }
}

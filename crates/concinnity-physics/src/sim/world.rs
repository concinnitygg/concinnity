// concinnity-physics/src/sim/world.rs
//
// The simulation itself: body storage, and the order the stages run in.
//
// Bodies live in a fixed-capacity pool, so a full world declines a body rather
// than growing under a caller's feet, and a slot's generation makes a handle
// to a removed body read as absent. That is also what lets every other stage
// index its own arrays by slot: the pool's slot is the one identity the whole
// step agrees on.
//
// Nothing on the step path allocates, and neither does a query. Every buffer
// is reserved when the simulation is built and reused, which is what lets the
// step run inside a frame budget rather than at the allocator's convenience.

use alloc::vec::Vec;

use concinnity_memory::{Pool, PoolHandle};

use crate::{
    BodyHandle, CharacterMove, CharacterMoveInput, ColliderShape, ContactHit, DynamicParams,
    Fanout, JointSpec, LayerMask, RayHit, SensorCrossing,
};

use super::body::Body;
use super::broadphase::{Proxy, Role, SweepPrune};
use super::ccd::{self, Ccd};
use super::character::{self, CharacterCapsule, CharacterConfig};
use super::collide::heightfield::{Heightfield, Heightfields};
use super::config::SimConfig;
use super::contact::{ContactCache, Manifold, carry_impulses};
use super::impact::Impacts;
use super::island::Islands;
use super::joint::{Joint, JointFrame, JointSet};
#[cfg(test)]
use super::math::Mat3;
use super::math::{Quat, Vec3};
use super::narrow::{self, Narrow};
use super::query::{self, RayQuery};
#[cfg(test)]
use super::query::{ShapeCast, ShapeCastHit};
use super::scene::Scene;
use super::sensor::Sensors;
use super::solver::{self, Solver, SolverBody};

// Friction of a character capsule against what it is driven into. A character
// is moved by resolving a translation rather than by the solver, so this only
// governs what the capsule pushes.
const CHARACTER_FRICTION: f32 = 0.5;

// What a step is worth handing out, in units of one body's integration -- a
// contact costs about twenty of those and a joint about sixteen.
//
// The threshold is high on purpose. Gathering a pool's workers from a thread
// that is not one of them costs about as much as three hundred of these units,
// and the gathering is serial: a step that only just covers it comes out level
// at best. So a step hands its work out once it is worth an order of magnitude
// more than the gathering, and keeps it otherwise.
const CONTACT_COST: usize = 20;
const JOINT_COST: usize = 16;
const MIN_FANOUT_COST: usize = 4000;

/// A rigid-body simulation: bodies fall under gravity, collide, and come to
/// rest.
///
/// The capacity given at construction is the whole reservation. Adding past it
/// returns `None`; stepping never allocates.
///
/// # Examples
///
/// ```
/// use concinnity_physics::{ColliderShape, DynamicParams, LayerMask, Simulation};
///
/// let mut sim = Simulation::with_capacity(2);
/// sim.add_fixed(
///     &ColliderShape::Cuboid { half_extents: [10.0, 0.5, 10.0] },
///     [0.0, -0.5, 0.0],
///     [0.0; 3],
///     0.8,
///     LayerMask::ALL,
/// );
/// let ball = sim
///     .add_dynamic(
///         &ColliderShape::Ball { radius: 0.5 },
///         [0.0, 5.0, 0.0],
///         [0.0; 3],
///         DynamicParams {
///             mass: 1.0,
///             friction: 0.5,
///             restitution: 0.0,
///             gravity_scale: 1.0,
///             linear_damping: 0.0,
///         },
///         LayerMask::ALL,
///     )
///     .expect("room in the pool");
///
/// for _ in 0..180 {
///     sim.step(1.0 / 60.0);
/// }
///
/// let (position, _rotation) = sim.body_pose_quat(ball).expect("a live body");
/// assert!(
///     (position[1] - 0.5).abs() < 0.02,
///     "the ball rests on the floor, at y = {}",
///     position[1]
/// );
/// ```
pub struct Simulation {
    config: SimConfig,
    character: CharacterConfig,
    bodies: Pool<Body>,
    broadphase: SweepPrune,
    contacts: ContactCache,
    fields: Heightfields,
    narrow: Narrow,
    joints: JointSet,
    islands: Islands,
    solver: Solver,
    sensors: Sensors,
    impacts: Impacts,
    ccd: Ccd,
    /// Workers the per-worker scratch was reserved for. One until a caller
    /// says otherwise, which is what makes the serial path the default.
    workers: usize,
    /// Steps asked to split further than the reservation allows.
    worker_overflows: u32,
}

impl core::fmt::Debug for Simulation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Simulation")
            .field("bodies", &self.bodies.len())
            .field("capacity", &self.bodies.capacity())
            .field("joints", &self.joints.len())
            .finish()
    }
}

impl Simulation {
    /// Reserve room for `capacity` bodies, with the default tuning.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(SimConfig::default(), capacity)
    }

    /// Reserve room for `capacity` bodies, tuned by `config`.
    pub fn new(config: SimConfig, capacity: usize) -> Self {
        Simulation {
            config,
            character: CharacterConfig::default(),
            bodies: Pool::with_capacity(capacity),
            broadphase: SweepPrune::with_capacity(capacity),
            // Two contacts per body covers a stack, which is what the
            // reservation is sized for. A scene denser than that grows these
            // buffers once, early, and never again.
            contacts: ContactCache::with_capacity(capacity * 2),
            fields: Heightfields::new(),
            narrow: Narrow::new(),
            // A world with more joints than bodies is unusual enough to pay
            // for one reservation while it is being built; nothing added
            // afterwards is on the step path.
            joints: JointSet::with_capacity(capacity),
            islands: Islands::with_capacity(capacity),
            solver: Solver::with_capacity(capacity),
            // A crossing is a boundary rather than a state, so a world
            // records far fewer of them than it has bodies. A hit is one per
            // contact pair, so the queue is reserved the way the contact list
            // above is.
            sensors: Sensors::with_capacity(capacity),
            impacts: Impacts::with_capacity(capacity * 2),
            ccd: Ccd::with_capacity(capacity),
            workers: 1,
            worker_overflows: 0,
        }
    }

    /// Reserve the scratch a step needs to split into `workers` pieces, and
    /// return how many it will actually use.
    ///
    /// Call this once while the world is built, with the worker count of the
    /// fan-out that will be stepping it. A simulation nobody calls this on
    /// reserves nothing and steps on the calling thread, which is what lets a
    /// host with no threads use the same simulation unchanged.
    ///
    /// Splitting never changes what a step produces, so the reserved count is
    /// a ceiling rather than a promise: a step handed a wider fan-out uses
    /// this many pieces of it and leaves the rest of it idle.
    pub fn reserve_workers(&mut self, workers: usize) -> usize {
        let capacity = self.bodies.capacity();
        self.workers = workers.clamp(1, solver::MAX_WORKERS);
        self.broadphase.reserve_workers(self.workers, capacity);
        self.narrow.reserve_workers(self.workers, capacity);
        self.workers
    }

    /// Workers the step splits into at most: what
    /// [`Simulation::reserve_workers`] settled on.
    pub fn workers(&self) -> usize {
        self.workers
    }

    #[cfg(test)]
    /// Steps that were handed a wider fan-out than the reservation covers,
    /// since the count was last cleared. Each one ran on the reserved number
    /// of workers instead, which changes nothing about the result.
    pub(crate) fn worker_overflows(&self) -> u32 {
        self.worker_overflows
    }

    #[cfg(test)]
    /// Forget the count above.
    pub(crate) fn clear_worker_overflows(&mut self) {
        self.worker_overflows = 0;
    }

    /// Tune the character controller. `grounded` is true for a gravity-bound
    /// character, which climbs steps and stays attached to the ground, and
    /// false for a free-flying camera, which does neither. A `max_slope_deg`
    /// of `0` disables the climb limit.
    pub fn configure_character(&mut self, max_slope_deg: f32, step_height: f32, grounded: bool) {
        self.character = CharacterConfig::new(max_slope_deg, step_height, grounded);
    }

    /// Build the capsule a character move is resolved against: a cylinder of
    /// `2 * half_height` capped by hemispheres of `radius`.
    ///
    /// A caller holds one per character across the fixed ticks rather than
    /// building one per move.
    pub fn character_shape(half_height: f32, radius: f32) -> CharacterCapsule {
        CharacterCapsule::new(half_height, radius)
    }

    /// The tuning this simulation steps with.
    pub fn config(&self) -> &SimConfig {
        &self.config
    }

    #[cfg(test)]
    /// Re-tune the simulation. Takes effect on the next step.
    pub(crate) fn set_config(&mut self, config: SimConfig) {
        self.config = config;
    }

    /// Bodies the pool can hold.
    pub fn capacity(&self) -> usize {
        self.bodies.capacity()
    }

    /// Bodies currently in the simulation.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Colliders currently in the simulation. A body carries exactly one
    /// shape, so this is the body count until compound shapes exist.
    pub fn collider_count(&self) -> usize {
        self.bodies.len()
    }

    /// Joints currently constraining bodies.
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    #[cfg(test)]
    /// Sensor pairs currently overlapping: what the regions are holding,
    /// rather than what crossed a boundary to get there.
    pub(crate) fn sensor_overlap_count(&self) -> usize {
        self.sensors.overlap_count()
    }

    /// Contact points the last step solved.
    #[cfg(test)]
    pub(crate) fn contact_count(&self) -> usize {
        self.contacts
            .manifolds()
            .iter()
            .map(|m| m.count as usize)
            .sum()
    }

    /// Bytes reserved for bodies, bounds, contacts, the solver's arrays, and
    /// the per-worker buffers [`Simulation::reserve_workers`] set aside: what
    /// the simulation costs whatever its occupancy.
    pub fn reserved_bytes(&self) -> u64 {
        self.bodies.reserved_bytes()
            + self.broadphase.reserved_bytes()
            + self.contacts.reserved_bytes()
            + self.fields.reserved_bytes()
            + self.narrow.reserved_bytes()
            + self.joints.reserved_bytes()
            + self.islands.reserved_bytes()
            + self.solver.reserved_bytes()
            + self.sensors.reserved_bytes()
            + self.impacts.reserved_bytes()
            + self.ccd.reserved_bytes()
    }

    /// Add an immovable body. `None` when the pool is full.
    pub fn add_fixed(
        &mut self,
        shape: &ColliderShape,
        pos: [f32; 3],
        euler_deg: [f32; 3],
        friction: f32,
        mask: LayerMask,
    ) -> Option<BodyHandle> {
        self.add(Body::fixed(
            *shape,
            Vec3::from_array(pos),
            Quat::from_euler_deg(euler_deg),
            friction,
            mask,
        ))
    }

    /// Add a body driven to a position rather than by forces: infinite mass,
    /// untouched by gravity or impulses, but it pushes what it is driven
    /// into. Move it with [`Simulation::set_kinematic_translation`]. `None`
    /// when the pool is full.
    pub fn add_kinematic(
        &mut self,
        shape: &ColliderShape,
        pos: [f32; 3],
        euler_deg: [f32; 3],
        friction: f32,
        mask: LayerMask,
    ) -> Option<BodyHandle> {
        self.add(Body::kinematic(
            *shape,
            Vec3::from_array(pos),
            Quat::from_euler_deg(euler_deg),
            friction,
            mask,
        ))
    }

    /// Add a position-driven character capsule centred on `center`: a
    /// cylinder of `2 * half_height` capped by hemispheres of `radius`.
    ///
    /// Gravity does not move it and the solver does not push it. Resolve a
    /// desired move with [`Simulation::move_character`] and apply the answer
    /// with [`Simulation::set_kinematic_translation`]. `None` when the pool is
    /// full.
    pub fn add_character(
        &mut self,
        half_height: f32,
        radius: f32,
        center: [f32; 3],
        mask: LayerMask,
    ) -> Option<BodyHandle> {
        self.add_kinematic(
            &ColliderShape::Capsule {
                half_height,
                radius,
            },
            center,
            [0.0; 3],
            CHARACTER_FRICTION,
            mask,
        )
    }

    /// Add a region that records what overlaps it and resists nothing.
    ///
    /// It never collides, never blocks a query, and never moves. What crosses
    /// its boundary is reported as a [`SensorCrossing`] carrying `tag`,
    /// collected with [`Simulation::drain_sensor_crossings_into`]. Freely
    /// simulated and position-driven bodies cross it; immovable geometry does
    /// not, and two overlapping regions record a crossing each.
    ///
    /// `None` when the pool is full.
    ///
    /// # Examples
    ///
    /// ```
    /// use concinnity_physics::{ColliderShape, DynamicParams, LayerMask, Simulation};
    ///
    /// let mut sim = Simulation::with_capacity(2);
    /// sim.add_sensor(
    ///     &ColliderShape::Cuboid { half_extents: [1.0, 1.0, 1.0] },
    ///     [0.0, 2.0, 0.0],
    ///     [0.0; 3],
    ///     7,
    ///     LayerMask::ALL,
    /// )
    /// .expect("room in the pool");
    ///
    /// // Nothing holds the ball up, so it falls through the region.
    /// sim.add_dynamic(
    ///     &ColliderShape::Ball { radius: 0.25 },
    ///     [0.0, 6.0, 0.0],
    ///     [0.0; 3],
    ///     DynamicParams {
    ///         mass: 1.0,
    ///         friction: 0.5,
    ///         restitution: 0.0,
    ///         gravity_scale: 1.0,
    ///         linear_damping: 0.0,
    ///     },
    ///     LayerMask::ALL,
    /// )
    /// .expect("room in the pool");
    ///
    /// let mut crossings = Vec::new();
    /// let (mut entered, mut left) = (false, false);
    /// for _ in 0..300 {
    ///     sim.step(1.0 / 60.0);
    ///     sim.drain_sensor_crossings_into(&mut crossings);
    ///     for crossing in &crossings {
    ///         assert_eq!(crossing.tag, 7);
    ///         if crossing.entered { entered = true } else { left = true }
    ///     }
    /// }
    /// assert!(entered && left, "the ball went in and came out again");
    /// ```
    pub fn add_sensor(
        &mut self,
        shape: &ColliderShape,
        pos: [f32; 3],
        euler_deg: [f32; 3],
        tag: u64,
        mask: LayerMask,
    ) -> Option<BodyHandle> {
        self.add(Body::sensor(
            *shape,
            Vec3::from_array(pos),
            Quat::from_euler_deg(euler_deg),
            tag,
            mask,
        ))
    }

    /// Move the boundary crossings recorded since the last drain into `out`,
    /// oldest first. `out` is cleared first, and both it and the queue keep
    /// their capacity, so a per-tick drain never reallocates.
    pub fn drain_sensor_crossings_into(&mut self, out: &mut Vec<SensorCrossing>) {
        self.sensors.drain_into(out);
    }

    #[cfg(test)]
    /// Crossings and overlaps the reservation had no room for, since the
    /// count was last cleared.
    ///
    /// Both are reserved against the body capacity, which covers a caller
    /// draining every step and a world whose regions hold fewer bodies than
    /// it has. A non-zero count means something crossed a region and no
    /// caller was told about it.
    pub(crate) fn sensor_overflows(&self) -> u32 {
        self.sensors.overflows()
    }

    #[cfg(test)]
    /// Reset the count of declined crossings.
    pub(crate) fn clear_sensor_overflows(&mut self) {
        self.sensors.clear_overflows();
    }

    /// Set the smallest contact impulse worth reporting as a
    /// [`ContactHit`], measured at a step of `tick_dt`.
    ///
    /// The simulation gates on the force that impulse stands for, so a pair
    /// leaning on another at rest stays silent while the same pair colliding
    /// does not. It applies to every body, whenever it is called.
    pub fn set_contact_min_impulse(&mut self, min_impulse: f32, tick_dt: f32) {
        self.impacts.set_min_impulse(min_impulse, tick_dt);
    }

    /// Move the contact hits recorded since the last drain into `out`, oldest
    /// first. `out` is cleared first, and both it and the queue keep their
    /// capacity.
    ///
    /// Only pairs with a freely simulated body on at least one side, carrying
    /// more than the force [`Simulation::set_contact_min_impulse`] set, appear.
    pub fn drain_contact_hits_into(&mut self, out: &mut Vec<ContactHit>) {
        self.impacts.drain_into(out);
    }

    #[cfg(test)]
    /// Contact hits the reservation had no room for, since the count was last
    /// cleared.
    ///
    /// The queue is reserved for one hit per contact pair, which covers a
    /// caller draining every step. A non-zero count means a collision
    /// happened that no caller was told about.
    pub(crate) fn contact_hit_overflows(&self) -> u32 {
        self.impacts.overflows()
    }

    /// Add a static height grid: terrain, addressed like any other body.
    ///
    /// `heights` is a `rows * cols` row-major grid of world-space `y` values,
    /// with rows running along `z` and columns along `x`. `scale` is the whole
    /// extent `[width, height_multiplier, depth]`, and the grid is centred on
    /// `pos`.
    ///
    /// `None` when the pool is full or the grid names no surface: fewer than
    /// two rows or columns, the wrong number of heights, or no footprint.
    ///
    /// The body is immovable and never rotates. Contacts against it are
    /// answered along the surface's own face normals, so a shape crossing the
    /// boundary between two cells is not caught by the edge they share.
    ///
    /// # Examples
    ///
    /// ```
    /// use concinnity_physics::{ColliderShape, DynamicParams, LayerMask, Simulation};
    ///
    /// let mut sim = Simulation::with_capacity(2);
    /// // A flat five-by-five grid twenty units square, its surface at y = 0.
    /// sim.add_heightfield(
    ///     5,
    ///     5,
    ///     vec![0.0; 25],
    ///     [20.0, 1.0, 20.0],
    ///     [0.0; 3],
    ///     LayerMask::ALL,
    /// )
    /// .expect("room in the pool");
    ///
    /// let ball = sim
    ///     .add_dynamic(
    ///         &ColliderShape::Ball { radius: 0.5 },
    ///         [1.0, 5.0, -2.0],
    ///         [0.0; 3],
    ///         DynamicParams {
    ///             mass: 1.0,
    ///             friction: 0.5,
    ///             restitution: 0.0,
    ///             gravity_scale: 1.0,
    ///             linear_damping: 0.0,
    ///         },
    ///         LayerMask::ALL,
    ///     )
    ///     .expect("room in the pool");
    ///
    /// for _ in 0..240 {
    ///     sim.step(1.0 / 60.0);
    /// }
    ///
    /// let (position, _) = sim.body_pose_quat(ball).expect("a live body");
    /// assert!(
    ///     (position[1] - 0.5).abs() < 0.02,
    ///     "the ball rests on the terrain, at y = {}",
    ///     position[1]
    /// );
    /// ```
    pub fn add_heightfield(
        &mut self,
        rows: usize,
        cols: usize,
        heights: Vec<f32>,
        scale: [f32; 3],
        pos: [f32; 3],
        mask: LayerMask,
    ) -> Option<BodyHandle> {
        if self.bodies.len() >= self.bodies.capacity() {
            return None;
        }
        let origin = Vec3::from_array(pos);
        let field = Heightfield::new(rows, cols, heights, Vec3::from_array(scale), origin)?;
        let bounds = field.bounds();
        let index = self.fields.push(field);
        // Terrain is fully rough: a slope holds whatever the body resting on
        // it brings, rather than the surface capping it.
        self.add(Body::terrain(index, bounds, origin, 1.0, mask))
    }

    #[cfg(test)]
    /// Queries that gave up with terrain still to look at, since the count was
    /// last cleared.
    ///
    /// A query walks a bounded number of the grid's triangles and stops rather
    /// than growing a buffer. A non-zero count means some question was asked
    /// of more surface than that, and the answer covered only part of it.
    pub(crate) fn heightfield_overflows(&self) -> u32 {
        self.fields.overflows()
    }

    #[cfg(test)]
    /// Reset the count of terrain queries that gave up.
    pub(crate) fn clear_heightfield_overflows(&mut self) {
        self.fields.clear_overflows();
    }

    #[cfg(test)]
    /// Fast bodies and swept region crossings the continuous-collision
    /// reservation had no room for, since the count was last cleared.
    ///
    /// The reservation holds one entry per body, which covers every body in
    /// the world moving fast at once. A non-zero count means a body took a
    /// step long enough to pass through something and was not swept.
    pub(crate) fn ccd_overflows(&self) -> u32 {
        self.ccd.overflows()
    }

    #[cfg(test)]
    /// Bodies the last step swept along their own path because they moved
    /// too far for its contact test to have seen what they crossed.
    ///
    /// Zero for a world at ordinary speeds, which is what the gate is for:
    /// a non-zero count is the number of bodies that paid for the expensive
    /// path on the last step.
    pub(crate) fn swept_body_count(&self) -> usize {
        self.ccd.mover_count()
    }

    /// Add a freely simulated body. `None` when the pool is full.
    pub fn add_dynamic(
        &mut self,
        shape: &ColliderShape,
        pos: [f32; 3],
        euler_deg: [f32; 3],
        params: DynamicParams,
        mask: LayerMask,
    ) -> Option<BodyHandle> {
        self.add(Body::dynamic(
            *shape,
            Vec3::from_array(pos),
            Quat::from_euler_deg(euler_deg),
            params,
            mask,
        ))
    }

    /// Constrain two bodies to each other. Anchors are in each body's own
    /// frame, and the joint holds the relative pose the bodies are in when it
    /// is made.
    ///
    /// Returns whether the joint was made. It needs two different live bodies;
    /// past that, degenerate input is repaired rather than refused, so a
    /// zero-length axis becomes `+Y` and limits given the wrong way round are
    /// read low to high.
    ///
    /// A joint is removed by removing either of the bodies it holds.
    ///
    /// # Examples
    ///
    /// ```
    /// use concinnity_physics::{
    ///     ColliderShape, DynamicParams, JointSpec, LayerMask, Simulation,
    /// };
    ///
    /// let mut sim = Simulation::with_capacity(2);
    /// let post = sim
    ///     .add_fixed(
    ///         &ColliderShape::Ball { radius: 0.1 },
    ///         [0.0, 4.0, 0.0],
    ///         [0.0; 3],
    ///         0.5,
    ///         LayerMask::ALL,
    ///     )
    ///     .expect("room in the pool");
    /// let bob = sim
    ///     .add_dynamic(
    ///         &ColliderShape::Ball { radius: 0.2 },
    ///         [1.0, 4.0, 0.0],
    ///         [0.0; 3],
    ///         DynamicParams {
    ///             mass: 1.0,
    ///             friction: 0.5,
    ///             restitution: 0.0,
    ///             gravity_scale: 1.0,
    ///             linear_damping: 0.0,
    ///         },
    ///         LayerMask::ALL,
    ///     )
    ///     .expect("room in the pool");
    ///
    /// assert!(sim.add_joint(post, bob, [0.0; 3], [-1.0, 0.0, 0.0], JointSpec::Spherical));
    ///
    /// for _ in 0..120 {
    ///     sim.step(1.0 / 60.0);
    /// }
    ///
    /// // The bob swings, but it stays one unit from the post it hangs off.
    /// let (position, _) = sim.body_pose_quat(bob).expect("a live body");
    /// let reach = ((position[0] - 0.0).powi(2)
    ///     + (position[1] - 4.0).powi(2)
    ///     + (position[2] - 0.0).powi(2))
    /// .sqrt();
    /// assert!((reach - 1.0).abs() < 0.01, "hanging {reach} from the post");
    /// ```
    pub fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        spec: JointSpec,
    ) -> bool {
        let (slot_a, slot_b) = (body_a.index(), body_b.index());
        if slot_a == slot_b {
            return false;
        }
        let (anchor_a, anchor_b) = (Vec3::from_array(anchor_a), Vec3::from_array(anchor_b));
        if !anchor_a.is_finite() || !anchor_b.is_finite() {
            return false;
        }
        let (Some(a), Some(b)) = (
            self.bodies.get(pool_handle(body_a)),
            self.bodies.get(pool_handle(body_b)),
        ) else {
            return false;
        };
        let frame = JointFrame::new(spec, a.orientation, b.orientation);
        self.joints.push(Joint {
            a: slot_a,
            b: slot_b,
            anchor_a,
            anchor_b,
            frame,
            impulses: Default::default(),
        });
        // A joint arriving on a settled body has to be able to disturb it.
        for slot in [slot_a, slot_b] {
            if let Some(body) = self.bodies.get_at_mut(slot as usize) {
                body.wake();
            }
        }
        true
    }

    /// Send a position-driven body to `pos` over the next step. It arrives
    /// exactly there, pushing whatever it meets on the way. Returns whether
    /// the handle named a live position-driven body.
    ///
    /// The target is consumed by the step, so a body that is to keep moving
    /// is given a fresh one each tick and a body that is left alone stops.
    pub fn set_kinematic_translation(&mut self, handle: BodyHandle, pos: [f32; 3]) -> bool {
        let Some(body) = self.bodies.get_mut(pool_handle(handle)) else {
            return false;
        };
        if !body.is_kinematic() {
            return false;
        }
        body.kinematic_target = Some(Vec3::from_array(pos));
        true
    }

    /// Switch a body to position-driven control, keeping its handle and the
    /// mass it was authored with. Returns whether the handle named a live
    /// body.
    pub fn make_kinematic(&mut self, handle: BodyHandle) -> bool {
        self.reclassify(handle, |body| body.make_kinematic())
    }

    /// Hand a body back to the solver with a launch velocity, restoring the
    /// mass it was authored with. Returns whether the handle named a live
    /// body.
    pub fn make_dynamic(&mut self, handle: BodyHandle, linear_velocity: [f32; 3]) -> bool {
        let velocity = Vec3::from_array(linear_velocity);
        self.reclassify(handle, move |body| body.make_dynamic(velocity))
    }

    /// Whether a body is driven by position rather than by forces.
    #[cfg(test)]
    pub(crate) fn is_kinematic(&self, handle: BodyHandle) -> Option<bool> {
        Some(self.bodies.get(pool_handle(handle))?.is_kinematic())
    }

    /// What a query reads: the bodies, the sweep order, and the height grids.
    fn scene(&self) -> Scene<'_> {
        Scene {
            bodies: &self.bodies,
            broadphase: &self.broadphase,
            fields: &self.fields,
        }
    }

    /// Cast a ray, returning the nearest hit within `max_dist`.
    ///
    /// `dir` need not be unit length; a zero direction misses. `exclude`
    /// leaves one body out, and `mask` restricts the hit set to layers the
    /// query interacts with. A ray that begins inside a body hits it at zero
    /// distance with the normal turned back along the ray.
    pub fn raycast(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        max_dist: f32,
        exclude: Option<BodyHandle>,
        mask: LayerMask,
    ) -> Option<RayHit> {
        query::raycast(
            self.scene(),
            &RayQuery {
                origin,
                dir,
                max_dist,
                exclude,
                mask,
            },
        )
    }

    /// Sweep a shape through the world, returning the nearest body it runs
    /// into and how far along the cast's motion that happens.
    #[cfg(test)]
    pub(crate) fn shape_cast(&self, cast: &ShapeCast) -> Option<ShapeCastHit> {
        query::shape_cast(self.scene(), cast)
    }

    /// Resolve a desired character move against the world without moving
    /// anything in it, returning the translation to apply and whether the
    /// capsule ends up on the ground.
    ///
    /// The capsule sweeps along the desired translation and slides along
    /// whatever it meets rather than stopping dead, up to a bounded number of
    /// deflections. `input.exclude` is the mover's own body, left out of the
    /// query so it does not collide with itself; other characters' capsules
    /// stay solid to it. What counts as ground, how high an obstacle is
    /// climbed, and whether the mover is gravity-bound at all come from
    /// [`Simulation::configure_character`].
    ///
    /// Apply the result with [`Simulation::set_kinematic_translation`].
    ///
    /// # Examples
    ///
    /// ```
    /// use concinnity_physics::{CharacterMoveInput, ColliderShape, LayerMask, Simulation};
    ///
    /// let mut sim = Simulation::with_capacity(3);
    /// sim.add_fixed(
    ///     &ColliderShape::Cuboid { half_extents: [10.0, 0.5, 10.0] },
    ///     [0.0, -0.5, 0.0],
    ///     [0.0; 3],
    ///     0.8,
    ///     LayerMask::ALL,
    /// );
    /// // A wall whose near face is at z = 1.
    /// sim.add_fixed(
    ///     &ColliderShape::Cuboid { half_extents: [4.0, 2.0, 0.5] },
    ///     [0.0, 2.0, 1.5],
    ///     [0.0; 3],
    ///     0.8,
    ///     LayerMask::ALL,
    /// );
    ///
    /// // The capsule stands on the floor: half height plus radius above it.
    /// let center = [0.0, 0.9, 0.0];
    /// let capsule = sim
    ///     .add_kinematic(
    ///         &ColliderShape::Capsule { half_height: 0.6, radius: 0.3 },
    ///         center,
    ///         [0.0; 3],
    ///         0.8,
    ///         LayerMask::ALL,
    ///     )
    ///     .expect("room in the pool");
    ///
    /// let shape = Simulation::character_shape(0.6, 0.3);
    /// let moved = sim.move_character(
    ///     &shape,
    ///     &CharacterMoveInput {
    ///         center,
    ///         desired: [0.0, -0.01, 2.0],
    ///         dt: 1.0 / 60.0,
    ///         exclude: capsule,
    ///         mask: LayerMask::ALL,
    ///     },
    /// );
    ///
    /// // Two units of walk, and the wall stops it a radius short of its face.
    /// assert!(
    ///     (moved.translation[2] - 0.7).abs() < 0.01,
    ///     "walked {}",
    ///     moved.translation[2]
    /// );
    /// assert!(moved.grounded, "the floor is still underfoot");
    /// ```
    pub fn move_character(
        &self,
        shape: &CharacterCapsule,
        input: &CharacterMoveInput,
    ) -> CharacterMove {
        character::resolve(self.scene(), &self.character, shape, input)
    }

    /// Remove a body, along with every joint it was in. Returns whether the
    /// handle named a live one.
    ///
    /// Purging the joints is part of the removal rather than something a
    /// caller does afterwards: a joint naming a slot whose body has gone would
    /// constrain whatever occupied that slot next.
    pub fn remove_body(&mut self, handle: BodyHandle) -> bool {
        let slot = handle.index();
        if self.bodies.remove(pool_handle(handle)).is_none() {
            return false;
        }
        self.broadphase.remove(slot);
        // Whatever this body was touching or holding has to be re-examined
        // without it, which has to happen before the joints are dropped.
        self.wake_neighbours(slot);
        self.joints.remove_incident(slot);
        true
    }

    #[cfg(test)]
    /// A body's world-space position and Euler-degree rotation.
    pub(crate) fn body_pose(&self, handle: BodyHandle) -> Option<([f32; 3], [f32; 3])> {
        let body = self.bodies.get(pool_handle(handle))?;
        Some((body.position.to_array(), body.orientation.to_euler_deg()))
    }

    /// A body's world-space position and `[x, y, z, w]` rotation quaternion.
    pub fn body_pose_quat(&self, handle: BodyHandle) -> Option<([f32; 3], [f32; 4])> {
        let body = self.bodies.get(pool_handle(handle))?;
        Some((body.position.to_array(), body.orientation.to_xyzw()))
    }

    #[cfg(test)]
    /// A body's linear velocity in world units per second.
    pub(crate) fn linear_velocity(&self, handle: BodyHandle) -> Option<[f32; 3]> {
        Some(
            self.bodies
                .get(pool_handle(handle))?
                .linear_velocity
                .to_array(),
        )
    }

    #[cfg(test)]
    /// A body's angular velocity in radians per second.
    pub(crate) fn angular_velocity(&self, handle: BodyHandle) -> Option<[f32; 3]> {
        Some(
            self.bodies
                .get(pool_handle(handle))?
                .angular_velocity
                .to_array(),
        )
    }

    /// A body's mass in kilograms. Immovable bodies report `0`.
    pub fn mass(&self, handle: BodyHandle) -> Option<f32> {
        Some(self.bodies.get(pool_handle(handle))?.mass)
    }

    #[cfg(test)]
    /// Whether a settled body has stopped being simulated.
    pub(crate) fn is_sleeping(&self, handle: BodyHandle) -> Option<bool> {
        Some(self.bodies.get(pool_handle(handle))?.sleeping)
    }

    #[cfg(test)]
    /// Set a body's linear velocity, waking it.
    pub(crate) fn set_linear_velocity(&mut self, handle: BodyHandle, velocity: [f32; 3]) {
        if let Some(body) = self.bodies.get_mut(pool_handle(handle)) {
            body.linear_velocity = Vec3::from_array(velocity);
            body.wake();
        }
    }

    #[cfg(test)]
    /// Set a body's angular velocity in radians per second, waking it.
    pub(crate) fn set_angular_velocity(&mut self, handle: BodyHandle, velocity: [f32; 3]) {
        if let Some(body) = self.bodies.get_mut(pool_handle(handle)) {
            body.angular_velocity = Vec3::from_array(velocity);
            body.wake();
        }
    }

    #[cfg(test)]
    /// Apply an impulse through a body's centre of mass, waking it.
    pub(crate) fn apply_impulse(&mut self, handle: BodyHandle, impulse: [f32; 3]) {
        if let Some(body) = self.bodies.get_mut(pool_handle(handle)) {
            body.linear_velocity += Vec3::from_array(impulse) * body.inv_mass;
            body.wake();
        }
    }

    #[cfg(test)]
    /// Kinetic plus gravitational potential energy over every dynamic body,
    /// with potential measured from `y = 0`.
    ///
    /// A settled world's total must not climb: a solver that injects energy
    /// says so here before it says so as a stack that will not stand.
    pub(crate) fn total_energy(&self) -> f32 {
        let mut total = 0.0;
        for (_, body) in self.bodies.iter() {
            if !body.is_dynamic() {
                continue;
            }
            let momentum = Mat3::diagonal_conjugated(body.orientation, body.inertia_local)
                .mul_vec3(body.angular_velocity);
            total += 0.5 * body.mass * body.linear_velocity.length_squared()
                + 0.5 * body.angular_velocity.dot(momentum)
                + body.mass * self.config.gravity * body.gravity_scale * body.position.y;
        }
        total
    }

    /// Advance the simulation by `dt` seconds on the calling thread.
    pub fn step(&mut self, dt: f32) {
        self.step_with(dt, &crate::Inline);
    }

    /// Advance the simulation by `dt` seconds, offering the step's independent
    /// work to `fanout`.
    ///
    /// A step splits the same way whatever it is handed, so a world stepped
    /// across a thread pool and the same world stepped on one thread land in
    /// exactly the same place. Nothing is reserved for a fan-out that
    /// [`Simulation::reserve_workers`] was not told about, so a wider one is
    /// used only as far as the reservation goes.
    ///
    /// # Examples
    ///
    /// ```
    /// use concinnity_physics::{ColliderShape, DynamicParams, Inline, LayerMask, Simulation};
    ///
    /// let mut sim = Simulation::with_capacity(1);
    /// let ball = sim
    ///     .add_dynamic(
    ///         &ColliderShape::Ball { radius: 0.5 },
    ///         [0.0, 10.0, 0.0],
    ///         [0.0; 3],
    ///         DynamicParams {
    ///             mass: 1.0,
    ///             friction: 0.4,
    ///             restitution: 0.2,
    ///             gravity_scale: 1.0,
    ///             linear_damping: 0.0,
    ///         },
    ///         LayerMask::ALL,
    ///     )
    ///     .expect("room for one body");
    /// sim.step_with(1.0 / 60.0, &Inline);
    /// assert!(sim.body_pose_quat(ball).expect("a live body").0[1] < 10.0, "it fell");
    /// ```
    pub fn step_with(&mut self, dt: f32, fanout: &impl Fanout) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let asked = fanout.workers().max(1);
        if asked > self.workers {
            self.worker_overflows = self.worker_overflows.saturating_add(1);
        }
        let workers = asked.min(self.workers);
        self.drive_kinematics(dt);
        let awake = self.refresh_bounds();
        // Reaching a fan-out's workers costs the same whatever is handed to
        // them, so a step with little to do is worth more on the thread it is
        // already on.
        if workers > 1 && self.step_cost(awake) >= MIN_FANOUT_COST {
            fanout.scope(|| self.advance(dt, fanout, workers));
        } else {
            self.advance(dt, &crate::Inline, 1);
        }
    }

    /// Roughly what the step ahead will cost, from what the world was holding
    /// last step.
    ///
    /// Only a moving body's contacts are work: a settled world keeps a full
    /// pair list and solves none of it, so the pairs are counted against what
    /// is still awake rather than taken at face value.
    fn step_cost(&self, awake: usize) -> usize {
        if awake == 0 {
            return 0;
        }
        let contacts = self.broadphase.pair_count().min(awake * 4);
        let joints = self.joints.len().min(awake * 2);
        awake + contacts * CONTACT_COST + joints * JOINT_COST
    }

    /// Everything a step does once the bounds are current and the workers, if
    /// any, have been gathered.
    fn advance(&mut self, dt: f32, fanout: &impl Fanout, workers: usize) {
        let Simulation {
            config,
            bodies,
            broadphase,
            contacts,
            fields,
            narrow,
            joints,
            islands,
            solver,
            sensors,
            impacts,
            ccd,
            ..
        } = self;
        let sweeping = ccd::enabled(config);

        let pairs = broadphase.sweep(fanout, workers);
        sensors.resolve(bodies, pairs.sensors);
        let (current, previous) = contacts.begin();
        narrow.build(
            narrow::Work {
                bodies,
                fields,
                pairs: pairs.contacts,
                previous,
                out: current,
                margin: config.speculative_margin,
            },
            fanout,
            workers,
        );
        carry_impulses(previous, current);
        wake_driven_contacts(bodies, current);

        solver.begin();
        gather(bodies, solver, current, joints.as_slice());
        solver.run(
            solver::Work {
                manifolds: current,
                joints: joints.as_mut_slice(),
                islands,
                config,
                dt,
            },
            fanout,
            workers,
        );
        // The solver is asked what it delivered, so a pair whose manifold was
        // carried forward untouched is not read as a fresh collision.
        impacts.collect(bodies, current, solver.loads(), dt);
        ccd.begin();
        for (handle, body) in bodies.iter_mut() {
            if !body.is_simulated() {
                continue;
            }
            let slot = handle.index() as u32;
            let solved = solver.body(slot);
            let began_at = body.position;
            body.linear_velocity = solved.linear_velocity;
            body.angular_velocity = solved.angular_velocity;
            body.position = solved.position;
            body.orientation = solved.rotation;
            // A position-driven body lands exactly where it was sent, not
            // wherever the substeps' arithmetic left it.
            if let Some(target) = body.kinematic_target.take() {
                body.position = target;
            }
            if sweeping {
                ccd.observe(slot, body, began_at, config.ccd_motion_ratio);
            }
        }

        // After the write-back, so a mover is swept along the path the step
        // actually took it on, and before sleep, so what it ran into is awake
        // when the islands are decided.
        if sweeping {
            let scene = Scene {
                bodies,
                broadphase,
                fields,
            };
            ccd.resolve(scene, config, dt);
            ccd.report_crossings(bodies, sensors);
            ccd.apply(bodies);
        }

        if !solver.is_idle() {
            update_sleep(config, bodies, islands, current, joints.as_slice(), dt);
        }
    }

    /// Change a body's kind and put the broad phase and its neighbours back
    /// in step with the change.
    fn reclassify(&mut self, handle: BodyHandle, change: impl FnOnce(&mut Body) -> bool) -> bool {
        let Some(body) = self.bodies.get_mut(pool_handle(handle)) else {
            return false;
        };
        if !change(body) {
            return true;
        }
        // Whether contact moves this body has changed, so the sweep's pair
        // filter and whatever was leaning on it both have to hear about it.
        let proxy = proxy_for(body);
        let slot = handle.index();
        self.broadphase.set_proxy(slot, proxy);
        self.wake_neighbours(slot);
        true
    }

    /// Give every driven body the velocity that carries it to its target over
    /// this step, so the solver sees the motion as motion.
    fn drive_kinematics(&mut self, dt: f32) {
        for (_, body) in self.bodies.iter_mut() {
            if body.is_kinematic() {
                body.drive_to_target(dt);
            }
        }
    }

    fn add(&mut self, mut body: Body) -> Option<BodyHandle> {
        body.refresh_bounds(self.config.bounds_margin);
        let proxy = proxy_for(&body);
        let handle = self.bodies.insert(body)?;
        let slot = handle.index() as u32;
        self.broadphase.insert(slot);
        self.broadphase.set_proxy(slot, proxy);
        // A body arriving inside a settled stack has to be able to disturb it.
        self.wake_neighbours(slot);
        Some(body_handle(handle))
    }

    /// Wake whatever the given slot was in contact with or jointed to, so a
    /// change at one body reaches the island it belonged to.
    fn wake_neighbours(&mut self, slot: u32) {
        let Simulation {
            bodies,
            contacts,
            joints,
            ..
        } = self;
        let mut wake = |other: u32| {
            if let Some(body) = bodies.get_at_mut(other as usize) {
                body.wake();
            }
        };
        for manifold in contacts.manifolds_mut() {
            let other = if manifold.a == slot {
                manifold.b
            } else if manifold.b == slot {
                manifold.a
            } else {
                continue;
            };
            wake(other);
        }
        for joint in joints.as_slice() {
            if let Some(other) = joint.other(slot) {
                wake(other);
            }
        }
    }

    /// Re-fatten the bounds of the bodies that moved, and tell the broad phase
    /// about the ones whose bounds actually changed.
    fn refresh_bounds(&mut self) -> usize {
        let margin = self.config.bounds_margin;
        let Simulation {
            bodies, broadphase, ..
        } = self;
        let mut awake = 0;
        for (handle, body) in bodies.iter_mut() {
            if !body.is_simulated() {
                continue;
            }
            awake += 1;
            if body.refresh_bounds(margin) {
                broadphase.set_proxy(handle.index() as u32, proxy_for(body));
            }
        }
        awake
    }
}

/// Hand the solver the bodies this step can reach: the ones it moves, and the
/// immovable ones a contact leans against. Everything else keeps whatever
/// state it had, because nothing will read it, which is what makes a settled
/// world cost close to nothing.
fn gather(bodies: &Pool<Body>, solver: &mut Solver, manifolds: &[Manifold], joints: &[Joint]) {
    for (handle, body) in bodies.iter() {
        if body.is_simulated() {
            solver.set_body(handle.index() as u32, SolverBody::from_body(body));
        }
    }
    for manifold in manifolds {
        gather_partner(bodies, solver, manifold.a, manifold.b);
    }
    for joint in joints {
        gather_partner(bodies, solver, joint.a, joint.b);
    }
}

/// Take the immovable half of a constraint, so the moving half has something
/// to lean on.
fn gather_partner(bodies: &Pool<Body>, solver: &mut Solver, a: u32, b: u32) {
    let simulated = |slot: u32| {
        bodies
            .get_at(slot as usize)
            .is_some_and(|body| body.is_simulated())
    };
    let (moves_a, moves_b) = (simulated(a), simulated(b));
    if moves_a == moves_b {
        // Both moving: already taken. Neither moving: the constraint is
        // skipped, so neither is needed.
        return;
    }
    let resting = if moves_a { b } else { a };
    if let Some(body) = bodies.get_at(resting as usize) {
        solver.set_body(resting, SolverBody::from_body(body));
    }
}

fn proxy_for(body: &Body) -> Proxy {
    let role = if body.is_sensor() {
        Role::Sensor
    } else if body.responds_to_contact() {
        Role::Dynamic
    } else if body.is_kinematic() {
        Role::Driven
    } else {
        Role::Static
    };
    Proxy {
        bounds: body.bounds,
        mask: body.mask,
        role,
    }
}

fn pool_handle(handle: BodyHandle) -> PoolHandle {
    PoolHandle::from_parts(handle.index(), handle.generation())
}

/// The body a handle still names, or `None` once it has been removed. What a
/// caller holding a handle from an earlier step asks.
pub(super) fn body_at(bodies: &Pool<Body>, handle: BodyHandle) -> Option<&Body> {
    bodies.get(pool_handle(handle))
}

fn body_handle(handle: PoolHandle) -> BodyHandle {
    BodyHandle::from_parts(handle.index() as u32, handle.generation())
}

/// The handle naming whatever occupies a slot, for a caller that walked the
/// broad phase and holds a slot rather than a handle.
pub(super) fn handle_at(bodies: &Pool<Body>, slot: u32) -> Option<BodyHandle> {
    bodies.handle_at(slot as usize).map(body_handle)
}

/// Wake whatever a driven body is about to move into, so a platform can
/// disturb a stack that has settled on it.
///
/// Without this a sleeping body would be gathered as immovable and the
/// platform would slide straight through it.
fn wake_driven_contacts(bodies: &mut Pool<Body>, manifolds: &[Manifold]) {
    for manifold in manifolds {
        for (slot, other) in [(manifold.a, manifold.b), (manifold.b, manifold.a)] {
            let driving = bodies.get_at(slot as usize).is_some_and(|body| {
                body.kinematic_target
                    .is_some_and(|target| target != body.position)
            });
            if driving && let Some(body) = bodies.get_at_mut(other as usize) {
                body.wake();
            }
        }
    }
}

/// Decide which islands have settled, and stop simulating the ones that have.
fn update_sleep(
    config: &SimConfig,
    bodies: &mut Pool<Body>,
    islands: &mut Islands,
    manifolds: &[Manifold],
    joints: &[Joint],
    dt: f32,
) {
    for (_, body) in bodies.iter_mut() {
        if !body.is_dynamic() {
            continue;
        }
        if !config.allow_sleep {
            body.wake();
            continue;
        }
        if body.sleeping {
            continue;
        }
        if body.is_still(config.sleep_linear_velocity, config.sleep_angular_velocity) {
            body.sleep_timer += dt;
        } else {
            body.sleep_timer = 0.0;
        }
    }
    if !config.allow_sleep {
        return;
    }

    islands.clear();
    let movable = |slot: u32| {
        bodies
            .get_at(slot as usize)
            .is_some_and(|body| body.is_dynamic())
    };
    for manifold in manifolds {
        if movable(manifold.a) && movable(manifold.b) {
            islands.union(manifold.a, manifold.b);
        }
    }
    // Two bodies a joint holds settle together or not at all, exactly as two
    // bodies leaning on each other do.
    for joint in joints {
        if movable(joint.a) && movable(joint.b) {
            islands.union(joint.a, joint.b);
        }
    }
    for (handle, body) in bodies.iter() {
        if body.is_dynamic() {
            islands.mark(
                handle.index() as u32,
                body.sleep_timer >= config.time_to_sleep,
            );
        }
    }
    // A motor with somewhere to go is still doing work however still the
    // bodies look, so its island stays awake.
    for joint in joints {
        if !joint.frame.is_driven() {
            continue;
        }
        for slot in [joint.a, joint.b] {
            if movable(slot) {
                islands.mark(slot, false);
            }
        }
    }
    for (handle, body) in bodies.iter_mut() {
        if !body.is_dynamic() {
            continue;
        }
        if islands.island_is_still(handle.index() as u32) {
            body.sleep();
        } else {
            body.sleeping = false;
        }
    }
}

#[cfg(test)]
mod tests {

    // A capsule swept at a floor stops where its lower cap meets it, and the
    // hit reports the surface it landed on.
    #[test]
    fn a_shape_cast_stops_at_the_first_body_in_its_path() {
        let mut sim = Simulation::with_capacity(1);
        sim.add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [10.0, 0.5, 10.0],
            },
            [0.0, -0.5, 0.0],
            [0.0; 3],
            0.8,
            LayerMask::ALL,
        );

        let capsule = ColliderShape::Capsule {
            half_height: 0.6,
            radius: 0.3,
        };
        let hit = sim
            .shape_cast(&ShapeCast::new(capsule, [0.0, 4.0, 0.0], [0.0, -8.0, 0.0]))
            .expect("the floor is down there");

        // The capsule reaches 0.9 below its centre, so it stops with its
        // centre 0.9 above the floor.
        let landed = 4.0 - hit.toi * 8.0;
        assert!((landed - 0.9).abs() < 0.01, "landed at {landed}");
        assert!(hit.normal[1] > 0.99, "standing on it: {:?}", hit.normal);
    }
    use super::*;

    const TICK: f32 = 1.0 / 60.0;

    fn params(restitution: f32, damping: f32) -> DynamicParams {
        DynamicParams {
            mass: 1.0,
            friction: 0.5,
            restitution,
            gravity_scale: 1.0,
            linear_damping: damping,
        }
    }

    fn floor(sim: &mut Simulation) -> BodyHandle {
        sim.add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [50.0, 1.0, 50.0],
            },
            [0.0, -1.0, 0.0],
            [0.0; 3],
            0.8,
            LayerMask::ALL,
        )
        .expect("room")
    }

    #[test]
    fn a_body_falls_under_gravity() {
        let mut sim = Simulation::with_capacity(1);
        let ball = sim
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 10.0, 0.0],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask::ALL,
            )
            .expect("room");
        for _ in 0..60 {
            sim.step(TICK);
        }
        let (pos, _) = sim.body_pose(ball).expect("live");
        // One second of free fall at g = 20 covers about ten units.
        assert!((pos[1] - (10.0 - 10.0)).abs() < 0.4, "y = {}", pos[1]);
        assert!(sim.linear_velocity(ball).expect("live")[1] < -19.0);
    }

    #[test]
    fn a_zero_or_negative_step_changes_nothing() {
        let mut sim = Simulation::with_capacity(1);
        let ball = sim
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 10.0, 0.0],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask::ALL,
            )
            .expect("room");
        sim.step(0.0);
        sim.step(-1.0);
        assert_eq!(sim.body_pose(ball).expect("live").0, [0.0, 10.0, 0.0]);
    }

    #[test]
    fn a_full_pool_declines_rather_than_growing() {
        let mut sim = Simulation::with_capacity(1);
        assert!(
            sim.add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 1.0, 0.0],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask::ALL
            )
            .is_some()
        );
        assert!(
            sim.add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 3.0, 0.0],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask::ALL
            )
            .is_none()
        );
        assert_eq!(sim.body_count(), 1);
        assert_eq!(sim.capacity(), 1);
        assert!(sim.reserved_bytes() > 0);
    }

    #[test]
    fn a_removed_bodys_handle_stops_naming_anything() {
        let mut sim = Simulation::with_capacity(2);
        let ball = sim
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 1.0, 0.0],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask::ALL,
            )
            .expect("room");
        assert!(sim.remove_body(ball));
        assert!(!sim.remove_body(ball));
        assert!(sim.body_pose(ball).is_none());
        assert_eq!(sim.body_count(), 0);
    }

    #[test]
    fn a_body_lands_on_the_floor_and_stays_on_it() {
        let mut sim = Simulation::with_capacity(2);
        floor(&mut sim);
        let ball = sim
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 6.0, 0.0],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask::ALL,
            )
            .expect("room");
        for _ in 0..240 {
            sim.step(TICK);
        }
        let (pos, _) = sim.body_pose(ball).expect("live");
        assert!((pos[1] - 0.5).abs() < 0.02, "y = {}", pos[1]);
        assert!(sim.contact_count() > 0);
    }

    #[test]
    fn layers_that_do_not_interact_pass_through_each_other() {
        let mut sim = Simulation::with_capacity(2);
        sim.add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [50.0, 1.0, 50.0],
            },
            [0.0, -1.0, 0.0],
            [0.0; 3],
            0.8,
            LayerMask {
                memberships: 0b01,
                filter: 0b01,
            },
        );
        let ball = sim
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 4.0, 0.0],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask {
                    memberships: 0b10,
                    filter: 0b10,
                },
            )
            .expect("room");
        for _ in 0..120 {
            sim.step(TICK);
        }
        assert!(sim.body_pose(ball).expect("live").0[1] < -2.0);
    }

    #[test]
    fn an_impulse_moves_a_body_and_wakes_it() {
        let mut sim = Simulation::with_capacity(2);
        floor(&mut sim);
        let ball = sim
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 0.5, 0.0],
                [0.0; 3],
                params(0.0, 0.5),
                LayerMask::ALL,
            )
            .expect("room");
        for _ in 0..120 {
            sim.step(TICK);
        }
        assert_eq!(sim.is_sleeping(ball), Some(true));
        sim.apply_impulse(ball, [0.0, 8.0, 0.0]);
        assert_eq!(sim.is_sleeping(ball), Some(false));
        sim.step(TICK);
        assert!(sim.body_pose(ball).expect("live").0[1] > 0.55);
    }

    #[test]
    fn velocity_can_be_set_and_read_back() {
        let mut sim = Simulation::with_capacity(1);
        let ball = sim
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 5.0, 0.0],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask::ALL,
            )
            .expect("room");
        sim.set_linear_velocity(ball, [1.0, 0.0, 0.0]);
        sim.set_angular_velocity(ball, [0.0, 2.0, 0.0]);
        assert_eq!(sim.linear_velocity(ball), Some([1.0, 0.0, 0.0]));
        assert_eq!(sim.angular_velocity(ball), Some([0.0, 2.0, 0.0]));
        assert!(sim.mass(ball).expect("live") > 0.0);
    }

    // The same world stepped twice must land on the same bits, or nothing
    // downstream can be reproduced.
    #[test]
    fn two_identical_runs_agree_bit_for_bit() {
        let run = || {
            let mut sim = Simulation::with_capacity(16);
            floor(&mut sim);
            let mut handles = Vec::new();
            for i in 0..12 {
                handles.push(
                    sim.add_dynamic(
                        &ColliderShape::Cuboid {
                            half_extents: [0.4, 0.4, 0.4],
                        },
                        [(i % 3) as f32 * 0.9, 1.0 + (i / 3) as f32 * 0.9, 0.0],
                        [0.0, i as f32 * 7.0, 0.0],
                        params(0.3, 0.0),
                        LayerMask::ALL,
                    )
                    .expect("room"),
                );
            }
            for _ in 0..90 {
                sim.step(TICK);
            }
            handles
                .iter()
                .map(|&h| {
                    let (p, r) = sim.body_pose(h).expect("live");
                    (
                        [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()],
                        [r[0].to_bits(), r[1].to_bits(), r[2].to_bits()],
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn a_ray_finds_the_nearest_body_along_it() {
        let mut sim = Simulation::with_capacity(4);
        for (index, z) in [2.0f32, 5.0, 9.0].into_iter().enumerate() {
            sim.add_fixed(
                &[
                    ColliderShape::Ball { radius: 0.5 },
                    ColliderShape::Cuboid {
                        half_extents: [0.5, 0.5, 0.5],
                    },
                    ColliderShape::Capsule {
                        half_height: 0.5,
                        radius: 0.25,
                    },
                ][index],
                [0.0, 0.0, z],
                [0.0; 3],
                0.5,
                LayerMask::ALL,
            )
            .expect("room");
        }
        let hit = sim
            .raycast(
                [0.0, 0.0, -5.0],
                [0.0, 0.0, 1.0],
                100.0,
                None,
                LayerMask::ALL,
            )
            .expect("a hit");
        assert!((hit.distance - 6.5).abs() < 1.0e-4, "{hit:?}");
        assert!((hit.normal[2] + 1.0).abs() < 1.0e-4, "{hit:?}");
        assert!((hit.point[2] - 1.5).abs() < 1.0e-4, "{hit:?}");
    }

    #[test]
    fn a_ray_answers_the_same_way_from_either_end_of_the_scene() {
        let mut sim = Simulation::with_capacity(4);
        for z in [2.0f32, 5.0, 9.0] {
            sim.add_fixed(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 0.0, z],
                [0.0; 3],
                0.5,
                LayerMask::ALL,
            )
            .expect("room");
        }
        sim.step(TICK);
        let forward = sim
            .raycast(
                [0.0, 0.0, -5.0],
                [0.0, 0.0, 1.0],
                100.0,
                None,
                LayerMask::ALL,
            )
            .expect("a hit");
        let backward = sim
            .raycast(
                [0.0, 0.0, 20.0],
                [0.0, 0.0, -1.0],
                100.0,
                None,
                LayerMask::ALL,
            )
            .expect("a hit");
        assert!((forward.distance - 6.5).abs() < 1.0e-4, "{forward:?}");
        assert!((backward.distance - 10.5).abs() < 1.0e-4, "{backward:?}");
    }

    #[test]
    fn a_ray_skips_layers_it_does_not_interact_with() {
        let mut sim = Simulation::with_capacity(2);
        let near = LayerMask {
            memberships: 0b01,
            filter: 0b11,
        };
        let far = LayerMask {
            memberships: 0b10,
            filter: 0b11,
        };
        sim.add_fixed(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 0.0, 0.0],
            [0.0; 3],
            0.5,
            near,
        )
        .expect("room");
        sim.add_fixed(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 0.0, 5.0],
            [0.0; 3],
            0.5,
            far,
        )
        .expect("room");

        let only_far = LayerMask {
            memberships: 0b11,
            filter: 0b10,
        };
        let hit = sim
            .raycast([0.0, 0.0, -5.0], [0.0, 0.0, 1.0], 100.0, None, only_far)
            .expect("a hit");
        assert!(
            (hit.distance - 9.5).abs() < 1.0e-4,
            "the near one is hidden"
        );
        // With no layer in common there is nothing to hit at all.
        assert!(
            sim.raycast(
                [0.0, 0.0, -5.0],
                [0.0, 0.0, 1.0],
                100.0,
                None,
                LayerMask {
                    memberships: 0b100,
                    filter: 0b100,
                },
            )
            .is_none()
        );
    }

    #[test]
    fn a_ray_can_be_told_to_leave_one_body_out() {
        let mut sim = Simulation::with_capacity(2);
        let near = sim
            .add_fixed(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 0.0, 0.0],
                [0.0; 3],
                0.5,
                LayerMask::ALL,
            )
            .expect("room");
        sim.add_fixed(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 0.0, 5.0],
            [0.0; 3],
            0.5,
            LayerMask::ALL,
        )
        .expect("room");

        let cast = |exclude| {
            sim.raycast(
                [0.0, 0.0, -5.0],
                [0.0, 0.0, 1.0],
                100.0,
                exclude,
                LayerMask::ALL,
            )
        };
        assert!((cast(None).expect("a hit").distance - 4.5).abs() < 1.0e-4);
        assert!((cast(Some(near)).expect("a hit").distance - 9.5).abs() < 1.0e-4);
    }

    // A handle to a removed body must not exclude whatever took its slot.
    #[test]
    fn a_stale_exclusion_does_not_hide_the_slots_new_occupant() {
        let mut sim = Simulation::with_capacity(1);
        let first = sim
            .add_fixed(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 0.0, 0.0],
                [0.0; 3],
                0.5,
                LayerMask::ALL,
            )
            .expect("room");
        assert!(sim.remove_body(first));
        sim.add_fixed(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 0.0, 0.0],
            [0.0; 3],
            0.5,
            LayerMask::ALL,
        )
        .expect("the freed slot");
        assert!(
            sim.raycast(
                [0.0, 0.0, -5.0],
                [0.0, 0.0, 1.0],
                100.0,
                Some(first),
                LayerMask::ALL,
            )
            .is_some(),
            "the stale handle names nobody"
        );
    }

    #[test]
    fn a_removed_body_stops_being_hit() {
        let mut sim = Simulation::with_capacity(1);
        let ball = sim
            .add_fixed(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 0.0, 0.0],
                [0.0; 3],
                0.5,
                LayerMask::ALL,
            )
            .expect("room");
        let cast = |sim: &Simulation| {
            sim.raycast(
                [0.0, 0.0, -5.0],
                [0.0, 0.0, 1.0],
                100.0,
                None,
                LayerMask::ALL,
            )
        };
        assert!(cast(&sim).is_some());
        sim.remove_body(ball);
        assert!(cast(&sim).is_none());
    }

    #[test]
    fn a_ray_respects_its_distance_limit_and_needs_a_direction() {
        let mut sim = Simulation::with_capacity(1);
        sim.add_fixed(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 0.0, 0.0],
            [0.0; 3],
            0.5,
            LayerMask::ALL,
        )
        .expect("room");
        let cast =
            |dir: [f32; 3], max| sim.raycast([0.0, 0.0, -5.0], dir, max, None, LayerMask::ALL);
        // The surface is exactly 4.5 away.
        assert!(cast([0.0, 0.0, 1.0], 4.5).is_some());
        assert!(cast([0.0, 0.0, 1.0], 4.4).is_none());
        assert!(cast([0.0, 0.0, 0.0], 100.0).is_none(), "no direction");
        assert!(cast([0.0, 0.0, 1.0], 0.0).is_none(), "no reach");
        assert!(cast([0.0, 0.0, 1.0], -1.0).is_none());
        assert!(cast([f32::NAN, 0.0, 0.0], 100.0).is_none());
        // An unnormalised direction is the same ray.
        assert!((cast([0.0, 0.0, 7.0], 100.0).expect("a hit").distance - 4.5).abs() < 1.0e-4);
    }

    #[test]
    fn a_ray_finds_a_body_added_since_the_last_step() {
        let mut sim = Simulation::with_capacity(2);
        floor(&mut sim);
        sim.step(TICK);
        sim.add_fixed(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 5.0, 0.0],
            [0.0; 3],
            0.5,
            LayerMask::ALL,
        )
        .expect("room");
        let hit = sim
            .raycast(
                [0.0, 9.0, 0.0],
                [0.0, -1.0, 0.0],
                100.0,
                None,
                LayerMask::ALL,
            )
            .expect("a hit");
        assert!((hit.distance - 3.5).abs() < 1.0e-4, "{hit:?}");
    }

    #[test]
    fn a_ray_hits_a_position_driven_body() {
        let mut sim = Simulation::with_capacity(1);
        sim.add_kinematic(
            &ColliderShape::Cuboid {
                half_extents: [1.0, 0.25, 1.0],
            },
            [0.0, 0.0, 0.0],
            [0.0; 3],
            0.5,
            LayerMask::ALL,
        )
        .expect("room");
        let hit = sim
            .raycast(
                [0.0, 5.0, 0.0],
                [0.0, -1.0, 0.0],
                100.0,
                None,
                LayerMask::ALL,
            )
            .expect("a hit");
        assert!((hit.distance - 4.75).abs() < 1.0e-4, "{hit:?}");
    }

    #[test]
    fn a_shape_cast_stops_at_the_nearest_body_and_names_it() {
        let mut sim = Simulation::with_capacity(3);
        let ground = floor(&mut sim);
        let ledge = sim
            .add_fixed(
                &ColliderShape::Cuboid {
                    half_extents: [2.0, 0.5, 2.0],
                },
                [0.0, 3.0, 0.0],
                [0.0; 3],
                0.5,
                LayerMask::ALL,
            )
            .expect("room");
        let capsule = ColliderShape::Capsule {
            half_height: 0.5,
            radius: 0.25,
        };
        let hit = sim
            .shape_cast(&ShapeCast::new(capsule, [0.0, 9.0, 0.0], [0.0, -12.0, 0.0]))
            .expect("a hit");
        assert_eq!(hit.body, ledge, "the ledge, not the ground under it");
        let landed = 9.0 - hit.toi * 12.0;
        assert!((landed - 4.25).abs() < 1.0e-2, "landed at {landed}");
        assert!(!hit.started_touching);
        assert_ne!(hit.body, ground);
    }

    #[test]
    fn a_shape_cast_that_reaches_nothing_reports_nothing() {
        let mut sim = Simulation::with_capacity(2);
        floor(&mut sim);
        let ball = ColliderShape::Ball { radius: 0.5 };
        assert!(
            sim.shape_cast(&ShapeCast::new(ball, [0.0, 9.0, 0.0], [0.0, -1.0, 0.0]))
                .is_none()
        );
        assert!(
            sim.shape_cast(&ShapeCast::new(ball, [0.0, 9.0, 0.0], [0.0, 5.0, 0.0]))
                .is_none(),
            "away from everything"
        );
    }

    #[test]
    fn a_shape_cast_says_when_it_began_in_contact() {
        let mut sim = Simulation::with_capacity(2);
        let ground = floor(&mut sim);
        let ball = ColliderShape::Ball { radius: 0.5 };
        let hit = sim
            .shape_cast(&ShapeCast::new(ball, [0.0, 0.4, 0.0], [3.0, 0.0, 0.0]))
            .expect("a hit");
        assert_eq!(hit.body, ground);
        assert_eq!(hit.toi, 0.0);
        assert!(hit.started_touching);
        assert!(hit.normal[1] > 0.9, "{hit:?}");
    }

    #[test]
    fn a_shape_cast_honours_its_layer_filter_and_its_exclusion() {
        let mut sim = Simulation::with_capacity(2);
        let near = sim
            .add_fixed(
                &ColliderShape::Cuboid {
                    half_extents: [2.0, 0.5, 2.0],
                },
                [0.0, 2.0, 0.0],
                [0.0; 3],
                0.5,
                LayerMask {
                    memberships: 0b01,
                    filter: 0b11,
                },
            )
            .expect("room");
        let far = sim
            .add_fixed(
                &ColliderShape::Cuboid {
                    half_extents: [2.0, 0.5, 2.0],
                },
                [0.0, 0.0, 0.0],
                [0.0; 3],
                0.5,
                LayerMask {
                    memberships: 0b10,
                    filter: 0b11,
                },
            )
            .expect("room");
        let ball = ColliderShape::Ball { radius: 0.5 };
        let straight = ShapeCast::new(ball, [0.0, 6.0, 0.0], [0.0, -8.0, 0.0]);
        assert_eq!(sim.shape_cast(&straight).expect("a hit").body, near);
        assert_eq!(
            sim.shape_cast(&ShapeCast {
                exclude: Some(near),
                ..straight
            })
            .expect("a hit")
            .body,
            far
        );
        assert_eq!(
            sim.shape_cast(&ShapeCast {
                mask: LayerMask {
                    memberships: 0b11,
                    filter: 0b10,
                },
                ..straight
            })
            .expect("a hit")
            .body,
            far
        );
    }

    #[test]
    fn a_driven_body_arrives_exactly_where_it_was_sent() {
        let mut sim = Simulation::with_capacity(1);
        let platform = sim
            .add_kinematic(
                &ColliderShape::Cuboid {
                    half_extents: [1.0, 0.25, 1.0],
                },
                [0.0, 0.0, 0.0],
                [0.0; 3],
                0.5,
                LayerMask::ALL,
            )
            .expect("room");
        assert!(sim.set_kinematic_translation(platform, [1.5, 2.0, -3.0]));
        sim.step(TICK);
        assert_eq!(sim.body_pose(platform).expect("live").0, [1.5, 2.0, -3.0]);
        // The target is spent, so a body left alone stops where it arrived.
        sim.step(TICK);
        assert_eq!(sim.body_pose(platform).expect("live").0, [1.5, 2.0, -3.0]);
        assert_eq!(sim.linear_velocity(platform), Some([0.0; 3]));
    }

    #[test]
    fn a_driven_body_ignores_gravity_and_impulses() {
        let mut sim = Simulation::with_capacity(1);
        let platform = sim
            .add_kinematic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 5.0, 0.0],
                [0.0; 3],
                0.5,
                LayerMask::ALL,
            )
            .expect("room");
        sim.apply_impulse(platform, [0.0, 100.0, 0.0]);
        for _ in 0..120 {
            sim.step(TICK);
        }
        assert_eq!(sim.body_pose(platform).expect("live").0, [0.0, 5.0, 0.0]);
        assert_eq!(sim.mass(platform), Some(0.0));
        assert_eq!(sim.is_kinematic(platform), Some(true));
    }

    // A character capsule is a position-driven body: it holds its place under
    // gravity and is moved by a target rather than by the solver.
    #[test]
    fn a_character_capsule_is_a_position_driven_body() {
        let mut sim = Simulation::with_capacity(1);
        let handle = sim
            .add_character(0.6, 0.3, [0.0, 4.0, 0.0], LayerMask::ALL)
            .expect("room in the pool");
        assert_eq!(sim.is_kinematic(handle), Some(true));

        for _ in 0..60 {
            sim.step(TICK);
        }
        let (position, _) = sim.body_pose(handle).expect("a live body");
        assert_eq!(position[1], 4.0, "gravity does not move a driven capsule");

        assert!(sim.set_kinematic_translation(handle, [0.0, 3.0, 0.0]));
        sim.step(TICK);
        let (position, _) = sim.body_pose(handle).expect("a live body");
        assert!((position[1] - 3.0).abs() < 1.0e-5, "{position:?}");
    }

    // A body set to a position it already holds must not read as motion, and
    // a body that is not driven by position must refuse the instruction.
    #[test]
    fn only_a_position_driven_body_takes_a_translation_target() {
        let mut sim = Simulation::with_capacity(2);
        let fixed = floor(&mut sim);
        let ball = sim
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 5.0, 0.0],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask::ALL,
            )
            .expect("room");
        assert!(!sim.set_kinematic_translation(fixed, [0.0, 9.0, 0.0]));
        assert!(!sim.set_kinematic_translation(ball, [0.0, 9.0, 0.0]));
        sim.step(TICK);
        assert!(sim.body_pose(ball).expect("live").0[1] < 5.0, "still falls");
    }

    #[test]
    fn a_driven_body_pushes_a_dynamic_one_it_is_moved_into() {
        let mut sim = Simulation::with_capacity(3);
        floor(&mut sim);
        let crate_body = sim
            .add_dynamic(
                &ColliderShape::Cuboid {
                    half_extents: [0.5, 0.5, 0.5],
                },
                [0.0, 0.5, 0.0],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask::ALL,
            )
            .expect("room");
        let blade = sim
            .add_kinematic(
                &ColliderShape::Cuboid {
                    half_extents: [0.5, 0.5, 0.5],
                },
                [-3.0, 0.5, 0.0],
                [0.0; 3],
                0.5,
                LayerMask::ALL,
            )
            .expect("room");

        // Long enough for the crate to settle and doze off before the blade
        // arrives, so this also proves the push wakes it.
        let mut x = -3.0f32;
        for step in 0..180 {
            if step == 60 {
                assert_eq!(sim.is_sleeping(crate_body), Some(true), "settled first");
            }
            if step >= 60 {
                x += 0.03;
                assert!(sim.set_kinematic_translation(blade, [x, 0.5, 0.0]));
            }
            sim.step(TICK);
        }
        assert!((sim.body_pose(blade).expect("live").0[0] - x).abs() < 1.0e-5);
        let pushed = sim.body_pose(crate_body).expect("live").0[0];
        assert!(pushed > 1.5, "the crate was shoved along: {pushed}");
        assert!(
            pushed > x,
            "and it stays ahead of the blade: {pushed} vs {x}"
        );
    }

    #[test]
    fn switching_a_bodys_kind_keeps_its_handle_and_leaves_the_world_standing() {
        let mut sim = Simulation::with_capacity(2);
        floor(&mut sim);
        let ball = sim
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 4.0, 0.0],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask::ALL,
            )
            .expect("room");
        for _ in 0..180 {
            sim.step(TICK);
        }
        let resting = sim.body_pose(ball).expect("live").0;
        assert!((resting[1] - 0.5).abs() < 0.02, "{resting:?}");

        // Picked up: held in the air by position alone.
        assert!(sim.make_kinematic(ball));
        assert_eq!(sim.is_kinematic(ball), Some(true));
        assert!(sim.set_kinematic_translation(ball, [0.0, 6.0, 0.0]));
        sim.step(TICK);
        for _ in 0..60 {
            sim.step(TICK);
        }
        assert_eq!(sim.body_pose(ball).expect("live").0, [0.0, 6.0, 0.0]);

        // Thrown: the same handle, back under gravity, and it lands again.
        assert!(sim.make_dynamic(ball, [0.0, 0.0, 2.0]));
        assert_eq!(sim.is_kinematic(ball), Some(false));
        assert!(sim.mass(ball).expect("live") > 0.0);
        for _ in 0..300 {
            sim.step(TICK);
        }
        let landed = sim.body_pose(ball).expect("live").0;
        assert!(
            (landed[1] - 0.5).abs() < 0.02,
            "back on the floor: {landed:?}"
        );
        assert!(landed[2] > 0.5, "and it travelled: {landed:?}");
        assert_eq!(sim.body_count(), 2);
        assert_eq!(sim.collider_count(), sim.body_count());
    }

    // The broad phase filters pairs by whether contact can move either body,
    // so a switch that did not reach it would leave a stack leaning on
    // nothing.
    #[test]
    fn a_stack_stays_up_when_the_body_under_it_is_switched() {
        let mut sim = Simulation::with_capacity(3);
        floor(&mut sim);
        let cube = ColliderShape::Cuboid {
            half_extents: [0.5, 0.5, 0.5],
        };
        let lower = sim
            .add_dynamic(
                &cube,
                [0.0, 0.5, 0.0],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask::ALL,
            )
            .expect("room");
        let upper = sim
            .add_dynamic(
                &cube,
                [0.0, 1.5, 0.0],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask::ALL,
            )
            .expect("room");
        for _ in 0..180 {
            sim.step(TICK);
        }
        let held = sim.body_pose(lower).expect("live").0;
        assert!(sim.make_kinematic(lower));
        for _ in 0..180 {
            sim.step(TICK);
        }
        let top = sim.body_pose(upper).expect("live").0;
        assert!(
            (top[1] - 1.5).abs() < 0.05,
            "the top box still rests: {top:?}"
        );
        assert_eq!(
            sim.body_pose(lower).expect("live").0,
            held,
            "and the one below it has not stirred"
        );

        // And a lift carries the box above it up too.
        assert!(sim.set_kinematic_translation(lower, [held[0], 1.5, held[2]]));
        for _ in 0..120 {
            sim.step(TICK);
        }
        let lifted = sim.body_pose(upper).expect("live").0;
        assert!(lifted[1] > 2.0, "carried up: {lifted:?}");
    }

    #[test]
    fn switching_the_kind_of_a_body_that_is_not_there_reports_so() {
        let mut sim = Simulation::with_capacity(1);
        let ball = sim
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 1.0, 0.0],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask::ALL,
            )
            .expect("room");
        assert!(sim.remove_body(ball));
        assert!(!sim.make_kinematic(ball));
        assert!(!sim.make_dynamic(ball, [0.0; 3]));
        assert!(!sim.set_kinematic_translation(ball, [0.0; 3]));
        assert_eq!(sim.is_kinematic(ball), None);
    }

    // Two runs of the same query on the same world must agree bit for bit,
    // or nothing built on a query can be reproduced.
    #[test]
    fn queries_answer_the_same_bits_twice_running() {
        let mut sim = Simulation::with_capacity(17);
        floor(&mut sim);
        // Terrain answers along its own faces rather than through the sweep
        // the convex shapes use, so it needs its own place in this check.
        let side = 9usize;
        let mut heights = Vec::with_capacity(side * side);
        for row in 0..side {
            for col in 0..side {
                heights.push((row as f32 * 0.37).sin() * 0.4 + (col as f32 * 0.21).cos() * 0.3);
            }
        }
        sim.add_heightfield(
            side,
            side,
            heights,
            [24.0, 1.0, 24.0],
            [0.0, -3.0, 0.0],
            LayerMask::ALL,
        )
        .expect("room");
        for i in 0..12 {
            sim.add_dynamic(
                &ColliderShape::Cuboid {
                    half_extents: [0.4, 0.4, 0.4],
                },
                [(i % 3) as f32 * 0.9, 1.0 + (i / 3) as f32 * 0.9, 0.0],
                [0.0, i as f32 * 7.0, 0.0],
                params(0.3, 0.0),
                LayerMask::ALL,
            )
            .expect("room");
        }
        for _ in 0..90 {
            sim.step(TICK);
        }
        let ray = |sim: &Simulation| {
            sim.raycast(
                [-6.0, 1.3, 0.1],
                [1.0, -0.1, 0.0],
                40.0,
                None,
                LayerMask::ALL,
            )
            .map(|hit| (hit.distance.to_bits(), hit.point, hit.normal))
        };
        let sweep = |sim: &Simulation| {
            sim.shape_cast(&ShapeCast::new(
                ColliderShape::Capsule {
                    half_height: 0.3,
                    radius: 0.2,
                },
                [-6.0, 1.3, 0.1],
                [12.0, 0.0, 0.0],
            ))
            .map(|hit| (hit.toi.to_bits(), hit.point, hit.normal, hit.body))
        };
        let shape = Simulation::character_shape(0.3, 0.2);
        let capsule = sim
            .add_kinematic(
                &ColliderShape::Capsule {
                    half_height: 0.3,
                    radius: 0.2,
                },
                [-6.0, 1.3, 0.1],
                [0.0; 3],
                0.8,
                LayerMask::ALL,
            )
            .expect("room");
        let walk = |sim: &Simulation| {
            let moved = sim.move_character(
                &shape,
                &CharacterMoveInput {
                    center: [-6.0, 1.3, 0.1],
                    desired: [12.0, -0.05, 0.0],
                    dt: TICK,
                    exclude: capsule,
                    mask: LayerMask::ALL,
                },
            );
            (moved.translation.map(f32::to_bits), moved.grounded)
        };
        // Fired into the terrain rather than at the stack, so the two answers
        // below are about the height grid and not about the boxes on it.
        let terrain_ray = |sim: &Simulation| {
            sim.raycast(
                [-1.7, 6.0, 2.3],
                [0.0, -1.0, 0.0],
                40.0,
                None,
                LayerMask::ALL,
            )
            .map(|hit| (hit.distance.to_bits(), hit.point, hit.normal))
        };
        let terrain_sweep = |sim: &Simulation| {
            sim.shape_cast(&ShapeCast::new(
                ColliderShape::Ball { radius: 0.4 },
                [-9.0, -2.2, 2.3],
                [18.0, 0.0, 0.0],
            ))
            .map(|hit| (hit.toi.to_bits(), hit.point, hit.normal, hit.body))
        };
        assert!(
            ray(&sim).is_some() && sweep(&sim).is_some(),
            "the scene is in the way"
        );
        assert!(
            terrain_ray(&sim).is_some() && terrain_sweep(&sim).is_some(),
            "and so is the terrain"
        );
        assert_eq!(ray(&sim), ray(&sim));
        assert_eq!(sweep(&sim), sweep(&sim));
        assert_eq!(walk(&sim), walk(&sim));
        assert_eq!(terrain_ray(&sim), terrain_ray(&sim));
        assert_eq!(terrain_sweep(&sim), terrain_sweep(&sim));
    }

    #[test]
    fn config_round_trips_and_takes_effect() {
        // Sleeping is off for the weightless stretch: a body that settles
        // there would stop being simulated, and the test would be measuring
        // that rather than the gravity change.
        let mut sim = Simulation::new(
            SimConfig {
                gravity: 0.0,
                allow_sleep: false,
                ..SimConfig::default()
            },
            1,
        );
        assert_eq!(sim.config().gravity, 0.0);
        let ball = sim
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 5.0, 0.0],
                [0.0; 3],
                params(0.0, 0.0),
                LayerMask::ALL,
            )
            .expect("room");
        for _ in 0..60 {
            sim.step(TICK);
        }
        assert_eq!(sim.body_pose(ball).expect("live").0[1], 5.0);
        sim.set_config(SimConfig::default());
        assert_eq!(sim.config().gravity, crate::GRAVITY);
        sim.step(TICK);
        assert!(sim.body_pose(ball).expect("live").0[1] < 5.0);
    }

    #[test]
    fn gravity_scale_and_damping_do_what_they_say() {
        let mut sim = Simulation::with_capacity(2);
        let floating = sim
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [0.0, 5.0, 0.0],
                [0.0; 3],
                DynamicParams {
                    gravity_scale: 0.0,
                    ..params(0.0, 0.0)
                },
                LayerMask::ALL,
            )
            .expect("room");
        let damped = sim
            .add_dynamic(
                &ColliderShape::Ball { radius: 0.5 },
                [5.0, 5.0, 0.0],
                [0.0; 3],
                DynamicParams {
                    gravity_scale: 0.0,
                    ..params(0.0, 4.0)
                },
                LayerMask::ALL,
            )
            .expect("room");
        sim.set_linear_velocity(floating, [3.0, 0.0, 0.0]);
        sim.set_linear_velocity(damped, [3.0, 0.0, 0.0]);
        for _ in 0..60 {
            sim.step(TICK);
        }
        assert_eq!(sim.body_pose(floating).expect("live").0[1], 5.0);
        let free = sim.linear_velocity(floating).expect("live")[0];
        let slowed = sim.linear_velocity(damped).expect("live")[0];
        assert!((free - 3.0).abs() < 1.0e-5, "{free}");
        assert!(slowed < 0.5, "damping must bleed the speed off: {slowed}");
    }
}

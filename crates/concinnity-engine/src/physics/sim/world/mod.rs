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

use concinnity_core::math::SaturatingFrom;

use concinnity_core::memory::{Pool, PoolHandle};

use crate::physics::BodyHandle;

use super::body::Body;
use super::broadphase::{Proxy, Role, SweepPrune};
use super::ccd::Ccd;
use super::character::{CharacterCapsule, CharacterConfig};
use super::collide::heightfield::Heightfields;
use super::config::SimConfig;
use super::contact::ContactCache;
use super::fluid::FluidLoad;
use super::impact::Impacts;
use super::island::Islands;
use super::joint::JointSet;
use super::narrow::Narrow;
use super::sensor::Sensors;
use super::solver::{self, Solver};
use alloc::vec::Vec;

/// A rigid-body simulation: bodies fall under gravity, collide, and come to
/// rest.
///
/// The capacity given at construction is the whole reservation. Adding past it
/// returns `None`; stepping never allocates.
///
/// # Examples
///
/// ```
/// use concinnity_engine::physics::{ColliderShape, DynamicParams, LayerMask, Simulation};
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
    /// The water loads the next step reads, each named by its body's
    /// `fluid`; emptied by the step. Reserved by `reserve_fluid_loads`.
    fluids: Vec<FluidLoad>,
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
            .finish_non_exhaustive()
    }
}

mod build;
mod control;
mod external;
mod queries;
mod step;
#[cfg(test)]
mod tests;

impl Simulation {
    /// Reserve room for `capacity` bodies, with the default tuning.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(SimConfig::default(), capacity)
    }

    /// Reserve room for `capacity` bodies, tuned by `config`.
    pub fn new(config: SimConfig, capacity: usize) -> Self {
        Self {
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
            fluids: Vec::new(),
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
    /// Sensor pairs the last step measured rather than carried over.
    pub(crate) fn sensor_pairs_measured(&self) -> usize {
        self.sensors.measured()
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

    fn add(&mut self, mut body: Body) -> Option<BodyHandle> {
        body.refresh_bounds(self.config.bounds_margin);
        let proxy = proxy_for(&body);
        let handle = self.bodies.insert(body)?;
        let slot = u32::saturating_from(handle.index());
        self.broadphase.insert(slot);
        self.broadphase.set_proxy(slot, proxy);
        // A body arriving inside a settled stack has to be able to disturb it.
        self.disturb(slot);
        Some(body_handle(handle))
    }

    /// Account for a body that was added, removed, moved or reclassified
    /// outside a step: its sensor pairs are measured afresh and its neighbors
    /// are woken.
    fn disturb(&mut self, slot: u32) {
        self.sensors.mark_moved(slot);
        self.wake_neighbors(slot);
    }

    /// Wake whatever the given slot was in contact with or jointed to, so a
    /// change at one body reaches the island it belonged to.
    fn wake_neighbors(&mut self, slot: u32) {
        let Self {
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
    BodyHandle::from_parts(u32::saturating_from(handle.index()), handle.generation())
}

/// The handle naming whatever occupies a slot, for a caller that walked the
/// broad phase and holds a slot rather than a handle.
pub(super) fn handle_at(bodies: &Pool<Body>, slot: u32) -> Option<BodyHandle> {
    bodies.handle_at(slot as usize).map(body_handle)
}

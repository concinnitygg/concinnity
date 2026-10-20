// Adding bodies, colliders, joints and sensors, and draining what they report.

use super::{Simulation, pool_handle};
use crate::physics::sim::body::Body;
use crate::physics::sim::collide::heightfield::Heightfield;
use crate::physics::sim::joint::JointImpulses;
use crate::physics::sim::joint::{Joint, JointFrame};
use crate::physics::sim::math::{Quat, Vec3};
use crate::physics::{
    BodyHandle, ColliderShape, ContactHit, DynamicParams, JointSpec, LayerMask, SensorCrossing,
};
use alloc::vec::Vec;

// Friction of a character capsule against what it is driven into. A character
// is moved by resolving a translation rather than by the solver, so this only
// governs what the capsule pushes.
const CHARACTER_FRICTION: f32 = 0.5;

impl Simulation {
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

    /// Add a position-driven character capsule centered on `center`: a
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
    /// use concinnity_core::physics::{ColliderShape, DynamicParams, LayerMask, Simulation};
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
    /// extent `[width, height_multiplier, depth]`, and the grid is centered on
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
    /// use concinnity_core::physics::{ColliderShape, DynamicParams, LayerMask, Simulation};
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
    pub(crate) fn clear_heightfield_overflows(&self) {
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
    /// use concinnity_core::physics::{
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
            impulses: JointImpulses::default(),
        });
        // A joint arriving on a settled body has to be able to disturb it.
        for slot in [slot_a, slot_b] {
            if let Some(body) = self.bodies.get_at_mut(slot as usize) {
                body.wake();
            }
        }
        true
    }
}

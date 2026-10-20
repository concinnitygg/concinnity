// Driving and reclassifying bodies after they are added.

use super::{Simulation, pool_handle, proxy_for};
use crate::physics::BodyHandle;
use crate::physics::sim::body::Body;
use crate::physics::sim::math::{Quat, Vec3};
use concinnity_core::math::SaturatingFrom;

impl Simulation {
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

    /// Move a body to `pos` with rotation `euler_deg`, discarding the motion
    /// it had. Returns whether the handle named a live body with a shape to
    /// move (terrain is the world, not something in it).
    ///
    /// The jump is not motion: both velocities are cleared and any pending
    /// kinematic target is dropped, so the body arrives at rest and is driven
    /// again only by what happens to it from there. Whatever it was leaning on
    /// where it stood is woken, as is the body itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use concinnity_engine::physics::{ColliderShape, DynamicParams, LayerMask, Simulation};
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
    ///             restitution: 0.0,
    ///             gravity_scale: 1.0,
    ///             linear_damping: 0.0,
    ///         },
    ///         LayerMask::ALL,
    ///     )
    ///     .expect("room for one body");
    /// for _ in 0..30 {
    ///     sim.step(1.0 / 60.0);
    /// }
    ///
    /// assert!(sim.teleport_body(ball, [4.0, 10.0, 0.0], [0.0; 3]));
    /// let (position, _) = sim.body_pose_quat(ball).expect("a live body");
    /// assert_eq!(position, [4.0, 10.0, 0.0], "it is where it was put");
    /// ```
    pub fn teleport_body(
        &mut self,
        handle: BodyHandle,
        pos: [f32; 3],
        euler_deg: [f32; 3],
    ) -> bool {
        let margin = self.config.bounds_margin;
        let Some(body) = self.bodies.get_mut(pool_handle(handle)) else {
            return false;
        };
        if body.convex().is_none() {
            return false;
        }
        body.position = Vec3::from_array(pos);
        body.orientation = Quat::from_euler_deg(euler_deg);
        body.linear_velocity = Vec3::ZERO;
        body.angular_velocity = Vec3::ZERO;
        body.kinematic_target = None;
        body.wake();
        body.refresh_bounds(margin);
        let proxy = proxy_for(body);
        let slot = handle.index();
        self.broadphase.set_proxy(slot, proxy);
        // Whatever was resting against the body where it stood has to be
        // re-examined without it.
        self.disturb(slot);
        true
    }

    /// Pull every body toward `center` from wherever it is, as a planet does,
    /// or straight down along `-Y` with `None`. The strength stays the
    /// configured `gravity`.
    pub fn set_gravity_center(&mut self, center: Option<[f32; 3]>) {
        self.config.gravity_center = center;
    }

    /// Carry every body into another frame: each position becomes
    /// `rotation * position + translation`, and orientations, velocities and
    /// drive targets turn with it. Nothing moves relative to anything else,
    /// so the step after behaves as if nothing had happened; a sleeping body
    /// stays asleep.
    ///
    /// Height grids cannot turn: they keep their place, and the caller
    /// replaces them ([`Simulation::replace_heightfield`]) with grids built in
    /// the new frame.
    pub fn rebase(&mut self, rotation: [f32; 4], translation: [f32; 3]) {
        let q = Quat::from_xyzw(rotation).normalize();
        let t = Vec3::from_array(translation);
        let center = self.config.gravity_center;
        let margin = self.config.bounds_margin;
        let Self {
            bodies, broadphase, ..
        } = self;
        for (handle, body) in bodies.iter_mut() {
            if body.convex().is_none() {
                continue;
            }
            body.position = q.rotate(body.position) + t;
            body.orientation = q.mul(body.orientation).normalize();
            body.linear_velocity = q.rotate(body.linear_velocity);
            body.angular_velocity = q.rotate(body.angular_velocity);
            body.kinematic_target = body.kinematic_target.map(|k| q.rotate(k) + t);
            body.bounds = body.tight_bounds().expanded(margin);
            broadphase.set_proxy(u32::saturating_from(handle.index()), proxy_for(body));
        }
        self.config.gravity_center = center.map(|c| (q.rotate(Vec3::from_array(c)) + t).to_array());
    }

    /// Put a new height grid in place of the one `handle` stands for, at
    /// `pos`: the same body, so whatever it was configured with stays.
    /// Returns whether `handle` named a height grid and the new one is valid.
    ///
    /// Whatever rested on the old grid is woken to settle on the new one.
    pub fn replace_heightfield(
        &mut self,
        handle: BodyHandle,
        rows: usize,
        cols: usize,
        heights: &[f32],
        scale: [f32; 3],
        pos: [f32; 3],
    ) -> bool {
        let margin = self.config.bounds_margin;
        let Some(index) = self
            .bodies
            .get(pool_handle(handle))
            .and_then(Body::terrain_index)
        else {
            return false;
        };
        let origin = Vec3::from_array(pos);
        let Some(field) = self.fields.get_mut(index) else {
            return false;
        };
        if !field.refill(rows, cols, heights, Vec3::from_array(scale), origin) {
            return false;
        }
        let bounds = field.bounds();
        let Some(body) = self.bodies.get_mut(pool_handle(handle)) else {
            return false;
        };
        *body = Body::terrain(index, bounds, origin, body.friction, body.mask);
        body.refresh_bounds(margin);
        let proxy = proxy_for(body);
        let slot = handle.index();
        self.broadphase.set_proxy(slot, proxy);
        self.disturb(slot);
        true
    }

    /// Switch a body to position-driven control, keeping its handle and the
    /// mass it was authored with. Returns whether the handle named a live
    /// body.
    pub fn make_kinematic(&mut self, handle: BodyHandle) -> bool {
        self.reclassify(handle, Body::make_kinematic)
    }

    /// Hand a body back to the solver with a launch velocity, restoring the
    /// mass it was authored with. Returns whether the handle named a live
    /// body.
    pub fn make_dynamic(&mut self, handle: BodyHandle, linear_velocity: [f32; 3]) -> bool {
        let velocity = Vec3::from_array(linear_velocity);
        self.reclassify(handle, move |body| body.make_dynamic(velocity))
    }

    /// Put a dynamic body to sleep where it stands. It stays put, gravity
    /// included, until something strikes it or [`wake_body`](Self::wake_body)
    /// wakes it. Returns whether the handle named a dynamic body.
    pub fn sleep_body(&mut self, handle: BodyHandle) -> bool {
        let time_to_sleep = self.config.time_to_sleep;
        let Some(body) = self.bodies.get_mut(pool_handle(handle)) else {
            return false;
        };
        if !body.is_dynamic() {
            return false;
        }
        body.sleep();
        // Already settled as far as the island pass is concerned, so the next
        // step keeps it down instead of waking it for a timer still at zero.
        body.sleep_timer = time_to_sleep;
        true
    }

    /// Wake a sleeping body so the simulation moves it again. Returns whether
    /// the handle named a live body.
    pub fn wake_body(&mut self, handle: BodyHandle) -> bool {
        let Some(body) = self.bodies.get_mut(pool_handle(handle)) else {
            return false;
        };
        body.wake();
        true
    }

    /// Whether a body is driven by position rather than by forces.
    #[cfg(test)]
    pub(crate) fn is_kinematic(&self, handle: BodyHandle) -> Option<bool> {
        Some(self.bodies.get(pool_handle(handle))?.is_kinematic())
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
        self.disturb(slot);
        self.joints.remove_incident(slot);
        true
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
    /// Apply an impulse through a body's center of mass, waking it.
    pub(crate) fn apply_impulse(&mut self, handle: BodyHandle, impulse: [f32; 3]) {
        if let Some(body) = self.bodies.get_mut(pool_handle(handle)) {
            body.linear_velocity += Vec3::from_array(impulse) * body.inv_mass;
            body.wake();
        }
    }

    /// Change a body's kind and put the broad phase and its neighbors back
    /// in step with the change.
    pub(super) fn reclassify(
        &mut self,
        handle: BodyHandle,
        change: impl FnOnce(&mut Body) -> bool,
    ) -> bool {
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
        self.disturb(slot);
        true
    }

    /// Give every driven body the velocity that carries it to its target over
    /// this step, so the solver sees the motion as motion.
    pub(super) fn drive_kinematics(&mut self, dt: f32) {
        for (_, body) in self.bodies.iter_mut() {
            if body.is_kinematic() {
                body.drive_to_target(dt);
            }
        }
    }
}

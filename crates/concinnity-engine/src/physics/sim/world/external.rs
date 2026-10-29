// What acts on a body from outside the solver between steps: its state read
// out, the water pressing on it, a push, and its velocity set outright.

use super::{Simulation, pool_handle};
use crate::physics::sim::math::Vec3;
use crate::physics::{BodyHandle, BodyState, FluidLoad};
use concinnity_core::math::SaturatingFrom;

impl Simulation {
    /// A dynamic body's state; `None` for any other body or a stale handle.
    pub fn body_state(&self, handle: BodyHandle) -> Option<BodyState> {
        let body = self.bodies.get(pool_handle(handle))?;
        if !body.is_dynamic() {
            return None;
        }
        Some(BodyState {
            position: body.position.to_array(),
            rotation: body.orientation.to_xyzw(),
            linear_velocity: body.linear_velocity.to_array(),
            angular_velocity: body.angular_velocity.to_array(),
            mass: body.mass,
            inertia: body.inertia_local.to_array(),
            sleeping: body.sleeping,
        })
    }

    /// The acceleration gravity gives a body at `point`, before its gravity
    /// scale: toward the planet's center on a planet, down `-Y` otherwise.
    pub fn gravity_at(&self, point: [f32; 3]) -> [f32; 3] {
        self.config.gravity_at(Vec3::from_array(point)).to_array()
    }

    /// Reserve room for `count` water loads a step, so setting them never
    /// allocates. A load past the reservation is refused.
    pub fn reserve_fluid_loads(&mut self, count: usize) {
        self.fluids
            .reserve_exact(count.saturating_sub(self.fluids.capacity()));
    }

    /// Have the water press on a dynamic body through the next step, as
    /// `load` describes: each substep reads how much of each piece lies under
    /// its water level at the body's pose then, and pushes it up out of the
    /// water against gravity by the water it displaces, its gravity scale
    /// included, turning it where the pieces sit unevenly. Setting a body's
    /// load again before the step replaces it. A sleeping body wakes; an
    /// awake one keeps its count toward sleep, so water that holds it still
    /// lets it settle. Returns whether the load was set: `false` for a body
    /// that is not dynamic, or with no room left in the reservation.
    ///
    /// # Examples
    ///
    /// ```
    /// use concinnity_engine::physics::{
    ///     ColliderShape, DynamicParams, FluidLoad, FluidPiece, FluidShape, LayerMask,
    ///     MAX_FLUID_PIECES, Simulation,
    /// };
    ///
    /// let mut sim = Simulation::with_capacity(1);
    /// sim.reserve_fluid_loads(1);
    /// let shape = ColliderShape::Cuboid { half_extents: [0.5; 3] };
    /// let params = DynamicParams { mass: 500.0, ..DynamicParams::default() };
    /// let raft = sim.add_dynamic(&shape, [0.0; 3], [0.0; 3], params, LayerMask::ALL).unwrap();
    /// // Water a thousand kilograms a cubic meter up to the raft's middle:
    /// // half submerged, it displaces its own weight and stays put.
    /// let piece = FluidPiece {
    ///     offset: [0.0; 3],
    ///     volume: 1.0,
    ///     shape: FluidShape::Cell { half_extents: [0.5; 3] },
    /// };
    /// let load = FluidLoad {
    ///     pieces: [piece; MAX_FLUID_PIECES],
    ///     count: 1,
    ///     levels: [[0.0; 3]; MAX_FLUID_PIECES],
    ///     up: [0.0, 1.0, 0.0],
    ///     density: 1000.0,
    /// };
    /// assert!(sim.set_fluid_load(raft, &load));
    /// sim.step(1.0 / 60.0);
    /// let state = sim.body_state(raft).unwrap();
    /// assert!(state.position[1].abs() < 1e-5 && state.linear_velocity[1].abs() < 1e-4);
    /// ```
    pub fn set_fluid_load(&mut self, handle: BodyHandle, load: &FluidLoad) -> bool {
        let Some(body) = self.bodies.get_mut(pool_handle(handle)) else {
            return false;
        };
        if !body.is_dynamic() {
            return false;
        }
        let at = body.fluid as usize;
        if at < self.fluids.len() {
            self.fluids[at] = *load;
        } else if self.fluids.len() < self.fluids.capacity() {
            body.fluid = u32::saturating_from(self.fluids.len());
            self.fluids.push(*load);
        } else {
            return false;
        }
        if body.sleeping {
            body.wake();
        }
        true
    }

    /// Push a dynamic body through the next step: `force` through its center
    /// of mass, in newtons, and `torque` about it, in newton meters, both in
    /// world axes. Each substep of the step applies them, as it does gravity,
    /// so contact and friction answer the push as it builds. Pushes set again
    /// before the step add up; the step clears them. A sleeping body wakes
    /// for a push that is not zero. Returns whether the handle named a
    /// dynamic body.
    ///
    /// # Examples
    ///
    /// ```
    /// use concinnity_engine::physics::{
    ///     ColliderShape, DynamicParams, LayerMask, SimConfig, Simulation,
    /// };
    ///
    /// let mut sim = Simulation::new(SimConfig { gravity: 0.0, ..SimConfig::default() }, 1);
    /// let shape = ColliderShape::Ball { radius: 0.5 };
    /// let ball = sim.add_dynamic(&shape, [0.0; 3], [0.0; 3], DynamicParams::default(), LayerMask::ALL).unwrap();
    /// // Ten newtons on a one-kilogram ball for a tenth of a second.
    /// assert!(sim.add_force(ball, [10.0, 0.0, 0.0], [0.0; 3]));
    /// sim.step(0.1);
    /// assert!((sim.body_state(ball).unwrap().linear_velocity[0] - 1.0).abs() < 1e-4);
    /// ```
    pub fn add_force(&mut self, handle: BodyHandle, force: [f32; 3], torque: [f32; 3]) -> bool {
        let Some(body) = self.bodies.get_mut(pool_handle(handle)) else {
            return false;
        };
        if !body.is_dynamic() {
            return false;
        }
        let (force, torque) = (Vec3::from_array(force), Vec3::from_array(torque));
        body.force += force;
        body.torque += torque;
        if body.sleeping && (force != Vec3::ZERO || torque != Vec3::ZERO) {
            body.wake();
        }
        true
    }

    /// Strike a dynamic body at once: `linear` through its center of mass,
    /// in newton seconds, and `angular` about it, in newton meter seconds,
    /// both in world axes. Its velocity changes before the next step. A
    /// sleeping body wakes. Returns whether the handle named a dynamic body.
    pub fn apply_impulse(
        &mut self,
        handle: BodyHandle,
        linear: [f32; 3],
        angular: [f32; 3],
    ) -> bool {
        let Some(body) = self.bodies.get_mut(pool_handle(handle)) else {
            return false;
        };
        if !body.is_dynamic() {
            return false;
        }
        body.linear_velocity += Vec3::from_array(linear) * body.inv_mass;
        body.angular_velocity += body.inv_inertia_world().mul_vec3(Vec3::from_array(angular));
        if body.sleeping {
            body.wake();
        }
        true
    }

    /// Set a dynamic body's linear and angular velocity outright. A sleeping
    /// body wakes; an awake one keeps its count toward sleep. Returns whether
    /// the handle named a dynamic body.
    pub fn set_body_velocity(
        &mut self,
        handle: BodyHandle,
        linear: [f32; 3],
        angular: [f32; 3],
    ) -> bool {
        let Some(body) = self.bodies.get_mut(pool_handle(handle)) else {
            return false;
        };
        if !body.is_dynamic() {
            return false;
        }
        body.linear_velocity = Vec3::from_array(linear);
        body.angular_velocity = Vec3::from_array(angular);
        if body.sleeping {
            body.wake();
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::physics::{
        ColliderShape, DynamicParams, FluidLoad, FluidPiece, FluidShape, LayerMask,
        MAX_FLUID_PIECES, SimConfig, Simulation,
    };

    fn ball(sim: &mut Simulation) -> crate::physics::BodyHandle {
        sim.add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 3.0, 0.0],
            [0.0; 3],
            DynamicParams {
                mass: 2.0,
                ..DynamicParams::default()
            },
            LayerMask::ALL,
        )
        .unwrap()
    }

    fn half_full(volume: f32) -> FluidLoad {
        let piece = FluidPiece {
            offset: [0.0; 3],
            volume,
            shape: FluidShape::Ball { radius: 0.5 },
        };
        FluidLoad {
            pieces: [piece; MAX_FLUID_PIECES],
            count: 1,
            levels: [[0.0, 3.0, 0.0]; MAX_FLUID_PIECES],
            up: [0.0, 1.0, 0.0],
            density: 1000.0,
        }
    }

    // A load wakes a sleeping body and lasts one step; past the reservation
    // it is refused, and setting it again replaces it.
    #[test]
    fn a_load_lasts_one_step() {
        let mut sim = Simulation::new(SimConfig::default(), 2);
        let b = ball(&mut sim);
        let volume = 4.0 / 3.0 * core::f32::consts::PI * 0.125;
        assert!(
            !sim.set_fluid_load(b, &half_full(volume)),
            "nothing reserved"
        );
        sim.reserve_fluid_loads(1);
        sim.sleep_body(b);
        assert!(sim.set_fluid_load(b, &half_full(volume)));
        assert!(
            sim.set_fluid_load(b, &half_full(volume)),
            "replaced, not added"
        );
        assert!(!sim.body_state(b).unwrap().sleeping);
        // Half under, the 2 kg ball displaces far more than its weight of
        // water: it rises.
        sim.step(1.0 / 60.0);
        assert!(sim.body_state(b).unwrap().linear_velocity[1] > 0.0);
        // The load is gone: next step it only falls.
        let before = sim.body_state(b).unwrap().linear_velocity[1];
        sim.step(1.0 / 60.0);
        assert!(sim.body_state(b).unwrap().linear_velocity[1] < before);
    }

    // A push lasts one step and adds up before it; an impulse lands at once.
    // Torque and an angular impulse turn a ball by its inertia, 2/5 m r^2.
    #[test]
    fn pushes_and_impulses_move_a_body_by_its_mass() {
        let mut sim = Simulation::new(
            SimConfig {
                gravity: 0.0,
                ..SimConfig::default()
            },
            1,
        );
        let b = ball(&mut sim);
        let inertia = 0.4 * 2.0 * 0.25;
        assert!(sim.add_force(b, [1.0, 0.0, 0.0], [0.0, inertia, 0.0]));
        assert!(sim.add_force(b, [1.0, 0.0, 0.0], [0.0; 3]), "adds up");
        sim.step(0.5);
        let s = sim.body_state(b).unwrap();
        assert!((s.linear_velocity[0] - 0.5).abs() < 1e-4, "{s:?}");
        assert!((s.angular_velocity[1] - 0.5).abs() < 1e-4, "{s:?}");
        sim.step(0.5);
        let after = sim.body_state(b).unwrap();
        assert_eq!(
            after.linear_velocity, s.linear_velocity,
            "cleared by the step"
        );

        sim.sleep_body(b);
        assert!(sim.apply_impulse(b, [0.0, 4.0, 0.0], [inertia, 0.0, 0.0]));
        let hit = sim.body_state(b).unwrap();
        assert!(!hit.sleeping);
        assert!((hit.linear_velocity[1] - 2.0).abs() < 1e-5);
        assert!((hit.angular_velocity[0] - 1.0).abs() < 1e-5);
    }

    // Only a dynamic body is pushed or read; its velocity can be set.
    #[test]
    fn only_dynamic_bodies_take_outside_pushes() {
        let mut sim = Simulation::with_capacity(2);
        let wall = sim
            .add_fixed(
                &ColliderShape::Ball { radius: 1.0 },
                [0.0; 3],
                [0.0; 3],
                0.5,
                LayerMask::ALL,
            )
            .unwrap();
        assert!(sim.body_state(wall).is_none());
        sim.reserve_fluid_loads(1);
        assert!(!sim.set_fluid_load(wall, &half_full(1.0)));
        assert!(!sim.set_body_velocity(wall, [1.0; 3], [0.0; 3]));
        assert!(!sim.add_force(wall, [1.0; 3], [1.0; 3]));
        assert!(!sim.apply_impulse(wall, [1.0; 3], [1.0; 3]));
        let b = ball(&mut sim);
        assert!(sim.set_body_velocity(b, [1.0, 2.0, 3.0], [0.0, 0.5, 0.0]));
        let s = sim.body_state(b).unwrap();
        assert_eq!(
            (s.linear_velocity, s.angular_velocity),
            ([1.0, 2.0, 3.0], [0.0, 0.5, 0.0])
        );
        assert_eq!(
            sim.gravity_at([0.0; 3]),
            [0.0, -crate::physics::GRAVITY, 0.0]
        );
    }
}

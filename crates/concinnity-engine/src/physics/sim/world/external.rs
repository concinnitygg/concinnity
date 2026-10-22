// What acts on a body from outside the solver between steps: its state read
// out, the water pressing on it, and its velocity set outright.

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

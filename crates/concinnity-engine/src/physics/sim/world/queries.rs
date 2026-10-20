// Ray and shape queries, character moves, and per-body readouts.

use super::{Simulation, pool_handle};
use crate::physics::sim::character::{self, CharacterCapsule};
#[cfg(test)]
use crate::physics::sim::math::Mat3;
use crate::physics::sim::query::{self, RayQuery};
#[cfg(test)]
use crate::physics::sim::query::{ShapeCast, ShapeCastHit};
use crate::physics::sim::scene::Scene;
use crate::physics::{BodyHandle, CharacterMove, CharacterMoveInput, LayerMask, RayHit};

impl Simulation {
    /// What a query reads: the bodies, the sweep order, and the height grids.
    pub(super) fn scene(&self) -> Scene<'_> {
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
    /// use concinnity_engine::physics::{CharacterMoveInput, ColliderShape, LayerMask, Simulation};
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
        character::resolve(self.scene(), &self.character, *shape, input)
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
}

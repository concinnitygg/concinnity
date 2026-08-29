// One body's state: what it looks like, where it is, how it moves, and how
// hard it is to move. It is a single struct rather than parallel arrays
// because the pool the bodies live in hands out one slot per body; the arrays
// the solver wants are gathered from it each step instead.
//
// Bounds are cached fat rather than recomputed tight every step, so a body
// that shifts a little inside its stored bounds costs the broad phase nothing.
//
// A body keeps the mass it was authored with even while it is standing still
// under position control, which is what lets it be handed back to the solver
// later without the caller re-stating what it weighs.
//
// A sensor is a body too, rather than a thing beside one. It has a shape, a
// pose and a layer mask like everything else, and the one thing that sets it
// apart -- the caller's tag -- is the field that says so.

use crate::physics::{ColliderShape, DynamicParams, LayerMask};

use super::aabb::{Aabb, shape_bounds};
use super::mass::MassProperties;
use super::math::{Mat3, Quat, Vec3};

/// How a body responds to being pushed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BodyKind {
    /// Immovable: infinite mass, never integrated.
    Fixed,
    /// Freely simulated.
    Dynamic,
    /// Driven to a caller-set position: infinite mass, unmoved by gravity or
    /// impulses, but it pushes whatever it is driven into.
    Kinematic,
}

/// What a body is made of.
///
/// Terrain is not a convex primitive and cannot be turned into one, so a body
/// standing for a height grid holds an index into the simulation's grid table
/// instead of a shape. It also holds the bounds that grid covers, because
/// terrain is fixed and unrotated and so its bounds are settled once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum BodyShape {
    /// A convex primitive in the body's own frame.
    Convex(ColliderShape),
    /// A height grid, by the index the table stored it at.
    Terrain { index: u32, bounds: Aabb },
}

#[derive(Debug, Clone)]
pub(crate) struct Body {
    pub(crate) shape: BodyShape,
    pub(crate) kind: BodyKind,
    pub(crate) mask: LayerMask,

    pub(crate) position: Vec3,
    pub(crate) orientation: Quat,
    pub(crate) linear_velocity: Vec3,
    pub(crate) angular_velocity: Vec3,

    pub(crate) mass: f32,
    pub(crate) inv_mass: f32,
    pub(crate) inertia_local: Vec3,
    pub(crate) inv_inertia_local: Vec3,

    /// The mass this body was authored with, kept through a spell under
    /// position control so switching back needs no caller to restate it.
    /// `0.0` derives the mass from the shape's volume.
    authored_mass: f32,

    pub(crate) friction: f32,
    pub(crate) restitution: f32,
    pub(crate) gravity_scale: f32,
    pub(crate) damping: f32,

    /// Where a position-driven body is to be by the end of the next step.
    pub(crate) kinematic_target: Option<Vec3>,

    /// The caller's tag when this body is a sensor region, which resists
    /// nothing and records what overlaps it instead.
    sensor: Option<u64>,

    pub(crate) bounds: Aabb,
    pub(crate) sleep_timer: f32,
    pub(crate) sleeping: bool,
}

impl Body {
    pub(crate) fn fixed(
        shape: ColliderShape,
        position: Vec3,
        orientation: Quat,
        friction: f32,
        mask: LayerMask,
    ) -> Self {
        Body {
            shape: BodyShape::Convex(shape),
            kind: BodyKind::Fixed,
            mask,
            position,
            orientation,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: 0.0,
            inv_mass: 0.0,
            inertia_local: Vec3::ZERO,
            inv_inertia_local: Vec3::ZERO,
            authored_mass: 0.0,
            friction: friction.max(0.0),
            restitution: 0.0,
            // Read only once the body is freely simulated, so that a body
            // handed to the solver later falls at the world's rate.
            gravity_scale: 1.0,
            damping: 0.0,
            kinematic_target: None,
            sensor: None,
            bounds: Aabb::EMPTY,
            sleep_timer: 0.0,
            sleeping: false,
        }
    }

    /// A body driven to a position rather than by forces. Infinite mass, so
    /// contact moves whatever it meets and never the body itself.
    pub(crate) fn kinematic(
        shape: ColliderShape,
        position: Vec3,
        orientation: Quat,
        friction: f32,
        mask: LayerMask,
    ) -> Self {
        Body {
            kind: BodyKind::Kinematic,
            ..Body::fixed(shape, position, orientation, friction, mask)
        }
    }

    /// A region that records what overlaps it and resists nothing. Immovable
    /// like a fixed body, but excluded from the narrow phase and from every
    /// query, so nothing ever leans on it or stops against it.
    pub(crate) fn sensor(
        shape: ColliderShape,
        position: Vec3,
        orientation: Quat,
        tag: u64,
        mask: LayerMask,
    ) -> Self {
        Body {
            sensor: Some(tag),
            ..Body::fixed(shape, position, orientation, 0.0, mask)
        }
    }

    /// An immovable height grid. It never moves and never rotates, so the
    /// bounds it covers are given once and kept.
    pub(crate) fn terrain(
        index: u32,
        bounds: Aabb,
        position: Vec3,
        friction: f32,
        mask: LayerMask,
    ) -> Self {
        Body {
            shape: BodyShape::Terrain { index, bounds },
            ..Body::fixed(
                ColliderShape::Ball { radius: 0.0 },
                position,
                Quat::IDENTITY,
                friction,
                mask,
            )
        }
    }

    pub(crate) fn dynamic(
        shape: ColliderShape,
        position: Vec3,
        orientation: Quat,
        params: DynamicParams,
        mask: LayerMask,
    ) -> Self {
        let properties = MassProperties::for_shape(&shape, params.mass);
        Body {
            shape: BodyShape::Convex(shape),
            kind: BodyKind::Dynamic,
            mask,
            position,
            orientation,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass: properties.mass,
            inv_mass: 1.0 / properties.mass,
            inertia_local: properties.inertia,
            inv_inertia_local: inverse_inertia(properties.inertia),
            authored_mass: params.mass,
            friction: params.friction.max(0.0),
            restitution: params.restitution.clamp(0.0, 1.0),
            gravity_scale: params.gravity_scale,
            damping: params.linear_damping.max(0.0),
            kinematic_target: None,
            sensor: None,
            bounds: Aabb::EMPTY,
            sleep_timer: 0.0,
            sleeping: false,
        }
    }

    /// The convex primitive this body is, or `None` for terrain.
    pub(crate) fn convex(&self) -> Option<&ColliderShape> {
        match &self.shape {
            BodyShape::Convex(shape) => Some(shape),
            BodyShape::Terrain { .. } => None,
        }
    }

    /// The height grid this body stands for, or `None` for a convex one.
    pub(crate) fn terrain_index(&self) -> Option<u32> {
        match self.shape {
            BodyShape::Terrain { index, .. } => Some(index),
            BodyShape::Convex(_) => None,
        }
    }

    pub(crate) fn is_dynamic(&self) -> bool {
        self.kind == BodyKind::Dynamic
    }

    pub(crate) fn is_kinematic(&self) -> bool {
        self.kind == BodyKind::Kinematic
    }

    /// The caller's tag, for a body that is a sensor region.
    pub(crate) fn sensor_tag(&self) -> Option<u64> {
        self.sensor
    }

    pub(crate) fn is_sensor(&self) -> bool {
        self.sensor.is_some()
    }

    /// Whether contact can change this body's motion. A pair where neither
    /// side responds is never worth reporting.
    pub(crate) fn responds_to_contact(&self) -> bool {
        self.is_dynamic()
    }

    /// Whether the solver moves this body at all this step. A position-driven
    /// body counts only while it is being driven, so a parked platform costs
    /// the step no more than a wall does.
    pub(crate) fn is_simulated(&self) -> bool {
        match self.kind {
            BodyKind::Fixed => false,
            BodyKind::Dynamic => !self.sleeping,
            BodyKind::Kinematic => self.kinematic_target.is_some(),
        }
    }

    /// Hand the body to the solver, giving it a launch velocity. Returns
    /// whether the kind actually changed.
    pub(crate) fn make_dynamic(&mut self, linear_velocity: Vec3) -> bool {
        let Some(shape) = self.convex().copied() else {
            // Terrain is the world, not something in it.
            return false;
        };
        if self.is_sensor() {
            // A region that resists nothing has nothing to hand the solver.
            return false;
        }
        let changed = self.kind != BodyKind::Dynamic;
        let properties = MassProperties::for_shape(&shape, self.authored_mass);
        self.kind = BodyKind::Dynamic;
        self.mass = properties.mass;
        self.inv_mass = 1.0 / properties.mass;
        self.inertia_local = properties.inertia;
        self.inv_inertia_local = inverse_inertia(properties.inertia);
        self.kinematic_target = None;
        self.linear_velocity = linear_velocity;
        self.angular_velocity = Vec3::ZERO;
        self.wake();
        changed
    }

    /// Take the body out of the solver's hands and drive it by position.
    /// Returns whether the kind actually changed.
    pub(crate) fn make_kinematic(&mut self) -> bool {
        if self.convex().is_none() || self.is_sensor() {
            return false;
        }
        let changed = self.kind != BodyKind::Kinematic;
        self.kind = BodyKind::Kinematic;
        self.mass = 0.0;
        self.inv_mass = 0.0;
        self.inertia_local = Vec3::ZERO;
        self.inv_inertia_local = Vec3::ZERO;
        self.kinematic_target = None;
        self.linear_velocity = Vec3::ZERO;
        self.angular_velocity = Vec3::ZERO;
        self.wake();
        changed
    }

    /// Set the velocity a driven body needs to reach its target over `dt`,
    /// or park it where it stands when nothing is driving it.
    pub(crate) fn drive_to_target(&mut self, dt: f32) {
        self.linear_velocity = match self.kinematic_target {
            Some(target) => (target - self.position) / dt,
            None => Vec3::ZERO,
        };
        self.angular_velocity = Vec3::ZERO;
    }

    pub(crate) fn inv_inertia_world(&self) -> Mat3 {
        if self.inv_inertia_local == Vec3::ZERO {
            return Mat3::ZERO;
        }
        Mat3::diagonal_conjugated(self.orientation, self.inv_inertia_local)
    }

    pub(crate) fn wake(&mut self) {
        self.sleeping = false;
        self.sleep_timer = 0.0;
    }

    pub(crate) fn sleep(&mut self) {
        self.sleeping = true;
        self.linear_velocity = Vec3::ZERO;
        self.angular_velocity = Vec3::ZERO;
    }

    /// Whether the body is moving slowly enough to be a sleep candidate.
    pub(crate) fn is_still(&self, linear: f32, angular: f32) -> bool {
        self.linear_velocity.length_squared() <= linear * linear
            && self.angular_velocity.length_squared() <= angular * angular
    }

    /// World-space bounds of the shape at the current pose, with no margin.
    /// A driven body's bounds also cover where it has been told to go, so the
    /// broad phase reports the pair before the move rather than after it.
    pub(crate) fn tight_bounds(&self) -> Aabb {
        let shape = match &self.shape {
            BodyShape::Convex(shape) => shape,
            BodyShape::Terrain { bounds, .. } => return *bounds,
        };
        let here = shape_bounds(shape, self.position, self.orientation);
        match self.kinematic_target {
            Some(target) => here.union(shape_bounds(shape, target, self.orientation)),
            None => here,
        }
    }

    /// Re-fatten the cached bounds when the body has moved out of them.
    /// Returns whether they were rebuilt.
    pub(crate) fn refresh_bounds(&mut self, margin: f32) -> bool {
        let tight = self.tight_bounds();
        if self.bounds.contains(tight) {
            return false;
        }
        self.bounds = tight.expanded(margin);
        true
    }
}

fn inverse_inertia(inertia: Vec3) -> Vec3 {
    let inverse = |i: f32| if i > 0.0 { 1.0 / i } else { 0.0 };
    Vec3::from_array([inverse(inertia.x), inverse(inertia.y), inverse(inertia.z)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::sqrt;
    use crate::physics::sim::math::vec3;

    fn params() -> DynamicParams {
        DynamicParams {
            mass: 0.0,
            friction: 0.5,
            restitution: 0.2,
            gravity_scale: 1.0,
            linear_damping: 0.0,
        }
    }

    #[test]
    fn a_fixed_body_has_no_inverse_mass_and_never_simulates() {
        let b = Body::fixed(
            ColliderShape::Ball { radius: 1.0 },
            Vec3::ZERO,
            Quat::IDENTITY,
            0.5,
            LayerMask::ALL,
        );
        assert_eq!(b.inv_mass, 0.0);
        assert_eq!(b.inv_inertia_world(), Mat3::ZERO);
        assert!(!b.is_dynamic());
        assert!(!b.is_simulated());
        assert!(!b.responds_to_contact());
    }

    #[test]
    fn a_driven_body_weighs_nothing_and_simulates_only_while_it_is_driven() {
        let mut b = Body::kinematic(
            ColliderShape::Cuboid {
                half_extents: [1.0, 0.2, 1.0],
            },
            Vec3::ZERO,
            Quat::IDENTITY,
            0.5,
            LayerMask::ALL,
        );
        assert!(b.is_kinematic() && !b.is_dynamic());
        assert_eq!(b.inv_mass, 0.0);
        assert!(!b.responds_to_contact(), "contact must not move it");
        assert!(
            !b.is_simulated(),
            "a parked platform costs the step nothing"
        );

        b.kinematic_target = Some(vec3(0.0, 1.0, 0.0));
        assert!(b.is_simulated());
        b.drive_to_target(0.5);
        assert_eq!(b.linear_velocity, vec3(0.0, 2.0, 0.0));

        b.kinematic_target = None;
        b.drive_to_target(0.5);
        assert_eq!(b.linear_velocity, Vec3::ZERO);
    }

    // The whole point of keeping the authored mass: a body handed back to the
    // solver weighs what it always weighed, with no caller restating it.
    #[test]
    fn switching_kinds_restores_the_mass_the_body_was_authored_with() {
        let mut b = Body::dynamic(
            ColliderShape::Ball { radius: 0.5 },
            Vec3::ZERO,
            Quat::IDENTITY,
            DynamicParams {
                mass: 7.0,
                ..params()
            },
            LayerMask::ALL,
        );
        let (mass, inertia) = (b.mass, b.inertia_local);
        assert_eq!(mass, 7.0);

        assert!(b.make_kinematic());
        assert_eq!(b.mass, 0.0);
        assert_eq!(b.inv_mass, 0.0);
        assert_eq!(b.inv_inertia_world(), Mat3::ZERO);
        assert!(!b.make_kinematic(), "already driven by position");

        assert!(b.make_dynamic(vec3(0.0, 3.0, 0.0)));
        assert_eq!(b.mass, mass);
        assert_eq!(b.inertia_local, inertia);
        assert_eq!(b.linear_velocity, vec3(0.0, 3.0, 0.0));
        assert!(!b.sleeping);
    }

    // A body authored fixed has no authored mass, so the shape's volume is
    // what it weighs once something hands it to the solver.
    #[test]
    fn a_fixed_body_made_dynamic_takes_its_mass_from_its_shape() {
        let mut b = Body::fixed(
            ColliderShape::Cuboid {
                half_extents: [0.5, 0.5, 0.5],
            },
            Vec3::ZERO,
            Quat::IDENTITY,
            0.5,
            LayerMask::ALL,
        );
        assert!(b.make_dynamic(Vec3::ZERO));
        assert!(b.mass > 0.0 && b.mass.is_finite(), "{}", b.mass);
        assert_eq!(b.gravity_scale, 1.0, "it has to fall");
        assert!(b.is_simulated());
    }

    // A driven body's bounds have to cover where it is going, or the broad
    // phase reports the pair only once the move has already happened.
    #[test]
    fn a_driven_bodys_bounds_cover_the_move_it_was_told_to_make() {
        let mut b = Body::kinematic(
            ColliderShape::Ball { radius: 0.5 },
            Vec3::ZERO,
            Quat::IDENTITY,
            0.5,
            LayerMask::ALL,
        );
        b.kinematic_target = Some(vec3(4.0, 0.0, 0.0));
        let bounds = b.tight_bounds();
        assert_eq!(bounds.min, vec3(-0.5, -0.5, -0.5));
        assert_eq!(bounds.max, vec3(4.5, 0.5, 0.5));
    }

    #[test]
    fn a_dynamic_body_reports_a_finite_positive_inverse_mass() {
        let b = Body::dynamic(
            ColliderShape::Ball { radius: 0.5 },
            Vec3::ZERO,
            Quat::IDENTITY,
            params(),
            LayerMask::ALL,
        );
        assert!(b.inv_mass > 0.0 && b.inv_mass.is_finite());
        assert!(b.is_simulated());
        assert!(b.inv_inertia_world().mul_vec3(Vec3::X).x > 0.0);
    }

    #[test]
    fn restitution_and_friction_are_clamped_into_range() {
        let b = Body::dynamic(
            ColliderShape::Ball { radius: 0.5 },
            Vec3::ZERO,
            Quat::IDENTITY,
            DynamicParams {
                friction: -1.0,
                restitution: 3.0,
                linear_damping: -2.0,
                ..params()
            },
            LayerMask::ALL,
        );
        assert_eq!(b.friction, 0.0);
        assert_eq!(b.restitution, 1.0);
        assert_eq!(b.damping, 0.0);
    }

    #[test]
    fn an_unrotated_box_bounds_its_half_extents() {
        let b = Body::fixed(
            ColliderShape::Cuboid {
                half_extents: [1.0, 2.0, 3.0],
            },
            vec3(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            0.5,
            LayerMask::ALL,
        );
        let bounds = b.tight_bounds();
        assert_eq!(bounds.min, vec3(0.0, -2.0, -3.0));
        assert_eq!(bounds.max, vec3(2.0, 2.0, 3.0));
    }

    // A box turned 45 degrees about Y must bound wider than the box itself.
    #[test]
    fn a_rotated_box_bounds_its_swept_extent() {
        let b = Body::fixed(
            ColliderShape::Cuboid {
                half_extents: [1.0, 0.5, 1.0],
            },
            Vec3::ZERO,
            Quat::from_euler_deg([0.0, 45.0, 0.0]),
            0.5,
            LayerMask::ALL,
        );
        let bounds = b.tight_bounds();
        let expected = sqrt(2.0);
        assert!((bounds.max.x - expected).abs() < 1.0e-5, "{bounds:?}");
        assert!((bounds.max.y - 0.5).abs() < 1.0e-5, "{bounds:?}");
    }

    #[test]
    fn a_capsule_bounds_both_caps() {
        let b = Body::fixed(
            ColliderShape::Capsule {
                half_height: 1.0,
                radius: 0.25,
            },
            Vec3::ZERO,
            Quat::from_euler_deg([0.0, 0.0, 90.0]),
            0.5,
            LayerMask::ALL,
        );
        let bounds = b.tight_bounds();
        // Rolled onto its side, the capsule is long in x and thin in y.
        assert!((bounds.max.x - 1.25).abs() < 1.0e-5, "{bounds:?}");
        assert!((bounds.max.y - 0.25).abs() < 1.0e-5, "{bounds:?}");
    }

    // The fat bounds exist so small motion is free: moving inside them must
    // not rebuild them, and moving out must.
    #[test]
    fn bounds_are_rebuilt_only_when_the_body_leaves_them() {
        let mut b = Body::dynamic(
            ColliderShape::Ball { radius: 0.5 },
            Vec3::ZERO,
            Quat::IDENTITY,
            params(),
            LayerMask::ALL,
        );
        assert!(
            b.refresh_bounds(0.1),
            "the first refresh always builds them"
        );
        b.position = vec3(0.05, 0.0, 0.0);
        assert!(!b.refresh_bounds(0.1), "a small move stays inside");
        b.position = vec3(0.5, 0.0, 0.0);
        assert!(b.refresh_bounds(0.1), "a large move leaves them");
        assert!(b.bounds.contains(b.tight_bounds()));
    }

    // A region is immovable, weightless, and never handed to the solver,
    // whatever a caller asks of it afterwards.
    #[test]
    fn a_sensor_is_immovable_and_stays_that_way() {
        let mut b = Body::sensor(
            ColliderShape::Cuboid {
                half_extents: [1.0, 1.0, 1.0],
            },
            vec3(0.0, 2.0, 0.0),
            Quat::IDENTITY,
            7,
            LayerMask::ALL,
        );
        assert!(b.is_sensor());
        assert_eq!(b.sensor_tag(), Some(7));
        assert_eq!(b.inv_mass, 0.0);
        assert!(!b.is_simulated());
        assert!(!b.responds_to_contact());

        assert!(!b.make_dynamic(Vec3::ZERO), "a region resists nothing");
        assert!(!b.make_kinematic());
        assert!(b.is_sensor() && !b.is_dynamic() && !b.is_kinematic());
    }

    #[test]
    fn a_body_that_is_not_a_sensor_carries_no_tag() {
        let b = Body::fixed(
            ColliderShape::Ball { radius: 1.0 },
            Vec3::ZERO,
            Quat::IDENTITY,
            0.5,
            LayerMask::ALL,
        );
        assert!(!b.is_sensor());
        assert_eq!(b.sensor_tag(), None);
    }

    #[test]
    fn sleeping_stops_a_body_and_waking_clears_the_timer() {
        let mut b = Body::dynamic(
            ColliderShape::Ball { radius: 0.5 },
            Vec3::ZERO,
            Quat::IDENTITY,
            params(),
            LayerMask::ALL,
        );
        b.linear_velocity = vec3(1.0, 0.0, 0.0);
        assert!(!b.is_still(0.05, 0.1));
        b.sleep_timer = 0.9;
        b.sleep();
        assert!(b.sleeping);
        assert_eq!(b.linear_velocity, Vec3::ZERO);
        assert!(!b.is_simulated());
        assert!(b.is_still(0.05, 0.1));
        b.wake();
        assert!(!b.sleeping);
        assert_eq!(b.sleep_timer, 0.0);
    }
}

// concinnity-physics/src/sim/mass.rs
//
// Mass and rotational inertia from a shape. Both are derived at unit density
// and then rescaled when a body names an explicit mass, so overriding the mass
// changes how heavy a body is without changing how its weight is distributed.
//
// Every shape here has its principal axes on its local frame, so inertia is a
// diagonal; it only becomes a full tensor once rotated into world space.

use crate::ColliderShape;

use super::math::{Vec3, vec3};

/// Density used when a body does not name a mass, matching the one-kilogram
/// per cubic unit convention the engine's authored densities assume.
const UNIT_DENSITY: f32 = 1.0;

/// The smallest mass a dynamic body may end up with, so a degenerate shape
/// cannot divide the solver by zero.
const MIN_MASS: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MassProperties {
    pub(crate) mass: f32,
    /// Principal moments about the local x, y, z axes.
    pub(crate) inertia: Vec3,
}

impl MassProperties {
    /// Mass properties for `shape`, overridden to `mass` when that is
    /// positive. The inertia scales with the override so the distribution the
    /// shape implies is preserved.
    pub(crate) fn for_shape(shape: &ColliderShape, mass: f32) -> Self {
        let derived = Self::at_unit_density(shape);
        if mass > 0.0 {
            let scale = mass / derived.mass.max(MIN_MASS);
            return MassProperties {
                mass: mass.max(MIN_MASS),
                inertia: derived.inertia * scale,
            };
        }
        MassProperties {
            mass: derived.mass.max(MIN_MASS),
            inertia: derived.inertia,
        }
    }

    fn at_unit_density(shape: &ColliderShape) -> Self {
        match *shape {
            ColliderShape::Cuboid { half_extents } => {
                let h = Vec3::from_array(half_extents).abs();
                let mass = UNIT_DENSITY * 8.0 * h.x * h.y * h.z;
                let third = mass / 3.0;
                MassProperties {
                    mass,
                    inertia: vec3(
                        third * (h.y * h.y + h.z * h.z),
                        third * (h.x * h.x + h.z * h.z),
                        third * (h.x * h.x + h.y * h.y),
                    ),
                }
            }
            ColliderShape::Ball { radius } => {
                let r = libm::fabsf(radius);
                let mass = UNIT_DENSITY * (4.0 / 3.0) * core::f32::consts::PI * r * r * r;
                MassProperties {
                    mass,
                    inertia: Vec3::splat(0.4 * mass * r * r),
                }
            }
            ColliderShape::Capsule {
                half_height,
                radius,
            } => {
                let r = libm::fabsf(radius);
                let h = libm::fabsf(half_height);
                let pi = core::f32::consts::PI;
                let cylinder = UNIT_DENSITY * pi * r * r * (2.0 * h);
                let caps = UNIT_DENSITY * (4.0 / 3.0) * pi * r * r * r;
                let mass = cylinder + caps;
                // The caps are two hemispheres shifted to y = +-h; the shift
                // uses the hemisphere centroid at 3r/8 from its flat face.
                let transverse = cylinder * (0.25 * r * r + h * h / 3.0)
                    + caps * (0.4 * r * r + h * h + 0.75 * h * r);
                let axial = 0.5 * cylinder * r * r + 0.4 * caps * r * r;
                MassProperties {
                    mass,
                    inertia: vec3(transverse, axial, transverse),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        libm::fabsf(a - b) <= 1.0e-4 * libm::fabsf(b).max(1.0)
    }

    #[test]
    fn a_unit_cube_weighs_its_volume_and_matches_the_textbook_inertia() {
        let p = MassProperties::for_shape(
            &ColliderShape::Cuboid {
                half_extents: [0.5, 0.5, 0.5],
            },
            0.0,
        );
        assert!(close(p.mass, 1.0), "{}", p.mass);
        // m/12 * (w^2 + d^2) with w = d = 1.
        assert!(close(p.inertia.x, 1.0 / 6.0), "{:?}", p.inertia);
        assert!(close(p.inertia.y, 1.0 / 6.0));
        assert!(close(p.inertia.z, 1.0 / 6.0));
    }

    #[test]
    fn an_oblong_box_is_hardest_to_turn_about_its_short_axis() {
        let p = MassProperties::for_shape(
            &ColliderShape::Cuboid {
                half_extents: [2.0, 0.25, 0.25],
            },
            0.0,
        );
        assert!(
            p.inertia.y > p.inertia.x && p.inertia.z > p.inertia.x,
            "{:?}",
            p.inertia
        );
        assert!(close(p.inertia.y, p.inertia.z));
    }

    #[test]
    fn a_ball_is_isotropic_and_matches_two_fifths_m_r_squared() {
        let p = MassProperties::for_shape(&ColliderShape::Ball { radius: 2.0 }, 0.0);
        let expected_mass = (4.0 / 3.0) * core::f32::consts::PI * 8.0;
        assert!(close(p.mass, expected_mass), "{}", p.mass);
        let expected = 0.4 * expected_mass * 4.0;
        assert!(close(p.inertia.x, expected), "{:?}", p.inertia);
        assert_eq!(p.inertia.x, p.inertia.y);
        assert_eq!(p.inertia.y, p.inertia.z);
    }

    // A capsule of zero cylinder length is a sphere, so its properties must
    // collapse onto the sphere's.
    #[test]
    fn a_zero_length_capsule_matches_a_ball() {
        let capsule = MassProperties::for_shape(
            &ColliderShape::Capsule {
                half_height: 0.0,
                radius: 0.7,
            },
            0.0,
        );
        let ball = MassProperties::for_shape(&ColliderShape::Ball { radius: 0.7 }, 0.0);
        assert!(close(capsule.mass, ball.mass));
        assert!(
            close(capsule.inertia.x, ball.inertia.x),
            "{:?}",
            capsule.inertia
        );
        assert!(close(capsule.inertia.y, ball.inertia.y));
    }

    #[test]
    fn a_long_capsule_turns_easiest_about_its_own_axis() {
        let p = MassProperties::for_shape(
            &ColliderShape::Capsule {
                half_height: 1.0,
                radius: 0.2,
            },
            0.0,
        );
        assert!(p.inertia.y < p.inertia.x, "{:?}", p.inertia);
        assert!(close(p.inertia.x, p.inertia.z));
    }

    // An explicit mass rescales the tensor rather than replacing it: the ratio
    // between axes is a property of the shape.
    #[test]
    fn an_explicit_mass_scales_inertia_without_reshaping_it() {
        let shape = ColliderShape::Cuboid {
            half_extents: [2.0, 0.5, 0.25],
        };
        let derived = MassProperties::for_shape(&shape, 0.0);
        let heavy = MassProperties::for_shape(&shape, derived.mass * 4.0);
        assert!(close(heavy.mass, derived.mass * 4.0));
        for axis in 0..3 {
            assert!(
                close(heavy.inertia.get(axis), derived.inertia.get(axis) * 4.0),
                "axis {axis}: {:?} vs {:?}",
                heavy.inertia,
                derived.inertia
            );
        }
    }

    // A shape with no volume must still produce a usable mass rather than a
    // division by zero downstream.
    #[test]
    fn a_degenerate_shape_still_has_a_positive_mass() {
        let p = MassProperties::for_shape(
            &ColliderShape::Cuboid {
                half_extents: [0.0, 0.0, 0.0],
            },
            0.0,
        );
        assert!(p.mass >= MIN_MASS, "{}", p.mass);
        assert!(p.mass.is_finite());
    }
}

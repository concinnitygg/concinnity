// concinnity-physics/src/sim/collide/sphere.rs
//
// The pairs that reduce to one point. A sphere against anything touches at a
// single place, so each routine here finds the closest point on the other
// shape and hands the result to the same sphere-against-sphere core.
//
// The one case that is not a closest-point query is a sphere whose centre is
// inside a box: there is no closest surface point to aim at, so the shallowest
// face is used as the way out.

use crate::sim::contact::{Manifold, ManifoldPoint};
use crate::sim::math::Vec3;

use super::support::{OrientedBox, Sphere};

/// A single-point contact has one feature, so its id never varies.
const SINGLE_POINT_ID: u32 = 0;

/// Contact between two spheres, with the normal pointing from `a` toward `b`.
/// `fallback` is the normal used when the centres coincide.
pub(crate) fn spheres(
    a: Sphere,
    b: Sphere,
    fallback: Vec3,
    margin: f32,
    out: &mut Manifold,
) -> bool {
    let delta = b.center - a.center;
    let distance = delta.length();
    let separation = distance - (a.radius + b.radius);
    if separation > margin {
        return false;
    }
    let normal = if distance > f32::MIN_POSITIVE {
        delta * (1.0 / distance)
    } else {
        fallback
    };
    let on_a = a.center + normal * a.radius;
    let on_b = b.center - normal * b.radius;
    out.normal = normal;
    out.push(ManifoldPoint {
        point: (on_a + on_b) * 0.5,
        separation,
        id: SINGLE_POINT_ID,
        ..Default::default()
    });
    true
}

/// Contact between a sphere and a box, with the normal pointing from the
/// sphere toward the box.
pub(crate) fn sphere_box(
    sphere: Sphere,
    shape: OrientedBox,
    margin: f32,
    out: &mut Manifold,
) -> bool {
    let half = shape.half;
    let local = shape.pose.to_local(sphere.center);
    let clamped = local.clamp(-half, half);
    let inside = clamped == local;
    if !inside {
        // The box's surface point stands in for a zero-radius sphere.
        let surface = Sphere {
            center: shape.pose.to_world(clamped),
            radius: 0.0,
        };
        return spheres(sphere, surface, Vec3::Y, margin, out);
    }

    // Centre inside the box: leave by the nearest face.
    let mut axis = 0usize;
    let mut depth = f32::INFINITY;
    for candidate in 0..3 {
        let d = half.get(candidate) - libm::fabsf(local.get(candidate));
        if d < depth {
            depth = d;
            axis = candidate;
        }
    }
    let sign = if local.get(axis) >= 0.0 { 1.0 } else { -1.0 };
    let exit = shape.pose.axis(axis) * sign;
    let mut surface_local = local;
    surface_local.set(axis, sign * half.get(axis));
    let surface = shape.pose.to_world(surface_local);

    out.normal = -exit;
    out.push(ManifoldPoint {
        point: (surface + sphere.center + exit * sphere.radius) * 0.5,
        separation: -(depth + sphere.radius),
        id: SINGLE_POINT_ID,
        ..Default::default()
    });
    true
}

/// Contact between a sphere and a capsule, with the normal pointing from the
/// sphere toward the capsule.
pub(crate) fn sphere_capsule(
    sphere: Sphere,
    segment: (Vec3, Vec3),
    capsule_radius: f32,
    margin: f32,
    out: &mut Manifold,
) -> bool {
    let (on_axis, _) =
        super::support::closest_point_on_segment(sphere.center, segment.0, segment.1);
    // A capsule is a sphere swept along its axis, so the closest axis point
    // carries the whole contact.
    spheres(
        sphere,
        Sphere {
            center: on_axis,
            radius: capsule_radius,
        },
        Vec3::Y,
        margin,
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::collide::support::Pose;
    use crate::sim::math::{Quat, vec3};

    fn ball(center: Vec3, radius: f32) -> Sphere {
        Sphere { center, radius }
    }

    fn cube(half: f32, pose: Pose) -> OrientedBox {
        OrientedBox {
            half: Vec3::splat(half),
            pose,
        }
    }

    fn manifold() -> Manifold {
        Manifold::new(0, 1)
    }

    fn identity(position: Vec3) -> Pose {
        Pose {
            position,
            rotation: Quat::IDENTITY,
        }
    }

    #[test]
    fn overlapping_spheres_report_the_overlap_along_the_centre_line() {
        let mut m = manifold();
        assert!(spheres(
            ball(Vec3::ZERO, 1.0),
            ball(vec3(1.5, 0.0, 0.0), 1.0),
            Vec3::Y,
            0.0,
            &mut m
        ));
        assert_eq!(m.count, 1);
        assert_eq!(m.normal, Vec3::X);
        assert!((m.points()[0].separation + 0.5).abs() < 1.0e-6);
        assert!((m.points()[0].point - vec3(0.75, 0.0, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn distant_spheres_report_nothing_until_the_margin_reaches_them() {
        let mut m = manifold();
        let (a, b) = (ball(Vec3::ZERO, 1.0), ball(vec3(2.1, 0.0, 0.0), 1.0));
        assert!(!spheres(a, b, Vec3::Y, 0.0, &mut m));
        assert_eq!(m.count, 0);
        // A speculative margin makes the same gap worth reporting.
        assert!(spheres(a, b, Vec3::Y, 0.2, &mut m));
        assert!(m.points()[0].separation > 0.0);
    }

    #[test]
    fn coincident_spheres_fall_back_to_the_given_normal_rather_than_nan() {
        let mut m = manifold();
        assert!(spheres(
            ball(Vec3::ZERO, 1.0),
            ball(Vec3::ZERO, 1.0),
            Vec3::Y,
            0.0,
            &mut m
        ));
        assert_eq!(m.normal, Vec3::Y);
        assert!(m.points()[0].point.is_finite());
    }

    #[test]
    fn a_sphere_above_a_box_face_contacts_along_the_face_normal() {
        let mut m = manifold();
        assert!(sphere_box(
            ball(vec3(0.0, 1.4, 0.0), 0.5),
            cube(1.0, identity(Vec3::ZERO)),
            0.0,
            &mut m
        ));
        assert!(
            (m.normal - vec3(0.0, -1.0, 0.0)).length() < 1.0e-5,
            "{:?}",
            m.normal
        );
        assert!((m.points()[0].separation + 0.1).abs() < 1.0e-5);
    }

    #[test]
    fn a_sphere_at_a_box_corner_contacts_along_the_corner_diagonal() {
        let mut m = manifold();
        assert!(sphere_box(
            ball(vec3(1.2, 1.2, 1.2), 0.6),
            cube(1.0, identity(Vec3::ZERO)),
            0.0,
            &mut m
        ));
        let expected = vec3(-1.0, -1.0, -1.0).normalize_or_zero();
        assert!((m.normal - expected).length() < 1.0e-4, "{:?}", m.normal);
    }

    // A sphere whose centre has sunk inside the box has no closest surface
    // point; it must still be pushed out by the shallowest face.
    #[test]
    fn a_sphere_inside_a_box_leaves_by_the_nearest_face() {
        let mut m = manifold();
        assert!(sphere_box(
            ball(vec3(0.0, 0.9, 0.0), 0.5),
            cube(1.0, identity(Vec3::ZERO)),
            0.0,
            &mut m
        ));
        // Pushing the sphere out means the normal points into the box.
        assert!(
            (m.normal - vec3(0.0, -1.0, 0.0)).length() < 1.0e-5,
            "{:?}",
            m.normal
        );
        assert!(
            (m.points()[0].separation + 0.6).abs() < 1.0e-5,
            "{:?}",
            m.points()[0]
        );
    }

    #[test]
    fn a_rotated_box_contacts_along_its_own_axes() {
        let mut m = manifold();
        let pose = Pose {
            position: Vec3::ZERO,
            rotation: Quat::from_euler_deg([0.0, 45.0, 0.0]),
        };
        let outward = vec3(1.0, 0.0, 1.0).normalize_or_zero();
        assert!(sphere_box(
            ball(outward * 1.4, 0.5),
            cube(1.0, pose),
            0.0,
            &mut m
        ));
        assert!((m.normal + outward).length() < 1.0e-4, "{:?}", m.normal);
    }

    #[test]
    fn a_sphere_beside_a_capsule_contacts_across_the_capsule_axis() {
        let mut m = manifold();
        let segment = (vec3(0.0, -1.0, 0.0), vec3(0.0, 1.0, 0.0));
        assert!(sphere_capsule(
            ball(vec3(0.7, 0.5, 0.0), 0.3),
            segment,
            0.5,
            0.0,
            &mut m
        ));
        assert!((m.normal + Vec3::X).length() < 1.0e-5, "{:?}", m.normal);
        assert!(m.points()[0].separation < 0.0);
    }

    #[test]
    fn a_sphere_past_a_capsule_end_contacts_against_the_cap() {
        let mut m = manifold();
        let segment = (vec3(0.0, -1.0, 0.0), vec3(0.0, 1.0, 0.0));
        assert!(sphere_capsule(
            ball(vec3(0.0, 1.6, 0.0), 0.3),
            segment,
            0.5,
            0.0,
            &mut m
        ));
        assert!((m.normal + Vec3::Y).length() < 1.0e-5, "{:?}", m.normal);
        assert!(!sphere_capsule(
            ball(vec3(0.0, 3.0, 0.0), 0.3),
            segment,
            0.5,
            0.0,
            &mut m
        ));
    }
}

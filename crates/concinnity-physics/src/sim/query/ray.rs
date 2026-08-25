// concinnity-physics/src/sim/query/ray.rs
//
// Where a ray first meets a shape, worked out in closed form.
//
// A ray is cheap enough against each of the three shapes to be solved exactly,
// so it is, rather than being handed to the iterative sweep next door. Every
// test runs in the shape's own frame, which is what lets an oriented box be
// the same slab test an axis-aligned one is.
//
// A ray that begins inside a shape hits it at once, at zero distance, with the
// normal turned back along the ray. That keeps the reported normal unit length
// whatever the caller did, and it is what a probe fired from inside a
// character's own capsule needs to hear.

use crate::ColliderShape;

use crate::sim::aabb::Aabb;
use crate::sim::collide::Pose;
use crate::sim::math::{Vec3, vec3};

use super::gjk::Support;

/// A ray in world space, with a unit direction.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Ray {
    pub(crate) origin: Vec3,
    /// Unit length: the primitives below solve in distance, not in parameter.
    pub(crate) direction: Vec3,
}

/// Where a ray met a shape.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RayImpact {
    pub(crate) distance: f32,
    pub(crate) normal: Vec3,
}

/// Direction components smaller than this are treated as parallel to the axis,
/// so a slab test divides by nothing.
const PARALLEL: f32 = 1.0e-8;

/// Where the ray first meets a shape at a pose, within `max_dist`.
pub(crate) fn cast(
    ray: Ray,
    shape: &ColliderShape,
    pose: Pose,
    max_dist: f32,
) -> Option<RayImpact> {
    // Inside is the same answer whatever the shape, and settling it first is
    // what lets each primitive below assume it is looking for a way in.
    if Support::new(shape, pose).contains(ray.origin) {
        return Some(RayImpact {
            distance: 0.0,
            normal: -ray.direction,
        });
    }

    let local = Ray {
        origin: pose.to_local(ray.origin),
        direction: pose.rotation.inverse_rotate(ray.direction),
    };
    let impact = match *shape {
        ColliderShape::Ball { radius } => sphere(local, libm::fabsf(radius)),
        ColliderShape::Cuboid { half_extents } => {
            cuboid(local, Vec3::from_array(half_extents).abs())
        }
        ColliderShape::Capsule {
            half_height,
            radius,
        } => capsule(local, libm::fabsf(half_height), libm::fabsf(radius)),
    }?;
    if impact.distance > max_dist {
        return None;
    }
    Some(RayImpact {
        distance: impact.distance,
        normal: pose.rotation.rotate(impact.normal),
    })
}

/// A ray with its reciprocal direction already worked out.
///
/// The bounds test runs once per candidate the broad phase offers, and the
/// three divisions it needs depend only on the ray. Doing them once per query
/// rather than once per candidate is most of what the test costs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BoundsProbe {
    origin: Vec3,
    /// Reciprocal per axis, infinite where the ray is parallel to it.
    inverse: Vec3,
}

impl BoundsProbe {
    pub(crate) fn new(ray: Ray) -> Self {
        let reciprocal = |d: f32| {
            if libm::fabsf(d) < PARALLEL {
                f32::INFINITY
            } else {
                1.0 / d
            }
        };
        BoundsProbe {
            origin: ray.origin,
            inverse: vec3(
                reciprocal(ray.direction.x),
                reciprocal(ray.direction.y),
                reciprocal(ray.direction.z),
            ),
        }
    }

    /// Whether the ray reaches the bounds at all, within `max_dist`. The broad
    /// phase's filter: cheap, and never rejects something the exact test wants.
    pub(crate) fn reaches(&self, bounds: Aabb, max_dist: f32) -> bool {
        let mut near = 0.0f32;
        let mut far = max_dist;
        for axis in 0..3 {
            let o = self.origin.get(axis);
            let inv = self.inverse.get(axis);
            if inv.is_infinite() {
                if o < bounds.min.get(axis) || o > bounds.max.get(axis) {
                    return false;
                }
                continue;
            }
            let mut low = (bounds.min.get(axis) - o) * inv;
            let mut high = (bounds.max.get(axis) - o) * inv;
            if low > high {
                core::mem::swap(&mut low, &mut high);
            }
            near = near.max(low);
            far = far.min(high);
            if near > far {
                return false;
            }
        }
        true
    }
}

/// A ray starting outside a sphere centred on the origin.
fn sphere(ray: Ray, radius: f32) -> Option<RayImpact> {
    let b = ray.origin.dot(ray.direction);
    if b >= 0.0 {
        // Outside and pointing away: nothing ahead to hit.
        return None;
    }
    let c = ray.origin.length_squared() - radius * radius;
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return None;
    }
    let distance = -b - libm::sqrtf(discriminant);
    if distance < 0.0 {
        return None;
    }
    let at = ray.origin + ray.direction * distance;
    Some(RayImpact {
        distance,
        normal: at.normalize_or(-ray.direction),
    })
}

/// A ray starting outside a box centred on the origin.
fn cuboid(ray: Ray, half: Vec3) -> Option<RayImpact> {
    let mut near = 0.0f32;
    let mut far = f32::INFINITY;
    let mut entry_axis = 0usize;
    let mut entry_sign = 1.0f32;
    for axis in 0..3 {
        let d = ray.direction.get(axis);
        let o = ray.origin.get(axis);
        if libm::fabsf(d) < PARALLEL {
            if libm::fabsf(o) > half.get(axis) {
                return None;
            }
            continue;
        }
        let inv = 1.0 / d;
        let mut low = (-half.get(axis) - o) * inv;
        let mut high = (half.get(axis) - o) * inv;
        // Entering through the face the ray points at.
        let mut sign = -1.0f32;
        if low > high {
            core::mem::swap(&mut low, &mut high);
            sign = 1.0;
        }
        if low > near {
            near = low;
            entry_axis = axis;
            entry_sign = sign;
        }
        far = far.min(high);
        if near > far {
            return None;
        }
    }
    if far < 0.0 {
        return None;
    }
    Some(RayImpact {
        distance: near,
        normal: Vec3::axis(entry_axis) * entry_sign,
    })
}

/// A ray starting outside a Y-axis capsule centred on the origin.
///
/// The capsule is the union of a finite cylinder and the two cap balls, so the
/// first way in is the nearest of the three, and the piece that wins carries
/// the normal.
fn capsule(ray: Ray, half_height: f32, radius: f32) -> Option<RayImpact> {
    let mut best: Option<RayImpact> = None;

    let a = ray.direction.x * ray.direction.x + ray.direction.z * ray.direction.z;
    if a > PARALLEL {
        let b = ray.origin.x * ray.direction.x + ray.origin.z * ray.direction.z;
        let c = ray.origin.x * ray.origin.x + ray.origin.z * ray.origin.z - radius * radius;
        let discriminant = b * b - a * c;
        if discriminant >= 0.0 {
            let distance = (-b - libm::sqrtf(discriminant)) / a;
            let y = ray.origin.y + ray.direction.y * distance;
            if distance >= 0.0 && libm::fabsf(y) <= half_height {
                let at = ray.origin + ray.direction * distance;
                best = Some(RayImpact {
                    distance,
                    normal: vec3(at.x, 0.0, at.z).normalize_or(-ray.direction),
                });
            }
        }
    }

    for cap in [half_height, -half_height] {
        let centre = vec3(0.0, cap, 0.0);
        let shifted = Ray {
            origin: ray.origin - centre,
            direction: ray.direction,
        };
        let Some(hit) = sphere(shifted, radius) else {
            continue;
        };
        if best.is_none_or(|current| hit.distance < current.distance) {
            best = Some(hit);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::math::Quat;

    const BALL: ColliderShape = ColliderShape::Ball { radius: 0.5 };
    const CUBE: ColliderShape = ColliderShape::Cuboid {
        half_extents: [0.5, 0.5, 0.5],
    };
    const CAPSULE: ColliderShape = ColliderShape::Capsule {
        half_height: 1.0,
        radius: 0.25,
    };

    fn ray(origin: Vec3, direction: Vec3) -> Ray {
        Ray {
            origin,
            direction: direction.normalize_or(Vec3::X),
        }
    }

    fn at(position: Vec3) -> Pose {
        Pose {
            position,
            rotation: Quat::IDENTITY,
        }
    }

    fn turned(position: Vec3, euler_deg: [f32; 3]) -> Pose {
        Pose {
            position,
            rotation: Quat::from_euler_deg(euler_deg),
        }
    }

    fn shapes() -> [ColliderShape; 3] {
        [BALL, CUBE, CAPSULE]
    }

    #[test]
    fn a_ray_meets_a_sphere_at_its_near_surface() {
        let hit = cast(
            ray(vec3(0.0, 5.0, 0.0), -Vec3::Y),
            &BALL,
            at(Vec3::ZERO),
            100.0,
        )
        .expect("a hit");
        assert!((hit.distance - 4.5).abs() < 1.0e-5, "{hit:?}");
        assert!((hit.normal - Vec3::Y).length() < 1.0e-5, "{hit:?}");
    }

    #[test]
    fn a_ray_meets_a_box_on_the_face_it_points_at() {
        for (direction, expected) in [
            (Vec3::Y, -Vec3::Y),
            (-Vec3::Y, Vec3::Y),
            (Vec3::X, -Vec3::X),
            (-Vec3::Z, Vec3::Z),
        ] {
            let hit = cast(
                ray(direction * -5.0, direction),
                &CUBE,
                at(Vec3::ZERO),
                100.0,
            )
            .expect("a hit");
            assert!((hit.distance - 4.5).abs() < 1.0e-5, "{direction:?} {hit:?}");
            assert!(
                (hit.normal - expected).length() < 1.0e-5,
                "{direction:?} {hit:?}"
            );
        }
    }

    #[test]
    fn a_ray_meets_a_capsule_on_its_barrel_and_on_its_caps() {
        let barrel = cast(
            ray(vec3(5.0, 0.0, 0.0), -Vec3::X),
            &CAPSULE,
            at(Vec3::ZERO),
            100.0,
        )
        .expect("a hit");
        assert!((barrel.distance - 4.75).abs() < 1.0e-5, "{barrel:?}");
        assert!((barrel.normal - Vec3::X).length() < 1.0e-5, "{barrel:?}");

        let cap = cast(
            ray(vec3(0.0, 5.0, 0.0), -Vec3::Y),
            &CAPSULE,
            at(Vec3::ZERO),
            100.0,
        )
        .expect("a hit");
        assert!((cap.distance - 3.75).abs() < 1.0e-5, "{cap:?}");
        assert!((cap.normal - Vec3::Y).length() < 1.0e-5, "{cap:?}");
    }

    // A ray shallow enough to cross the infinite cylinder above the barrel
    // has to come back with the cap, not with the cylinder's own root.
    #[test]
    fn a_ray_crossing_a_capsules_barrel_line_above_it_meets_the_cap() {
        let origin = vec3(5.0, 1.15, 0.0);
        let direction = vec3(-1.0, -0.02, 0.0);
        let hit = cast(ray(origin, direction), &CAPSULE, at(Vec3::ZERO), 100.0).expect("a hit");
        let point = origin + direction.normalize_or(Vec3::X) * hit.distance;
        assert!(point.y > 1.0, "the cap, not the barrel: {point:?}");
        let from_cap = (point - vec3(0.0, 1.0, 0.0)).length();
        assert!((from_cap - 0.25).abs() < 1.0e-4, "{point:?} {from_cap}");
    }

    #[test]
    fn every_shape_misses_when_the_ray_points_away() {
        for shape in shapes() {
            assert!(
                cast(
                    ray(vec3(0.0, 5.0, 0.0), Vec3::Y),
                    &shape,
                    at(Vec3::ZERO),
                    100.0
                )
                .is_none(),
                "{shape:?}"
            );
        }
    }

    #[test]
    fn every_shape_misses_when_the_ray_passes_beside_it() {
        for shape in shapes() {
            assert!(
                cast(
                    ray(vec3(9.0, 5.0, 0.0), -Vec3::Y),
                    &shape,
                    at(Vec3::ZERO),
                    100.0
                )
                .is_none(),
                "{shape:?}"
            );
        }
    }

    // Beginning inside is a hit at once, and the normal has to stay unit
    // length so a caller can reflect off it.
    #[test]
    fn a_ray_starting_inside_every_shape_hits_at_zero_facing_back() {
        for shape in shapes() {
            let hit = cast(
                ray(vec3(0.0, 0.05, 0.0), -Vec3::Y),
                &shape,
                at(Vec3::ZERO),
                100.0,
            )
            .expect("a hit");
            assert_eq!(hit.distance, 0.0, "{shape:?}");
            assert!((hit.normal - Vec3::Y).length() < 1.0e-6, "{shape:?}");
        }
    }

    // Grazing the surface exactly: the answer may be a hit or a miss, but it
    // must be finite and never a NaN distance.
    #[test]
    fn a_tangent_ray_answers_without_producing_a_nan() {
        for shape in shapes() {
            let tangent = cast(
                ray(vec3(0.5, 5.0, 0.0), -Vec3::Y),
                &shape,
                at(Vec3::ZERO),
                100.0,
            );
            if let Some(hit) = tangent {
                assert!(hit.distance.is_finite() && hit.distance >= 0.0, "{shape:?}");
                assert!(hit.normal.is_finite(), "{shape:?}");
                assert!(
                    (hit.normal.length() - 1.0).abs() < 1.0e-3,
                    "{shape:?} {hit:?}"
                );
            }
        }
    }

    #[test]
    fn the_distance_limit_is_inclusive_at_its_boundary() {
        let origin = vec3(0.0, 5.0, 0.0);
        // The sphere's near surface is exactly 4.5 away.
        assert!(cast(ray(origin, -Vec3::Y), &BALL, at(Vec3::ZERO), 4.5).is_some());
        assert!(cast(ray(origin, -Vec3::Y), &BALL, at(Vec3::ZERO), 4.4).is_none());
    }

    // An oriented box is the same slab test in its own frame, and the normal
    // has to come back rotated into the world.
    #[test]
    fn a_turned_box_reports_a_turned_normal() {
        let hit = cast(
            ray(vec3(5.0, 0.0, 0.0), -Vec3::X),
            &CUBE,
            turned(Vec3::ZERO, [0.0, 45.0, 0.0]),
            100.0,
        )
        .expect("a hit");
        let reach = 0.5 * libm::sqrtf(2.0);
        assert!((hit.distance - (5.0 - reach)).abs() < 1.0e-4, "{hit:?}");
        // The corner faces the ray, so the normal is one of the two faces
        // meeting there, each 45 degrees off the ray.
        assert!(hit.normal.x > 0.6, "{hit:?}");
    }

    #[test]
    fn a_turned_capsule_is_met_along_its_new_axis() {
        let hit = cast(
            ray(vec3(5.0, 0.0, 0.0), -Vec3::X),
            &CAPSULE,
            turned(Vec3::ZERO, [0.0, 0.0, 90.0]),
            100.0,
        )
        .expect("a hit");
        // Rolled onto x, the capsule reaches 1.25 along it.
        assert!((hit.distance - 3.75).abs() < 1.0e-4, "{hit:?}");
        assert!((hit.normal - Vec3::X).length() < 1.0e-4, "{hit:?}");
    }

    #[test]
    fn a_shape_away_from_the_origin_is_met_at_its_own_place() {
        let hit = cast(
            ray(vec3(10.0, 3.0, -2.0), -Vec3::Y),
            &BALL,
            at(vec3(10.0, 0.0, -2.0)),
            100.0,
        )
        .expect("a hit");
        assert!((hit.distance - 2.5).abs() < 1.0e-5, "{hit:?}");
    }

    #[test]
    fn bounds_rejection_agrees_with_reach_and_direction() {
        let bounds = Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(1.0));
        let probe = |origin, direction| BoundsProbe::new(ray(origin, direction));
        assert!(probe(vec3(0.0, 5.0, 0.0), -Vec3::Y).reaches(bounds, 10.0));
        assert!(!probe(vec3(0.0, 5.0, 0.0), -Vec3::Y).reaches(bounds, 3.0));
        assert!(!probe(vec3(0.0, 5.0, 0.0), Vec3::Y).reaches(bounds, 100.0));
        // Parallel to an axis, beside the box on it.
        assert!(!probe(vec3(0.0, 5.0, 0.0), Vec3::X).reaches(bounds, 100.0));
        // Parallel to two axes and inside the box on both: still a reach.
        assert!(probe(vec3(0.5, -0.5, -5.0), Vec3::Z).reaches(bounds, 100.0));
        // Starting inside is always a reach.
        assert!(probe(Vec3::ZERO, Vec3::X).reaches(bounds, 100.0));
    }

    // The precomputed reciprocals must not change the answer, whatever the
    // ray, or the broad-phase filter starts rejecting real hits.
    #[test]
    fn the_bounds_filter_never_rejects_a_ray_that_hits() {
        let bounds = Aabb::from_center_half_extents(vec3(2.0, -1.0, 0.5), Vec3::splat(1.0));
        let shape = ColliderShape::Cuboid {
            half_extents: [1.0, 1.0, 1.0],
        };
        for i in 0..64 {
            let angle = i as f32 * core::f32::consts::TAU / 64.0;
            let origin = vec3(libm::cosf(angle) * 6.0, libm::sinf(angle) * 6.0, -4.0);
            let direction = vec3(2.0, -1.0, 0.5) - origin;
            let r = ray(origin, direction);
            let hit = cast(r, &shape, at(vec3(2.0, -1.0, 0.5)), 100.0);
            if hit.is_some() {
                assert!(
                    BoundsProbe::new(r).reaches(bounds, 100.0),
                    "ray {i} hit the shape but not its bounds"
                );
            }
        }
    }
}

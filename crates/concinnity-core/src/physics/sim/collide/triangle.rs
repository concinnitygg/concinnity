// A convex shape against one triangle, always along the triangle's own face
// normal.
//
// That choice is the whole module. A general convex-convex test finds the
// least separating axis, which for a shape resting across two triangles of the
// same surface is the edge they share -- and an edge normal points sideways,
// so a capsule sliding over it is stopped by a surface it should have slid
// along. Answering only along the face normal makes the two triangles push the
// same way they would if the surface were one piece, which is what a height
// grid is.
//
// What it costs is the outer boundary of a mesh. A shape meeting a triangle
// from beyond the last edge is answered as though the triangle's plane
// continued, because there is no neighbour to own that region. Inside the
// grid, where every edge is shared, there is nothing to lose.

use crate::math::sqrt;
use crate::physics::ColliderShape;

use crate::physics::sim::math::{Vec3, vec3};

use super::support::{Pose, face_corners, support_vertex};

/// Points a clipped incident face can leave behind: a quad against three edge
/// planes gains at most one point per plane.
pub(crate) const MAX_TRIANGLE_POINTS: usize = 8;

/// How far outside an edge a point may fall and still be owned by the
/// triangle. Small: it covers the arithmetic of the clip and lets two
/// triangles both claim a contact exactly on the edge they share, which is
/// steadier than either of them claiming it in alternate steps.
const EDGE_SLACK: f32 = 1.0e-3;

/// How close a clipped point has to be to a feature to be called that feature.
const FEATURE_EPSILON: f32 = 1.0e-4;

/// A triangle of the surface, with the outward normal that answers for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Triangle {
    pub(crate) corners: [Vec3; 3],
    /// Unit outward normal. Degenerate triangles never reach here.
    pub(crate) normal: Vec3,
}

impl Triangle {
    /// A triangle, or `None` when the three corners name no plane.
    pub(crate) fn new(corners: [Vec3; 3]) -> Option<Self> {
        let normal = (corners[1] - corners[0]).cross(corners[2] - corners[0]);
        if !normal.is_finite() {
            return None;
        }
        let length = normal.length();
        if length <= f32::MIN_POSITIVE {
            return None;
        }
        Some(Triangle {
            corners,
            normal: normal * (1.0 / length),
        })
    }

    /// Height of a point above the triangle's plane.
    pub(crate) fn height_of(&self, point: Vec3) -> f32 {
        (point - self.corners[0]).dot(self.normal)
    }

    /// The three half-spaces a point has to be inside for this triangle to own
    /// it, as vertical planes through its edges, widened by `slack`.
    ///
    /// Vertical rather than extruded along the face normal, because the
    /// surface these triangles come from is a height function: seen from
    /// above they tile the ground with no gaps, so every point is owned by
    /// exactly one of them. Extruding along each face normal instead leaves a
    /// wedge over every ridge that neither of the two triangles meeting there
    /// reaches into, and a shape resting on the ridge falls through it.
    fn edge_planes(&self, slack: f32) -> [Plane; 3] {
        let mut planes = [Plane {
            normal: Vec3::X,
            offset: 0.0,
        }; 3];
        for (index, plane) in planes.iter_mut().enumerate() {
            let (from, to) = (self.corners[index], self.corners[(index + 1) % 3]);
            let edge = to - from;
            // An edge with no ground plan has no vertical plane of its own, so
            // the face normal is the only direction left to extrude along.
            let outward = edge
                .cross(Vec3::Y)
                .normalize_or(edge.cross(self.normal).normalize_or_zero());
            *plane = Plane {
                normal: outward,
                offset: outward.dot(from) + slack,
            };
        }
        planes
    }
}

/// One side of the triangle's extent: `normal . p <= offset` is inside.
#[derive(Debug, Clone, Copy)]
struct Plane {
    normal: Vec3,
    offset: f32,
}

impl Plane {
    fn distance(&self, point: Vec3) -> f32 {
        self.normal.dot(point) - self.offset
    }
}

/// The shape's surface point furthest along `direction`.
pub(crate) fn support_point(shape: &ColliderShape, pose: Pose, direction: Vec3) -> Vec3 {
    let unit = direction.normalize_or(Vec3::Y);
    match *shape {
        ColliderShape::Ball { radius } => pose.position + unit * radius.abs(),
        ColliderShape::Cuboid { half_extents } => {
            let local = pose.rotation.inverse_rotate(unit);
            pose.to_world(support_vertex(Vec3::from_array(half_extents).abs(), local))
        }
        ColliderShape::Capsule {
            half_height,
            radius,
        } => {
            let local = pose.rotation.inverse_rotate(unit);
            let end = vec3(
                0.0,
                if local.y >= 0.0 {
                    half_height.abs()
                } else {
                    -half_height.abs()
                },
                0.0,
            );
            pose.to_world(end) + unit * radius.abs()
        }
    }
}

/// The part of the shape's core facing `-normal`, and how far its surface
/// stands off that core: the face for a box, the axis for a capsule, the
/// centre for a ball.
///
/// The core rather than the surface, because which triangle owns a contact is
/// decided by where it stands on the ground and the surface point does not
/// stand anywhere fixed: it sits a radius along the triangle's own normal,
/// which leans one way on one side of a ridge and the other way on the other,
/// so both triangles would disown a shape resting on it. The core does not
/// lean, so exactly one triangle owns it.
fn incident_core(shape: &ColliderShape, pose: Pose, normal: Vec3) -> ([Vec3; 4], usize, f32) {
    let mut points = [Vec3::ZERO; 4];
    match *shape {
        ColliderShape::Ball { radius } => {
            points[0] = pose.position;
            (points, 1, radius.abs())
        }
        ColliderShape::Cuboid { half_extents } => {
            let half = Vec3::from_array(half_extents).abs();
            let local = pose.rotation.inverse_rotate(-normal);
            let face = super::support::best_face(local);
            for (slot, corner) in points.iter_mut().zip(face_corners(half, face)) {
                *slot = pose.to_world(corner);
            }
            (points, 4, 0.0)
        }
        ColliderShape::Capsule {
            half_height,
            radius,
        } => {
            let axis = pose.rotation.rotate(Vec3::Y) * half_height.abs();
            points[0] = pose.position - axis;
            points[1] = pose.position + axis;
            (points, 2, radius.abs())
        }
    }
}

/// One contact between a shape and a triangle, before it becomes a manifold
/// point.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TriangleContact {
    /// Point on the shape's surface.
    pub(crate) point: Vec3,
    /// Height of that point above the triangle's plane; negative when the
    /// shape has sunk into the surface.
    pub(crate) separation: f32,
    /// Which incident and triangle features produced it, for warm starting.
    pub(crate) feature: u32,
}

/// The contacts a shape makes with one triangle, measured along the triangle's
/// face normal. `count` is zero when the shape does not reach the triangle's
/// own extent.
///
/// `offset` displaces the shape before the test, which is what lets a sweep
/// ask the question at the moment of impact rather than at the moment it
/// started.
pub(crate) fn contacts(
    triangle: &Triangle,
    shape: &ColliderShape,
    pose: Pose,
    offset: Vec3,
    margin: f32,
    out: &mut [TriangleContact; MAX_TRIANGLE_POINTS],
) -> usize {
    let moved = Pose {
        position: pose.position + offset,
        rotation: pose.rotation,
    };
    let (core, core_count, radius) = incident_core(shape, moved, triangle.normal);
    // A rounded shape's core may stand a radius outside the ground the surface
    // touches, and only ever that far, so the slack that keeps a cliff face
    // reachable is bounded by how far the surface leans off the core.
    let leaning =
        sqrt(triangle.normal.x * triangle.normal.x + triangle.normal.z * triangle.normal.z);
    let planes = triangle.edge_planes(EDGE_SLACK + radius * leaning);

    let mut clipped = [Vec3::ZERO; MAX_TRIANGLE_POINTS];
    let clipped_count = match core_count {
        1 => usize::from(planes.iter().all(|p| p.distance(core[0]) <= 0.0)),
        2 => clip_segment([core[0], core[1]], &planes, &mut clipped),
        _ => clip_polygon(&core[..core_count], &planes, &mut clipped),
    };
    if core_count == 1 && clipped_count == 1 {
        clipped[0] = core[0];
    }

    let mut kept = 0usize;
    for point in &clipped[..clipped_count] {
        let separation = triangle.height_of(*point) - radius;
        if separation > margin {
            continue;
        }
        out[kept] = TriangleContact {
            point: *point - triangle.normal * radius,
            separation,
            feature: feature_of(*point, &core[..core_count], &planes),
        };
        kept += 1;
    }
    kept
}

/// Which incident corner a clipped point came from, and which of the
/// triangle's edges it ended up on.
///
/// Derived from where the point is rather than from how the clip got there, so
/// the same geometry gives the same id and a warm-started impulse follows it.
fn feature_of(point: Vec3, incident: &[Vec3], planes: &[Plane; 3]) -> u32 {
    let mut corner = 0x7u32;
    for (index, candidate) in incident.iter().enumerate() {
        if (point - *candidate).length_squared() <= FEATURE_EPSILON * FEATURE_EPSILON {
            corner = index as u32;
            break;
        }
    }
    let mut edges = 0u32;
    for (bit, plane) in planes.iter().enumerate() {
        if (plane.distance(point)).abs() <= FEATURE_EPSILON {
            edges |= 1 << bit;
        }
    }
    (corner << 3) | edges
}

/// Clip a segment to the three half-spaces, keeping the part inside.
fn clip_segment(
    segment: [Vec3; 2],
    planes: &[Plane; 3],
    out: &mut [Vec3; MAX_TRIANGLE_POINTS],
) -> usize {
    let (mut low, mut high) = (0.0f32, 1.0f32);
    let direction = segment[1] - segment[0];
    for plane in planes {
        let start = plane.distance(segment[0]);
        let rate = plane.normal.dot(direction);
        if rate.abs() <= f32::MIN_POSITIVE {
            if start > 0.0 {
                return 0;
            }
            continue;
        }
        let crossing = -start / rate;
        if rate > 0.0 {
            high = high.min(crossing);
        } else {
            low = low.max(crossing);
        }
        if low > high {
            return 0;
        }
    }
    out[0] = segment[0] + direction * low;
    out[1] = segment[0] + direction * high;
    if (out[1] - out[0]).length_squared() <= FEATURE_EPSILON * FEATURE_EPSILON {
        1
    } else {
        2
    }
}

/// Clip a convex polygon to the three half-spaces.
fn clip_polygon(
    polygon: &[Vec3],
    planes: &[Plane; 3],
    out: &mut [Vec3; MAX_TRIANGLE_POINTS],
) -> usize {
    let mut input = [Vec3::ZERO; MAX_TRIANGLE_POINTS];
    let mut length = polygon.len().min(MAX_TRIANGLE_POINTS);
    input[..length].copy_from_slice(&polygon[..length]);

    for plane in planes {
        let mut kept = 0usize;
        for index in 0..length {
            let current = input[index];
            let next = input[(index + 1) % length];
            let (here, there) = (plane.distance(current), plane.distance(next));
            if here <= 0.0 && kept < MAX_TRIANGLE_POINTS {
                out[kept] = current;
                kept += 1;
            }
            if (here <= 0.0) != (there <= 0.0) && kept < MAX_TRIANGLE_POINTS {
                let span = here - there;
                let t = if span.abs() > f32::MIN_POSITIVE {
                    here / span
                } else {
                    0.0
                };
                out[kept] = current + (next - current) * t;
                kept += 1;
            }
        }
        input[..kept].copy_from_slice(&out[..kept]);
        length = kept;
        if length == 0 {
            return 0;
        }
    }
    out[..length].copy_from_slice(&input[..length]);
    length
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::sim::math::Quat;

    /// A unit triangle lying flat, covering the corner of the xz square.
    fn flat() -> Triangle {
        Triangle::new([
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 1.0),
            vec3(1.0, 0.0, 0.0),
        ])
        .expect("a real triangle")
    }

    fn at(position: Vec3) -> Pose {
        Pose {
            position,
            rotation: Quat::IDENTITY,
        }
    }

    fn empty() -> [TriangleContact; MAX_TRIANGLE_POINTS] {
        [TriangleContact {
            point: Vec3::ZERO,
            separation: 0.0,
            feature: 0,
        }; MAX_TRIANGLE_POINTS]
    }

    #[test]
    fn a_triangle_faces_up_and_a_degenerate_one_is_no_triangle() {
        assert!((flat().normal - Vec3::Y).length() < 1.0e-6);
        assert!(Triangle::new([Vec3::ZERO, Vec3::X, Vec3::X * 2.0]).is_none());
        assert!(Triangle::new([Vec3::ZERO; 3]).is_none());
        assert!((flat().height_of(vec3(0.2, 3.0, 0.2)) - 3.0).abs() < 1.0e-6);
    }

    #[test]
    fn a_support_point_is_on_the_surface_it_names() {
        let ball = ColliderShape::Ball { radius: 0.5 };
        assert!(
            (support_point(&ball, at(Vec3::ZERO), Vec3::Y) - vec3(0.0, 0.5, 0.0)).length() < 1.0e-6
        );

        let cube = ColliderShape::Cuboid {
            half_extents: [0.5; 3],
        };
        let corner = support_point(&cube, at(Vec3::ZERO), vec3(1.0, 1.0, 1.0));
        assert!((corner - Vec3::splat(0.5)).length() < 1.0e-6);

        let capsule = ColliderShape::Capsule {
            half_height: 0.6,
            radius: 0.3,
        };
        let foot = support_point(&capsule, at(Vec3::ZERO), -Vec3::Y);
        assert!((foot - vec3(0.0, -0.9, 0.0)).length() < 1.0e-6, "{foot:?}");
    }

    // The resting case: a box over the middle of a triangle keeps the corners
    // of its lowest face that lie over the triangle.
    #[test]
    fn a_box_over_a_triangle_contacts_through_its_lowest_face() {
        let cube = ColliderShape::Cuboid {
            half_extents: [0.1, 0.1, 0.1],
        };
        let mut out = empty();
        let count = contacts(
            &flat(),
            &cube,
            at(vec3(0.2, 0.05, 0.2)),
            Vec3::ZERO,
            0.02,
            &mut out,
        );
        assert_eq!(count, 4, "the whole face is over the triangle");
        for contact in &out[..count] {
            assert!((contact.separation + 0.05).abs() < 1.0e-5, "{contact:?}");
            assert!(contact.point.y < 0.0, "the corners are under it");
        }
        // Every corner has to be told apart, or a warm start would follow the
        // wrong one.
        for i in 0..count {
            for j in i + 1..count {
                assert_ne!(out[i].feature, out[j].feature, "{i} and {j}");
            }
        }
    }

    #[test]
    fn a_shape_clear_of_the_triangles_plane_makes_no_contact() {
        let ball = ColliderShape::Ball { radius: 0.2 };
        let mut out = empty();
        assert_eq!(
            contacts(
                &flat(),
                &ball,
                at(vec3(0.2, 4.0, 0.2)),
                Vec3::ZERO,
                0.02,
                &mut out
            ),
            0
        );
    }

    // A shape beyond the triangle's own extent belongs to a neighbour, so this
    // triangle must not claim it.
    #[test]
    fn a_shape_beyond_the_triangles_extent_is_left_to_its_neighbour() {
        let ball = ColliderShape::Ball { radius: 0.05 };
        let mut out = empty();
        // Well outside the hypotenuse from (0,0,1) to (1,0,0).
        assert_eq!(
            contacts(
                &flat(),
                &ball,
                at(vec3(0.9, 0.0, 0.9)),
                Vec3::ZERO,
                0.02,
                &mut out
            ),
            0
        );
        // And inside it, the same ball does make contact.
        assert_eq!(
            contacts(
                &flat(),
                &ball,
                at(vec3(0.2, 0.0, 0.2)),
                Vec3::ZERO,
                0.02,
                &mut out
            ),
            1
        );
        assert!((out[0].separation + 0.05).abs() < 1.0e-5, "{:?}", out[0]);
    }

    // A capsule lying across the surface has to hold on two points, or it
    // rocks on the one it has.
    #[test]
    fn a_capsule_lying_across_a_triangle_contacts_at_both_ends() {
        let big = Triangle::new([
            vec3(-5.0, 0.0, -5.0),
            vec3(-5.0, 0.0, 5.0),
            vec3(5.0, 0.0, -5.0),
        ])
        .expect("a real triangle");
        let capsule = ColliderShape::Capsule {
            half_height: 0.5,
            radius: 0.1,
        };
        let lying = Pose {
            position: vec3(-1.0, 0.1, -1.0),
            rotation: Quat::from_euler_deg([0.0, 0.0, 90.0]),
        };
        let mut out = empty();
        let count = contacts(&big, &capsule, lying, Vec3::ZERO, 0.02, &mut out);
        assert_eq!(count, 2, "{out:?}");
        assert!(
            (out[0].point.x - out[1].point.x).abs() > 0.9,
            "two distinct ends: {out:?}"
        );
        for contact in &out[..count] {
            assert!(contact.separation.abs() < 1.0e-5, "{contact:?}");
        }
    }

    // Stood on its end a capsule is a ball as far as the surface is concerned.
    #[test]
    fn a_capsule_stood_on_end_contacts_at_one_point() {
        let big = Triangle::new([
            vec3(-5.0, 0.0, -5.0),
            vec3(-5.0, 0.0, 5.0),
            vec3(5.0, 0.0, -5.0),
        ])
        .expect("a real triangle");
        let capsule = ColliderShape::Capsule {
            half_height: 0.5,
            radius: 0.1,
        };
        let mut out = empty();
        let count = contacts(
            &big,
            &capsule,
            at(vec3(-1.0, 0.55, -1.0)),
            Vec3::ZERO,
            0.02,
            &mut out,
        );
        assert_eq!(count, 1);
        assert!((out[0].separation + 0.05).abs() < 1.0e-5, "{:?}", out[0]);
    }

    // The offset is what a sweep asks its question with: the same shape, moved.
    #[test]
    fn an_offset_asks_the_question_at_a_moved_pose() {
        let ball = ColliderShape::Ball { radius: 0.05 };
        let mut out = empty();
        assert_eq!(
            contacts(
                &flat(),
                &ball,
                at(vec3(0.2, 2.0, 0.2)),
                Vec3::ZERO,
                0.02,
                &mut out
            ),
            0
        );
        assert_eq!(
            contacts(
                &flat(),
                &ball,
                at(vec3(0.2, 2.0, 0.2)),
                vec3(0.0, -2.0, 0.0),
                0.02,
                &mut out
            ),
            1
        );
    }

    #[test]
    fn a_segment_clip_keeps_the_part_inside_and_drops_the_rest() {
        let planes = flat().edge_planes(EDGE_SLACK);
        let mut out = [Vec3::ZERO; MAX_TRIANGLE_POINTS];
        // Crossing the hypotenuse: the far end is clipped back onto it.
        let count = clip_segment(
            [vec3(0.1, 0.0, 0.1), vec3(2.0, 0.0, 2.0)],
            &planes,
            &mut out,
        );
        assert_eq!(count, 2);
        assert!((out[0] - vec3(0.1, 0.0, 0.1)).length() < 1.0e-5, "{out:?}");
        assert!(out[1].x + out[1].z <= 1.0 + 2.0 * EDGE_SLACK, "{out:?}");
        // Entirely outside.
        assert_eq!(
            clip_segment(
                [vec3(3.0, 0.0, 3.0), vec3(4.0, 0.0, 4.0)],
                &planes,
                &mut out
            ),
            0
        );
    }

    #[test]
    fn a_polygon_clip_trims_a_quad_to_the_triangle() {
        let planes = flat().edge_planes(EDGE_SLACK);
        let mut out = [Vec3::ZERO; MAX_TRIANGLE_POINTS];
        let quad = [
            vec3(-1.0, 0.0, -1.0),
            vec3(-1.0, 0.0, 2.0),
            vec3(2.0, 0.0, 2.0),
            vec3(2.0, 0.0, -1.0),
        ];
        let count = clip_polygon(&quad, &planes, &mut out);
        assert!((3..=MAX_TRIANGLE_POINTS).contains(&count), "{count}");
        for point in &out[..count] {
            assert!(
                point.x >= -2.0 * EDGE_SLACK && point.z >= -2.0 * EDGE_SLACK,
                "{point:?}"
            );
            assert!(point.x + point.z <= 1.0 + 4.0 * EDGE_SLACK, "{point:?}");
        }
        // A quad entirely outside survives nothing.
        let away = [
            vec3(5.0, 0.0, 5.0),
            vec3(6.0, 0.0, 5.0),
            vec3(6.0, 0.0, 6.0),
            vec3(5.0, 0.0, 6.0),
        ];
        assert_eq!(clip_polygon(&away, &planes, &mut out), 0);
    }
}

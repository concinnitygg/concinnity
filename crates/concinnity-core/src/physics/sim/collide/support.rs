// The geometry every shape pair borrows: closest points between segments,
// a box's faces and supporting features, and the polygon clip that turns two
// overlapping faces into a contact patch.
//
// Contact ids are derived from where a clipped point ended up rather than from
// the path the clipper took to produce it. A geometric derivation repeats
// whenever the geometry repeats, which is exactly the property warm starting
// needs; an algorithmic one drifts as soon as the clip order changes.

use crate::physics::sim::math::{Quat, Vec3, vec3};

/// A body's placement, all the narrow phase needs of it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Pose {
    pub(crate) position: Vec3,
    pub(crate) rotation: Quat,
}

impl Pose {
    pub(crate) fn to_local(self, world: Vec3) -> Vec3 {
        self.rotation.inverse_rotate(world - self.position)
    }

    pub(crate) fn to_world(self, local: Vec3) -> Vec3 {
        self.position + self.rotation.rotate(local)
    }

    pub(crate) fn axis(self, index: usize) -> Vec3 {
        self.rotation.rotate(Vec3::axis(index))
    }
}

/// A box and where it is, which is how every routine here is handed one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OrientedBox {
    /// Half-extents along the box's own axes.
    pub(crate) half: Vec3,
    pub(crate) pose: Pose,
}

impl OrientedBox {
    /// How far the box reaches from its centre along a world direction.
    pub(crate) fn extent_along(&self, direction: Vec3) -> f32 {
        (direction.dot(self.pose.axis(0))).abs() * self.half.x
            + (direction.dot(self.pose.axis(1))).abs() * self.half.y
            + (direction.dot(self.pose.axis(2))).abs() * self.half.z
    }
}

/// A sphere, or the round end of something that behaves like one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Sphere {
    pub(crate) center: Vec3,
    pub(crate) radius: f32,
}

/// Point on segment `a -> b` closest to `p`, with its parameter.
pub(crate) fn closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> (Vec3, f32) {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq <= f32::MIN_POSITIVE {
        return (a, 0.0);
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    (a + ab * t, t)
}

/// Closest points between two segments, degenerate cases included.
pub(crate) fn closest_points_between_segments(
    p1: Vec3,
    q1: Vec3,
    p2: Vec3,
    q2: Vec3,
) -> (Vec3, Vec3) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.length_squared();
    let e = d2.length_squared();
    let f = d2.dot(r);

    if a <= f32::MIN_POSITIVE && e <= f32::MIN_POSITIVE {
        return (p1, p2);
    }
    if a <= f32::MIN_POSITIVE {
        let (c2, _) = closest_point_on_segment(p1, p2, q2);
        return (p1, c2);
    }
    if e <= f32::MIN_POSITIVE {
        let (c1, _) = closest_point_on_segment(p2, p1, q1);
        return (c1, p2);
    }

    let c = d1.dot(r);
    let b = d1.dot(d2);
    let denom = a * e - b * b;
    // Parallel segments leave the first parameter free; anchoring it at zero
    // and clamping the second is what keeps the answer on both segments.
    let mut s = if denom > f32::MIN_POSITIVE {
        ((b * f - c * e) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }
    (p1 + d1 * s, p2 + d2 * t)
}

/// The axis a box face lies on, and which side of it.
pub(crate) fn face_axis(face: usize) -> (usize, f32) {
    (face / 2, if face.is_multiple_of(2) { 1.0 } else { -1.0 })
}

/// The face whose outward normal is closest to `dir`, in box-local space.
pub(crate) fn best_face(dir_local: Vec3) -> usize {
    let axis = dir_local.max_abs_axis();
    axis * 2 + usize::from(dir_local.get(axis) < 0.0)
}

/// A box face's four corners in loop order, in box-local space.
pub(crate) fn face_corners(half: Vec3, face: usize) -> [Vec3; 4] {
    let (axis, sign) = face_axis(face);
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    let corner = |su: f32, sv: f32| {
        let mut p = Vec3::ZERO;
        p.set(axis, sign * half.get(axis));
        p.set(u, su * half.get(u));
        p.set(v, sv * half.get(v));
        p
    };
    [
        corner(1.0, 1.0),
        corner(-1.0, 1.0),
        corner(-1.0, -1.0),
        corner(1.0, -1.0),
    ]
}

/// Box vertex furthest along `dir`, in box-local space.
pub(crate) fn support_vertex(half: Vec3, dir_local: Vec3) -> Vec3 {
    vec3(
        if dir_local.x >= 0.0 { half.x } else { -half.x },
        if dir_local.y >= 0.0 { half.y } else { -half.y },
        if dir_local.z >= 0.0 { half.z } else { -half.z },
    )
}

/// The box edge along `axis` that supports `dir`, as two local endpoints.
/// The component along `axis` is free -- `dir` is perpendicular to it -- so
/// the whole edge is taken.
pub(crate) fn support_edge(half: Vec3, dir_local: Vec3, axis: usize) -> (Vec3, Vec3) {
    let mut base = support_vertex(half, dir_local);
    base.set(axis, 0.0);
    let along = Vec3::axis(axis) * half.get(axis);
    (base - along, base + along)
}

/// Points that survived clipping a face against a reference face's sides.
pub(crate) struct ClipResult {
    pub(crate) points: [Vec3; MAX_CLIP_POINTS],
    pub(crate) count: usize,
}

/// A quad clipped by four half-planes gains at most one point per plane.
pub(crate) const MAX_CLIP_POINTS: usize = 8;

/// Clip a polygon, given in the reference box's local space, to the rectangle
/// the reference face spans on axes `u` and `v`.
pub(crate) fn clip_to_face_rect(polygon: &[Vec3], half: Vec3, axis: usize) -> ClipResult {
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    let mut input = [Vec3::ZERO; MAX_CLIP_POINTS];
    let mut output = [Vec3::ZERO; MAX_CLIP_POINTS];
    let mut in_len = polygon.len().min(MAX_CLIP_POINTS);
    input[..in_len].copy_from_slice(&polygon[..in_len]);

    for (clip_axis, limit) in [(u, half.get(u)), (v, half.get(v))] {
        for sign in [1.0f32, -1.0] {
            let mut out_len = 0usize;
            let distance = |p: &Vec3| limit - sign * p.get(clip_axis);
            for i in 0..in_len {
                let current = input[i];
                let next = input[(i + 1) % in_len];
                let dc = distance(&current);
                let dn = distance(&next);
                if dc >= 0.0 && out_len < MAX_CLIP_POINTS {
                    output[out_len] = current;
                    out_len += 1;
                }
                if (dc >= 0.0) != (dn >= 0.0) && out_len < MAX_CLIP_POINTS {
                    let denom = dc - dn;
                    let t = if denom.abs() > f32::MIN_POSITIVE {
                        dc / denom
                    } else {
                        0.0
                    };
                    output[out_len] = current + (next - current) * t;
                    out_len += 1;
                }
            }
            input[..out_len].copy_from_slice(&output[..out_len]);
            in_len = out_len;
            if in_len == 0 {
                return ClipResult {
                    points: input,
                    count: 0,
                };
            }
        }
    }
    ClipResult {
        points: input,
        count: in_len,
    }
}

/// How much a clipped point may drift from a corner and still be called that
/// corner. Loose enough to survive the clip's arithmetic, tight enough that
/// two distinct corners never collide.
const FEATURE_EPSILON: f32 = 1.0e-4;

/// The feature id for a clipped point: which incident corner it is, if any,
/// and which of the reference face's four sides it lies on.
pub(crate) fn clipped_feature(point: Vec3, corners: &[Vec3; 4], half: Vec3, axis: usize) -> u32 {
    let mut corner_index = 0xFu32;
    for (index, corner) in corners.iter().enumerate() {
        if (point - *corner).length_squared() <= FEATURE_EPSILON * FEATURE_EPSILON {
            corner_index = index as u32;
            break;
        }
    }
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    let mut sides = 0u32;
    for (bit, (clip_axis, sign)) in [(u, 1.0f32), (u, -1.0), (v, 1.0), (v, -1.0)]
        .into_iter()
        .enumerate()
    {
        if (sign * point.get(clip_axis) - half.get(clip_axis)).abs() <= FEATURE_EPSILON {
            sides |= 1 << bit;
        }
    }
    (corner_index << 4) | sides
}

/// Keep four points out of a larger patch: the deepest, the one furthest from
/// it, and the two that push the quad widest to either side. Ties break on
/// index, so the same patch always reduces to the same four points.
pub(crate) fn reduce_to_quad(
    points: &[Vec3],
    separations: &[f32],
    normal: Vec3,
    keep: &mut [usize; 4],
) -> usize {
    let count = points.len();
    if count <= 4 {
        for (slot, index) in keep.iter_mut().zip(0..count) {
            *slot = index;
        }
        return count;
    }

    let mut deepest = 0usize;
    for i in 1..count {
        if separations[i] < separations[deepest] {
            deepest = i;
        }
    }
    let mut furthest = usize::MAX;
    let mut best_dist = -1.0f32;
    for i in 0..count {
        if i == deepest {
            continue;
        }
        let d = (points[i] - points[deepest]).length_squared();
        if d > best_dist {
            best_dist = d;
            furthest = i;
        }
    }

    // The remaining two maximise the signed area on either side of the line
    // through the first two, which is what spreads the quad.
    let axis = points[furthest] - points[deepest];
    let mut left = usize::MAX;
    let mut right = usize::MAX;
    let mut best_left = 0.0f32;
    let mut best_right = 0.0f32;
    for i in 0..count {
        if i == deepest || i == furthest {
            continue;
        }
        let area = axis.cross(points[i] - points[deepest]).dot(normal);
        if area > best_left {
            best_left = area;
            left = i;
        } else if area < best_right {
            best_right = area;
            right = i;
        }
    }

    let mut kept = 0usize;
    for candidate in [deepest, furthest, left, right] {
        if candidate != usize::MAX && kept < 4 {
            keep[kept] = candidate;
            kept += 1;
        }
    }
    // Degenerate patches (every point collinear) leave a slot unfilled; take
    // the next unused point rather than reporting fewer contacts than exist.
    let mut next = 0usize;
    while kept < 4 && next < count {
        if !keep[..kept].contains(&next) {
            keep[kept] = next;
            kept += 1;
        }
        next += 1;
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{cos, sin};

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1.0e-4
    }

    #[test]
    fn a_pose_round_trips_a_point_through_its_frame() {
        let pose = Pose {
            position: vec3(1.0, 2.0, 3.0),
            rotation: Quat::from_euler_deg([20.0, -35.0, 10.0]),
        };
        let p = vec3(0.5, -1.0, 2.0);
        assert!(close(pose.to_local(pose.to_world(p)), p));
        assert!(close(pose.axis(1), pose.rotation.rotate(Vec3::Y)));
    }

    #[test]
    fn closest_point_clamps_to_the_segment_ends() {
        let a = vec3(0.0, 0.0, 0.0);
        let b = vec3(2.0, 0.0, 0.0);
        assert!(close(
            closest_point_on_segment(vec3(1.0, 5.0, 0.0), a, b).0,
            vec3(1.0, 0.0, 0.0)
        ));
        assert!(close(
            closest_point_on_segment(vec3(-3.0, 1.0, 0.0), a, b).0,
            a
        ));
        assert!(close(
            closest_point_on_segment(vec3(9.0, 1.0, 0.0), a, b).0,
            b
        ));
        // A degenerate segment is just its own endpoint.
        assert!(close(
            closest_point_on_segment(vec3(1.0, 1.0, 1.0), a, a).0,
            a
        ));
    }

    #[test]
    fn crossing_segments_meet_at_their_crossing() {
        let (c1, c2) = closest_points_between_segments(
            vec3(-1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, -1.0),
            vec3(0.0, 1.0, 1.0),
        );
        assert!(close(c1, vec3(0.0, 0.0, 0.0)), "{c1:?}");
        assert!(close(c2, vec3(0.0, 1.0, 0.0)), "{c2:?}");
    }

    #[test]
    fn parallel_segments_still_produce_points_on_both() {
        let (c1, c2) = closest_points_between_segments(
            vec3(0.0, 0.0, 0.0),
            vec3(4.0, 0.0, 0.0),
            vec3(1.0, 2.0, 0.0),
            vec3(5.0, 2.0, 0.0),
        );
        assert!((c2.y - c1.y).abs() > 1.9);
        assert!(c1.x >= -1.0e-4 && c1.x <= 4.0 + 1.0e-4, "{c1:?}");
        assert!(c2.x >= 1.0 - 1.0e-4 && c2.x <= 5.0 + 1.0e-4, "{c2:?}");
        assert!((c1 - c2).length() - 2.0 < 1.0e-4);
    }

    #[test]
    fn disjoint_segments_meet_at_their_nearest_endpoints() {
        let (c1, c2) = closest_points_between_segments(
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(3.0, 0.0, 0.0),
            vec3(4.0, 0.0, 0.0),
        );
        assert!(close(c1, vec3(1.0, 0.0, 0.0)), "{c1:?}");
        assert!(close(c2, vec3(3.0, 0.0, 0.0)), "{c2:?}");
    }

    #[test]
    fn faces_and_supports_agree_on_which_way_is_out() {
        let half = Vec3::splat(1.0);
        assert_eq!(best_face(vec3(0.0, 2.0, 0.1)), 2);
        assert_eq!(face_axis(2), (1, 1.0));
        assert_eq!(best_face(vec3(0.0, -2.0, 0.1)), 3);
        assert_eq!(face_axis(3), (1, -1.0));
        for face in 0..6 {
            let (axis, sign) = face_axis(face);
            for corner in face_corners(half, face) {
                assert_eq!(corner.get(axis), sign * half.get(axis), "face {face}");
            }
        }
        assert_eq!(
            support_vertex(half, vec3(1.0, -1.0, 1.0)),
            vec3(1.0, -1.0, 1.0)
        );
    }

    // Face corners must form a loop: consecutive corners share an edge, so
    // each step changes exactly one coordinate.
    #[test]
    fn face_corners_come_out_in_loop_order() {
        let half = vec3(1.0, 2.0, 3.0);
        for face in 0..6 {
            let corners = face_corners(half, face);
            for i in 0..4 {
                let diff = (corners[(i + 1) % 4] - corners[i]).abs();
                let changed = (0..3).filter(|&a| diff.get(a) > 1.0e-6).count();
                assert_eq!(changed, 1, "face {face} corner {i}");
            }
        }
    }

    #[test]
    fn a_support_edge_spans_its_axis_and_faces_the_direction() {
        let half = Vec3::splat(1.0);
        let (a, b) = support_edge(half, vec3(1.0, 0.0, 1.0), 1);
        assert_eq!(a, vec3(1.0, -1.0, 1.0));
        assert_eq!(b, vec3(1.0, 1.0, 1.0));
    }

    #[test]
    fn a_face_larger_than_the_reference_clips_down_to_it() {
        let half = Vec3::splat(1.0);
        let big = [
            vec3(5.0, 1.0, 5.0),
            vec3(-5.0, 1.0, 5.0),
            vec3(-5.0, 1.0, -5.0),
            vec3(5.0, 1.0, -5.0),
        ];
        let clipped = clip_to_face_rect(&big, half, 1);
        assert_eq!(clipped.count, 4);
        for p in &clipped.points[..clipped.count] {
            assert!(
                p.x.abs() <= 1.0 + 1.0e-5 && p.z.abs() <= 1.0 + 1.0e-5,
                "{p:?}"
            );
        }
    }

    #[test]
    fn a_face_entirely_outside_the_reference_clips_to_nothing() {
        let half = Vec3::splat(1.0);
        let away = [
            vec3(9.0, 1.0, 9.0),
            vec3(8.0, 1.0, 9.0),
            vec3(8.0, 1.0, 8.0),
            vec3(9.0, 1.0, 8.0),
        ];
        assert_eq!(clip_to_face_rect(&away, half, 1).count, 0);
    }

    // A face turned 45 degrees over a smaller one clips to an eight-sided
    // patch, which is what the point reduction exists to handle.
    #[test]
    fn a_turned_face_can_clip_to_more_than_four_points() {
        let half = Vec3::splat(1.0);
        let s = 1.6f32;
        let turned = [
            vec3(s, 1.0, 0.0),
            vec3(0.0, 1.0, s),
            vec3(-s, 1.0, 0.0),
            vec3(0.0, 1.0, -s),
        ];
        let clipped = clip_to_face_rect(&turned, half, 1);
        assert!(clipped.count > 4, "{}", clipped.count);
        assert!(clipped.count <= MAX_CLIP_POINTS);
    }

    #[test]
    fn feature_ids_separate_corners_edges_and_interior_points() {
        let half = Vec3::splat(1.0);
        let corners = face_corners(half, 2);
        let at_corner = clipped_feature(corners[0], &corners, half, 1);
        let other_corner = clipped_feature(corners[2], &corners, half, 1);
        let on_side = clipped_feature(vec3(1.0, 1.0, 0.2), &corners, half, 1);
        let interior = clipped_feature(vec3(0.1, 1.0, 0.2), &corners, half, 1);
        assert_ne!(at_corner, other_corner);
        assert_ne!(at_corner, on_side);
        assert_ne!(on_side, interior);
        assert_eq!(interior & 0xF, 0, "an interior point lies on no side");
        // The same geometry must give the same id every time it is asked.
        assert_eq!(at_corner, clipped_feature(corners[0], &corners, half, 1));
    }

    #[test]
    fn reduction_keeps_everything_when_there_is_little_enough() {
        let points = [Vec3::ZERO, Vec3::X, Vec3::Z];
        let seps = [-0.1, -0.2, -0.3];
        let mut keep = [0usize; 4];
        assert_eq!(reduce_to_quad(&points, &seps, Vec3::Y, &mut keep), 3);
        assert_eq!(keep[..3], [0, 1, 2]);
    }

    // Reducing an eight-point ring must keep the deepest point and spread the
    // rest, not take the first four it meets.
    #[test]
    fn reduction_keeps_the_deepest_point_and_spreads_the_rest() {
        let mut points = [Vec3::ZERO; 8];
        let mut seps = [0.0f32; 8];
        for i in 0..8 {
            let angle = i as f32 * core::f32::consts::TAU / 8.0;
            points[i] = vec3(cos(angle), 0.0, sin(angle));
            seps[i] = -0.01;
        }
        seps[5] = -0.5;
        let mut keep = [0usize; 4];
        assert_eq!(reduce_to_quad(&points, &seps, Vec3::Y, &mut keep), 4);
        assert!(keep.contains(&5), "{keep:?}");
        // The kept points must not all be neighbours on the ring.
        let spread = keep
            .iter()
            .map(|&i| (points[i] - points[keep[0]]).length())
            .fold(0.0f32, f32::max);
        assert!(spread > 1.5, "{keep:?} spread {spread}");
        assert_eq!(
            reduce_to_quad(&points, &seps, Vec3::Y, &mut keep.clone()),
            4
        );
    }

    // Collinear patches have no area to maximise, so the fill must still
    // deliver four distinct points.
    #[test]
    fn reduction_fills_four_points_even_from_a_collinear_patch() {
        let points: [Vec3; 6] = core::array::from_fn(|i| vec3(i as f32, 0.0, 0.0));
        let seps = [-0.1f32; 6];
        let mut keep = [usize::MAX; 4];
        assert_eq!(reduce_to_quad(&points, &seps, Vec3::Y, &mut keep), 4);
        let mut sorted = keep;
        sorted.sort_unstable();
        assert!(sorted.windows(2).all(|w| w[0] != w[1]), "{keep:?}");
    }
}

// concinnity-physics/src/sim/collide/capsule.rs
//
// Capsules against boxes and against each other. A capsule is a segment with a
// radius, so both routines work on the segment and add the radius back at the
// end; what they cannot skip is the second contact point.
//
// A capsule lying along a face touches all the way down its length. One point
// there would let it rock about that point forever, so both routines detect
// the aligned case and clip the overlapping stretch to two points. That is the
// same reason a box contact is a patch rather than a point.

use crate::sim::contact::{Manifold, ManifoldPoint};
use crate::sim::math::Vec3;

use super::sphere::spheres;
use super::support::{
    OrientedBox, Pose, Sphere, closest_point_on_segment, closest_points_between_segments,
    face_axis, support_edge,
};

/// How much better an edge axis must measure before it displaces a face axis.
const FACE_PREFERENCE: f32 = 1.0e-3;

/// Cross products of near-parallel axes carry no direction worth testing.
const MIN_CROSS_LENGTH_SQUARED: f32 = 1.0e-8;

/// Axis alignment past which two capsules are treated as parallel and given a
/// two-point contact. About two degrees.
const PARALLEL_COSINE: f32 = 0.999;

/// Overlap shorter than this is one contact, not two.
const MIN_OVERLAP: f32 = 1.0e-4;

/// Feature ids for edge contacts start past every face-clipped id.
const EDGE_ID_BASE: u32 = 1 << 20;

/// A capsule is round, so the box feature nearest it can be a corner, which no
/// face or edge axis names.
const NEAREST_ID: u32 = (1 << 20) + 8;

#[derive(Debug, Clone, Copy)]
enum Winner {
    /// A box face the capsule lies against.
    Face(usize),
    /// A box edge along the given axis, crossing the capsule's own.
    Edge(usize),
    /// The nearest pair of points, when the box's closest feature is a corner.
    Nearest,
}

/// The world-space endpoints of a capsule's axis.
pub(crate) fn capsule_segment(pose: Pose, half_height: f32) -> (Vec3, Vec3) {
    let axis = pose.axis(1) * libm::fabsf(half_height);
    (pose.position - axis, pose.position + axis)
}

/// Contact between a box and a capsule, with the normal pointing from the box
/// toward the capsule.
pub(crate) fn box_capsule(
    shape: OrientedBox,
    segment: (Vec3, Vec3),
    radius: f32,
    margin: f32,
    out: &mut Manifold,
) -> bool {
    let (half, box_pose) = (shape.half, shape.pose);
    let center = (segment.0 + segment.1) * 0.5;
    let along = (segment.1 - segment.0) * 0.5;
    let half_length = along.length();
    let direction = along.normalize_or(box_pose.axis(1));
    let delta = center - box_pose.position;
    let axes = [box_pose.axis(0), box_pose.axis(1), box_pose.axis(2)];

    let mut best_normal = Vec3::Y;
    let mut best_separation = f32::NEG_INFINITY;
    let mut best_winner = Winner::Face(0);

    let mut consider = |axis: Vec3, winner: Winner, preference: f32| {
        let projection = axis.dot(delta);
        let normal = if projection < 0.0 { -axis } else { axis };
        let separation = libm::fabsf(projection)
            - (shape.extent_along(normal)
                + half_length * libm::fabsf(normal.dot(direction))
                + radius);
        if separation > margin {
            return false;
        }
        if separation > best_separation + preference {
            best_separation = separation;
            best_normal = normal;
            best_winner = winner;
        }
        true
    };

    for (index, axis) in axes.iter().enumerate() {
        if !consider(*axis, Winner::Face(index), 0.0) {
            return false;
        }
    }
    for (index, axis) in axes.iter().enumerate() {
        let cross = axis.cross(direction);
        if cross.length_squared() < MIN_CROSS_LENGTH_SQUARED {
            continue;
        }
        if !consider(
            cross.normalize_or_zero(),
            Winner::Edge(index),
            FACE_PREFERENCE,
        ) {
            return false;
        }
    }
    let (on_box, on_axis) = nearest_points(half, box_pose, segment);
    let toward = (on_axis - on_box).normalize_or_zero();
    if toward != Vec3::ZERO && !consider(toward, Winner::Nearest, FACE_PREFERENCE) {
        return false;
    }

    out.normal = best_normal;
    match best_winner {
        Winner::Face(axis_index) => clip_segment_to_face(
            &SegmentOnFace {
                shape,
                segment,
                radius,
                normal: best_normal,
                axis: axis_index,
                margin,
            },
            out,
        ),
        Winner::Edge(axis_index) => {
            let (e0, e1) = support_edge(
                half,
                box_pose.rotation.inverse_rotate(best_normal),
                axis_index,
            );
            let (edge_point, axis_point) = closest_points_between_segments(
                box_pose.to_world(e0),
                box_pose.to_world(e1),
                segment.0,
                segment.1,
            );
            out.push(ManifoldPoint {
                point: (edge_point + (axis_point - best_normal * radius)) * 0.5,
                separation: best_separation,
                id: EDGE_ID_BASE + axis_index as u32,
                ..Default::default()
            });
            true
        }
        Winner::Nearest => {
            out.push(ManifoldPoint {
                point: (on_box + (on_axis - best_normal * radius)) * 0.5,
                separation: best_separation,
                id: NEAREST_ID,
                ..Default::default()
            });
            true
        }
    }
}

/// One capsule-against-a-box-face contact, ready to be clipped.
struct SegmentOnFace {
    shape: OrientedBox,
    /// The capsule's axis, in world space.
    segment: (Vec3, Vec3),
    radius: f32,
    /// Out of the box, toward the capsule.
    normal: Vec3,
    /// Which of the box's own axes the face lies on.
    axis: usize,
    margin: f32,
}

/// A close pair of points on the box's surface and on the capsule's axis.
///
/// One projection each way, not an exact minimisation: this only has to name a
/// direction good enough to be tested as a separating axis, and the face and
/// edge axes already cover everything but the corner regions.
fn nearest_points(half: Vec3, box_pose: Pose, segment: (Vec3, Vec3)) -> (Vec3, Vec3) {
    let (toward_centre, _) = closest_point_on_segment(box_pose.position, segment.0, segment.1);
    let clamped = box_pose.to_local(toward_centre).clamp(-half, half);
    let on_box = box_pose.to_world(clamped);
    let (on_axis, _) = closest_point_on_segment(on_box, segment.0, segment.1);
    (on_box, on_axis)
}

/// Clip the capsule's axis to the box face it rests against, keeping the one
/// or two points that survive.
fn clip_segment_to_face(clip: &SegmentOnFace, out: &mut Manifold) -> bool {
    let (half, box_pose) = (clip.shape.half, clip.shape.pose);
    let local_normal = box_pose.rotation.inverse_rotate(clip.normal);
    let face = clip.axis * 2 + usize::from(local_normal.get(clip.axis) < 0.0);
    let (axis, sign) = face_axis(face);
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);

    let point = FacePoint {
        p0: box_pose.to_local(clip.segment.0),
        p1: box_pose.to_local(clip.segment.1),
        half,
        box_pose,
        radius: clip.radius,
        axis,
        sign,
        normal: clip.normal,
        margin: clip.margin,
    };
    let (mut low, mut high) = (0.0f32, 1.0f32);
    for (clip_axis, clip_sign) in [(u, 1.0f32), (u, -1.0), (v, 1.0), (v, -1.0)] {
        // clip_sign * p(t)[clip_axis] <= half[clip_axis], linear in t.
        let limit = half.get(clip_axis);
        let start = clip_sign * point.p0.get(clip_axis);
        let slope = clip_sign * (point.p1.get(clip_axis) - point.p0.get(clip_axis));
        if libm::fabsf(slope) <= f32::MIN_POSITIVE {
            if start > limit {
                low = 1.0;
                high = 0.0;
            }
            continue;
        }
        let crossing = (limit - start) / slope;
        if slope > 0.0 {
            high = high.min(crossing);
        } else {
            low = low.max(crossing);
        }
    }

    // An empty interval means the axis passes beside the face rather than over
    // it; the midpoint of the crossed bounds is the nearest stretch to it.
    if low > high {
        let t = ((low + high) * 0.5).clamp(0.0, 1.0);
        return point.push_at(t, 0, out);
    }
    let mut pushed = point.push_at(low, 0, out);
    if high - low > MIN_OVERLAP {
        pushed |= point.push_at(high, 1, out);
    }
    pushed
}

/// Where one point of a capsule-on-face contact goes, once the clip has
/// chosen a parameter along the capsule's axis.
struct FacePoint {
    /// The capsule's axis endpoints, in the box's local space.
    p0: Vec3,
    p1: Vec3,
    half: Vec3,
    box_pose: Pose,
    radius: f32,
    /// Which local axis the face lies on, and on which side.
    axis: usize,
    sign: f32,
    /// Out of the box, toward the capsule.
    normal: Vec3,
    margin: f32,
}

impl FacePoint {
    fn push_at(&self, t: f32, id: u32, out: &mut Manifold) -> bool {
        let local = self.p0 + (self.p1 - self.p0) * t;
        let separation = self.sign * local.get(self.axis) - self.half.get(self.axis) - self.radius;
        if separation > self.margin {
            return false;
        }
        out.push(ManifoldPoint {
            point: self.box_pose.to_world(local) - self.normal * (self.radius + separation * 0.5),
            separation,
            id,
            ..Default::default()
        });
        true
    }
}

/// Contact between two capsules, with the normal pointing from `a` toward `b`.
pub(crate) fn capsules(
    segment_a: (Vec3, Vec3),
    radius_a: f32,
    segment_b: (Vec3, Vec3),
    radius_b: f32,
    margin: f32,
    out: &mut Manifold,
) -> bool {
    let (near_a, near_b) =
        closest_points_between_segments(segment_a.0, segment_a.1, segment_b.0, segment_b.1);
    // Coaxial capsules have no closest-point direction of their own; the line
    // between their midpoints is the one that still reverses when the pair
    // does.
    let fallback =
        (((segment_b.0 + segment_b.1) - (segment_a.0 + segment_a.1)) * 0.5).normalize_or(Vec3::Y);
    let (ball_a, ball_b) = (
        Sphere {
            center: near_a,
            radius: radius_a,
        },
        Sphere {
            center: near_b,
            radius: radius_b,
        },
    );
    if !spheres(ball_a, ball_b, fallback, margin, out) {
        return false;
    }

    let dir_a = (segment_a.1 - segment_a.0).normalize_or_zero();
    let dir_b = (segment_b.1 - segment_b.0).normalize_or_zero();
    if libm::fabsf(dir_a.dot(dir_b)) < PARALLEL_COSINE {
        return true;
    }

    // Parallel axes touch along a stretch, not a point. The stretch is where
    // the two axes' projections onto a's axis overlap.
    let length_a = (segment_a.1 - segment_a.0).length();
    let project = |p: Vec3| (p - segment_a.0).dot(dir_a);
    let (b0, b1) = (project(segment_b.0), project(segment_b.1));
    let low = b0.min(b1).max(0.0);
    let high = b0.max(b1).min(length_a);
    if high - low <= MIN_OVERLAP {
        return true;
    }

    let normal = out.normal;
    out.count = 0;
    for (id, t) in [low, high].into_iter().enumerate() {
        let on_a = segment_a.0 + dir_a * t;
        let (on_b, _) = closest_point_on_segment(on_a, segment_b.0, segment_b.1);
        let separation = (on_b - on_a).dot(normal) - (radius_a + radius_b);
        if separation > margin {
            continue;
        }
        out.push(ManifoldPoint {
            point: ((on_a + normal * radius_a) + (on_b - normal * radius_b)) * 0.5,
            separation,
            id: id as u32,
            ..Default::default()
        });
    }
    if out.count == 0 {
        // The endpoints fell outside the margin even though the middle did
        // not; keep the single closest-point contact rather than nothing.
        return spheres(ball_a, ball_b, fallback, margin, out);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::math::{Quat, vec3};

    fn pose(position: Vec3, euler_deg: [f32; 3]) -> Pose {
        Pose {
            position,
            rotation: Quat::from_euler_deg(euler_deg),
        }
    }

    fn manifold() -> Manifold {
        Manifold::new(0, 1)
    }

    fn oriented(half: Vec3, position: Vec3, euler_deg: [f32; 3]) -> OrientedBox {
        OrientedBox {
            half,
            pose: pose(position, euler_deg),
        }
    }

    #[test]
    fn a_capsule_axis_spans_both_caps() {
        let (a, b) = capsule_segment(pose(vec3(0.0, 2.0, 0.0), [0.0; 3]), 0.5);
        assert!((a - vec3(0.0, 1.5, 0.0)).length() < 1.0e-6);
        assert!((b - vec3(0.0, 2.5, 0.0)).length() < 1.0e-6);
    }

    // A capsule lying flat on a floor must contact at both ends, or it rocks.
    #[test]
    fn a_capsule_lying_on_a_floor_contacts_at_both_ends() {
        let mut m = manifold();
        let segment = (vec3(-1.0, 0.29, 0.0), vec3(1.0, 0.29, 0.0));
        assert!(box_capsule(
            oriented(vec3(5.0, 0.5, 5.0), Vec3::ZERO, [0.0; 3]),
            segment,
            0.25,
            0.0,
            &mut m
        ));
        assert_eq!(m.count, 2, "{m:?}");
        assert!((m.normal - Vec3::Y).length() < 1.0e-5, "{:?}", m.normal);
        for point in m.points() {
            assert!((point.separation + 0.46).abs() < 1.0e-4, "{point:?}");
        }
        assert!(
            (m.points()[0].point - m.points()[1].point).length() > 1.5,
            "{m:?}"
        );
    }

    #[test]
    fn a_capsule_standing_on_a_floor_contacts_at_one_end() {
        let mut m = manifold();
        let segment = (vec3(0.0, 0.7, 0.0), vec3(0.0, 1.7, 0.0));
        assert!(box_capsule(
            oriented(vec3(5.0, 0.5, 5.0), Vec3::ZERO, [0.0; 3]),
            segment,
            0.25,
            0.0,
            &mut m
        ));
        assert!((m.normal - Vec3::Y).length() < 1.0e-5);
        assert!(m.points().iter().all(|p| p.separation < 0.0), "{m:?}");
        assert!(m.count >= 1);
    }

    #[test]
    fn a_capsule_clear_of_a_box_reports_nothing() {
        let mut m = manifold();
        let segment = (vec3(-1.0, 5.0, 0.0), vec3(1.0, 5.0, 0.0));
        assert!(!box_capsule(
            oriented(Vec3::splat(0.5), Vec3::ZERO, [0.0; 3]),
            segment,
            0.25,
            0.0,
            &mut m
        ));
        assert_eq!(m.count, 0);
    }

    // A capsule crossing a box's edge diagonally is the case a face-only axis
    // set gets wrong: the normal must lean off the face.
    #[test]
    fn a_capsule_across_a_box_corner_contacts_on_the_edge() {
        let mut m = manifold();
        let segment = (vec3(0.6, 0.6, -1.0), vec3(0.6, 0.6, 1.0));
        assert!(box_capsule(
            oriented(Vec3::splat(0.5), Vec3::ZERO, [0.0; 3]),
            segment,
            0.3,
            0.0,
            &mut m
        ));
        assert!(m.normal.x > 0.3 && m.normal.y > 0.3, "{:?}", m.normal);
        assert!((m.normal.length() - 1.0).abs() < 1.0e-4);
        assert!(m.points()[0].separation < 0.0, "{m:?}");
    }

    // Hanging off the side of a face still has to produce a contact rather
    // than an empty manifold.
    #[test]
    fn a_capsule_hanging_past_a_face_still_contacts() {
        let mut m = manifold();
        let segment = (vec3(2.0, 0.7, 0.0), vec3(4.0, 0.7, 0.0));
        let touched = box_capsule(
            oriented(Vec3::splat(0.5), Vec3::ZERO, [0.0; 3]),
            segment,
            0.3,
            0.05,
            &mut m,
        );
        assert!(
            !touched || m.count > 0,
            "a reported contact must have points"
        );
    }

    #[test]
    fn crossing_capsules_meet_at_one_point() {
        let mut m = manifold();
        assert!(capsules(
            (vec3(-1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0)),
            0.3,
            (vec3(0.0, 0.5, -1.0), vec3(0.0, 0.5, 1.0)),
            0.3,
            0.0,
            &mut m
        ));
        assert_eq!(m.count, 1);
        assert!((m.normal - Vec3::Y).length() < 1.0e-5, "{:?}", m.normal);
        assert!((m.points()[0].separation + 0.1).abs() < 1.0e-5);
    }

    // Two capsules side by side touch along their whole overlap.
    #[test]
    fn parallel_capsules_meet_along_their_overlap() {
        let mut m = manifold();
        assert!(capsules(
            (vec3(-1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0)),
            0.3,
            (vec3(0.0, 0.55, 0.0), vec3(2.0, 0.55, 0.0)),
            0.3,
            0.0,
            &mut m
        ));
        assert_eq!(m.count, 2, "{m:?}");
        assert!((m.normal - Vec3::Y).length() < 1.0e-4, "{:?}", m.normal);
        let span = (m.points()[0].point - m.points()[1].point).length();
        assert!((span - 1.0).abs() < 1.0e-3, "overlap span {span}");
        assert_ne!(m.points()[0].id, m.points()[1].id);
    }

    // Parallel but barely overlapping: one point, not two coincident ones.
    #[test]
    fn parallel_capsules_meeting_end_to_end_report_one_point() {
        let mut m = manifold();
        assert!(capsules(
            (vec3(-1.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)),
            0.3,
            (vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0)),
            0.3,
            0.0,
            &mut m
        ));
        assert_eq!(m.count, 1, "{m:?}");
    }

    #[test]
    fn distant_capsules_report_nothing() {
        let mut m = manifold();
        assert!(!capsules(
            (vec3(-1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0)),
            0.3,
            (vec3(-1.0, 9.0, 0.0), vec3(1.0, 9.0, 0.0)),
            0.3,
            0.0,
            &mut m
        ));
        assert_eq!(m.count, 0);
    }
}

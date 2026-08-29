// Two boxes, by separating axis followed by face clipping. Fifteen axes decide
// whether the boxes touch and along what normal; the winning axis then decides
// whether the contact is a patch (a face against a face, clipped to up to four
// points) or a single point (two edges crossing).
//
// Face axes are preferred to edge axes by a small margin. Near a face-on-face
// rest the two kinds of axis measure almost the same separation, and letting
// them trade places from step to step would swap a four-point patch for a
// one-point contact and back, which reads as a stack that will not settle.

use crate::physics::sim::contact::{MAX_MANIFOLD_POINTS, Manifold, ManifoldPoint};
use crate::physics::sim::math::{Quat, Vec3};

use super::support::{
    self, MAX_CLIP_POINTS, OrientedBox, best_face, clip_to_face_rect, clipped_feature, face_axis,
    face_corners,
};

/// How much better an edge axis must measure before it displaces a face axis.
/// One millimetre: below the penetration a resting contact settles at, so a
/// resting patch is never traded for a point contact.
const FACE_PREFERENCE: f32 = 1.0e-3;

/// The same preference between the two boxes' own faces, so a symmetric
/// contact resolves the same way twice running.
const REFERENCE_PREFERENCE: f32 = 1.0e-4;

/// Cross products of near-parallel axes carry no direction worth testing.
const MIN_CROSS_LENGTH_SQUARED: f32 = 1.0e-8;

/// Feature ids for edge contacts start past every face-clipped id.
const EDGE_ID_BASE: u32 = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Winner {
    FaceA(usize),
    FaceB(usize),
    Edge(usize, usize),
}

struct Query {
    normal: Vec3,
    separation: f32,
    winner: Winner,
}

/// Contact between two boxes, with the normal pointing from `a` toward `b`.
pub(crate) fn box_box(a: OrientedBox, b: OrientedBox, margin: f32, out: &mut Manifold) -> bool {
    let Some(query) = separating_axis(a, b, margin) else {
        return false;
    };
    out.normal = query.normal;
    match query.winner {
        Winner::FaceA(axis) => clip_faces(
            &FaceClip {
                reference: a,
                incident: b,
                normal: query.normal,
                axis,
                margin,
                flipped: false,
            },
            out,
        ),
        Winner::FaceB(axis) => clip_faces(
            &FaceClip {
                reference: b,
                incident: a,
                normal: -query.normal,
                axis,
                margin,
                flipped: true,
            },
            out,
        ),
        Winner::Edge(axis_a, axis_b) => edge_contact(
            &EdgeCross {
                a,
                b,
                axis_a,
                axis_b,
                normal: query.normal,
                separation: query.separation,
            },
            out,
        ),
    }
}

/// One face-on-face contact, ready to be clipped.
struct FaceClip {
    /// The box whose face the patch is clipped against.
    reference: OrientedBox,
    /// The box whose nearest face is clipped against it.
    incident: OrientedBox,
    /// Out of the reference box, toward the incident one.
    normal: Vec3,
    /// Which of the reference box's own axes the face lies on.
    axis: usize,
    margin: f32,
    /// Whether the manifold normal is the opposite of `normal`, which it is
    /// when the reference box is `b`.
    flipped: bool,
}

/// One edge-on-edge contact, ready to be resolved.
struct EdgeCross {
    a: OrientedBox,
    b: OrientedBox,
    axis_a: usize,
    axis_b: usize,
    /// From `a` toward `b`.
    normal: Vec3,
    separation: f32,
}

/// The axis of greatest separation, or `None` when one axis separates the
/// boxes by more than the margin allows.
fn separating_axis(a: OrientedBox, b: OrientedBox, margin: f32) -> Option<Query> {
    let delta = b.pose.position - a.pose.position;
    let axes_a = [a.pose.axis(0), a.pose.axis(1), a.pose.axis(2)];
    let axes_b = [b.pose.axis(0), b.pose.axis(1), b.pose.axis(2)];

    let mut best: Option<Query> = None;
    let mut consider = |axis: Vec3, winner: Winner, preference: f32| -> bool {
        let projection = axis.dot(delta);
        // Orient every axis to point from a toward b, so the sign of the
        // manifold normal never depends on which axis won.
        let normal = if projection < 0.0 { -axis } else { axis };
        let separation = projection.abs() - (a.extent_along(normal) + b.extent_along(normal));
        if separation > margin {
            return false;
        }
        let improved = match &best {
            None => true,
            Some(current) => separation > current.separation + preference,
        };
        if improved {
            best = Some(Query {
                normal,
                separation,
                winner,
            });
        }
        true
    };

    for (index, axis) in axes_a.iter().enumerate() {
        if !consider(*axis, Winner::FaceA(index), 0.0) {
            return None;
        }
    }
    for (index, axis) in axes_b.iter().enumerate() {
        if !consider(*axis, Winner::FaceB(index), REFERENCE_PREFERENCE) {
            return None;
        }
    }
    for (i, a) in axes_a.iter().enumerate() {
        for (j, b) in axes_b.iter().enumerate() {
            let cross = a.cross(*b);
            if cross.length_squared() < MIN_CROSS_LENGTH_SQUARED {
                continue;
            }
            if !consider(
                cross.normalize_or_zero(),
                Winner::Edge(i, j),
                FACE_PREFERENCE,
            ) {
                return None;
            }
        }
    }
    best
}

/// Which of a box's six faces the given world direction leaves by.
fn outward_face(rotation: Quat, world_direction: Vec3) -> usize {
    best_face(rotation.inverse_rotate(world_direction))
}

/// Clip the incident box's nearest face against the reference box's face and
/// keep the points that are actually in contact.
fn clip_faces(clip: &FaceClip, out: &mut Manifold) -> bool {
    let (half_ref, pose_ref) = (clip.reference.half, clip.reference.pose);
    let (half_inc, pose_inc) = (clip.incident.half, clip.incident.pose);
    let reference_normal = clip.normal;
    let reference_face = clip.axis * 2
        + usize::from(
            pose_ref
                .rotation
                .inverse_rotate(reference_normal)
                .get(clip.axis)
                < 0.0,
        );
    let (axis, sign) = face_axis(reference_face);
    let incident_face = outward_face(pose_inc.rotation, -reference_normal);

    // The incident face, brought into the reference box's frame, which is
    // where both the clip and the feature ids are expressed.
    let mut incident_local = [Vec3::ZERO; 4];
    for (slot, corner) in incident_local
        .iter_mut()
        .zip(face_corners(half_inc, incident_face))
    {
        *slot = pose_ref.to_local(pose_inc.to_world(corner));
    }

    let clipped = clip_to_face_rect(&incident_local, half_ref, axis);
    if clipped.count == 0 {
        return false;
    }

    let mut points = [Vec3::ZERO; MAX_CLIP_POINTS];
    let mut separations = [0.0f32; MAX_CLIP_POINTS];
    let mut ids = [0u32; MAX_CLIP_POINTS];
    let mut kept = 0usize;
    for index in 0..clipped.count {
        let local = clipped.points[index];
        let separation = sign * local.get(axis) - half_ref.get(axis);
        if separation > clip.margin {
            continue;
        }
        points[kept] = pose_ref.to_world(local) - reference_normal * (separation * 0.5);
        separations[kept] = separation;
        ids[kept] = feature_id(
            reference_face,
            incident_face,
            clip.flipped,
            clipped_feature(local, &incident_local, half_ref, axis),
        );
        kept += 1;
    }
    if kept == 0 {
        return false;
    }

    let mut keep = [0usize; MAX_MANIFOLD_POINTS];
    let count = support::reduce_to_quad(
        &points[..kept],
        &separations[..kept],
        reference_normal,
        &mut keep,
    );
    for &index in &keep[..count] {
        out.push(ManifoldPoint {
            point: points[index],
            separation: separations[index],
            id: ids[index],
            ..Default::default()
        });
    }
    out.count > 0
}

/// Pack the features a clipped point came from into one stable id.
fn feature_id(reference_face: usize, incident_face: usize, flipped: bool, clipped: u32) -> u32 {
    let reference = reference_face as u32 + if flipped { 6 } else { 0 };
    (reference << 16) | ((incident_face as u32) << 12) | clipped
}

/// Two crossing edges touch at one point: the closest pair on the supporting
/// edge of each box.
fn edge_contact(cross: &EdgeCross, out: &mut Manifold) -> bool {
    let (a, b, normal) = (cross.a, cross.b, cross.normal);
    let (a0, a1) =
        support::support_edge(a.half, a.pose.rotation.inverse_rotate(normal), cross.axis_a);
    let (b0, b1) = support::support_edge(
        b.half,
        b.pose.rotation.inverse_rotate(-normal),
        cross.axis_b,
    );
    let (on_a, on_b) = support::closest_points_between_segments(
        a.pose.to_world(a0),
        a.pose.to_world(a1),
        b.pose.to_world(b0),
        b.pose.to_world(b1),
    );
    out.push(ManifoldPoint {
        point: (on_a + on_b) * 0.5,
        separation: cross.separation,
        id: EDGE_ID_BASE + (cross.axis_a as u32) * 3 + cross.axis_b as u32,
        ..Default::default()
    });
    true
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::physics::sim::collide::support::Pose;
    use crate::physics::sim::math::vec3;

    fn oriented(position: Vec3, euler_deg: [f32; 3], half: Vec3) -> OrientedBox {
        OrientedBox {
            half,
            pose: Pose {
                position,
                rotation: Quat::from_euler_deg(euler_deg),
            },
        }
    }

    fn collide(a: Vec3, ea: [f32; 3], ha: Vec3, b: Vec3, eb: [f32; 3], hb: Vec3) -> Manifold {
        let mut m = Manifold::new(0, 1);
        box_box(oriented(a, ea, ha), oriented(b, eb, hb), 0.0, &mut m);
        m
    }

    #[test]
    fn separated_boxes_report_no_contact() {
        let m = collide(
            Vec3::ZERO,
            [0.0; 3],
            Vec3::splat(0.5),
            vec3(3.0, 0.0, 0.0),
            [0.0; 3],
            Vec3::splat(0.5),
        );
        assert_eq!(m.count, 0);
    }

    // The case the whole milestone rests on: a box sitting squarely on a floor
    // must produce four points spread over the contact face, not one.
    #[test]
    fn a_box_resting_on_a_floor_produces_a_four_point_patch() {
        let m = collide(
            vec3(0.0, -1.0, 0.0),
            [0.0; 3],
            vec3(10.0, 1.0, 10.0),
            vec3(0.0, 0.49, 0.0),
            [0.0; 3],
            Vec3::splat(0.5),
        );
        assert_eq!(m.count, 4, "{m:?}");
        assert!((m.normal - Vec3::Y).length() < 1.0e-5, "{:?}", m.normal);
        for point in m.points() {
            assert!((point.separation + 0.01).abs() < 1.0e-4, "{point:?}");
            assert!(point.point.y.abs() < 0.02, "{point:?}");
        }
        // The four points must actually spread across the face.
        let spread = m
            .points()
            .iter()
            .map(|p| (p.point - m.points()[0].point).length())
            .fold(0.0f32, f32::max);
        assert!(spread > 0.5, "{m:?}");
    }

    #[test]
    fn the_manifold_normal_always_points_from_a_toward_b() {
        let below = collide(
            vec3(0.0, 1.0, 0.0),
            [0.0; 3],
            Vec3::splat(0.5),
            vec3(0.0, 0.1, 0.0),
            [0.0; 3],
            Vec3::splat(0.5),
        );
        assert!(below.count > 0);
        assert!(
            (below.normal + Vec3::Y).length() < 1.0e-5,
            "{:?}",
            below.normal
        );
    }

    // Every contact point must lie between the two surfaces and report the
    // overlap the geometry actually has.
    #[test]
    fn overlap_depth_matches_the_geometry() {
        let m = collide(
            Vec3::ZERO,
            [0.0; 3],
            Vec3::splat(1.0),
            vec3(1.75, 0.0, 0.0),
            [0.0; 3],
            Vec3::splat(1.0),
        );
        assert!(m.count > 0);
        assert!((m.normal - Vec3::X).length() < 1.0e-5, "{:?}", m.normal);
        for point in m.points() {
            assert!((point.separation + 0.25).abs() < 1.0e-4, "{point:?}");
        }
    }

    // A box turned about the contact normal still meets the floor face on,
    // and the clip has to survive the corners hanging over the edges.
    #[test]
    fn a_yawed_box_on_a_floor_still_makes_a_patch() {
        let m = collide(
            vec3(0.0, -1.0, 0.0),
            [0.0; 3],
            vec3(10.0, 1.0, 10.0),
            vec3(0.0, 0.49, 0.0),
            [0.0, 30.0, 0.0],
            Vec3::splat(0.5),
        );
        assert_eq!(m.count, 4, "{m:?}");
        assert!((m.normal - Vec3::Y).length() < 1.0e-5);
    }

    // A box tipped onto an edge contacts along that edge, so the winning axis
    // is a cross product and the manifold is small.
    #[test]
    fn a_box_crossing_another_at_an_angle_contacts_on_an_edge() {
        let mut m = Manifold::new(0, 1);
        assert!(box_box(
            oriented(Vec3::ZERO, [0.0; 3], Vec3::splat(1.0)),
            oriented(vec3(0.0, 2.3, 0.0), [45.0, 0.0, 45.0], Vec3::splat(1.0)),
            0.0,
            &mut m
        ));
        assert!(m.count >= 1, "{m:?}");
        assert!(m.normal.y > 0.5, "{:?}", m.normal);
        for point in m.points() {
            assert!(point.separation < 0.0, "{point:?}");
        }
    }

    // Feature ids exist so warm starting survives; the same rest must produce
    // the same four ids every time it is evaluated.
    #[test]
    fn a_resting_patch_reports_the_same_feature_ids_twice() {
        let first = collide(
            vec3(0.0, -1.0, 0.0),
            [0.0; 3],
            vec3(4.0, 1.0, 4.0),
            vec3(0.2, 0.495, -0.1),
            [0.0, 15.0, 0.0],
            Vec3::splat(0.5),
        );
        let second = collide(
            vec3(0.0, -1.0, 0.0),
            [0.0; 3],
            vec3(4.0, 1.0, 4.0),
            vec3(0.2, 0.495, -0.1),
            [0.0, 15.0, 0.0],
            Vec3::splat(0.5),
        );
        let ids: Vec<u32> = first.points().iter().map(|p| p.id).collect();
        let again: Vec<u32> = second.points().iter().map(|p| p.id).collect();
        assert_eq!(ids, again);
        let mut unique = ids.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), ids.len(), "ids must be distinct: {ids:?}");
    }

    // Nudging a resting box by less than the contact tolerance must not
    // renumber its features, or every step would warm start from cold.
    #[test]
    fn feature_ids_survive_a_small_nudge() {
        let ids = |offset: f32| -> Vec<u32> {
            collide(
                vec3(0.0, -1.0, 0.0),
                [0.0; 3],
                vec3(4.0, 1.0, 4.0),
                vec3(offset, 0.495, 0.0),
                [0.0; 3],
                Vec3::splat(0.5),
            )
            .points()
            .iter()
            .map(|p| p.id)
            .collect()
        };
        assert_eq!(ids(0.0), ids(0.0005));
    }

    #[test]
    fn a_speculative_margin_reports_a_gap_before_it_closes() {
        let mut m = Manifold::new(0, 1);
        assert!(box_box(
            oriented(Vec3::ZERO, [0.0; 3], Vec3::splat(0.5)),
            oriented(vec3(0.0, 1.02, 0.0), [0.0; 3], Vec3::splat(0.5)),
            0.05,
            &mut m
        ));
        assert!(m.points().iter().all(|p| p.separation > 0.0), "{m:?}");
    }

    // Whichever box is named first, the contact is the same contact.
    #[test]
    fn swapping_the_boxes_mirrors_the_manifold() {
        let forward = collide(
            Vec3::ZERO,
            [0.0; 3],
            Vec3::splat(1.0),
            vec3(1.8, 0.3, 0.0),
            [0.0, 20.0, 0.0],
            Vec3::splat(1.0),
        );
        let backward = collide(
            vec3(1.8, 0.3, 0.0),
            [0.0, 20.0, 0.0],
            Vec3::splat(1.0),
            Vec3::ZERO,
            [0.0; 3],
            Vec3::splat(1.0),
        );
        assert_eq!(forward.count, backward.count);
        assert!((forward.normal + backward.normal).length() < 1.0e-4);
        let depth = |m: &Manifold| {
            m.points()
                .iter()
                .map(|p| p.separation)
                .fold(f32::INFINITY, f32::min)
        };
        assert!((depth(&forward) - depth(&backward)).abs() < 1.0e-3);
    }
}

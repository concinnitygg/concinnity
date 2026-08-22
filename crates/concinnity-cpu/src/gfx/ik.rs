//! Analytic two-bone inverse kinematics: bend a root-mid-end joint chain (a
//! leg or an arm) so the end joint lands on a target, with a pole vector
//! picking the bend side. Operates on the sampled local pose matrices after
//! blending and before `skinning_matrices`, so the solve composes with any
//! animation. Pure math, no ECS or backend types.

use crate::geometry::vec3::{add, cross, dot, length, scale, sub};

use crate::gfx::skinning::{Mat4, Skeleton, mat4_affine_inverse, mat4_mul};

type Vec3 = [f32; 3];
type Mat3 = [[f32; 3]; 3];

const EPS: f32 = 1.0e-5;

fn normalize(v: Vec3) -> Option<Vec3> {
    let len = length(v);
    (len > EPS).then(|| scale(v, 1.0 / len))
}

// Rodrigues rotation: column-major 3x3 rotating about the unit `axis` by
// `angle` radians.
fn rotate_about(axis: Vec3, angle: f32) -> Mat3 {
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    let [x, y, z] = axis;
    [
        [t * x * x + c, t * x * y + s * z, t * x * z - s * y],
        [t * x * y - s * z, t * y * y + c, t * y * z + s * x],
        [t * x * z + s * y, t * y * z - s * x, t * z * z + c],
    ]
}

const MAT3_IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

// Shortest-arc rotation taking direction `a` onto direction `b` (inputs need
// not be unit length). Identity when either is degenerate or they already
// align; a half-turn about any perpendicular when they oppose.
fn from_to(a: Vec3, b: Vec3) -> Mat3 {
    let (Some(a), Some(b)) = (normalize(a), normalize(b)) else {
        return MAT3_IDENTITY;
    };
    let c = cross(a, b);
    let angle = length(c).atan2(dot(a, b));
    match normalize(c) {
        Some(axis) => rotate_about(axis, angle),
        // Parallel or anti-parallel: no rotation, or a half-turn about any
        // axis perpendicular to `a`.
        None if dot(a, b) > 0.0 => MAT3_IDENTITY,
        None => rotate_about(any_perpendicular(a), core::f32::consts::PI),
    }
}

// Some unit vector perpendicular to unit `v`.
fn any_perpendicular(v: Vec3) -> Vec3 {
    let candidate = if v[0].abs() < 0.9 {
        cross(v, [1.0, 0.0, 0.0])
    } else {
        cross(v, [0.0, 1.0, 0.0])
    };
    normalize(candidate).unwrap_or([0.0, 0.0, 1.0])
}

fn mat3_apply(m: Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

fn mat3_mul(a: Mat3, b: Mat3) -> Mat3 {
    let mut out = [[0.0f32; 3]; 3];
    for col in 0..3 {
        for row in 0..3 {
            for k in 0..3 {
                out[col][row] += a[k][row] * b[col][k];
            }
        }
    }
    out
}

// Rotate a full affine matrix about a world-space pivot point: the linear
// part is premultiplied by `r`, the translation orbits the pivot.
fn rotate_mat4_about(r: Mat3, pivot: Vec3, m: Mat4) -> Mat4 {
    let mut out = m;
    for col in 0..3 {
        let rotated = mat3_apply(r, [m[col][0], m[col][1], m[col][2]]);
        out[col][0] = rotated[0];
        out[col][1] = rotated[1];
        out[col][2] = rotated[2];
    }
    let p = mat3_apply(r, sub([m[3][0], m[3][1], m[3][2]], pivot));
    out[3][0] = pivot[0] + p[0];
    out[3][1] = pivot[1] + p[1];
    out[3][2] = pivot[2] + p[2];
    out
}

/// One two-bone chain resolved to joint indices. `mid` must be the direct
/// child of `root` and `end` the direct child of `mid` (the local-pose
/// write-back assumes it); `pole` is the bend direction in mesh space.
#[derive(Debug, Clone)]
pub struct TwoBoneChain {
    /// Index of the chain's root joint.
    pub root: usize,
    /// Index of the middle joint, a direct child of `root`.
    pub mid: usize,
    /// Index of the end joint, a direct child of `mid`.
    pub end: usize,
    /// Bend direction in mesh space.
    pub pole: Vec3,
}

// Solve the chain analytically: given the current joint positions and a
// target for the end joint, return the delta rotations to apply about the
// root and mid joint origins (in the same space as the positions). The
// target is reach-clamped, so an out-of-range target straightens the chain
// toward it. `None` when the chain is degenerate (zero-length bones or a
// target on top of the root).
pub(crate) fn solve_two_bone(
    root: Vec3,
    mid: Vec3,
    end: Vec3,
    target: Vec3,
    pole: Vec3,
) -> Option<(Mat3, Mat3)> {
    let upper = sub(mid, root);
    let lower = sub(end, mid);
    let a = length(upper);
    let b = length(lower);
    if a <= EPS || b <= EPS {
        return None;
    }
    let to_target = sub(target, root);
    let dist = length(to_target);
    if dist <= EPS {
        return None;
    }
    let t_dir = scale(to_target, 1.0 / dist);
    let t = dist.clamp((a - b).abs() + 1.0e-4, a + b - 1.0e-4);

    let u = normalize(sub(root, mid)).expect("a > EPS");
    let v = normalize(lower).expect("b > EPS");

    // Bend axis: the current bend plane's normal. A straight chain has no
    // bend plane; fall back to an axis perpendicular to the bone line and
    // as close to the pole plane as possible (the signed-angle math below
    // requires the axis to be perpendicular to both bones, and for a
    // straight chain u = -v).
    let axis = normalize(cross(upper, lower))
        .or_else(|| normalize(cross(v, pole)))
        .unwrap_or_else(|| any_perpendicular(v));

    // Interior knee angle from the law of cosines, then rotate the lower
    // bone so the root-to-end distance becomes exactly `t`. The signed
    // current angle keeps the existing bend side; the pole twist below
    // picks the final plane, so an ambiguous side here cannot stick.
    let desired = ((a * a + b * b - t * t) / (2.0 * a * b))
        .clamp(-1.0, 1.0)
        .acos();
    let current_cos = dot(u, v).clamp(-1.0, 1.0);
    let current_sin = dot(cross(u, v), axis);
    let current = current_sin.atan2(current_cos);
    let signed_desired = if current >= 0.0 { desired } else { -desired };
    let r_mid = rotate_about(axis, signed_desired - current);

    // Aim the whole (now correctly-shortened) chain from the root onto the
    // target direction.
    let end_bent = add(mid, mat3_apply(r_mid, lower));
    let r_aim = from_to(sub(end_bent, root), to_target);

    // Twist about the root-to-target axis so the mid joint lies on the pole
    // side of the chain.
    let mid_aimed = mat3_apply(r_aim, upper);
    let bend_current = sub(mid_aimed, scale(t_dir, dot(mid_aimed, t_dir)));
    let bend_pole = sub(pole, scale(t_dir, dot(pole, t_dir)));
    let r_root = match (normalize(bend_current), normalize(bend_pole)) {
        (Some(c), Some(p)) => {
            let twist = dot(cross(c, p), t_dir).atan2(dot(c, p));
            mat3_mul(rotate_about(t_dir, twist), r_aim)
        }
        _ => r_aim,
    };
    Some((r_root, r_mid))
}

/// Apply one chain to a sampled local pose in place. `target` is the desired
/// end-joint position in mesh space; `weight` in `[0, 1]` fades the solve by
/// pulling the effective target from the animated end position toward
/// `target`. Locals shorter than the chain's joints grow from the bind pose
/// first, so a partial sample still solves correctly. `world` is a reusable
/// buffer the hierarchy composes through, so a steady-state solve allocates
/// nothing.
pub fn apply_two_bone_ik(
    skeleton: &Skeleton,
    locals: &mut Vec<Mat4>,
    chain: &TwoBoneChain,
    target: Vec3,
    weight: f32,
    world: &mut Vec<Mat4>,
) {
    let weight = weight.clamp(0.0, 1.0);
    let n = skeleton.len();
    if weight <= 0.0 || chain.root >= n || chain.mid >= n || chain.end >= n {
        return;
    }
    while locals.len() < n {
        let i = locals.len();
        locals.push(skeleton.joints()[i].bind.to_matrix());
    }
    skeleton.world_matrices_into(locals, world);
    let pos = |m: &Mat4| [m[3][0], m[3][1], m[3][2]];
    let p_root = pos(&world[chain.root]);
    let p_mid = pos(&world[chain.mid]);
    let p_end = pos(&world[chain.end]);
    let effective = add(p_end, scale(sub(target, p_end), weight));
    let Some((r_root, r_mid)) = solve_two_bone(p_root, p_mid, p_end, effective, chain.pole) else {
        return;
    };

    // New world transforms: the root turns about its own origin; the mid
    // inherits that and additionally bends about its (pre-solve) origin.
    let root_world = rotate_mat4_about(r_root, p_root, world[chain.root]);
    let mid_world = rotate_mat4_about(
        r_root,
        p_root,
        rotate_mat4_about(r_mid, p_mid, world[chain.mid]),
    );

    // Back to locals. The root's parent world is untouched by the solve; the
    // mid's parent is the root (direct parentage, validated at resolution).
    locals[chain.root] = match skeleton.joints()[chain.root].parent {
        Some(p) => mat4_mul(mat4_affine_inverse(world[p]), root_world),
        None => root_world,
    };
    locals[chain.mid] = mat4_mul(mat4_affine_inverse(root_world), mid_world);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::skinning::{Joint, JointPose};

    // A three-joint leg hanging straight down: hip at y=2, knee at y=1,
    // foot at y=0, all along -Y.
    fn leg() -> Skeleton {
        let joint = |name: &str, parent: Option<usize>, ty: f32| Joint {
            name: name.to_string(),
            parent,
            bind: JointPose {
                translation: [0.0, ty, 0.0],
                ..JointPose::default()
            },
        };
        Skeleton::new(vec![
            joint("hip", None, 2.0),
            joint("knee", Some(0), -1.0),
            joint("foot", Some(1), -1.0),
        ])
    }

    fn bind_locals(skeleton: &Skeleton) -> Vec<Mat4> {
        skeleton
            .joints()
            .iter()
            .map(|j| j.bind.to_matrix())
            .collect()
    }

    fn joint_pos(skeleton: &Skeleton, locals: &[Mat4], i: usize) -> [f32; 3] {
        let mut worlds = Vec::new();
        skeleton.world_matrices_into(locals, &mut worlds);
        let w = worlds[i];
        [w[3][0], w[3][1], w[3][2]]
    }

    fn solve(
        skeleton: &Skeleton,
        locals: &mut Vec<Mat4>,
        chain: &TwoBoneChain,
        target: [f32; 3],
        weight: f32,
    ) {
        apply_two_bone_ik(skeleton, locals, chain, target, weight, &mut Vec::new());
    }

    fn assert_close(a: [f32; 3], b: [f32; 3], tol: f32) {
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() < tol, "{a:?} vs {b:?}");
        }
    }

    const CHAIN: TwoBoneChain = TwoBoneChain {
        root: 0,
        mid: 1,
        end: 2,
        pole: [0.0, 0.0, 1.0],
    };

    #[test]
    fn reachable_target_lands_the_foot_and_bends_toward_the_pole() {
        let skeleton = leg();
        let mut locals = bind_locals(&skeleton);
        // Pull the foot up half a unit: the leg must bend.
        solve(&skeleton, &mut locals, &CHAIN, [0.0, 0.5, 0.0], 1.0);
        assert_close(joint_pos(&skeleton, &locals, 2), [0.0, 0.5, 0.0], 1e-3);
        // The knee moved out on the pole side (+Z) and the hip stayed put.
        let knee = joint_pos(&skeleton, &locals, 1);
        assert!(knee[2] > 0.1, "knee bends toward the pole: {knee:?}");
        assert_close(joint_pos(&skeleton, &locals, 0), [0.0, 2.0, 0.0], 1e-4);
        // Bone lengths are preserved by the solve.
        let hip = joint_pos(&skeleton, &locals, 0);
        let foot = joint_pos(&skeleton, &locals, 2);
        let len = |a: [f32; 3], b: [f32; 3]| length(sub(a, b));
        assert!((len(hip, knee) - 1.0).abs() < 1e-3);
        assert!((len(knee, foot) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn out_of_reach_target_straightens_toward_it() {
        let skeleton = leg();
        let mut locals = bind_locals(&skeleton);
        solve(&skeleton, &mut locals, &CHAIN, [3.0, 2.0, 0.0], 1.0);
        // Max reach is ~2 along +X from the hip at (0,2,0).
        let foot = joint_pos(&skeleton, &locals, 2);
        assert_close(foot, [2.0, 2.0, 0.0], 2e-2);
    }

    #[test]
    fn sideways_target_respects_the_pole_plane() {
        let skeleton = leg();
        let mut locals = bind_locals(&skeleton);
        solve(&skeleton, &mut locals, &CHAIN, [1.0, 0.8, 0.0], 1.0);
        assert_close(joint_pos(&skeleton, &locals, 2), [1.0, 0.8, 0.0], 1e-3);
        let knee = joint_pos(&skeleton, &locals, 1);
        assert!(knee[2] > 0.05, "knee stays on the +Z pole side: {knee:?}");
    }

    #[test]
    fn weight_blends_between_animated_and_solved() {
        let skeleton = leg();
        let mut half = bind_locals(&skeleton);
        solve(&skeleton, &mut half, &CHAIN, [0.0, 0.5, 0.0], 0.5);
        // Half weight pins the foot halfway between animated (y=0) and the
        // target (y=0.5).
        assert_close(joint_pos(&skeleton, &half, 2), [0.0, 0.25, 0.0], 1e-3);

        let mut off = bind_locals(&skeleton);
        solve(&skeleton, &mut off, &CHAIN, [0.0, 0.5, 0.0], 0.0);
        assert_close(joint_pos(&skeleton, &off, 2), [0.0, 0.0, 0.0], 1e-6);
    }

    #[test]
    fn degenerate_targets_leave_the_pose_untouched() {
        let skeleton = leg();
        let mut locals = bind_locals(&skeleton);
        let before = locals.clone();
        // Target exactly on the root: no solvable direction.
        solve(&skeleton, &mut locals, &CHAIN, [0.0, 2.0, 0.0], 1.0);
        assert_eq!(locals, before);
        // Out-of-range chain indices are ignored.
        let bad = TwoBoneChain {
            root: 0,
            mid: 9,
            end: 2,
            pole: [0.0, 0.0, 1.0],
        };
        solve(&skeleton, &mut locals, &bad, [0.0, 0.5, 0.0], 1.0);
        assert_eq!(locals, before);
    }

    #[test]
    fn solve_composes_with_an_animated_pose() {
        // Rotate the hip 45 degrees about Z first (the leg swings toward +X),
        // then pin the foot back under the hip: the solve must land on target
        // from the animated (not bind) configuration.
        let skeleton = leg();
        let mut locals = bind_locals(&skeleton);
        locals[0] = JointPose {
            translation: [0.0, 2.0, 0.0],
            rotation_deg: [0.0, 0.0, 45.0],
            ..JointPose::default()
        }
        .to_matrix();
        solve(&skeleton, &mut locals, &CHAIN, [0.0, 0.2, 0.0], 1.0);
        assert_close(joint_pos(&skeleton, &locals, 2), [0.0, 0.2, 0.0], 1e-3);
    }
}

// Bind-pose bone frames and per-vertex region weights: the geometry every
// synthesizer reads. A bone frame is the joint's model-space origin, the unit
// axis toward its first child (or along its parent's bone for a leaf), and
// the bone length. A vertex's region weight is the skin weight it gives the
// region's joints, so a region boundary is exactly as smooth as the skinning.

use crate::components::{SkeletonJoint, SkinnedVertexData};
use concinnity_core::gfx::transform::Mat4;
use concinnity_core::math::vec3;

// One joint's bind-pose frame in model space.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoneFrame {
    pub origin: [f32; 3],
    pub axis: [f32; 3],
    pub length: f32,
}

impl BoneFrame {
    // Where `p` sits along the bone as a fraction of its length (0 at the
    // joint, 1 at the child), unclamped.
    pub(crate) fn along(&self, p: [f32; 3]) -> f32 {
        vec3::dot(vec3::sub(p, self.origin), self.axis) / self.length.max(1e-6)
    }

    // The component of `p - origin` perpendicular to the axis.
    pub(crate) fn radial(&self, p: [f32; 3]) -> [f32; 3] {
        let d = vec3::sub(p, self.origin);
        vec3::sub(d, vec3::scale(self.axis, vec3::dot(d, self.axis)))
    }
}

// Model-space joint origins from the bind pose.
fn joint_origins(skeleton: &[SkeletonJoint]) -> Vec<[f32; 3]> {
    let sk = concinnity_core::components::build_skeleton_from_joint_defs(skeleton);
    let mut world: Vec<Mat4> = Vec::new();
    sk.world_matrices_into(sk.bind_locals(), &mut world);
    world.iter().map(|m| [m[3][0], m[3][1], m[3][2]]).collect()
}

// Bind-pose frames for every joint. A leaf's axis continues its parent's
// bone and its length is the reach of its own vertices along that axis (or
// the parent's length when it has none).
pub(crate) fn bone_frames(
    skeleton: &[SkeletonJoint],
    vertices: &[SkinnedVertexData],
) -> Vec<BoneFrame> {
    let origins = joint_origins(skeleton);
    let n = skeleton.len();
    let first_child: Vec<Option<usize>> = (0..n)
        .map(|j| (0..n).find(|&c| skeleton[c].parent == j as i32))
        .collect();
    let mut frames: Vec<BoneFrame> = Vec::with_capacity(n);
    for j in 0..n {
        let (axis, length) = match first_child[j] {
            Some(c) => {
                let d = vec3::sub(origins[c], origins[j]);
                let len = vec3::length(d);
                if len > 1e-6 {
                    (vec3::scale(d, 1.0 / len), len)
                } else {
                    ([0.0, 1.0, 0.0], 0.0)
                }
            }
            None => {
                let parent = skeleton[j].parent;
                let axis = if parent >= 0 {
                    let p = &frames[parent as usize];
                    let d = vec3::sub(origins[j], p.origin);
                    let len = vec3::length(d);
                    if len > 1e-6 {
                        vec3::scale(d, 1.0 / len)
                    } else {
                        p.axis
                    }
                } else {
                    [0.0, 1.0, 0.0]
                };
                let reach = vertices
                    .iter()
                    .filter(|v| dominant_joint(v) == j)
                    .map(|v| vec3::dot(vec3::sub(v.pos, origins[j]), axis))
                    .fold(0.0_f32, f32::max);
                let fallback = if parent >= 0 {
                    frames[parent as usize].length
                } else {
                    0.0
                };
                (axis, if reach > 1e-4 { reach } else { fallback })
            }
        };
        frames.push(BoneFrame {
            origin: origins[j],
            axis,
            length,
        });
    }
    frames
}

// The joint a vertex gives the most weight (the first on a tie).
pub(crate) fn dominant_joint(v: &SkinnedVertexData) -> usize {
    let mut best = 0;
    for k in 1..4 {
        if v.weights[k] > v.weights[best] {
            best = k;
        }
    }
    v.joints[best] as usize
}

// Per-joint membership flags for a region, from its joint names.
pub(crate) fn region_joints(skeleton: &[SkeletonJoint], joints: &[String]) -> Vec<bool> {
    skeleton.iter().map(|j| joints.contains(&j.name)).collect()
}

// The skin weight a vertex gives the region's joints, in `[0, 1]`.
pub(crate) fn region_weight(v: &SkinnedVertexData, members: &[bool]) -> f32 {
    let sum: f32 = v.weights.iter().sum();
    if sum <= 1e-6 {
        return 0.0;
    }
    (0..4)
        .filter(|&k| members.get(v.joints[k] as usize).copied().unwrap_or(false))
        .map(|k| v.weights[k])
        .sum::<f32>()
        / sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::character::synth::test_support::{cylinder, joint, vertex};

    #[test]
    fn frames_follow_the_first_child_and_leaves_continue_their_parent() {
        let skeleton = vec![
            joint("root", -1, [0.0, 0.0, 0.0]),
            joint("mid", 0, [0.0, 1.0, 0.0]),
            joint("tip", 1, [0.0, 2.0, 0.0]),
        ];
        let verts = vec![
            vertex([0.1, 3.5, 0.0], 2),
            vertex([0.0, 2.5, 0.0], 2),
            vertex([0.0, 0.5, 0.0], 0),
        ];
        let frames = bone_frames(&skeleton, &verts);
        assert_eq!(frames[0].origin, [0.0, 0.0, 0.0]);
        assert_eq!(frames[0].axis, [0.0, 1.0, 0.0]);
        assert_eq!(frames[0].length, 1.0);
        assert_eq!(frames[1].origin, [0.0, 1.0, 0.0]);
        assert_eq!(frames[1].length, 2.0);
        // The leaf keeps pointing up and reaches its farthest vertex.
        assert_eq!(frames[2].axis, [0.0, 1.0, 0.0]);
        assert!((frames[2].length - 0.5).abs() < 1e-6);
        assert!((frames[1].along([0.0, 2.0, 0.0]) - 0.5).abs() < 1e-6);
        assert_eq!(frames[1].radial([0.3, 1.7, 0.0]), [0.3, 0.0, 0.0]);
    }

    #[test]
    fn a_leaf_without_vertices_borrows_its_parents_length() {
        let skeleton = vec![
            joint("root", -1, [0.0, 0.0, 0.0]),
            joint("tip", 0, [0.0, 0.0, 2.0]),
        ];
        let frames = bone_frames(&skeleton, &[]);
        assert_eq!(frames[1].axis, [0.0, 0.0, 1.0]);
        assert_eq!(frames[1].length, 2.0);
    }

    #[test]
    fn region_weights_and_membership_follow_the_skin_weights() {
        let skeleton = vec![
            joint("a", -1, [0.0, 0.0, 0.0]),
            joint("b", 0, [0.0, 1.0, 0.0]),
            joint("c", 1, [0.0, 2.0, 0.0]),
        ];
        let members = region_joints(&skeleton, &["b".to_string(), "c".to_string()]);
        assert_eq!(members, [false, true, true]);
        let mut v = vertex([0.0, 0.0, 0.0], 0);
        v.joints = [0, 1, 2, 0];
        v.weights = [0.5, 0.3, 0.2, 0.0];
        assert!((region_weight(&v, &members) - 0.5).abs() < 1e-6);
        assert_eq!(dominant_joint(&v), 0);
        v.weights = [0.2, 0.5, 0.3, 0.0];
        assert_eq!(dominant_joint(&v), 1);
        // Unnormalised weights are normalised; no weights give nothing.
        v.weights = [0.4, 1.0, 0.6, 0.0];
        assert!((region_weight(&v, &members) - 0.8).abs() < 1e-6);
        v.weights = [0.0; 4];
        assert_eq!(region_weight(&v, &members), 0.0);
        // The cylinder fixture is fully inside its own region.
        let (verts, _, _) = cylinder(8, 4, 1.0, 2.0);
        assert!(verts.iter().all(|v| region_weight(v, &[true]) == 1.0));
    }
}

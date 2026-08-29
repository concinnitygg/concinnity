//! Per-joint proportion changes applied to a posed skeleton: a uniform scale
//! on a joint's local matrix and a length offset pushing its children along
//! the bone. The bind pose and inverse bind matrices stay as authored, so the
//! change rides under every clip sampled on the skeleton.

use alloc::vec::Vec;

use crate::components::JointProportion;

use crate::gfx::skeleton::Skeleton;
use crate::gfx::transform::Mat4;

// One child pushed along its bind direction by the parent's `length`.
#[derive(Debug, Clone, PartialEq)]
struct ChildOffset {
    child: usize,
    offset: [f32; 3],
}

#[derive(Debug, Clone, PartialEq)]
struct Entry {
    joint: usize,
    scale: f32,
    offsets: Vec<ChildOffset>,
}

/// Proportion entries resolved against one skeleton, ready to apply to a
/// local-pose buffer every frame without further lookups.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProportionLayer {
    entries: Vec<Entry>,
}

impl ProportionLayer {
    /// Resolve `proportions` against `skeleton` by joint name. Entries naming
    /// an unknown joint are skipped (the caller reports them); entries that
    /// change nothing are dropped.
    pub fn resolve(skeleton: &Skeleton, proportions: &[JointProportion]) -> Self {
        let joints = skeleton.joints();
        let entries = proportions
            .iter()
            .filter(|p| p.scale != 1.0 || p.length != 0.0)
            .filter_map(|p| {
                let joint = skeleton.joint_index(&p.joint)?;
                let offsets = joints
                    .iter()
                    .enumerate()
                    .filter(|(_, j)| j.parent == Some(joint))
                    .filter_map(|(child, j)| {
                        let dir = normalize(j.bind.translation)?;
                        Some(ChildOffset {
                            child,
                            offset: crate::math::vec3::scale(dir, p.length),
                        })
                    })
                    .collect();
                Some(Entry {
                    joint,
                    scale: p.scale,
                    offsets,
                })
            })
            .collect();
        Self { entries }
    }

    /// Whether no joint is changed.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Apply the layer to `locals` (one local matrix per joint, as sampled
    /// from a clip or copied from the bind pose). A joint's scale multiplies
    /// its local basis; its children's local translations gain the length
    /// offset. Entries past the end of `locals` are ignored.
    pub fn apply(&self, locals: &mut [Mat4]) {
        for e in &self.entries {
            if let Some(m) = locals.get_mut(e.joint) {
                for col in m.iter_mut().take(3) {
                    col[0] *= e.scale;
                    col[1] *= e.scale;
                    col[2] *= e.scale;
                }
            }
            for o in &e.offsets {
                if let Some(m) = locals.get_mut(o.child) {
                    m[3][0] += o.offset[0];
                    m[3][1] += o.offset[1];
                    m[3][2] += o.offset[2];
                }
            }
        }
    }

    /// Uniform scale of the skeleton's first root joint, or `1` when no
    /// entry changes it.
    pub fn root_scale(&self, skeleton: &Skeleton) -> f32 {
        let root = skeleton.joints().iter().position(|j| j.parent.is_none());
        self.entries
            .iter()
            .find(|e| Some(e.joint) == root)
            .map_or(1.0, |e| e.scale)
    }

    /// Ratio of the proportioned skeleton's height to the bind height, both
    /// measured as the highest joint position (model Y) in the rest pose.
    /// `1` when the bind skeleton has no height.
    pub fn height_ratio(&self, skeleton: &Skeleton) -> f32 {
        let mut world = Vec::new();
        skeleton.world_matrices_into(skeleton.bind_locals(), &mut world);
        let bind_top = top(&world);
        let mut locals: Vec<Mat4> = skeleton.bind_locals().to_vec();
        self.apply(&mut locals);
        skeleton.world_matrices_into(&locals, &mut world);
        if bind_top <= 1e-6 {
            1.0
        } else {
            top(&world) / bind_top
        }
    }
}

fn top(world: &[Mat4]) -> f32 {
    world.iter().map(|m| m[3][1]).fold(0.0, f32::max)
}

fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let len = crate::math::vec3::length(v);
    (len > 1e-6).then(|| crate::math::vec3::scale(v, 1.0 / len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::skeleton::{Joint, JointPose};
    use alloc::string::String;
    use alloc::vec;

    // root -> mid -> tip, each 1 unit up the Y axis.
    fn chain() -> Skeleton {
        let joint = |name: &str, parent: Option<usize>, y: f32| Joint {
            name: String::from(name),
            parent,
            bind: JointPose {
                translation: [0.0, y, 0.0],
                ..Default::default()
            },
        };
        Skeleton::new(vec![
            joint("root", None, 0.0),
            joint("mid", Some(0), 1.0),
            joint("tip", Some(1), 1.0),
        ])
    }

    fn proportion(joint: &str, scale: f32, length: f32) -> JointProportion {
        JointProportion {
            joint: String::from(joint),
            scale,
            length,
        }
    }

    fn positions(skeleton: &Skeleton, layer: &ProportionLayer) -> Vec<[f32; 3]> {
        let mut locals = skeleton.bind_locals().to_vec();
        layer.apply(&mut locals);
        let mut world = Vec::new();
        skeleton.world_matrices_into(&locals, &mut world);
        world.iter().map(|m| [m[3][0], m[3][1], m[3][2]]).collect()
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-5)
    }

    #[test]
    fn length_pushes_only_the_children_along_the_bone() {
        let s = chain();
        let layer = ProportionLayer::resolve(&s, &[proportion("mid", 1.0, 0.5)]);
        let p = positions(&s, &layer);
        // mid itself stays; tip moves 0.5 further up and nothing stretches
        // beyond that.
        assert!(close(p[1], [0.0, 1.0, 0.0]), "{p:?}");
        assert!(close(p[2], [0.0, 2.5, 0.0]), "{p:?}");
        assert!((layer.height_ratio(&s) - 1.25).abs() < 1e-5);
    }

    #[test]
    fn scale_propagates_to_descendants() {
        let s = chain();
        let layer = ProportionLayer::resolve(&s, &[proportion("mid", 2.0, 0.0)]);
        let p = positions(&s, &layer);
        // The mid joint's frame is doubled, so the tip's 1-unit offset
        // becomes 2 units.
        assert!(close(p[2], [0.0, 3.0, 0.0]), "{p:?}");
        // Scaling the root doubles the whole chain and reports as root scale.
        let layer = ProportionLayer::resolve(&s, &[proportion("root", 2.0, 0.0)]);
        let p = positions(&s, &layer);
        assert!(close(p[2], [0.0, 4.0, 0.0]), "{p:?}");
        assert_eq!(layer.root_scale(&s), 2.0);
        assert!((layer.height_ratio(&s) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn unknown_and_identity_entries_are_dropped() {
        let s = chain();
        let layer = ProportionLayer::resolve(
            &s,
            &[proportion("tail", 2.0, 1.0), proportion("mid", 1.0, 0.0)],
        );
        assert!(layer.is_empty());
        assert_eq!(layer.root_scale(&s), 1.0);
        assert_eq!(layer.height_ratio(&s), 1.0);
        let mut locals = s.bind_locals().to_vec();
        layer.apply(&mut locals);
        assert_eq!(locals, s.bind_locals());
    }

    #[test]
    fn a_short_local_buffer_is_applied_in_range() {
        let s = chain();
        let layer = ProportionLayer::resolve(&s, &[proportion("mid", 2.0, 0.5)]);
        let mut locals = s.bind_locals()[..2].to_vec();
        layer.apply(&mut locals);
        assert_eq!(locals[1][0][0], 2.0);
    }
}

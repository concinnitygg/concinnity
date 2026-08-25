// src/components/skinned_mesh.rs
//
// Runtime behavior for the SkinnedMesh asset. The authored schema (SkinnedMesh,
// its SkinnedVertexData / SkeletonJoint / CharacterCapsule, and their Defaults) lives
// in concinnity-asset; SkinnedMesh is a resource now (compiled by cook into the
// blob's resource stream, no `Component` impl), so this file keeps only the
// skeleton builder and the `SkinnedMeshGeometry` extension trait that needs
// `gfx::skeleton`.

use crate::components::{SkeletonJoint, SkinnedMesh};

/// Build a runtime `Skeleton` from authored joint definitions. Mirrors the
/// conversion `GraphicsSystem::init` does at world load time: each
/// `SkeletonJoint.parent` becomes `Some(usize)` for valid indices (negative values
/// mark roots), and each `SkeletonJoint`'s translation / rotation / scale becomes the
/// joint's bind `JointPose`. Used at init and by the asset hot-reload's
/// skeleton-shape change path.
pub fn build_skeleton_from_joint_defs(defs: &[SkeletonJoint]) -> crate::gfx::skeleton::Skeleton {
    use crate::gfx::skeleton as skinning;
    let joints = defs
        .iter()
        .map(|jd| skinning::Joint {
            name: jd.name.clone(),
            parent: (jd.parent >= 0).then_some(jd.parent as usize),
            bind: skinning::JointPose {
                translation: jd.translation,
                rotation_deg: jd.rotation_deg,
                scale: jd.scale,
            },
        })
        .collect();
    skinning::Skeleton::new(joints)
}

/// Column-major world matrix from a SkinnedMesh's transform. Kept in core (not
/// the schema crate) because the matrix build goes through `gfx::skeleton`, which
/// needs std transcendentals. Exposed as an extension trait so call sites keep
/// method syntax (`sm.model_matrix()`), matching `geometry.rs`.
pub trait SkinnedMeshGeometry {
    /// Column-major world matrix built from the mesh's transform.
    fn model_matrix(&self) -> [[f32; 4]; 4];
}

impl SkinnedMeshGeometry for SkinnedMesh {
    // Same construction order (scale, YXZ rotation, translate) as
    // `Prop::model_matrix`.
    fn model_matrix(&self) -> [[f32; 4]; 4] {
        crate::gfx::skeleton::JointPose {
            translation: self.position,
            rotation_deg: self.rotation_deg,
            scale: self.scale,
        }
        .to_matrix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{CharacterCapsule, SkinnedVertexData};
    use alloc::vec;

    #[test]
    fn build_skeleton_from_joint_defs_preserves_count_and_parent_links() {
        let defs = vec![
            SkeletonJoint {
                name: "root".into(),
                parent: -1,
                translation: [0.0, 0.0, 0.0],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
            },
            SkeletonJoint {
                name: "tip".into(),
                parent: 0,
                translation: [0.0, 1.0, 0.0],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
            },
            SkeletonJoint {
                name: "tail".into(),
                parent: 1,
                translation: [0.0, 1.0, 0.0],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
            },
        ];
        let skel = build_skeleton_from_joint_defs(&defs);
        assert_eq!(skel.len(), 3);
        let joints = skel.joints();
        assert_eq!(joints[0].parent, None);
        assert_eq!(joints[1].parent, Some(0));
        assert_eq!(joints[2].parent, Some(1));
    }

    #[test]
    fn build_skeleton_from_joint_defs_treats_negative_parent_as_root() {
        // Any negative parent (not just -1) collapses to None; mirrors the
        // init-time semantics so a hot-reload from the same SkeletonJoint shape
        // produces the same Skeleton.
        let defs = vec![SkeletonJoint {
            name: "root".into(),
            parent: -42,
            translation: [1.0, 2.0, 3.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }];
        let skel = build_skeleton_from_joint_defs(&defs);
        assert_eq!(skel.joints()[0].parent, None);
    }

    #[test]
    fn model_matrix_places_translation_in_last_column() {
        let mesh = SkinnedMesh {
            position: [2.0, 3.0, 4.0],
            scale: [1.0, 1.0, 1.0],
            ..SkinnedMesh::default()
        };
        let m = mesh.model_matrix();
        // Column-major: the translation lives in the last column, identity
        // scale keeps the diagonal at 1.
        assert_eq!([m[3][0], m[3][1], m[3][2]], [2.0, 3.0, 4.0]);
        assert_eq!(m[3][3], 1.0);
        assert_eq!(m[0][0], 1.0);
    }

    #[test]
    fn skinned_vertex_defaults_fill_color_uv_and_weights() {
        // A vertex authored with only a position picks up the serde defaults:
        // white colour, zero uv, and full weight on joint 0.
        let v: SkinnedVertexData =
            serde_json::from_value(serde_json::json!({"pos": [0.0, 0.0, 0.0]})).unwrap();
        assert_eq!(v.color, [1.0, 1.0, 1.0]);
        assert_eq!(v.uv, [0.0, 0.0]);
        assert_eq!(v.weights, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(v.joints, [0, 0, 0, 0]);
    }

    #[test]
    fn capsule_joint_defaults() {
        let cap = CharacterCapsule::default();
        assert_eq!(cap.half_height, 0.5);
        assert_eq!(cap.radius, 0.3);

        let jd: SkeletonJoint = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(jd.parent, -1);
        assert_eq!(jd.scale, [1.0, 1.0, 1.0]);
    }
}

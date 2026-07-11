// src/assets/skinned_mesh.rs
//
// Runtime behavior for the SkinnedMesh asset. The authored schema (SkinnedMesh,
// its SkinnedVertexData / JointDef / CharacterCapsule, and their Defaults) lives
// in concinnity-asset; this file keeps the `Component` impl, the `SourceBacked`
// impl, the skeleton builder, and the `SkinnedMeshGeometry` extension trait that
// needs `gfx::skinning`.

use crate::assets::{JointDef, SkinnedMesh};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

fn unit_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

// Build a runtime `Skeleton` from authored joint definitions. Mirrors the
// conversion `GraphicsSystem::init` does at world load time: each
// `JointDef.parent` becomes `Some(usize)` for valid indices (negative values
// mark roots), and each `JointDef`'s translation / rotation / scale becomes the
// joint's bind `JointPose`. Used at init and by the asset hot-reload's
// skeleton-shape change path.
pub fn build_skeleton_from_joint_defs(defs: &[JointDef]) -> crate::gfx::skinning::Skeleton {
    use crate::gfx::skinning;
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

// Column-major world matrix from a SkinnedMesh's transform. Kept in core (not
// the schema crate) because the matrix build goes through `gfx::skinning`, which
// needs std transcendentals. Exposed as an extension trait so call sites keep
// method syntax (`sm.model_matrix()`), matching `geometry.rs`.
pub trait SkinnedMeshGeometry {
    fn model_matrix(&self) -> [[f32; 4]; 4];
}

impl SkinnedMeshGeometry for SkinnedMesh {
    // Same construction order (scale, YXZ rotation, translate) as
    // `Prop::model_matrix`.
    fn model_matrix(&self) -> [[f32; 4]; 4] {
        crate::gfx::skinning::JointPose {
            translation: self.position,
            rotation_deg: self.rotation_deg,
            scale: self.scale,
        }
        .to_matrix()
    }
}

impl Component for SkinnedMesh {
    const NAME: &'static str = "SkinnedMesh";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    const PAYLOAD: AssetPayload = AssetPayload::Compiled;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(mut args: Self) -> Self {
        // A zero scale would collapse the world matrix; clamp to a sane unit.
        if args.scale == [0.0, 0.0, 0.0] {
            args.scale = unit_scale();
        }
        // `lod_levels` defaults to 0 via #[derive(Default)] on u32. 0 and 1
        // are equivalent (no alternates); cap at 8 to mirror the static
        // `Mesh` LOD ceiling.
        if args.lod_levels == 0 {
            args.lod_levels = 1;
        }
        args.lod_levels = args.lod_levels.min(8);
        // A runaway reserve would pre-expand the skinned vertex buffer without
        // bound; cap it. The skinned index buffer is u16, so init applies a
        // further per-mesh limit when the copies would overflow that.
        args.max_instances = args.max_instances.min(4096);
        args
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }
    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

impl crate::build::SourceBacked for SkinnedMesh {
    // A glTF-sourced SkinnedMesh needs its `.glb` fetched before the build's
    // desugar pass can expand it; an inline-authored mesh has no source.
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        args.get("source")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{CharacterCapsule, SkinnedVertexData};

    #[test]
    fn build_skeleton_from_joint_defs_preserves_count_and_parent_links() {
        let defs = vec![
            JointDef {
                name: "root".into(),
                parent: -1,
                translation: [0.0, 0.0, 0.0],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
            },
            JointDef {
                name: "tip".into(),
                parent: 0,
                translation: [0.0, 1.0, 0.0],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
            },
            JointDef {
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
        // init-time semantics so a hot-reload from the same JointDef shape
        // produces the same Skeleton.
        let defs = vec![JointDef {
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
    fn from_args_clamps_transform_and_reserve() {
        // A zero scale becomes unit; lod_levels floors at 1; max_instances is
        // capped at 4096.
        let clamped = SkinnedMesh::from_args(SkinnedMesh {
            scale: [0.0, 0.0, 0.0],
            lod_levels: 0,
            max_instances: 999_999,
            ..SkinnedMesh::default()
        });
        assert_eq!(clamped.scale, [1.0, 1.0, 1.0]);
        assert_eq!(clamped.lod_levels, 1);
        assert_eq!(clamped.max_instances, 4096);

        // lod_levels is also capped at the 8-level ceiling.
        let capped = SkinnedMesh::from_args(SkinnedMesh {
            lod_levels: 99,
            scale: [1.0, 1.0, 1.0],
            ..SkinnedMesh::default()
        });
        assert_eq!(capped.lod_levels, 8);
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
    fn capsule_joint_defaults_and_source_path() {
        let cap = CharacterCapsule::default();
        assert_eq!(cap.half_height, 0.5);
        assert_eq!(cap.radius, 0.3);

        let jd: JointDef = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(jd.parent, -1);
        assert_eq!(jd.scale, [1.0, 1.0, 1.0]);

        use crate::build::{Platform, SourceBacked};
        assert_eq!(
            SkinnedMesh::source_path(&serde_json::json!({"source": "hero.glb"}), Platform::Metal),
            Some("hero.glb".to_string())
        );
        assert_eq!(
            SkinnedMesh::source_path(&serde_json::json!({}), Platform::Metal),
            None
        );
    }
}

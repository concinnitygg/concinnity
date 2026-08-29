// src/components/skinned_mesh.rs
//
// Runtime behavior for the SkinnedMesh asset. The authored schema (SkinnedMesh,
// its SkinnedVertexData / SkeletonJoint / CharacterCapsule, and their Defaults) lives
// above; SkinnedMesh is a resource (compiled by cook into the
// blob's resource stream, no `Component` impl), so this file keeps only the
// skeleton builder and the `SkinnedMeshGeometry` extension trait that needs
// `gfx::skeleton`.

use crate::ecs::MaterialHandle;
use crate::ecs::PayloadLocator;
use crate::ecs::TextureHandle;
use crate::ecs::asset_id::AssetId;
use crate::ecs::de_opt_material_handle;
use crate::ecs::de_opt_texture_handle;
use alloc::string::String;
use alloc::vec::Vec;

fn white() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

fn first_weight() -> [f32; 4] {
    [1.0, 0.0, 0.0, 0.0]
}

/// One vertex of a skinned mesh. Beyond position / colour / uv it carries up
/// to four joint bindings: `joints[k]` indexes the skeleton, `weights[k]` is
/// its blend weight. Weights are normalised at build time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SkinnedVertexData {
    /// Vertex position `[x, y, z]` in model space.
    pub pos: [f32; 3],
    /// Vertex colour `[r, g, b]` in [0, 1]. Defaults to white.
    #[serde(default = "white")]
    pub color: [f32; 3],
    /// Texture coordinates in [0, 1] space. Defaults to [0, 0].
    #[serde(default)]
    pub uv: [f32; 2],
    /// Joint indices this vertex is bound to. Unused slots can be 0.
    #[serde(default)]
    pub joints: [u32; 4],
    /// Blend weights parallel to `joints`. Defaults to fully bound to joint 0.
    #[serde(default = "first_weight")]
    pub weights: [f32; 4],
}

/// One morph-target vertex delta: offsets added to the bind-pose position and
/// normal, scaled by the target's weight at runtime.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MorphDelta {
    /// Position offset `[x, y, z]` in model space.
    pub position: [f32; 3],
    /// Normal offset; the deformed normal is re-normalised after adding it.
    pub normal: [f32; 3],
}

/// One joint of a skeleton's bind pose.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SkeletonJoint {
    /// Human-readable joint name (animation tracks may reference it later).
    pub name: String,
    /// Parent joint index, or -1 for a root. Parents must appear before their
    /// children in the `skeleton` list.
    pub parent: i32,
    /// Local bind translation relative to the parent.
    pub translation: [f32; 3],
    /// Local bind rotation, Euler degrees [pitch, yaw, roll], YXZ order.
    pub rotation_deg: [f32; 3],
    /// Local bind scale.
    pub scale: [f32; 3],
}

impl Default for SkeletonJoint {
    fn default() -> Self {
        Self {
            name: String::new(),
            parent: -1,
            translation: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

/// A skeletally animated mesh placed directly in the world.
///
/// Unlike a [Mesh](#mesh), a `SkinnedMesh` carries its own world transform and a
/// `skeleton` (a joint hierarchy with a bind pose). Each vertex is bound to up
/// to four joints; an [Animation](#animation) targeting this mesh deforms it at
/// runtime. With no animation the mesh renders in its bind pose.
///
/// The geometry + skeleton may be authored inline (`vertices` / `indices` /
/// `skeleton`) or imported with `source` from a glTF (`.glb` / `.gltf`) or
/// binary `.fbx` file. The import fills the mesh, the skeleton bind pose,
/// and (for glTF) any morph targets; animations are imported separately by
/// [Animation](#animation) assets referencing the same file.
///
/// The `customize_character` example ships a neutral unclothed body
/// (`base_humanoid.glb`, about 19k vertices, A-pose bind) with a 25-joint
/// skeleton (`root`, `hips`, `spine`, `chest`,
/// `upper_chest`, `neck`, `head`, and `clavicle` / `upper_arm` / `forearm` /
/// `hand` / `thumb` / `thigh` / `shin` / `foot` / `toe` with an `_l` / `_r`
/// suffix), the morph targets a [CharacterShape](#charactershape) slider set
/// names (`weight+/-`, `muscle`, `shoulders+/-`, `hips+/-`, `chest+/-`,
/// `belly`, `head+/-`, `jaw+/-`, `nose+/-`, `brow`, `cheeks+/-`,
/// `chin+/-`), and a rotation-only `idle` clip an Animation can import from
/// the same file; a [CharacterModel](#charactermodel) is the usual way to
/// declare it.
///
/// The `skeleton` (joint hierarchy and bind pose) is provided as an arg
/// (authored inline alongside `vertices`/`indices`, or filled in from the
/// imported `.glb`) and is baked into the mesh at build time.
///
/// Normals and tangents are computed automatically at build time. Do not
/// supply them.
///
/// ```rust
/// # use concinnity_core::components::SkinnedMesh;
/// SkinnedMesh {
///     position: [0.0, 1.0, 0.0],
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SkinnedMesh {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Optional path to a `.glb` / `.gltf` / `.fbx` file. When set, the
    /// build imports `vertices` / `indices` / `skeleton` from it; an
    /// inline-authored mesh leaves this empty.
    pub source: String,
    /// Which skinned mesh of `source` to import, in file declaration order
    /// (default 0). A character split into several meshes bound to one
    /// skeleton (body, hair, clothes) needs one `SkinnedMesh` per part, each
    /// naming its own index.
    pub skin_index: u32,
    /// Skinned vertex list.
    pub vertices: Vec<SkinnedVertexData>,
    /// Triangle index list.
    pub indices: Vec<u16>,
    /// Morph-target names, one per target, in target order. Filled from the
    /// source file's target names when importing; empty for a mesh without
    /// morph targets.
    pub morph_target_names: Vec<String>,
    /// Dense morph-target deltas, target-major: entry `t * vertex_count + v`
    /// is target `t`'s delta for vertex `v`. Length must be
    /// `morph_target_names.len() * vertices.len()`. An [Animation](#animation)
    /// with a `morph_track` drives the per-target weights at runtime.
    pub morph_deltas: Vec<MorphDelta>,
    /// [Material](#material); provides the albedo texture plus lighting
    /// parameters.
    #[serde(deserialize_with = "de_opt_material_handle")]
    pub material: Option<MaterialHandle>,
    /// [Texture](#texture) (older path); ignored when `material` is set.
    #[serde(deserialize_with = "de_opt_texture_handle")]
    pub texture: Option<TextureHandle>,
    /// World-space position.
    pub position: [f32; 3],
    /// World rotation, Euler degrees [pitch, yaw, roll], YXZ order.
    pub rotation_deg: [f32; 3],
    /// World scale.
    pub scale: [f32; 3],
    /// Number of level-of-detail versions to generate, including the original.
    /// `1` (the default) generates none; values are clamped to `[1, 8]`.
    pub lod_levels: u32,
    /// Camera distances at which to switch to each lower-detail version. When
    /// non-empty, must have exactly `lod_levels - 1` entries; empty lets the
    /// build choose defaults.
    #[serde(default)]
    pub lod_distances: Vec<f32>,
    /// How many runtime copies of this mesh may exist at once beyond the
    /// authored one. `0` (the default) means the mesh is not runtime-spawnable.
    /// A non-zero value pre-reserves that many extra instance slots at load: the
    /// engine appends that many hidden bind-pose copies to the skinned geometry
    /// so a runtime spawn can claim one without growing any GPU buffer, and a
    /// despawn returns it to the pool. Spawns past the reserve are dropped (a
    /// warning is logged). Capped at 4096.
    pub max_instances: u32,
    /// Optional character capsule. When set, the mesh collides with the
    /// scene as a kinematic character and is moved by the root motion of its
    /// [Animation](#animation) clips (those with `root_motion` set): the
    /// capsule slides along obstacles and settles under gravity, and the
    /// rendered mesh follows it. The capsule stands on the mesh origin (its
    /// feet), centred `half_height + radius` above it.
    pub capsule: Option<CharacterCapsule>,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

/// A kinematic character capsule for a [SkinnedMesh](#skinnedmesh), in world
/// units (after the mesh's `scale`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CharacterCapsule {
    /// Half-height of the capsule's cylindrical section.
    pub half_height: f32,
    /// Capsule radius.
    pub radius: f32,
}

impl Default for CharacterCapsule {
    fn default() -> Self {
        Self {
            half_height: 0.5,
            radius: 0.3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn a_vertex_with_only_a_position_binds_fully_to_its_first_joint() {
        // Importers emit position-only vertices for unweighted geometry; the
        // defaults have to make that render white and rigid rather than black
        // and collapsed to the origin.
        let v: SkinnedVertexData = serde_json::from_str(r#"{"pos":[1,2,3]}"#).unwrap();
        assert_eq!(v.pos, [1.0, 2.0, 3.0]);
        assert_eq!(v.color, [1.0, 1.0, 1.0]);
        assert_eq!(v.uv, [0.0, 0.0]);
        assert_eq!(v.joints, [0, 0, 0, 0]);
        assert_eq!(v.weights, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_weighted_vertex_keeps_its_authored_joints_and_weights() {
        let v: SkinnedVertexData = serde_json::from_str(
            r#"{"pos":[0,0,0],"color":[0.5,0.5,0.5],"uv":[0.25,0.75],
                "joints":[3,4,0,0],"weights":[0.6,0.4,0,0]}"#,
        )
        .unwrap();
        assert_eq!(v.color, [0.5, 0.5, 0.5]);
        assert_eq!(v.uv, [0.25, 0.75]);
        assert_eq!(v.joints, [3, 4, 0, 0]);
        assert_eq!(v.weights, [0.6, 0.4, 0.0, 0.0]);
    }

    #[test]
    fn a_blank_joint_is_a_root_at_the_bind_pose_origin() {
        let j = SkeletonJoint::default();
        assert!(j.name.is_empty());
        // -1 is the root marker; 0 would make every joint a child of joint 0.
        assert_eq!(j.parent, -1);
        assert_eq!(j.translation, [0.0, 0.0, 0.0]);
        assert_eq!(j.scale, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn a_blank_morph_delta_moves_nothing() {
        let d = MorphDelta::default();
        assert_eq!(
            d,
            MorphDelta {
                position: [0.0; 3],
                normal: [0.0; 3],
            }
        );
    }

    #[test]
    fn a_blank_mesh_has_no_geometry_and_no_capsule() {
        let m = SkinnedMesh::default();
        assert!(m.vertices.is_empty());
        assert!(m.indices.is_empty());
        assert!(m.morph_target_names.is_empty());
        assert!(m.capsule.is_none());
        assert!(m.locator.is_none());
        assert_eq!(m.scale, [0.0, 0.0, 0.0]);
        let c = CharacterCapsule::default();
        assert_eq!((c.half_height, c.radius), (0.5, 0.3));
    }

    #[test]
    fn an_imported_mesh_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let m: SkinnedMesh = serde_json::from_str(
            r#"{"source":"hero.glb","skin_index":1,"material":"skin_mat","texture":"skin_tex",
                "vertices":[{"pos":[0,0,0]}],"indices":[0],
                "morph_target_names":["smile"],"morph_deltas":[{"position":[0,0.1,0]}],
                "position":[1,0,2],"scale":[1,1,1],"lod_levels":2,"lod_distances":[10],
                "max_instances":4,"capsule":{"half_height":0.9,"radius":0.35}}"#,
        )
        .unwrap();
        assert_eq!(m.material, Some(MaterialHandle(8)));
        assert_eq!(m.texture, Some(TextureHandle(8)));
        assert_eq!(m.morph_target_names, ["smile"]);

        let bytes = postcard::to_allocvec(&m).unwrap();
        let back: SkinnedMesh = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.source, "hero.glb");
        assert_eq!(back.skin_index, 1);
        assert_eq!(back.vertices[0].weights, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            back.morph_deltas,
            vec![MorphDelta {
                position: [0.0, 0.1, 0.0],
                normal: [0.0; 3],
            }]
        );
        assert_eq!(back.lod_distances, [10.0]);
        assert_eq!(back.max_instances, 4);
        assert_eq!(back.capsule.expect("capsule").half_height, 0.9);
        // Identity and payload location are injected at load, never authored.
        assert_eq!(back.asset_id, AssetId::default());
        assert!(back.locator.is_none());
    }
}

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
/// the schema half) because the matrix build goes through `gfx::skeleton`, which
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
mod runtime_tests {
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

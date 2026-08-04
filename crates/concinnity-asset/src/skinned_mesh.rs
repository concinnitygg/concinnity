// Skinned-mesh schema: skeletal geometry, its bind-pose joints, and an optional
// character capsule.

use crate::{
    AssetId, MaterialHandle, PayloadLocator, TextureHandle, de_opt_material_handle,
    de_opt_texture_handle,
};
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
pub struct JointDef {
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

impl Default for JointDef {
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
/// The `skeleton` (joint hierarchy and bind pose) is provided as an arg
/// (authored inline alongside `vertices`/`indices`, or filled in from the
/// imported `.glb`) and is baked into the mesh at build time.
///
/// Normals and tangents are computed automatically at build time. Do not
/// supply them.
///
/// ```jsonl
/// {"name":"flag","type":"SkinnedMesh","args":{"position":[0,1,0],"material":"mat_cloth","skeleton":[{"parent":-1},{"parent":0,"translation":[0,1,0]}],"vertices":[{"pos":[0,0,0],"joints":[0,0,0,0],"weights":[1,0,0,0]}],"indices":[0,0,0]}}
/// {"name":"hero","type":"SkinnedMesh","args":{"source":"models/hero.glb","position":[0,0,0],"material":"mat_skin"}}
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SkinnedMesh {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Optional path to a `.glb` file. When set, the build imports
    /// `vertices` / `indices` / `skeleton` from it; an inline-authored mesh
    /// leaves this empty.
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
        let j = JointDef::default();
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

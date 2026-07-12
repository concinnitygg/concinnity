// Skinned-mesh schema: skeletal geometry, its bind-pose joints, and an optional
// character capsule.

use crate::{AssetId, PayloadLocator, TextureHandle, de_opt_asset_ref, de_opt_texture_handle};
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
/// `skeleton`) or imported from a binary glTF file with `source`. Only the
/// `.glb` container is supported, and only the mesh + skeleton bind pose are
/// imported (glTF animations are not yet brought in).
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
    /// Skinned vertex list.
    pub vertices: Vec<SkinnedVertexData>,
    /// Triangle index list.
    pub indices: Vec<u16>,
    /// [Material](#material); provides the albedo texture plus lighting
    /// parameters.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub material: Option<AssetId>,
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

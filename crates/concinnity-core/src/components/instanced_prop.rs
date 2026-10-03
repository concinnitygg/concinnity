// Instanced-prop schema.

use crate::ecs::MaterialHandle;
use crate::ecs::MeshHandle;
use alloc::vec::Vec;

/// Per-instance transform within an `InstancedProp`.
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct InstanceTransform {
    /// World-space position `[x, y, z]`.
    pub position: [f32; 3],
    /// Euler rotation in degrees `[pitch, yaw, roll]`, applied in YXZ order.
    pub rotation_deg: [f32; 3],
    /// Non-uniform scale `[x, y, z]`.
    #[asset(default = [1.0, 1.0, 1.0])]
    pub scale: [f32; 3],
}

/// A single mesh + material drawn at many world-space transforms.
///
/// Use for foliage, debris, projectiles, or any content that repeats the same
/// shape with varied placement. Each instance gets its own world transform and
/// culling without the overhead of declaring many separate [Prop](#prop)s.
///
/// Each `instances` entry has the shape `{"position":[x,y,z], "rotation_deg":[p,y,r], "scale":[sx,sy,sz]}`.
/// `rotation_deg` and `scale` may be omitted (defaults `[0,0,0]` and `[1,1,1]`).
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct InstancedProp {
    /// A [Mesh](#mesh), [ProceduralMesh](#proceduralmesh),
    /// [VoxelChunk](#voxelchunk), or mesh-kind [File](#file) asset.
    pub mesh: Option<MeshHandle>,
    /// A [Material](#material) providing the albedo texture plus lighting parameters.
    pub material: Option<MaterialHandle>,
    /// Per-instance transforms. Empty list renders nothing.
    pub instances: Vec<InstanceTransform>,
    /// View-distance cutoff in world units per instance. 0 = always draw.
    pub cull_distance: f32,
}

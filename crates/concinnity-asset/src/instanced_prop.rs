// Instanced-prop schema.

use crate::{
    AssetId, MaterialHandle, MeshHandle, TextureHandle, de_opt_material_handle, de_opt_mesh_handle,
    de_opt_texture_handle,
};
use alloc::vec::Vec;

/// Per-instance transform within an `InstancedProp`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct InstanceTransform {
    /// World-space position `[x, y, z]`.
    pub position: [f32; 3],
    /// Euler rotation in degrees `[pitch, yaw, roll]`, applied in YXZ order.
    pub rotation_deg: [f32; 3],
    /// Non-uniform scale `[x, y, z]`.
    pub scale: [f32; 3],
}

impl Default for InstanceTransform {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

/// A single mesh + material drawn at many world-space transforms.
///
/// Use for foliage, debris, projectiles, or any content that repeats the same
/// shape with varied placement. Each instance gets its own world transform and
/// culling without the overhead of declaring many separate [Prop](#prop)s.
///
/// Each `instances` entry has the shape `{"position":[x,y,z], "rotation_deg":[p,y,r], "scale":[sx,sy,sz]}`.
/// `rotation_deg` and `scale` may be omitted (defaults `[0,0,0]` and `[1,1,1]`).
///
/// ```jsonl
/// {"name":"rock_mesh","type":"ProceduralMesh","args":{"generator":"sphere","radius":0.4,"rings":8,"segments":10}}
/// {"name":"mat_stone","type":"Material","args":{"albedo":"tex_stone","roughness":0.9}}
/// {"name":"rocks","type":"InstancedProp","args":{
///   "mesh":"rock_mesh",
///   "material":"mat_stone",
///   "cull_distance":80.0,
///   "instances":[
///     {"position":[ 2.0, 0.4, -3.0]},
///     {"position":[-5.0, 0.4,  1.0], "rotation_deg":[0, 45, 0]},
///     {"position":[ 4.0, 0.4,  7.0], "scale":[1.5, 1.5, 1.5]}
///   ]
/// }}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct InstancedProp {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// A [Mesh](#mesh), [ProceduralMesh](#proceduralmesh),
    /// [VoxelChunk](#voxelchunk), or mesh-kind [File](#file) asset.
    #[serde(deserialize_with = "de_opt_mesh_handle")]
    pub mesh: Option<MeshHandle>,
    /// A [Material](#material); takes precedence over `texture` when set.
    #[serde(deserialize_with = "de_opt_material_handle")]
    pub material: Option<MaterialHandle>,
    /// Older texture-only reference; ignored when `material` is set.
    #[serde(deserialize_with = "de_opt_texture_handle")]
    pub texture: Option<TextureHandle>,
    /// Per-instance transforms. Empty list renders nothing.
    pub instances: Vec<InstanceTransform>,
    /// View-distance cutoff in world units per instance. 0 = always draw.
    pub cull_distance: f32,
}

impl Default for InstancedProp {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            mesh: None,
            material: None,
            texture: None,
            instances: Vec::new(),
            cull_distance: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_instance_is_an_untransformed_copy() {
        let t = InstanceTransform::default();
        assert_eq!(t.position, [0.0, 0.0, 0.0]);
        assert_eq!(t.rotation_deg, [0.0, 0.0, 0.0]);
        // Unit scale, not zero: an omitted scale must not collapse the copy.
        assert_eq!(t.scale, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn a_blank_prop_draws_nothing_and_never_culls() {
        let p = InstancedProp::default();
        assert!(p.instances.is_empty());
        assert!(p.mesh.is_none());
        assert!(p.material.is_none());
        assert!(p.texture.is_none());
        // Zero means "no distance cull", not "cull everything".
        assert_eq!(p.cull_distance, 0.0);
    }

    #[test]
    fn an_authored_instance_list_parses_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let p: InstancedProp = serde_json::from_str(
            r#"{"mesh":"tree_mesh","material":"bark","texture":"bark_tex","cull_distance":120,
                "instances":[{"position":[1,0,2]},{"position":[3,0,4],"scale":[2,2,2]}]}"#,
        )
        .unwrap();
        assert_eq!(p.mesh, Some(MeshHandle(9)));
        assert_eq!(p.material, Some(MaterialHandle(4)));
        assert_eq!(p.texture, Some(TextureHandle(8)));
        assert_eq!(p.instances.len(), 2);
        // A copy that mentions only its position keeps unit scale.
        assert_eq!(p.instances[0].scale, [1.0, 1.0, 1.0]);
        assert_eq!(p.instances[1].scale, [2.0, 2.0, 2.0]);

        let bytes = postcard::to_allocvec(&p).unwrap();
        let back: InstancedProp = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.instances.len(), 2);
        assert_eq!(back.instances[1].position, [3.0, 0.0, 4.0]);
        assert_eq!(back.cull_distance, 120.0);
        assert_eq!(back.asset_id, AssetId::default());
    }
}

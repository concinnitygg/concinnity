// Multi-mesh model schema.

use crate::{AssetId, MeshHandle, de_opt_asset_ref, de_opt_mesh_handle};
use alloc::vec::Vec;

/// One geometric part of a Model, referencing a mesh and its surface material.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SubMeshRef {
    /// A [Mesh](#mesh) or [ProceduralMesh](#proceduralmesh) asset.
    #[serde(default, deserialize_with = "de_opt_mesh_handle")]
    pub mesh: Option<MeshHandle>,
    /// A [Material](#material) asset.  `None` uses the default material.
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub material: Option<AssetId>,
}

/// An ordered list of sub-meshes, each with its own material.
///
/// Use via the `model` field on a [Prop](#prop) instead of `mesh`. Each
/// sub-mesh is drawn with its own material, all sharing the prop's transform.
///
/// Each `mesh` must name a [Mesh](#mesh) or [ProceduralMesh](#proceduralmesh)
/// asset present in the scene. `material` may be empty to use the default
/// material.
///
/// ```jsonl
/// {"name":"crate_body","type":"ProceduralMesh","args":{"generator":"box","half_extents":[0.3,0.3,0.3]}}
/// {"name":"crate_bands","type":"ProceduralMesh","args":{"generator":"box","half_extents":[0.31,0.04,0.31]}}
/// {"name":"mat_wood","type":"Material","args":{"albedo":"tex_wood","roughness":0.75,"metallic":0.0}}
/// {"name":"mat_metal","type":"Material","args":{"albedo":"tex_metal","roughness":0.4,"metallic":1.0}}
/// {"name":"wooden_crate","type":"Model","args":{"meshes":[
///   {"mesh":"crate_body", "material":"mat_wood"},
///   {"mesh":"crate_bands","material":"mat_metal"}
/// ]}}
/// {"name":"crate_a","type":"Prop","args":{"model":"wooden_crate","position":[2.0,0.3,-4.0]}}
/// {"name":"crate_b","type":"Prop","args":{"model":"wooden_crate","position":[-1.5,0.3,-6.0],"rotation_deg":[0,45,0]}}
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Model {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Ordered list of sub-meshes that make up this model.
    pub meshes: Vec<SubMeshRef>,
}

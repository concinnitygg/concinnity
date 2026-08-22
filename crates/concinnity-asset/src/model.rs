// Multi-mesh model schema.

use crate::{AssetId, MaterialHandle, MeshHandle, de_opt_material_handle, de_opt_mesh_handle};
use alloc::vec::Vec;

/// One geometric part of a Model, referencing a mesh and its surface material.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SubMeshRef {
    /// A [Mesh](#mesh) or [ProceduralMesh](#proceduralmesh) asset.
    #[serde(default, deserialize_with = "de_opt_mesh_handle")]
    pub mesh: Option<MeshHandle>,
    /// A [Material](#material) asset.  `None` uses the default material.
    #[serde(default, deserialize_with = "de_opt_material_handle")]
    pub material: Option<MaterialHandle>,
}

/// An ordered list of sub-meshes, each with its own material.
///
/// Use via the `model` field on a [Prop](#prop) instead of `mesh`. Each
/// sub-mesh is drawn with its own material, all sharing the prop's transform.
///
/// Each `mesh` must name a [Mesh](#mesh) or [ProceduralMesh](#proceduralmesh)
/// asset present in the scene. `material` may be empty to use the default
/// material.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Model {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Ordered list of sub-meshes that make up this model.
    pub meshes: Vec<SubMeshRef>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_model_has_no_sub_meshes() {
        let m = Model::default();
        assert!(m.meshes.is_empty());
        assert_eq!(m.asset_id, AssetId::default());
    }

    #[test]
    fn a_sub_mesh_may_declare_geometry_without_a_material() {
        crate::test_support::install_resolvers();
        let s: SubMeshRef = serde_json::from_str(r#"{"mesh":"body"}"#).unwrap();
        assert_eq!(s.mesh, Some(MeshHandle(4)));
        // No material means the sub-mesh inherits whatever the prop supplies.
        assert_eq!(s.material, None);
        let s: SubMeshRef = serde_json::from_str("{}").unwrap();
        assert_eq!(s.mesh, None);
        assert_eq!(s.material, None);
    }

    #[test]
    fn an_imported_model_keeps_its_sub_mesh_order_through_postcard() {
        crate::test_support::install_resolvers();
        let m: Model = serde_json::from_str(
            r#"{"meshes":[{"mesh":"body","material":"skin"},
                         {"mesh":"trim","material":"metal"}]}"#,
        )
        .unwrap();
        assert_eq!(m.meshes.len(), 2);

        let bytes = postcard::to_allocvec(&m).unwrap();
        let back: Model = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.meshes[0].mesh, Some(MeshHandle(4)));
        assert_eq!(back.meshes[0].material, Some(MaterialHandle(4)));
        assert_eq!(back.meshes[1].mesh, Some(MeshHandle(4)));
        assert_eq!(back.meshes[1].material, Some(MaterialHandle(5)));
        assert_eq!(back.asset_id, AssetId::default());
    }
}

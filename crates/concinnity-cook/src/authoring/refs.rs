//! Asset reference-graph extraction: the names an asset references, from the
//! same two sources the cross-reference validator resolves -- the reference
//! fields the registry derives from each schema's field types and the
//! structured `CrossReferenced` impls. Their union is the complete reference set
//! by construction (a hand impl never re-checks a derived field), so consumers
//! (scene partitioning, provenance tooling) get the full edge list without
//! per-type knowledge.

use crate::authoring::field_path::string_leaves;
use crate::authoring::world::WorldJsonlAsset;
use crate::check::asset_refs::CrossRef;
use crate::check::cross_reference::cross_refs_for;

/// The names of every asset `asset` references, in field/declaration order.
/// Names are returned as authored; callers resolve them against the world's
/// asset list. Unresolvable or empty references are omitted, not errors.
pub fn referenced_names(asset: &WorldJsonlAsset) -> Vec<String> {
    let mut names = Vec::new();

    for field in asset.asset_type.ref_fields() {
        for (_, name) in string_leaves(&asset.args, &field.path) {
            names.push(name.to_string());
        }
    }

    for cross_ref in cross_refs_for(asset.asset_type, &asset.id, &asset.args) {
        if let CrossRef::Resolve { target, .. } = cross_ref {
            names.push(target);
        }
    }

    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authoring::registry::RegisteredType;

    fn asset(name: &str, asset_type: RegisteredType, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            id: name.to_string(),
            asset_type,
            args,
        }
    }

    #[test]
    fn prop_typed_and_structured_refs_are_unioned() {
        // `material` is a typed field; `mesh` is structured (polymorphic
        // MeshSource, kept in the hand impl).
        let refs = referenced_names(&asset(
            "p",
            RegisteredType::Prop,
            serde_json::json!({"mesh":"box","material":"mat"}),
        ));
        assert!(refs.contains(&"box".to_string()));
        assert!(refs.contains(&"mat".to_string()));
    }

    #[test]
    fn material_texture_slots_come_from_the_resource_registry() {
        let refs = referenced_names(&asset(
            "m",
            RegisteredType::Material,
            serde_json::json!({"albedo":"tex_a","normal_map":"tex_n"}),
        ));
        assert_eq!(refs, vec!["tex_a".to_string(), "tex_n".to_string()]);
    }

    #[test]
    fn model_submesh_list_is_structured() {
        let refs = referenced_names(&asset(
            "mdl",
            RegisteredType::Model,
            serde_json::json!({"meshes":[{"mesh":"m0","material":"mat0"},{"mesh":"m1"}]}),
        ));
        assert!(refs.contains(&"m0".to_string()));
        assert!(refs.contains(&"mat0".to_string()));
        assert!(refs.contains(&"m1".to_string()));
    }

    #[test]
    fn list_and_nested_references_are_derived() {
        let refs = referenced_names(&asset(
            "c",
            RegisteredType::VoxelChunk,
            serde_json::json!({"palette":["stone","dirt"]}),
        ));
        assert_eq!(refs, vec!["stone".to_string(), "dirt".to_string()]);
        let refs = referenced_names(&asset(
            "cam",
            RegisteredType::Camera3D,
            serde_json::json!({"controller":{"follow":{"target":"hero"}}}),
        ));
        assert_eq!(refs, vec!["hero".to_string()]);
    }

    #[test]
    fn empty_and_absent_fields_are_omitted() {
        let refs = referenced_names(&asset(
            "p",
            RegisteredType::Prop,
            serde_json::json!({"model":""}),
        ));
        assert!(refs.is_empty());
    }

    #[test]
    fn a_type_without_references_has_no_refs() {
        let light = asset(
            "x",
            RegisteredType::PointLight,
            serde_json::json!({"model": "m"}),
        );
        assert!(referenced_names(&light).is_empty());
    }
}

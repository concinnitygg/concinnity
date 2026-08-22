//! Asset reference-graph extraction: the names an asset references, from the
//! same two sources the cross-reference validator resolves — the registries'
//! flat `refs:` metadata and the structured `CrossReferenced` impls. Their
//! union is the complete reference set by construction (a hand impl never
//! re-checks a registry-declared field), so consumers (scene partitioning,
//! provenance tooling) get the full edge list without per-type knowledge.

use crate::check::asset_refs::CrossRef;
use crate::check::cross_reference::cross_refs_for;
use crate::registry::ComponentType;
use crate::resource_type::ResourceAssetType;
use crate::world::WorldJsonlAsset;

/// The names of every asset `asset` references, in field/declaration order.
/// Names are returned as authored; callers resolve them against the world's
/// asset list. Unresolvable or empty references are omitted, not errors.
pub fn referenced_names(asset: &WorldJsonlAsset) -> Vec<String> {
    let norm = |t: &str| t.to_lowercase().replace('_', "");
    let type_norm = norm(&asset.asset_type);

    let mut names = Vec::new();

    let flat_refs = ComponentType::all()
        .iter()
        .map(|t| (t.as_str(), t.ref_fields()))
        .chain(
            ResourceAssetType::all()
                .iter()
                .map(|t| (t.as_str(), t.ref_fields())),
        )
        .filter(|(ty, _)| norm(ty) == type_norm)
        .flat_map(|(_, refs)| refs.iter());
    for &(field, _target) in flat_refs {
        if let Some(name) = asset
            .args
            .get(field)
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
        {
            names.push(name.to_string());
        }
    }

    for cross_ref in cross_refs_for(&type_norm, &asset.name, &asset.args) {
        if let CrossRef::Resolve { target, .. } = cross_ref {
            names.push(target);
        }
    }

    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str, ty: &str, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            name: name.to_string(),
            asset_type: ty.to_string(),
            args,
        }
    }

    #[test]
    fn prop_flat_and_structured_refs_are_unioned() {
        // `model` is registry-declared; `mesh` is structured (polymorphic
        // MeshSource, kept in the hand impl).
        let refs = referenced_names(&asset(
            "p",
            "Prop",
            serde_json::json!({"mesh":"box","material":"mat"}),
        ));
        assert!(refs.contains(&"box".to_string()));
        assert!(refs.contains(&"mat".to_string()));
    }

    #[test]
    fn material_texture_slots_come_from_the_resource_registry() {
        let refs = referenced_names(&asset(
            "m",
            "Material",
            serde_json::json!({"albedo":"tex_a","normal_map":"tex_n"}),
        ));
        assert_eq!(refs, vec!["tex_a".to_string(), "tex_n".to_string()]);
    }

    #[test]
    fn model_submesh_list_is_structured() {
        let refs = referenced_names(&asset(
            "mdl",
            "Model",
            serde_json::json!({"meshes":[{"mesh":"m0","material":"mat0"},{"mesh":"m1"}]}),
        ));
        assert!(refs.contains(&"m0".to_string()));
        assert!(refs.contains(&"mat0".to_string()));
        assert!(refs.contains(&"m1".to_string()));
    }

    #[test]
    fn empty_and_absent_fields_are_omitted() {
        let refs = referenced_names(&asset("p", "Prop", serde_json::json!({"model":""})));
        assert!(refs.is_empty());
    }

    #[test]
    fn unknown_type_has_no_refs() {
        assert!(referenced_names(&asset("x", "NotAType", serde_json::json!({}))).is_empty());
    }
}

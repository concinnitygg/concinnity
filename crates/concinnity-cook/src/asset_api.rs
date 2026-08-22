//! Shared asset construction API.
//!
//! This module is the single place where "type name + JSON args → BlobAssetDef"
//! is implemented.
use crate::ecs::{AssetKind, AssetOrigin, BlobAssetDef};
use crate::registry::ComponentType;
use crate::registry::Registration;
use crate::result::CnResult;

/// Incoming request to construct an asset from an external caller
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssetRequest {
    /// type name as it appears in the world declaration ("Mesh", "Material", ...)
    /// case-insensitive; underscores ignored
    pub asset_type: String,
    /// constructor args. If None, the type's default_args are used
    #[serde(default)]
    pub args: Option<serde_json::Value>,
}

// Describes one addable asset type: its name, what it is, and the authoring
// metadata a caller needs to construct one. The summary is the first line of the
// type's reference documentation, so a picker can say what an asset does rather
// than only naming it.
#[cfg(test)]
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct AssetTypeEntry {
    pub asset_type: String,
    pub summary: String,
    pub registration: Registration,
}

/// Validate an AssetRequest and produce a BlobAssetDef
///
/// Returns Err if:
/// - The type name is unknown
/// - The type's origin is not External (not addable)
/// - The resolved args cannot be serialized
///
/// Does not perform payload compilation (shaders, images, etc.). The build
/// step calls this first, then runs its compilation pass over the resulting
/// defs. The HTTP API follows the same two-step pattern
pub fn create_asset_def(req: &AssetRequest) -> Result<BlobAssetDef, CnResult> {
    if let Some(ct) = ComponentType::parse(&req.asset_type) {
        let reg = ct.registration();
        if reg.origin != AssetOrigin::External {
            return Err(CnResult::InvalidArgument);
        }
        let args = resolve_args(&reg, &req.args);
        // Every record is baked. For a pass-through type the baked component is
        // its reserialized args (the component IS its args); a divergent type
        // (`Args != Self`) bakes the translated component instead.
        let args_bytes = match crate::registry::bake_divergent(ct, &args)? {
            Some(bytes) => bytes,
            None => ct.reserialize_args(&args)?,
        };
        return Ok(BlobAssetDef {
            name: None,
            kind: AssetKind::Component,
            discriminant: ct.discriminant(),
            args_bytes,
            payload: None,
        });
    }

    tracing::error!("asset_api: unknown asset type '{}'", req.asset_type);
    Err(CnResult::AssetInvalidType)
}

// List every externally-addable component type with its registration metadata.
#[cfg(test)]
pub(crate) fn list_addable_types() -> Vec<AssetTypeEntry> {
    let mut entries: Vec<AssetTypeEntry> = ComponentType::addable_types()
        .map(|(ct, reg)| AssetTypeEntry {
            asset_type: ct.as_str().to_string(),
            summary: concinnity_docs::summary(ct.as_str())
                .unwrap_or_default()
                .to_string(),
            registration: reg,
        })
        .collect();

    entries.sort_by(|a, b| a.asset_type.cmp(&b.asset_type));
    entries
}

// Resolve the args to use for construction.
//
// Merges supplied args over the type's defaults so that missing keys are filled
// in automatically. This lets callers supply partial args (including `{}`) and
// still get sensible values for any fields they omit.
fn resolve_args(reg: &Registration, supplied: &Option<serde_json::Value>) -> serde_json::Value {
    let mut base = reg
        .default_args
        .clone()
        .unwrap_or_else(|| serde_json::Value::Object(Default::default()));

    if let Some(serde_json::Value::Object(supplied_map)) = supplied {
        if let serde_json::Value::Object(ref mut base_map) = base {
            for (k, v) in supplied_map {
                base_map.insert(k.clone(), v.clone());
            }
        }
    } else if let Some(v) = supplied {
        base = v.clone();
    }

    base
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader_reg() -> Registration {
        Registration {
            type_name: "VertexStage",
            origin: AssetOrigin::External,
            payload: crate::ecs::AssetPayload::Compiled,
            default_args: Some(serde_json::json!({ "source": "user.metal" })),
        }
    }

    #[test]
    fn resolve_args_none_uses_default() {
        let reg = shader_reg();
        let result = resolve_args(&reg, &None);
        assert_eq!(result["source"], "user.metal");
    }

    #[test]
    fn resolve_args_empty_object_fills_from_default() {
        let reg = shader_reg();
        let supplied = Some(serde_json::json!({}));
        let result = resolve_args(&reg, &supplied);
        assert_eq!(result["source"], "user.metal");
    }

    #[test]
    fn resolve_args_supplied_value_wins() {
        let reg = shader_reg();
        let supplied = Some(serde_json::json!({ "source": "custom.metal" }));
        let result = resolve_args(&reg, &supplied);
        assert_eq!(result["source"], "custom.metal");
    }

    #[test]
    fn resolve_args_partial_keeps_default_for_missing_keys() {
        let reg = Registration {
            type_name: "Fake",
            origin: AssetOrigin::External,
            payload: crate::ecs::AssetPayload::None,
            default_args: Some(serde_json::json!({ "a": 1, "b": 2 })),
        };
        let supplied = Some(serde_json::json!({ "b": 99 }));
        let result = resolve_args(&reg, &supplied);
        assert_eq!(result["a"], 1);
        assert_eq!(result["b"], 99);
    }

    #[test]
    fn resolve_args_non_object_supplied_replaces_the_default() {
        let reg = shader_reg();
        let supplied = Some(serde_json::json!([1, 2, 3]));
        let result = resolve_args(&reg, &supplied);
        assert_eq!(result, serde_json::json!([1, 2, 3]));
    }

    // Merging is only defined between two objects: supplied args cannot be
    // folded into a non-object default, so the default is kept as-is.
    #[test]
    fn resolve_args_keeps_a_non_object_default_when_an_object_is_supplied() {
        let reg = Registration {
            type_name: "Fake",
            origin: AssetOrigin::External,
            payload: crate::ecs::AssetPayload::None,
            default_args: Some(serde_json::json!([1, 2, 3])),
        };
        let supplied = Some(serde_json::json!({ "a": 1 }));
        assert_eq!(resolve_args(&reg, &supplied), serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn create_asset_def_rejects_unknown_types() {
        let req = AssetRequest {
            asset_type: "NotARealAsset".to_string(),
            args: None,
        };
        assert_eq!(
            create_asset_def(&req).unwrap_err(),
            CnResult::AssetInvalidType
        );
    }

    #[test]
    fn create_asset_def_rejects_runtime_only_types() {
        // Transform is registered but RuntimeOnly, so it is not addable.
        let req = AssetRequest {
            asset_type: "Transform".to_string(),
            args: None,
        };
        assert_eq!(
            create_asset_def(&req).unwrap_err(),
            CnResult::InvalidArgument
        );
    }

    #[test]
    fn create_asset_def_builds_a_component_def() {
        let req = AssetRequest {
            asset_type: "ProceduralMesh".to_string(),
            args: None,
        };
        let def = create_asset_def(&req).unwrap();
        assert_eq!(def.kind, AssetKind::Component);
        assert_eq!(
            def.discriminant,
            ComponentType::parse("ProceduralMesh")
                .unwrap()
                .discriminant()
        );
        assert!(def.name.is_none());
        assert!(def.payload.is_none());
        // The baked bytes decode as the component (postcard).
        postcard::from_bytes::<crate::assets::ProceduralMesh>(&def.args_bytes).unwrap();
    }

    // Every addable type builds a baked def from its default args: the def's
    // bytes reconstruct through `from_baked` at load. A new type whose baked
    // form cannot round-trip its defaults fails here.
    #[test]
    fn every_addable_type_builds_a_baked_def() {
        for (ct, _) in ComponentType::addable_types() {
            let def = create_asset_def(&AssetRequest {
                asset_type: ct.as_str().to_string(),
                args: None,
            })
            .unwrap();
            assert_eq!(def.kind, AssetKind::Component, "{}", ct.as_str());
            assert!(!def.args_bytes.is_empty(), "{}", ct.as_str());
        }
    }

    #[test]
    fn create_asset_def_merges_supplied_args_over_defaults() {
        let req = AssetRequest {
            asset_type: "ProceduralMesh".to_string(),
            args: Some(serde_json::json!({ "generator": "box" })),
        };
        let def = create_asset_def(&req).unwrap();
        let baked: crate::assets::ProceduralMesh = postcard::from_bytes(&def.args_bytes).unwrap();
        assert_eq!(baked.generator, "box");
        // Defaults fill the fields the caller omitted.
        let defaults = crate::assets::ProceduralMesh::default();
        assert_eq!(baked.half_width, defaults.half_width);
        assert_eq!(baked.ceiling_height, defaults.ceiling_height);
    }

    #[test]
    fn create_asset_def_rejects_mistyped_args() {
        let req = AssetRequest {
            asset_type: "ProceduralMesh".to_string(),
            args: Some(serde_json::json!({ "generator": 42 })),
        };
        assert_eq!(
            create_asset_def(&req).unwrap_err(),
            CnResult::InvalidArgument
        );
    }

    #[test]
    fn addable_type_listing_is_sorted_and_external_only() {
        let entries = list_addable_types();
        assert!(!entries.is_empty());
        assert!(
            entries
                .windows(2)
                .all(|w| w[0].asset_type <= w[1].asset_type)
        );
        assert!(
            entries
                .iter()
                .all(|e| e.registration.origin == AssetOrigin::External)
        );
        assert!(entries.iter().any(|e| e.asset_type == "ProceduralMesh"));
        assert!(entries.iter().all(|e| e.asset_type != "Transform"));
    }

    // Every addable type carries its reference summary, so a picker can describe
    // what it offers. An addable type is authorable, so it always has a page.
    #[test]
    fn addable_type_listing_carries_summaries() {
        for e in list_addable_types() {
            assert!(
                !e.summary.is_empty(),
                "{} has no summary in the asset reference",
                e.asset_type
            );
            assert!(
                !e.summary.contains('\n'),
                "{} summary is multi-line",
                e.asset_type
            );
        }
    }
}

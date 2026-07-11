// Shared asset construction API.
//
// This module is the single place where "type name + JSON args → BlobAssetDef"
// is implemented.
use crate::ecs::{AssetKind, AssetOrigin, BlobAssetDef, RecordKind, Registration};
use crate::registry::ComponentType;
use crate::result::CnResult;

// Incoming request to construct an asset from an external caller
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssetRequest {
    // type name as it appears in the world declaration ("Mesh", "Material", ...)
    // case-insensitive; underscores ignored
    pub asset_type: String,
    // constructor args. If None, the type's default_args are used
    #[serde(default)]
    pub args: Option<serde_json::Value>,
}

// Describes one addable asset type. Returned by list_addable_types() and
// used by HTTP GET /assets/types and CLI help output
#[derive(Debug, Clone, serde::Serialize)]
#[allow(dead_code)]
pub struct AssetTypeEntry {
    pub asset_type: String,
    pub registration: Registration,
}

// Validate an AssetRequest and produce a BlobAssetDef
//
// Returns Err if:
// - The type name is unknown
// - The type's origin is not External (not addable)
// - The resolved args cannot be serialized
//
// Does not perform payload compilation (shaders, images, etc.). The build
// step calls this first, then runs its compilation pass over the resulting
// defs. The HTTP API follows the same two-step pattern
pub fn create_asset_def(req: &AssetRequest) -> Result<BlobAssetDef, CnResult> {
    if let Some(ct) = ComponentType::parse(&req.asset_type) {
        let reg = ct.registration();
        if reg.origin != AssetOrigin::External {
            return Err(CnResult::InvalidArgument);
        }
        let args = resolve_args(&reg, &req.args);
        // A type that has migrated to the baked path emits a `Baked` record. For
        // a pass-through leaf the baked component is its authored args, so the
        // bytes are the same reserialization; only the record kind differs, and
        // the runtime reconstructs it identically via `from_baked`.
        let record = if ct.baked() {
            RecordKind::Baked
        } else {
            RecordKind::Authored
        };
        return Ok(BlobAssetDef {
            name: None,
            kind: AssetKind::Component,
            record,
            discriminant: ct.discriminant(),
            args_bytes: ct.reserialize_args(&args)?,
            payload: None,
        });
    }

    tracing::error!("asset_api: unknown asset type '{}'", req.asset_type);
    Err(CnResult::AssetInvalidType)
}

// List every externally-addable component type with its registration metadata.
#[allow(dead_code)]
pub fn list_addable_types() -> Vec<AssetTypeEntry> {
    let mut entries: Vec<AssetTypeEntry> = ComponentType::addable_types()
        .map(|(ct, reg)| AssetTypeEntry {
            asset_type: ct.as_str().to_string(),
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
            default_args: Some(serde_json::json!({ "source": "default.metal" })),
        }
    }

    #[test]
    fn resolve_args_none_uses_default() {
        let reg = shader_reg();
        let result = resolve_args(&reg, &None);
        assert_eq!(result["source"], "default.metal");
    }

    #[test]
    fn resolve_args_empty_object_fills_from_default() {
        let reg = shader_reg();
        let supplied = Some(serde_json::json!({}));
        let result = resolve_args(&reg, &supplied);
        assert_eq!(result["source"], "default.metal");
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
            asset_type: "Mesh".to_string(),
            args: None,
        };
        let def = create_asset_def(&req).unwrap();
        assert_eq!(def.kind, AssetKind::Component);
        assert_eq!(
            def.discriminant,
            ComponentType::parse("Mesh").unwrap().discriminant()
        );
        assert!(def.name.is_none());
        assert!(def.payload.is_none());
        let args: serde_json::Value = serde_json::from_slice(&def.args_bytes).unwrap();
        assert!(args.is_object());
    }

    // The record kind a def gets follows the type's `baked()` flag, so a
    // migrated type emits `Baked` and an unmigrated one stays `Authored`. Driven
    // by the flag rather than a hard-coded type list, so it stays correct as
    // more types migrate. PointLight is pinned as an already-migrated leaf.
    #[test]
    fn create_asset_def_record_kind_follows_baked_flag() {
        for (ct, _) in ComponentType::addable_types() {
            let def = create_asset_def(&AssetRequest {
                asset_type: ct.as_str().to_string(),
                args: None,
            })
            .unwrap();
            let expected = if ct.baked() {
                RecordKind::Baked
            } else {
                RecordKind::Authored
            };
            assert_eq!(
                def.record,
                expected,
                "{} record kind must follow its baked() flag",
                ct.as_str()
            );
        }
        assert!(ComponentType::PointLight.baked());
    }

    #[test]
    fn create_asset_def_merges_supplied_args_over_defaults() {
        let req = AssetRequest {
            asset_type: "Mesh".to_string(),
            args: Some(serde_json::json!({ "source": "a.glb" })),
        };
        let def = create_asset_def(&req).unwrap();
        let args: serde_json::Value = serde_json::from_slice(&def.args_bytes).unwrap();
        assert_eq!(args["source"], "a.glb");
        // Defaults fill the fields the caller omitted.
        assert_eq!(args["lod_levels"], 1);
    }

    #[test]
    fn create_asset_def_rejects_mistyped_args() {
        let req = AssetRequest {
            asset_type: "Mesh".to_string(),
            args: Some(serde_json::json!({ "source": 42 })),
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
        assert!(entries.iter().any(|e| e.asset_type == "Mesh"));
        assert!(entries.iter().all(|e| e.asset_type != "Transform"));
    }
}

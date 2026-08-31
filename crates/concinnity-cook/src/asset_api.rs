//! Shared asset construction API.
//!
//! This module is the single place where "type name + JSON args → BlobAssetDef"
//! is implemented.
use crate::ecs::{AssetKind, AssetOrigin, BlobAssetDef};
use crate::registry::RegisteredType;
use crate::registry::Registration;
use crate::result::CnResult;
use concinnity_core::platform::Platform;

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
///
/// `platform` is the shader platform the world is cooked for; the bake reads it
/// for the types that resolve a per-backend shader source.
pub fn create_asset_def(req: &AssetRequest, platform: Platform) -> Result<BlobAssetDef, CnResult> {
    if let Some(ct) = RegisteredType::parse(&req.asset_type) {
        let reg = ct.registration();
        if reg.origin != AssetOrigin::External {
            return Err(CnResult::InvalidArgument);
        }
        // A resource asset is External too, but compiles into the resource
        // stream rather than a component record, so it has no tag to carry.
        let discriminant = ct.discriminant().ok_or(CnResult::InvalidArgument)?;
        let args = resolve_args(&reg, &req.args);
        // Every record is baked. For a pass-through type the baked component is
        // its reserialized args (the component IS its args); a divergent type
        // (`Args != Self`) bakes the translated component instead.
        let args_bytes = match crate::registry::bake_divergent(ct, &args)? {
            Some(bytes) => bytes,
            None => ct.reserialize_args(&args, platform)?,
        };
        return Ok(BlobAssetDef {
            name: None,
            kind: AssetKind::Component,
            discriminant,
            args_bytes,
            payload: None,
        });
    }

    tracing::error!("asset_api: unknown asset type '{}'", req.asset_type);
    Err(CnResult::AssetInvalidType)
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
            create_asset_def(&req, Platform::Metal).unwrap_err(),
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
            create_asset_def(&req, Platform::Metal).unwrap_err(),
            CnResult::InvalidArgument
        );
    }

    #[test]
    fn create_asset_def_builds_a_component_def() {
        let req = AssetRequest {
            asset_type: "ProceduralMesh".to_string(),
            args: None,
        };
        let def = create_asset_def(&req, Platform::Metal).unwrap();
        assert_eq!(def.kind, AssetKind::Component);
        assert_eq!(
            Some(def.discriminant),
            RegisteredType::parse("ProceduralMesh")
                .unwrap()
                .discriminant()
        );
        assert!(def.name.is_none());
        assert!(def.payload.is_none());
        // The baked bytes decode as the component (postcard).
        postcard::from_bytes::<crate::components::ProceduralMesh>(&def.args_bytes).unwrap();
    }

    // Every addable component type builds a baked def from its default args: the
    // def's bytes reconstruct through `from_baked` at load. A new type whose
    // baked form cannot round-trip its defaults fails here. A resource asset has
    // no component def; it compiles into the resource stream instead.
    #[test]
    fn every_addable_component_type_builds_a_baked_def() {
        for (ct, _) in RegisteredType::addable_types().filter(|(t, _)| !t.is_resource()) {
            let def = create_asset_def(
                &AssetRequest {
                    asset_type: ct.as_str().to_string(),
                    args: None,
                },
                Platform::Metal,
            )
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
        let def = create_asset_def(&req, Platform::Metal).unwrap();
        let baked: crate::components::ProceduralMesh =
            postcard::from_bytes(&def.args_bytes).unwrap();
        assert_eq!(baked.generator, "box");
        // Defaults fill the fields the caller omitted.
        let defaults = crate::components::ProceduralMesh::default();
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
            create_asset_def(&req, Platform::Metal).unwrap_err(),
            CnResult::InvalidArgument
        );
    }

    // Every addable type is authorable, and only the authorable ones are
    // addable. Order is the registry's, which is the discriminant order, not
    // alphabetical.
    #[test]
    fn addable_types_are_external_only() {
        let names: Vec<&str> = RegisteredType::addable_types()
            .inspect(|(_, reg)| assert_eq!(reg.origin, AssetOrigin::External))
            .map(|(ct, _)| ct.as_str())
            .collect();
        assert!(!names.is_empty());
        assert!(names.contains(&"ProceduralMesh"));
        assert!(!names.contains(&"Transform"));
    }
}

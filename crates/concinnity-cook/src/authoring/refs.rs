//! Asset reference-graph extraction: the names an asset references, from the
//! same two sources the cross-reference validator resolves -- the reference
//! fields the registry derives from each schema's field types and the
//! structured `CrossReferenced` impls. Their union is the complete reference set
//! by construction (a hand impl never re-checks a derived field), so consumers
//! (scene partitioning, provenance tooling) get the full edge list without
//! per-type knowledge.

use crate::authoring::field_path::{retarget_leaves, string_leaves};
use crate::authoring::registry::RegisteredType;
use crate::authoring::resource_type::is_mesh_source;
use crate::authoring::world::WorldJsonlAsset;
use crate::check::asset_refs::{CrossRef, RefKind};
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

/// Point every reference field of `entry` that may name a `target_type` asset
/// and names `from` at `to` instead, or, with `None`, drop it so the field
/// takes its default. Returns how many references changed. Covers the fields
/// the registry derives from the schema, not the structured
/// `CrossReferenced` ones.
pub fn retarget_references(
    entry: &mut serde_json::Value,
    target_type: &str,
    from: &str,
    to: Option<&str>,
) -> usize {
    let Some(ty) = entry
        .get("type")
        .and_then(|t| t.as_str())
        .and_then(RegisteredType::parse)
    else {
        return 0;
    };
    let Some(args) = entry.get_mut("args") else {
        return 0;
    };
    ty.ref_fields()
        .iter()
        .filter(|f| f.targets.is_empty() || f.targets.contains(&target_type))
        .map(|f| retarget_leaves(args, &f.path, from, to))
        .sum()
}

/// Point every reference in `entry` naming `from` at `to`, for a rename of
/// that asset, a `target` declared with `target_args`: the fields the registry
/// derives and the structured references the build resolves by name (a Prop's
/// `mesh`, a Model's submeshes, a Behavior's nodes and trigger sources, an
/// AnimationGraph's blendspace clips). Returns how many references changed.
pub fn rename_references(
    entry: &mut serde_json::Value,
    target: RegisteredType,
    target_args: &serde_json::Value,
    from: &str,
    to: &str,
) -> usize {
    let mut moved = retarget_references(entry, target.as_str(), from, Some(to));
    let Some(ty) = entry
        .get("type")
        .and_then(|t| t.as_str())
        .and_then(RegisteredType::parse)
    else {
        return moved;
    };
    let Some(args) = entry.get_mut("args") else {
        return moved;
    };
    let accepts = |kind: RefKind| match kind {
        RefKind::MeshSource => is_mesh_source(target, target_args),
        RefKind::Scene => target == RegisteredType::Scene,
        RefKind::Animation => target == RegisteredType::Animation,
        RefKind::AudioClip => target == RegisteredType::AudioClip,
        RefKind::Screen => target == RegisteredType::Screen,
        RefKind::TriggerVolume => target == RegisteredType::TriggerVolume,
        RefKind::AnyAsset => true,
    };
    let naming = |args: &serde_json::Value| {
        cross_refs_for(ty, "", args)
            .iter()
            .filter(|r| {
                matches!(r, CrossRef::Resolve { kind, target, .. } if target == from && accepts(*kind))
            })
            .count()
    };
    // A string reading `from` is a reference exactly when the build reads it
    // as one: rewriting it leaves one reference fewer to `from`.
    let mut left = naming(args);
    for pointer in pointers_to(args, from) {
        if left == 0 {
            break;
        }
        set_pointer(args, &pointer, to);
        let now = naming(args);
        if now < left {
            left = now;
            moved += 1;
        } else {
            set_pointer(args, &pointer, from);
        }
    }
    moved
}

// The JSON pointer of every string in `value` that reads `text`.
fn pointers_to(value: &serde_json::Value, text: &str) -> Vec<String> {
    fn walk(value: &serde_json::Value, at: String, text: &str, out: &mut Vec<String>) {
        match value {
            serde_json::Value::String(s) if s == text => out.push(at),
            serde_json::Value::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    walk(item, format!("{at}/{i}"), text, out);
                }
            }
            serde_json::Value::Object(map) => {
                for (key, item) in map {
                    let key = key.replace('~', "~0").replace('/', "~1");
                    walk(item, format!("{at}/{key}"), text, out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(value, String::new(), text, &mut out);
    out
}

fn set_pointer(value: &mut serde_json::Value, pointer: &str, text: &str) {
    if let Some(leaf) = value.pointer_mut(pointer) {
        *leaf = serde_json::Value::String(text.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn a_retarget_rewrites_only_fields_that_may_name_the_type() {
        let mut material = serde_json::json!({"type": "Material", "args": {
            "$id": "m", "shader": "water", "albedo": "water",
        }});
        assert_eq!(
            retarget_references(&mut material, "Shader", "water", Some("sea")),
            1
        );
        assert_eq!(material["args"]["shader"], "sea");
        assert_eq!(material["args"]["albedo"], "water", "a texture slot");
        assert_eq!(retarget_references(&mut material, "Shader", "sea", None), 1);
        assert!(material["args"].get("shader").is_none());
        let mut unknown = serde_json::json!({"type": "Nope", "args": {"shader": "sea"}});
        assert_eq!(retarget_references(&mut unknown, "Shader", "sea", None), 0);
    }

    // A rename follows the structured references too, and only the strings the
    // build reads as references: a label that happens to read the old name
    // stays.
    #[test]
    fn a_rename_follows_structured_references() {
        let mesh = serde_json::json!({"generator": "box"});
        let mut prop = serde_json::json!({"type": "Prop", "args": {
            "$id": "p", "mesh": "box", "material": "box",
        }});
        assert_eq!(
            rename_references(
                &mut prop,
                RegisteredType::ProceduralMesh,
                &mesh,
                "box",
                "crate"
            ),
            1
        );
        assert_eq!(prop["args"]["mesh"], "crate");
        assert_eq!(prop["args"]["material"], "box", "a Material slot");

        let mut behavior = serde_json::json!({"type": "Behavior", "args": {
            "$id": "b",
            "on": {"enter": "gate"},
            "do": [
                {"despawn": {"target": {"named": "gate"}}},
                {"log": {"text": "gate"}},
            ],
        }});
        let volume = serde_json::json!({});
        let moved = rename_references(
            &mut behavior,
            RegisteredType::TriggerVolume,
            &volume,
            "gate",
            "door",
        );
        assert_eq!(moved, 2);
        assert_eq!(behavior["args"]["on"]["enter"], "door");
        assert_eq!(
            behavior["args"]["do"][0]["despawn"]["target"]["named"],
            "door"
        );
        assert_eq!(
            behavior["args"]["do"][1]["log"]["text"], "gate",
            "plain text"
        );
    }
}

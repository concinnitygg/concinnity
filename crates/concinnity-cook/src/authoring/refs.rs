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
use concinnity_core::ecs::RefField;

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
    let Some(ty) = entry_type(entry) else {
        return 0;
    };
    let Some(args) = entry.get_mut("args") else {
        return 0;
    };
    ty.ref_fields()
        .iter()
        .filter(|f| may_name(f, target_type))
        .map(|f| retarget_leaves(args, &f.path, from, to))
        .sum()
}

/// Point every reference in `entry` naming `from` at `to`, for a rename of
/// that asset, a `target` declared with `target_args`: the fields the registry
/// derives and the structured references the build resolves by name (a Prop's
/// or a Prefab entry's `mesh`, a Model's submeshes, a Behavior's nodes and
/// trigger sources, an AnimationGraph's blendspace clips). Returns how many
/// references changed.
pub fn rename_references(
    entry: &mut serde_json::Value,
    target: RegisteredType,
    target_args: &serde_json::Value,
    from: &str,
    to: &str,
) -> usize {
    let mut moved = retarget_references(entry, target.as_str(), from, Some(to));
    let Some(ty) = entry_type(entry) else {
        return moved;
    };
    let Some(args) = entry.get_mut("args") else {
        return moved;
    };
    let pointers = pointers_to(args, from);
    if pointers.is_empty() {
        return moved;
    }
    // A string reading `from` is a reference exactly when the build reads it
    // as one: rewriting it leaves one reference fewer to `from`.
    let mut left = structured_naming(ty, args, target, target_args, from);
    for pointer in pointers {
        if left == 0 {
            break;
        }
        set_pointer(args, &pointer, to);
        let now = structured_naming(ty, args, target, target_args, from);
        if now < left {
            left = now;
            moved += 1;
        } else {
            set_pointer(args, &pointer, from);
        }
    }
    moved
}

/// How many references in `entry` name `name`, a `target` declared with
/// `target_args`: the references [`rename_references`] would move.
pub fn count_references(
    entry: &serde_json::Value,
    target: RegisteredType,
    target_args: &serde_json::Value,
    name: &str,
) -> usize {
    let (Some(ty), Some(args)) = (entry_type(entry), entry.get("args")) else {
        return 0;
    };
    if pointers_to(args, name).is_empty() {
        return 0;
    }
    let typed: usize = ty
        .ref_fields()
        .iter()
        .filter(|f| may_name(f, target.as_str()))
        .map(|f| {
            string_leaves(args, &f.path)
                .iter()
                .filter(|(_, leaf)| *leaf == name)
                .count()
        })
        .sum();
    typed + structured_naming(ty, args, target, target_args, name)
}

fn may_name(field: &RefField, target_type: &str) -> bool {
    field.targets.is_empty() || field.targets.contains(&target_type)
}

fn entry_type(entry: &serde_json::Value) -> Option<RegisteredType> {
    entry
        .get("type")
        .and_then(|t| t.as_str())
        .and_then(RegisteredType::parse)
}

// How many of the structured references in `args`, a `ty` entry's, name
// `name` and may resolve to a `target` declared with `target_args`.
fn structured_naming(
    ty: RegisteredType,
    args: &serde_json::Value,
    target: RegisteredType,
    target_args: &serde_json::Value,
    name: &str,
) -> usize {
    let accepts = |kind: RefKind| match kind {
        RefKind::MeshSource => is_mesh_source(target, target_args),
        RefKind::Scene => target == RegisteredType::Scene,
        RefKind::Animation => target == RegisteredType::Animation,
        RefKind::AudioClip => target == RegisteredType::AudioClip,
        RefKind::Screen => target == RegisteredType::Screen,
        RefKind::TriggerVolume => target == RegisteredType::TriggerVolume,
        RefKind::AnyAsset => true,
    };
    cross_refs_for(ty, "", args)
        .iter()
        .filter(|r| {
            matches!(r, CrossRef::Resolve { kind, target, .. } if target == name && accepts(*kind))
        })
        .count()
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
mod tests;

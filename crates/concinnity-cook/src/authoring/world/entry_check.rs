// The structural rules on one world entry, checked before any schema reads its
// args: no schema rejects an unknown field, so a misspelled `$id` would
// otherwise pass as an anonymous asset.

use concinnity_core::components::GRAPHICS_SECTIONS;
use concinnity_core::ecs::AssetOrigin;
use serde_json::Value;

use super::WORLD_JSONL;
use super::identity::{ID_KEY, check_declared_id, entry_id};
use crate::authoring::registry::RegisteredType;

/// Every structural problem with the entry at `index`: a registered type that
/// is not `RuntimeOnly`, and at most a well-formed `$id` among the `$` keys of its
/// args. Uniqueness is a whole-world rule and not checked here.
pub fn entry_errors(value: &Value, index: usize) -> Vec<String> {
    let label = entry_id(value).map_or_else(|| format!("asset[{index}]"), |id| format!("'{id}'"));
    let mut errors = Vec::new();
    if let Some(args) = value.get("args").and_then(Value::as_object) {
        if let Some(Err(e)) = args.get(ID_KEY).map(check_declared_id) {
            errors.push(format!("{label}: {e}"));
        }
        for key in args
            .keys()
            .filter(|k| k.starts_with('$') && k.as_str() != ID_KEY)
        {
            errors.push(format!(
                "{label}: unknown key `{key}` in `args`; `{ID_KEY}` is the only `$` key"
            ));
        }
    }

    let Some(type_str) = value.get("type").and_then(Value::as_str) else {
        errors.push(format!("{label}: missing `type`"));
        return errors;
    };
    match RegisteredType::parse(type_str).map(|t| t.registration().origin) {
        None => errors.push(match section_key(type_str) {
            Some(key) => format!(
                "{label}: '{type_str}' is a section of GraphicsConfig; write \
                 [\"GraphicsConfig\",{{\"{key}\":{{...}}}}]"
            ),
            None => format!("{label}: unknown type '{type_str}'"),
        }),
        Some(AssetOrigin::RuntimeOnly) => errors.push(format!(
            "{label}: '{type_str}' is RuntimeOnly: it is pushed by a system at runtime \
             and cannot be declared in {WORLD_JSONL}"
        )),
        Some(_) => {}
    }
    errors
}

// The `GraphicsConfig` key of the section type named `type_str`, if it is one.
fn section_key(type_str: &str) -> Option<&'static str> {
    GRAPHICS_SECTIONS
        .iter()
        .find(|(ty, _)| *ty == type_str)
        .map(|(_, key)| *key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_top_level_section_line_points_at_graphics_config() {
        let line = serde_json::json!({"type": "Shadows", "args": {"map_size": 2048}});
        assert_eq!(
            entry_errors(&line, 3),
            [
                r#"asset[3]: 'Shadows' is a section of GraphicsConfig; write ["GraphicsConfig",{"shadows":{...}}]"#
            ]
        );
        let line = serde_json::json!({"type": "Shadowz", "args": {}});
        assert_eq!(entry_errors(&line, 0), ["asset[0]: unknown type 'Shadowz'"]);
        let line = serde_json::json!({"type": "GraphicsConfig", "args": {}});
        assert!(entry_errors(&line, 0).is_empty());
    }

    // The table the error reads names each section by the type its
    // `GraphicsConfig` field documents, and no section type is an asset.
    #[cfg(feature = "schema")]
    #[test]
    fn every_section_is_named_by_its_graphics_config_field() {
        use concinnity_core::components::GraphicsConfig;
        use concinnity_core::ecs::schema::{Body, FieldType, Schema};
        let Body::Fields(fields) = GraphicsConfig::SCHEMA.body else {
            panic!("GraphicsConfig has fields");
        };
        assert_eq!(fields.len(), GRAPHICS_SECTIONS.len());
        for (field, (ty, key)) in fields.iter().zip(GRAPHICS_SECTIONS) {
            assert_eq!(field.key, key);
            let FieldType::Nested(section) = (field.ty)() else {
                panic!("{key} is a nested section");
            };
            assert_eq!(section.name, ty);
            assert!(RegisteredType::parse(ty).is_none(), "{ty}");
        }
    }
}

// The structural rules on one world entry, checked before any schema reads its
// args: no schema rejects an unknown field, so a misplaced or misspelled `$id`
// would otherwise pass as an anonymous asset.

use concinnity_core::ecs::AssetOrigin;
use serde_json::Value;

use super::WORLD_JSONL;
use super::identity::{ID_KEY, check_declared_id, entry_id};
use crate::authoring::registry::RegisteredType;

/// Every structural problem with the entry at `index`: it must be
/// `{"type", "args"}` with an object (or absent) `args`, a registered type
/// that is not RuntimeOnly, and at most a well-formed `$id` among its `$` keys.
/// Uniqueness is a whole-world rule and not checked here.
pub fn entry_errors(value: &Value, index: usize) -> Vec<String> {
    let label = entry_id(value)
        .map(|id| format!("'{id}'"))
        .unwrap_or_else(|| format!("asset[{index}]"));
    let Some(obj) = value.as_object() else {
        return vec![format!("{label}: an entry must be a JSON object")];
    };
    let mut errors = Vec::new();
    for key in obj
        .keys()
        .filter(|k| !matches!(k.as_str(), "type" | "args"))
    {
        errors.push(match key.as_str() {
            "name" => format!(
                "{label}: `name` is not an entry key; declare identity as \
                 `\"{ID_KEY}\"` inside `args`"
            ),
            ID_KEY => format!("{label}: `{ID_KEY}` belongs inside `args`"),
            other => format!("{label}: unknown entry key `{other}`"),
        });
    }

    match obj.get("args") {
        None | Some(Value::Null) => {}
        Some(Value::Object(args)) => {
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
        Some(_) => errors.push(format!("{label}: `args` must be an object")),
    }

    let Some(type_str) = obj.get("type").and_then(|v| v.as_str()) else {
        errors.push(format!("{label}: missing `type` field"));
        return errors;
    };
    match RegisteredType::parse(type_str).map(|t| t.registration().origin) {
        None => errors.push(format!("{label}: unknown type '{type_str}'")),
        Some(AssetOrigin::RuntimeOnly) => errors.push(format!(
            "{label}: '{type_str}' is RuntimeOnly: it is pushed by a system at runtime \
             and cannot be declared in {WORLD_JSONL}"
        )),
        Some(_) => {}
    }
    errors
}

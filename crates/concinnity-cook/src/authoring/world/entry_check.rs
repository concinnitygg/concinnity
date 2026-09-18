// The structural rules on one world entry, checked before any schema reads its
// args: no schema rejects an unknown field, so a misspelled `$id` would
// otherwise pass as an anonymous asset.

use concinnity_core::ecs::AssetOrigin;
use serde_json::Value;

use super::WORLD_JSONL;
use super::identity::{ID_KEY, check_declared_id, entry_id};
use crate::authoring::registry::RegisteredType;

/// Every structural problem with the entry at `index`: a registered type that
/// is not RuntimeOnly, and at most a well-formed `$id` among the `$` keys of its
/// args. Uniqueness is a whole-world rule and not checked here.
pub fn entry_errors(value: &Value, index: usize) -> Vec<String> {
    let label = entry_id(value)
        .map(|id| format!("'{id}'"))
        .unwrap_or_else(|| format!("asset[{index}]"));
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
        None => errors.push(format!("{label}: unknown type '{type_str}'")),
        Some(AssetOrigin::RuntimeOnly) => errors.push(format!(
            "{label}: '{type_str}' is RuntimeOnly: it is pushed by a system at runtime \
             and cannot be declared in {WORLD_JSONL}"
        )),
        Some(_) => {}
    }
    errors
}

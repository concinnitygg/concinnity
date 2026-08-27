// src/editor/live/diff.rs
//
// What changed between the entry list the live preview world was built from
// and the working one. Only an args-only change on entries that kept their
// place, name, and type can be applied to a running world; a line added,
// removed, renamed, retyped, or reordered changes what the expansion produces
// and needs the world rebuilt.
//
// Both sides are compared as EFFECTIVE args -- the authored line merged over
// its template baseline, over the type's defaults -- not as written. An
// authoring form writes back every field it knows, so a line gains keys that
// were only ever holding their default; comparing what the asset actually
// amounts to keeps those out of the change set, and lets a key going away
// register as the move back to the value it uncovers.

use super::ShadowBaselines;
use crate::editor::form;
use serde_json::{Map, Value};

/// One entry whose args changed in place.
pub(crate) struct ArgsChange {
    /// The entry's authored name.
    pub(crate) name: String,
    /// The entry's registry type name.
    pub(crate) ty: String,
    /// The entry's effective args before the edit.
    pub(crate) before: Map<String, Value>,
    /// The entry's effective args after the edit.
    pub(crate) args: Map<String, Value>,
    /// The arg keys whose effective value differs from the pre-edit line.
    pub(crate) keys: Vec<String>,
}

fn field<'a>(entry: &'a Value, key: &str) -> Option<&'a str> {
    entry.get(key).and_then(|v| v.as_str())
}

fn args_of(entry: &Value) -> Map<String, Value> {
    entry
        .get("args")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default()
}

/// Whether the two lists hold the same assets in the same order, so only their
/// args can differ. The cheap half of the comparison, asked first so a
/// structural edit never pays for an expansion it is going to discard.
pub(crate) fn same_assets(before: &[Value], after: &[Value]) -> bool {
    before.len() == after.len()
        && before.iter().zip(after).all(|(old, new)| {
            field(old, "name") == field(new, "name")
                && field(old, "type") == field(new, "type")
                && field(new, "name").is_some()
                && field(new, "type").is_some()
        })
}

/// The args-only changes from `before` to `after`, or `None` when the lists
/// differ structurally. An empty list means the two amount to the same world.
pub(crate) fn args_changes(
    before: &[Value],
    after: &[Value],
    shadows: &ShadowBaselines,
) -> Option<Vec<ArgsChange>> {
    if !same_assets(before, after) {
        return None;
    }
    let mut out = Vec::new();
    for (old, new) in before.iter().zip(after) {
        let (name, ty) = (field(new, "name")?, field(new, "type")?);
        let old_args = effective(ty, name, &args_of(old), shadows);
        let new_args = effective(ty, name, &args_of(new), shadows);
        let keys: Vec<String> = new_args
            .keys()
            .chain(old_args.keys())
            .filter(|k| old_args.get(*k) != new_args.get(*k))
            .cloned()
            .collect::<std::collections::BTreeSet<String>>()
            .into_iter()
            .collect();
        if !keys.is_empty() {
            out.push(ArgsChange {
                name: name.to_string(),
                ty: ty.to_string(),
                before: old_args,
                args: new_args,
                keys,
            });
        }
    }
    Some(out)
}

// What the asset's args amount to: the authored line over the baseline its
// expansion produced (when it patches one), over the type's defaults.
fn effective(
    ty: &str,
    name: &str,
    args: &Map<String, Value>,
    shadows: &ShadowBaselines,
) -> Map<String, Value> {
    let authored = match shadows.get(name) {
        Some(baseline) => crate::world::merge_args(baseline, &Value::Object(args.clone()))
            .as_object()
            .cloned()
            .unwrap_or_else(|| args.clone()),
        None => args.clone(),
    };
    form::working_args(ty, Some(&authored))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry(name: &str, ty: &str, args: Value) -> Value {
        json!({ "name": name, "type": ty, "args": args })
    }

    fn changes(before: &[Value], after: &[Value]) -> Option<Vec<ArgsChange>> {
        args_changes(before, after, &ShadowBaselines::new())
    }

    #[test]
    fn an_unchanged_list_yields_no_changes() {
        let list = vec![entry("a", "Sprite", json!({ "width": 4.0 }))];
        assert!(changes(&list, &list).unwrap().is_empty());
    }

    #[test]
    fn a_changed_value_reports_only_its_key() {
        let before = vec![entry("a", "Sprite", json!({ "width": 4.0, "height": 4.0 }))];
        let after = vec![entry("a", "Sprite", json!({ "width": 8.0, "height": 4.0 }))];
        let changes = changes(&before, &after).unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].name, "a");
        assert_eq!(changes[0].keys, ["width"]);
    }

    // The form writes back every field it knows, so a committed edit spells out
    // keys the line never carried. A key that only ever held its default is not
    // a change, and must not drag the edit into a rebuild.
    #[test]
    fn a_key_written_out_at_its_default_is_not_a_change() {
        let before = vec![entry("a", "Sprite", json!({ "width": 4.0 }))];
        let defaults = form::base_args("Sprite");
        let mut spelled_out = defaults.clone();
        spelled_out.insert("width".to_string(), json!(8.0));
        let after = vec![entry("a", "Sprite", Value::Object(spelled_out))];
        assert_eq!(changes(&before, &after).unwrap()[0].keys, ["width"]);
    }

    // Dropping a key uncovers what it was overriding, which is a change like
    // any other -- the reverting half of the override loop.
    #[test]
    fn a_dropped_key_reports_the_value_it_uncovers() {
        let before = vec![entry("a", "Sprite", json!({ "width": 8.0 }))];
        let after = vec![entry("a", "Sprite", json!({}))];
        let changes = changes(&before, &after).unwrap();
        assert_eq!(changes[0].keys, ["width"]);
        assert_eq!(
            changes[0].args.get("width"),
            form::base_args("Sprite").get("width"),
            "the type default is what it falls back to"
        );
    }

    // A patch line's effective args are its baseline with the patch over it, so
    // an untouched field compares as the template's value, not a type default.
    #[test]
    fn a_patch_line_compares_against_its_baseline() {
        let mut shadows = ShadowBaselines::new();
        shadows.insert("a".to_string(), json!({ "width": 4.0, "height": 12.0 }));
        let before = vec![entry("a", "Sprite", json!({}))];
        let after = vec![entry("a", "Sprite", json!({ "width": 9.0 }))];
        let changed = args_changes(&before, &after, &shadows).unwrap();
        assert_eq!(changed[0].keys, ["width"]);
        assert_eq!(
            changed[0].args.get("height"),
            Some(&json!(12.0)),
            "the baseline fills what the patch leaves out"
        );
    }

    #[test]
    fn structural_edits_report_none() {
        let one = vec![entry("a", "Sprite", json!({}))];
        let two = vec![
            entry("a", "Sprite", json!({})),
            entry("b", "Sprite", json!({})),
        ];
        assert!(changes(&one, &two).is_none(), "a line was added");
        assert!(
            changes(&one, &[entry("z", "Sprite", json!({}))]).is_none(),
            "the line was renamed"
        );
        assert!(
            changes(&one, &[entry("a", "TextLabel", json!({}))]).is_none(),
            "the line changed type"
        );
    }

    // Two lines swapping places is a reorder, not two edits: the expansion
    // reads the list in order, so the rebuild decides it.
    #[test]
    fn a_reorder_reports_none() {
        let before = vec![
            entry("a", "Sprite", json!({})),
            entry("b", "Sprite", json!({})),
        ];
        let after = vec![
            entry("b", "Sprite", json!({})),
            entry("a", "Sprite", json!({})),
        ];
        assert!(changes(&before, &after).is_none());
    }

    // The cheap pre-check agrees with the full comparison about what is
    // structural, and rejects a line missing its identity.
    #[test]
    fn the_pre_check_matches_on_identity_alone() {
        let before = vec![entry("a", "Sprite", json!({ "width": 1.0 }))];
        let after = vec![entry("a", "Sprite", json!({ "width": 2.0 }))];
        assert!(same_assets(&before, &after));
        assert!(!same_assets(&before, &[entry("b", "Sprite", json!({}))]));
        assert!(!same_assets(&before, &[json!({ "type": "Sprite" })]));
    }
}

// src/world/shadow.rs
// Patch-merge shadowing: an authored world.jsonl line with the same name and
// type as a generated or injected asset is a sparse patch over it. The
// authored args win per key; keys the patch omits keep the generated values,
// so an instance tracks its template except where it overrides.

use super::expand::asset_name_str;

/// Deep-merge a sparse authored patch over a template's args.
///
/// Objects merge recursively with the patch winning per key. Everything else
/// (scalars, arrays, explicit null) replaces the template value wholesale: an
/// array override is a full replacement, and an authored `null` overrides a
/// template object (preserving markers like a Camera3D's `controller: null`).
pub fn merge_args(template: &serde_json::Value, patch: &serde_json::Value) -> serde_json::Value {
    match (template, patch) {
        (serde_json::Value::Object(t), serde_json::Value::Object(p)) => {
            let mut out = t.clone();
            for (key, patch_val) in p {
                let merged = match out.get(key) {
                    Some(existing) => merge_args(existing, patch_val),
                    None => patch_val.clone(),
                };
                out.insert(key.clone(), merged);
            }
            serde_json::Value::Object(out)
        }
        _ => patch.clone(),
    }
}

// Apply a template's args under the authored line named `name`: the line's
// args become merge_args(template, authored). Idempotent, since re-merging a
// merged result changes nothing. A missing line is a no-op (the caller
// recorded the shadow from the same list, so it only happens in tests).
pub(crate) fn merge_into_authored(
    assets: &mut [serde_json::Value],
    name: &str,
    template_args: &serde_json::Value,
) {
    let Some(line) = assets.iter_mut().find(|v| asset_name_str(v) == name) else {
        return;
    };
    let authored = line.get("args").cloned().unwrap_or(serde_json::json!({}));
    line["args"] = merge_args(template_args, &authored);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn patch_keys_win_and_omitted_keys_keep_template_values() {
        let template = json!({"a": 1, "b": [1, 2], "c": "x"});
        let patch = json!({"b": [9]});
        assert_eq!(
            merge_args(&template, &patch),
            json!({"a": 1, "b": [9], "c": "x"})
        );
    }

    #[test]
    fn nested_objects_merge_recursively() {
        let template = json!({"controller": {"move_speed": 4.0, "free_fly": false}, "far": 100});
        let patch = json!({"controller": {"move_speed": 9.0}});
        assert_eq!(
            merge_args(&template, &patch),
            json!({"controller": {"move_speed": 9.0, "free_fly": false}, "far": 100})
        );
    }

    #[test]
    fn an_explicit_null_overrides_a_template_object() {
        let template = json!({"controller": {"move_speed": 4.0}});
        let patch = json!({"controller": null});
        assert_eq!(merge_args(&template, &patch), json!({"controller": null}));
    }

    #[test]
    fn arrays_replace_wholesale() {
        let template = json!({"waves": [{"amplitude": 1.0}, {"amplitude": 2.0}]});
        let patch = json!({"waves": [{"amplitude": 5.0}]});
        assert_eq!(
            merge_args(&template, &patch),
            json!({"waves": [{"amplitude": 5.0}]})
        );
    }

    #[test]
    fn patch_may_add_keys_the_template_lacks() {
        let template = json!({"a": 1});
        let patch = json!({"b": {"c": 2}});
        assert_eq!(
            merge_args(&template, &patch),
            json!({"a": 1, "b": {"c": 2}})
        );
    }

    #[test]
    fn merging_a_merged_result_is_idempotent() {
        let template = json!({"a": 1, "b": {"c": 2, "d": 3}});
        let patch = json!({"b": {"c": 9}});
        let once = merge_args(&template, &patch);
        assert_eq!(merge_args(&template, &once), once);
    }

    #[test]
    fn merge_into_authored_patches_the_named_line_only() {
        let mut assets = vec![
            json!({"name": "other", "type": "Prop", "args": {"scale": [2, 2, 2]}}),
            json!({"name": "inst_table", "type": "Prop", "args": {"position": [5, 0, 0]}}),
        ];
        let template = json!({"position": [1, 0, 0], "mesh": "box"});
        merge_into_authored(&mut assets, "inst_table", &template);
        assert_eq!(
            assets[1]["args"],
            json!({"position": [5, 0, 0], "mesh": "box"})
        );
        assert_eq!(assets[0]["args"], json!({"scale": [2, 2, 2]}));
    }

    #[test]
    fn merge_into_authored_tolerates_a_line_without_args() {
        let mut assets = vec![json!({"name": "x", "type": "Font"})];
        merge_into_authored(&mut assets, "x", &json!({"size_px": 48}));
        assert_eq!(assets[0]["args"], json!({"size_px": 48}));
    }
}

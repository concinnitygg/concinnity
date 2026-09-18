//! Reading an args value at a dotted field path, the address the registry's
//! reference and vocabulary tables use.
//!
//! A path names object keys only (`controller.follow.target`); a list met on
//! the way is walked element by element, so one path covers every entry of a
//! `Vec` field and of a `Vec` of nested objects alike.

use serde_json::Value;

/// Every non-empty string at `path` in `args`, each with the location it was
/// found at: the path itself for a plain field, with an index per list walked
/// through (`meshes[1].material`, `rows[0][2]`). Anything that is not a string
/// (an absent key, null, an already-resolved integer id) is skipped.
pub fn string_leaves<'a>(args: &'a Value, path: &str) -> Vec<(String, &'a str)> {
    let mut out = Vec::new();
    let segments: Vec<&str> = path.split('.').collect();
    walk(args, &segments, String::new(), &mut out);
    out
}

fn walk<'a>(value: &'a Value, segments: &[&str], at: String, out: &mut Vec<(String, &'a str)>) {
    match value {
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                walk(item, segments, format!("{at}[{i}]"), out);
            }
        }
        Value::String(s) if segments.is_empty() && !s.is_empty() => out.push((at, s)),
        Value::Object(map) => {
            let Some((first, rest)) = segments.split_first() else {
                return;
            };
            if let Some(next) = map.get(*first) {
                let at = if at.is_empty() {
                    (*first).to_string()
                } else {
                    format!("{at}.{first}")
                };
                walk(next, rest, at, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn leaves(args: Value, path: &str) -> Vec<(String, String)> {
        string_leaves(&args, path)
            .into_iter()
            .map(|(at, s)| (at, s.to_string()))
            .collect()
    }

    fn pair(at: &str, s: &str) -> (String, String) {
        (at.to_string(), s.to_string())
    }

    #[test]
    fn a_top_level_field_reads_its_string() {
        assert_eq!(
            leaves(json!({"screen": "menu"}), "screen"),
            [pair("screen", "menu")]
        );
    }

    #[test]
    fn empty_absent_null_and_resolved_values_are_skipped() {
        for args in [
            json!({"screen": ""}),
            json!({}),
            json!({"screen": null}),
            json!({"screen": 4}),
            json!({"screen": {"nested": "x"}}),
        ] {
            assert!(leaves(args, "screen").is_empty());
        }
    }

    #[test]
    fn a_nested_path_walks_objects() {
        let args = json!({"controller": {"follow": {"target": "hero"}}});
        assert_eq!(
            leaves(args, "controller.follow.target"),
            [pair("controller.follow.target", "hero")]
        );
        assert!(leaves(json!({"controller": null}), "controller.follow.target").is_empty());
    }

    #[test]
    fn lists_are_walked_with_their_indices() {
        assert_eq!(
            leaves(json!({"palette": ["stone", "", "dirt"]}), "palette"),
            [pair("palette[0]", "stone"), pair("palette[2]", "dirt")]
        );
        assert_eq!(
            leaves(
                json!({"meshes": [{"mesh": "a"}, {"mesh": "b", "material": "m"}]}),
                "meshes.material"
            ),
            [pair("meshes[1].material", "m")]
        );
        assert_eq!(
            leaves(json!({"rows": [["a", "b"], ["c"]]}), "rows"),
            [
                pair("rows[0][0]", "a"),
                pair("rows[0][1]", "b"),
                pair("rows[1][0]", "c")
            ]
        );
    }
}

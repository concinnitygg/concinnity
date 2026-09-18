//! The authored world model: world.jsonl I/O (`WorldJsonlAsset`,
//! parse/write/patch_world_jsonl, find_world_jsonl, the path consts), $include
//! resolution, and structural validation (`load_world`). What sits on top of
//! this -- expansion passes, injection, and `prepare_world` -- is
//! `crate::build_only`; the shipped runtime plays compiled blobs and never sees
//! any of this.
mod entry_check;
mod find;
mod identity;
mod io;

pub use entry_check::entry_errors;
pub use find::{WORLD_JSONL, find_world_jsonl};
pub use identity::{
    ID_KEY, anonymous_label, args_with_id, args_without_id, entry_handle, entry_handles, entry_id,
    find_entry, is_label_of, replace_args, set_entry_id, take_entry_id,
};
pub use io::{
    WorldJsonlAsset, known_names, parse_world_jsonl, patch_world_jsonl, patch_world_jsonl_to,
    write_world_jsonl,
};

/// Asset name derived from a file path: the file stem with dots replaced by
/// underscores. Companion injection and `cn add` share this so a generated asset
/// is named exactly as if the user had added the same file.
pub fn asset_name_from_path(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.replace('.', "_"))
        .unwrap_or_else(|| path.to_string())
}

/// Resolve $include directives in a flat asset list.
///
/// An entry of the form `{"$include": "path/to/file"}` is replaced inline by
/// the entries from that file. The included file may be a JSON array or a
/// single JSON object. Includes are resolved relative to cwd. The result is
/// always a flat list with no $include entries remaining.
pub fn resolve_includes(assets: Vec<serde_json::Value>) -> std::io::Result<Vec<serde_json::Value>> {
    let mut out = Vec::with_capacity(assets.len());
    for entry in assets {
        if let Some(path_val) = entry.get("$include") {
            let path = path_val.as_str().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "$include value must be a string path",
                )
            })?;

            let content = std::fs::read_to_string(path).map_err(|e| {
                std::io::Error::new(e.kind(), format!("$include '{}': {}", path, e))
            })?;

            let parsed: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("$include '{}': {}", path, e),
                )
            })?;

            match parsed {
                serde_json::Value::Array(items) => out.extend(items),
                obj @ serde_json::Value::Object(_) => out.push(obj),
                other => {
                    let kind = match &other {
                        serde_json::Value::Null => "null",
                        serde_json::Value::Bool(_) => "bool",
                        serde_json::Value::Number(_) => "number",
                        serde_json::Value::String(_) => "string",
                        _ => "unknown",
                    };
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "$include '{}': expected object or array, got {}",
                            path, kind
                        ),
                    ));
                }
            }
        } else {
            out.push(entry);
        }
    }
    Ok(out)
}

/// Parse a world.jsonl string, resolve $include directives, and run structural
/// validation. On success returns the raw (pre-expansion) asset list; on failure
/// returns every structural error found, not just the first, so an upstream
/// caller (e.g. the infra agentic loop) gets all feedback in a single pass.
///
/// Structural validation covers what must hold before a world can be expanded
/// or built: each entry is `{"type", "args"}` with an object (or absent)
/// `args`, the type is registered, the type is not RuntimeOnly (those are
/// pushed by a system at runtime and cannot be authored), and every `$id`
/// declared is a well-formed string no other entry declares. The `$id` is
/// checked here, before any schema reads the args: no schema rejects an
/// unknown field, so a misspelled key would otherwise pass as an anonymous
/// asset. Semantic validation of the expanded world (cross-references,
/// per-asset args) is a separate stage; see crate::check.
///
/// Every anonymous entry leaves here carrying its `<Type>#<ordinal>` label as
/// its `$id`, so the passes after it address every entry the same way.
pub(crate) fn load_world(content: &str) -> Result<Vec<serde_json::Value>, Vec<String>> {
    let parsed = parse_world_jsonl(content).map_err(|e| vec![format!("syntax error: {e}")])?;
    let mut raw = resolve_includes(parsed).map_err(|e| vec![e.to_string()])?;

    let mut errors: Vec<String> = Vec::new();
    let mut seen_ids: std::collections::HashMap<&str, usize> = Default::default();
    for (i, value) in raw.iter().enumerate() {
        errors.extend(entry_errors(value, i));
        if let Some(id) = entry_id(value) {
            let count = seen_ids.entry(id).or_insert(0);
            *count += 1;
            if *count == 2 {
                errors.push(format!(
                    "duplicate `{ID_KEY}` '{id}': an id names one asset"
                ));
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }
    let handles = entry_handles(&raw);
    for (value, handle) in raw.iter_mut().zip(handles) {
        if let Some(handle) = handle.filter(|_| entry_id(value).is_none()) {
            set_entry_id(value, &handle);
        }
    }
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_world_accepts_valid_world() {
        let content = r#"{"type":"Window","args":{"$id":"a"}}
{"type":"Window","args":{"$id":"b"}}
"#;
        let raw = load_world(content).unwrap();
        assert_eq!(raw.len(), 2);
    }

    #[test]
    fn load_world_collects_all_errors() {
        let content = r#"{"args":{"$id":"a"}}
{"type":"Window","args":{"$id":7}}
{"type":"Nope"}
"#;
        let errs = load_world(content).unwrap_err();
        assert_eq!(errs.len(), 3, "{errs:?}");
        assert!(errs[0].contains("'a'") && errs[0].contains("missing `type`"));
        assert!(errs[1].contains("must be a string"));
        assert!(errs[2].contains("asset[2]") && errs[2].contains("unknown type"));
    }

    // An anonymous entry is the common case: it loads carrying its label as its
    // `$id`, counted among the anonymous entries of its type only.
    #[test]
    fn load_world_labels_every_anonymous_entry() {
        let content = r#"{"type":"Prop","args":{"mesh":"m"}}
{"type":"Prop","args":{"$id":"hero"}}
{"type":"Window"}
{"type":"Prop","args":null}
"#;
        let raw = load_world(content).unwrap();
        let ids: Vec<Option<&str>> = raw.iter().map(entry_id).collect();
        assert_eq!(
            ids,
            [
                Some("Prop#0"),
                Some("hero"),
                Some("Window#0"),
                Some("Prop#1")
            ]
        );
        assert_eq!(raw[0]["args"]["mesh"], "m");
    }

    #[test]
    fn load_world_rejects_duplicate_ids() {
        let content = r#"{"type":"Window","args":{"$id":"a"}}
{"type":"Scene","args":{"$id":"a"}}
"#;
        let errs = load_world(content).unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("duplicate `$id` 'a'")),
            "{errs:?}"
        );
    }

    // The old top-level `name` is refused with the move spelled out, as is a
    // `$id` placed beside `args` rather than in it.
    #[test]
    fn load_world_rejects_identity_outside_args() {
        let content = r#"{"name":"a","type":"Window"}
{"$id":"b","type":"Window"}
{"type":"Window","label":"c"}
"#;
        let errs = load_world(content).unwrap_err();
        assert_eq!(errs.len(), 3, "{errs:?}");
        assert!(
            errs[0].contains("`name` is not an entry key") && errs[0].contains("inside `args`")
        );
        assert!(errs[1].contains("`$id` belongs inside `args`"));
        assert!(errs[2].contains("unknown entry key `label`"));
    }

    // No schema rejects an unknown field, so a misspelled `$id` is caught here
    // or it would pass as an anonymous asset.
    #[test]
    fn load_world_rejects_a_misspelled_id_key() {
        let errs = load_world(r#"{"type":"Window","args":{"$ID":"w"}}"#).unwrap_err();
        assert!(errs[0].contains("unknown key `$ID`"), "{errs:?}");
    }

    #[test]
    fn load_world_rejects_a_label_shaped_or_empty_id() {
        let errs = load_world(r#"{"type":"Prop","args":{"$id":"Prop#0"}}"#).unwrap_err();
        assert!(errs[0].contains("reserved"), "{errs:?}");
        let errs = load_world(r#"{"type":"Prop","args":{"$id":""}}"#).unwrap_err();
        assert!(errs[0].contains("must not be empty"), "{errs:?}");
    }

    #[test]
    fn load_world_rejects_non_object_args() {
        let errs = load_world(r#"{"type":"Window","args":[]}"#).unwrap_err();
        assert!(errs[0].contains("`args` must be an object"), "{errs:?}");
    }

    #[test]
    fn load_world_rejects_runtime_only_type() {
        // Transform is pushed by a system at runtime (RuntimeOnly), so it may
        // not be authored in the world file.
        let content = r#"{"type":"Transform","args":{"$id":"t"}}"#;
        let errs = load_world(content).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("RuntimeOnly")));
    }

    #[test]
    fn load_world_rejects_unknown_type() {
        let content = r#"{"type":"NotARealType","args":{"$id":"x"}}"#;
        let errs = load_world(content).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("unknown type")));
    }

    // asset_name_from_path

    #[test]
    fn asset_name_from_path_replaces_dots_in_stem() {
        assert_eq!(asset_name_from_path("/a/b/hero.model.glb"), "hero_model");
        assert_eq!(asset_name_from_path("hero.png"), "hero");
    }

    #[test]
    fn asset_name_from_path_falls_back_to_the_raw_path() {
        // A path with no file stem (e.g. "..") falls back to the input.
        assert_eq!(asset_name_from_path(".."), "..");
    }

    // resolve_includes

    fn write_temp(dir: &tempfile::TempDir, name: &str, contents: &str) -> String {
        let path = dir.path().join(name);
        std::fs::write(&path, contents).unwrap();
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn resolve_includes_passes_through_non_include_entries() {
        let entries = vec![serde_json::json!({"type": "Window", "args": {"$id": "a"}})];
        let out = resolve_includes(entries).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["args"]["$id"], "a");
    }

    #[test]
    fn resolve_includes_inlines_an_array_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(
            &dir,
            "chunk.json",
            r#"[{"type":"Window","args":{"$id":"a"}},{"type":"Window","args":{"$id":"b"}}]"#,
        );
        let entries = vec![
            serde_json::json!({"$include": path}),
            serde_json::json!({"type": "Window", "args": {"$id": "c"}}),
        ];
        let out = resolve_includes(entries).unwrap();
        let names: Vec<&str> = out
            .iter()
            .filter_map(|v| v["args"]["$id"].as_str())
            .collect();
        assert_eq!(names, ["a", "b", "c"]);
    }

    #[test]
    fn resolve_includes_inlines_a_single_object_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(
            &dir,
            "one.json",
            r#"{"type":"Window","args":{"$id":"solo"}}"#,
        );
        let out = resolve_includes(vec![serde_json::json!({"$include": path})]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["args"]["$id"], "solo");
    }

    #[test]
    fn resolve_includes_rejects_non_string_path() {
        let err =
            resolve_includes(vec![serde_json::json!({"$include": 42})]).expect_err("non-string");
        assert!(err.to_string().contains("must be a string path"));
    }

    #[test]
    fn resolve_includes_reports_a_read_error_with_the_path() {
        let err = resolve_includes(vec![
            serde_json::json!({"$include": "/no/such/include.json"}),
        ])
        .expect_err("missing file");
        assert!(err.to_string().contains("/no/such/include.json"));
    }

    #[test]
    fn resolve_includes_reports_bad_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "broken.json", "{ not valid json");
        let err =
            resolve_includes(vec![serde_json::json!({"$include": path})]).expect_err("bad json");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn resolve_includes_rejects_a_non_object_non_array_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "scalar.json", "42");
        let err = resolve_includes(vec![serde_json::json!({"$include": path})])
            .expect_err("kind mismatch");
        assert!(err.to_string().contains("expected object or array"));
        assert!(err.to_string().contains("number"));
    }
}

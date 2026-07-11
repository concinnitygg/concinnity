// The build-time world surface: world.jsonl I/O (`WorldJsonlAsset`,
// parse/write/patch_world_jsonl, find_world_jsonl, the path consts), preset
// loading, and the build front-half (structural load, the expansion passes, and
// prepare_world = load + expand + validate). All authoring input handling; the
// shipped runtime plays compiled blobs and never sees any of this.
mod find;
mod io;
pub mod preset;

pub use find::{WORLD_JSONL, find_world_jsonl};
pub use io::{
    WorldJsonlAsset, known_names, parse_world_jsonl, patch_world_jsonl, patch_world_jsonl_to,
    write_world_jsonl,
};

pub(crate) mod application;
pub(crate) mod camera_shot;
pub(crate) mod companion;
pub(crate) mod companion_specs;
pub(crate) mod config;
pub(crate) mod defaults;

pub(crate) mod light_rig;
pub(crate) mod main_menu;
pub(crate) mod material_palette;
pub(crate) mod option_select;
pub(crate) mod panel;
pub(crate) mod prefab;
pub(crate) mod room;
pub(crate) mod scene_import;
pub(crate) mod shader;
pub(crate) mod slider;
pub(crate) mod story;
pub(crate) mod ui_spec;

pub(crate) mod expand;

#[allow(unused_imports)]
pub use config::{DEFAULT_MAX_BLOB_BYTES, WorldConfig};
pub use expand::{ExpandReport, InjectedAsset, expand_world, expand_world_from_str};
pub use shader::normalize_single_shader_type;

use crate::ecs::AssetOrigin;
use crate::registry::ComponentType;

// Asset name derived from a file path: the file stem with dots replaced by
// underscores. Companion injection and `cn add` share this so a generated asset
// is named exactly as if the user had added the same file.
pub fn asset_name_from_path(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.replace('.', "_"))
        .unwrap_or_else(|| path.to_string())
}

// A world.jsonl that has been loaded, structurally validated, expanded, and
// semantically checked: everything the compile stage needs, computed once.
pub struct LoadedWorld {
    // The same assets as typed entries, consumed by the build pipeline.
    pub assets: Vec<WorldJsonlAsset>,
    // Assets added by the injection passes (companions, engine defaults),
    // recorded in world-lock.json so the user can see and override them.
    pub injected: Vec<InjectedAsset>,
    // Names declared in the world file itself (pre-expansion), for
    // provenance listings.
    pub authored: Vec<String>,
}

// Resolve $include directives in a flat asset list.
//
// An entry of the form `{"$include": "path/to/file"}` is replaced inline by
// the entries from that file. The included file may be a JSON array or a
// single JSON object. Includes are resolved relative to cwd. The result is
// always a flat list with no $include entries remaining.
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

// Parse a world.jsonl string, resolve $include directives, and run structural
// validation. On success returns the raw (pre-expansion) asset list; on failure
// returns every structural error found, not just the first, so an upstream
// caller (e.g. the infra agentic loop) gets all feedback in a single pass.
//
// Structural validation covers what must hold before a world can be expanded
// or built: each entry has a string `name` and `type`, the type is registered,
// the type is not RuntimeOnly (those are pushed by a system at runtime and
// cannot be authored), and names are unique. Semantic validation of the
// expanded world (cross-references, per-asset args) is a separate stage; see
// crate::check.
pub fn load_world(content: &str) -> Result<Vec<serde_json::Value>, Vec<String>> {
    let parsed = parse_world_jsonl(content).map_err(|e| vec![format!("syntax error: {e}")])?;
    let raw = resolve_includes(parsed).map_err(|e| vec![e.to_string()])?;

    let mut errors: Vec<String> = Vec::new();
    let mut seen_names: std::collections::HashMap<&str, usize> = Default::default();

    for (i, value) in raw.iter().enumerate() {
        let name = value.get("name").and_then(|v| v.as_str());
        let type_str = value.get("type").and_then(|v| v.as_str());

        let label = name
            .map(|n| format!("'{}'", n))
            .unwrap_or_else(|| format!("asset[{}]", i));

        if name.is_none() {
            errors.push(format!("{}: missing `name` field", label));
        }

        let Some(type_str) = type_str else {
            errors.push(format!("{}: missing `type` field", label));
            continue;
        };

        let origin = if let Some(ct) = ComponentType::parse(type_str) {
            Some(ct.registration().origin)
        } else {
            errors.push(format!("{}: unknown type '{}'", label, type_str));
            None
        };

        if matches!(origin, Some(AssetOrigin::RuntimeOnly)) {
            errors.push(format!(
                "{}: '{}' is RuntimeOnly: it is pushed by a system at runtime \
                 and cannot be declared in {}",
                label, type_str, WORLD_JSONL
            ));
        }

        if let Some(n) = name {
            let count = seen_names.entry(n).or_insert(0);
            *count += 1;
            if *count == 2 {
                errors.push(format!(
                    "duplicate name '{}': asset names must be unique",
                    n
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(raw)
    } else {
        Err(errors)
    }
}

// Run the read-only front half of the build pipeline: parse and structurally
// validate the world (`load_world`), expand all build-time assets, then run
// semantic validation (`crate::check::check_world`). Returns everything the
// compile stage needs, computed exactly once. Errors from every stage are
// collected, so the caller gets the full picture in a single pass.
pub fn prepare_world(content: &str) -> Result<LoadedWorld, Vec<String>> {
    let mut expanded = load_world(content)?;
    let authored: Vec<String> = expanded
        .iter()
        .filter_map(|v| v.get("name").and_then(|n| n.as_str()))
        .map(str::to_string)
        .collect();
    let report = expand_world(&mut expanded).map_err(|e| vec![e])?;

    let assets: Vec<WorldJsonlAsset> = expanded.iter().map(WorldJsonlAsset::from_value).collect();

    crate::check::check_world(&assets)?;

    Ok(LoadedWorld {
        assets,
        injected: report.injected,
        authored,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_world_accepts_valid_world() {
        let content = r#"{"name":"a","type":"Window"}
{"name":"b","type":"Window"}
"#;
        let raw = load_world(content).unwrap();
        assert_eq!(raw.len(), 2);
    }

    #[test]
    fn load_world_collects_all_errors() {
        let content = r#"{"name":"a"}
{"type":"Window"}
"#;
        let errs = load_world(content).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("missing `type`")));
        assert!(errs.iter().any(|e| e.contains("missing `name`")));
    }

    #[test]
    fn load_world_rejects_duplicate_names() {
        let content = r#"{"name":"a","type":"Window"}
{"name":"a","type":"Window"}
"#;
        let errs = load_world(content).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("duplicate name")));
    }

    #[test]
    fn prepare_world_expands_and_validates() {
        let content = r#"{"name":"gfx","type":"GraphicsConfig","args":{}}"#;
        let loaded = prepare_world(content).unwrap();
        // GraphicsConfig pulls in its companions, so the prepared world holds
        // more than the single declared asset.
        assert!(loaded.assets.len() > 1);
        assert!(
            loaded
                .assets
                .iter()
                .any(|a| a.asset_type == "GraphicsConfig")
        );
    }

    #[test]
    fn load_world_rejects_runtime_only_type() {
        // Transform is pushed by a system at runtime (RuntimeOnly), so it may
        // not be authored in the world file.
        let content = r#"{"name":"t","type":"Transform"}"#;
        let errs = load_world(content).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("RuntimeOnly")));
    }

    #[test]
    fn load_world_rejects_unknown_type() {
        let content = r#"{"name":"x","type":"NotARealType"}"#;
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
        let entries = vec![serde_json::json!({"name": "a", "type": "Window"})];
        let out = resolve_includes(entries).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["name"], "a");
    }

    #[test]
    fn resolve_includes_inlines_an_array_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(
            &dir,
            "chunk.json",
            r#"[{"name":"a","type":"Window"},{"name":"b","type":"Window"}]"#,
        );
        let entries = vec![
            serde_json::json!({"$include": path}),
            serde_json::json!({"name": "c", "type": "Window"}),
        ];
        let out = resolve_includes(entries).unwrap();
        let names: Vec<&str> = out.iter().filter_map(|v| v["name"].as_str()).collect();
        assert_eq!(names, ["a", "b", "c"]);
    }

    #[test]
    fn resolve_includes_inlines_a_single_object_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp(&dir, "one.json", r#"{"name":"solo","type":"Window"}"#);
        let out = resolve_includes(vec![serde_json::json!({"$include": path})]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["name"], "solo");
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

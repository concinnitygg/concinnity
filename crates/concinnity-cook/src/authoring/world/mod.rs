//! The authored world model: world.jsonl I/O (`WorldJsonlAsset`, the
//! `["Type", {args}]` line format, parse/write/patch_world_jsonl,
//! find_world_jsonl, the path consts) and structural validation
//! (`load_world`), which resolves `Include` lines first. What sits on top of
//! this -- expansion passes, injection, and `prepare_world` -- is
//! `crate::build_only`; the shipped runtime plays compiled blobs and never sees
//! any of this.
mod entry_check;
mod find;
mod identity;
mod io;
mod line;
mod source;

pub use entry_check::entry_errors;
pub use find::{WORLD_JSONL, find_world_jsonl};
pub use identity::{
    ID_KEY, anonymous_label, args_with_id, args_without_id, entry_handle, entry_handles, entry_id,
    find_entry, is_label_of, replace_args, set_entry_id, take_entry_id,
};
pub use io::{WorldJsonlAsset, known_names, patch_world_jsonl, patch_world_jsonl_to};
pub use line::{
    LineError, ParseError, entry_from_line, entry_line, parse_entry, parse_world_jsonl,
    write_world_jsonl,
};
pub use source::WorldSource;

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

/// Parse world text, resolve its `Include` lines, and run structural
/// validation. On success returns the raw (pre-expansion) asset list; on failure
/// returns every structural error found, not just the first, so an upstream
/// caller (e.g. the infra agentic loop) gets all feedback in a single pass.
///
/// Structural validation covers what must hold before a world can be expanded
/// or built: each line is a `["Type", {args}]` entry, the type is registered,
/// the type is not RuntimeOnly (those are pushed by a system at runtime and
/// cannot be authored), and every `$id` declared is a well-formed string no other entry declares. The `$id` is
/// checked here, before any schema reads the args: no schema rejects an
/// unknown field, so a misspelled key would otherwise pass as an anonymous
/// asset. Semantic validation of the expanded world (cross-references,
/// per-asset args) is a separate stage; see crate::check.
///
/// Every anonymous entry leaves here carrying its `<Type>#<ordinal>` label as
/// its `$id`, so the passes after it address every entry the same way.
pub fn load_world(source: WorldSource<'_>) -> Result<Vec<serde_json::Value>, Vec<String>> {
    let parsed = parse_world_jsonl(source.text).map_err(|e| {
        e.0.iter()
            .map(|l| format!("syntax error: {l}"))
            .collect::<Vec<_>>()
    })?;
    let mut raw =
        crate::build_only::include::resolve_includes(parsed, source.file).map_err(|e| vec![e])?;

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

    fn load(text: &str) -> Result<Vec<serde_json::Value>, Vec<String>> {
        load_world(text.into())
    }

    #[test]
    fn load_world_accepts_valid_world() {
        let raw = load("[\"Window\",{\"$id\":\"a\"}]\n[\"Window\",{\"$id\":\"b\"}]\n").unwrap();
        assert_eq!(raw.len(), 2);
    }

    #[test]
    fn load_world_reports_every_malformed_line() {
        let errs = load("{\"type\":\"Window\"}\n[\"Scene\"]\n[3]\n").unwrap_err();
        assert_eq!(errs.len(), 2, "{errs:?}");
        assert!(errs[0].starts_with("syntax error: line 1: "), "{errs:?}");
        assert!(errs[1].starts_with("syntax error: line 3: "), "{errs:?}");
    }

    #[test]
    fn load_world_collects_all_errors() {
        let errs = load("[\"Window\",{\"$id\":7}]\n[\"Nope\"]\n").unwrap_err();
        assert_eq!(errs.len(), 2, "{errs:?}");
        assert!(errs[0].contains("must be a string"));
        assert!(errs[1].contains("asset[1]") && errs[1].contains("unknown type"));
    }

    // An anonymous entry is the common case: it loads carrying its label as its
    // `$id`, counted among the anonymous entries of its type only.
    #[test]
    fn load_world_labels_every_anonymous_entry() {
        let raw = load(
            "[\"Prop\",{\"mesh\":\"m\"}]\n[\"Prop\",{\"$id\":\"hero\"}]\n[\"Window\"]\n[\"Prop\",{}]\n",
        )
        .unwrap();
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

    // Included entries are labeled where they land, so an anonymous entry after
    // an include counts the ones the include brought in.
    #[test]
    fn load_world_labels_after_resolving_includes() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("world.jsonl");
        std::fs::write(dir.path().join("props.jsonl"), "[\"Prop\"]\n[\"Prop\"]\n").unwrap();
        let text = "[\"Prop\",{\"mesh\":\"first\"}]\n[\"Include\",{\"path\":\"props.jsonl\"}]\n[\"Prop\",{\"mesh\":\"last\"}]\n";
        let raw = load_world(WorldSource::file(text, &root)).unwrap();
        let ids: Vec<Option<&str>> = raw.iter().map(entry_id).collect();
        assert_eq!(
            ids,
            [
                Some("Prop#0"),
                Some("Prop#1"),
                Some("Prop#2"),
                Some("Prop#3")
            ]
        );
        assert_eq!(raw[3]["args"]["mesh"], "last");
    }

    #[test]
    fn load_world_reports_an_include_failure() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("world.jsonl");
        let errs = load_world(WorldSource::file(
            "[\"Include\",{\"path\":\"missing.jsonl\"}]\n",
            &root,
        ))
        .unwrap_err();
        assert!(errs[0].contains("missing.jsonl"), "{errs:?}");
    }

    #[test]
    fn load_world_rejects_duplicate_ids() {
        let errs = load("[\"Window\",{\"$id\":\"a\"}]\n[\"Scene\",{\"$id\":\"a\"}]\n").unwrap_err();
        assert!(
            errs.iter().any(|e| e.contains("duplicate `$id` 'a'")),
            "{errs:?}"
        );
    }

    // No schema rejects an unknown field, so a misspelled `$id` is caught here
    // or it would pass as an anonymous asset.
    #[test]
    fn load_world_rejects_a_misspelled_id_key() {
        let errs = load(r#"["Window",{"$ID":"w"}]"#).unwrap_err();
        assert!(errs[0].contains("unknown key `$ID`"), "{errs:?}");
    }

    #[test]
    fn load_world_rejects_a_label_shaped_or_empty_id() {
        let errs = load(r#"["Prop",{"$id":"Prop#0"}]"#).unwrap_err();
        assert!(errs[0].contains("reserved"), "{errs:?}");
        let errs = load(r#"["Prop",{"$id":""}]"#).unwrap_err();
        assert!(errs[0].contains("must not be empty"), "{errs:?}");
    }

    #[test]
    fn load_world_rejects_runtime_only_type() {
        // Transform is pushed by a system at runtime (RuntimeOnly), so it may
        // not be authored in the world file.
        let errs = load(r#"["Transform",{"$id":"t"}]"#).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("RuntimeOnly")));
    }

    #[test]
    fn load_world_rejects_unknown_type() {
        let errs = load(r#"["NotARealType",{"$id":"x"}]"#).unwrap_err();
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
}

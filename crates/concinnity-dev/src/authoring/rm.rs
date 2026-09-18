//! Remove an asset from a world JSONL by its handle and rebuild.

use std::path::Path;

use concinnity_cook::authoring::world::{
    WORLD_JSONL, find_entry, known_names, patch_world_jsonl_to,
};
use concinnity_cook::build_only::include::with_includes;

/// Remove the asset `name` addresses from `world_path` and rebuild: the entry
/// declaring it as its `$id`, or the anonymous entry it labels (`Prop#3`).
///
/// Errors if `name` is not present. When it isn't, the error message includes
/// the known handles from the world so the caller can suggest a fix. An entry
/// that an `Include` brings in is refused: its line is in the included file.
pub(crate) fn rm_at_path(world_path: &str, name: &str) -> std::io::Result<()> {
    let mut removed = false;

    patch_world_jsonl_to(world_path, world_path, |assets| {
        let found = line_index(assets, Path::new(world_path), name)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        if let Some(i) = found {
            let asset = assets.remove(i);
            tracing::info!(
                "Removed '{}' (type: {})",
                name,
                asset.get("type").and_then(|v| v.as_str()).unwrap_or("?"),
            );
            removed = true;
        }
        Ok(())
    })?;

    if !removed {
        let known = known_names(world_path).unwrap_or_default();
        if known.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "no asset named '{}' in {} (no assets declared)",
                    name, WORLD_JSONL
                ),
            ));
        }
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "no asset named '{}' in {}\nKnown names: {}",
                name,
                WORLD_JSONL,
                known.join(", ")
            ),
        ));
    }

    super::build_world_file(world_path)
}

// The position among `lines` (the entries of `world_file` itself) of the entry
// `name` addresses. Handles are matched over the world with its includes
// resolved, so a label counts the anonymous entries an include brings in, as
// the build does. `None` when nothing is named so; an error when the entry is
// declared in an included file rather than this one.
fn line_index(
    lines: &[serde_json::Value],
    world_file: &Path,
    name: &str,
) -> Result<Option<usize>, String> {
    let sourced = with_includes(lines.to_vec(), Some(world_file))?;
    let entries: Vec<serde_json::Value> = sourced.iter().map(|s| s.entry.clone()).collect();
    let Some(i) = find_entry(&entries, name) else {
        return Ok(None);
    };
    if let Some(file) = &sourced[i].file {
        return Err(format!(
            "'{name}' is declared in {}, which {} includes; remove it there",
            file.display(),
            world_file.display()
        ));
    }
    Ok(Some(
        sourced[..i].iter().filter(|s| s.file.is_none()).count(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Removal failures surface before any rebuild runs, so these tests never
    // touch the compile pipeline.

    #[test]
    fn rm_of_a_missing_world_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.jsonl");
        assert!(rm_at_path(path.to_str().unwrap(), "anything").is_err());
    }

    #[test]
    fn rm_of_an_unknown_name_lists_the_known_names() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world.jsonl");
        std::fs::write(
            &path,
            concat!(
                "[\"Logger\",{\"$id\":\"log\"}]\n",
                "[\"Logger\",{\"$id\":\"log2\"}]\n",
            ),
        )
        .unwrap();

        let err = rm_at_path(path.to_str().unwrap(), "ghost").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        let msg = err.to_string();
        assert!(msg.contains("no asset named 'ghost'"), "got: {msg}");
        assert!(msg.contains("Known names: log, log2"), "got: {msg}");
        // The world file itself is left intact.
        let survived = std::fs::read_to_string(&path).unwrap();
        assert!(survived.contains("\"log\""));
        assert!(survived.contains("\"log2\""));
    }

    #[test]
    fn rm_from_an_empty_world_reports_no_assets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world.jsonl");
        std::fs::write(&path, "").unwrap();

        let err = rm_at_path(path.to_str().unwrap(), "ghost").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("no assets declared"), "got: {err}");
    }

    // A label counts the anonymous entries an include brings in, so it names
    // the line the build would, and an included entry is refused by name.
    #[test]
    fn a_label_is_counted_across_includes() {
        let dir = tempfile::tempdir().unwrap();
        let world = dir.path().join("world.jsonl");
        std::fs::write(
            dir.path().join("props.jsonl"),
            "[\"Prop\",{\"mesh\":\"inc\"}]\n",
        )
        .unwrap();
        let lines = vec![
            serde_json::json!({"type": "Prop", "args": {"mesh": "a"}}),
            serde_json::json!({"type": "Include", "args": {"path": "props.jsonl"}}),
            serde_json::json!({"type": "Prop", "args": {"mesh": "b"}}),
        ];
        assert_eq!(line_index(&lines, &world, "Prop#0"), Ok(Some(0)));
        assert_eq!(line_index(&lines, &world, "Prop#2"), Ok(Some(2)));
        assert_eq!(line_index(&lines, &world, "Include#0"), Ok(Some(1)));
        assert_eq!(line_index(&lines, &world, "Prop#3"), Ok(None));
        let err = line_index(&lines, &world, "Prop#1").unwrap_err();
        assert!(
            err.contains("props.jsonl") && err.contains("remove it there"),
            "{err}"
        );
    }
}

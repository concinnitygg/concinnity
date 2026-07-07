// src/app/rm.rs
// Remove an asset from a world JSONL by its unique `name` field and rebuild.

use crate::world::{WORLD_JSONL, known_names, patch_world_jsonl};
use concinnity_cook::build_from_path;

// Remove the asset named `name` from `world_path` and rebuild.
//
// Errors if `name` is not present. When it isn't, the error message includes
// the known asset names from the world so the caller can suggest a fix.
pub fn rm_at_path(world_path: &str, name: &str) -> std::io::Result<()> {
    let mut removed = false;

    patch_world_jsonl(world_path, |assets| {
        if let Some(i) = assets
            .iter()
            .position(|a| a.get("name").and_then(|v| v.as_str()) == Some(name))
        {
            let asset = assets.remove(i);
            tracing::info!(
                "Removed '{}' (type: {})",
                name,
                asset.get("type").and_then(|v| v.as_str()).unwrap_or("?"),
            );
            removed = true;
        }
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

    build_from_path(world_path)
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
                "{\"name\":\"log\",\"type\":\"Logger\",\"args\":{}}\n",
                "{\"name\":\"log2\",\"type\":\"Logger\",\"args\":{}}\n",
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
}

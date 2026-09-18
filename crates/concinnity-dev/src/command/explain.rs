//! Print one asset's effective entry from the expanded world: the full JSONL
//! line as the build sees it, pasteable into world.jsonl verbatim. This is the
//! override path for injected defaults and expanded assets, which have no line
//! in the authored file to copy from. An anonymous asset is named by its
//! `<Type>#<ordinal>` label, and its line declares no `$id`.

use concinnity_cook::authoring::world::entry_line;

use crate::command::{provenance, resolve_world_path};

/// Print one asset's effective entry from the expanded world, with where
/// each value came from.
pub fn explain(name: &str, json_path: Option<&str>) -> std::io::Result<()> {
    let json_path = resolve_world_path(json_path)?;
    let content = std::fs::read_to_string(&json_path)?;
    let source = concinnity_cook::WorldSource::file(&content, std::path::Path::new(&json_path));

    let loaded = concinnity_cook::prepare_world(source, crate::project::assets_dir().as_deref())
        .map_err(|errs| crate::authoring::report_validation_errors(&errs))?;

    let Some(asset) = loaded.assets.iter().find(|a| a.id == name) else {
        let mut close: Vec<&str> = loaded
            .assets
            .iter()
            .map(|a| a.id.as_str())
            .filter(|n| n.contains(name))
            .take(5)
            .collect();
        close.sort_unstable();
        let hint = if close.is_empty() {
            String::new()
        } else {
            format!("; close matches: {}", close.join(", "))
        };
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no asset named '{}' in the expanded world{}", name, hint),
        ));
    };

    let line = entry_line(&asset.to_entry())?;

    println!("// {}", provenance(&loaded, &asset.id));
    println!("{line}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_world(content: &str) -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world.jsonl");
        std::fs::write(&path, content).unwrap();
        (dir, path.to_string_lossy().into_owned())
    }

    #[test]
    fn explain_prints_a_known_asset() {
        let (_dir, path) = write_world("[\"GraphicsConfig\",{\"$id\":\"gfx\"}]\n");
        explain("gfx", Some(&path)).unwrap();
    }

    #[test]
    fn explain_resolves_an_anonymous_label() {
        let (_dir, path) = write_world("[\"GraphicsConfig\",{}]\n[\"Scene\"]\n");
        explain("Scene#0", Some(&path)).unwrap();
        assert!(explain("Scene#1", Some(&path)).is_err());
    }

    #[test]
    fn explain_of_an_unknown_name_offers_close_matches() {
        let (_dir, path) = write_world("[\"GraphicsConfig\",{\"$id\":\"gfx\"}]\n");
        let err = explain("gf", Some(&path)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let msg = err.to_string();
        assert!(msg.contains("close matches"), "got: {msg}");
        assert!(msg.contains("gfx"), "got: {msg}");
    }

    #[test]
    fn explain_of_an_unknown_name_without_matches_has_no_hint() {
        let (_dir, path) = write_world("[\"GraphicsConfig\",{\"$id\":\"gfx\"}]\n");
        let err = explain("zzz_nothing", Some(&path)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(!err.to_string().contains("close matches"), "got: {err}");
    }

    #[test]
    fn explain_surfaces_validation_failures() {
        let (_dir, path) = write_world("[\"NotARealAssetType\",{\"$id\":\"odd\"}]\n");
        assert!(explain("odd", Some(&path)).is_err());
    }
}

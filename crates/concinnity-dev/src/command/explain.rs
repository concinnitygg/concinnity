// src/cli/explain.rs
// Print one asset's effective entry from the expanded world: the full JSONL
// line as the build sees it, pasteable into world.jsonl verbatim. This is the
// override path for injected defaults and expanded assets, which have no line
// in the authored file to copy from.

use crate::command::{provenance, resolve_world_path};

/// Print one asset's effective entry from the expanded world, with where
/// each value came from.
pub fn explain(name: &str, json_path: Option<&str>) -> std::io::Result<()> {
    let json_path = resolve_world_path(json_path)?;
    let content = std::fs::read_to_string(&json_path)?;

    let loaded =
        concinnity_cook::prepare_world(&content, concinnity_cook::paths::assets_dir().as_deref())
            .map_err(|errs| concinnity_cook::check::report_validation_errors(&errs))?;

    let Some(asset) = loaded.assets.iter().find(|a| a.name == name) else {
        let mut close: Vec<&str> = loaded
            .assets
            .iter()
            .map(|a| a.name.as_str())
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

    let line = serde_json::json!({
        "name": asset.name,
        "type": asset.asset_type,
        "args": asset.args,
    });

    println!("// {}", provenance(&loaded, &asset.name));
    println!("{}", serde_json::to_string(&line)?);
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
        let (_dir, path) =
            write_world("{\"name\":\"gfx\",\"type\":\"GraphicsConfig\",\"args\":{}}\n");
        explain("gfx", Some(&path)).unwrap();
    }

    #[test]
    fn explain_of_an_unknown_name_offers_close_matches() {
        let (_dir, path) =
            write_world("{\"name\":\"gfx\",\"type\":\"GraphicsConfig\",\"args\":{}}\n");
        let err = explain("gf", Some(&path)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let msg = err.to_string();
        assert!(msg.contains("close matches"), "got: {msg}");
        assert!(msg.contains("gfx"), "got: {msg}");
    }

    #[test]
    fn explain_of_an_unknown_name_without_matches_has_no_hint() {
        let (_dir, path) =
            write_world("{\"name\":\"gfx\",\"type\":\"GraphicsConfig\",\"args\":{}}\n");
        let err = explain("zzz_nothing", Some(&path)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(!err.to_string().contains("close matches"), "got: {err}");
    }

    #[test]
    fn explain_surfaces_validation_failures() {
        let (_dir, path) =
            write_world("{\"name\":\"odd\",\"type\":\"NotARealAssetType\",\"args\":{}}\n");
        assert!(explain("odd", Some(&path)).is_err());
    }
}

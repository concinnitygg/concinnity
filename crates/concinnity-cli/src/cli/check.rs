// src/cli/check.rs: discovery wrapper around concinnity_editor::check_at_path
//
// `cn test` accepts an optional --file path. When the path is missing or
// doesn't exist on disk, fall back to discovery via find_world_jsonl.

use concinnity_cook::world::find_world_jsonl;
use concinnity_editor::check_at_path;

pub(crate) fn check(json_path: &str) -> std::io::Result<()> {
    let resolved;
    let json_path = if !std::path::Path::new(json_path).exists() {
        resolved = find_world_jsonl(None)?;
        resolved.as_str()
    } else {
        json_path
    };
    check_at_path(json_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    // An explicit, existing path is validated in place -- the discovery branch
    // (and its process-global path anchors) is never touched.
    #[test]
    fn check_validates_an_explicit_existing_world() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world.jsonl");
        std::fs::write(
            &path,
            "{\"name\":\"phys\",\"type\":\"PhysicsConfig\",\"args\":{}}\n",
        )
        .unwrap();
        check(path.to_str().unwrap()).unwrap();
    }

    #[test]
    fn check_reports_an_invalid_explicit_world() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world.jsonl");
        std::fs::write(
            &path,
            "{\"name\":\"x\",\"type\":\"NotARealAssetType\",\"args\":{}}\n",
        )
        .unwrap();
        assert!(check(path.to_str().unwrap()).is_err());
    }
}

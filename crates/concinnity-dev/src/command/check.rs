//! Discovery wrapper around `crate::authoring::check_at_path`.

use crate::authoring::check_at_path;
use crate::command::resolve_world_path;

/// Validate a world and report its errors without building blobs.
///
/// `json_path` names the world explicitly and must exist; `None` discovers it.
pub fn check(json_path: Option<&str>) -> std::io::Result<()> {
    check_at_path(&resolve_world_path(json_path)?)
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
        std::fs::write(&path, "[\"PhysicsConfig\",{\"$id\":\"phys\"}]\n").unwrap();
        check(path.to_str()).unwrap();
    }

    #[test]
    fn check_reports_an_invalid_explicit_world() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world.jsonl");
        std::fs::write(&path, "[\"NotARealAssetType\",{\"$id\":\"x\"}]\n").unwrap();
        assert!(check(path.to_str()).is_err());
    }

    #[test]
    fn check_of_a_missing_explicit_world_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.jsonl");
        let err = check(path.to_str()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }
}

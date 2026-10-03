use crate::command::resolve_world_path;

pub use crate::build_status::Verbosity;

/// Compile a world to binary blobs and write `world-lock.json`, showing each
/// step's progress as it runs and the warnings the build raises.
///
/// `json_path` names the world explicitly and must exist; `None` discovers it.
pub fn build(json_path: Option<&str>, verbosity: Verbosity) -> std::io::Result<()> {
    let json_path = resolve_world_path(json_path)?;
    crate::authoring::build_world_file_as(&json_path, &json_path, verbosity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_of_a_missing_explicit_world_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.jsonl");
        let err = build(path.to_str(), Verbosity::Normal).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }
}

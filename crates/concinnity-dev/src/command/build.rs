use crate::command::resolve_world_path;

/// Compile a world to binary blobs and write `world-lock.json`.
///
/// `json_path` names the world explicitly and must exist; `None` discovers it.
pub fn build(json_path: Option<&str>) -> std::io::Result<()> {
    // The cook reports what does not fail the build, such as a shader
    // warning, through the log.
    concinnity_engine::app::run::init_logging();
    let json_path = resolve_world_path(json_path)?;
    crate::authoring::build_world_file(&json_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_of_a_missing_explicit_world_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.jsonl");
        let err = build(path.to_str()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }
}

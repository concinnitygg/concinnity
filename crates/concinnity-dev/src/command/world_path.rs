//! Which world file a subcommand operates on.

use concinnity_cook::authoring::world::find_world_jsonl;

/// The world a subcommand should load: the `-f` path when the caller gave one,
/// otherwise whatever discovery finds.
///
/// A `-f` that does not exist is an error rather than a fall back to discovery.
/// Falling back runs a different world under the name the caller asked for, and
/// says nothing.
pub(crate) fn resolve_world_path(file: Option<&str>) -> std::io::Result<String> {
    match file {
        Some(p) if std::path::Path::new(p).exists() => Ok(p.to_string()),
        Some(p) => Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("world file not found: {p}"),
        )),
        None => discover_world_path(),
    }
}

/// The project's world, found from `worlds/` or the working directory.
pub(crate) fn discover_world_path() -> std::io::Result<String> {
    find_world_jsonl(crate::project::worlds_dir().as_deref(), None)
}

#[cfg(test)]
mod tests {
    use super::resolve_world_path;

    #[test]
    fn an_existing_path_is_taken_as_given() {
        let dir = tempfile::tempdir().expect("temp dir");
        let world = dir.path().join("scene.jsonl");
        std::fs::write(&world, "").expect("write");
        let given = world.to_string_lossy().into_owned();
        assert_eq!(resolve_world_path(Some(&given)).expect("resolves"), given);
    }

    // A named world that is not there fails loudly instead of silently becoming
    // whatever discovery turns up.
    #[test]
    fn a_missing_path_errors_rather_than_falling_back() {
        let dir = tempfile::tempdir().expect("temp dir");
        let missing = dir.path().join("absent.jsonl");
        let given = missing.to_string_lossy().into_owned();
        let err = resolve_world_path(Some(&given)).expect_err("missing world is an error");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(err.to_string().contains("absent.jsonl"), "{err}");
    }
}

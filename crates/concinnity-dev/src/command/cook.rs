use crate::command::Target;

pub(crate) use crate::build_status::Verbosity;

/// Compile the file `target` names to binary blobs and write `cook-lock.json`,
/// showing each step's progress as it runs and the warnings the build raises.
pub(crate) fn cook(target: Target<'_>, verbosity: Verbosity) -> std::io::Result<()> {
    cook_path(&target.resolve()?, verbosity)
}

/// Compile the file at `path` the way [`cook`] does.
pub(crate) fn cook_path(path: &std::path::Path, verbosity: Verbosity) -> std::io::Result<()> {
    crate::authoring::build_world_file_as(path, path, verbosity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cook_of_a_missing_explicit_file_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.jsonl");
        let err = cook(Target::file(path.to_str().unwrap()), Verbosity::Normal).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }
}

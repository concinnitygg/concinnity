//! Resolving the world a player run reads.

use std::path::{Path, PathBuf};

// Where a resolved world lives, owning its path so it outlives the borrow the
// engine takes. A directory holds blob 0 and any overflow siblings; a file is
// a single self-contained blob.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ResolvedBlob {
    Directory(PathBuf),
    File(PathBuf),
}

impl ResolvedBlob {
    pub(crate) fn as_source(&self) -> concinnity_engine::BlobSource<'_> {
        match self {
            ResolvedBlob::Directory(dir) => concinnity_engine::BlobSource::Directory(dir),
            ResolvedBlob::File(file) => concinnity_engine::BlobSource::File(file),
        }
    }
}

// Classify `path` as a world: a directory holds blob 0, a file is blob 0
// itself. `None` when nothing is there, which is the same answer for a missing
// `data` entry and a mistyped argument.
pub(crate) fn blob_source(path: &Path) -> Option<ResolvedBlob> {
    if path.is_dir() {
        return Some(ResolvedBlob::Directory(path.to_path_buf()));
    }
    if path.is_file() {
        return Some(ResolvedBlob::File(path.to_path_buf()));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // A `data` directory is the overflow-capable layout; a `data` file is one
    // self-contained blob. The same two rules decide a positional argument.
    #[test]
    fn a_data_directory_and_a_data_file_resolve_to_their_own_forms() {
        let tmp = tempfile::tempdir().unwrap();

        let dir = tmp.path().join("data");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("0"), b"blob").unwrap();
        assert_eq!(
            blob_source(&dir),
            Some(ResolvedBlob::Directory(dir.clone()))
        );

        let single = tmp.path().join("single").join("data");
        std::fs::create_dir_all(single.parent().unwrap()).unwrap();
        std::fs::write(&single, b"blob").unwrap();
        assert_eq!(
            blob_source(&single),
            Some(ResolvedBlob::File(single.clone()))
        );

        // The directory form owns blob 0 plus its siblings; the file form is
        // blob 0 itself.
        assert_eq!(
            ResolvedBlob::Directory(dir.clone()).as_source(),
            concinnity_engine::BlobSource::Directory(&dir)
        );
        assert_eq!(
            ResolvedBlob::File(single.clone()).as_source(),
            concinnity_engine::BlobSource::File(&single)
        );
    }

    // Nothing there is not an empty world: the caller turns this into a
    // not-found naming the path it looked at.
    #[test]
    fn a_missing_path_resolves_to_no_world() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(blob_source(&tmp.path().join("data")), None);
    }
}

//! Where a shipped app's compiled world sits, and the path a run names it by.

use std::path::{Path, PathBuf};

use crate::app::startup_error::StartupError;

/// Where a shipped app's compiled world sits. Both forms make the same file
/// blob 0; they differ in whether the world is allowed to spill into overflow
/// payload blobs, which are always siblings named by index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlobSource {
    /// A directory holding blob 0 plus any overflow blobs beside it.
    Directory(PathBuf),
    /// A single self-contained blob file. A world that needs overflow blobs is
    /// refused rather than half-loaded, since its siblings would be written
    /// into whatever directory the file happens to sit in.
    File(PathBuf),
}

impl BlobSource {
    /// Classify `path` as a world: a directory holds blob 0, a file is blob 0
    /// itself. `None` when nothing is there, which is the same answer for a
    /// missing `data` entry and a mistyped argument.
    pub fn classify(path: &Path) -> Option<Self> {
        if path.is_dir() {
            return Some(Self::Directory(path.to_path_buf()));
        }
        if path.is_file() {
            return Some(Self::File(path.to_path_buf()));
        }
        None
    }

    // The primary blob file: the named file itself, or blob 0 in the directory.
    pub(crate) fn primary(&self) -> PathBuf {
        match self {
            Self::Directory(dir) => dir.join("0"),
            Self::File(file) => file.clone(),
        }
    }

    // The refusal a single-file source owes a world that spans more blobs.
    pub(crate) fn check_span(&self, max_blob_index: u32) -> Option<StartupError> {
        match self {
            Self::File(file) if max_blob_index > 0 => Some(StartupError::OverflowUnsupported {
                blob: file.clone(),
                needed: max_blob_index,
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A `data` directory is the overflow-capable layout; a `data` file is one
    // self-contained blob. The same two rules decide a positional argument.
    #[test]
    fn a_data_directory_and_a_data_file_classify_as_their_own_forms() {
        let tmp = tempfile::tempdir().unwrap();

        let dir = tmp.path().join("data");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("0"), b"blob").unwrap();
        assert_eq!(
            BlobSource::classify(&dir),
            Some(BlobSource::Directory(dir.clone()))
        );

        let single = tmp.path().join("single").join("data");
        std::fs::create_dir_all(single.parent().unwrap()).unwrap();
        std::fs::write(&single, b"blob").unwrap();
        assert_eq!(
            BlobSource::classify(&single),
            Some(BlobSource::File(single.clone()))
        );
    }

    // Nothing there is not an empty world: the caller turns this into a
    // not-found naming the path it looked at.
    #[test]
    fn a_missing_path_classifies_as_no_world() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(BlobSource::classify(&tmp.path().join("data")), None);
    }

    // Both forms make the same file blob 0: the file itself, or `0` inside the
    // directory. This is what lets one runtime entry point serve both.
    #[test]
    fn each_blob_source_names_the_same_primary_file() {
        let path = PathBuf::from("/apps/MyGame/data");
        assert_eq!(
            BlobSource::Directory(path.clone()).primary(),
            path.join("0"),
            "a directory holds blob 0"
        );
        assert_eq!(
            BlobSource::File(path.clone()).primary(),
            path,
            "a single file is blob 0"
        );
    }

    // Overflow blobs are siblings named by index, so only the directory form
    // has somewhere to hold them. A single file whose world spans more is
    // refused rather than half-loaded.
    #[test]
    fn a_single_file_source_refuses_a_world_that_overflows() {
        let path = PathBuf::from("/apps/MyGame/data");
        let file = BlobSource::File(path.clone());
        assert!(file.check_span(0).is_none());
        assert!(matches!(
            file.check_span(2),
            Some(StartupError::OverflowUnsupported { blob, needed: 2 }) if blob == path
        ));
        let dir = BlobSource::Directory(path);
        assert!(dir.check_span(0).is_none());
        assert!(dir.check_span(7).is_none());
    }
}

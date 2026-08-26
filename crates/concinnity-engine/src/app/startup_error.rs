// Classification of a fatal startup failure into the two things it needs to
// produce: a line for the log, and a sentence for the person looking at the
// window. `CnResult` is the FFI-facing status enum and carries no context, so
// the classification happens here where the paths involved are still known.

use crate::result::CnResult;
use std::path::PathBuf;

// Why the runtime could not reach a playable state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StartupError {
    /// No compiled world data where the runtime expected it. The usual causes
    /// are a build that never ran and an installation missing its data folder.
    MissingData { blob: PathBuf },
    /// The data is present but did not load: a truncated file, a schema the
    /// binary no longer understands, or a failed read.
    UnreadableData { blob: PathBuf, cause: CnResult },
    /// The world was packaged as one self-contained blob file, but it needs
    /// overflow payload blobs, which only the directory layout can hold. Their
    /// siblings would land beside the executable, so this is refused rather
    /// than half-loaded.
    OverflowUnsupported { blob: PathBuf, needed: u32 },
    /// Nothing anchored the state tree, so there is nowhere to look for data.
    NoStateRoot,
}

impl StartupError {
    // Classify a blob-load failure, distinguishing absent data from data that
    // is present but unusable, since only the first is the user's to fix.
    // `blob` is the primary blob's path (`concinnity_store::blob::blob_path(0)`),
    // passed in rather than resolved here so the classification stays a pure
    // function of its inputs.
    pub(crate) fn from_blob_failure(blob: PathBuf, cause: CnResult) -> Self {
        if blob.exists() {
            StartupError::UnreadableData { blob, cause }
        } else {
            StartupError::MissingData { blob }
        }
    }

    // How the failure surfaces to the process's exit status.
    pub(crate) fn io_kind(&self) -> std::io::ErrorKind {
        match self {
            StartupError::MissingData { .. } | StartupError::NoStateRoot => {
                std::io::ErrorKind::NotFound
            }
            StartupError::UnreadableData { .. } | StartupError::OverflowUnsupported { .. } => {
                std::io::ErrorKind::InvalidData
            }
        }
    }

    // The sentence shown on the error screen. Names the path, because the
    // path is the actionable part, and stays free of internal vocabulary.
    pub(crate) fn user_message(&self) -> String {
        match self {
            StartupError::MissingData { blob } => {
                format!("Failed to find the data blob:\n{}", blob.display())
            }
            StartupError::UnreadableData { blob, .. } => {
                format!("Failed to read the data blob:\n{}", blob.display())
            }
            StartupError::OverflowUnsupported { blob, .. } => {
                format!("This app's data is incomplete:\n{}", blob.display())
            }
            StartupError::NoStateRoot => "Failed to find this app's data.".to_string(),
        }
    }

    // The developer-facing line, carrying the status the user message omits.
    pub(crate) fn log_line(&self) -> String {
        match self {
            StartupError::MissingData { blob } => {
                format!(
                    "no compiled world data at {} -- run `concinnity build` first",
                    blob.display()
                )
            }
            StartupError::UnreadableData { blob, cause } => {
                format!(
                    "compiled world data at {} failed to load: {cause}",
                    blob.display()
                )
            }
            StartupError::OverflowUnsupported { blob, needed } => {
                format!(
                    "{} is a single blob file, but this world spans {} more; \
                     re-export it so the player ships a `data/` directory",
                    blob.display(),
                    needed
                )
            }
            StartupError::NoStateRoot => {
                "no state directory was installed, so there is nowhere to read world data from"
                    .to_string()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_blob_classifies_as_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let blob = dir.path().join("data").join("0");

        let err = StartupError::from_blob_failure(blob, CnResult::FileIo);
        assert!(matches!(err, StartupError::MissingData { .. }));
        assert!(err.user_message().contains("Failed to find"));
        assert!(err.log_line().contains("concinnity build"));
    }

    #[test]
    fn a_present_but_broken_blob_classifies_as_unreadable() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("data")).expect("data dir");
        let blob = dir.path().join("data").join("0");
        std::fs::write(&blob, b"garbage").expect("write blob");

        let err = StartupError::from_blob_failure(blob, CnResult::FileIo);
        assert!(matches!(err, StartupError::UnreadableData { .. }));
        assert!(err.user_message().contains("Failed to read"));
        // The status the user message deliberately omits stays in the log line.
        assert!(err.log_line().contains(&CnResult::FileIo.to_string()));
    }

    // Both messages name the path, which is the part the reader can act on.
    #[test]
    fn every_message_names_the_blob_path() {
        for err in [
            StartupError::MissingData {
                blob: PathBuf::from("/somewhere/data/0"),
            },
            StartupError::UnreadableData {
                blob: PathBuf::from("/somewhere/data/0"),
                cause: CnResult::FileIo,
            },
            StartupError::OverflowUnsupported {
                blob: PathBuf::from("/somewhere/data/0"),
                needed: 2,
            },
        ] {
            assert!(err.user_message().contains("/somewhere/data/0"));
            assert!(err.log_line().contains("/somewhere/data/0"));
        }
    }

    // The refusal has to say what to do about it, since a player cannot tell
    // from a half-loaded world that its data was packaged in the wrong shape.
    #[test]
    fn the_overflow_refusal_names_the_fix() {
        let err = StartupError::OverflowUnsupported {
            blob: PathBuf::from("/apps/MyGame/data"),
            needed: 3,
        };
        assert!(err.log_line().contains("single blob file"), "{err:?}");
        assert!(err.log_line().contains("`data/` directory"), "{err:?}");
        assert!(err.log_line().contains('3'), "{err:?}");
        assert_eq!(err.io_kind(), std::io::ErrorKind::InvalidData);
    }

    // A missing state root is a not-found, not a corrupt-data report: there is
    // no path to name because nothing anchored one.
    #[test]
    fn no_state_root_reports_not_found_without_a_path() {
        let err = StartupError::NoStateRoot;
        assert_eq!(err.io_kind(), std::io::ErrorKind::NotFound);
        assert!(err.log_line().contains("no state directory"));
    }
}

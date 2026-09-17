//! Classification of a fatal startup failure into the two things it needs to
//! produce: a line for the log, and a sentence for the person looking at the
//! window. The classification happens here, where the paths involved are still
//! known, and carries the load failure itself as its cause.

use crate::blob::WorldLoadError;
use std::fmt;
use std::path::PathBuf;

/// Why the runtime could not reach a playable state.
#[derive(Debug)]
pub enum StartupError {
    /// No compiled world data where the runtime expected it. The usual causes
    /// are a build that never ran and an installation missing its data folder.
    MissingData {
        /// The primary blob file that was looked for.
        blob: PathBuf,
    },
    /// The data is present but did not load: a truncated file, a schema the
    /// binary no longer understands, or a failed read.
    UnreadableData {
        /// The primary blob file that was read.
        blob: PathBuf,
        /// What the read reported.
        cause: WorldLoadError,
    },
    /// The world was packaged as one self-contained blob file, but it needs
    /// overflow payload blobs, which only the directory layout can hold. Their
    /// siblings would land beside the executable, so this is refused rather
    /// than half-loaded.
    OverflowUnsupported {
        /// The single blob file the world was read from.
        blob: PathBuf,
        /// How many further blobs the world spans.
        needed: u32,
    },
    /// Nothing anchored the state tree, so there is nowhere to look for data.
    NoStateRoot,
}

impl StartupError {
    /// Classify a blob-load failure, distinguishing absent data from data that
    /// is present but unusable, since only the first is the user's to fix.
    /// `blob` is the primary blob's path, passed in rather than resolved here
    /// so the classification stays a pure function of its inputs.
    pub fn from_blob_failure(blob: PathBuf, cause: WorldLoadError) -> Self {
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
}

/// The developer-facing line, carrying the status the user message omits.
impl fmt::Display for StartupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StartupError::MissingData { blob } => write!(
                f,
                "no compiled world data at {} -- run `concinnity build` first",
                blob.display()
            ),
            StartupError::UnreadableData { blob, cause } => write!(
                f,
                "compiled world data at {} failed to load: {cause}",
                blob.display()
            ),
            StartupError::OverflowUnsupported { blob, needed } => write!(
                f,
                "{} is a single blob file, but this world spans {} more; \
                 re-export it so the player ships a `data/` directory",
                blob.display(),
                needed
            ),
            StartupError::NoStateRoot => f.write_str(
                "no state directory was installed, so there is nowhere to read world data from",
            ),
        }
    }
}

/// The load failure stays reachable underneath the classification, so a caller
/// walking the chain reaches the file and the format verdict below it.
impl std::error::Error for StartupError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StartupError::UnreadableData { cause, .. } => Some(cause),
            StartupError::MissingData { .. }
            | StartupError::OverflowUnsupported { .. }
            | StartupError::NoStateRoot => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::error::AssetError;

    // A load failure to classify. Which one it is does not matter here; that
    // it survives into the log line does.
    fn cause() -> WorldLoadError {
        WorldLoadError::Asset(AssetError::UnknownComponent { discriminant: 9 })
    }

    #[test]
    fn a_missing_blob_classifies_as_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let blob = dir.path().join("data").join("0");

        let err = StartupError::from_blob_failure(blob, cause());
        assert!(matches!(err, StartupError::MissingData { .. }));
        assert!(err.user_message().contains("Failed to find"));
        assert!(err.to_string().contains("concinnity build"));
    }

    #[test]
    fn a_present_but_broken_blob_classifies_as_unreadable() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("data")).expect("data dir");
        let blob = dir.path().join("data").join("0");
        std::fs::write(&blob, b"garbage").expect("write blob");

        let err = StartupError::from_blob_failure(blob, cause());
        assert!(matches!(err, StartupError::UnreadableData { .. }));
        assert!(err.user_message().contains("Failed to read"));
        // The cause the user message deliberately omits stays in the log line,
        // and stays reachable as a source.
        assert!(err.to_string().contains(&cause().to_string()));
        assert!(std::error::Error::source(&err).is_some());
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
                cause: cause(),
            },
            StartupError::OverflowUnsupported {
                blob: PathBuf::from("/somewhere/data/0"),
                needed: 2,
            },
        ] {
            assert!(err.user_message().contains("/somewhere/data/0"));
            assert!(err.to_string().contains("/somewhere/data/0"));
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
        assert!(err.to_string().contains("single blob file"), "{err:?}");
        assert!(err.to_string().contains("`data/` directory"), "{err:?}");
        assert!(err.to_string().contains('3'), "{err:?}");
        assert_eq!(err.io_kind(), std::io::ErrorKind::InvalidData);
    }

    // A missing state root is a not-found, not a corrupt-data report: there is
    // no path to name because nothing anchored one.
    #[test]
    fn no_state_root_reports_not_found_without_a_path() {
        let err = StartupError::NoStateRoot;
        assert_eq!(err.io_kind(), std::io::ErrorKind::NotFound);
        assert!(err.to_string().contains("no state directory"));
    }
}

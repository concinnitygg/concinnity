//! Why a blob file could not be read.

use concinnity_core::blob::BlobError;
use thiserror::Error;

/// Why a blob file could not be read.
///
/// The format crate is I/O-free and never learns which file its bytes came
/// from, so the path and the failing read belong here, wrapped around the
/// format's own verdict rather than flattened into a status.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BlobLoadError {
    /// Nothing anchored the blob layout, so there is no path to read from.
    #[error("no blob layout is anchored, so there is nowhere to read world data from")]
    NoStateRoot,

    /// The file could not be read.
    #[error("reading {path} failed")]
    Io {
        /// The file that was being read.
        path: String,
        /// What the read reported.
        #[source]
        source: std::io::Error,
    },

    /// The file was read but is not usable world data.
    #[error("{path} is not usable world data: {source}")]
    Format {
        /// The file that was read.
        path: String,
        /// What the format rejected.
        #[source]
        source: BlobError,
    },
}

impl BlobLoadError {
    pub(super) fn io(path: &str, source: std::io::Error) -> Self {
        BlobLoadError::Io {
            path: path.to_string(),
            source,
        }
    }

    pub(super) fn format(path: &str, source: BlobError) -> Self {
        BlobLoadError::Format {
            path: path.to_string(),
            source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    // The path is the actionable half and the cause is what the flat status
    // used to drop, so both have to survive.
    #[test]
    fn every_failure_that_names_a_file_names_it_and_keeps_its_cause() {
        let io = BlobLoadError::io(
            "data/0",
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        );
        assert!(io.to_string().contains("data/0"), "{io}");
        assert_eq!(io.source().expect("the io error").to_string(), "denied");

        let format = BlobLoadError::format("data/0", BlobError::ValidityMismatch(7));
        assert!(format.to_string().contains("data/0"), "{format}");
        assert!(format.to_string().contains("different version"), "{format}");
        assert!(format.source().is_some());
    }

    // An unanchored layout names no file, which is what distinguishes it from
    // a read that failed.
    #[test]
    fn an_unanchored_layout_names_no_file() {
        let error = BlobLoadError::NoStateRoot;
        assert!(error.source().is_none());
        assert!(error.to_string().contains("no blob layout"), "{error}");
    }
}

//! The failure an [`App`](crate::App) reports, on every build of the crate.

#[cfg(feature = "std")]
use std::path::PathBuf;

use concinnity_core::result::CnResult;

/// Why an application could not load its world, or could not run it.
///
/// The variants naming a file exist only where there is a filesystem to name
/// one in, so a `no_std` build reports [`Runtime`](Error::Runtime) alone.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// No compiled world data where the app expected it. The usual causes are
    /// a build that never ran and an installation missing its data folder.
    #[cfg(feature = "std")]
    #[error("no compiled world data at {}", blob.display())]
    MissingData {
        /// The primary blob file that was looked for.
        blob: PathBuf,
    },

    /// The data is present but did not load: a truncated file, a schema the
    /// binary no longer understands, or a failed read.
    #[cfg(feature = "std")]
    #[error("compiled world data at {} failed to load: {cause}", blob.display())]
    UnreadableData {
        /// The primary blob file that was read.
        blob: PathBuf,
        /// What the read reported.
        cause: CnResult,
    },

    /// The world was packaged as one self-contained blob file, but it needs
    /// overflow payload blobs, which only the directory layout can hold.
    #[cfg(feature = "std")]
    #[error("{} is a single blob file, but this world spans {needed} more", blob.display())]
    OverflowUnsupported {
        /// The single blob file the world was read from.
        blob: PathBuf,
        /// How many further blobs the world spans.
        needed: u32,
    },

    /// Nothing anchored the state tree, so there is nowhere to look for data.
    #[cfg(feature = "std")]
    #[error("no state directory was installed, so there is nowhere to read world data from")]
    NoStateRoot,

    /// The world refused to start, or a system stopped it with a failure.
    #[error(transparent)]
    Runtime(#[from] CnResult),
}

#[cfg(feature = "std")]
impl Error {
    // How the failure surfaces to a process's exit status: absent data is a
    // not-found, present-but-unusable data is invalid.
    fn io_kind(&self) -> std::io::ErrorKind {
        match self {
            Error::MissingData { .. } | Error::NoStateRoot => std::io::ErrorKind::NotFound,
            Error::UnreadableData { .. } | Error::OverflowUnsupported { .. } => {
                std::io::ErrorKind::InvalidData
            }
            Error::Runtime(_) => std::io::ErrorKind::Other,
        }
    }
}

/// The kind is what a caller reading an `io::Error` acts on, so the
/// distinction between data that is absent and data that is unusable survives
/// the conversion.
#[cfg(feature = "std")]
impl From<Error> for std::io::Error {
    fn from(error: Error) -> Self {
        std::io::Error::new(error.io_kind(), error.to_string())
    }
}

// The engine classifies a load failure while the paths involved are still in
// hand. A free function rather than a `From` impl, which would put the engine's
// type on this crate's public surface.
#[cfg(feature = "std")]
pub(crate) fn from_startup(error: concinnity_engine::StartupError) -> Error {
    use concinnity_engine::StartupError as S;
    match error {
        S::MissingData { blob } => Error::MissingData { blob },
        S::UnreadableData { blob, cause } => Error::UnreadableData { blob, cause },
        S::OverflowUnsupported { blob, needed } => Error::OverflowUnsupported { blob, needed },
        S::NoStateRoot => Error::NoStateRoot,
    }
}

#[cfg(test)]
mod tests {
    use super::Error;
    use alloc::string::ToString;
    use concinnity_core::result::CnResult;

    // The one variant every tier reports, so the signature ports whether or
    // not there is an operating system underneath it.
    #[test]
    fn a_runtime_failure_carries_the_status_it_was_built_from() {
        let error = Error::from(CnResult::InvalidState);
        assert_eq!(error, Error::Runtime(CnResult::InvalidState));
        assert_eq!(error.to_string(), CnResult::InvalidState.to_string());
    }

    #[cfg(feature = "std")]
    mod std_tier {
        use super::super::Error;
        use concinnity_core::result::CnResult;
        use std::io::ErrorKind;
        use std::path::PathBuf;

        fn blob() -> PathBuf {
            PathBuf::from("/apps/MyGame/data/0")
        }

        // The path is the actionable part of a load failure, so every variant
        // that has one names it.
        #[test]
        fn every_load_failure_names_the_blob_it_could_not_use() {
            for error in [
                Error::MissingData { blob: blob() },
                Error::UnreadableData {
                    blob: blob(),
                    cause: CnResult::FileIo,
                },
                Error::OverflowUnsupported {
                    blob: blob(),
                    needed: 2,
                },
            ] {
                assert!(error.to_string().contains("/apps/MyGame/data/0"), "{error}");
            }
        }

        // The status a message would otherwise drop stays readable.
        #[test]
        fn an_unreadable_blob_reports_what_the_read_said() {
            let error = Error::UnreadableData {
                blob: blob(),
                cause: CnResult::FileIo,
            };
            assert!(
                error.to_string().contains(&CnResult::FileIo.to_string()),
                "{error}"
            );
        }

        // A single-file world that spans more blobs says how many, since that
        // is what tells the reader the export shape was wrong.
        #[test]
        fn an_overflowing_world_reports_how_far_it_spans() {
            let error = Error::OverflowUnsupported {
                blob: blob(),
                needed: 3,
            };
            assert!(error.to_string().contains('3'), "{error}");
        }

        // Absent data and unusable data are different to whoever is handling
        // the failure, and the conversion is where that distinction is easiest
        // to lose.
        #[test]
        fn the_io_conversion_keeps_the_kind() {
            let cases = [
                (Error::MissingData { blob: blob() }, ErrorKind::NotFound),
                (Error::NoStateRoot, ErrorKind::NotFound),
                (
                    Error::UnreadableData {
                        blob: blob(),
                        cause: CnResult::FileIo,
                    },
                    ErrorKind::InvalidData,
                ),
                (
                    Error::OverflowUnsupported {
                        blob: blob(),
                        needed: 2,
                    },
                    ErrorKind::InvalidData,
                ),
                (Error::Runtime(CnResult::InvalidState), ErrorKind::Other),
            ];

            for (error, kind) in cases {
                let message = error.to_string();
                let io: std::io::Error = error.into();
                assert_eq!(io.kind(), kind, "{message}");
                assert_eq!(io.to_string(), message);
            }
        }

        // The engine classifies a load failure while the paths are still in
        // hand; the facade carries that classification rather than flattening
        // it on the way out.
        #[test]
        fn an_engine_startup_failure_maps_variant_for_variant() {
            use concinnity_engine::StartupError as S;

            let cases = [
                (
                    S::MissingData { blob: blob() },
                    Error::MissingData { blob: blob() },
                ),
                (
                    S::UnreadableData {
                        blob: blob(),
                        cause: CnResult::FileIo,
                    },
                    Error::UnreadableData {
                        blob: blob(),
                        cause: CnResult::FileIo,
                    },
                ),
                (
                    S::OverflowUnsupported {
                        blob: blob(),
                        needed: 4,
                    },
                    Error::OverflowUnsupported {
                        blob: blob(),
                        needed: 4,
                    },
                ),
                (S::NoStateRoot, Error::NoStateRoot),
            ];

            for (startup, expected) in cases {
                assert_eq!(super::super::from_startup(startup), expected);
            }
        }
    }
}

//! The facade's failure type, on every build of the crate.

use alloc::string::String;
#[cfg(feature = "cook")]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::path::PathBuf;

use concinnity_core::error::WorldError;

/// The facade's failure type: why a value could not be baked, a world could not
/// be compiled or loaded, or an app could not run it.
///
/// The variants naming a file exist only where there is a filesystem to name
/// one in, so a `no_std` build reports [`Bake`](Error::Bake) and
/// [`Runtime`](Error::Runtime) alone.
#[derive(Debug, thiserror::Error)]
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
        #[source]
        cause: concinnity_engine::WorldLoadError,
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

    /// A value the [`bake`](crate::bake) functions could not compute, usually
    /// one that needs the `cook` module's importers.
    #[error("{0}")]
    Bake(String),

    /// The declared world failed validation, one message per problem found.
    #[cfg(feature = "cook")]
    #[error("world validation failed:\n{}", .0.join("\n"))]
    Validation(Vec<String>),

    /// Compiling or writing the declared world failed.
    #[cfg(feature = "cook")]
    #[error("{message}")]
    Build {
        /// The category of the underlying failure.
        kind: std::io::ErrorKind,
        /// What the failing step reported.
        message: String,
    },

    /// The world refused to start, or a system stopped it with a failure.
    #[error(transparent)]
    Runtime(#[from] WorldError),
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
            // A world that could not read its own data is the runtime half of
            // the same failure the load variants above report.
            Error::Runtime(WorldError::Asset(_) | WorldError::Payload(_)) => {
                std::io::ErrorKind::InvalidData
            }
            Error::Bake(_) => std::io::ErrorKind::InvalidInput,
            #[cfg(feature = "cook")]
            Error::Validation(_) => std::io::ErrorKind::InvalidData,
            #[cfg(feature = "cook")]
            Error::Build { kind, .. } => *kind,
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

// The cook classifies a build failure at the seam it happened; the facade
// carries that classification onto the variant of its own type that holds it.
// A free function for the same reason as `from_startup`: a `From` impl would
// put the cook's type on this crate's public surface.
#[cfg(feature = "cook")]
pub(crate) fn from_cook(error: concinnity_cook::WorldBuildError) -> Error {
    use alloc::string::ToString;
    use concinnity_cook::WorldBuildError as W;
    match error {
        W::Validation(messages) => Error::Validation(messages),
        W::Build { kind, message } => Error::Build { kind, message },
        W::Asset(cause) => Error::Runtime(cause.into()),
        other => Error::Build {
            kind: std::io::ErrorKind::Other,
            message: other.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::Error;
    use alloc::string::ToString;
    use concinnity_core::error::WorldError;

    // The one variant every tier reports, so the signature ports whether or
    // not there is an operating system underneath it.
    #[test]
    fn a_runtime_failure_carries_the_failure_it_was_built_from() {
        let error = Error::from(WorldError::AlreadyStarted);
        assert!(matches!(error, Error::Runtime(WorldError::AlreadyStarted)));
        assert_eq!(error.to_string(), WorldError::AlreadyStarted.to_string());
    }

    // A bake failure is its message, since that message is the direction to
    // the cook module.
    #[test]
    fn a_bake_failure_displays_its_message() {
        let error = Error::Bake("compile it with the cook module".into());
        assert_eq!(error.to_string(), "compile it with the cook module");
    }

    // A load failure to build errors from. Which one it is does not matter to
    // these tests; that it is a typed cause rather than a message does.
    #[cfg(feature = "std")]
    fn load_failure() -> concinnity_engine::WorldLoadError {
        concinnity_engine::WorldLoadError::Asset(
            concinnity_core::error::AssetError::UnknownComponent { discriminant: 9 },
        )
    }

    #[cfg(feature = "std")]
    mod std_tier {
        use super::super::Error;
        use super::load_failure;
        use concinnity_core::error::{AssetError, WorldError};
        use core::error::Error as _;
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
                    cause: load_failure(),
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
                cause: load_failure(),
            };
            assert!(
                error.to_string().contains(&load_failure().to_string()),
                "{error}"
            );
            assert!(error.source().is_some(), "the cause stays reachable");
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
                        cause: load_failure(),
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
                (
                    Error::Runtime(WorldError::Asset(AssetError::UnknownComponent {
                        discriminant: 9,
                    })),
                    ErrorKind::InvalidData,
                ),
                (Error::Runtime(WorldError::AlreadyStarted), ErrorKind::Other),
                (Error::Bake("unbakeable".into()), ErrorKind::InvalidInput),
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
                        cause: load_failure(),
                    },
                    Error::UnreadableData {
                        blob: blob(),
                        cause: load_failure(),
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

            // The errors carry an `io::Error` underneath, which has no
            // equality, so the pairs are matched on variant and message.
            for (startup, expected) in cases {
                let mapped = super::super::from_startup(startup);
                assert_eq!(
                    core::mem::discriminant(&mapped),
                    core::mem::discriminant(&expected)
                );
                assert_eq!(mapped.to_string(), expected.to_string());
            }
        }

        // Every validation problem reaches the reader, not only the first.
        #[cfg(feature = "cook")]
        #[test]
        fn a_validation_failure_lists_every_message_as_invalid_data() {
            let error = Error::Validation(vec!["first problem".into(), "second problem".into()]);
            let message = error.to_string();
            assert!(message.contains("first problem"), "{message}");
            assert!(message.contains("second problem"), "{message}");
            let io: std::io::Error = error.into();
            assert_eq!(io.kind(), ErrorKind::InvalidData);
        }

        // A build failure keeps the kind the failing step reported.
        #[cfg(feature = "cook")]
        #[test]
        fn a_cook_build_failure_becomes_a_build_error_of_the_same_kind() {
            let error = super::super::from_cook(concinnity_cook::WorldBuildError::Build {
                kind: ErrorKind::PermissionDenied,
                message: "data/0 is read-only".into(),
            });
            assert!(
                matches!(
                    &error,
                    Error::Build {
                        kind: ErrorKind::PermissionDenied,
                        message
                    } if message == "data/0 is read-only"
                ),
                "{error:?}"
            );
            let io: std::io::Error = error.into();
            assert_eq!(io.kind(), ErrorKind::PermissionDenied);
        }

        // The cook's three failure modes each land on the variant that holds
        // them, so nothing flattens into a message on the way out.
        #[cfg(feature = "cook")]
        #[test]
        fn every_cook_failure_maps_onto_the_variant_that_carries_it() {
            use concinnity_cook::WorldBuildError as W;

            let validation = super::super::from_cook(W::Validation(vec!["orphan".into()]));
            let Error::Validation(messages) = &validation else {
                panic!("expected a validation failure, got {validation:?}");
            };
            assert_eq!(messages.len(), 1);
            assert!(messages[0].contains("orphan"), "{messages:?}");

            let asset =
                super::super::from_cook(W::Asset(AssetError::UnknownComponent { discriminant: 9 }));
            assert!(
                matches!(
                    asset,
                    Error::Runtime(WorldError::Asset(AssetError::UnknownComponent {
                        discriminant: 9
                    }))
                ),
                "{asset:?}"
            );
        }
    }
}

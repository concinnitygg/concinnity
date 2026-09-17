//! Why a declared world did not compile.

use concinnity_core::error::AssetError;
use thiserror::Error;

/// Why a declared world did not compile into a runnable world or a blob file.
///
/// The three failure modes are distinct to whoever handles them: the world was
/// rejected before anything was built, a build step failed, or a compiled
/// record did not resolve back into a component.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WorldBuildError {
    /// The declared world failed validation, one message per problem found.
    #[error("world validation failed:\n{}", .0.join("\n"))]
    Validation(Vec<String>),

    /// A build step failed: serializing a declaration, expanding the world,
    /// compiling a payload, or writing the result.
    #[error("{message}")]
    Build {
        /// The category of the underlying failure.
        kind: std::io::ErrorKind,
        /// What the failing step reported.
        message: String,
    },

    /// A compiled record did not resolve back into a component.
    #[error(transparent)]
    Asset(#[from] AssetError),
}

// A free function rather than a `From` impl, which would let any io failure
// pass as a build failure.
pub(super) fn from_io(error: std::io::Error) -> WorldBuildError {
    WorldBuildError::Build {
        kind: error.kind(),
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every problem the validation found reaches the reader, not only the
    // first, since an author fixes them in one pass.
    #[test]
    fn a_validation_failure_lists_every_message() {
        let error = WorldBuildError::Validation(vec!["first".into(), "second".into()]);
        let message = error.to_string();
        assert!(message.contains("first"), "{message}");
        assert!(message.contains("second"), "{message}");
    }

    // The kind is what a caller acts on, so it survives the conversion from
    // the io failure the build step reported.
    #[test]
    fn an_io_failure_keeps_its_kind_and_message() {
        let error = from_io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "data/0 is read-only",
        ));
        assert!(
            matches!(
                &error,
                WorldBuildError::Build {
                    kind: std::io::ErrorKind::PermissionDenied,
                    message,
                } if message == "data/0 is read-only"
            ),
            "{error:?}"
        );
    }

    // An unresolvable record keeps the asset failure itself, so the reader
    // sees what the record was rather than a rewritten message.
    #[test]
    fn an_asset_failure_displays_the_failure_it_wraps() {
        let cause = AssetError::UnknownComponent { discriminant: 9 };
        let error = WorldBuildError::from(AssetError::UnknownComponent { discriminant: 9 });
        assert_eq!(error.to_string(), cause.to_string());
    }
}

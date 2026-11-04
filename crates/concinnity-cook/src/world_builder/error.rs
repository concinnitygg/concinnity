//! Why a declared world did not validate or compile.

use concinnity_engine::error::AssetError;
use thiserror::Error;

/// Why a declared world, or a single asset, failed validation: one message
/// per problem found, so an author can fix them in one pass.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{}", .0.join("\n"))]
pub struct WorldValidationError(Vec<String>);

impl WorldValidationError {
    pub(crate) fn new(messages: Vec<String>) -> Self {
        Self(messages)
    }

    /// Every problem found, in the order the checks reported them.
    pub fn messages(&self) -> &[String] {
        &self.0
    }
}

/// Why a declared world did not compile into a runnable world or blob files.
///
/// The three failure modes are distinct to whoever handles them: the world was
/// rejected before anything was built, a build step failed, or a compiled
/// record did not resolve back into a component.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WorldBuildError {
    /// The declared world failed validation.
    #[error("world validation failed:\n{0}")]
    Validation(#[from] WorldValidationError),

    /// A build step failed: reading the world, serializing a declaration,
    /// importing or compiling a payload, or writing the result.
    #[error(transparent)]
    Build(std::io::Error),

    /// A compiled record did not resolve back into a component.
    #[error(transparent)]
    Asset(#[from] AssetError),
}

/// The bridge for a host whose own boundary is `std::io::Result`: a build
/// step failure keeps its io error, and the rest become `InvalidData`.
impl From<WorldBuildError> for std::io::Error {
    fn from(error: WorldBuildError) -> Self {
        match error {
            WorldBuildError::Build(io) => io,
            other => Self::new(std::io::ErrorKind::InvalidData, other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_validation_failure_lists_every_message() {
        let error = WorldBuildError::from(WorldValidationError::new(vec![
            "first".into(),
            "second".into(),
        ]));
        let message = error.to_string();
        assert!(message.contains("first\nsecond"), "{message}");
    }

    // The kind is what a caller acts on, so a build step failure reaches an
    // io host unchanged.
    #[test]
    fn a_build_failure_keeps_its_io_error() {
        let error = WorldBuildError::Build(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "data/0 is read-only",
        ));
        assert_eq!(error.to_string(), "data/0 is read-only");
        let io = std::io::Error::from(error);
        assert_eq!(io.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(io.to_string(), "data/0 is read-only");
    }

    #[test]
    fn a_validation_failure_reaches_an_io_host_as_invalid_data() {
        let error = WorldBuildError::from(WorldValidationError::new(vec!["bad".into()]));
        let io = std::io::Error::from(error);
        assert_eq!(io.kind(), std::io::ErrorKind::InvalidData);
        assert!(io.to_string().ends_with("bad"), "{io}");
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

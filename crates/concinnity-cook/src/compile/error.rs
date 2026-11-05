//! Why a payload did not compile: a source file that could not be read, kept
//! with its I/O cause, or args and contents the compile does not accept.

/// A payload compile's failure.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    /// A source file the compile reads could not be read.
    #[error("{context} '{}': {source}", path.display())]
    Source {
        /// What the compile was doing, such as "failed to open DDS texture".
        context: String,
        /// The path as the compile was handed it.
        path: std::path::PathBuf,
        /// The read's own failure, whose kind tells a missing file apart.
        #[source]
        source: std::io::Error,
    },
    /// The args or a source's contents are not something the compile accepts.
    #[error("{0}")]
    Invalid(String),
}

impl CompileError {
    /// The bytes of `path`, or a [`Source`](Self::Source) error naming what the
    /// compile was doing.
    pub(crate) fn read(path: impl AsRef<std::path::Path>, context: &str) -> Result<Vec<u8>, Self> {
        let path = path.as_ref();
        std::fs::read(path).map_err(|source| Self::source(path, context, source))
    }

    /// The text of `path`, or a [`Source`](Self::Source) error naming what the
    /// compile was doing.
    pub(crate) fn read_text(
        path: impl AsRef<std::path::Path>,
        context: &str,
    ) -> Result<String, Self> {
        let path = path.as_ref();
        std::fs::read_to_string(path).map_err(|source| Self::source(path, context, source))
    }

    /// The same failure with `prefix` ahead of its message, keeping its kind.
    #[must_use]
    pub(crate) fn context(self, prefix: impl core::fmt::Display) -> Self {
        match self {
            Self::Source {
                context,
                path,
                source,
            } => Self::Source {
                context: format!("{prefix}: {context}"),
                path,
                source,
            },
            Self::Invalid(message) => Self::Invalid(format!("{prefix}: {message}")),
        }
    }

    /// A [`Source`](Self::Source) error for a read of `path` that failed.
    pub(crate) fn source(
        path: impl AsRef<std::path::Path>,
        context: &str,
        source: std::io::Error,
    ) -> Self {
        Self::Source {
            context: context.to_string(),
            path: path.as_ref().to_path_buf(),
            source,
        }
    }
}

impl From<String> for CompileError {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}

/// A source read keeps its I/O kind, so a missing file stays `NotFound`;
/// anything else is invalid data.
impl From<CompileError> for std::io::Error {
    fn from(error: CompileError) -> Self {
        let kind = match &error {
            CompileError::Source { source, .. } => source.kind(),
            CompileError::Invalid(_) => std::io::ErrorKind::InvalidData,
        };
        Self::new(kind, error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_source_stays_not_found_through_io() {
        let tree = concinnity_testing::TempTree::new();
        let path = tree.join("absent.png");
        let path = path.to_str().expect("a utf-8 temp path");

        let error = CompileError::read(path, "failed to open texture").unwrap_err();
        let io = std::io::Error::from(error.context("heightmap Texture 'hills'"));
        assert_eq!(io.kind(), std::io::ErrorKind::NotFound);
        assert!(
            io.to_string()
                .starts_with("heightmap Texture 'hills': failed to open texture '"),
            "{io}"
        );
    }

    #[test]
    fn invalid_args_read_as_invalid_data() {
        let io = std::io::Error::from(CompileError::from("missing 'source'".to_string()));
        assert_eq!(io.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(io.to_string(), "missing 'source'");
    }
}

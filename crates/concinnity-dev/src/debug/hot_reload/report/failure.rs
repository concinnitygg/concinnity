//! Why a reload left the live pipelines as they were.

use concinnity_cook::compile::program::{CompileFailure, ProgramError};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReloadFailure {
    // The compiler rejected the files. Its diagnostics name the subject's own
    // files by the resolved path the recompile read.
    Compile(CompileFailure),
    // The backend refused to build a pipeline from the compiled programs.
    Rejected(String),
    // A file could not be read, or the compile could not start.
    Unstarted(String),
}

impl ReloadFailure {
    // The first error's file name and line, for a message too short for the
    // whole report.
    pub(crate) fn first_error_at(&self) -> Option<String> {
        let Self::Compile(failed) = self else {
            return None;
        };
        let error = failed.errors().next()?;
        let file = std::path::Path::new(&error.path)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or(&error.path);
        Some(format!("{file}:{}", error.line))
    }
}

impl From<ProgramError> for ReloadFailure {
    fn from(e: ProgramError) -> Self {
        match e {
            ProgramError::Compile(failed) => Self::Compile(failed),
            other => Self::Unstarted(other.to_string()),
        }
    }
}

impl fmt::Display for ReloadFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(failed) => failed.fmt(f),
            Self::Rejected(e) => write!(f, "pipeline rebuild rejected: {e}"),
            Self::Unstarted(e) => f.write_str(e),
        }
    }
}

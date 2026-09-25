//! Why an authored program did not compile.

use std::fmt;

use concinnity_shader::diagnostics::{self, Diagnostic, Severity};
use thiserror::Error;

/// Why an asset's programs did not compile.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProgramError {
    /// The scratch directory the compile runs in could not be made.
    #[error("{owner}: no scratch directory to compile in: {source}")]
    Scratch {
        /// The asset, as a message names it.
        owner: String,
        /// Why the directory could not be made.
        #[source]
        source: std::io::Error,
    },

    /// The compiler rejected one or more entries.
    #[error(transparent)]
    Compile(#[from] CompileFailure),
}

impl From<ProgramError> for std::io::Error {
    fn from(e: ProgramError) -> Self {
        let kind = match &e {
            ProgramError::Scratch { source, .. } => source.kind(),
            ProgramError::Compile(_) => std::io::ErrorKind::InvalidData,
        };
        std::io::Error::new(kind, e)
    }
}

/// One entry the compiler rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryFailure {
    /// The entry point.
    pub entry: String,
    /// Everything the compiler printed, verbatim: the fallback for whatever
    /// does not read as a diagnostic.
    pub output: String,
}

/// Every entry of one asset the compiler rejected, and what it said.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileFailure {
    /// The asset, as a message names it (`Shader 'water'`).
    pub owner: String,
    /// Each rejected entry, in job order.
    pub failures: Vec<EntryFailure>,
    /// Every located error and warning across the failures, each once, in the
    /// order first reported. Several entries compile one authored file, so the
    /// same error arrives once per entry.
    pub diagnostics: Vec<Diagnostic>,
    /// What the asset adds for this failure, such as the signature of a hook
    /// the file never defined, or empty.
    pub hint: &'static str,
}

impl CompileFailure {
    /// The failure for `failures`, with `hint` asked of each one's output and
    /// the first answer kept.
    pub(crate) fn new(
        owner: &str,
        failures: Vec<EntryFailure>,
        hint: impl Fn(&str) -> &'static str,
    ) -> Self {
        let diagnostics =
            diagnostics::dedup(failures.iter().flat_map(|f| diagnostics::parse(&f.output)));
        let hint = failures
            .iter()
            .map(|f| hint(&f.output))
            .find(|h| !h.is_empty())
            .unwrap_or("");
        Self {
            owner: owner.to_string(),
            failures,
            diagnostics,
            hint,
        }
    }

    /// The errors among [`diagnostics`](Self::diagnostics).
    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
    }
}

impl fmt::Display for CompileFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: compiling ", self.owner)?;
        for (i, failure) in self.failures.iter().enumerate() {
            let sep = if i == 0 { "" } else { ", " };
            write!(f, "{sep}'{}'", failure.entry)?;
        }
        f.write_str(":")?;
        for diagnostic in &self.diagnostics {
            write!(f, "\n{diagnostic}")?;
        }
        // An entry whose output located no error failed for a reason only its
        // raw output says.
        for failure in &self.failures {
            let located = diagnostics::parse(&failure.output)
                .iter()
                .any(|d| d.severity == Severity::Error);
            if !located {
                write!(f, "\n{}", failure.output.trim_end())?;
            }
        }
        f.write_str(self.hint)
    }
}

impl std::error::Error for CompileFailure {}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "hlsl: dxc failed for main_bindless.hlsl:\n";
    const TYPO: &str =
        "shaders/water.hlsl:4:5: error: unknown type name 'flaot'\n    flaot x;\n    ^\n";
    const WARNING: &str = "shaders/water.hlsl:2:9: warning: implicit truncation of vector type\n";

    fn failure(entry: &str, output: String) -> EntryFailure {
        EntryFailure {
            entry: entry.to_string(),
            output,
        }
    }

    // Both entries compile the same file, so its error arrives twice and is
    // reported once, with the warning beside it.
    #[test]
    fn one_error_in_every_entry_is_reported_once() {
        let failed = CompileFailure::new(
            "Shader 'water'",
            vec![
                failure("vertex_main", format!("{HEADER}{WARNING}{TYPO}")),
                failure("fragment_main", format!("{HEADER}{TYPO}{WARNING}")),
            ],
            |_| "",
        );
        let located: Vec<(&str, u32, Severity)> = failed
            .diagnostics
            .iter()
            .map(|d| (d.path.as_str(), d.line, d.severity))
            .collect();
        assert_eq!(
            located,
            [
                ("shaders/water.hlsl", 2, Severity::Warning),
                ("shaders/water.hlsl", 4, Severity::Error),
            ]
        );
        assert_eq!(failed.errors().count(), 1);
        assert_eq!(
            failed.to_string(),
            "Shader 'water': compiling 'vertex_main', 'fragment_main':\n\
             shaders/water.hlsl:2:9: warning: implicit truncation of vector type\n\
             shaders/water.hlsl:4:5: error: unknown type name 'flaot'\n    flaot x;\n    ^"
        );
    }

    // Output that locates no error is shown as the compiler printed it, and
    // the first hint any output asks for ends the message.
    #[test]
    fn unlocated_output_falls_back_to_the_raw_text_and_the_hint_follows() {
        let failed = CompileFailure::new(
            "Shader 'water'",
            vec![
                failure("vertex_main", format!("{HEADER}{TYPO}")),
                failure(
                    "fragment_main",
                    "hlsl: spirv-cross failed: bad\n".to_string(),
                ),
            ],
            |out| {
                if out.contains("spirv-cross") {
                    "\nhint"
                } else {
                    ""
                }
            },
        );
        assert_eq!(failed.hint, "\nhint");
        let message = failed.to_string();
        assert!(message.contains("unknown type name 'flaot'"), "{message}");
        assert!(
            message.ends_with("\nhlsl: spirv-cross failed: bad\nhint"),
            "{message}"
        );
        assert!(
            !message.contains(HEADER),
            "a located error needs no raw text"
        );
    }

    // A failure reaches an asset's build as an io error that still carries
    // the typed failure.
    #[test]
    fn a_failure_converts_to_an_io_error_keeping_the_type() {
        let failed = CompileFailure::new(
            "Shader 'water'",
            vec![failure("fragment_main", TYPO.to_string())],
            |_| "",
        );
        let io: std::io::Error = ProgramError::from(failed.clone()).into();
        assert_eq!(io.kind(), std::io::ErrorKind::InvalidData);
        let inner = io
            .get_ref()
            .and_then(|e| e.downcast_ref::<ProgramError>())
            .expect("the typed error rides inside");
        assert!(matches!(inner, ProgramError::Compile(f) if *f == failed));
        assert_eq!(io.to_string(), failed.to_string());
    }
}

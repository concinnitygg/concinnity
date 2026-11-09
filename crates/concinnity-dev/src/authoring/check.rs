//! Validate a world JSONL without producing blob files.
//!
//! Runs the validation front half of the build pipeline (load, expand, and
//! the semantic checks in `crate::check`) and reports the outcome. Used by
//! `cn check`.

use concinnity_cook::Diagnostic;
use std::path::Path;

/// Read `world_path`, run validation, and report results. Returns Ok if every
/// asset passes; otherwise an error whose Display contains a human-readable
/// summary of every failure (one per asset).
pub(crate) fn check_at_path(world_path: &Path) -> std::io::Result<()> {
    let content = std::fs::read_to_string(world_path)?;
    let source = concinnity_cook::WorldSource::file(&content, world_path);
    check_source(source, &world_path.display().to_string())
}

/// Run validation against world text. `label` is the origin used in messages
/// (typically the source path).
fn check_source(source: concinnity_cook::WorldSource<'_>, label: &str) -> std::io::Result<()> {
    match concinnity_cook::prepare_world(source, crate::project::assets_dir().as_deref()) {
        Ok(loaded) => {
            println!("ok: {} asset(s) passed in {}", loaded.assets.len(), label);
            Ok(())
        }
        Err(errors) => Err(report_validation_errors(errors.diagnostics())),
    }
}

/// Print each validation error in CLI form and collapse them into a single
/// `io::Error` naming the count, so a failed world surfaces every problem in one
/// pass.
pub(crate) fn report_validation_errors(errors: &[Diagnostic]) -> std::io::Error {
    for e in errors {
        eprintln!("error:   {e}");
    }
    eprintln!("\nvalidation failed ({} error(s))", errors.len());
    validation_failed(errors)
}

/// The error a world that failed validation with `errors` returns, once they
/// have been shown. It carries them, for a tool that points at what failed.
pub(crate) fn validation_failed(errors: &[Diagnostic]) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        ValidationFailed(errors.to_vec()),
    )
}

/// A world that failed validation: every problem found, in the order the
/// checks reported them.
#[derive(Debug)]
pub(crate) struct ValidationFailed(Vec<Diagnostic>);

impl ValidationFailed {
    /// The validation failure `e` carries, if it is one.
    pub(crate) fn of(e: &std::io::Error) -> Option<&Self> {
        e.get_ref()?.downcast_ref()
    }

    /// Every problem found.
    pub(crate) fn diagnostics(&self) -> &[Diagnostic] {
        &self.0
    }
}

impl std::fmt::Display for ValidationFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "validation failed with {} error(s)", self.0.len())
    }
}

impl std::error::Error for ValidationFailed {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_validation_errors_names_the_count() {
        let errors = vec![Diagnostic::from("first"), Diagnostic::from("second")];
        let err = report_validation_errors(&errors);
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("2 error(s)"), "got: {err}");
    }

    #[test]
    fn check_source_accepts_a_valid_world() {
        check_source("[\"PhysicsConfig\",{\"$id\":\"phys\"}]\n".into(), "test").unwrap();
    }

    #[test]
    fn check_source_rejects_an_unknown_type() {
        let err =
            check_source("[\"NotARealAssetType\",{\"$id\":\"odd\"}]\n".into(), "test").unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn check_at_path_reports_a_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.jsonl");
        assert!(check_at_path(&path).is_err());
    }

    #[test]
    fn check_at_path_accepts_a_valid_world_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world.jsonl");
        std::fs::write(&path, "[\"PhysicsConfig\",{\"$id\":\"phys\"}]\n").unwrap();
        check_at_path(&path).unwrap();
    }
}

//! Validate a world JSONL without producing blob files.
//!
//! Runs the validation front half of the build pipeline (load, expand, and
//! the semantic checks in `crate::check`) and reports the outcome. Used by
//! `cn test`.

/// Read `world_path`, run validation, and report results. Returns Ok if every
/// asset passes; otherwise an error whose Display contains a human-readable
/// summary of every failure (one per asset).
pub(crate) fn check_at_path(world_path: &str) -> std::io::Result<()> {
    let content = std::fs::read_to_string(world_path)?;
    check_from_str(&content, world_path)
}

/// Run validation against an in-memory world JSONL string. `label` is the
/// origin used in messages (typically the source path).
fn check_from_str(content: &str, label: &str) -> std::io::Result<()> {
    match concinnity_cook::prepare_world(content, crate::project::assets_dir().as_deref()) {
        Ok(loaded) => {
            println!("ok: {} asset(s) passed in {}", loaded.assets.len(), label);
            Ok(())
        }
        Err(errors) => Err(report_validation_errors(&errors)),
    }
}

/// Print each validation error in CLI form and collapse them into a single
/// io::Error naming the count, so a failed world surfaces every problem in one
/// pass.
pub(crate) fn report_validation_errors(errors: &[String]) -> std::io::Error {
    for e in errors {
        eprintln!("error:   {}", e);
    }
    eprintln!("\nvalidation failed ({} error(s))", errors.len());
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("validation failed with {} error(s)", errors.len()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_validation_errors_names_the_count() {
        let errors = vec!["first".to_string(), "second".to_string()];
        let err = report_validation_errors(&errors);
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("2 error(s)"), "got: {err}");
    }

    #[test]
    fn check_from_str_accepts_a_valid_world() {
        check_from_str(
            "{\"type\":\"PhysicsConfig\",\"args\":{\"$id\":\"phys\"}}\n",
            "test",
        )
        .unwrap();
    }

    #[test]
    fn check_from_str_rejects_an_unknown_type() {
        let err = check_from_str(
            "{\"type\":\"NotARealAssetType\",\"args\":{\"$id\":\"odd\"}}\n",
            "test",
        )
        .unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn check_at_path_reports_a_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.jsonl");
        assert!(check_at_path(path.to_str().unwrap()).is_err());
    }

    #[test]
    fn check_at_path_accepts_a_valid_world_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world.jsonl");
        std::fs::write(
            &path,
            "{\"type\":\"PhysicsConfig\",\"args\":{\"$id\":\"phys\"}}\n",
        )
        .unwrap();
        check_at_path(path.to_str().unwrap()).unwrap();
    }
}

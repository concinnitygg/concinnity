//! Which vendor directory a tool works in: the workspace's own `vendor/` when
//! it runs inside a Cargo workspace, so what a checkout uses is a property of
//! the checkout, and otherwise one per-user directory shared by every project.

use std::fs;
use std::path::{Path, PathBuf};

/// The vendor directory for a tool started in `cwd`: `vendor/` beside the
/// manifest of the nearest enclosing Cargo workspace, else `concinnity/vendor`
/// under `per_user_base`. `None` when neither exists.
pub(crate) fn vendor_root(cwd: &Path, per_user_base: Option<&Path>) -> Option<PathBuf> {
    workspace_root(cwd)
        .map(|root| root.join("vendor"))
        .or_else(|| per_user_base.map(|base| base.join("concinnity").join("vendor")))
}

// The nearest ancestor of `start` whose manifest declares a workspace.
fn workspace_root(start: &Path) -> Option<&Path> {
    start.ancestors().find(|dir| {
        fs::read_to_string(dir.join("Cargo.toml"))
            .is_ok_and(|manifest| declares_workspace(&manifest))
    })
}

// Whether a manifest has a `[workspace]` table.
fn declares_workspace(manifest: &str) -> bool {
    manifest.lines().any(|line| {
        line.split('#')
            .next()
            .is_some_and(|code| code.trim() == "[workspace]")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_workspace_keeps_its_own_vendor_directory() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("checkout");
        let member = root.join("crates").join("app");
        fs::create_dir_all(&member).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/*\"]\n",
        )
        .unwrap();
        fs::write(member.join("Cargo.toml"), "[package]\nname = \"app\"\n").unwrap();
        let base = dir.path().join("user");
        assert_eq!(vendor_root(&member, Some(&base)), Some(root.join("vendor")));
    }

    #[test]
    fn outside_a_workspace_the_per_user_directory_is_used() {
        let dir = tempfile::tempdir().unwrap();
        let project = dir.path().join("game");
        fs::create_dir_all(&project).unwrap();
        fs::write(project.join("Cargo.toml"), "[package]\nname = \"game\"\n").unwrap();
        let base = dir.path().join("user");
        assert_eq!(
            vendor_root(&project, Some(&base)),
            Some(base.join("concinnity").join("vendor"))
        );
        assert_eq!(vendor_root(&project, None), None);
    }

    #[test]
    fn only_a_workspace_table_marks_a_workspace() {
        assert!(declares_workspace("[package]\n\n[workspace]\n"));
        assert!(declares_workspace("[workspace]  # the checkout\n"));
        assert!(!declares_workspace(
            "[workspace.dependencies]\nfoo = \"1\"\n"
        ));
        assert!(!declares_workspace("# [workspace]\n"));
    }
}

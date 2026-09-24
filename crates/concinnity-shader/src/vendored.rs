//! Releases unpacked under a workspace's `vendor/` directory.
//!
//! `scripts/vendor.py` leaves each one at `vendor/<name>-<version>-<slug>/`,
//! where `<slug>` is the `<os>-<arch>` it was built for.

use std::path::{Path, PathBuf};

/// Every `<name>-<version>-<slug>` directory under `vendor`, newest first.
///
/// Versions compare by component rather than by string: 1.10.0 is newer than
/// 1.9.0, which sorts the other way lexically. An absent `vendor` holds none.
#[must_use]
pub fn vendored_releases(vendor: &Path, name: &str, slug: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(vendor) else {
        return Vec::new();
    };
    let mut found: Vec<(Vec<u32>, PathBuf)> = entries
        .flatten()
        .filter_map(|entry| {
            let file_name = entry.file_name();
            let version = release_version(file_name.to_str()?, name, slug)?;
            entry.path().is_dir().then_some((version, entry.path()))
        })
        .collect();
    found.sort_by(|a, b| b.0.cmp(&a.0));
    found.into_iter().map(|(_, path)| path).collect()
}

// The dotted version of a `<name>-<version>-<slug>` directory, as components.
fn release_version(dir: &str, name: &str, slug: &str) -> Option<Vec<u32>> {
    let version = dir
        .strip_prefix(name)?
        .strip_prefix('-')?
        .strip_suffix(slug)?
        .strip_suffix('-')?;
    let parts: Option<Vec<u32>> = version.split('.').map(|p| p.parse().ok()).collect();
    parts.filter(|p| !p.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_testing::TempTree;

    fn names(found: &[PathBuf]) -> Vec<String> {
        found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect()
    }

    // The pin is the point of vendoring, so a tree holding several releases
    // has to lead with the newest -- and "newer" is a numeric compare, which a
    // plain string sort gets backwards past a minor of 9.
    #[test]
    fn the_newest_release_leads() {
        let tree = TempTree::new();
        tree.dir("dxc-1.9.2607-linux-x86_64");
        tree.dir("dxc-1.10.2605-linux-x86_64");
        tree.dir("dxc-1.8.2505-linux-x86_64");

        let found = vendored_releases(tree.path(), "dxc", "linux-x86_64");
        assert_eq!(
            names(&found),
            [
                "dxc-1.10.2605-linux-x86_64",
                "dxc-1.9.2607-linux-x86_64",
                "dxc-1.8.2505-linux-x86_64",
            ]
        );
    }

    // Every release vendors into the same directory, so a scan that matched on
    // the version alone would hand one component's root to another.
    #[test]
    fn another_components_or_platforms_release_is_not_a_match() {
        let tree = TempTree::new();
        tree.dir("xess-3.0.1-windows-x86_64");
        tree.dir("streamline-2.11.1-windows-x86_64");
        tree.dir("xess-3.0.1-linux-x86_64");

        let found = vendored_releases(tree.path(), "xess", "windows-x86_64");
        assert_eq!(names(&found), ["xess-3.0.1-windows-x86_64"]);
    }

    // `fidelityfx` and `fidelityfx-vk` share a pin and sit side by side: a
    // name that merely starts with another's is a different component.
    #[test]
    fn a_longer_name_sharing_the_prefix_is_not_a_match() {
        let tree = TempTree::new();
        tree.dir("fidelityfx-1.1.4-windows-x86_64");
        tree.dir("fidelityfx-vk-1.1.4-windows-x86_64");

        let found = |name| names(&vendored_releases(tree.path(), name, "windows-x86_64"));
        assert_eq!(found("fidelityfx"), ["fidelityfx-1.1.4-windows-x86_64"]);
        assert_eq!(
            found("fidelityfx-vk"),
            ["fidelityfx-vk-1.1.4-windows-x86_64"]
        );
    }

    // A file named like a release is not one.
    #[test]
    fn only_directories_are_releases() {
        let tree = TempTree::new();
        tree.write("dxc-1.9.2607-linux-x86_64", b"");
        assert!(vendored_releases(tree.path(), "dxc", "linux-x86_64").is_empty());
    }

    // Nothing vendored is the ordinary case on a host that installed its
    // tools, and `vendor/` may not exist at all.
    #[test]
    fn an_absent_or_empty_vendor_directory_yields_nothing() {
        let tree = TempTree::new();
        assert!(vendored_releases(&tree.join("absent"), "dxc", "linux-x86_64").is_empty());
        assert!(vendored_releases(tree.path(), "dxc", "linux-x86_64").is_empty());
    }

    #[test]
    fn a_release_directory_parses_to_version_components() {
        let parse = |d| release_version(d, "dxc", "linux-x86_64");
        assert_eq!(parse("dxc-1.9.2607-linux-x86_64"), Some(vec![1, 9, 2607]));
        assert_eq!(parse("dxc-3-linux-x86_64"), Some(vec![3]));
        assert_eq!(parse("dxc-linux-x86_64"), None);
        assert_eq!(parse("dxc-1.9.2607-windows-x86_64"), None);
        assert_eq!(parse("dxc-main-linux-x86_64"), None);
    }
}

//! Graphics SDKs unpacked under the workspace's `vendor/` directory.
//!
//! A vendoring run leaves each SDK at `vendor/<name>-<version>-<os>-<arch>/`,
//! unpacked exactly as its vendor ships it, so the paths the SDK setup joins
//! onto a root are the same whether the root came from here, from the
//! environment, or from a hand install. Resolution prefers an explicit
//! `CN_<VENDOR>_SDK` over anything found here, and falls back past here to the
//! default install path.
//!
//! The scan is its own module rather than shared with `concinnity-slang`'s: that
//! one carries a compiler's version floor and a PATH fallback, neither of which
//! means anything for a directory of DLLs, and the crates cannot depend on each
//! other in the direction that would let one call the other.

use std::path::{Path, PathBuf};

/// The `<os>-<arch>` suffix vendored SDK directories carry. Every one of these
/// is Windows-only, so there is one.
const SLUG: &str = "windows-x86_64";

/// The newest vendored release of `name` under `workspace`, if any.
pub(crate) fn newest(workspace: Option<&Path>, name: &str) -> Option<PathBuf> {
    newest_in(&workspace?.join("vendor"), name, SLUG)
}

// Split out so the scan is testable against a synthetic tree rather than
// whatever the running machine happens to vendor.
fn newest_in(vendor: &Path, name: &str, slug: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(vendor).ok()?;
    entries
        .flatten()
        .filter_map(|entry| {
            let file_name = entry.file_name();
            let version = release_version(file_name.to_str()?, name, slug)?;
            entry.path().is_dir().then_some((version, entry.path()))
        })
        // By component rather than by string: 1.619.3 is newer than 1.9.0,
        // which sorts the other way lexically.
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, path)| path)
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

    fn vendored(tree: &TempTree, dir: &str) {
        tree.dir(dir);
    }

    // The pin is the point of vendoring, so a tree holding two releases has to
    // resolve the newer one -- and "newer" is a numeric compare, which a plain
    // string sort gets backwards past a minor of 9.
    #[test]
    fn the_newest_release_wins() {
        let tree = TempTree::new();
        vendored(&tree, "agility-1.9.0-windows-x86_64");
        vendored(&tree, "agility-1.619.3-windows-x86_64");

        let found = newest_in(tree.path(), "agility", "windows-x86_64");
        assert_eq!(
            found.as_deref().and_then(Path::file_name),
            Some("agility-1.619.3-windows-x86_64".as_ref())
        );
    }

    // Every SDK vendors into the same directory, so a scan that matched on the
    // version alone would hand the Agility root to the XeSS setup.
    #[test]
    fn another_components_release_is_not_a_match() {
        let tree = TempTree::new();
        vendored(&tree, "xess-3.0.1-windows-x86_64");
        vendored(&tree, "streamline-2.11.1-windows-x86_64");

        let found = newest_in(tree.path(), "xess", "windows-x86_64");
        assert_eq!(
            found.as_deref().and_then(Path::file_name),
            Some("xess-3.0.1-windows-x86_64".as_ref())
        );
    }

    // A name that merely starts with another's is a different component, and
    // matching it would resolve a root the build cannot use.
    #[test]
    fn a_longer_name_sharing_the_prefix_is_not_a_match() {
        let tree = TempTree::new();
        vendored(&tree, "xess-fg-3.0.1-windows-x86_64");

        assert!(newest_in(tree.path(), "xess", "windows-x86_64").is_none());
    }

    // `fidelityfx` and `fidelityfx-vk` share a pin and sit side by side, and
    // only the latter holds a Vulkan runtime: resolving one as the other hands
    // a root whose payload is not there.
    #[test]
    fn the_fidelityfx_pair_resolves_apart() {
        let tree = TempTree::new();
        vendored(&tree, "fidelityfx-1.1.4-windows-x86_64");
        vendored(&tree, "fidelityfx-vk-1.1.4-windows-x86_64");

        let name = |component| {
            newest_in(tree.path(), component, "windows-x86_64")
                .as_deref()
                .and_then(Path::file_name)
                .map(|n| n.to_string_lossy().into_owned())
        };
        assert_eq!(
            name("fidelityfx").as_deref(),
            Some("fidelityfx-1.1.4-windows-x86_64")
        );
        assert_eq!(
            name("fidelityfx-vk").as_deref(),
            Some("fidelityfx-vk-1.1.4-windows-x86_64")
        );
    }

    // Nothing vendored is the ordinary case on a host that installed its SDKs,
    // and `vendor/` may not exist at all.
    #[test]
    fn an_absent_or_empty_vendor_directory_yields_nothing() {
        let tree = TempTree::new();
        assert!(newest_in(&tree.join("absent"), "xess", "windows-x86_64").is_none());
        assert!(newest_in(tree.path(), "xess", "windows-x86_64").is_none());
    }

    // Without a workspace above it there is nothing to scan, which is what a
    // build from a registry copy sees.
    #[test]
    fn no_workspace_yields_nothing() {
        assert!(newest(None, "xess").is_none());
    }

    #[test]
    fn a_release_directory_parses_to_version_components() {
        let parse = |d| release_version(d, "xess", "windows-x86_64");
        assert_eq!(parse("xess-3.0.1-windows-x86_64"), Some(vec![3, 0, 1]));
        assert_eq!(parse("xess-3-windows-x86_64"), Some(vec![3]));
        assert_eq!(parse("xess-windows-x86_64"), None);
        assert_eq!(parse("xess-3.0.1-linux-x86_64"), None);
        assert_eq!(parse("xess-main-windows-x86_64"), None);
    }
}

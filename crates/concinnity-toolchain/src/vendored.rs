//! Graphics SDKs unpacked under the workspace's `vendor/` directory.
//!
//! A vendoring run leaves each SDK at `vendor/<name>-<version>-<os>-<arch>/`,
//! unpacked exactly as its vendor ships it, so the paths the SDK setup joins
//! onto a root are the same whether the root came from here, from the
//! environment, or from a hand install. Resolution prefers an explicit
//! `CN_<VENDOR>_SDK` over anything found here, and finds nothing past here:
//! there is no guessed install path to fall back to.

use std::path::{Path, PathBuf};

/// The `<os>-<arch>` suffix vendored SDK directories carry. Every one of these
/// is Windows-only, so there is one.
const SLUG: &str = "windows-x86_64";

/// The newest vendored release of `name` under `workspace`, if any.
pub(crate) fn newest(workspace: Option<&Path>, name: &str) -> Option<PathBuf> {
    concinnity_shader::vendored_releases(&workspace?.join("vendor"), name, SLUG)
        .into_iter()
        .next()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Without a workspace above it there is nothing to scan, which is what a
    // build from a registry copy sees.
    #[test]
    fn no_workspace_yields_nothing() {
        assert!(newest(None, "xess").is_none());
    }
}

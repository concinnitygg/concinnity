//! Where to look for slangc, in order.
//!
//! `$CN_SLANG_SDK`, then `slang/` beside the running executable, then the
//! workspace's vendored releases, then PATH, then `$VULKAN_SDK/bin`. The
//! environment variable is the same override every other third-party SDK takes
//! (`CN_AGILITY_SDK`, `CN_XESS_SDK`, ...), and leads for the same reason: an
//! explicit answer beats a discovered one.
//!
//! The two after it are the same idea for the two ways this code ships. An
//! install carries its compiler in `slang/` beside the binaries, so it works
//! with nothing set up; a checkout carries a pinned release under `vendor/`, so
//! a given revision compiles its shaders with a given compiler rather than with
//! whatever an individual machine installed. Both beat PATH, because both are
//! the answer their own distribution shipped. Neither exists for a consumer
//! building from a registry copy, which starts at PATH.

use std::path::{Path, PathBuf};

/// The slangc file name on this platform.
pub(crate) const EXE: &str = if cfg!(windows) {
    "slangc.exe"
} else {
    "slangc"
};

// WORKSPACE_ROOT: the engine checkout this crate was built from, and `None` for
// a registry copy.
include!(concat!(env!("OUT_DIR"), "/workspace_root.rs"));

/// The environment variable naming a Slang install, matching the
/// `<VENDOR>_SDK_ROOT` shape the graphics SDKs use.
pub(crate) const ROOT_VAR: &str = "CN_SLANG_SDK";

/// The directory an install puts its Slang release in, beside the binaries.
const BUNDLED_DIR: &str = "slang";

/// Every slangc to try, in resolution order.
///
/// Resolution runs on the host when this crate is a build dependency and on the
/// target when it is linked into the runtime; `std::env::consts` names the right
/// machine either way, so a cross-build looks for the release its output will
/// run beside rather than the one that built it.
pub(crate) fn slangc_candidates() -> Vec<PathBuf> {
    let vendored = match WORKSPACE_ROOT {
        Some(root) => vendored_in(&Path::new(root).join("vendor"), host_slug()),
        None => Vec::new(),
    };
    candidates(
        std::env::var(ROOT_VAR).ok().as_deref(),
        exe_dir().as_deref(),
        vendored,
        std::env::var("VULKAN_SDK").ok().as_deref(),
    )
}

// Split from `slangc_candidates` so the order is testable without reading the
// process environment, the running machine's `vendor/`, or where the test
// binary happens to sit.
fn candidates(
    slang_root: Option<&str>,
    exe_dir: Option<&Path>,
    vendored: Vec<PathBuf>,
    vulkan_sdk: Option<&str>,
) -> Vec<PathBuf> {
    let bin = |root: &Path| root.join("bin").join(EXE);
    let mut found = Vec::new();
    found.extend(
        slang_root
            .filter(|r| !r.is_empty())
            .map(|r| bin(Path::new(r))),
    );
    found.extend(exe_dir.map(|dir| bin(&dir.join(BUNDLED_DIR))));
    found.extend(vendored);
    found.push(PathBuf::from(EXE));
    found.extend(
        vulkan_sdk
            .filter(|s| !s.is_empty())
            .map(|s| bin(Path::new(s))),
    );
    found
}

// The directory holding the running executable. A build script gets its own
// binary's directory under `target/`, which carries no `slang/`, so the
// candidate costs a failed stat there and nothing more.
fn exe_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.to_path_buf())
}

// The `<os>-<arch>` half of a release directory name. `None` on a platform
// Slang publishes no build for.
fn host_slug() -> Option<&'static str> {
    Some(match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "macos-aarch64",
        ("macos", "x86_64") => "macos-x86_64",
        ("linux", "aarch64") => "linux-aarch64",
        ("linux", "x86_64") => "linux-x86_64",
        ("windows", "aarch64") => "windows-aarch64",
        ("windows", "x86_64") => "windows-x86_64",
        _ => return None,
    })
}

// Split out so the scan is testable against a synthetic tree rather than
// whatever the running machine happens to vendor.
fn vendored_in(vendor: &Path, slug: Option<&str>) -> Vec<PathBuf> {
    let Some(slug) = slug else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(vendor) else {
        return Vec::new();
    };
    let mut found: Vec<(Vec<u32>, PathBuf)> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            let version = release_version(name.to_str()?, slug)?;
            let exe = entry.path().join("bin").join(EXE);
            exe.is_file().then_some((version, exe))
        })
        .collect();
    // Descending, and by component rather than by string: 2026.16.1 is newer
    // than 2026.9.1, which sorts the other way lexically.
    found.sort_by(|a, b| b.0.cmp(&a.0));
    found.into_iter().map(|(_, exe)| exe).collect()
}

// The dotted version of a `slang-<version>-<slug>` directory, as components.
fn release_version(name: &str, slug: &str) -> Option<Vec<u32>> {
    let version = name
        .strip_prefix("slang-")?
        .strip_suffix(slug)?
        .strip_suffix('-')?;
    let parts: Option<Vec<u32>> = version.split('.').map(|p| p.parse().ok()).collect();
    parts.filter(|p| !p.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_testing::TempTree;

    fn release(tree: &TempTree, name: &str) {
        tree.write(&format!("{name}/bin/{EXE}"), b"");
    }

    fn names(found: &[PathBuf]) -> Vec<String> {
        found
            .iter()
            .filter_map(|p| p.parent()?.parent()?.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .collect()
    }

    // The point of vendoring is a pinned compiler, so a checkout holding two
    // releases has to resolve the newer one -- and "newer" is a numeric
    // compare, which a plain string sort gets backwards past a minor of 9.
    #[test]
    fn the_newest_release_for_this_platform_wins() {
        let tree = TempTree::new();
        release(&tree, "slang-2026.9.1-linux-x86_64");
        release(&tree, "slang-2026.16.1-linux-x86_64");
        release(&tree, "slang-2026.2-linux-x86_64");

        assert_eq!(
            names(&vendored_in(tree.path(), Some("linux-x86_64"))),
            [
                "slang-2026.16.1-linux-x86_64",
                "slang-2026.9.1-linux-x86_64",
                "slang-2026.2-linux-x86_64",
            ]
        );
    }

    // Every platform's release unpacks under the same `bin/slangc`, and running
    // one host's binary on another fails in a way the version probe cannot
    // explain.
    #[test]
    fn another_platforms_release_is_not_a_candidate() {
        let tree = TempTree::new();
        release(&tree, "slang-2026.16.1-macos-aarch64");
        release(&tree, "slang-2026.16.1-windows-x86_64");

        assert_eq!(
            names(&vendored_in(tree.path(), Some("macos-aarch64"))),
            ["slang-2026.16.1-macos-aarch64"]
        );
    }

    // `vendor/` holds whatever else a checkout needs, and a neighbour can share
    // the prefix without being a release for this platform.
    #[test]
    fn unrelated_vendor_entries_are_ignored() {
        let tree = TempTree::new();
        release(&tree, "slang-2026.16.1-linux-x86_64");
        release(&tree, "slang");
        release(&tree, "slang-main-linux-x86_64");

        assert_eq!(
            names(&vendored_in(tree.path(), Some("linux-x86_64"))),
            ["slang-2026.16.1-linux-x86_64"]
        );
    }

    // A release directory with no `bin/slangc` under it is a half-finished
    // download, not a compiler: naming it as a candidate turns a clear "not
    // found" into a launch failure.
    #[test]
    fn a_release_without_the_binary_is_not_a_candidate() {
        let tree = TempTree::new();
        tree.write("slang-2026.16.1-linux-x86_64/lib/libslang.so", b"");

        assert!(vendored_in(tree.path(), Some("linux-x86_64")).is_empty());
    }

    // Slang publishes no build for every platform the engine can target, and a
    // host it does not cover falls through to PATH rather than scanning for a
    // directory name that can never exist.
    #[test]
    fn an_unpublished_platform_yields_no_candidates() {
        let tree = TempTree::new();
        release(&tree, "slang-2026.16.1-linux-x86_64");

        assert!(vendored_in(tree.path(), None).is_empty());
    }

    // A checkout that vendors nothing is the ordinary case, and the directory
    // itself may not exist at all.
    #[test]
    fn a_missing_vendor_directory_is_not_an_error() {
        let tree = TempTree::new();

        assert!(vendored_in(&tree.join("absent"), Some("linux-x86_64")).is_empty());
    }

    // The override exists so a host can point at an install neither vendored
    // nor on PATH, which only works if it is tried before both.
    // An install ships its compiler beside its binaries and a checkout vendors
    // one; both must beat PATH, or a machine with an unrelated slangc installed
    // silently compiles with that instead of what it was shipped.
    #[test]
    fn every_source_is_tried_in_order() {
        let vendored = vec![PathBuf::from("/checkout/vendor/slang-1.0-x/bin/slangc")];
        let found = candidates(
            Some("/opt/slang"),
            Some(Path::new("/install")),
            vendored.clone(),
            Some("/opt/vk"),
        );

        assert_eq!(
            found,
            [
                Path::new("/opt/slang").join("bin").join(EXE),
                Path::new("/install").join("slang").join("bin").join(EXE),
                vendored[0].clone(),
                PathBuf::from(EXE),
                Path::new("/opt/vk").join("bin").join(EXE),
            ]
        );
    }

    // An unset variable must not become a candidate rooted at the empty path,
    // which resolves to `bin/slangc` relative to the working directory and
    // would make resolution depend on where cargo was invoked from.
    #[test]
    fn an_unset_or_empty_override_adds_nothing() {
        for empty in [None, Some("")] {
            let found = candidates(empty, None, Vec::new(), empty);

            assert_eq!(found, [PathBuf::from(EXE)], "{empty:?}");
        }
    }

    #[test]
    fn a_release_directory_parses_to_version_components() {
        assert_eq!(
            release_version("slang-2026.16.1-macos-aarch64", "macos-aarch64"),
            Some(vec![2026, 16, 1])
        );
        assert_eq!(
            release_version("slang-2026.2-macos-aarch64", "macos-aarch64"),
            Some(vec![2026, 2])
        );
        assert_eq!(
            release_version("slang-macos-aarch64", "macos-aarch64"),
            None
        );
        assert_eq!(
            release_version("slang-2026.16.1-macos-aarch64", "linux-x86_64"),
            None
        );
        assert_eq!(release_version("slangc", "macos-aarch64"), None);
    }
}

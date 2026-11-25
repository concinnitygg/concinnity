//! Where to look for `dxc`, in order.
//!
//! `dxc/` beside the running executable, then the releases vendored in the
//! directory the dev CLI named, then those under the checkout's `vendor/`, then
//! PATH, then `$VULKAN_SDK/bin`. `concinnity vendor` pins one release, so a
//! given revision compiles its shaders with a given compiler rather than with
//! whatever a machine installed. Each beats PATH, because each is the answer
//! nearest to the thing being run.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The dxc file name on this platform.
pub(crate) const EXE: &str = if cfg!(windows) { "dxc.exe" } else { "dxc" };

// WORKSPACE_ROOT: the engine checkout this crate was built from, and `None` for
// a registry copy.
include!(concat!(env!("OUT_DIR"), "/workspace_root.rs"));

/// The directory a release stages its dxc in, beside the binaries.
const BUNDLED_DIR: &str = "dxc";

/// The name `concinnity vendor` gives a dxc release directory, before `-<version>-<slug>`.
const VENDORED_NAME: &str = "dxc";

static VENDOR_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Look for dxc among the releases vendored in `vendor` before any other
/// vendored release, for the life of the process. Call it before the first
/// compile; only the first call counts.
pub fn use_vendor_dir(vendor: &Path) {
    let _first = VENDOR_DIR.set(vendor.to_path_buf());
}

/// Every vendored dxc for this machine under `vendor`, newest first.
pub fn vendored_dxc(vendor: &Path) -> Vec<PathBuf> {
    host_slug().map_or_else(Vec::new, |slug| vendored_in(vendor, slug))
}

/// Every dxc to try, in resolution order.
///
/// Resolution runs on the machine doing the compile, which
/// `std::env::consts` names.
#[expect(
    clippy::disallowed_methods,
    reason = "the Vulkan SDK installer publishes its location only as VULKAN_SDK"
)]
pub(crate) fn dxc_candidates() -> Vec<PathBuf> {
    let named = VENDOR_DIR.get().map(PathBuf::as_path);
    let workspace = WORKSPACE_ROOT.map(|root| Path::new(root).join("vendor"));
    let mut vendored = named.map_or_else(Vec::new, vendored_dxc);
    if workspace.as_deref() != named {
        vendored.extend(workspace.as_deref().map_or_else(Vec::new, vendored_dxc));
    }
    candidates(
        exe_dir().as_deref(),
        vendored,
        std::env::var("VULKAN_SDK").ok().as_deref(),
    )
}

// Split from `dxc_candidates` so the order is testable without reading the
// process environment, the running machine's `vendor/`, or where the test
// binary happens to sit.
fn candidates(
    exe_dir: Option<&Path>,
    vendored: Vec<PathBuf>,
    vulkan_sdk: Option<&str>,
) -> Vec<PathBuf> {
    let bin = |root: &Path| root.join("bin").join(EXE);
    let mut found = Vec::new();
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

// The directory holding the running executable. A test binary's directory
// under `target/` carries no `dxc/`, so the candidate costs a failed probe there
// and nothing more.
fn exe_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.to_path_buf())
}

// The `<os>-<arch>` half of a release directory name. `None` on a platform
// `concinnity vendor` vendors no dxc for.
fn host_slug() -> Option<&'static str> {
    Some(match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => "macos-aarch64",
        ("linux", "x86_64") => "linux-x86_64",
        ("windows", "x86_64") => "windows-x86_64",
        _ => return None,
    })
}

// Where a vendored release keeps its executable: Microsoft's Windows archive
// ships one per architecture under `bin/<arch>/`, every other layout `bin/`.
fn vendored_exe(release: &Path, slug: &str) -> PathBuf {
    let bin = release.join("bin");
    if slug.starts_with("windows") {
        bin.join("x64").join(EXE)
    } else {
        bin.join(EXE)
    }
}

// Every vendored release's executable for `slug`, newest first. Split out so
// the scan is testable against a synthetic tree rather than whatever the
// running machine happens to vendor.
fn vendored_in(vendor: &Path, slug: &str) -> Vec<PathBuf> {
    crate::vendored_releases(vendor, VENDORED_NAME, slug)
        .into_iter()
        .map(|release| vendored_exe(&release, slug))
        .filter(|exe| exe.is_file())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_testing::TempTree;

    fn release(tree: &TempTree, name: &str, slug: &str) {
        let exe = vendored_exe(Path::new(name), slug);
        tree.write(&exe.to_string_lossy().replace('\\', "/"), b"");
    }

    #[test]
    fn what_sits_beside_the_binaries_leads_then_vendor_then_path_then_the_vulkan_sdk() {
        let exe = PathBuf::from("/apps/game");
        let vendored = PathBuf::from("/checkout/vendor/dxc-1.9.2607-linux-x86_64/bin").join(EXE);
        let found = candidates(Some(&exe), vec![vendored.clone()], Some("/sdk/vulkan"));
        assert_eq!(
            found,
            [
                exe.join(BUNDLED_DIR).join("bin").join(EXE),
                vendored,
                PathBuf::from(EXE),
                Path::new("/sdk/vulkan").join("bin").join(EXE),
            ]
        );
    }

    #[test]
    fn an_absent_or_empty_vulkan_sdk_contributes_no_candidate() {
        assert_eq!(candidates(None, Vec::new(), None), [PathBuf::from(EXE)]);
        assert_eq!(candidates(None, Vec::new(), Some("")), [PathBuf::from(EXE)]);
    }

    // Running another host's binary fails in a way the version probe cannot
    // explain, so it is never a candidate.
    #[test]
    fn another_platforms_release_is_not_a_candidate() {
        let tree = TempTree::new();
        release(&tree, "dxc-1.9.2607-macos-aarch64", "macos-aarch64");
        release(&tree, "dxc-1.9.2607-windows-x86_64", "windows-x86_64");

        let found = vendored_in(tree.path(), "macos-aarch64");
        assert_eq!(found.len(), 1);
        assert!(found[0].starts_with(tree.path().join("dxc-1.9.2607-macos-aarch64")));
    }

    // Microsoft's Windows archive nests the executable one level deeper than
    // every other layout, under the architecture it was built for.
    #[test]
    fn a_windows_release_resolves_its_x64_executable() {
        let tree = TempTree::new();
        release(&tree, "dxc-1.9.2607-windows-x86_64", "windows-x86_64");
        let found = vendored_in(tree.path(), "windows-x86_64");
        assert_eq!(
            found,
            [tree
                .path()
                .join("dxc-1.9.2607-windows-x86_64")
                .join("bin")
                .join("x64")
                .join(EXE)]
        );
    }

    // `vendor/` holds every other pinned SDK too, a half-synced release can
    // carry its licenses without its binaries, and a build directory sits
    // beside them while one is being built.
    #[test]
    fn a_directory_without_an_executable_or_another_components_is_ignored() {
        let tree = TempTree::new();
        tree.write("dxc-1.9.2607-linux-x86_64/LICENSE-MS.txt", b"");
        tree.dir("xess-3.0.1-windows-x86_64");
        tree.dir(".build/dxc-1.9.2607");
        release(&tree, "dxcx-1.9.2607-linux-x86_64", "linux-x86_64");

        assert!(vendored_in(tree.path(), "linux-x86_64").is_empty());
    }
}

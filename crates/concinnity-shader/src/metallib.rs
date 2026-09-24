//! MSL to a Metal library: the one place the engine runs the Metal toolchain,
//! for the built-in libraries a build script embeds and for the world shaders a
//! renderer compiles at runtime alike.
//!
//! The toolchain resolves once per process. Its binaries are invoked directly
//! against the macOS SDK rather than through `xcrun` per compile, which emits
//! the same bytes without a lookup for every step.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use crate::{run, stdout, stdout_line};

// The MSL dialect `msl::translate` emits, and the one the Metal backend's
// pipelines are built against.
const STANDARD: &str = "-std=metal3.0";

// A usable Metal toolchain: its two binaries, the SDK they compile against,
// and the id a cache keys their output under.
struct Toolchain {
    metal: PathBuf,
    metallib: PathBuf,
    sdk: PathBuf,
    id: String,
}

fn resolved() -> Option<&'static Toolchain> {
    static TOOLCHAIN: OnceLock<Option<Toolchain>> = OnceLock::new();
    TOOLCHAIN.get_or_init(probe).as_ref()
}

fn probe() -> Option<Toolchain> {
    let metal = PathBuf::from(xcrun(&["-f", "metal"])?);
    let metallib = PathBuf::from(xcrun(&["-f", "metallib"])?);
    let sdk = PathBuf::from(xcrun(&["--show-sdk-path"])?);
    let sdk_version = xcrun(&["--show-sdk-version"])?;
    let version = stdout(Command::new(&metal).arg("--version"))?;
    let id = toolchain_id_from(&version, &sdk_version)?;
    Some(Toolchain {
        metal,
        metallib,
        sdk,
        id,
    })
}

// One `xcrun --sdk macosx` query, answered on the first line of stdout.
fn xcrun(args: &[&str]) -> Option<String> {
    stdout_line(Command::new("xcrun").args(["--sdk", "macosx"]).args(args))
}

// The release line of `metal --version`, which reads
// `Apple metal version 32023.864 (metalfe-32023.864)`, and the SDK version,
// which a library records as its target SDK. The whole release line is kept
// rather than a version parsed out of it: a field that failed to parse would
// silently stop separating the releases it exists to separate.
fn toolchain_id_from(version_output: &str, sdk_version: &str) -> Option<String> {
    let line = version_output
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())?;
    let sdk = sdk_version.trim();
    (!sdk.is_empty()).then(|| format!("metal {line} sdk {sdk}"))
}

/// Whether this host carries the Metal compiler (full Xcode, not just the
/// Command Line Tools).
#[must_use]
pub fn toolchain_present() -> bool {
    resolved().is_some()
}

/// Identifies the Metal toolchain for a content-addressed cache of its output,
/// or `None` when none resolves.
///
/// Since Xcode 16 the toolchain is a separately versioned download, so it
/// moves without dxc, the source, or Xcode changing, and a library from a
/// superseded release still loads: a key without this replays it unnoticed.
#[must_use]
pub fn toolchain_id() -> Option<&'static str> {
    resolved().map(|t| t.id.as_str())
}

/// Compile MSL text to a metallib under a scratch directory of its own in
/// `work_dir`, returning the library's bytes.
pub fn compile(msl: &str, work_dir: &Path) -> Result<Vec<u8>, String> {
    crate::in_scratch(work_dir, |scratch| build(scratch, msl))
}

// Compile `msl` in `scratch`. Two steps rather than `metal -o x.metallib`: the
// one-step driver stamps a fresh UUID into every library, so no two builds of
// the same source would agree.
pub(crate) fn build(scratch: &Path, msl: &str) -> Result<Vec<u8>, String> {
    let toolchain = resolved().ok_or("hlsl: no Metal toolchain (full Xcode required)")?;
    let (msl_name, air_name, lib_name) = ("artifact.metal", "artifact.air", "artifact.metallib");
    std::fs::write(scratch.join(msl_name), msl)
        .map_err(|e| format!("hlsl: write {msl_name}: {e}"))?;
    run(
        Command::new(&toolchain.metal)
            .current_dir(scratch)
            .arg("-isysroot")
            .arg(&toolchain.sdk)
            .args([STANDARD, "-c", msl_name, "-o", air_name]),
        "metal",
        msl_name,
    )?;
    run(
        Command::new(&toolchain.metallib)
            .current_dir(scratch)
            .args([air_name, "-o", lib_name]),
        "metallib",
        msl_name,
    )?;
    std::fs::read(scratch.join(lib_name)).map_err(|e| format!("hlsl: read {lib_name}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELEASE: &str = "Apple metal version 32023.864 (metalfe-32023.864)\n\
                           Target: air64-apple-darwin25.6.0\n";

    // A library's bytes are not guaranteed identical across releases, so two
    // releases must key apart.
    #[test]
    fn two_toolchain_releases_key_differently() {
        let older = toolchain_id_from(
            "Apple metal version 32023.404 (metalfe-32023.404)\n",
            "26.2",
        );
        let newer = toolchain_id_from(RELEASE, "26.2");
        assert!(older.is_some() && newer.is_some(), "{older:?} {newer:?}");
        assert_ne!(older, newer);
    }

    // A library records the SDK it targeted, so an SDK update moves its bytes
    // under an unchanged compiler.
    #[test]
    fn two_sdks_key_differently() {
        assert_ne!(
            toolchain_id_from(RELEASE, "26.2"),
            toolchain_id_from(RELEASE, "26.4")
        );
    }

    #[test]
    fn the_id_names_the_metal_toolchain_and_its_sdk() {
        let id = toolchain_id_from(RELEASE, "26.2\n").expect("a version line yields an id");
        assert_eq!(
            id,
            "metal Apple metal version 32023.864 (metalfe-32023.864) sdk 26.2"
        );
    }

    // A toolchain that answered with nothing usable must not collapse every
    // release onto one key.
    #[test]
    fn an_empty_report_yields_no_id() {
        assert_eq!(toolchain_id_from("", "26.2"), None);
        assert_eq!(toolchain_id_from("\n  \n\t\n", "26.2"), None);
        assert_eq!(toolchain_id_from(RELEASE, " \n"), None);
    }

    // Leading blank lines are not a version, and trailing whitespace on the one
    // that is would key two runs of the same toolchain apart.
    #[test]
    fn the_release_line_is_taken_trimmed() {
        assert_eq!(
            toolchain_id_from(
                "\n\n  Apple metal version 32023.864  \nTarget: air64\n",
                "26.2"
            ),
            Some("metal Apple metal version 32023.864 sdk 26.2".to_string())
        );
    }

    // The two answers are one fact.
    #[test]
    fn a_present_toolchain_has_an_id() {
        assert_eq!(toolchain_present(), toolchain_id().is_some());
    }

    // A library out of a trivial kernel, twice: the output is non-empty and the
    // same both times, which is what lets a build embed it reproducibly.
    #[test]
    fn a_kernel_compiles_to_the_same_library_every_time() {
        if !toolchain_present() {
            return;
        }
        let tree = concinnity_testing::TempTree::new();
        let msl =
            "#include <metal_stdlib>\nkernel void k(device uint* o [[buffer(0)]]) { o[0] = 1; }\n";
        let first = compile(msl, tree.path()).expect("compiles");
        assert!(!first.is_empty());
        assert_eq!(first, compile(msl, tree.path()).expect("compiles"));
    }

    #[test]
    fn a_syntax_error_names_the_metal_compiler() {
        if !toolchain_present() {
            return;
        }
        let tree = concinnity_testing::TempTree::new();
        let err = compile("kernel void k( {\n", tree.path()).unwrap_err();
        assert!(err.contains("metal failed"), "{err}");
    }
}

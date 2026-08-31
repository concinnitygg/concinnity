//! Nothing passed to cargo may change which lines of this crate compile, apart
//! from the two development tiers (`detail` and `debug_assertions`, both about
//! cost rather than platform).
//!
//! The crate is the leaf every other one depends on, so a configuration axis
//! here recompiles the foundation differently for anyone whose graph enables it.
//! A backend belongs to the crate that owns the backend; this one takes the
//! shader platform as a parameter instead.

use std::path::{Path, PathBuf};

// Every `.rs` file the crate compiles, plus the build script that generates its
// tables.
fn crate_sources() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = vec![root.join("build.rs")];
    collect(&root.join("src"), &mut files);
    files
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("directory entry reads").path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn no_source_reads_the_target_or_a_backend() {
    // `CARGO_CFG_TARGET_OS` covers the build-script spelling of the same read.
    const FORBIDDEN: &[&str] = &[
        "target_os",
        "target_arch",
        "TARGET_OS",
        "backend_metal",
        "backend_dx",
        "backend_vk",
    ];

    let mut offenders = Vec::new();
    for path in crate_sources() {
        let text = read(&path);
        for (line_no, line) in text.lines().enumerate() {
            for needle in FORBIDDEN {
                if line.contains(needle) {
                    offenders.push(format!("{}:{}: {}", path.display(), line_no + 1, needle));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "the runtime foundation must not vary with the target or a render backend:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn detail_is_the_only_feature_gate() {
    let mut offenders = Vec::new();
    for path in crate_sources() {
        let text = read(&path);
        for (line_no, line) in text.lines().enumerate() {
            let mut rest = line;
            while let Some(at) = rest.find("feature = \"") {
                rest = &rest[at + "feature = \"".len()..];
                let name = rest.split('"').next().unwrap_or_default();
                if name != "detail" {
                    offenders.push(format!("{}:{}: {name}", path.display(), line_no + 1));
                }
            }
            // The build-script spelling: `CARGO_FEATURE_<NAME>`.
            if line.contains("CARGO_FEATURE_") {
                offenders.push(format!(
                    "{}:{}: CARGO_FEATURE_",
                    path.display(),
                    line_no + 1
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "`detail` is the crate's only feature gate; found:\n{}",
        offenders.join("\n")
    );
}

// The scanner is only as good as the tree it walks, so pin that it reaches a
// real file set rather than silently passing on an empty one.
#[test]
fn the_scan_reaches_the_whole_crate() {
    let files = crate_sources();
    assert!(files.len() > 100, "found only {} sources", files.len());
    assert!(files.iter().any(|p| p.ends_with("build.rs")));
    assert!(files.iter().any(|p| p.ends_with("platform.rs")));
}

//! Where code is allowed to read and write files.
//!
//! A path picked by hand under the system temporary directory survives the run,
//! is the same path on the next one, and is shared with every other process on
//! the machine. Two runs then collide: one resets a directory the other is
//! part-way through writing, and whatever a failure leaves behind is inherited
//! rather than cleaned. Left alone it also accumulates -- one machine reached
//! 4,375 leftover directories and 1.6 GB before the rule was written down.
//!
//! So exactly two modules name that directory, and each owns a kind of root:
//! `concinnity_host::scratch` hands out the ephemeral paths an external tool is
//! given, unique per call and removed when their guard drops;
//! `concinnity_testing::shared_dirs` owns the roots a suite keeps between runs,
//! the content-addressed cache and the per-process state. Everything else asks
//! one of them. This is a whole-workspace rule, not a test-only one: the shape
//! goes wrong the same way in a shipped tool, where it is a user's two
//! concurrent runs rather than two test processes.
//!
//! The other half is reading. A test may read its own package's tree -- the
//! lint scans in this workspace are built that way, and so is the shader
//! directory a backend checks itself against. Escaping it is what goes wrong: a
//! test that walks up out of its own crate reads files no test wrote, so it
//! passes or fails on a checkout's state. Two of those read pages `cn docs`
//! generates, and were removed rather than repaired.
//!
//! This is a source scan for the same reason the others are: nothing fails when
//! a test leaks a directory, so there is no runtime moment to assert at.

use std::path::{Path, PathBuf};

use concinnity_testing::source;

// This file names the forbidden call as test data, so scanning it would report
// itself.
const SELF: &str = "file_access_discipline.rs";

// Naming the system temporary directory at all.
const RAW_TEMP: &str = "env::temp_dir()";

// The two modules that own a root under it, by workspace-relative path. Both
// are the implementation of an exception, so both name the directory.
const TEMP_DIR_OWNERS: [&str; 2] = [
    "crates/concinnity-host/src/scratch.rs",
    "crates/concinnity-testing/src/shared_dirs.rs",
];

// Whether `path` is one of those, spelled with either separator.
fn owns_a_temp_root(path: &Path) -> bool {
    let text = path.to_string_lossy().replace('\\', "/");
    TEMP_DIR_OWNERS.iter().any(|owner| text.ends_with(owner))
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn rust_sources() -> Vec<PathBuf> {
    let root = workspace_root();
    source::rust_sources(
        &[root.join("src"), root.join("tests"), root.join("crates")],
        SELF,
    )
}

#[test]
fn only_the_two_root_owners_name_the_system_temp_dir() {
    let root = workspace_root();
    let mut offenders = Vec::new();

    for path in rust_sources() {
        if owns_a_temp_root(&path) {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

        for (number, line) in text.lines().enumerate() {
            if line.contains(RAW_TEMP) {
                let rel = path.strip_prefix(&root).unwrap_or(&path);
                offenders.push(format!("{}:{}", rel.display(), number + 1));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "these name the system temporary directory, which only the two root \
         owners may do. For a path an external tool is handed, take \
         `concinnity_host::scratch::Scratch`, which is unique per call and \
         removes itself; for a root a suite keeps, take \
         `concinnity_testing::shared_cache_dir` / `shared_state_dir`; for a \
         test's own files, take `concinnity_testing::TempTree`:\n  {}",
        offenders.join("\n  ")
    );
}

// The scan is only worth as much as its ability to see the shape it forbids.
#[test]
fn the_scan_recognises_a_hand_built_temp_path() {
    let offending = "let dir = std::env::temp_dir().join(\"cn-thing\");";

    assert!(offending.contains(RAW_TEMP));
    assert!(!"let dir = concinnity_host::scratch::path(\"thing\");".contains(RAW_TEMP));
}

// The owners are recognised by where they are, so a file that merely sounds
// like one is still scanned.
#[test]
fn the_scan_spares_the_owners_and_nothing_else() {
    assert!(owns_a_temp_root(Path::new(
        "/w/crates/concinnity-host/src/scratch.rs"
    )));
    assert!(owns_a_temp_root(Path::new(
        "\\w\\crates\\concinnity-testing\\src\\shared_dirs.rs"
    )));
    assert!(!owns_a_temp_root(Path::new(
        "/w/crates/other/src/scratch.rs"
    )));
    assert!(!owns_a_temp_root(Path::new("/w/src/shared_dirs.rs")));
}

// A path built from this package's own directory, walked upward out of it.
// `CARGO_MANIFEST_DIR` alone is fine -- it names the package's own tree.
fn escapes_its_own_package(body: &str) -> bool {
    body.lines().any(|line| {
        line.contains("CARGO_MANIFEST_DIR") && (line.contains("..") || line.contains("parent()"))
    })
}

#[test]
fn no_test_reads_outside_its_own_package() {
    let root = workspace_root();
    let mut offenders = Vec::new();

    for path in rust_sources() {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

        // A helper may build the path the test then reads, so the whole file's
        // functions are in scope, not just the `#[test]` ones.
        let helpers: Vec<_> = source::fn_bodies(&text)
            .into_iter()
            .filter(|b| escapes_its_own_package(&b.text))
            .collect();

        for body in helpers {
            let rel = path.strip_prefix(&root).unwrap_or(&path);
            offenders.push(format!("{}:{} {}", rel.display(), body.line, body.name));
        }
    }

    assert!(
        offenders.is_empty(),
        "these read outside their own package, so they pass or fail on files no \
         test wrote. Build what the test needs instead -- `docs::tests` writes a \
         vocabulary of its own rather than reading the engine's:\n  {}",
        offenders.join("\n  ")
    );
}

#[test]
fn the_scan_recognises_a_walk_out_of_the_package() {
    let escaping = "fn repo_root() -> PathBuf {\n    Path::new(env!(\"CARGO_MANIFEST_DIR\")).join(\"../..\")\n}\n";
    let bodies = source::fn_bodies(escaping);
    assert_eq!(bodies.len(), 1);
    assert!(escapes_its_own_package(&bodies[0].text));

    // Reading the package's own tree is what the lint scans do.
    let own = "fn crate_sources() -> PathBuf {\n    Path::new(env!(\"CARGO_MANIFEST_DIR\")).join(\"src\")\n}\n";
    assert!(!escapes_its_own_package(&source::fn_bodies(own)[0].text));
}

#[test]
fn the_scan_reads_the_whole_workspace() {
    let files = rust_sources();
    assert!(
        files.len() > 300,
        "only {} sources found; the walk is not reaching the crates",
        files.len()
    );
    assert!(
        !files.iter().any(|p| p.ends_with(SELF)),
        "the guard does not scan itself"
    );
}

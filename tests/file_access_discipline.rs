//! Where a test is allowed to read and write files.
//!
//! A test writes into a directory that deletes itself, and nowhere else. A path
//! built by hand under the system temporary directory is not that: it survives
//! the run, it is the same path on the next one, and two runs of the same test
//! share it. Left alone this accumulates -- one machine reached 4,375 leftover
//! directories and 1.6 GB before the rule was written down.
//!
//! The exceptions are the two roots a suite keeps outside any one test:
//! `concinnity_testing::shared_cache_dir` for a content-addressed cache, which
//! exists to be kept, and `shared_state_dir` for process-wide state, which is
//! emptied on the first call in a run.
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

// Building a path under the system temporary directory by hand. The harness's
// own `TempTree`, `shared_cache_dir` and `shared_state_dir` are what a test
// uses instead.
const RAW_TEMP: &str = "env::temp_dir()";

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

// Production code may name the system temporary directory: a shader compile
// writes its intermediates there, and a mesh stream needs scratch space. The
// rule is about tests, so only `#[cfg(test)]` code is scanned -- and the tests
// that assert *about* that production behaviour name the directory too, which
// is why the check is for a test that BUILDS a path rather than one that
// mentions the call.
fn builds_a_temp_path(body: &str) -> bool {
    body.lines()
        .any(|line| line.contains(RAW_TEMP) && (line.contains(".join(") || line.contains("let ")))
}

#[test]
fn no_test_builds_its_own_path_under_the_system_temp_dir() {
    let root = workspace_root();
    let mut offenders = Vec::new();

    for path in rust_sources() {
        // The harness implements the exception, so it names the directory.
        if path.to_string_lossy().contains("concinnity-testing") {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

        for body in source::test_bodies(&text) {
            if builds_a_temp_path(&body.text) {
                let rel = path.strip_prefix(&root).unwrap_or(&path);
                offenders.push(format!("{}:{} {}", rel.display(), body.line, body.name));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "these tests build their own path under the system temporary directory, \
         which survives the run and is shared with the next one. Use \
         `concinnity_testing::TempTree`, or `shared_cache_dir` / \
         `shared_state_dir` for a root a suite keeps:\n  {}",
        offenders.join("\n  ")
    );
}

// The scan is only worth as much as its ability to see the shape it forbids.
#[test]
fn the_scan_recognises_a_hand_built_temp_path() {
    let offending =
        "\n#[test]\nfn leaks() {\n    let dir = std::env::temp_dir().join(\"cn-thing\");\n}\n";
    let bodies = source::test_bodies(offending);

    assert_eq!(bodies.len(), 1);
    assert!(builds_a_temp_path(&bodies[0].text));

    let fixed = offending.replace(
        "std::env::temp_dir().join(\"cn-thing\")",
        "concinnity_testing::TempTree::new()",
    );
    assert!(!builds_a_temp_path(&source::test_bodies(&fixed)[0].text));
}

// A test may assert *about* the production code that uses the temporary
// directory without building a path of its own.
#[test]
fn the_scan_allows_asserting_on_the_directory() {
    let asserting =
        "\n#[test]\nfn stays_in_temp() {\n    assert!(a.starts_with(std::env::temp_dir()));\n}\n";
    let bodies = source::test_bodies(asserting);

    assert!(
        !builds_a_temp_path(&bodies[0].text),
        "naming the directory is not building a path in it"
    );
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

//! How a test is allowed to reach this binary's process-global state.
//!
//! Cargo runs a binary's tests on parallel threads, so the working directory,
//! the installed state root, the engine's development flags and the window
//! policy are shared by every test running at that moment. `concinnity-testing`
//! puts one reader/writer lock over all of it: readers stay parallel, writers
//! run alone.
//!
//! Neither guard is reentrant. Taking two on one thread deadlocks the test, and
//! a deadlocked test does not fail -- it hangs, which on a coverage run or a
//! constrained container reads as a suite that never finishes. That is the
//! failure this scan exists to make impossible.
//!
//! It reads the workspace's own test code, in the same style as
//! `configuration_surface.rs`, `double_drive_audit.rs` and
//! `headless_discipline.rs`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use concinnity_testing::source::{self, FnBody};

// This file names the forbidden shapes as test data, so scanning it would
// report itself.
const SELF: &str = "global_state_discipline.rs";

// Every spelling of "I hold the one exclusive guard". Taking any two of these
// live at once on a thread is the deadlock.
//
// The guards a caller may import and then call bare are matched unqualified, so
// `write_access()` counts however it was reached. `lock` stays qualified: a
// bare `lock()` is also how the cook and the host take their own mutexes, which
// are not this one.
const EXCLUSIVE: &[&str] = &[
    "test_support::lock()",
    "lock_in_temp_cwd()",
    "GlobalState::acquire()",
    "concinnity_testing::exclusive()",
    "write_access()",
    "Output::new()",
];

// Shared access. Also not reentrant against the exclusive guard.
const SHARED: &[&str] = &["concinnity_testing::shared()", "read_access()"];

// Writes to process-global state. A test reaching one of these without an
// exclusive guard races every other test in its binary.
const GLOBAL_WRITES: &[&str] = &[
    "set_state_dir(",
    "set_current_dir(",
    "dev_flags::set_",
    "set_pending_animations(",
];

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

// A guard bound at the body's own top level lives to the end of the test. One
// taken inside a nested block or a closure is scoped to that block, and a
// second one after it is sequential rather than nested -- which is fine, and is
// how the harness's own tests prove the lock recovers from poison.
fn guards_held_at_top_level(body: &str, needles: &[&str]) -> usize {
    let Some(base) = body.lines().nth(1).map(|l| l.len() - l.trim_start().len()) else {
        return 0;
    };
    body.lines()
        .skip(1)
        .filter(|line| {
            let indent = line.len() - line.trim_start().len();
            indent == base
                && line.trim_start().starts_with("let ")
                && needles.iter().any(|n| line.contains(n))
        })
        .count()
}

fn mentions(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| text.contains(n))
}

// A test's own body plus the bodies of the same-file helpers it calls. One
// level deep: enough to see through a `fn world_with_reload_entry()` that
// writes a flag on the test's behalf.
fn reach(body: &FnBody, helpers: &HashMap<String, String>) -> String {
    let mut text = body.text.clone();
    for (name, helper_body) in helpers {
        if body.text.contains(&format!("{name}(")) && *name != body.name {
            text.push('\n');
            text.push_str(helper_body);
        }
    }
    text
}

fn helper_map(text: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for body in source::fn_bodies(text) {
        out.entry(body.name).or_insert(body.text);
    }
    out
}

#[test]
fn no_test_takes_the_exclusive_guard_twice() {
    let root = workspace_root();
    let mut offenders = Vec::new();

    for path in rust_sources() {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        for body in source::test_bodies(&text) {
            let taken = guards_held_at_top_level(&body.text, EXCLUSIVE);
            if taken > 1 {
                let rel = path.strip_prefix(&root).unwrap_or(&path);
                offenders.push(format!(
                    "{}:{} {} holds {taken}",
                    rel.display(),
                    body.line,
                    body.name
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "these tests take the one exclusive guard more than once, which \
         deadlocks the thread rather than failing it. Take exactly one \
         (`lock_in_temp_cwd` covers the lock and the working directory \
         together):\n  {}",
        offenders.join("\n  ")
    );
}

#[test]
fn no_test_mixes_the_shared_and_exclusive_guards() {
    let root = workspace_root();
    let mut offenders = Vec::new();

    for path in rust_sources() {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        for body in source::test_bodies(&text) {
            if guards_held_at_top_level(&body.text, EXCLUSIVE) > 0
                && guards_held_at_top_level(&body.text, SHARED) > 0
            {
                let rel = path.strip_prefix(&root).unwrap_or(&path);
                offenders.push(format!("{}:{} {}", rel.display(), body.line, body.name));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "these tests hold the exclusive guard and then ask for the shared one. \
         The lock is not reentrant, so this deadlocks:\n  {}",
        offenders.join("\n  ")
    );
}

// Which test binary a source file belongs to. Cargo compiles a crate's `src`
// into one test binary and every file directly under its `tests/` into its own,
// so a process-global race is only ever between tests of the same binary.
fn test_binary_of(path: &Path, root: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();

    // `crates/<name>/tests/<file>.rs` and `tests/<file>.rs` are each their own
    // binary; everything under a crate's `src` shares that crate's lib binary.
    if parts.len() >= 2 && parts[parts.len() - 2] == "tests" {
        return format!("integration:{}", rel.display());
    }
    match parts.first().map(String::as_str) {
        Some("crates") => format!("lib:{}", parts.get(1).cloned().unwrap_or_default()),
        _ => "lib:concinnity".to_string(),
    }
}

#[test]
fn a_test_that_writes_a_global_holds_the_exclusive_guard() {
    let root = workspace_root();
    // Per binary: the tests that write a global without holding the guard.
    let mut unguarded: HashMap<String, Vec<String>> = HashMap::new();

    for path in rust_sources() {
        // The guards' own definitions write these globals to implement them.
        let as_str = path.to_string_lossy().to_string();
        if as_str.contains("concinnity-testing") || as_str.ends_with("dev_flags.rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let helpers = helper_map(&text);
        let binary = test_binary_of(&path, &root);

        for body in source::test_bodies(&text) {
            let reachable = reach(&body, &helpers);
            if mentions(&reachable, GLOBAL_WRITES) && !mentions(&reachable, EXCLUSIVE) {
                let rel = path.strip_prefix(&root).unwrap_or(&path);
                unguarded.entry(binary.clone()).or_default().push(format!(
                    "{}:{} {}",
                    rel.display(),
                    body.line,
                    body.name
                ));
            }
        }
    }

    // One unguarded writer in a binary has nothing to race: it is the only test
    // in that process that moves the global. Two or more race each other, and
    // the guard is what orders them.
    let mut offenders = Vec::new();
    for (binary, tests) in &unguarded {
        if tests.len() > 1 {
            offenders.push(format!("{binary}:\n    {}", tests.join("\n    ")));
        }
    }
    offenders.sort();

    assert!(
        offenders.is_empty(),
        "these test binaries have more than one test writing process-global \
         state without the exclusive guard, so they race each other:\n  {}",
        offenders.join("\n  ")
    );
}

// The scan is only worth as much as its ability to see the shapes it forbids.
#[test]
fn the_scan_recognises_two_guards_held_at_once() {
    let offending = "\n#[test]\nfn deadlocks() {\n    let _guard = crate::test_support::lock();\n    let _cwd = concinnity_testing::GlobalState::acquire().with_cwd();\n}\n";
    let bodies = source::test_bodies(offending);
    assert_eq!(bodies.len(), 1);
    assert_eq!(
        guards_held_at_top_level(&bodies[0].text, EXCLUSIVE),
        2,
        "two live guards in one body"
    );

    let fixed = offending.replace("    let _guard = crate::test_support::lock();\n", "");
    assert_eq!(
        guards_held_at_top_level(&source::test_bodies(&fixed)[0].text, EXCLUSIVE),
        1
    );
}

// A guard taken inside a closure or block is dropped with it, so a later one is
// sequential, not nested. That is how the harness proves the lock survives a
// poisoning panic, and it must not read as a deadlock.
#[test]
fn a_scoped_guard_is_not_counted_as_held() {
    let sequential = "\n#[test]\nfn poison_then_retake() {\n    let r = catch_unwind(|| {\n        let _guard = concinnity_testing::exclusive();\n        panic!();\n    });\n    let _second = concinnity_testing::exclusive();\n}\n";
    let bodies = source::test_bodies(sequential);

    assert_eq!(bodies.len(), 1);
    assert_eq!(
        guards_held_at_top_level(&bodies[0].text, EXCLUSIVE),
        1,
        "only the top-level binding is held to the end"
    );
}

// A `{` inside a fixture string must not run the body into its neighbours, or
// the scan reports whatever the next test does.
#[test]
fn a_malformed_json_fixture_does_not_extend_the_body() {
    let source_text = "\n#[test]\nfn reads_bad_json() {\n    write(\"{ not json\");\n}\n\n#[test]\nfn takes_a_guard() {\n    let _g = concinnity_testing::exclusive();\n}\n";
    let bodies = source::test_bodies(source_text);

    assert_eq!(bodies.len(), 2);
    assert_eq!(
        guards_held_at_top_level(&bodies[0].text, EXCLUSIVE),
        0,
        "the first test holds nothing: {:?}",
        bodies[0].text
    );
}

#[test]
fn the_scan_sees_a_global_write_through_a_helper() {
    let source_text = "fn make_world() {\n    dev_flags::set_enabled(true);\n}\n\n#[test]\nfn writes_a_flag_indirectly() {\n    make_world();\n}\n";
    let helpers = helper_map(source_text);
    let body = &source::test_bodies(source_text)[0];
    let reachable = reach(body, &helpers);

    assert!(
        mentions(&reachable, GLOBAL_WRITES),
        "the helper's write is reachable from the test"
    );
    assert!(!mentions(&reachable, EXCLUSIVE), "and it holds no guard");
}

// Most `write_access()` call sites import it and call it bare, so a needle
// carrying the module path would see two of nine.
#[test]
fn the_scan_sees_a_guard_called_without_its_module_path() {
    let bare = "\n#[test]\nfn writes_a_flag() {\n    let _flags = write_access();\n    let _also = concinnity_testing::exclusive();\n}\n";
    let bodies = source::test_bodies(bare);

    assert_eq!(
        guards_held_at_top_level(&bodies[0].text, EXCLUSIVE),
        2,
        "a bare guard call counts the same as a qualified one"
    );
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

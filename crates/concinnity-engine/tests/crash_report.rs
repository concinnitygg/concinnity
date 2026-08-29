//! End-to-end panic reporting: a child process (this test binary re-run with
//! the ignored probe selected) installs the crash hooks and panics; the parent
//! asserts a complete report lands under the state root's crashes dir.
//!
//! The engine reads no environment of its own, so the parent hands the child a
//! scratch root through a variable this file owns and the probe installs it.

use std::process::Command;

// How the parent tells the spawned probe where to write. Read here and nowhere
// else: the state root is installed through the engine's own API below.
const PROBE_ROOT_ENV: &str = "CN_CRASH_PROBE_ROOT";

// An integration test links the engine as an ordinary dependency, so it
// inherits nothing from the engine's own `#[cfg(test)]` allocator, and without
// one the report below would carry no heap figures to assert on.
concinnity_core::install_global_allocator!();

#[test]
#[ignore = "probe body: spawned by crash_report_lands_for_a_panicking_process"]
fn panicking_probe() {
    if let Some(root) = std::env::var_os(PROBE_ROOT_ENV) {
        concinnity_engine::paths::set_state_dir(root);
    }
    concinnity_engine::crash::install();
    panic!("crash report end to end probe");
}

// The exact byte count from a `key: <bytes> (<scale>)` header line.
fn header_bytes(text: &str, key: &str) -> Option<u64> {
    let rest = text.lines().find_map(|l| l.strip_prefix(key))?;
    rest.split_whitespace().next()?.parse().ok()
}

#[test]
fn crash_report_lands_for_a_panicking_process() {
    let root = tempfile::tempdir().unwrap();
    let exe = std::env::current_exe().unwrap();
    let output = Command::new(exe)
        .args(["--exact", "panicking_probe", "--ignored"])
        .env(PROBE_ROOT_ENV, root.path())
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "the probe must die of its panic: {}",
        String::from_utf8_lossy(&output.stdout)
    );

    let crashes = root.path().join("crashes");
    let paths: Vec<_> = std::fs::read_dir(&crashes)
        .expect("crashes dir created by the hook")
        .flatten()
        .map(|e| e.path())
        .collect();

    let reports: Vec<_> = paths
        .iter()
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    assert_eq!(reports.len(), 1, "one report for one panic: {paths:?}");
    let text = std::fs::read_to_string(reports[0]).unwrap();
    assert!(text.contains("kind: panic"));
    assert!(text.contains("message: crash report end to end probe"));
    assert!(text.contains("thread: panicking_probe"));
    assert!(text.contains("crash_report.rs"));
    assert!(text.contains("backtrace:"));
    assert!(text.trim_end().ends_with("(end of report)"));

    // The memory figures come from a real crashing process, so they must be
    // plausible rather than merely present: the tracked heap is a part of the
    // resident set, never the whole of it or more.
    let heap_live =
        header_bytes(&text, "heap-live: ").expect("this test binary installs the allocator");
    let rss = header_bytes(&text, "rss: ").expect("RSS is queryable on a supported platform");
    assert!(heap_live > 0, "a running process holds a tracked heap");
    assert!(
        heap_live < rss,
        "tracked heap {heap_live} must fit inside RSS {rss}"
    );
    assert!(text.contains("heap-peak: "));
    assert!(text.contains("heap-churn: "));

    #[cfg(any(target_os = "macos", target_os = "windows"))]
    {
        let dumps: Vec<_> = paths
            .iter()
            .filter(|p| p.extension().is_some_and(|x| x == "dmp"))
            .collect();
        assert_eq!(dumps.len(), 1, "a minidump beside the report: {paths:?}");
        let len = std::fs::metadata(dumps[0]).unwrap().len();
        assert!(len > 0, "minidump has content");
    }
}

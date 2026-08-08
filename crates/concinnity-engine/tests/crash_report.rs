// End-to-end panic reporting: a child process (this test binary re-run with
// the ignored probe selected) installs the crash hooks and panics; the parent
// asserts a complete report lands under the state root's crashes dir.

use std::process::Command;

#[test]
#[ignore = "probe body: spawned by crash_report_lands_for_a_panicking_process"]
fn panicking_probe() {
    concinnity_engine::crash::install();
    panic!("crash report end to end probe");
}

#[test]
fn crash_report_lands_for_a_panicking_process() {
    let root = tempfile::tempdir().unwrap();
    let exe = std::env::current_exe().unwrap();
    let output = Command::new(exe)
        .args(["--exact", "panicking_probe", "--ignored"])
        .env("CN_HOME", root.path())
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "the probe must die of its panic: {}",
        String::from_utf8_lossy(&output.stdout)
    );

    let crashes = root.path().join(".concinnity").join("crashes");
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

// src/crash/write.rs
//
// Report emission and retention. Reports land under `paths::crashes_dir()`;
// each section is written and flushed before the next begins, so a report
// interrupted mid-write still carries its most valuable sections. The
// directory is pruned to the newest reports, minidump siblings included.

use super::report::CrashReport;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};

pub(crate) const RETAINED_REPORTS: usize = 20;

// A stem under `dir` whose `.txt` and `.dmp` are both unclaimed, so a report
// and its minidump always pair up. Falls back to the base stem after a few
// suffixes; `create_new` still guards the file itself.
pub(crate) fn unique_stem(dir: &Path, base: &str) -> String {
    let free = |stem: &String| {
        !dir.join(format!("{stem}.txt")).exists() && !dir.join(format!("{stem}.dmp")).exists()
    };
    let plain = base.to_string();
    if free(&plain) {
        return plain;
    }
    (2..10)
        .map(|n| format!("{base}-{n}"))
        .find(free)
        .unwrap_or(plain)
}

// Write `report` to `<dir>/<stem>.txt`, section by section with a flush after
// each. Never overwrites an existing file.
pub(crate) fn write_report_named(
    dir: &Path,
    stem: &str,
    report: &CrashReport,
) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{stem}.txt"));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    for section in report.sections() {
        file.write_all(section.as_bytes())?;
        file.flush()?;
    }
    let _ = file.sync_all();
    Ok(path)
}

// Write `report` under the crashes dir and prune. The common path for report
// kinds that carry no minidump.
pub(crate) fn emit(report: &CrashReport) -> Option<PathBuf> {
    let dir = concinnity_store::paths::crashes_dir();
    let stem = unique_stem(&dir, &report.file_stem());
    let path = write_report_named(&dir, &stem, report).ok()?;
    prune(&dir, RETAINED_REPORTS);
    Some(path)
}

// Keep the newest `keep` reports (by file-name timestamp) and delete the
// rest, treating a `.txt`/`.dmp` pair as one report. A `.dmp` without its
// sidecar still counts, so a dump from a failed report write survives until
// it ages out.
pub(crate) fn prune(dir: &Path, keep: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut stems: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str());
        if !matches!(ext, Some("txt") | Some("dmp")) {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            && stem.starts_with("crash-")
            && !stems.iter().any(|s| s == stem)
        {
            stems.push(stem.to_string());
        }
    }
    if stems.len() <= keep {
        return;
    }
    // Newest first: the stem embeds a zero-padded timestamp.
    stems.sort_unstable_by(|a, b| b.cmp(a));
    for stem in &stems[keep..] {
        let _ = std::fs::remove_file(dir.join(format!("{stem}.txt")));
        let _ = std::fs::remove_file(dir.join(format!("{stem}.dmp")));
    }
}

// Create the minidump file for a report stem. Returns the open file plus its
// path so a failed dump can be cleaned up.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub(crate) fn create_dump_file(
    dir: &Path,
    stem: &str,
) -> std::io::Result<(std::fs::File, PathBuf)> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{stem}.dmp"));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)?;
    Ok((file, path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crash::report::{ReportKind, UtcTime};

    fn report_at(secs: u64) -> CrashReport {
        CrashReport {
            kind: ReportKind::Panic,
            time: UtcTime::from_unix(secs),
            message: "boom".to_string(),
            thread: None,
            location: None,
            backtrace: None,
            notes: Vec::new(),
            recent_logs: Vec::new(),
        }
    }

    #[test]
    fn a_report_lands_with_all_sections() {
        let dir = tempfile::tempdir().unwrap();
        let report = report_at(1_786_192_496);
        let stem = unique_stem(dir.path(), &report.file_stem());
        let path = write_report_named(dir.path(), &stem, &report).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("concinnity crash report\n"));
        assert!(text.contains("message: boom"));
        assert!(text.ends_with("(end of report)\n"));
    }

    #[test]
    fn colliding_stems_get_a_suffix() {
        let dir = tempfile::tempdir().unwrap();
        let report = report_at(1_786_192_496);
        let base = report.file_stem();
        let first = unique_stem(dir.path(), &base);
        write_report_named(dir.path(), &first, &report).unwrap();
        let second = unique_stem(dir.path(), &base);
        assert_ne!(first, second);
        assert!(second.starts_with(&base));
        // A dump claims the stem too, even without a sidecar report.
        std::fs::write(dir.path().join(format!("{second}.dmp")), b"dump").unwrap();
        let third = unique_stem(dir.path(), &base);
        assert_ne!(third, second);
    }

    #[test]
    fn prune_keeps_the_newest_and_removes_dump_siblings() {
        let dir = tempfile::tempdir().unwrap();
        for hour in 0..6 {
            let stem = format!("crash-20260808-{hour:02}0000-1");
            std::fs::write(dir.path().join(format!("{stem}.txt")), b"r").unwrap();
            std::fs::write(dir.path().join(format!("{stem}.dmp")), b"d").unwrap();
        }
        // An orphan dump counts as a report of its own.
        std::fs::write(dir.path().join("crash-20260807-000000-1.dmp"), b"d").unwrap();
        // Unrelated files are never touched.
        std::fs::write(dir.path().join("notes.md"), b"keep").unwrap();

        prune(dir.path(), 3);

        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"notes.md".to_string()));
        assert!(names.contains(&"crash-20260808-050000-1.txt".to_string()));
        assert!(names.contains(&"crash-20260808-030000-1.dmp".to_string()));
        assert!(!names.iter().any(|n| n.starts_with("crash-20260808-02")));
        assert!(!names.iter().any(|n| n.starts_with("crash-20260807-")));
        // 3 stems survive, each with its pair intact.
        assert_eq!(names.iter().filter(|n| n.starts_with("crash-")).count(), 6);
    }

    #[test]
    fn prune_under_the_cap_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("crash-20260808-000000-1.txt"), b"r").unwrap();
        prune(dir.path(), 20);
        assert!(dir.path().join("crash-20260808-000000-1.txt").exists());
        // A missing directory is tolerated.
        prune(&dir.path().join("nope"), 20);
    }
}

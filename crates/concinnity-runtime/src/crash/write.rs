//! Crash log emission. The log lands at the path `crash::install` was given.
//! The first report of a process replaces whatever an earlier run left there;
//! any later one in the same process is appended, so a crash that follows from
//! the first never buries its root cause. The minidump is claimed apart from
//! the log: the first report able to capture one writes it, even after a
//! report that could not. Each section is written and flushed before the next
//! begins, so a log interrupted mid-write still carries its most valuable
//! sections.

use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use super::report::CrashReport;

// How a report reaches the log: the first of a process starts it afresh, a
// later one is added after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogWrite {
    Replace,
    Append,
}

// Something a process does once per run: start the log, or write the dump.
#[derive(Debug, Default)]
pub(crate) struct FirstClaim(AtomicBool);

impl FirstClaim {
    pub(crate) const fn new() -> Self {
        Self(AtomicBool::new(false))
    }

    // Whether this caller is the first. Exactly one ever sees `true`.
    pub(crate) fn claim(&self) -> bool {
        !self.0.swap(true, Ordering::SeqCst)
    }

    fn claimed(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    // How the next report reaches the log this claim guards.
    pub(crate) fn log_write(&self) -> LogWrite {
        if self.claim() {
            LogWrite::Replace
        } else {
            LogWrite::Append
        }
    }
}

// The process's crash log, and its minidump, which only the panic hook and the
// native fault handler can capture.
pub(crate) static LOG: FirstClaim = FirstClaim::new();
pub(crate) static DUMP: FirstClaim = FirstClaim::new();

// Write `report` to `path`, section by section with a flush after each.
#[expect(
    clippy::let_underscore_must_use,
    reason = "the log is already written; syncing it is best effort"
)]
pub(crate) fn write_report(
    path: &Path,
    report: &CrashReport,
    mode: LogWrite,
) -> std::io::Result<()> {
    let mut file = match mode {
        LogWrite::Replace => create_truncated(path)?,
        LogWrite::Append => {
            let mut file = OpenOptions::new().create(true).append(true).open(path)?;
            file.write_all(b"\n")?;
            file
        }
    };
    for section in report.sections() {
        file.write_all(section.as_bytes())?;
        file.flush()?;
    }
    let _ = file.sync_all();
    Ok(())
}

// Write `report` to the crash log, reporting whether it landed. The path for
// report kinds that carry no minidump, so it leaves the dump to a later crash
// that can capture one.
pub(crate) fn emit(report: &CrashReport) -> bool {
    let Some(path) = super::log_path() else {
        return false;
    };
    emit_at(&path, report, &LOG, &DUMP)
}

// `emit` against the given claims. As the run's first report, while no dump
// has been claimed, it removes the dump an earlier run left rather than leave
// it beside a log it does not belong to.
fn emit_at(path: &Path, report: &CrashReport, log: &FirstClaim, dump: &FirstClaim) -> bool {
    let mode = log.log_write();
    if mode == LogWrite::Replace && !dump.claimed() {
        remove_dump(path);
    }
    write_report(path, report, mode).is_ok()
}

// The minidump paired with the crash log at `log`.
pub(crate) fn dump_path(log: &Path) -> PathBuf {
    log.with_extension("dmp")
}

// Remove the dump beside `log`, if any.
#[expect(
    clippy::let_underscore_must_use,
    reason = "removing a dump is best effort; the text log is what matters"
)]
pub(crate) fn remove_dump(log: &Path) {
    let _ = std::fs::remove_file(dump_path(log));
}

// Open `path` for writing from empty, creating its directory first: the
// writable root of a fresh per-user install may not exist yet.
pub(crate) fn create_truncated(path: &Path) -> std::io::Result<File> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crash::memory::MemorySnapshot;
    use crate::crash::report::ReportKind;
    use concinnity_core::utc::UtcTime;

    fn report(message: &str) -> CrashReport {
        CrashReport {
            kind: ReportKind::Panic,
            time: UtcTime::from_unix(1_786_192_496),
            message: message.to_string(),
            thread: None,
            location: None,
            backtrace: None,
            notes: Vec::new(),
            recent_logs: Vec::new(),
            memory: MemorySnapshot::default(),
        }
    }

    #[test]
    fn a_report_lands_with_all_sections() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crash.log");
        write_report(&path, &report("boom"), LogWrite::Replace).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("concinnity crash report\n"));
        assert!(text.contains("message: boom"));
        assert!(text.ends_with("(end of report)\n"));
    }

    // One log, rewritten by each run's first crash: it replaces what an
    // earlier run left whole rather than appending to it or landing beside it.
    #[test]
    fn a_new_run_replaces_the_log() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crash.log");
        write_report(
            &path,
            &report(&"earlier-crash ".repeat(500)),
            LogWrite::Replace,
        )
        .unwrap();
        write_report(&path, &report("second"), LogWrite::Replace).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("message: second"));
        assert!(!text.contains("earlier-crash"));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    // A writable root that does not exist yet is created for the log.
    #[test]
    fn the_log_creates_its_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state").join("crash.log");
        write_report(&path, &report("boom"), LogWrite::Replace).unwrap();
        assert!(path.is_file());
    }

    // Only the first report of a process starts the log; every later one
    // claims an append, so the root cause stays at the top.
    #[test]
    fn only_the_first_claim_replaces() {
        let slot = FirstClaim::new();
        assert_eq!(slot.log_write(), LogWrite::Replace);
        assert_eq!(slot.log_write(), LogWrite::Append);
        assert_eq!(slot.log_write(), LogWrite::Append);
    }

    // A device loss starts the log without a dump: it clears the dump an
    // earlier run left, and the crash that follows still claims the dump.
    #[test]
    fn a_report_without_a_dump_leaves_the_dump_to_the_next_crash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crash.log");
        std::fs::write(dump_path(&path), b"stale dump").unwrap();
        let (log, dump) = (FirstClaim::new(), FirstClaim::new());

        assert!(emit_at(&path, &report("device lost"), &log, &dump));
        assert!(
            !dump_path(&path).exists(),
            "an earlier run's dump is cleared"
        );
        assert!(dump.claim(), "the next crash still captures the dump");
        assert_eq!(log.log_write(), LogWrite::Append);
    }

    // Once a dump is claimed, a later first log write leaves it in place.
    #[test]
    fn a_claimed_dump_is_not_cleared() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crash.log");
        std::fs::write(dump_path(&path), b"this run's dump").unwrap();
        let (log, dump) = (FirstClaim::new(), FirstClaim::new());
        assert!(dump.claim());
        assert!(emit_at(&path, &report("device lost"), &log, &dump));
        assert!(dump_path(&path).exists());
    }

    // A crash that follows the first in one process is added after it, never
    // over it.
    #[test]
    fn a_later_crash_in_the_process_keeps_the_first() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crash.log");
        std::fs::write(&path, b"stale log from an earlier run").unwrap();
        let slot = FirstClaim::new();
        write_report(&path, &report("root cause"), slot.log_write()).unwrap();
        write_report(&path, &report("poisoned lock"), slot.log_write()).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("earlier run"));
        let first = text.find("message: root cause").expect("first report kept");
        let later = text
            .find("message: poisoned lock")
            .expect("later report added");
        assert!(first < later);
    }
}

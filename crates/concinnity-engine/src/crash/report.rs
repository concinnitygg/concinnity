// src/crash/report.rs
//
// The crash report data and its plain-text rendering. Rendering is pure and
// section-based: the writer emits sections in order of forensic value and
// flushes between them, so a partial report still leads with what matters.

use super::memory::MemorySnapshot;
use std::time::{SystemTime, UNIX_EPOCH};

// Caps keep the hook's allocations bounded no matter what panicked.
pub(crate) const MAX_MESSAGE_BYTES: usize = 4 * 1024;
pub(crate) const MAX_BACKTRACE_BYTES: usize = 192 * 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ReportKind {
    Panic,
    DeviceLost,
    // Only the native fault handler constructs this; Linux has none.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    NativeFault,
}

impl ReportKind {
    pub(crate) fn label(self) -> &'static str {
        match self {
            ReportKind::Panic => "panic",
            ReportKind::DeviceLost => "gpu-device-lost",
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            ReportKind::NativeFault => "native-fault",
        }
    }
}

pub(crate) struct CrashReport {
    pub kind: ReportKind,
    pub time: UtcTime,
    pub message: String,
    pub thread: Option<String>,
    pub location: Option<String>,
    pub(crate) backtrace: Option<String>,
    pub notes: Vec<(String, String)>,
    pub(crate) recent_logs: Vec<String>,
    pub memory: MemorySnapshot,
}

impl CrashReport {
    // A report with the ambient context filled in: timestamp, context notes,
    // and the recent-log ring. The caller adds the event-specific fields.
    pub(crate) fn gather(kind: ReportKind, message: String) -> Self {
        Self {
            kind,
            time: UtcTime::now(),
            message,
            thread: None,
            location: None,
            backtrace: None,
            notes: super::notes_snapshot(),
            recent_logs: super::ring::global().snapshot(),
            memory: MemorySnapshot::capture(),
        }
    }

    // Like `gather`, but never blocks on a lock: used by the native fault
    // handler, where the crashing thread may hold the ring or notes lock.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    pub(crate) fn gather_nonblocking(kind: ReportKind, message: String) -> Self {
        Self {
            kind,
            time: UtcTime::now(),
            message,
            thread: None,
            location: None,
            backtrace: None,
            notes: super::try_notes_snapshot(),
            recent_logs: super::ring::global().try_snapshot(),
            memory: MemorySnapshot::capture(),
        }
    }

    // Preferred file stem for this report: sortable timestamp plus pid, so
    // retention pruning can order reports by name alone.
    pub(crate) fn file_stem(&self) -> String {
        file_stem_at(self.time)
    }

    // The report text as ordered sections. The writer flushes after each one.
    pub(crate) fn sections(&self) -> [String; 4] {
        [
            self.header(),
            self.summary(),
            self.backtrace_section(),
            self.logs_section(),
        ]
    }

    fn header(&self) -> String {
        let mut out = String::with_capacity(256);
        out.push_str("concinnity crash report\n");
        out.push_str(&format!("kind: {}\n", self.kind.label()));
        out.push_str(&format!("time: {} UTC\n", self.time.display()));
        out.push_str(&format!("engine: {}\n", env!("CARGO_PKG_VERSION")));
        out.push_str(&format!(
            "schema: {:#010x}\n",
            concinnity_host::store::blob::SCHEMA_HASH
        ));
        out.push_str(&format!(
            "os: {} {}\n",
            std::env::consts::OS,
            std::env::consts::ARCH
        ));
        self.memory.write_into(&mut out);
        for (key, value) in &self.notes {
            out.push_str(&format!("{key}: {value}\n"));
        }
        out.push('\n');
        out
    }

    fn summary(&self) -> String {
        let mut out = String::with_capacity(self.message.len().min(MAX_MESSAGE_BYTES) + 64);
        out.push_str("message: ");
        out.push_str(truncated(&self.message, MAX_MESSAGE_BYTES));
        out.push('\n');
        if let Some(thread) = &self.thread {
            out.push_str(&format!("thread: {thread}\n"));
        }
        if let Some(location) = &self.location {
            out.push_str(&format!("location: {location}\n"));
        }
        out.push('\n');
        out
    }

    fn backtrace_section(&self) -> String {
        match &self.backtrace {
            Some(bt) => format!("backtrace:\n{}\n", truncated(bt, MAX_BACKTRACE_BYTES)),
            None => "backtrace: <none>\n".to_string(),
        }
    }

    fn logs_section(&self) -> String {
        let mut out = String::new();
        out.push_str("\nrecent logs (oldest first):\n");
        if self.recent_logs.is_empty() {
            out.push_str("<none>\n");
        }
        for line in &self.recent_logs {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("(end of report)\n");
        out
    }
}

// The file stem a report written at `time` claims. Free-standing so the fault
// handler can name (and write) the minidump before it builds the report.
pub(crate) fn file_stem_at(time: UtcTime) -> String {
    format!("crash-{}-{}", time.stem(), std::process::id())
}

// Truncate to `cap` bytes on a char boundary.
fn truncated(s: &str, cap: usize) -> &str {
    if s.len() <= cap {
        return s;
    }
    let mut end = cap;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

// A civil UTC timestamp, derived from the system clock without a date
// dependency. Only meaningful for post-epoch times, which is all a crash
// report needs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct UtcTime {
    pub(crate) year: i64,
    pub(crate) month: u32,
    pub day: u32,
    pub(crate) hour: u32,
    pub minute: u32,
    pub second: u32,
}

impl UtcTime {
    pub(crate) fn now() -> Self {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self::from_unix(secs)
    }

    // Civil-from-days conversion (Gregorian, proleptic; Hinnant's algorithm).
    pub(crate) fn from_unix(secs: u64) -> Self {
        let days = (secs / 86_400) as i64;
        let rem = secs % 86_400;
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097) as u64;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
        let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
        let year = yoe as i64 + era * 400 + i64::from(month <= 2);
        Self {
            year,
            month,
            day,
            hour: (rem / 3600) as u32,
            minute: (rem % 3600 / 60) as u32,
            second: (rem % 60) as u32,
        }
    }

    pub(crate) fn display(&self) -> String {
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }

    // Compact, lexicographically sortable form for file names.
    pub(crate) fn stem(&self) -> String {
        format!(
            "{:04}{:02}{:02}-{:02}{:02}{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_converts_to_civil_origin() {
        let t = UtcTime::from_unix(0);
        assert_eq!(t.display(), "1970-01-01 00:00:00");
    }

    #[test]
    fn known_timestamps_convert() {
        // 2026-08-08 12:34:56 UTC.
        let t = UtcTime::from_unix(1_786_192_496);
        assert_eq!(t.display(), "2026-08-08 12:34:56");
        assert_eq!(t.stem(), "20260808-123456");
        // A leap day: 2024-02-29 23:59:59 UTC.
        let t = UtcTime::from_unix(1_709_251_199);
        assert_eq!(t.display(), "2024-02-29 23:59:59");
    }

    fn sample() -> CrashReport {
        CrashReport {
            kind: ReportKind::Panic,
            time: UtcTime::from_unix(1_786_192_496),
            message: "index out of bounds".to_string(),
            thread: Some("main".to_string()),
            location: Some("src/lib.rs:10:5".to_string()),
            backtrace: Some("0: frame_a\n1: frame_b".to_string()),
            notes: vec![("backend".to_string(), "metal".to_string())],
            recent_logs: vec!["+1.000s INFO app: started".to_string()],
            memory: MemorySnapshot {
                heap: Some(concinnity_memory::MemStats {
                    live_bytes: 400 * 1024 * 1024,
                    peak_bytes: 512 * 1024 * 1024,
                    alloc_count: 900,
                    free_count: 400,
                }),
                rss_bytes: Some(6 * 1024 * 1024 * 1024),
            },
        }
    }

    #[test]
    fn sections_carry_the_report_in_order() {
        let text = sample().sections().join("");
        let order = [
            "kind: panic",
            "time: 2026-08-08 12:34:56 UTC",
            "schema: 0x",
            "heap-live: 419430400 (400.0 MiB)",
            "rss: 6442450944 (6.0 GiB)",
            "backend: metal",
            "message: index out of bounds",
            "thread: main",
            "location: src/lib.rs:10:5",
            "backtrace:\n0: frame_a",
            "recent logs (oldest first):\n+1.000s INFO app: started",
            "(end of report)",
        ];
        let mut at = 0;
        for needle in order {
            let found = text[at..].find(needle);
            assert!(found.is_some(), "missing or out of order: {needle}");
            at += found.unwrap();
        }
    }

    #[test]
    fn oversized_fields_truncate_on_char_boundaries() {
        let mut report = sample();
        report.message = "\u{e9}".repeat(MAX_MESSAGE_BYTES);
        report.backtrace = Some("x".repeat(MAX_BACKTRACE_BYTES + 100));
        let text = report.sections().join("");
        assert!(text.len() < MAX_MESSAGE_BYTES + MAX_BACKTRACE_BYTES + 4096);
        // Two-byte char at the boundary: truncation stays on a boundary.
        assert!(text.contains("message: \u{e9}"));
    }

    #[test]
    fn file_stem_orders_by_time() {
        let older = sample().file_stem();
        let mut newer = sample();
        newer.time = UtcTime::from_unix(1_786_192_497);
        assert!(newer.file_stem() > older);
    }
}

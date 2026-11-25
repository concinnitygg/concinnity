//! Crash reporting: a panic hook, native fault capture, and one local crash log
//! at the path a host names at install. A run's first crash rewrites the log,
//! and any later one in the same run is appended after it, so the log holds
//! the latest run's crashes with the root cause first. The log is plain text
//! written section by section, so a partial log still leads with what matters.
//! On macOS and Windows the run's first panic or native fault also writes a
//! minidump beside the log, sharing its stem. Local files only: nothing is
//! uploaded, and no hostname or username is recorded.

mod hook;
mod memory;
mod report;
pub(crate) use report::utc_now;
mod ring;
mod write;

#[cfg(any(target_os = "macos", target_os = "windows"))]
mod minidump;
#[cfg(any(target_os = "macos", target_os = "windows"))]
mod native;

pub use ring::RingLayer;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

// Where the log lands. Process state, and it has to be: a fault handler runs on
// a compromised stack with no caller to take a path from, so the path is
// resolved once, up front, by whoever installs the hooks.
static LOG_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Install the process-wide crash hooks writing the crash log at `log`: a panic
/// hook that writes the log and chains to the previously installed hook, plus
/// (macOS/Windows) a native fault handler that writes a minidump. Binaries call
/// this first thing at startup, naming the crash log of the state tree they
/// resolved; later calls re-point the log but install nothing twice.
///
/// Without a path (`None`) the hooks still chain and the notes still
/// accumulate, but nothing is written: an embedder that has chosen no place for
/// them gets no files.
pub fn install(log: Option<&Path>) {
    *LOG_PATH
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = log.map(Path::to_path_buf);
    hook::install();
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    native::install();
}

const MAX_NOTES: usize = 16;
const MAX_NOTE_KEY_BYTES: usize = 32;
const MAX_NOTE_VALUE_BYTES: usize = 128;

// Bounded key-value context stamped into every report (backend, GPU, world
// identity). Bounded so the crash path never carries unbounded state.
static NOTES: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

/// Record a context note included in subsequent crash reports, replacing any
/// previous value for `key`. Bounded: at most 16 keys, keys truncate at 32
/// bytes and values at 128; further keys are dropped.
pub fn note(key: &str, value: &str) {
    let key = key[..key.floor_char_boundary(MAX_NOTE_KEY_BYTES)].to_string();
    let value = value[..value.floor_char_boundary(MAX_NOTE_VALUE_BYTES)].to_string();
    let mut notes = NOTES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(slot) = notes.iter_mut().find(|(k, _)| *k == key) {
        slot.1 = value;
    } else if notes.len() < MAX_NOTES {
        notes.push((key, value));
    }
}

// Write a crash-style report for a lost GPU device. The process itself is
// healthy, so no minidump is captured; the report exists so a device loss on
// another machine leaves the same local evidence as a crash.
pub(crate) fn report_device_lost(detail: &str) {
    let report = report::CrashReport::gather(report::ReportKind::DeviceLost, detail.to_string());
    if !write::emit(&report) {
        tracing::warn!("crash log for device loss could not be written");
    }
}

// The crash log, or `None` when no host named one.
pub(crate) fn log_path() -> Option<PathBuf> {
    LOG_PATH.lock().ok()?.clone()
}

pub(crate) fn notes_snapshot() -> Vec<(String, String)> {
    NOTES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

// Snapshot that gives up instead of blocking, for the native fault path.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub(crate) fn try_notes_snapshot() -> Vec<(String, String)> {
    match NOTES.try_lock() {
        Ok(notes) => notes.clone(),
        Err(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The notes map is process-global; a single test drives it so mutations
    // never race another test.
    #[test]
    fn notes_replace_by_key_and_stay_bounded() {
        note("probe-key", "first");
        note("probe-key", "second");
        let snap = notes_snapshot();
        assert_eq!(snap.iter().filter(|(k, _)| k == "probe-key").count(), 1);
        assert!(snap.contains(&("probe-key".to_string(), "second".to_string())));

        note("probe-long", &"v".repeat(1000));
        let snap = notes_snapshot();
        let long = snap.iter().find(|(k, _)| k == "probe-long").unwrap();
        assert_eq!(long.1.len(), MAX_NOTE_VALUE_BYTES);

        for i in 0..2 * MAX_NOTES {
            note(&format!("probe-fill-{i}"), "x");
        }
        assert!(notes_snapshot().len() <= MAX_NOTES);
    }
}

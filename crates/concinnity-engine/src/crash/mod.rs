// src/crash/mod.rs
//
// Crash reporting: a panic hook, native fault capture, and local report files
// under the crashes dir (`.concinnity/crashes/` in a dev project). Reports
// are plain text written section by section, so a partial report still leads
// with what matters; macOS and Windows also write a minidump beside the
// report. The directory is pruned to the newest reports after each write.
// Local files only: nothing is uploaded, and no hostname or username is
// recorded.

mod hook;
mod report;
mod ring;
mod write;

#[cfg(any(target_os = "macos", target_os = "windows"))]
mod minidump;
#[cfg(any(target_os = "macos", target_os = "windows"))]
mod native;

pub use ring::RingLayer;

use std::sync::Mutex;

/// Install the process-wide crash hooks: a panic hook that writes a report
/// and chains to the previously installed hook, plus (macOS/Windows) a native
/// fault handler that writes a minidump. Binaries call this first thing at
/// startup; later calls are no-ops.
pub fn install() {
    hook::install();
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    native::install();
    #[cfg(backend_metal)]
    note("backend", "metal");
    #[cfg(backend_dx)]
    note("backend", "directx");
    #[cfg(backend_vk)]
    note("backend", "vulkan");
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
    let key = clamp(key, MAX_NOTE_KEY_BYTES);
    let value = clamp(value, MAX_NOTE_VALUE_BYTES);
    let mut notes = NOTES.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(slot) = notes.iter_mut().find(|(k, _)| *k == key) {
        slot.1 = value;
    } else if notes.len() < MAX_NOTES {
        notes.push((key, value));
    }
}

/// Write a crash-style report for a lost GPU device. The process itself is
/// healthy, so no minidump is captured; the report exists so a device loss on
/// another machine leaves the same local evidence as a crash.
pub fn report_device_lost(detail: &str) {
    let report = report::CrashReport::gather(report::ReportKind::DeviceLost, detail.to_string());
    if write::emit(&report).is_none() {
        tracing::warn!("crash report for device loss could not be written");
    }
}

pub(crate) fn notes_snapshot() -> Vec<(String, String)> {
    NOTES.lock().unwrap_or_else(|p| p.into_inner()).clone()
}

// Snapshot that gives up instead of blocking, for the native fault path.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub(crate) fn try_notes_snapshot() -> Vec<(String, String)> {
    match NOTES.try_lock() {
        Ok(notes) => notes.clone(),
        Err(_) => Vec::new(),
    }
}

fn clamp(s: &str, cap: usize) -> String {
    let mut end = s.len().min(cap);
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
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

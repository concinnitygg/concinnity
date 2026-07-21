// Filesystem identity for memoizing repeated reads of a source file.
//
// A build reads one source file from many assets -- every Mesh imported from a
// single `.fbx`, every sibling lookup for one `.gltf` -- so the readers memoize
// on a cheap stat instead of re-reading. The stat alone is not a sound key:
// filesystem timestamps advance in discrete ticks (~3ms on NTFS, and the
// Windows system timer is coarser), so two writes landing in one tick carry the
// same mtime. When such a rewrite also preserves length -- retargeting a glTF
// buffer URI from `a.bin` to `b.bin`, say -- the stamp repeats across different
// contents and a memo keyed on it serves stale data.
//
// `settled` closes that window. Once wall-clock has advanced past the tick a
// file was written in, every later write must land on a different mtime, so the
// stamp becomes sound. Callers memoize only settled observations and re-read
// the rest, paying a repeat read for at most `SETTLE_WINDOW` after a write and
// nothing after that.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

// How long after a write a file's stamp stays untrusted. Comfortably above the
// ~3ms NTFS tick and the 15.6ms default Windows system timer, while keeping the
// re-read window short enough that a hot-reload rebuild pays it once or twice.
const SETTLE_WINDOW: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct FileStamp {
    mtime_nanos: u64,
    len: u64,
}

impl FileStamp {
    // Stat `path`, or `None` when it cannot be read.
    pub(crate) fn read(path: &str) -> Option<Self> {
        let meta = std::fs::metadata(path).ok()?;
        Some(Self {
            mtime_nanos: meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            len: meta.len(),
        })
    }

    // Whether this stamp is old enough to key a memo entry on.
    pub(crate) fn settled(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        self.settled_at(now)
    }

    // A stamp mtime in the future (clock skew, or a timestamp copied from a
    // newer source) reads as unsettled, costing a re-read rather than risking a
    // stale hit.
    fn settled_at(&self, now_nanos: u64) -> bool {
        now_nanos.saturating_sub(self.mtime_nanos) > SETTLE_WINDOW.as_nanos() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp(mtime_nanos: u64, len: u64) -> FileStamp {
        FileStamp { mtime_nanos, len }
    }

    #[test]
    fn read_reports_length_and_a_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.bin");
        std::fs::write(&path, b"12345").unwrap();
        let s = FileStamp::read(path.to_str().unwrap()).expect("stat");
        assert_eq!(s.len, 5);
        assert!(FileStamp::read("/no/such/file.bin").is_none());
    }

    #[test]
    fn equal_length_rewrite_can_repeat_the_stamp() {
        // The bug this module exists for: same length, same tick, same stamp.
        // Equality is what a memo keys on, so the guard cannot come from here.
        assert_eq!(stamp(1_000, 5), stamp(1_000, 5));
        assert_ne!(stamp(1_000, 5), stamp(1_000, 6), "length must separate");
        assert_ne!(stamp(1_000, 5), stamp(2_000, 5), "mtime must separate");
    }

    #[test]
    fn settled_only_once_the_window_has_elapsed() {
        let window = SETTLE_WINDOW.as_nanos() as u64;
        let s = stamp(1_000_000, 5);
        assert!(!s.settled_at(1_000_000), "a just-written file is unsettled");
        assert!(
            !s.settled_at(1_000_000 + window),
            "the boundary itself is still unsettled"
        );
        assert!(s.settled_at(1_000_000 + window + 1), "past the window");
    }

    #[test]
    fn a_future_mtime_is_never_settled() {
        // saturating_sub keeps a skewed clock from reading as long-settled.
        assert!(!stamp(u64::MAX, 5).settled_at(1_000));
    }

    #[test]
    fn a_file_settles_after_the_window() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.bin");
        std::fs::write(&path, b"12345").unwrap();
        let s = FileStamp::read(path.to_str().unwrap()).expect("stat");
        assert!(!s.settled(), "a file written just now is unsettled");
        std::thread::sleep(SETTLE_WINDOW + Duration::from_millis(50));
        assert!(s.settled(), "the same stamp settles once the window passes");
    }
}

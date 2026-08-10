// src/crash/memory.rs
//
// The memory figures a crash report carries: the tracked Rust heap and the
// process resident set. Neither number means much alone. Together they say
// whose growth it was -- 400 MiB of heap inside a 6 GiB resident set is the
// driver, the mapped assets or the binary, and the same 6 GiB with a 5 GiB heap
// is ours.
//
// Capture is a sum of relaxed atomics plus one platform query: no locks, no
// allocation, and nothing that can block on a thread the fault suspended.
// Rendering allocates and runs later, with the report's other sections.

use concinnity_memory::MemStats;

// The tracked heap and process RSS at crash time. Either half is `None` when
// its source is unavailable: no binary installed the tracking allocator, or the
// platform RSS query failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) struct MemorySnapshot {
    pub heap: Option<MemStats>,
    pub rss_bytes: Option<u64>,
}

impl MemorySnapshot {
    pub(crate) fn capture() -> Self {
        Self {
            heap: concinnity_memory::stats(),
            rss_bytes: crate::app::sysmem::process_resident_bytes(),
        }
    }

    // Header lines in the same `key: value` form as the rest of the header. An
    // unavailable source says so rather than reporting a zero, which would read
    // as a real measurement of an empty heap.
    pub(crate) fn write_into(&self, out: &mut String) {
        match self.heap {
            Some(h) => {
                out.push_str(&format!("heap-live: {}\n", bytes(h.live_bytes)));
                out.push_str(&format!("heap-peak: {}\n", bytes(h.peak_bytes)));
                out.push_str(&format!(
                    "heap-churn: {} alloc / {} free\n",
                    h.alloc_count, h.free_count
                ));
            }
            None => out.push_str("heap: <unavailable>\n"),
        }
        match self.rss_bytes {
            Some(rss) => out.push_str(&format!("rss: {}\n", bytes(rss))),
            None => out.push_str("rss: <unavailable>\n"),
        }
    }
}

// Exact bytes with a binary scale beside them: the exact figure is what two
// reports are compared on, the scale is what a reader takes in at a glance.
fn bytes(value: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    let (div, unit) = if value >= GIB {
        (GIB, "GiB")
    } else if value >= MIB {
        (MIB, "MiB")
    } else if value >= KIB {
        (KIB, "KiB")
    } else {
        return format!("{value} B");
    };
    format!("{value} ({:.1} {unit})", value as f64 / div as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    fn rendered(snap: MemorySnapshot) -> String {
        let mut out = String::new();
        snap.write_into(&mut out);
        out
    }

    // The engine installs the tracking allocator, so a live capture carries
    // both halves on every supported platform.
    #[test]
    fn a_live_capture_carries_both_halves() {
        let snap = MemorySnapshot::capture();
        let heap = snap
            .heap
            .expect("the engine installs the tracking allocator");
        assert!(heap.alloc_count > 0);
        assert!(heap.peak_bytes >= heap.live_bytes);
        if cfg!(any(
            target_os = "macos",
            target_os = "linux",
            target_os = "windows"
        )) {
            assert!(snap.rss_bytes.expect("RSS query works here") > 0);
        }
    }

    // The pair a report is read for: a small heap inside a large resident set
    // is the reading that says the growth was not ours.
    #[test]
    fn both_figures_render_with_exact_bytes_and_a_scale() {
        let text = rendered(MemorySnapshot {
            heap: Some(MemStats {
                live_bytes: 400 * MIB,
                peak_bytes: 512 * MIB,
                alloc_count: 8_123_456,
                free_count: 7_901_234,
            }),
            rss_bytes: Some(6 * 1024 * MIB),
        });
        assert!(text.contains("heap-live: 419430400 (400.0 MiB)\n"));
        assert!(text.contains("heap-peak: 536870912 (512.0 MiB)\n"));
        assert!(text.contains("heap-churn: 8123456 alloc / 7901234 free\n"));
        assert!(text.contains("rss: 6442450944 (6.0 GiB)\n"));
    }

    // A zero here would read as a measured empty heap, which is the one thing
    // the report must not claim.
    #[test]
    fn an_unavailable_source_says_so_rather_than_reporting_zero() {
        let text = rendered(MemorySnapshot::default());
        assert!(text.contains("heap: <unavailable>\n"));
        assert!(text.contains("rss: <unavailable>\n"));
        assert!(!text.contains(" 0 "));
    }

    #[test]
    fn small_values_read_as_plain_bytes() {
        assert_eq!(bytes(0), "0 B");
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(2048), "2048 (2.0 KiB)");
    }
}

// src/debug/memory.rs
//
// The `memory` query: what the process is holding, and who reports holding it.
//
// The same numbers the Health panel draws, on a surface that needs no window --
// a headless check can read the heap counters, the per-tag ledger, and the
// allocation histogram straight out of a running engine. Built from values
// passed in rather than read from the globals here, so the reply shape is
// testable without a live process.

use concinnity_engine::ecs::ScratchStats;
use concinnity_memory::{LedgerSnapshot, MemStats, Realm, SizeClass};

pub(super) fn report(
    frame: u64,
    heap: Option<MemStats>,
    tags: &LedgerSnapshot,
    hot_class: Option<SizeClass>,
    scratch: ScratchStats,
) -> serde_json::Value {
    let rows: Vec<_> = Realm::ALL
        .into_iter()
        .flat_map(|realm| tags.reported(realm))
        .map(|usage| {
            serde_json::json!({
                "tag": usage.tag.name(),
                "realm": usage.realm.name(),
                "bytes": usage.bytes,
                "peak_bytes": usage.peak_bytes,
                "budget": usage.budget,
                "over_budget": usage.over_budget(),
            })
        })
        .collect();

    serde_json::json!({
        "ok": true,
        "frame": frame,
        // Null when no binary installed the tracking allocator, which is not
        // the same as a heap holding nothing.
        "heap": heap.map(|h| serde_json::json!({
            "live_bytes": h.live_bytes,
            "peak_bytes": h.peak_bytes,
            "alloc_count": h.alloc_count,
            "free_count": h.free_count,
        })),
        "tags": rows,
        // The frame scratch reserve and the most any single frame drew from it:
        // `peak` is what the reserve constant should be sized from, and a
        // non-zero `overflows` means some frame wanted more than it got.
        "scratch": {
            "capacity_bytes": scratch.capacity,
            "peak_bytes": scratch.peak,
            "overflows": scratch.overflows,
        },
        // Null unless the allocator was built with its `detail` feature.
        "hot_class": hot_class.map(|c| serde_json::json!({
            "min_bytes": c.min_bytes,
            "max_bytes": c.max_bytes,
            "allocs": c.allocs,
            "live_blocks": c.live_blocks,
        })),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_memory::{Ledger, MemTag};

    fn ledger() -> Ledger {
        let ledger = Ledger::new();
        ledger.set(MemTag::Textures, Realm::Device, 4_096);
        ledger.set_budget(MemTag::Textures, Realm::Device, Some(2_048));
        ledger.set(MemTag::Scratch, Realm::Host, 512);
        ledger
    }

    #[test]
    fn the_reply_lists_every_reported_tag_with_its_budget() {
        let value = report(7, None, &ledger().snapshot(), None, ScratchStats::default());
        assert_eq!(value["ok"], true);
        assert_eq!(value["frame"], 7);

        let tags = value["tags"].as_array().expect("tags is an array");
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0]["tag"], "Scratch");
        assert_eq!(tags[0]["realm"], "RAM");
        assert_eq!(tags[0]["bytes"], 512);
        assert!(tags[0]["budget"].is_null());
        assert_eq!(tags[1]["tag"], "Textures");
        assert_eq!(tags[1]["realm"], "VRAM");
        assert_eq!(tags[1]["budget"], 2_048);
        assert_eq!(tags[1]["over_budget"], true);
    }

    // An absent instrument reads as null, never as a zero that would look like
    // a measurement.
    #[test]
    fn absent_instruments_read_as_null() {
        let value = report(
            0,
            None,
            &LedgerSnapshot::default(),
            None,
            ScratchStats::default(),
        );
        assert!(value["heap"].is_null());
        assert!(value["hot_class"].is_null());
        assert_eq!(value["tags"].as_array().expect("tags is an array").len(), 0);
    }

    #[test]
    fn the_heap_and_histogram_are_reported_when_measured() {
        let heap = MemStats {
            live_bytes: 1_000,
            peak_bytes: 2_000,
            alloc_count: 30,
            free_count: 10,
        };
        let hot = SizeClass {
            min_bytes: 64,
            max_bytes: 127,
            allocs: 900,
            live_blocks: 42,
        };
        let scratch = ScratchStats {
            capacity: 1 << 20,
            peak: 4_096,
            overflows: 0,
        };
        let value = report(
            1,
            Some(heap),
            &LedgerSnapshot::default(),
            Some(hot),
            scratch,
        );

        assert_eq!(value["heap"]["live_bytes"], 1_000);
        assert_eq!(value["heap"]["free_count"], 10);
        assert_eq!(value["hot_class"]["min_bytes"], 64);
        assert_eq!(value["hot_class"]["live_blocks"], 42);
        assert_eq!(value["scratch"]["capacity_bytes"], 1 << 20);
        assert_eq!(value["scratch"]["peak_bytes"], 4_096);
    }

    // The reserve holding is the normal case, so the count that says it did not
    // has to survive to the reply rather than being cleared with the frame.
    #[test]
    fn a_reserve_that_overflowed_says_so() {
        let scratch = ScratchStats {
            capacity: 1024,
            peak: 1024,
            overflows: 17,
        };
        let value = report(2, None, &LedgerSnapshot::default(), None, scratch);
        assert_eq!(value["scratch"]["overflows"], 17);
    }
}

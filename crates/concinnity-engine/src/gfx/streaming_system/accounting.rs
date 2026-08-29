// src/gfx/streaming_system/accounting.rs
//
// Streaming's report into the shared memory ledger: what each pool holds in
// device memory, and the budget it holds it against.
//
// The pools already track resident bytes and byte budgets to drive their own
// residency policy. Publishing those under the shared tags costs a few atomic
// stores a frame and is what lets one readout break device memory down by what
// is holding it, beside the process RAM the tracking allocator counts.

use concinnity_core::memory::{Ledger, MemTag, Realm};

// One pool's device footprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PoolReport {
    pub(crate) tag: MemTag,
    pub(crate) resident_bytes: u64,
    // The pool's resident-byte budget, or `None` when it runs count-only.
    pub(crate) byte_budget: Option<u64>,
}

// Publish each pool's footprint. A pool that is not streaming is absent from
// `reports` and its tag is left alone: nothing is known about it this frame,
// which is not the same as it holding nothing.
pub(crate) fn publish(ledger: &Ledger, reports: impl IntoIterator<Item = PoolReport>) {
    for report in reports {
        ledger.set(report.tag, Realm::Device, report.resident_bytes);
        ledger.set_budget(report.tag, Realm::Device, report.byte_budget);
    }
}

// The `Textures` row, which two things contribute to: the texture streamer
// (`(resident, budget)` when it is streaming) and the render graph's transient
// pool. The pool is not a streaming pool -- it sits off the device allocator,
// because its slots alias on purpose -- so nothing else accounts for it, and
// without this it would show up only in the device-wide total.
//
// The pool's bytes are added to the budget as well as the usage. The budget is
// the streamer's own residency cap, and the pool is not competing for it, so
// counting the pool on one side only would make the streamer read over budget
// on memory it does not control.
//
// `None` when there is nothing to say: no streamer and an empty pool. That is
// not the same as reporting zero, which would claim the tag holds nothing.
pub(crate) fn textures_report(
    streamer: Option<(u64, Option<u64>)>,
    transient_pool_bytes: u64,
) -> Option<PoolReport> {
    if streamer.is_none() && transient_pool_bytes == 0 {
        return None;
    }
    let (resident, budget) = streamer.unwrap_or((0, None));
    Some(PoolReport {
        tag: MemTag::Textures,
        resident_bytes: resident + transient_pool_bytes,
        byte_budget: budget.map(|b| b + transient_pool_bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(tag: MemTag, resident_bytes: u64, byte_budget: Option<u64>) -> PoolReport {
        PoolReport {
            tag,
            resident_bytes,
            byte_budget,
        }
    }

    #[test]
    fn each_pool_lands_under_its_own_tag() {
        let ledger = Ledger::new();
        publish(
            &ledger,
            [
                report(MemTag::Textures, 4_096, Some(8_192)),
                report(MemTag::Meshes, 1_024, None),
            ],
        );

        let textures = ledger.usage(MemTag::Textures, Realm::Device);
        assert_eq!(textures.bytes, 4_096);
        assert_eq!(textures.budget, Some(8_192));

        let meshes = ledger.usage(MemTag::Meshes, Realm::Device);
        assert_eq!(meshes.bytes, 1_024);
        assert_eq!(meshes.budget, None);
    }

    // Published every frame, so a pool that evicted must read lower rather than
    // accumulating what it used to hold.
    #[test]
    fn republishing_replaces_the_previous_frame() {
        let ledger = Ledger::new();
        publish(&ledger, [report(MemTag::Chunks, 10_000, Some(16_000))]);
        publish(&ledger, [report(MemTag::Chunks, 2_000, Some(16_000))]);

        assert_eq!(ledger.usage(MemTag::Chunks, Realm::Device).bytes, 2_000);
        assert_eq!(
            ledger.usage(MemTag::Chunks, Realm::Device).peak_bytes,
            10_000
        );
    }

    // Reports are device-side: streaming residency is a GPU upload, and the
    // host row must not claim it.
    #[test]
    fn reports_land_in_device_memory_only() {
        let ledger = Ledger::new();
        publish(&ledger, [report(MemTag::Textures, 4_096, None)]);
        assert_eq!(ledger.usage(MemTag::Textures, Realm::Host).bytes, 0);
        assert_eq!(ledger.snapshot().realm_bytes(Realm::Device), 4_096);
    }

    // A pool that is not streaming says nothing, leaving whatever another
    // reporter published under that tag intact.
    #[test]
    fn an_absent_pool_leaves_its_tag_untouched() {
        let ledger = Ledger::new();
        publish(&ledger, [report(MemTag::Meshes, 512, None)]);
        publish(&ledger, []);
        assert_eq!(ledger.usage(MemTag::Meshes, Realm::Device).bytes, 512);
    }

    #[test]
    fn the_transient_pool_joins_the_texture_row() {
        // Both contributors present: the row is their sum, and the budget rises
        // by the pool's share so the streamer's headroom against its own cap is
        // exactly what it was.
        let r = textures_report(Some((4_000, Some(10_000))), 1_000).expect("reports");
        assert_eq!(r.tag, MemTag::Textures);
        assert_eq!(r.resident_bytes, 5_000);
        assert_eq!(r.byte_budget, Some(11_000));
        let headroom = r.byte_budget.unwrap() - r.resident_bytes;
        assert_eq!(headroom, 10_000 - 4_000, "streamer headroom is unchanged");
    }

    #[test]
    fn the_transient_pool_reports_without_a_streamer() {
        // A world with no streamed textures still has a transient pool, and it
        // is the only thing that would otherwise account for those bytes.
        let r = textures_report(None, 2_048).expect("reports");
        assert_eq!(r.resident_bytes, 2_048);
        assert_eq!(r.byte_budget, None, "no streamer means no cap to report");

        // A count-only streamer keeps its absent budget rather than gaining one.
        let r = textures_report(Some((100, None)), 50).expect("reports");
        assert_eq!(r.resident_bytes, 150);
        assert_eq!(r.byte_budget, None);
    }

    #[test]
    fn nothing_to_say_says_nothing() {
        // No streamer and an empty pool is not the same as "the tag holds
        // nothing": staying silent leaves whatever is there alone.
        assert_eq!(textures_report(None, 0), None);
        // A streaming pool holding zero *is* a claim, and still reports.
        assert!(textures_report(Some((0, Some(8_000))), 0).is_some());
    }
}

// src/gfx/streaming_system/accounting.rs
//
// Streaming's report into the shared memory ledger: what each pool holds in
// device memory, and the budget it holds it against.
//
// The pools already track resident bytes and byte budgets to drive their own
// residency policy. Publishing those under the shared tags costs a few atomic
// stores a frame and is what lets one readout break device memory down by what
// is holding it, beside the process RAM the tracking allocator counts.

use concinnity_memory::{Ledger, MemTag, Realm};

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
}

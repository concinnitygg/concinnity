// concinnity-memory/src/lib.rs
//
// The engine's allocation layer: what the process is holding, who is holding it,
// and the allocators that hand memory out in bulk instead of one block at a
// time.
//
// Five things live here, in files that do not depend on each other:
//
//   counters   the global heap's live / peak / churn, sharded per thread and
//              driven by `TrackingAlloc` (`tracking`)
//   ledger     tagged byte accounting -- textures, meshes, audio, scratch --
//              in host and device memory, against optional budgets
//   arena      a bump allocator for per-frame working memory
//   pool       fixed-capacity storage for a population that churns
//   inline_vec a sequence that keeps its first element inline, for the many
//              per-entity collections that hold exactly one thing
//
// The counters measure the Rust heap, not "the engine". They see every
// allocation the process makes through Rust -- engine, tools, and third-party
// crates alike -- and none of the memory Rust never allocated: GPU driver
// allocations, mapped asset files, thread stacks, and the binary image itself.
// The gap between `MemStats::live_bytes` and the process resident size is that
// non-Rust remainder, not untracked engine waste. The ledger is the other half
// of that story: it explains a portion of both realms by name, and what it
// explains is always a floor, since it holds only what someone reports.
//
// GPU memory is accounted here and allocated elsewhere, deliberately. A device
// allocator returns a heap and an offset rather than a pointer, its frees must
// wait for frames in flight to retire, and its placement rules differ per
// backend; that belongs behind concinnity-device. What both sides share is the
// vocabulary they report into, which is what lets one readout show RAM and VRAM
// through the same lens.

#![no_std]

extern crate alloc;

#[cfg(test)]
extern crate std;

mod arena;
mod counters;
mod detail;
mod inline_vec;
mod ledger;
mod pool;
mod tag;
mod tracking;

pub use arena::{Arena, ArenaVec};
pub use counters::{Counters, MemStats};
pub use detail::{CLASS_COUNT, SizeClass, SizeClasses, size_classes};
pub use inline_vec::{InlineVec, IntoIter as InlineVecIntoIter};
pub use ledger::{Ledger, LedgerSnapshot, TagUsage};
pub use pool::{Pool, PoolHandle};
pub use tag::{MemTag, Realm};
pub use tracking::TrackingAlloc;

static LEDGER: Ledger = Ledger::new();

// The tracked heap as of now, or `None` when no binary installed
// `TrackingAlloc` as its `#[global_allocator]`.
pub fn stats() -> Option<MemStats> {
    tracking::COUNTERS.snapshot()
}

// The process-wide tagged accounting. Subsystems report what they hold into it
// and readouts break the process down by tag; unlike `stats`, it is live
// whether or not a binary installed the tracking allocator.
pub fn ledger() -> &'static Ledger {
    &LEDGER
}

#[cfg(test)]
mod tests {
    use super::*;

    // The one global instance is reachable and is the same one every caller
    // reports into.
    #[test]
    fn the_global_ledger_is_shared_by_every_caller() {
        ledger().add(MemTag::Other, Realm::Host, 4_096);
        assert!(ledger().usage(MemTag::Other, Realm::Host).bytes >= 4_096);
        ledger().release(MemTag::Other, Realm::Host, 4_096);
    }
}

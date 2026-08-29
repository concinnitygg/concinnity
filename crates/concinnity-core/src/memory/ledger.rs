// Tagged byte accounting: who is holding memory, in which realm, against what
// budget.
//
// The global counters say how much the heap holds; they cannot say what for. A
// `GlobalAlloc` only ever sees a `Layout`, so attribution has to come from the
// subsystems that know what they are holding -- the streaming pools know their
// resident bytes, a GPU backend knows what it placed on the device -- and they
// report it here. Host and device reports share one vocabulary so a readout can
// show RAM and VRAM through the same lens.
//
// Reports are per-pool events (a load, an eviction, a budget change), not
// per-allocation, so one atomic per tag is not a hot line and needs none of the
// sharding the global counters use.

use core::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::memory::tag::{MemTag, Realm};

// One tag's counters in one realm. A budget of zero is "no budget set": a
// subsystem budgeted at zero bytes is not a case worth modelling.
struct Cell {
    bytes: AtomicU64,
    peak: AtomicU64,
    budget: AtomicU64,
}

impl Cell {
    const fn new() -> Self {
        Self {
            bytes: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            budget: AtomicU64::new(0),
        }
    }

    fn observe(&self, bytes: u64) {
        self.peak.fetch_max(bytes, Relaxed);
    }
}

// What one tag holds in one realm, as read at a moment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TagUsage {
    pub tag: MemTag,
    pub realm: Realm,
    pub bytes: u64,
    // High-water mark of `bytes` since process start.
    pub peak_bytes: u64,
    // The ceiling this tag is expected to stay under, when one was declared.
    pub budget: Option<u64>,
}

impl TagUsage {
    // Nothing reported. Distinct from a reported zero only in that no reporter
    // has touched the tag, which is why a readout hides it rather than drawing
    // an empty row.
    pub const fn empty(tag: MemTag, realm: Realm) -> Self {
        Self {
            tag,
            realm,
            bytes: 0,
            peak_bytes: 0,
            budget: None,
        }
    }

    pub(crate) fn is_reported(&self) -> bool {
        self.bytes > 0 || self.peak_bytes > 0 || self.budget.is_some()
    }

    pub fn over_budget(&self) -> bool {
        self.budget.is_some_and(|b| self.bytes > b)
    }

    // How much of the budget is spent, or `None` when the tag has no budget.
    pub fn fraction(&self) -> Option<f32> {
        self.budget
            .filter(|&b| b > 0)
            .map(|b| self.bytes as f32 / b as f32)
    }
}

/// Every tag's usage in every realm, read in one pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LedgerSnapshot {
    entries: [TagUsage; MemTag::COUNT * Realm::COUNT],
}

impl LedgerSnapshot {
    /// One tag's usage in one realm.
    pub fn get(&self, tag: MemTag, realm: Realm) -> TagUsage {
        self.entries[entry_index(tag, realm)]
    }

    /// The tags something has reported into, in realm order. What a readout
    /// lists: an unreported tag is absent rather than a row of zeroes.
    pub fn reported(&self, realm: Realm) -> impl Iterator<Item = TagUsage> + '_ {
        MemTag::ALL
            .into_iter()
            .map(move |tag| self.get(tag, realm))
            .filter(TagUsage::is_reported)
    }

    /// Bytes attributed across every tag in `realm`. Always a floor on the
    /// realm's real usage: it counts what reporters explain, not everything
    /// allocated.
    pub fn realm_bytes(&self, realm: Realm) -> u64 {
        MemTag::ALL
            .into_iter()
            .map(|tag| self.get(tag, realm).bytes)
            .sum()
    }

    /// Whether nothing has reported into any tag.
    pub fn is_empty(&self) -> bool {
        !self.entries.iter().any(TagUsage::is_reported)
    }
}

impl Default for LedgerSnapshot {
    fn default() -> Self {
        Self {
            entries: core::array::from_fn(|i| {
                TagUsage::empty(MemTag::ALL[i / Realm::COUNT], Realm::ALL[i % Realm::COUNT])
            }),
        }
    }
}

const fn entry_index(tag: MemTag, realm: Realm) -> usize {
    tag.index() * Realm::COUNT + realm.index()
}

/// The tagged accounting itself. One process-global instance backs the engine
/// (`crate::memory::ledger()`); the type is public so the accounting is testable on its
/// own instance.
pub struct Ledger {
    cells: [Cell; MemTag::COUNT * Realm::COUNT],
}

impl Ledger {
    /// A ledger with every cell at zero and no budgets set.
    pub const fn new() -> Self {
        Self {
            cells: [const { Cell::new() }; MemTag::COUNT * Realm::COUNT],
        }
    }

    fn cell(&self, tag: MemTag, realm: Realm) -> &Cell {
        &self.cells[entry_index(tag, realm)]
    }

    /// Take on `bytes` under `tag`. For a subsystem that grows and shrinks in
    /// increments; one that knows its total should `set` it instead.
    pub fn add(&self, tag: MemTag, realm: Realm, bytes: u64) {
        let cell = self.cell(tag, realm);
        let total = cell.bytes.fetch_add(bytes, Relaxed).saturating_add(bytes);
        cell.observe(total);
    }

    /// Give up `bytes` under `tag`. Saturates at zero: a reporter that
    /// double-releases should read as holding nothing, never as holding an
    /// enormous amount.
    pub fn release(&self, tag: MemTag, realm: Realm, bytes: u64) {
        let cell = self.cell(tag, realm);
        let _ = cell
            .bytes
            .fetch_update(Relaxed, Relaxed, |held| Some(held.saturating_sub(bytes)));
    }

    /// Declare the tag's whole holding. Idempotent, so a pool that recomputes
    /// its resident bytes each frame cannot drift the way paired add/release
    /// calls can.
    pub fn set(&self, tag: MemTag, realm: Realm, bytes: u64) {
        let cell = self.cell(tag, realm);
        cell.bytes.store(bytes, Relaxed);
        cell.observe(bytes);
    }

    /// The ceiling the tag is expected to stay under. `None` clears it.
    pub fn set_budget(&self, tag: MemTag, realm: Realm, budget: Option<u64>) {
        self.cell(tag, realm)
            .budget
            .store(budget.unwrap_or(0), Relaxed);
    }

    /// Set (or clear, with `None`) the byte budget for one tag and realm.
    pub fn usage(&self, tag: MemTag, realm: Realm) -> TagUsage {
        let cell = self.cell(tag, realm);
        let budget = cell.budget.load(Relaxed);
        TagUsage {
            tag,
            realm,
            bytes: cell.bytes.load(Relaxed),
            peak_bytes: cell.peak.load(Relaxed),
            budget: (budget > 0).then_some(budget),
        }
    }

    /// One tag's current usage in one realm.
    pub fn snapshot(&self) -> LedgerSnapshot {
        LedgerSnapshot {
            entries: core::array::from_fn(|i| {
                self.usage(MemTag::ALL[i / Realm::COUNT], Realm::ALL[i % Realm::COUNT])
            }),
        }
    }

    /// Drop every report and budget, including the peaks. For a host tearing a
    /// world down and building another, whose old numbers describe nothing.
    pub fn clear(&self) {
        for cell in &self.cells {
            cell.bytes.store(0, Relaxed);
            cell.peak.store(0, Relaxed);
            cell.budget.store(0, Relaxed);
        }
    }
}

impl Default for Ledger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_untouched_ledger_reports_nothing() {
        let ledger = Ledger::new();
        let snap = ledger.snapshot();
        assert!(snap.is_empty());
        assert_eq!(snap.reported(Realm::Host).count(), 0);
        assert_eq!(snap.realm_bytes(Realm::Device), 0);
    }

    #[test]
    fn increments_accumulate_and_release_gives_them_back() {
        let ledger = Ledger::new();
        ledger.add(MemTag::Audio, Realm::Host, 1024);
        ledger.add(MemTag::Audio, Realm::Host, 512);
        assert_eq!(ledger.usage(MemTag::Audio, Realm::Host).bytes, 1536);

        ledger.release(MemTag::Audio, Realm::Host, 512);
        assert_eq!(ledger.usage(MemTag::Audio, Realm::Host).bytes, 1024);
    }

    // A reporter that gives back more than it took reads as holding nothing.
    // The alternative -- wrapping past zero -- would show as exabytes held.
    #[test]
    fn releasing_more_than_held_saturates_at_zero() {
        let ledger = Ledger::new();
        ledger.add(MemTag::Meshes, Realm::Device, 100);
        ledger.release(MemTag::Meshes, Realm::Device, 4096);
        assert_eq!(ledger.usage(MemTag::Meshes, Realm::Device).bytes, 0);
    }

    // The idempotent form: a pool republishing its total every frame must not
    // accumulate.
    #[test]
    fn set_replaces_rather_than_accumulates() {
        let ledger = Ledger::new();
        ledger.set(MemTag::Textures, Realm::Device, 8_000);
        ledger.set(MemTag::Textures, Realm::Device, 6_000);
        assert_eq!(ledger.usage(MemTag::Textures, Realm::Device).bytes, 6_000);
    }

    #[test]
    fn peak_holds_the_high_water_mark_across_both_report_styles() {
        let ledger = Ledger::new();
        ledger.set(MemTag::Textures, Realm::Device, 9_000);
        ledger.set(MemTag::Textures, Realm::Device, 1_000);
        ledger.add(MemTag::Textures, Realm::Device, 500);

        let usage = ledger.usage(MemTag::Textures, Realm::Device);
        assert_eq!(usage.bytes, 1_500);
        assert_eq!(usage.peak_bytes, 9_000);
    }

    #[test]
    fn a_budget_reads_back_and_clears() {
        let ledger = Ledger::new();
        ledger.set_budget(MemTag::Chunks, Realm::Device, Some(4_096));
        assert_eq!(
            ledger.usage(MemTag::Chunks, Realm::Device).budget,
            Some(4_096)
        );

        ledger.set_budget(MemTag::Chunks, Realm::Device, None);
        assert_eq!(ledger.usage(MemTag::Chunks, Realm::Device).budget, None);
    }

    #[test]
    fn over_budget_and_fraction_track_the_ceiling() {
        let ledger = Ledger::new();
        ledger.set_budget(MemTag::Meshes, Realm::Device, Some(1_000));
        ledger.set(MemTag::Meshes, Realm::Device, 500);
        let under = ledger.usage(MemTag::Meshes, Realm::Device);
        assert!(!under.over_budget());
        assert_eq!(under.fraction(), Some(0.5));

        ledger.set(MemTag::Meshes, Realm::Device, 1_200);
        assert!(ledger.usage(MemTag::Meshes, Realm::Device).over_budget());
    }

    // An unbudgeted tag has no fraction to draw; a readout shows the bytes
    // alone rather than inventing a scale.
    #[test]
    fn an_unbudgeted_tag_has_no_fraction() {
        let ledger = Ledger::new();
        ledger.set(MemTag::Scratch, Realm::Host, 4_096);
        let usage = ledger.usage(MemTag::Scratch, Realm::Host);
        assert_eq!(usage.fraction(), None);
        assert!(!usage.over_budget());
    }

    // Host and device are separate accounts under one tag: a streamed texture
    // costs device bytes without touching the host row.
    #[test]
    fn the_realms_are_counted_separately() {
        let ledger = Ledger::new();
        ledger.set(MemTag::Textures, Realm::Device, 2_048);
        ledger.set(MemTag::Textures, Realm::Host, 64);

        let snap = ledger.snapshot();
        assert_eq!(snap.get(MemTag::Textures, Realm::Device).bytes, 2_048);
        assert_eq!(snap.get(MemTag::Textures, Realm::Host).bytes, 64);
        assert_eq!(snap.realm_bytes(Realm::Device), 2_048);
        assert_eq!(snap.realm_bytes(Realm::Host), 64);
    }

    // A readout lists reported tags only, in the fixed vocabulary order, so
    // rows never reorder underneath a reader.
    #[test]
    fn reported_lists_touched_tags_in_vocabulary_order() {
        let ledger = Ledger::new();
        ledger.set(MemTag::Chunks, Realm::Device, 3);
        ledger.set(MemTag::Textures, Realm::Device, 1);
        ledger.set(MemTag::Meshes, Realm::Device, 2);
        ledger.set(MemTag::Audio, Realm::Host, 9);

        let tags: std::vec::Vec<MemTag> = ledger
            .snapshot()
            .reported(Realm::Device)
            .map(|u| u.tag)
            .collect();
        assert_eq!(tags, [MemTag::Textures, MemTag::Meshes, MemTag::Chunks]);
    }

    // A tag that has held memory stays listed at zero: "loaded nothing right
    // now" is a different reading from "nobody reports this".
    #[test]
    fn a_tag_that_dropped_to_zero_stays_reported() {
        let ledger = Ledger::new();
        ledger.set(MemTag::Textures, Realm::Device, 4_096);
        ledger.set(MemTag::Textures, Realm::Device, 0);

        let snap = ledger.snapshot();
        assert!(snap.get(MemTag::Textures, Realm::Device).is_reported());
        assert_eq!(snap.reported(Realm::Device).count(), 1);
    }

    #[test]
    fn clear_drops_every_report_and_budget() {
        let ledger = Ledger::new();
        ledger.set(MemTag::Textures, Realm::Device, 4_096);
        ledger.set_budget(MemTag::Textures, Realm::Device, Some(8_192));
        ledger.clear();

        assert!(ledger.snapshot().is_empty());
        assert_eq!(ledger.usage(MemTag::Textures, Realm::Device).peak_bytes, 0);
    }

    // Every cell must be its own account; an indexing slip would show as one
    // tag's bytes appearing under another.
    #[test]
    fn every_tag_and_realm_addresses_its_own_cell() {
        let ledger = Ledger::new();
        for (i, tag) in MemTag::ALL.into_iter().enumerate() {
            for (j, realm) in Realm::ALL.into_iter().enumerate() {
                ledger.set(tag, realm, (i * Realm::COUNT + j + 1) as u64);
            }
        }
        let snap = ledger.snapshot();
        for (i, tag) in MemTag::ALL.into_iter().enumerate() {
            for (j, realm) in Realm::ALL.into_iter().enumerate() {
                let usage = snap.get(tag, realm);
                assert_eq!(usage.bytes, (i * Realm::COUNT + j + 1) as u64);
                assert_eq!(usage.tag, tag);
                assert_eq!(usage.realm, realm);
            }
        }
    }
}

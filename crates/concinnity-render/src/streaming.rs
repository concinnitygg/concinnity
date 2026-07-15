// src/streaming.rs
//
// Asset-streaming policy core.
//
// Pure decision logic: given each streamable item's priority score (camera
// distance), a per-frame load budget, and a cap on how many items may be
// resident at once, this decides *which* items to load and *which* to evict.
// It performs no I/O, spawns no threads, and touches no backend.
//
// This module is deliberately written against `core` + `alloc` constructs
// only (`Vec`, primitives, slices) so it can move into a future `no_std`
// client runtime unchanged. The `std`-side half -- the background fetch
// thread, the channels, and the GPU upload -- lives in concinnity-engine's
// `app::texture_stream`. Keep that boundary: no `std::`-only types
// (threads, files, `HashMap`, `Instant`, ...) belong in this file.

// Residency state of a single streamable item.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StreamState {
    // Not on the GPU; eligible to be loaded.
    Unloaded,
    // A background load has been dispatched but has not completed.
    Pending,
    // On the GPU and ready to sample.
    Resident,
}

#[derive(Clone, Copy, Debug)]
struct Item {
    state: StreamState,
    // Priority score; lower = more urgent. The driver feeds squared camera
    // distance, so "closer to the camera" sorts first and no `sqrt` (which
    // lives in `std`, not `core`) is needed here.
    score: f32,
    // Frame this item was last referenced; the LRU tiebreak when two resident
    // items have an equal score during eviction.
    last_touch: u64,
    // Resident GPU footprint in bytes, reported by the driver on load
    // completion. Zero until the item first becomes Resident; retained as the
    // re-load estimate after an eviction. Only counted while Resident, so a
    // stale weight on an Unloaded item never inflates `resident_bytes`.
    bytes: u64,
}

// The load / evict decisions produced by one [`StreamPlanner::plan`] call.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct StreamPlan {
    // Item ids whose background load should be dispatched this frame.
    pub to_load: Vec<usize>,
    // Item ids that should be evicted from the GPU this frame.
    pub to_evict: Vec<usize>,
}

// Decides what to stream in and out of a fixed-size residency pool.
//
// The planner owns only residency *bookkeeping*: it never reads or writes a
// GPU resource. Each frame the driver updates scores, calls [`plan`], and
// reports completed loads back via [`mark_resident`].
//
// [`plan`]: StreamPlanner::plan
// [`mark_resident`]: StreamPlanner::mark_resident
pub struct StreamPlanner {
    items: Vec<Item>,
    // Max number of loads `plan` will dispatch in a single call.
    load_budget: usize,
    // Max number of items allowed Resident-or-Pending simultaneously. Once the
    // pool is full a load can only proceed by evicting a lower-priority item.
    resident_cap: usize,
    // Optional cap on total resident bytes. When `Some(b)`, `plan` treats the
    // pool as full whenever loading a candidate would push resident bytes past
    // `b`, evicting farther residents until it fits. `None` (the default)
    // disables byte accounting entirely, leaving the count-only policy.
    byte_budget: Option<u64>,
}

impl StreamPlanner {
    // Create a planner tracking `count` items, all initially `Unloaded`.
    //
    // `load_budget` and `resident_cap` are both clamped to at least 1 so a
    // zero from a misconfigured asset cannot wedge streaming permanently.
    pub fn new(count: usize, load_budget: usize, resident_cap: usize) -> Self {
        Self {
            items: vec![
                Item {
                    state: StreamState::Unloaded,
                    score: 0.0,
                    last_touch: 0,
                    bytes: 0,
                };
                count
            ],
            load_budget: load_budget.max(1),
            resident_cap: resident_cap.max(1),
            byte_budget: None,
        }
    }

    // Set (or clear with `None`) the total resident-byte budget. `None` keeps
    // the count-only policy; `Some(b)` additionally evicts to hold resident
    // bytes at or under `b`. Off by default so worlds that never set it behave
    // exactly as the count-only planner.
    pub fn set_byte_budget(&mut self, budget: Option<u64>) {
        self.byte_budget = budget;
    }

    // Number of tracked items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    // Whether the planner tracks no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    // Residency state of item `id`, or `None` if `id` is out of range.
    pub fn state(&self, id: usize) -> Option<StreamState> {
        self.items.get(id).map(|i| i.state)
    }

    // Set item `id`'s priority score (lower = loaded sooner / evicted later).
    // Out-of-range ids are ignored.
    pub fn set_score(&mut self, id: usize, score: f32) {
        if let Some(item) = self.items.get_mut(id) {
            item.score = score;
        }
    }

    // Record that item `id` was referenced on `frame`. Refreshes the LRU
    // tiebreak used when evicting equally-scored resident items.
    pub fn touch(&mut self, id: usize, frame: u64) {
        if let Some(item) = self.items.get_mut(id) {
            item.last_touch = frame;
        }
    }

    // Report that a dispatched load for item `id` has completed and the
    // resource is now on the GPU, occupying `bytes` of GPU memory. The driver
    // knows the exact resident size at completion (decoded pixel bytes, or
    // vertex + index buffer bytes); `bytes` may be 0 when the size is unknown
    // or nothing was actually uploaded (e.g. a failed fetch left a placeholder).
    pub fn mark_resident(&mut self, id: usize, frame: u64, bytes: u64) {
        if let Some(item) = self.items.get_mut(id) {
            item.state = StreamState::Resident;
            item.last_touch = frame;
            item.bytes = bytes;
        }
    }

    // Force item `id` back to `Unloaded` (e.g. after a failed load that
    // should be retried). Out-of-range ids are ignored. The item's last-known
    // byte weight is retained as the estimate for a future re-load; it no
    // longer counts toward `resident_bytes` while Unloaded.
    pub fn mark_unloaded(&mut self, id: usize) {
        if let Some(item) = self.items.get_mut(id) {
            item.state = StreamState::Unloaded;
        }
    }

    // Total bytes of all currently Resident items, for diagnostics and the
    // byte-budget policy. Pending and Unloaded items are excluded.
    pub fn resident_bytes(&self) -> u64 {
        self.items
            .iter()
            .filter(|it| it.state == StreamState::Resident)
            .map(|it| it.bytes)
            .sum()
    }

    // `(resident, pending, unloaded)` item counts, for diagnostics.
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut resident = 0;
        let mut pending = 0;
        let mut unloaded = 0;
        for item in &self.items {
            match item.state {
                StreamState::Resident => resident += 1,
                StreamState::Pending => pending += 1,
                StreamState::Unloaded => unloaded += 1,
            }
        }
        (resident, pending, unloaded)
    }

    // Decide which items to load and evict this frame.
    //
    // `Unloaded` items are considered best-score-first. While the pool has
    // spare capacity -- under both the count cap and (when set) the byte budget
    // -- they are simply scheduled to load. Once the pool is full a candidate
    // can still load by evicting worst-scored residents, but only ones strictly
    // farther than the candidate, so equal-priority items never churn. A large
    // candidate may evict several small residents to fit under the byte budget.
    // At most `load_budget` loads are scheduled per call.
    //
    // With no byte budget set this reduces exactly to the count-only policy.
    //
    // This method mutates planner state: scheduled items become `Pending` and
    // evicted items become `Unloaded`, so a later `plan` call in the same
    // frame (or the next frame) will not re-pick them.
    pub fn plan(&mut self) -> StreamPlan {
        let mut plan = StreamPlan::default();

        // Candidate loads: every Unloaded item, best (lowest) score first.
        let mut candidates: Vec<usize> = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, it)| it.state == StreamState::Unloaded)
            .map(|(id, _)| id)
            .collect();
        candidates.sort_by(|&a, &b| {
            self.items[a]
                .score
                .partial_cmp(&self.items[b].score)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        for &id in &candidates {
            if plan.to_load.len() >= self.load_budget {
                break;
            }
            let cand_score = self.items[id].score;
            let cand_bytes = self.items[id].bytes;

            // Tentatively pick victims -- worst-scored resident first, and only
            // ones strictly farther than the candidate -- until the candidate
            // would fit under both the count cap and the byte budget. Track the
            // occupancy and resident-byte totals as if each tentative victim
            // were already gone; commit only if the candidate actually fits.
            let mut occ = self.occupied();
            let mut resident_bytes = self.resident_bytes();
            let mut victims: Vec<usize> = Vec::new();
            let fits = loop {
                let count_ok = occ < self.resident_cap;
                let byte_ok = self
                    .byte_budget
                    .is_none_or(|b| resident_bytes + cand_bytes <= b);
                if count_ok && byte_ok {
                    break true;
                }
                match self.worst_resident(&plan.to_evict, &victims) {
                    Some(victim) if self.items[victim].score > cand_score => {
                        occ -= 1;
                        resident_bytes -= self.items[victim].bytes;
                        victims.push(victim);
                    }
                    // No farther resident left to shed: this candidate cannot
                    // be placed.
                    _ => break false,
                }
            };

            if fits {
                for &victim in &victims {
                    self.items[victim].state = StreamState::Unloaded;
                    plan.to_evict.push(victim);
                }
                self.items[id].state = StreamState::Pending;
                plan.to_load.push(id);
            } else if self.byte_budget.is_none() {
                // Count-only: candidates are score-sorted, so if the best
                // remaining one cannot displace the worst resident, none can.
                break;
            }
            // Byte budget set: a later, smaller candidate may still fit, so
            // keep scanning rather than stopping here.
        }

        plan
    }

    // Items occupying (or about to occupy) a pool slot: everything not Unloaded.
    fn occupied(&self) -> usize {
        self.items
            .iter()
            .filter(|it| it.state != StreamState::Unloaded)
            .count()
    }

    // The resident item with the worst (highest) score, breaking ties toward
    // the least-recently-touched. Items in either `excluded` slice are skipped
    // so a single `plan` call never evicts the same victim twice (one slice is
    // the committed evictions, the other this candidate's tentative picks).
    fn worst_resident(&self, excluded: &[usize], tentative: &[usize]) -> Option<usize> {
        let mut worst: Option<usize> = None;
        for (id, item) in self.items.iter().enumerate() {
            if item.state != StreamState::Resident
                || excluded.contains(&id)
                || tentative.contains(&id)
            {
                continue;
            }
            match worst {
                None => worst = Some(id),
                Some(w) => {
                    let cur = &self.items[w];
                    let better_victim = item.score > cur.score
                        || (item.score == cur.score && item.last_touch < cur.last_touch);
                    if better_victim {
                        worst = Some(id);
                    }
                }
            }
        }
        worst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_planner_has_all_items_unloaded() {
        let p = StreamPlanner::new(3, 4, 8);
        assert_eq!(p.len(), 3);
        for id in 0..3 {
            assert_eq!(p.state(id), Some(StreamState::Unloaded));
        }
        assert_eq!(p.state(3), None);
        assert_eq!(p.counts(), (0, 0, 3));
    }

    #[test]
    fn zero_budget_and_cap_are_clamped_to_one() {
        let mut p = StreamPlanner::new(2, 0, 0);
        let plan = p.plan();
        // A budget/cap of 0 would wedge streaming; clamped to 1 it still moves.
        assert_eq!(plan.to_load.len(), 1);
    }

    #[test]
    fn plan_loads_nearest_items_first_within_budget() {
        let mut p = StreamPlanner::new(4, 2, 8);
        p.set_score(0, 30.0);
        p.set_score(1, 10.0);
        p.set_score(2, 20.0);
        p.set_score(3, 40.0);
        let plan = p.plan();
        // Budget is 2; the two lowest scores (ids 1 then 2) are picked in order.
        assert_eq!(plan.to_load, vec![1, 2]);
        assert!(plan.to_evict.is_empty());
        assert_eq!(p.state(1), Some(StreamState::Pending));
        assert_eq!(p.state(2), Some(StreamState::Pending));
        assert_eq!(p.state(0), Some(StreamState::Unloaded));
    }

    #[test]
    fn pending_items_are_not_re_dispatched() {
        let mut p = StreamPlanner::new(3, 1, 8);
        let first = p.plan();
        assert_eq!(first.to_load.len(), 1);
        let dispatched = first.to_load[0];
        let second = p.plan();
        assert!(!second.to_load.contains(&dispatched));
    }

    #[test]
    fn resident_cap_blocks_loading_when_no_eviction_is_worthwhile() {
        let mut p = StreamPlanner::new(3, 4, 2);
        // Two near items become resident.
        p.set_score(0, 1.0);
        p.set_score(1, 2.0);
        p.set_score(2, 99.0);
        let plan = p.plan();
        assert_eq!(plan.to_load, vec![0, 1]);
        p.mark_resident(0, 1, 0);
        p.mark_resident(1, 1, 0);
        // The far item cannot displace either resident; they are both closer.
        let plan = p.plan();
        assert!(plan.to_load.is_empty());
        assert!(plan.to_evict.is_empty());
    }

    #[test]
    fn closer_candidate_evicts_a_farther_resident() {
        let mut p = StreamPlanner::new(3, 4, 2);
        p.set_score(0, 50.0);
        p.set_score(1, 60.0);
        p.set_score(2, 99.0);
        let plan = p.plan();
        assert_eq!(plan.to_load, vec![0, 1]);
        p.mark_resident(0, 1, 0);
        p.mark_resident(1, 1, 0);
        // Item 2 walks closer than resident item 1.
        p.set_score(2, 10.0);
        let plan = p.plan();
        assert_eq!(plan.to_load, vec![2]);
        assert_eq!(plan.to_evict, vec![1]);
        assert_eq!(p.state(1), Some(StreamState::Unloaded));
        assert_eq!(p.state(2), Some(StreamState::Pending));
    }

    #[test]
    fn eviction_breaks_score_ties_toward_least_recently_touched() {
        let mut p = StreamPlanner::new(3, 4, 2);
        p.set_score(0, 5.0);
        p.set_score(1, 5.0);
        p.set_score(2, 99.0); // far away initially, so 0 and 1 become resident
        let plan = p.plan();
        assert_eq!(plan.to_load, vec![0, 1]);
        p.mark_resident(0, 1, 0);
        p.mark_resident(1, 1, 0);
        // Item 0 is referenced more recently than item 1.
        p.touch(0, 100);
        p.touch(1, 50);
        // A closer candidate forces one eviction; the staler resident loses.
        p.set_score(2, 1.0);
        let plan = p.plan();
        assert_eq!(plan.to_evict, vec![1]);
    }

    #[test]
    fn counts_track_state_transitions() {
        let mut p = StreamPlanner::new(3, 1, 8);
        assert_eq!(p.counts(), (0, 0, 3));
        let plan = p.plan();
        let id = plan.to_load[0];
        assert_eq!(p.counts(), (0, 1, 2));
        p.mark_resident(id, 1, 0);
        assert_eq!(p.counts(), (1, 0, 2));
        p.mark_unloaded(id);
        assert_eq!(p.counts(), (0, 0, 3));
    }

    #[test]
    fn empty_planner_plans_nothing() {
        let mut p = StreamPlanner::new(0, 4, 8);
        assert_eq!(p.len(), 0);
        assert_eq!(p.plan(), StreamPlan::default());
    }

    // Give an Unloaded item a known byte weight, as if it had been resident and
    // then evicted: `mark_resident` records the size, `mark_unloaded` frees the
    // slot but keeps the weight as the re-load estimate.
    fn seed_bytes(p: &mut StreamPlanner, id: usize, bytes: u64) {
        p.mark_resident(id, 0, bytes);
        p.mark_unloaded(id);
    }

    #[test]
    fn resident_bytes_sums_resident_items_only() {
        let mut p = StreamPlanner::new(3, 4, 8);
        assert_eq!(p.resident_bytes(), 0);
        p.mark_resident(0, 1, 100);
        p.mark_resident(1, 1, 250);
        assert_eq!(p.resident_bytes(), 350);
        // An unloaded item stops counting even though it keeps its weight.
        p.mark_unloaded(0);
        assert_eq!(p.resident_bytes(), 250);
    }

    #[test]
    fn no_byte_budget_ignores_item_bytes() {
        // Generous count cap, no byte budget: even huge items never evict.
        let mut p = StreamPlanner::new(3, 4, 8);
        p.set_score(0, 5.0);
        p.set_score(1, 6.0);
        p.mark_resident(0, 1, 10_000);
        p.mark_resident(1, 1, 10_000);
        seed_bytes(&mut p, 2, 10_000);
        p.set_score(2, 7.0);
        let plan = p.plan();
        // A count slot is free, so it simply loads with no eviction.
        assert_eq!(plan.to_load, vec![2]);
        assert!(plan.to_evict.is_empty());
    }

    #[test]
    fn byte_budget_evicts_worst_scored_resident_to_fit() {
        // Count cap is generous; the byte budget is the only pressure.
        let mut p = StreamPlanner::new(3, 4, 100);
        p.set_byte_budget(Some(100));
        // Two residents fill the 100-byte budget: a near one and a far one.
        p.set_score(0, 5.0);
        p.set_score(1, 80.0);
        p.mark_resident(0, 1, 50);
        p.mark_resident(1, 1, 50);
        assert_eq!(p.resident_bytes(), 100);
        // A mid-distance 50-byte candidate wants in.
        seed_bytes(&mut p, 2, 50);
        p.set_score(2, 10.0);
        let plan = p.plan();
        // It must evict the worst-scored resident (far id 1), not near id 0.
        assert_eq!(plan.to_evict, vec![1]);
        assert_eq!(plan.to_load, vec![2]);
    }

    #[test]
    fn count_cap_binds_first_for_small_items() {
        // Tight count cap, roomy byte budget: the count cap drives eviction.
        let mut p = StreamPlanner::new(3, 4, 2);
        p.set_byte_budget(Some(10_000));
        p.set_score(0, 10.0);
        p.set_score(1, 20.0);
        p.set_score(2, 99.0);
        let plan = p.plan();
        assert_eq!(plan.to_load, vec![0, 1]);
        p.mark_resident(0, 1, 8);
        p.mark_resident(1, 1, 8);
        // id 2 walks in closer than id 1. Resident bytes (16) are nowhere near
        // the 10_000 budget, so the full count cap is what forces the swap.
        seed_bytes(&mut p, 2, 8);
        p.set_score(2, 15.0);
        let plan = p.plan();
        assert_eq!(plan.to_evict, vec![1]);
        assert_eq!(plan.to_load, vec![2]);
        assert_eq!(p.resident_bytes(), 8); // only id 0 remains resident
    }

    #[test]
    fn byte_budget_binds_first_for_large_items() {
        // Roomy count cap, tight byte budget: bytes drive eviction.
        let mut p = StreamPlanner::new(3, 4, 100);
        p.set_byte_budget(Some(100));
        p.set_score(0, 10.0);
        p.set_score(1, 20.0);
        p.mark_resident(0, 1, 50);
        p.mark_resident(1, 1, 50); // 100 bytes: budget full, 2/100 slots used
        seed_bytes(&mut p, 2, 50);
        p.set_score(2, 15.0);
        let plan = p.plan();
        // 98 count slots are free, so only the byte budget can force this.
        assert_eq!(plan.to_evict, vec![1]);
        assert_eq!(plan.to_load, vec![2]);
        assert!(p.resident_bytes() + 50 <= 100);
    }

    #[test]
    fn large_candidate_evicts_multiple_farther_residents_but_not_a_closer_one() {
        let mut p = StreamPlanner::new(5, 4, 100);
        p.set_byte_budget(Some(100));
        // Four 20-byte residents: one near (score 5), three far (30/40/50).
        p.set_score(0, 5.0);
        p.set_score(1, 30.0);
        p.set_score(2, 40.0);
        p.set_score(3, 50.0);
        p.mark_resident(0, 1, 20);
        p.mark_resident(1, 1, 20);
        p.mark_resident(2, 1, 20);
        p.mark_resident(3, 1, 20);
        assert_eq!(p.resident_bytes(), 80);
        // A big 60-byte candidate (score 10) needs 40 bytes freed: it evicts
        // the two worst-scored residents (far id 3 then id 2), never the nearer
        // id 0 or id 1.
        seed_bytes(&mut p, 4, 60);
        p.set_score(4, 10.0);
        let plan = p.plan();
        assert_eq!(plan.to_load, vec![4]);
        assert_eq!(plan.to_evict, vec![3, 2]);
        assert_eq!(p.state(0), Some(StreamState::Resident));
        assert_eq!(p.state(1), Some(StreamState::Resident));
        assert!(p.resident_bytes() + 60 <= 100);
    }

    #[test]
    fn byte_budget_does_not_evict_a_resident_closer_than_the_candidate() {
        let mut p = StreamPlanner::new(2, 4, 100);
        p.set_byte_budget(Some(50));
        // A single near resident already fills the budget.
        p.set_score(0, 5.0);
        p.mark_resident(0, 1, 50);
        // A farther, large candidate cannot displace the closer resident.
        seed_bytes(&mut p, 1, 50);
        p.set_score(1, 99.0);
        let plan = p.plan();
        assert!(plan.to_load.is_empty());
        assert!(plan.to_evict.is_empty());
        assert_eq!(p.state(0), Some(StreamState::Resident));
    }
}

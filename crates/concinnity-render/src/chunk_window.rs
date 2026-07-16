// src/chunk_window.rs
//
// Sliding-window streaming policy for an infinite voxel world.
//
// `gfx::streaming::StreamPlanner` streams a *fixed* pool of items known at
// init (textures, build-time meshes). An infinite chunk world is different:
// the item set is unbounded and only a bounded *window* around the camera is
// ever resident. This module is that policy -- given the camera's chunk and a
// view radius it decides which chunks to load (nearest first, budget-limited)
// and which to evict (those that have fallen well outside the window).
//
// Two concentric bands: chunks within `near_radius` stream at full voxel
// detail; chunks beyond it but within `far_radius` stream as cheap coarse
// "impostors" (a low-poly surface mesh). As the camera moves a chunk crosses
// the near/far boundary and is *re-detailed* -- evicted and reloaded at the new
// detail. A small detail hysteresis keeps a chunk pacing across the boundary
// from thrashing. When `far_radius == near_radius` (the default) the far band
// is empty, so the window behaves exactly as the original single-detail one.
//
// Like `crate::streaming` and `crate::chunk_coord` this is written against
// `core` + `alloc` only -- no threads, no I/O, no `std` collections (a
// `BTreeMap`, not a `HashMap`) -- so it can move into a future `no_std`
// runtime unchanged. The `std`-side driver (background generation thread,
// GPU upload) lives in concinnity-engine's `app::chunk_stream`.

// `BTreeMap` is an `alloc` collection (re-exported here through `std`); a
// `HashMap` would pull in `std`-only hashing. `Vec` comes from the prelude.
use std::collections::BTreeMap;

use crate::chunk_coord::ChunkCoord;

// Extra chunk rings a chunk may drift beyond the view radius before it is
// evicted. The gap between the load radius and the evict radius is hysteresis:
// without it a chunk straddling the boundary would load and evict on
// alternating frames as the camera jitters across a chunk edge.
const EVICT_HYSTERESIS: i32 = 2;

// Extra rings a currently-full (Near) chunk may drift past `near_radius` before
// it is downgraded to a Far impostor. Without it a camera pacing back and forth
// across the near/far boundary would re-detail a chunk every step.
const DETAIL_HYSTERESIS: i32 = 1;

// Low-water mark (percent of the byte budget) the effective window regrows at.
// The window shrinks a ring whenever resident bytes exceed the budget and only
// regrows once they fall back under this fraction of it; the gap is hysteresis
// that stops the radius oscillating a ring in and out at the boundary.
const BYTE_BUDGET_LOW_PCT: u64 = 75;

// Residency state of a chunk the window is currently tracking.
//
// A chunk not in the window's map is simply unloaded -- there is no explicit
// `Unloaded` state, since the grid is infinite and tracking every never-seen
// chunk would be unbounded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChunkState {
    // A background generation+upload has been dispatched but not completed.
    Pending,
    // The chunk's mesh is resident on the GPU.
    Resident,
}

// Which representation a chunk is streamed at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChunkDetail {
    // Full voxel geometry for chunks within `near_radius`.
    Near,
    // A coarse distant-impostor surface mesh for chunks beyond `near_radius` but
    // within `far_radius`.
    Far,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Slot {
    state: ChunkState,
    detail: ChunkDetail,
    // Resident GPU footprint in bytes, reported by the driver on load
    // completion. Zero while Pending; only counted toward `resident_bytes`
    // once the slot is Resident.
    bytes: u64,
}

// The load / evict decisions produced by one [`ChunkWindow::plan`] call.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ChunkPlan {
    // Chunks whose background load should be dispatched this frame, nearest
    // to the camera first, each tagged with the detail to generate it at.
    // Already marked [`ChunkState::Pending`].
    pub to_load: Vec<(ChunkCoord, ChunkDetail)>,
    // Chunks removed from the GPU this frame: those that fell outside the
    // evict radius, plus those crossing the near/far boundary (which reload at
    // the new detail). Already dropped from the window's tracking map.
    pub to_evict: Vec<ChunkCoord>,
}

// Decides which chunks stream in and out of the camera-centred view window,
// and at which detail.
//
// The window owns only residency *bookkeeping* -- it never generates a chunk
// or touches a GPU resource. Each frame the driver calls [`plan`] with the
// camera's current chunk, dispatches the loads, applies the evictions, and
// reports completed loads back via [`mark_resident`].
//
// [`plan`]: ChunkWindow::plan
// [`mark_resident`]: ChunkWindow::mark_resident
pub struct ChunkWindow {
    // Tracked chunks only: a chunk absent from the map is unloaded.
    states: BTreeMap<ChunkCoord, Slot>,
    // Chebyshev radius (in chunks) of the full-detail square window.
    near_radius: i32,
    // Chebyshev radius of the outer impostor window (>= near_radius).
    far_radius: i32,
    // Max chunk loads dispatched per `plan` call.
    load_budget: usize,
    // Optional cap on total resident chunk bytes. When `Some(b)`, `plan`
    // clamps the effective window down (shrinking the far impostor band before
    // the near full-detail band) whenever resident bytes exceed `b`, evicting
    // the outermost chunks until they fit. `None` (the default) disables byte
    // accounting entirely, leaving the pure radius-based window.
    byte_budget: Option<u64>,
    // Rings the effective window is currently shrunk by under byte pressure,
    // in `[0, far_radius]`. Zero (the default, and always so with no byte
    // budget) means the effective window is exactly the configured one. Each
    // ring shrinks the far band first, then the near band once the two meet.
    shrink: i32,
}

impl ChunkWindow {
    // A window with a full-detail radius of `near_radius`, an outer impostor
    // radius of `far_radius`, and a per-frame load budget of `load_budget`
    // chunks.
    //
    // `near_radius` is floored at 0 (a lone chunk), `far_radius` at
    // `near_radius` (so it never undercuts the full-detail band; equal means
    // "no impostors"), and `load_budget` at 1 so a stray 0 cannot wedge
    // streaming permanently.
    pub fn new(near_radius: i32, far_radius: i32, load_budget: usize) -> Self {
        let near_radius = near_radius.max(0);
        let far_radius = far_radius.max(near_radius);
        Self {
            states: BTreeMap::new(),
            near_radius,
            far_radius,
            load_budget: load_budget.max(1),
            byte_budget: None,
            shrink: 0,
        }
    }

    // Set (or clear with `None`) the total resident-chunk-byte budget. `None`
    // keeps the pure radius-based window; `Some(b)` additionally clamps the
    // effective view radius down until resident bytes fit `b`. Off by default
    // so worlds that never set it behave exactly as the radius-only window.
    pub fn set_byte_budget(&mut self, budget: Option<u64>) {
        self.byte_budget = budget;
    }

    // The active resident-byte budget, or `None` when byte accounting is off
    // (the pure radius window). For diagnostics.
    pub fn byte_budget(&self) -> Option<u64> {
        self.byte_budget
    }

    // The detail a chunk should currently be at, given its distance from the
    // camera, the effective full-detail radius, and (for hysteresis) the detail
    // it is currently tracked at.
    fn target_detail(
        &self,
        c: ChunkCoord,
        camera: ChunkCoord,
        current: Option<ChunkDetail>,
        near_radius: i32,
    ) -> ChunkDetail {
        let d = c.chebyshev_distance(camera);
        if d <= near_radius {
            ChunkDetail::Near
        } else if matches!(current, Some(ChunkDetail::Near)) && d <= near_radius + DETAIL_HYSTERESIS
        {
            // A currently-full chunk keeps full detail through the hysteresis
            // band rather than re-detailing the instant it leaves near_radius.
            ChunkDetail::Near
        } else {
            ChunkDetail::Far
        }
    }

    // The effective `(near_radius, far_radius)` after applying the byte-pressure
    // `shrink`. A shrink first trims the far impostor band down to the near
    // band, then shrinks the near band (with the far band held equal to it), so
    // cheap distant impostors are dropped before full-detail near chunks.
    fn effective_radii(&self) -> (i32, i32) {
        let far_span = self.far_radius - self.near_radius;
        if self.shrink <= far_span {
            (self.near_radius, self.far_radius - self.shrink)
        } else {
            let near = (self.near_radius - (self.shrink - far_span)).max(0);
            (near, near)
        }
    }

    // Adjust the byte-pressure `shrink` from resident bytes vs the budget, with
    // hysteresis so the effective radius does not oscillate at the boundary. A
    // no-op (shrink pinned to 0) when no byte budget is set.
    fn adjust_shrink(&mut self) {
        let Some(budget) = self.byte_budget else {
            self.shrink = 0;
            return;
        };
        let resident = self.resident_bytes();
        if resident > budget {
            // Over budget: shrink one more ring (far impostor band first, then
            // the near full-detail band), never past a lone camera chunk.
            self.shrink = (self.shrink + 1).min(self.far_radius);
            return;
        }
        // Under budget: regrow one ring only once resident bytes fall
        // comfortably below the low-water margin AND no loads are in flight
        // (a pending load's bytes are not yet counted, so regrowing before it
        // lands could overshoot and force an immediate re-shrink). The margin
        // plus the in-flight gate are the hysteresis that keeps the effective
        // radius from oscillating ring in and out frame to frame.
        if self.shrink == 0 {
            return;
        }
        let pending = self
            .states
            .values()
            .filter(|slot| slot.state == ChunkState::Pending)
            .count();
        if pending == 0 && resident.saturating_mul(100) < budget.saturating_mul(BYTE_BUDGET_LOW_PCT)
        {
            self.shrink -= 1;
        }
    }

    // Decide this frame's chunk loads and evictions for a camera in chunk
    // `camera`.
    //
    // First reconciles the effective view radius against the byte budget (a
    // no-op when none is set), then evicts every tracked chunk now beyond the
    // effective evict radius, re-details any tracked chunk that has crossed the
    // near/far boundary (evict + reload at the new detail), and dispatches the
    // nearest in-window chunks not yet tracked, up to the load budget, marking
    // each `Pending`.
    pub fn plan(&mut self, camera: ChunkCoord) -> ChunkPlan {
        // 0. Reconcile the effective window with the byte budget. The clamp only
        //    ever shrinks below the configured radii, so a world with no budget
        //    (shrink pinned to 0) plans exactly the configured window.
        self.adjust_shrink();
        let (near_radius, far_radius) = self.effective_radii();
        let evict_radius = far_radius + EVICT_HYSTERESIS;

        let mut to_evict: Vec<ChunkCoord> = Vec::new();

        // 1. Evict chunks that have drifted past the evict radius.
        let gone: Vec<ChunkCoord> = self
            .states
            .keys()
            .copied()
            .filter(|c| c.chebyshev_distance(camera) > evict_radius)
            .collect();
        for c in &gone {
            self.states.remove(c);
        }
        to_evict.extend_from_slice(&gone);

        // 2. Re-detail: a tracked chunk whose target detail no longer matches
        //    is dropped + evicted so the candidate scan reloads it at the new
        //    detail (a near<->far crossing as the camera moves).
        let redetail: Vec<ChunkCoord> = self
            .states
            .iter()
            .filter(|(c, slot)| {
                self.target_detail(**c, camera, Some(slot.detail), near_radius) != slot.detail
            })
            .map(|(c, _)| *c)
            .collect();
        for c in &redetail {
            self.states.remove(c);
        }
        to_evict.extend_from_slice(&redetail);

        // 3. Collect in-window chunks (within far_radius) not yet tracked,
        //    nearest first.
        let mut candidates: Vec<ChunkCoord> = Vec::new();
        for dz in -far_radius..=far_radius {
            for dx in -far_radius..=far_radius {
                let c = camera.offset(dx, dz);
                if !self.states.contains_key(&c) {
                    candidates.push(c);
                }
            }
        }
        candidates.sort_unstable_by(|a, b| {
            a.sq_distance(camera)
                .cmp(&b.sq_distance(camera))
                // Stable tiebreak on the coordinate so the plan is deterministic.
                .then(a.cmp(b))
        });
        candidates.truncate(self.load_budget);

        let mut to_load = Vec::with_capacity(candidates.len());
        for &c in &candidates {
            let detail = self.target_detail(c, camera, None, near_radius);
            self.states.insert(
                c,
                Slot {
                    state: ChunkState::Pending,
                    detail,
                    bytes: 0,
                },
            );
            to_load.push((c, detail));
        }

        to_evict.sort_unstable();
        ChunkPlan { to_load, to_evict }
    }

    // Mark a dispatched chunk resident once its mesh is on the GPU, recording
    // its GPU footprint in `bytes` (the decoded vertex + index buffer size).
    // `bytes` may be 0 when nothing was uploaded (e.g. a generation that
    // deterministically failed and is being retired to stop retrying it).
    //
    // A no-op if the chunk is no longer tracked -- the camera may have moved
    // far enough to evict it while its load was still in flight.
    pub fn mark_resident(&mut self, coord: ChunkCoord, bytes: u64) {
        if let Some(slot) = self.states.get_mut(&coord) {
            slot.state = ChunkState::Resident;
            slot.bytes = bytes;
        }
    }

    // Total bytes of all currently Resident chunks, for diagnostics and the
    // byte-budget clamp. Pending chunks (not yet uploaded) are excluded.
    pub fn resident_bytes(&self) -> u64 {
        self.states
            .values()
            .filter(|slot| slot.state == ChunkState::Resident)
            .map(|slot| slot.bytes)
            .sum()
    }

    // Drop `coord` from tracking so a later [`plan`](Self::plan) will
    // re-dispatch it.
    //
    // The driver calls this when a dispatch could not be delivered to the
    // background worker, so the chunk is retried rather than stuck `Pending`.
    pub fn forget(&mut self, coord: ChunkCoord) {
        self.states.remove(&coord);
    }

    // Whether the window is still tracking `coord` (pending or resident).
    //
    // The driver checks this when a background load completes: a chunk
    // evicted mid-flight is no longer tracked and its mesh should be dropped.
    pub fn is_tracked(&self, coord: ChunkCoord) -> bool {
        self.states.contains_key(&coord)
    }

    // `(resident, pending)` chunk counts -- for diagnostics.
    pub fn counts(&self) -> (usize, usize) {
        let mut resident = 0;
        let mut pending = 0;
        for slot in self.states.values() {
            match slot.state {
                ChunkState::Resident => resident += 1,
                ChunkState::Pending => pending += 1,
            }
        }
        (resident, pending)
    }

    // `(near_resident, far_resident)` counts -- resident full chunks vs
    // resident impostors, for diagnostics / verifying the far band is active.
    pub fn counts_by_detail(&self) -> (usize, usize) {
        let mut near = 0;
        let mut far = 0;
        for slot in self.states.values() {
            if slot.state == ChunkState::Resident {
                match slot.detail {
                    ChunkDetail::Near => near += 1,
                    ChunkDetail::Far => far += 1,
                }
            }
        }
        (near, far)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cc(x: i32, z: i32) -> ChunkCoord {
        ChunkCoord::new(x, z)
    }

    // Coords in a plan's load list, detail dropped, for set-style assertions.
    fn load_coords(plan: &ChunkPlan) -> Vec<ChunkCoord> {
        plan.to_load.iter().map(|(c, _)| *c).collect()
    }

    // Fill the whole configured window at `camera`, marking every dispatched
    // chunk resident with `bytes`. Assumes a load budget covering the window.
    fn fill(w: &mut ChunkWindow, camera: ChunkCoord, bytes: u64) {
        for (c, _) in w.plan(camera).to_load {
            w.mark_resident(c, bytes);
        }
    }

    // Replan at a stationary `camera`, marking each newly loaded chunk resident,
    // until the byte clamp reaches a fixed point. Convergence is a streak of
    // plans that neither load nor evict: a single quiet plan is not enough,
    // because the effective radius shrinks two frames before the `+2` evict
    // hysteresis actually drops the outer ring. Panics if it never settles --
    // which doubles as the assertion that the clamp does not oscillate.
    fn settle(w: &mut ChunkWindow, camera: ChunkCoord, bytes: u64) {
        let mut quiet = 0;
        for _ in 0..200 {
            let plan = w.plan(camera);
            for (c, _) in &plan.to_load {
                w.mark_resident(*c, bytes);
            }
            if plan.to_load.is_empty() && plan.to_evict.is_empty() {
                quiet += 1;
                if quiet >= 4 {
                    return;
                }
            } else {
                quiet = 0;
            }
        }
        panic!("byte-budget clamp did not converge (oscillating?)");
    }

    #[test]
    fn plan_loads_nearest_in_window_chunks_within_budget() {
        // near=far=2 -> a 5x5 window of 25 chunks, impostors off; budget 4.
        let mut w = ChunkWindow::new(2, 2, 4);
        let plan = w.plan(cc(0, 0));
        assert!(plan.to_evict.is_empty());
        assert_eq!(plan.to_load.len(), 4);
        // The camera's own chunk is distance 0 -- it must be dispatched first.
        assert_eq!(plan.to_load[0], (cc(0, 0), ChunkDetail::Near));
        // Every dispatched chunk is within the load radius and full-detail.
        for (c, detail) in &plan.to_load {
            assert!(c.chebyshev_distance(cc(0, 0)) <= 2);
            assert_eq!(*detail, ChunkDetail::Near);
        }
    }

    #[test]
    fn plan_does_not_redispatch_tracked_chunks() {
        let mut w = ChunkWindow::new(3, 3, 100);
        let first = w.plan(cc(0, 0));
        // A generous budget loads the whole 7x7 window at once.
        assert_eq!(first.to_load.len(), 49);
        // Nothing left to dispatch on the next frame at the same position.
        let second = w.plan(cc(0, 0));
        assert!(second.to_load.is_empty());
        assert!(second.to_evict.is_empty());
    }

    #[test]
    fn plan_evicts_chunks_past_the_hysteresis_band() {
        let mut w = ChunkWindow::new(2, 2, 100);
        w.plan(cc(0, 0)); // load the 5x5 window around the origin
        // Move far enough that the origin chunk is past radius 2 + hysteresis 2.
        let plan = w.plan(cc(6, 0));
        assert!(plan.to_evict.contains(&cc(0, 0)));
    }

    #[test]
    fn evicted_chunk_can_be_reloaded_after_returning() {
        let mut w = ChunkWindow::new(1, 1, 100);
        w.plan(cc(0, 0));
        w.plan(cc(20, 0)); // evicts the origin window entirely
        assert!(!w.is_tracked(cc(0, 0)));
        let plan = w.plan(cc(0, 0));
        assert!(load_coords(&plan).contains(&cc(0, 0)));
    }

    #[test]
    fn mark_resident_promotes_a_pending_chunk() {
        let mut w = ChunkWindow::new(0, 0, 1);
        let plan = w.plan(cc(0, 0));
        assert_eq!(plan.to_load, vec![(cc(0, 0), ChunkDetail::Near)]);
        assert_eq!(w.counts(), (0, 1));
        w.mark_resident(cc(0, 0), 0);
        assert_eq!(w.counts(), (1, 0));
    }

    #[test]
    fn mark_resident_of_an_untracked_chunk_is_a_noop() {
        let mut w = ChunkWindow::new(0, 0, 1);
        w.mark_resident(cc(9, 9), 0); // never planned -- must not panic or insert
        assert_eq!(w.counts(), (0, 0));
        assert!(!w.is_tracked(cc(9, 9)));
    }

    #[test]
    fn forget_lets_a_chunk_be_redispatched() {
        let mut w = ChunkWindow::new(0, 0, 1);
        w.plan(cc(0, 0));
        assert!(w.is_tracked(cc(0, 0)));
        w.forget(cc(0, 0));
        assert!(!w.is_tracked(cc(0, 0)));
        let plan = w.plan(cc(0, 0));
        assert_eq!(plan.to_load, vec![(cc(0, 0), ChunkDetail::Near)]);
    }

    #[test]
    fn zero_radius_and_budget_are_floored() {
        // near floored to 0 (just the camera chunk), far to near, budget to 1.
        let mut w = ChunkWindow::new(-5, -5, 0);
        let plan = w.plan(cc(0, 0));
        assert_eq!(plan.to_load, vec![(cc(0, 0), ChunkDetail::Near)]);
    }

    #[test]
    fn far_band_chunks_load_as_impostors() {
        // near 1, far 3: the 3x3 core is full detail, the surrounding rings are
        // impostors. A generous budget loads the whole 7x7 window at once.
        let mut w = ChunkWindow::new(1, 3, 100);
        let plan = w.plan(cc(0, 0));
        assert_eq!(plan.to_load.len(), 49);
        for (c, detail) in &plan.to_load {
            let d = c.chebyshev_distance(cc(0, 0));
            let expected = if d <= 1 {
                ChunkDetail::Near
            } else {
                ChunkDetail::Far
            };
            assert_eq!(*detail, expected, "chunk {:?} at distance {}", c, d);
        }
    }

    #[test]
    fn crossing_the_boundary_redetails_a_chunk() {
        let mut w = ChunkWindow::new(1, 3, 100);
        // Load + resolve the whole window around the origin.
        let plan = w.plan(cc(0, 0));
        let crossing = cc(2, 0); // distance 2 -> Far impostor at the origin
        assert!(plan.to_load.contains(&(crossing, ChunkDetail::Far)));
        for (c, _) in plan.to_load.clone() {
            w.mark_resident(c, 0);
        }
        let (near0, far0) = w.counts_by_detail();
        assert!(near0 > 0 && far0 > 0);

        // Step toward `crossing` so it falls inside near_radius: it must be
        // re-detailed (evicted) and re-dispatched as Near.
        let plan = w.plan(cc(1, 0));
        assert!(plan.to_evict.contains(&crossing));
        assert!(plan.to_load.contains(&(crossing, ChunkDetail::Near)));
    }

    #[test]
    fn detail_hysteresis_holds_a_full_chunk_through_the_band() {
        // near 2, far 5. A chunk at the origin starts Near (camera at origin).
        let mut w = ChunkWindow::new(2, 5, 200);
        for (c, _) in w.plan(cc(0, 0)).to_load {
            w.mark_resident(c, 0);
        }
        // Camera steps to (3,0): origin chunk is now chebyshev distance 3 =
        // near_radius(2) + hysteresis(1), so it stays Near, no re-detail.
        let plan = w.plan(cc(3, 0));
        assert!(!plan.to_evict.contains(&cc(0, 0)));
        // Step once more to (4,0): distance 4 > 2 + 1, so it downgrades to Far.
        let plan = w.plan(cc(4, 0));
        assert!(plan.to_evict.contains(&cc(0, 0)));
        assert!(plan.to_load.contains(&(cc(0, 0), ChunkDetail::Far)));
    }

    #[test]
    fn equal_radii_disable_the_far_band() {
        // far == near -> every in-window chunk is Near, exactly as the original
        // single-detail window.
        let mut w = ChunkWindow::new(3, 3, 100);
        let plan = w.plan(cc(0, 0));
        assert!(plan.to_load.iter().all(|(_, d)| *d == ChunkDetail::Near));
        assert_eq!(w.counts_by_detail().1, 0); // never any far chunks
    }

    #[test]
    fn resident_bytes_counts_only_resident_chunks() {
        let mut w = ChunkWindow::new(1, 1, 100); // 3x3 window
        w.plan(cc(0, 0)); // all 9 dispatched Pending
        assert_eq!(w.resident_bytes(), 0); // nothing uploaded yet
        w.mark_resident(cc(0, 0), 500);
        w.mark_resident(cc(1, 0), 250);
        assert_eq!(w.resident_bytes(), 750);
        // The 7 still-pending chunks contribute nothing to the resident total.
        assert_eq!(w.counts(), (2, 7));
    }

    #[test]
    fn byte_budget_accessor_reflects_set_and_clear() {
        let mut w = ChunkWindow::new(1, 1, 4);
        assert_eq!(w.byte_budget(), None);
        w.set_byte_budget(Some(4096));
        assert_eq!(w.byte_budget(), Some(4096));
        w.set_byte_budget(None);
        assert_eq!(w.byte_budget(), None);
    }

    #[test]
    fn no_byte_budget_never_shrinks_the_window() {
        // Absurd resident bytes but no budget: the window stays the configured
        // radius and a stationary replan neither loads nor evicts -- byte-for-byte
        // the pure radius-only behavior the existing tests pin.
        let mut w = ChunkWindow::new(1, 3, 1000);
        fill(&mut w, cc(0, 0), 10_000_000);
        assert_eq!(w.counts().0, 49); // full 7x7 window resident
        for _ in 0..4 {
            let plan = w.plan(cc(0, 0));
            assert!(plan.to_load.is_empty());
            assert!(plan.to_evict.is_empty());
        }
        assert_eq!(w.counts().0, 49);
    }

    #[test]
    fn byte_budget_evicts_the_far_band_before_the_near_band() {
        let mut w = ChunkWindow::new(2, 6, 1000);
        fill(&mut w, cc(0, 0), 100);
        let (near_full, far_full) = w.counts_by_detail();
        assert_eq!(near_full, 25); // the 5x5 full-detail core
        assert!(far_full > 0);
        // A budget that cannot hold the whole impostor band but comfortably fits
        // the near core: the far band shrinks first, leaving the core intact.
        w.set_byte_budget(Some(9000));
        settle(&mut w, cc(0, 0), 100);
        let (near_after, far_after) = w.counts_by_detail();
        assert_eq!(near_after, 25, "full-detail core must survive");
        assert!(far_after < far_full, "impostor band must shrink");
        assert!(w.resident_bytes() <= 9000);
        assert!(w.is_tracked(cc(0, 0)), "camera chunk is never evicted");
        assert!(!w.is_tracked(cc(6, 0)), "outermost ring evicted");
    }

    #[test]
    fn tighter_byte_budget_shrinks_the_window_further() {
        // A tighter budget must keep strictly fewer chunks resident -- once the
        // far band is exhausted the clamp shrinks into the near band too.
        let build = |budget: u64| {
            let mut w = ChunkWindow::new(2, 6, 1000);
            fill(&mut w, cc(0, 0), 100);
            w.set_byte_budget(Some(budget));
            settle(&mut w, cc(0, 0), 100);
            w
        };
        let loose = build(9000);
        let tight = build(6000);
        assert!(loose.resident_bytes() <= 9000);
        assert!(tight.resident_bytes() <= 6000);
        assert!(
            tight.counts().0 < loose.counts().0,
            "tighter budget must shrink further: tight {} vs loose {}",
            tight.counts().0,
            loose.counts().0
        );
        assert!(loose.is_tracked(cc(0, 0)) && tight.is_tracked(cc(0, 0)));
    }

    #[test]
    fn byte_budget_clamp_settles_without_oscillating() {
        let mut w = ChunkWindow::new(1, 5, 1000);
        fill(&mut w, cc(0, 0), 100);
        w.set_byte_budget(Some(5000));
        settle(&mut w, cc(0, 0), 100); // panics if it never converges
        // Past the fixed point a stationary replan is a no-op: the effective
        // radius neither regrows nor re-shrinks frame to frame.
        for _ in 0..8 {
            let plan = w.plan(cc(0, 0));
            assert!(plan.to_load.is_empty(), "regrew: {:?}", plan.to_load);
            assert!(plan.to_evict.is_empty(), "evicted: {:?}", plan.to_evict);
        }
        assert!(w.resident_bytes() <= 5000);
    }
}

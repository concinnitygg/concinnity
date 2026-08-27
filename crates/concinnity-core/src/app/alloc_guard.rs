// src/app/alloc_guard.rs
//
// The headless loop's steady-state allocation invariant, checked in dev builds.
//
// A world allocates while it is built and started; a tick of a world that has
// settled must allocate nothing, or the cost recurs every frame for as long as
// the app runs. That is the property this asserts.
//
// The counters behind it are process-wide, so a tick's delta is an upper bound
// on what the loop itself allocated: another thread's allocation can only ADD
// to it. One tick that allocates nothing therefore proves the loop's own cost
// is nothing, and a window of ticks that all allocate proves a per-tick cost
// only where the loop is the one allocating thread. Two things keep the check
// on the sound side of that:
//
//   - it reads nothing until some binary installs the tracking allocator, so a
//     host that opts out of counting is never judged;
//   - none of the loop's own work happens between two ticks, so allocations
//     counted in that gap are another thread's, and a window holding one is
//     abandoned rather than reported.

use concinnity_memory::alloc_count;

// Ticks a run is given to reach its steady state before the invariant applies.
// The same warmup the engine's per-frame allocation pins use: every system has
// reserved its working memory and every recycled buffer has reached its size
// well inside it.
pub(crate) const WARMUP_TICKS: u64 = 64;

// Consecutive allocating ticks that make a per-tick cost. A settled loop lands
// a tick that allocates nothing long before this, whatever else the process is
// doing.
pub(crate) const QUIET_WINDOW_TICKS: u64 = 64;

pub(crate) struct AllocGuard {
    tick: u64,
    // Counter reading at the top of the current tick.
    started: Option<u64>,
    // Counter reading when the last tick ended, against which the gap before
    // the next one exposes another allocating thread.
    ended: Option<u64>,
    // Whether such a gap preceded the current tick.
    foreign: bool,
    // Consecutive judged ticks that allocated.
    since_quiet: u64,
    // The least any tick in the current window allocated, and which tick.
    quietest: (u64, u64),
}

impl AllocGuard {
    pub(crate) const fn new() -> Self {
        Self {
            tick: 0,
            started: None,
            ended: None,
            foreign: false,
            since_quiet: 0,
            quietest: (0, 0),
        }
    }

    pub(crate) fn begin_tick(&mut self) {
        let now = alloc_count();
        self.foreign = match (self.ended, now) {
            (Some(ended), Some(now)) => now > ended,
            _ => false,
        };
        self.started = now;
    }

    pub(crate) fn end_tick(&mut self) {
        let now = alloc_count();
        let sampled = match (self.started, now) {
            (Some(started), Some(now)) => Some(now.saturating_sub(started)),
            _ => None,
        };
        self.ended = now;
        match sampled {
            Some(allocated) => self.observe(allocated),
            None => self.tick = self.tick.saturating_add(1),
        }
    }

    // The invariant itself, split from the sampling so it can be driven with
    // known deltas.
    fn observe(&mut self, allocated: u64) {
        let tick = self.tick;
        self.tick = self.tick.saturating_add(1);
        if tick < WARMUP_TICKS {
            return;
        }
        if allocated == 0 || self.foreign {
            self.since_quiet = 0;
            return;
        }
        if self.since_quiet == 0 || allocated < self.quietest.1 {
            self.quietest = (tick, allocated);
        }
        self.since_quiet += 1;
        assert!(
            self.since_quiet < QUIET_WINDOW_TICKS,
            "the headless loop allocates every tick: none of the last \
             {QUIET_WINDOW_TICKS} allocated nothing, and the quietest, tick {}, \
             allocated {} time(s)",
            self.quietest.0,
            self.quietest.1
        );
    }
}

// Whether the counters the guard judges a tick by are live, which they are in
// a binary that installed the tracking allocator and nowhere else.
#[cfg(test)]
pub(crate) fn armed() -> bool {
    alloc_count().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    // `#[global_allocator]` is a per-program item, so a library declares one
    // only for its own test binary. This crate's tests read the counters the
    // guard is built on -- here, and in the driver's `run_tests` -- so the test
    // binary is the one build of this crate that installs the allocator.
    concinnity_memory::install_global_allocator!();

    // Drive the guard past its warmup without judging anything.
    fn warmed() -> AllocGuard {
        let mut guard = AllocGuard::new();
        for _ in 0..WARMUP_TICKS {
            guard.observe(1);
        }
        guard
    }

    // The warmup is where a world reserves what it reuses, so allocation there
    // is expected and unjudged: many times the window's worth of allocating
    // ticks passes without a word.
    #[test]
    fn the_warmup_is_not_judged() {
        let mut guard = AllocGuard::new();
        for _ in 0..WARMUP_TICKS {
            guard.observe(64);
        }
        assert_eq!(guard.tick, WARMUP_TICKS);
    }

    // A tick that allocates nothing proves the loop's own steady-state cost,
    // whatever the ticks around it did, so it clears the window.
    #[test]
    fn a_quiet_tick_clears_the_window() {
        let mut guard = warmed();
        for _ in 0..(QUIET_WINDOW_TICKS * 4) {
            for _ in 0..(QUIET_WINDOW_TICKS - 1) {
                guard.observe(3);
            }
            guard.observe(0);
        }
        assert_eq!(guard.since_quiet, 0);
    }

    // A whole window with no quiet tick is a loop that allocates every tick.
    // The panic names the tick that allocated least, which is the strongest
    // claim the process-wide counters support.
    #[test]
    #[should_panic(expected = "allocated 2 time(s)")]
    fn a_window_of_allocating_ticks_fails_naming_the_quietest() {
        let mut guard = warmed();
        for i in 0..QUIET_WINDOW_TICKS {
            guard.observe(if i == 3 { 2 } else { 9 });
        }
    }

    // Allocations counted between two ticks belong to another thread, and a
    // thread allocating alongside the loop is what makes a tick's delta
    // untrustworthy. Such a window is abandoned rather than reported.
    #[test]
    fn another_allocating_thread_abandons_the_window() {
        let mut guard = warmed();
        for _ in 0..(QUIET_WINDOW_TICKS * 4) {
            guard.foreign = true;
            guard.observe(7);
        }
        assert_eq!(guard.since_quiet, 0);
    }

    // A tick the counters could not be read around is not judged: a binary
    // that installs no tracking allocator runs the loop unguarded rather than
    // on zeros. Ending a tick that never began is that same unsampled case.
    #[test]
    fn an_unsampled_tick_is_not_judged() {
        let mut guard = warmed();
        guard.since_quiet = QUIET_WINDOW_TICKS - 1;
        guard.end_tick();
        assert_eq!(guard.tick, WARMUP_TICKS + 1, "the tick still counts");
        assert_eq!(guard.since_quiet, QUIET_WINDOW_TICKS - 1);
    }

    // The counters this binary judges its own ticks by are live, which is what
    // makes the driver's long-run test an assertion rather than a formality.
    #[test]
    fn the_test_binary_arms_the_guard() {
        assert!(armed());
    }
}

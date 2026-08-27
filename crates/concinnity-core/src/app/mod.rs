//! The headless driver: a world, the fixed virtual timestep it runs on, and the
//! loop that steps the two together.
//!
//! Time here is virtual. Every tick publishes the same [`SimTiming`] budget and
//! steps the world once, with no sleep and no wall clock, so a run is
//! reproducible and goes as fast as the host can step it. Pacing a frame
//! against a display, accumulating real elapsed time, and catching a signal are
//! a windowed host's concerns; a host that has them drives the world itself.
//!
//! [`SimTiming`]: crate::ecs::SimTiming

#[cfg(debug_assertions)]
mod alloc_guard;
mod fixed_timestep;

#[cfg(test)]
mod headless_world_tests;
#[cfg(test)]
mod run_tests;

use crate::ecs::{StepResult, SystemTable, World};
use crate::result::CnResult;
use fixed_timestep::FixedTimestep;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppStatus {
    Created,
    Started,
}

/// A world and the headless loop that runs it.
///
/// [`run`](App::run) drives the world until a system stops it or its last
/// system finishes; [`run_for`](App::run_for) is the bounded form, for a test
/// or a tool that wants a known number of ticks. Both start the world if the
/// caller has not.
///
/// In dev builds the loop holds itself to a steady state of no allocation per
/// tick, which is asserted rather than assumed. See [`run`](App::run).
pub struct App {
    world: World,
    // The systems the world is started with. A host that contributes none runs
    // its world's content and nothing over it.
    table: &'static SystemTable,
    status: AppStatus,
    sim: FixedTimestep,
    #[cfg(debug_assertions)]
    allocs: alloc_guard::AllocGuard,
}

impl core::fmt::Debug for App {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("App")
            .field("ticks", &self.sim.ticks())
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}

impl App {
    /// An app that runs `world` with no systems over it: what a caller gets
    /// when the host contributes no system table.
    pub fn from_world(world: World) -> Self {
        Self::with_systems(world, &SystemTable::EMPTY)
    }

    /// An app that runs `world` under `table`, which is what gives it systems.
    pub fn with_systems(world: World, table: &'static SystemTable) -> Self {
        Self {
            world,
            table,
            status: AppStatus::Created,
            sim: FixedTimestep::default(),
            #[cfg(debug_assertions)]
            allocs: alloc_guard::AllocGuard::new(),
        }
    }

    /// Borrow the world being run.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Ticks stepped so far.
    pub fn ticks(&self) -> u64 {
        self.sim.ticks()
    }

    /// Build the world's systems and run their `init`. Runs once: a second
    /// call is [`InvalidState`](CnResult::InvalidState) rather than a second
    /// `init` over the running world.
    pub fn start(&mut self) -> Result<(), CnResult> {
        if self.status != AppStatus::Created {
            return Err(CnResult::InvalidState);
        }
        self.world.start(self.table)?;
        self.status = AppStatus::Started;
        Ok(())
    }

    /// Step until a system stops the world or the last one finishes, returning
    /// which of the two ended the run.
    ///
    /// Under `debug_assertions` the loop asserts its own steady state: past a
    /// warmup, a settled world's tick allocates nothing, and a stretch of ticks
    /// that all allocate is a cost that would recur every frame for the life of
    /// the app, so the run panics naming the tick that allocated least. The
    /// counters behind that are process-wide, so the check stands down where it
    /// cannot trust them: where no binary installed the tracking allocator, and
    /// where another thread is allocating alongside the loop.
    pub fn run(&mut self) -> Result<StepResult, CnResult> {
        self.start_if_created()?;
        loop {
            let result = self.tick();
            if result != StepResult::Continue {
                return Ok(result);
            }
        }
    }

    /// Step at most `ticks` times, returning what the last tick reported:
    /// `Continue` when the full count ran, `Stop` or `Done` when the world
    /// ended the run early. The bounded form of [`run`](App::run).
    pub fn run_for(&mut self, ticks: u64) -> Result<StepResult, CnResult> {
        self.start_if_created()?;
        let mut result = StepResult::Continue;
        for _ in 0..ticks {
            result = self.tick();
            if result != StepResult::Continue {
                break;
            }
        }
        Ok(result)
    }

    // Start the world unless the caller already did, so a run is one call.
    fn start_if_created(&mut self) -> Result<(), CnResult> {
        if self.status == AppStatus::Created {
            self.start()?;
        }
        Ok(())
    }

    // One tick: the frame's fixed timing budget, then the world's step. The
    // allocation invariant brackets both, since publishing the budget is as
    // much a per-tick cost as stepping is.
    fn tick(&mut self) -> StepResult {
        #[cfg(debug_assertions)]
        self.allocs.begin_tick();
        self.world.insert_resource(self.sim.advance());
        let result = self.world.step();
        #[cfg(debug_assertions)]
        self.allocs.end_tick();
        result
    }
}

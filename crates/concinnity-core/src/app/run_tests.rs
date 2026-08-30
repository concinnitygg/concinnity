// src/app/run_tests.rs
//
// The headless driver over a small world: what one tick publishes, the bounded
// and unbounded runs, a system halting the world, and the steady-state
// allocation invariant holding across a long run.
//
// The systems here move a `Transform` rather than pushing components, so a
// settled tick allocates nothing and the invariant has something honest to
// judge. The tracking allocator this crate's test binary installs (see
// `alloc_guard`) is what arms it.

use alloc::boxed::Box;

use crate::app::App;
use crate::components::Transform;
use crate::ecs::{PipelineContext, SimTiming, StepResult, System, SystemEntry, SystemTable, World};
use crate::result::CnResult;

// The step count the halting world stops at.
const STOP_AT: f32 = 3.0;

// Advances every transform one unit per tick, so the world's own state counts
// the ticks that have run.
#[derive(Debug)]
struct Ticker;

impl System for Ticker {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        for transform in ctx.query_mut::<Transform>() {
            transform.position[0] += 1.0;
        }
        StepResult::Continue
    }
}

// Halts the world once the ticker has counted far enough.
#[derive(Debug)]
struct Stopper;

impl System for Stopper {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        match ctx.query::<Transform>().next() {
            Some(t) if t.position[0] >= STOP_AT => StepResult::Stop,
            _ => StepResult::Continue,
        }
    }
}

fn ticker(world: &World) -> Option<Box<dyn System>> {
    world.query::<Transform>().next()?;
    Some(Box::new(Ticker))
}

fn stopper(world: &World) -> Option<Box<dyn System>> {
    world.query::<Transform>().next()?;
    Some(Box::new(Stopper))
}

const fn entry(name: &'static str, gate: fn(&World) -> Option<Box<dyn System>>) -> SystemEntry {
    SystemEntry {
        name,
        present_when: "the world holds a Transform",
        gate,
        after: &[],
        before: &[],
    }
}

const fn table(entries: &'static [SystemEntry]) -> SystemTable {
    SystemTable {
        entries,
        complete_world: None,
        before_init: None,
        prepare_events: None,
    }
}

static TICKING: SystemTable = table(&[entry("Ticker", ticker)]);
static HALTING: SystemTable = table(&[entry("Ticker", ticker), entry("Stopper", stopper)]);

// A world holding one transform at the origin, which is both what gates the
// systems in and what counts their steps.
fn counting_world() -> World {
    let mut world = World::new();
    world.push(Transform::default());
    world
}

fn count(app: &App) -> f32 {
    app.world()
        .query::<Transform>()
        .next()
        .expect("the world holds its transform")
        .position[0]
}

// The bounded run steps exactly what it was asked for, and the world's own
// state agrees with the app's tick count.
#[test]
fn a_bounded_run_steps_the_ticks_it_was_asked_for() {
    let mut app = App::with_systems(counting_world(), &TICKING);
    assert_eq!(app.run_for(5), Ok(StepResult::Continue));
    assert_eq!(app.ticks(), 5);
    assert_eq!(count(&app), 5.0);
}

// A second bounded run continues where the first left off rather than
// restarting the world.
#[test]
fn bounded_runs_continue_one_another() {
    let mut app = App::with_systems(counting_world(), &TICKING);
    app.run_for(2).expect("the app runs");
    app.run_for(3).expect("the app runs again");
    assert_eq!(app.ticks(), 5);
    assert_eq!(count(&app), 5.0);
}

// Each tick publishes the fixed virtual budget the simulation systems read:
// one step, at the fixed rate, with nothing left over to blend against.
#[test]
fn a_tick_publishes_the_fixed_virtual_timing() {
    let mut app = App::with_systems(counting_world(), &TICKING);
    app.run_for(1).expect("the app runs");

    let timing = app
        .world()
        .resource::<SimTiming>()
        .expect("the tick published its budget");
    assert_eq!(timing.ticks, 1);
    assert_eq!(timing.tick_dt, SimTiming::TICK_DT);
    assert_eq!(timing.alpha, 1.0);
}

// A system halting the world ends an unbounded run at the tick it halted on,
// and the run reports why it ended.
#[test]
fn a_system_stopping_the_world_ends_the_run() {
    let mut app = App::with_systems(counting_world(), &HALTING);
    assert_eq!(app.run(), Ok(StepResult::Stop));
    assert_eq!(app.ticks(), STOP_AT as u64);
}

// Stop is honored inside a bounded run too: the remaining ticks are not run.
#[test]
fn a_bounded_run_honors_a_stop_before_its_count() {
    let mut app = App::with_systems(counting_world(), &HALTING);
    assert_eq!(app.run_for(100), Ok(StepResult::Stop));
    assert_eq!(app.ticks(), STOP_AT as u64);
}

// A world with no systems is done as soon as it is stepped, which is what an
// app built from a world alone runs: its content, and nothing over it.
#[test]
fn a_world_with_no_systems_is_done_on_its_first_tick() {
    let mut app = App::from_world(counting_world());
    assert_eq!(app.run(), Ok(StepResult::Done));
    assert_eq!(app.ticks(), 1);
    assert_eq!(count(&app), 0.0, "no system ran over the world");
}

// A run starts the world for the caller, and starting it first is equally
// fine -- the run picks up the already-started world rather than refusing it.
#[test]
fn a_run_starts_the_world_and_tolerates_one_already_started() {
    let mut app = App::with_systems(counting_world(), &TICKING);
    assert_eq!(app.start(), Ok(()));
    assert_eq!(app.run_for(2), Ok(StepResult::Continue));
    assert_eq!(app.ticks(), 2);

    let mut unstarted = App::with_systems(counting_world(), &TICKING);
    assert_eq!(unstarted.run_for(2), Ok(StepResult::Continue));
    assert_eq!(count(&unstarted), 2.0);
}

// Starting twice is refused rather than running every system's `init` a second
// time over the running world.
#[test]
fn starting_twice_is_refused() {
    let mut app = App::with_systems(counting_world(), &TICKING);
    assert_eq!(app.start(), Ok(()));
    assert_eq!(app.start(), Err(CnResult::InvalidState));
}

// The long run the invariant exists for: past the warmup and a full window
// beyond it, a settled world's ticks allocate nothing, so the assertion inside
// `run_for` holds. It is reading live counters, which is what makes that
// something rather than nothing.
#[cfg(debug_assertions)]
#[test]
fn the_allocation_invariant_arms_and_holds_across_a_long_run() {
    use crate::app::alloc_guard::{QUIET_WINDOW_TICKS, WARMUP_TICKS, armed};

    let mut app = App::with_systems(counting_world(), &TICKING);
    let ticks = WARMUP_TICKS + QUIET_WINDOW_TICKS + 8;
    assert_eq!(app.run_for(ticks), Ok(StepResult::Continue));
    assert_eq!(app.ticks(), ticks);
    assert!(armed(), "the test binary installs the tracking allocator");
}

// The Debug impl reports where the run has got to rather than dumping the
// world through it.
#[test]
fn the_debug_impl_reports_the_runs_progress() {
    let mut app = App::with_systems(counting_world(), &TICKING);
    app.run_for(2).expect("the app runs");
    let text = alloc::format!("{app:?}");
    assert!(text.contains("ticks: 2"), "{text}");
    assert!(text.contains("Started"), "{text}");
}

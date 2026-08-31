// The headless driver reached through the trait: what a host holds when the
// loop it runs is a runtime value rather than a compile-time type.

use alloc::boxed::Box;

use crate::app::{App, Driver};
use crate::components::Transform;
use crate::ecs::{PipelineContext, StepResult, System, SystemEntry, SystemTable, World};
use crate::result::CnResult;

// The step count the halting world stops at.
const STOP_AT: f32 = 3.0;

// Advances every transform one unit per tick, then halts the world once it has
// counted far enough, so an unbounded run through the trait ends.
#[derive(Debug)]
struct Counter;

impl System for Counter {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        for transform in ctx.query_mut::<Transform>() {
            transform.position[0] += 1.0;
            if transform.position[0] >= STOP_AT {
                return StepResult::Stop;
            }
        }
        StepResult::Continue
    }
}

fn counter(world: &World) -> Option<Box<dyn System>> {
    world.query::<Transform>().next()?;
    Some(Box::new(Counter))
}

// A completion pass that refuses the world, which is what a start failure looks
// like from outside the loop.
fn refuse(_: &mut PipelineContext) -> Result<(), CnResult> {
    Err(CnResult::InvalidState)
}

const ENTRIES: &[SystemEntry] = &[SystemEntry {
    name: "Counter",
    present_when: "the world holds a Transform",
    gate: counter,
    after: &[],
    before: &[],
}];

static COUNTING: SystemTable = SystemTable {
    entries: ENTRIES,
    complete_world: None,
    before_init: None,
    prepare_events: None,
};

static REFUSING: SystemTable = SystemTable {
    entries: &[],
    complete_world: Some(refuse),
    before_init: None,
    prepare_events: None,
};

// A world holding one transform at the origin, which is both what gates the
// system in and what counts its steps.
fn counting_world() -> World {
    let mut world = World::new();
    world.push(Transform::default());
    world
}

fn driver(table: &'static SystemTable) -> Box<dyn Driver> {
    Box::new(App::with_systems(counting_world(), table))
}

// Starting through the trait builds the world's systems, and the second call is
// refused the same way the inherent one is.
#[test]
fn a_driver_starts_the_world_it_holds() {
    let mut driver = driver(&COUNTING);
    assert_eq!(driver.start(), Ok(()));
    assert_eq!(driver.start(), Err(CnResult::InvalidState));
}

// An unbounded run through the trait reports that it ran, not which of the
// readings ended it.
#[test]
fn a_driver_runs_the_world_to_its_end() {
    assert_eq!(driver(&COUNTING).run(), Ok(()));
}

// A run starts the world for the caller, so a world that cannot be started
// fails the run rather than stepping a half-built world.
#[test]
fn a_run_reports_a_refused_start() {
    assert_eq!(driver(&REFUSING).run(), Err(CnResult::InvalidState));
}

// The other way out: the world comes back as it was handed over, so a caller
// can put it on a different loop.
#[test]
fn a_driver_hands_its_world_back_unrun() {
    let world = driver(&COUNTING).into_world();
    let transform = world
        .query::<Transform>()
        .next()
        .expect("the world keeps its content");
    assert_eq!(transform.position[0], 0.0, "no tick ran over the world");
}

// src/ecs/world_run_tests.rs
//
// Starting and stepping a world from a table, with no host beyond this crate:
// the gates build the systems, the load-time pass runs before their init, the
// clock seam times each step, and a finished system leaves the set.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::components::TextLabel;
use crate::ecs::{
    Access, Clock, EventStore, PipelineContext, StepResult, System, SystemEntry, SystemTable, World,
};

// Steps until it has run `stop_after` times, then reports Done. Each phase
// leaves a labelled marker in the world, so a test reads the order the world
// ran them in out of the component column.
#[derive(Debug)]
struct Marker {
    steps: u32,
    stop_after: Option<u32>,
}

impl System for Marker {
    fn init(&mut self, ctx: &mut PipelineContext) {
        mark(ctx, "init");
    }

    fn step(&mut self, _ctx: &mut PipelineContext) -> StepResult {
        self.steps += 1;
        match self.stop_after {
            Some(n) if self.steps >= n => StepResult::Done,
            _ => StepResult::Continue,
        }
    }
}

fn mark(ctx: &mut PipelineContext, what: &str) {
    ctx.push(TextLabel {
        content: what.to_string(),
        ..Default::default()
    });
}

// Present whenever the world holds the seed marker the test adds.
fn gate(world: &World) -> Option<Box<dyn System>> {
    world.query::<TextLabel>().next()?;
    Some(Box::new(Marker {
        steps: 0,
        stop_after: None,
    }))
}

// The same gate for a system that finishes after one step.
fn finishing_gate(world: &World) -> Option<Box<dyn System>> {
    world.query::<TextLabel>().next()?;
    Some(Box::new(Marker {
        steps: 0,
        stop_after: Some(1),
    }))
}

fn before_init(ctx: &mut PipelineContext) {
    mark(ctx, "before_init");
}

fn prepare_events(store: &mut EventStore, _access: Access) {
    store.get_mut_or_create::<u8>();
}

const fn entry(name: &'static str, gate: fn(&World) -> Option<Box<dyn System>>) -> SystemEntry {
    SystemEntry {
        name,
        present_when: "the world holds a TextLabel",
        gate,
        after: &[],
        before: &[],
    }
}

const TABLE: SystemTable = SystemTable {
    entries: &[entry("Marker", gate)],
    complete_world: None,
    before_init: Some(before_init),
    prepare_events: Some(prepare_events),
};

const FINISHING: SystemTable = SystemTable {
    entries: &[entry("Marker", finishing_gate)],
    complete_world: None,
    before_init: None,
    prepare_events: None,
};

// A world holding the seed marker, so the table's gate holds.
fn seeded() -> World {
    let mut world = World::new();
    world.add_component(TextLabel {
        content: "seed".to_string(),
        ..Default::default()
    });
    world
}

fn marks(world: &World) -> Vec<&str> {
    world
        .query::<TextLabel>()
        .map(|l| l.content.as_str())
        .collect()
}

// The whole start sequence, in order: gates build from the world's content, the
// table's load-time pass runs, then each system inits.
#[test]
fn start_runs_the_load_pass_before_system_init() {
    let mut world = seeded();
    world.start(&TABLE).expect("the world starts");

    assert_eq!(marks(&world), ["seed", "before_init", "init"]);
    assert_eq!(world.system_count(), 1);
    assert_eq!(world.systems()[0].name(), "Marker");
}

// The manifest is the same gated set `start` builds, from the same table.
#[test]
fn the_manifest_matches_what_start_builds() {
    let mut world = seeded();
    let manifest = world.system_manifest(&TABLE);
    world.start(&TABLE).expect("the world starts");
    let built: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
    assert_eq!(manifest, built);
}

// A gate that does not hold leaves the world with no systems, and a world with
// no systems is Done on its first step.
#[test]
fn an_ungated_world_builds_nothing_and_is_done() {
    let mut world = World::new();
    world.start(&TABLE).expect("the world starts");
    assert_eq!(world.system_count(), 0);
    assert_eq!(world.step(), StepResult::Done);
}

// Starting twice does not build the systems twice: the gating content can
// survive `init`, and the guard is what stops a second start duplicating it.
#[test]
fn starting_twice_builds_the_systems_once() {
    let mut world = seeded();
    world.start(&TABLE).expect("the world starts");
    world.start(&TABLE).expect("the world starts again");
    assert_eq!(world.system_count(), 1);
}

// A system reporting Done leaves the set, and the world reports Done once the
// last one has.
#[test]
fn a_finished_system_leaves_the_set() {
    let mut world = seeded();
    world.start(&FINISHING).expect("the world starts");
    assert_eq!(world.system_count(), 1);
    assert_eq!(world.step(), StepResult::Done);
    assert_eq!(world.system_count(), 0);
}

// A table with no systems and no passes is a world that starts and steps
// without a host contributing anything.
#[test]
fn an_empty_table_starts_a_world_that_does_nothing() {
    let mut world = seeded();
    world.start(&SystemTable::EMPTY).expect("the world starts");
    assert_eq!(marks(&world), ["seed"], "no load pass ran");
    assert_eq!(world.step(), StepResult::Done);
}

// The table's event-queue pass runs for each scheduled system, so a queue a
// declared access can touch exists before the first step.
#[test]
fn start_pre_creates_the_declared_event_queues() {
    let mut world = seeded();
    assert!(world.events::<u8>().is_none());
    world.start(&TABLE).expect("the world starts");
    assert!(world.events::<u8>().is_some());
}

// Every event queue rotates each step, whatever its type: an event sent once
// retires after two steps, so a queue written every frame stays bounded at two
// frames of events instead of growing for the session.
#[test]
fn event_queues_rotate_every_step() {
    let mut world = seeded();
    world.start(&TABLE).expect("the world starts");
    for _ in 0..5 {
        world.events_mut::<u16>().send(1);
        world.events_mut::<u32>().send(2);
        world.step();
    }
    for len in [
        world.events::<u16>().expect("queue exists").len(),
        world.events::<u32>().expect("queue exists").len(),
    ] {
        assert!(len <= 2, "queue holds at most two frames of events: {len}");
    }
}

// Micros counted off the installed clock: this one ticks 10 per read, so the
// system's step reads as 10 micros in the completed frame's timings.
#[test]
fn the_clock_resource_times_each_system() {
    static NOW: AtomicU64 = AtomicU64::new(0);
    fn ticking() -> u64 {
        NOW.fetch_add(10, Ordering::Relaxed)
    }

    let mut world = seeded();
    world.insert_resource(Clock(ticking));
    world.start(&TABLE).expect("the world starts");
    // Two steps: the first frame's timings become readable when the second
    // rotates the profile's buffers.
    world.step();
    world.step();

    assert_eq!(world.profile().system_timings(), [("Marker", 10)]);
}

// Without a clock the same step is recorded at zero rather than dropping the
// system from the profile.
#[test]
fn an_absent_clock_records_zero_micros() {
    let mut world = seeded();
    world.start(&TABLE).expect("the world starts");
    world.step();
    world.step();

    assert_eq!(world.profile().system_timings(), [("Marker", 0)]);
}

// The profile is exposed before any step has run, and holds no timings.
#[test]
fn the_profile_is_exposed_before_any_step() {
    let world = World::new();
    assert!(world.profile().system_timings().is_empty());
}

// The Debug impl reports component and system counts rather than dumping their
// contents.
#[test]
fn the_debug_impl_reports_counts() {
    let mut world = seeded();
    world.start(&TABLE).expect("the world starts");
    let text: String = alloc::format!("{world:?}");
    assert!(text.contains("components: 3"), "{text}");
    assert!(text.contains("systems: 1"), "{text}");
}

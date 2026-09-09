// src/ecs/registration_tests.rs
//
// Systems registered on a world from outside its table: where in the tick they
// land, and what the merge refuses.
//
// The table under test is synthetic -- one entry per phase, each gated on the
// world holding a TextLabel -- so the placement rule is read off the run order
// directly rather than through the engine's twenty-entry schedule.

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec::Vec;

use crate::components::TextLabel;
use crate::ecs::{Phase, PipelineContext, StepResult, System, SystemEntry, SystemTable, World};

// A system that does nothing; the tests read placement off the built set's
// names, not off anything it writes.
#[derive(Debug)]
struct Inert;

impl System for Inert {
    fn step(&mut self, _ctx: &mut PipelineContext) -> StepResult {
        StepResult::Continue
    }
}

fn gate(world: &World) -> Option<Box<dyn System>> {
    world.query::<TextLabel>().next()?;
    Some(Box::new(Inert))
}

const fn entry(name: &'static str, phase: Phase) -> SystemEntry {
    SystemEntry {
        name,
        present_when: "the world holds a TextLabel",
        phase,
        gate,
        after: &[],
        before: &[],
    }
}

// One entry per phase, in phase order, like a real table.
const TABLE: SystemTable = SystemTable {
    entries: &[
        entry("TableEarly", Phase::Early),
        entry("TableLogic", Phase::Logic),
        entry("TablePreRender", Phase::PreRender),
        entry("TableLate", Phase::Late),
    ],
    complete_world: None,
    before_init: None,
    prepare_events: None,
};

fn seeded() -> World {
    let mut world = World::new();
    world.add_component(TextLabel {
        content: "seed".to_string(),
        ..Default::default()
    });
    world
}

fn names(world: &World) -> Vec<&'static str> {
    world.systems().iter().map(|s| s.name()).collect()
}

// The placement rule, whole: phases run in order, and within a phase the
// table's entries run before the systems registered into it.
#[test]
fn a_registration_runs_after_the_table_entries_of_its_phase() {
    let mut world = seeded();
    world.add_system(Phase::Logic, "MyLogic", Inert);
    world.add_system(Phase::Early, "MyEarly", Inert);
    world.start(&TABLE).unwrap();

    assert_eq!(
        names(&world),
        [
            "TableEarly",
            "MyEarly",
            "TableLogic",
            "MyLogic",
            "TablePreRender",
            "TableLate",
        ]
    );
}

// Registration order is run order within one phase.
#[test]
fn registrations_in_one_phase_keep_their_order() {
    let mut world = seeded();
    world.add_system(Phase::Late, "First", Inert);
    world.add_system(Phase::Late, "Second", Inert);
    world.add_system(Phase::Late, "Third", Inert);
    world.start(&TABLE).unwrap();

    let built = names(&world);
    let tail = &built[built.len() - 3..];
    assert_eq!(tail, ["First", "Second", "Third"]);
}

// The default phase is the end of the tick, after every table entry.
#[test]
fn the_default_phase_runs_last() {
    let mut world = seeded();
    world.add_system(Phase::default(), "Mine", Inert);
    world.start(&TABLE).unwrap();

    assert_eq!(names(&world).last(), Some(&"Mine"));
}

// A registration into a phase no table entry occupies still lands in it, in
// phase order among the rest.
#[test]
fn a_registration_lands_in_an_empty_phase() {
    const SPARSE: SystemTable = SystemTable {
        entries: &[entry("TableLate", Phase::Late)],
        complete_world: None,
        before_init: None,
        prepare_events: None,
    };

    let mut world = seeded();
    world.add_system(Phase::Early, "MyEarly", Inert);
    world.start(&SPARSE).unwrap();

    assert_eq!(names(&world), ["MyEarly", "TableLate"]);
}

// A registered system steps like any other, from the first tick.
#[test]
fn a_registered_system_steps() {
    #[derive(Debug, Default)]
    struct Counting(u32);

    impl System for Counting {
        fn step(&mut self, _ctx: &mut PipelineContext) -> StepResult {
            self.0 += 1;
            StepResult::Continue
        }
    }

    let mut world = seeded();
    world.add_system(Phase::Late, "Counting", Counting::default());
    world.start(&TABLE).unwrap();
    world.step();
    world.step();

    let counted = world
        .systems()
        .iter()
        .find_map(|s| s.downcast_ref::<Counting>())
        .expect("the registered system is in the built set");
    assert_eq!(counted.0, 2);
}

// The manifest reports what `start` builds, registrations included and in the
// same order, so the schedule a tool prints is the schedule that runs.
#[test]
fn the_manifest_matches_the_built_set() {
    let mut world = seeded();
    world.add_system(Phase::Early, "MyEarly", Inert);
    world.add_system(Phase::Late, "MyLate", Inert);

    let manifest = world.system_manifest(&TABLE);
    world.start(&TABLE).unwrap();
    assert_eq!(manifest, names(&world));
}

// Registrations are readable before the world starts, in run order.
#[test]
fn registered_systems_are_listed_in_run_order() {
    let mut world = seeded();
    world.add_system(Phase::Late, "MyLate", Inert);
    world.add_system(Phase::Early, "MyEarly", Inert);
    assert_eq!(world.registered_systems(), ["MyEarly", "MyLate"]);
}

// A name a table entry already uses would inherit that entry's ordering edges
// and shadow it in every name lookup, so the merge refuses it.
#[test]
#[should_panic(expected = "already registered as 'TableLate'")]
fn a_name_a_table_entry_uses_is_refused() {
    let mut world = seeded();
    world.add_system(Phase::Late, "TableLate", Inert);
    let _ = world.start(&TABLE);
}

// Two registrations under one name are equally unaddressable.
#[test]
#[should_panic(expected = "already registered as 'Mine'")]
fn a_repeated_registration_name_is_refused() {
    let mut world = seeded();
    world.add_system(Phase::Late, "Mine", Inert);
    world.add_system(Phase::Early, "Mine", Inert);
    let _ = world.start(&TABLE);
}

// Registrations are read once, by `start`: a world already running does not
// grow a system, and the one it was started with is not built twice.
#[test]
fn registering_after_start_changes_nothing() {
    let mut world = seeded();
    world.add_system(Phase::Late, "Mine", Inert);
    world.start(&TABLE).unwrap();
    let before = names(&world);

    world.add_system(Phase::Late, "TooLate", Inert);
    world.start(&TABLE).unwrap();
    assert_eq!(names(&world), before);
}

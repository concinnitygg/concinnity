// Systems registered from outside the engine, run on the headless app.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::app::App;
use crate::components::{TextLabel, Transform};
use crate::ecs::{Phase, PipelineContext, StepResult, System, World};

// Counts its steps into shared state, then finishes. A world whose only
// system is this one ends the run when it reports Done, so the count the
// caller reads back is the whole run.
#[derive(Debug)]
struct Counting {
    steps: Arc<AtomicU32>,
    stop_after: u32,
}

impl System for Counting {
    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        let seen = self.steps.fetch_add(1, Ordering::Relaxed) + 1;
        for label in ctx.query_mut::<TextLabel>() {
            label.content = alloc::format!("step {seen}");
        }
        if seen >= self.stop_after {
            StepResult::Done
        } else {
            StepResult::Continue
        }
    }
}

// Code registered on a world runs on the same loop the engine's own systems
// run on, over the same world, and ends the run when it is finished.
#[test]
fn a_registered_system_runs_on_the_headless_loop() {
    let steps = Arc::new(AtomicU32::new(0));

    let mut world = World::new();
    world.add_component(TextLabel::default());
    world.add_system(
        Phase::Late,
        "Counting",
        Counting {
            steps: Arc::clone(&steps),
            stop_after: 3,
        },
    );

    App::from_world(world).run().expect("the run ends");
    assert_eq!(steps.load(Ordering::Relaxed), 3);
}

// The same registration reaches an app rather than a world.
#[test]
fn an_app_takes_a_registration_too() {
    let steps = Arc::new(AtomicU32::new(0));

    let app = App::from_world(World::new()).with_system(
        Phase::Early,
        "Counting",
        Counting {
            steps: Arc::clone(&steps),
            stop_after: 2,
        },
    );

    app.run().expect("the run ends");
    assert_eq!(steps.load(Ordering::Relaxed), 2);
}

#[derive(Debug)]
struct Countdown(u32);

crate::declare_components!(Countdown);

// Ticks every Countdown down, moving its Transform as it goes, and finishes
// once none is left above zero. `moved` is what the run is read back through,
// since `run` consumes the world.
#[derive(Debug)]
struct Tick {
    moved: Arc<AtomicU32>,
}

impl System for Tick {
    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        let live: Vec<_> = ctx
            .join2::<Countdown, Transform>()
            .filter(|(_, c, _)| c.0 > 0)
            .map(|(entity, _, _)| entity)
            .collect();
        if live.is_empty() {
            return StepResult::Done;
        }
        for entity in live {
            if let Some(c) = ctx.get_mut::<Countdown>(entity) {
                c.0 -= 1;
            }
            if let Some(t) = ctx.get_mut::<Transform>(entity) {
                t.position[0] += 1.0;
            }
            self.moved.fetch_add(1, Ordering::Relaxed);
        }
        StepResult::Continue
    }
}

// An application's own component type, declared, seeded beside a vocabulary
// component, and read and written by a registered system.
#[test]
fn a_declared_component_drives_a_registered_system() {
    let moved = Arc::new(AtomicU32::new(0));
    let mut world = World::new();
    let mover = world.spawn();
    world.insert(mover, Countdown(3));
    world.insert(mover, Transform::default());

    let app = App::from_world(world).with_system(
        Phase::Late,
        "Tick",
        Tick {
            moved: Arc::clone(&moved),
        },
    );
    app.run().expect("the run ends");
    assert_eq!(
        moved.load(Ordering::Relaxed),
        3,
        "the join reached the entity on each of the three ticks its countdown ran"
    );
}

// A world of authored content, run headless: no window, no host, no clock.
//
// The systems in `run_tests` are synthetic, written to exercise the driver.
// These run the real thing -- a `Behavior` compiled and evaluated by the
// system core's own table gates in -- so what they prove is that the loop, the
// table, and a simulation system fit together into a world that does something.

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::app::App;
use crate::components::{
    Behavior, BehaviorExpr, BehaviorNode, BehaviorSource, BodyDynamics, Collider, PhysicsConfig,
    PropCollider, PropInstance, StoryCommand, StoryPlayback, Transform,
};
use crate::ecs::{Entity, HEADLESS_SYSTEMS, StepResult, World};

// Each tick this behavior drifts every prop one unit along x and asks the story
// to advance, so the world's own state counts the ticks two ways: a component
// it wrote, and an event it sent.
fn drifting_behavior() -> Behavior {
    Behavior {
        on: BehaviorSource::Tick,
        scope: vec!["Prop".to_string()],
        body: vec![
            BehaviorNode::SetTransform {
                entity: BehaviorExpr::SelfEntity,
                position: Some(BehaviorExpr::Add(
                    Box::new(BehaviorExpr::Position(Box::new(BehaviorExpr::SelfEntity))),
                    Box::new(BehaviorExpr::Vec3([1.0, 0.0, 0.0])),
                )),
                rotation_deg: None,
                scale: None,
            },
            BehaviorNode::Story(StoryPlayback::Continue),
        ],
        ..Default::default()
    }
}

// One prop at the origin, and the behavior that moves it. The prop carries the
// runtime marker a scope of "Prop" resolves against, since nothing decomposes
// authored content here.
fn drifting_world() -> (World, Entity) {
    let mut world = World::new();
    world.add_component(drifting_behavior());
    let prop = world.push(PropInstance);
    world.insert(prop, Transform::default());
    (world, prop)
}

fn drift(app: &App, prop: Entity) -> f32 {
    app.world()
        .get::<Transform>(prop)
        .expect("the prop keeps its transform")
        .position[0]
}

// The headless table gates the behavior system in and nothing else, which is
// the whole of what this world runs.
#[test]
fn a_behavior_world_starts_with_just_the_behavior_system() {
    let (world, _) = drifting_world();
    let mut app = App::with_systems(world, HEADLESS_SYSTEMS);
    app.start().expect("the world starts");

    let names: Vec<&str> = app.world().systems().iter().map(|s| s.name()).collect();
    assert_eq!(names, ["BehaviorSystem"]);
}

// The behavior fires once per tick of virtual time and its effects land on the
// world: the transform it wrote, and the story command it sent.
#[test]
fn a_behavior_fires_every_tick_of_a_headless_run() {
    let (world, prop) = drifting_world();
    let mut app = App::with_systems(world, HEADLESS_SYSTEMS);
    assert_eq!(app.run_for(5), Ok(StepResult::Continue));

    assert_eq!(drift(&app, prop), 5.0, "one unit per tick, five ticks");
    assert!(
        app.world()
            .events::<StoryCommand>()
            .is_some_and(|q| !q.is_empty()),
        "the tick's story command reached the event queue",
    );
}

// A second run continues the first rather than restarting the world, so the
// behavior's clocks and instances survive between them.
#[test]
fn bounded_runs_over_a_behavior_world_continue_one_another() {
    let (world, prop) = drifting_world();
    let mut app = App::with_systems(world, HEADLESS_SYSTEMS);
    app.run_for(2).expect("the app runs");
    app.run_for(3).expect("the app runs again");
    assert_eq!(app.ticks(), 5);
    assert_eq!(drift(&app, prop), 5.0);
}

// A `start` source fires once however long the run is, which is the firing rule
// the virtual clock is what drives.
#[test]
fn a_start_sourced_behavior_fires_once_across_a_long_run() {
    let mut world = World::new();
    world.add_component(Behavior {
        on: BehaviorSource::Start,
        scope: vec!["Prop".to_string()],
        body: vec![BehaviorNode::SetTransform {
            entity: BehaviorExpr::SelfEntity,
            position: Some(BehaviorExpr::Vec3([7.0, 0.0, 0.0])),
            rotation_deg: None,
            scale: None,
        }],
        ..Default::default()
    });
    let prop = world.push(PropInstance);
    world.insert(prop, Transform::default());

    let mut app = App::with_systems(world, HEADLESS_SYSTEMS);
    app.run_for(20).expect("the app runs");
    assert_eq!(drift(&app, prop), 7.0);
}

// The long run the driver's allocation invariant exists for, over a real
// simulation system: past the warmup, a settled behavior tick allocates
// nothing, so the assertion inside `run_for` holds rather than firing.
#[cfg(debug_assertions)]
#[test]
fn a_behavior_world_settles_into_an_allocation_free_tick() {
    use crate::app::alloc_guard::{QUIET_WINDOW_TICKS, WARMUP_TICKS, armed};

    let (world, prop) = drifting_world();
    let mut app = App::with_systems(world, HEADLESS_SYSTEMS);
    let ticks = WARMUP_TICKS + QUIET_WINDOW_TICKS + 8;
    assert_eq!(app.run_for(ticks), Ok(StepResult::Continue));
    assert_eq!(drift(&app, prop), ticks as f32);
    assert!(armed(), "the test binary installs the tracking allocator");
}

// The world the headless CI run covers: a behavior that fires every tick
// beside real physics content -- four dynamic balls dropped just above the
// flat floor -- so a run exercises both simulation systems the headless table
// gates in.
fn simulating_world() -> (World, Entity, Vec<Entity>) {
    let mut world = World::new();
    world.add_component(drifting_behavior());
    let drifter = world.push(PropInstance);
    world.insert(drifter, Transform::default());

    world.add_component(PhysicsConfig::default());
    let balls: Vec<Entity> = (0..4)
        .map(|i| {
            let ball = world.push(Transform {
                position: [i as f32 * 3.0, 1.5, 0.0],
                rotation_deg: [0.0; 3],
                scale: [1.0; 3],
            });
            world.insert(
                ball,
                Collider(PropCollider {
                    shape: "ball".to_string(),
                    radius: 0.5,
                    ..Default::default()
                }),
            );
            world.insert(ball, BodyDynamics::default());
            ball
        })
        .collect();
    (world, drifter, balls)
}

// Only the settling test reads a body's height, and that test is itself
// debug-only.
#[cfg(debug_assertions)]
fn height(app: &App, entity: Entity) -> f32 {
    app.world()
        .get::<Transform>(entity)
        .expect("the body keeps its transform")
        .position[1]
}

// The headless CI run, and the whole claim the headless tier makes: a world of
// authored content with no host behind it runs its simulation, and once it has
// settled a tick of it costs no allocation.
//
// Three things are asserted, and the third is the one `run_for` makes on its
// own: the bodies fell and came to rest on the floor, the behavior fired once
// per tick throughout, and the driver's steady-state allocation invariant held
// across a full window past the warmup (see `alloc_guard`). The counters that
// invariant reads are live in this binary, which is what makes it an assertion
// rather than a formality.
#[cfg(debug_assertions)]
#[test]
fn a_simulating_world_settles_and_runs_an_allocation_free_tick() {
    use crate::app::alloc_guard::{QUIET_WINDOW_TICKS, WARMUP_TICKS, armed};

    let (world, drifter, balls) = simulating_world();
    let mut app = App::with_systems(world, HEADLESS_SYSTEMS);
    let ticks = WARMUP_TICKS + QUIET_WINDOW_TICKS + 8;
    assert_eq!(app.run_for(ticks), Ok(StepResult::Continue));

    assert!(armed(), "the test binary installs the tracking allocator");
    assert_eq!(
        drift(&app, drifter),
        ticks as f32,
        "the behavior fired once per tick of the run"
    );
    for &ball in &balls {
        let y = height(&app, ball);
        assert!(
            (y - 0.5).abs() < 0.05,
            "the ball fell from 1.5 and rests on the floor (y = {y})"
        );
    }

    // And it is still settled: another window's worth of ticks leaves the
    // bodies where they are, which is the state the invariant was judged over.
    let before: Vec<f32> = balls.iter().map(|&b| height(&app, b)).collect();
    assert_eq!(app.run_for(QUIET_WINDOW_TICKS), Ok(StepResult::Continue));
    for (&ball, y) in balls.iter().zip(before) {
        assert!(
            (height(&app, ball) - y).abs() < 1.0e-4,
            "a settled body must stay put",
        );
    }
}

// Both systems run over the one world, in table order.
#[test]
fn a_simulating_world_starts_both_headless_systems() {
    let (world, ..) = simulating_world();
    let mut app = App::with_systems(world, HEADLESS_SYSTEMS);
    app.start().expect("the world starts");

    let names: Vec<&str> = app.world().systems().iter().map(|s| s.name()).collect();
    assert_eq!(names, ["BehaviorSystem", "PhysicsSystem"]);
}

// The per-frame allocation-regression gate: these tests pin the steady-state
// heap churn of a world under the runtime's table, so a change that
// reintroduces per-frame allocation (a rebuilt HashMap, a fresh Vec per tick)
// fails loudly instead of eroding it back. The extraction pass's own pin sits
// beside the engine's `FrameExtractor`.
//
// Each pin steps through `quietest_allocs`, since other test threads allocate
// against the same process-wide counters.

use concinnity_engine::components::{
    Behavior, BehaviorExpr, BehaviorLiteral, BehaviorNode, BehaviorSource, Binding, Collider,
    PhysicsConfig, Prop, PropCollider, TextLabel, Transform, VariableDecl, Variables,
};
use concinnity_engine::ecs::World;
use concinnity_runtime::SYSTEMS;

const WARMUP_FRAMES: usize = 64;

// The pinned steady-state heap cost of one `World::step` of the static world
// below. Zero: every system reserves its working memory at init and reuses it,
// and the simulation reserves the world's whole body budget the same way, so a
// frame that changes nothing allocates nothing. Re-pin (with a commit-message
// explanation) only when a change legitimately moves it; a bump without one is
// the regression this gate exists to catch.
const STATIC_WORLD_ALLOCS_PER_FRAME: u64 = 0;

// A headless static world: nothing moves, nothing spawns, nothing streams.
// Props give the behavior scope a real population, the colliders give physics
// a settled static scene, and the tick behavior keeps the eval path hot
// without changing world state.
fn static_world() -> World {
    let mut world = World::new();
    world.add_component(Behavior {
        on: BehaviorSource::Tick,
        body: vec![BehaviorNode::Set {
            var: "beat".into(),
            value: BehaviorExpr::Int(1),
            add: true,
        }],
        ..Default::default()
    });
    for i in 0..200 {
        world.add_component(Prop {
            position: [i as f32 * 2.0, 0.5, 0.0],
            ..Default::default()
        });
    }
    // The variable the behavior moves every tick, shown by a bound label, and
    // a vector variable published beside it.
    world.add_component(Variables {
        vars: vec![
            VariableDecl {
                name: "beat".into(),
                value: BehaviorLiteral::Int(0),
            },
            VariableDecl {
                name: "spot".into(),
                value: BehaviorLiteral::Vec3([1.0, 2.0, 3.0]),
            },
        ],
    });
    world.add_component(TextLabel {
        text_binding: Some(Binding::Var("beat".into())),
        ..Default::default()
    });
    world.add_component(PhysicsConfig::default());
    for i in 0..8 {
        let e = world.push(Transform {
            position: [i as f32 * 25.0, 0.0, 0.0],
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
        });
        world.insert(
            e,
            Collider(PropCollider {
                shape: concinnity_engine::components::PropColliderShape::Cuboid,
                half_extents: [10.0, 0.5, 10.0],
                ..Default::default()
            }),
        );
    }
    world
}

// A static world's steady-state frame must stay at the pinned allocation
// count. This is the audit's guardrail: the sim-side systems run every frame
// against unchanged state, so their working memory must be persistent buffers
// or frame scratch, not fresh heap blocks.
#[test]
fn static_world_frame_allocs_stay_pinned() {
    let mut world = static_world();
    world.start(SYSTEMS).unwrap();
    for _ in 0..WARMUP_FRAMES {
        world.step();
    }

    let min = concinnity_testing::quietest_allocs(STATIC_WORLD_ALLOCS_PER_FRAME, || {
        world.step();
    });
    assert_eq!(
        min,
        STATIC_WORLD_ALLOCS_PER_FRAME,
        "static world's quietest frame allocated {min} times \
         (pinned at {STATIC_WORLD_ALLOCS_PER_FRAME}); a per-frame allocation crept in.\n\
         last frame by system: {:?}",
        world.profile().system_allocs(),
    );
    // The bound label kept up with the variable the whole time.
    let shown = world
        .query::<TextLabel>()
        .find(|l| l.text_binding.is_some())
        .map(|l| l.content.clone())
        .unwrap_or_default();
    assert!(
        shown.parse::<i32>().is_ok_and(|beat| beat > 0),
        "label shows {shown:?}"
    );
}

// The dev-build frame loop samples allocation deltas into the profile: the
// whole-frame count and one per-system entry beside each timing, in the same
// order. This is what the debug server's `profile` command and the HUD read.
#[cfg(debug_assertions)]
#[test]
fn frame_loop_samples_alloc_deltas_into_the_profile() {
    let mut world = static_world();
    world.start(SYSTEMS).unwrap();
    world.step();
    // The per-system list rotates with the timings, so it is readable after
    // the second step; the frame total is written as each step ends.
    world.step();

    let profile = world.profile();
    assert!(profile.frame_allocs().is_some());
    let timed: Vec<&str> = profile.system_timings().iter().map(|&(n, _)| n).collect();
    let counted: Vec<&str> = profile.system_allocs().iter().map(|&(n, _)| n).collect();
    assert_eq!(timed, counted, "alloc entries mirror the timing entries");
    assert!(!counted.is_empty(), "the static world runs systems");
}

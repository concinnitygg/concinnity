// src/bench/alloc_budget.rs
//
// The per-frame allocation-regression gate. The 2026-08 allocation audit cut
// the engine's steady-state heap churn to a small floor; these tests pin that
// floor so a change that reintroduces per-frame allocation (a rebuilt HashMap,
// a fresh Vec per tick) fails loudly instead of eroding it back.
//
// Unlike the microbenches beside this file, the pins are NOT ignored: they run
// in every `cargo test`. The allocation counters are process-wide and other
// test threads allocate concurrently, so a single frame's delta is only an
// upper bound on the world's own cost. Concurrency can only ADD allocations,
// never hide them, so each pin keeps stepping until one frame lands at or
// under its budget: that frame proves the world's own cost is within it. A
// regression can never produce such a frame, while a run polluted by parallel
// tests finds its quiet frame as the rest of the suite drains -- only a whole
// deadline with no quiet frame fails.

use super::BenchWorld;
use crate::components::{
    Behavior, BehaviorExpr, BehaviorNode, BehaviorSource, Collider, GlobalTransform, PhysicsConfig,
    Prop, PropCollider, RenderHandle, Transform,
};
use crate::ecs::SYSTEMS;
use crate::ecs::World;
use crate::gfx::graphics_system::GraphicsSystem;
use crate::gfx::snapshot::RenderSnapshot;

const WARMUP_FRAMES: usize = 64;

// How long a pin keeps looking for a quiet frame before giving up. Generous on
// purpose: the whole suite runs in about a second, so a real steady state is
// found in the first few frames once the concurrent noise drains, and only a
// genuine regression spends the full window.
const QUIET_FRAME_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

// The pinned steady-state heap cost of one `World::step` of the static world
// below. Zero: every system reserves its working memory at init and reuses it,
// and the simulation reserves the world's whole body budget the same way, so a
// frame that changes nothing allocates nothing. Re-pin (with a commit-message
// explanation) only when a change legitimately moves it; a bump without one is
// the regression this gate exists to catch.
const STATIC_WORLD_ALLOCS_PER_FRAME: u64 = 0;

// Step until one frame allocates at most `budget` times, returning the
// quietest delta seen: at most `budget` when the pin holds, or the closest
// the deadline's worth of frames ever came when it does not.
fn quietest_frame(budget: u64, mut step: impl FnMut()) -> u64 {
    let deadline = std::time::Instant::now() + QUIET_FRAME_DEADLINE;
    let mut min = u64::MAX;
    loop {
        let before = concinnity_memory::alloc_count().expect("test binary tracks its heap");
        step();
        let after = concinnity_memory::alloc_count().expect("allocator stays installed");
        min = min.min(after - before);
        if min <= budget || std::time::Instant::now() >= deadline {
            return min;
        }
    }
}

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
                shape: "cuboid".into(),
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

    let min = quietest_frame(STATIC_WORLD_ALLOCS_PER_FRAME, || {
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
}

// Static render extraction is allocation-free: the change gate drops every
// slot, so a frame where nothing moved must not touch the heap. The
// extraction microbench prints this as a column; this is the loud version.
#[test]
fn static_extraction_allocates_nothing() {
    let mut world = BenchWorld::new();
    for i in 0..100u32 {
        let entity = world.components.push_typed(Prop::default());
        world
            .components
            .insert_typed(entity, GlobalTransform(glam_identity_at(i)));
        world
            .components
            .insert_typed(entity, RenderHandle { draws: [i].into() });
    }
    let mut gs = GraphicsSystem::new();
    let mut snap = RenderSnapshot::default();
    for _ in 0..WARMUP_FRAMES {
        gs.extract(&mut world.ctx(), &mut snap);
    }

    let min = quietest_frame(0, || {
        gs.extract(&mut world.ctx(), &mut snap);
    });
    assert_eq!(
        min, 0,
        "static extraction's quietest pass allocated {min} times; \
         the zero-alloc extraction contract broke"
    );
}

fn glam_identity_at(i: u32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [i as f32, 0.0, 0.0, 1.0],
    ]
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

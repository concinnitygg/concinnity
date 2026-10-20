use super::*;
use alloc::string::{String, ToString};
use alloc::vec;

use crate::components::{
    CameraController, CharacterRig, Collider, FollowController, Pickup, PropCollider,
    PropColliderShape,
};
use crate::ecs::{Ref, SkinnedMeshHandle, World};
use crate::physics::LayerMask;
use crate::physics::capacity::{record_of, scan_counts};

// Push a decomposed prop (Transform + Collider, plus the Pickup tag when
// asked) exactly as the load-time decomposition would, and identify it as
// `id` so joints can resolve it.
fn spawn_prop(world: &mut World, id: AssetId, position: [f32; 3], pickup: bool) -> Entity {
    let entity = world.push(Transform {
        position,
        ..Default::default()
    });
    world.insert(
        entity,
        Collider(PropCollider {
            shape: PropColliderShape::Ball,
            half_extents: [0.5; 3],
            radius: 0.5,
            half_height: 0.0,
            layer: String::new(),
        }),
    );
    if pickup {
        world.insert(entity, Pickup);
    }
    world.identify(entity, id);
    entity
}

// Make a spawned prop dynamic, exactly as the load-time PropBody
// decomposition would.
fn make_dynamic(world: &mut World, entity: Entity) {
    world.insert(entity, ball_dynamics());
}

fn controlled_camera() -> Camera3D {
    Camera3D {
        fov_y_degrees: 75.0,
        near: 0.05,
        view_distance: None,
        view_matrix: [[0.0; 4]; 4],
        position: [0.0, 1.0, 0.0],
        yaw: 0.0,
        pitch: 0.0,
        up: [0.0, 1.0, 0.0],
        desired_move: [0.0; 3],
        jump_requested: false,
        interact_requested: false,
        controller: Some(CameraController::default()),
    }
}

// A third-person camera is a virtual orbit: no player capsule is created
// for it. (Regression: the spectator capsule spawned at the camera eye
// overlapped the followed rig's capsule and squeezed it through the
// floor.) A first-person camera keeps its capsule. The schedule gate that
// builds the system at all is covered by the engine's schedule tests.
#[test]
fn third_person_camera_gets_no_player_capsule() {
    // Third-person (follow) camera: a virtual orbit, so no player capsule.
    let mut world = World::new();
    let mut camera = controlled_camera();
    camera.controller = Some(CameraController {
        follow: Some(FollowController {
            target: Some(SkinnedMeshHandle::new(1)),
            ..FollowController::default()
        }),
        ..CameraController::default()
    });
    world.push(camera);
    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());
    assert!(physics.player.is_none(), "no capsule for the orbit camera");

    // First-person camera keeps its capsule.
    let mut world = World::new();
    world.push(controlled_camera());
    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());
    assert!(
        physics.player.is_some(),
        "first-person camera keeps its capsule"
    );
}

// A walker standing on a planet's ground patch has the simulated frame
// moved under it, with the ground rebuilt in the new frame: it keeps
// standing where it was carried to, with gravity toward the carried center.
#[test]
fn a_walker_on_a_planet_stands_through_a_frame_move() {
    use crate::planet::{
        FrameRebases, LocalFrame, PlanetGravity, PlanetGround, PlanetShape, ground_patch,
    };
    let shape = PlanetShape {
        center: [0.0, -5_000.0, 0.0],
        radius: 5_000.0,
        amplitude: 0.0,
        feature_size: 100.0,
        octaves: 1,
        seed: 0,
    };
    let mut world = World::new();
    world.push(RigidBody::default());
    world.push(Camera3D {
        position: [400.0, 2.0, 0.0],
        ..controlled_camera()
    });
    let region = world.push(TriggerVolume {
        position: [404.0, 1.0, -2.0],
        rotation_deg: [0.0, 30.0, 0.0],
        ..Default::default()
    });
    let publish = |world: &mut World, frame: &LocalFrame, around: [f32; 3], revision| {
        let patch = ground_patch(&shape, frame, around, 32, 16.0).unwrap();
        world.insert_resource(PlanetGround { revision, patch });
        world.insert_resource(PlanetGravity {
            center: frame.to_local(shape.center),
            strength: 20.0,
        });
    };
    let authored = LocalFrame::AUTHORED;
    publish(&mut world, &authored, [400.0, 0.0, 0.0], 0);
    world.insert_resource(FrameRebases::default());
    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());
    for _ in 0..180 {
        physics.step(&mut world.context());
    }
    let eye = |world: &World| world.query::<Camera3D>().next().unwrap().position;
    let height = |p: [f32; 3], frame: &LocalFrame| {
        let world_p = frame.to_world(p);
        let d = world_p
            .iter()
            .zip(shape.center)
            .map(|(a, c)| (a - c) * (a - c))
            .sum::<f64>();
        libm::sqrt(d) - shape.radius
    };
    let standing = height(eye(&world), &authored);
    assert!(
        standing > 0.5 && standing < 3.0,
        "on the ground: {standing}"
    );

    let at = eye(&world);
    let (moved, rebase) = authored.recentered(at, shape.up_at(authored.to_world(at)));
    for camera in world.context().query_mut::<Camera3D>() {
        camera.position = rebase.apply_point(camera.position);
    }
    let mut rebases = FrameRebases::default();
    rebases.push(rebase);
    world.insert_resource(rebases);
    publish(&mut world, &moved, [0.0; 3], 1);
    for _ in 0..120 {
        physics.step(&mut world.context());
    }
    let after = eye(&world);
    assert!(
        after.iter().all(|c| c.abs() < 0.05),
        "carried, not moved: {after:?}"
    );
    let still = height(after, &moved);
    assert!((still - standing).abs() < 0.02, "{still} vs {standing}");
    let volume = world.get::<TriggerVolume>(region).unwrap();
    let expected = rebase.apply_point([404.0, 1.0, -2.0]);
    assert!((0..3).all(|i| (volume.position[i] - expected[i]).abs() < 1e-3));
    assert_eq!(
        volume.rotation_deg,
        rebase.apply_euler_deg([0.0, 30.0, 0.0])
    );
}

fn ball_dynamics() -> BodyDynamics {
    BodyDynamics {
        mass: 1.0,
        friction: 0.5,
        linear_damping: 0.0,
        ..Default::default()
    }
}

// A camera looking down -Z with the interact field latched on, so the
// PhysicsSystem (which reads Camera3D.interact_requested) triggers a pickup.
fn interacting_camera(position: [f32; 3]) -> Camera3D {
    Camera3D {
        interact_requested: true,
        controller: None,
        position,
        ..controlled_camera()
    }
}

// The simulated pose is written back to the prop's Transform. With no
// `SimTiming` published, each step runs exactly one fixed tick.
#[test]
fn dynamic_prop_writes_transform() {
    let id = AssetId(1);
    let mut world = World::new();
    let entity = spawn_prop(&mut world, id, [0.0, 5.0, 0.0], false);
    make_dynamic(&mut world, entity);

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());
    for _ in 0..10 {
        physics.step(&mut world.context());
    }

    let transform_y = world.get::<Transform>(entity).unwrap().position[1];
    assert!(
        transform_y < 5.0,
        "the simulated pose falls the Transform (y={transform_y})"
    );
}

// A menu freezes the solve: while `MenuActive(true)` is published the body
// does not fall, and clearing it resumes the fall from where it froze.
// (The runtime-level simulation clock additionally holds its accumulator
// across the pause, so a live run resumes without a catch-up burst.)
#[test]
fn menu_active_freezes_then_resumes_physics() {
    let id = AssetId(1);
    let mut world = World::new();
    let entity = spawn_prop(&mut world, id, [0.0, 5.0, 0.0], false);
    make_dynamic(&mut world, entity);

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());

    // Paused: the body stays put however many frames pass.
    world.insert_resource(MenuActive(true));
    for _ in 0..5 {
        physics.step(&mut world.context());
    }
    let y_paused = world.get::<Transform>(entity).unwrap().position[1];
    assert!(
        (y_paused - 5.0).abs() < 1e-3,
        "the body must not fall while a menu is active (y={y_paused})"
    );

    // Resumed: the body falls again.
    world.insert_resource(MenuActive(false));
    for _ in 0..5 {
        physics.step(&mut world.context());
    }
    let y_resumed = world.get::<Transform>(entity).unwrap().position[1];
    assert!(
        y_resumed < y_paused - 1e-3,
        "the body must fall once the menu closes (y={y_resumed})"
    );
}

// A zero-tick frame (the accumulator has not crossed a tick) advances
// nothing; the written Transform blends between the last two ticks by the
// frame's alpha.
#[test]
fn zero_tick_frames_blend_between_the_last_two_ticks() {
    let id = AssetId(1);
    let mut world = World::new();
    let entity = spawn_prop(&mut world, id, [0.0, 5.0, 0.0], false);
    make_dynamic(&mut world, entity);

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());

    let tick = |ticks, alpha| SimTiming {
        ticks,
        tick_dt: SimTiming::TICK_DT,
        alpha,
    };
    world.insert_resource(tick(1, 1.0));
    physics.step(&mut world.context());
    let y_prev = world.get::<Transform>(entity).unwrap().position[1];
    physics.step(&mut world.context());
    let y_curr = world.get::<Transform>(entity).unwrap().position[1];
    assert!(y_curr < y_prev, "the body falls tick over tick");

    // No tick, alpha 0: the write-back returns to the previous tick's pose.
    world.insert_resource(tick(0, 0.0));
    physics.step(&mut world.context());
    let y_alpha0 = world.get::<Transform>(entity).unwrap().position[1];
    assert!(
        (y_alpha0 - y_prev).abs() < 1e-6,
        "alpha 0 samples the previous tick"
    );

    // No tick, alpha 0.5: halfway between the two ticks.
    world.insert_resource(tick(0, 0.5));
    physics.step(&mut world.context());
    let y_mid = world.get::<Transform>(entity).unwrap().position[1];
    let expected = (y_prev + y_curr) * 0.5;
    assert!(
        (y_mid - expected).abs() < 1e-6,
        "alpha 0.5 blends the tick poses (y={y_mid}, expected {expected})"
    );
}

// The same number of fixed ticks produces bit-identical world state
// however they are grouped into frames: 30 fps frames (two ticks each)
// against 120 fps frames (a tick every other frame).
#[test]
fn tick_grouping_does_not_change_the_outcome() {
    let run = |frames: &[u32]| -> ([f32; 3], [f32; 3]) {
        let id = AssetId(1);
        let mut world = World::new();
        let entity = spawn_prop(&mut world, id, [0.3, 5.0, 0.1], false);
        make_dynamic(&mut world, entity);

        let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
        physics.init(&mut world.context());
        for &ticks in frames {
            world.insert_resource(SimTiming {
                ticks,
                tick_dt: SimTiming::TICK_DT,
                alpha: 1.0,
            });
            physics.step(&mut world.context());
        }
        let t = world.get::<Transform>(entity).unwrap();
        (t.position, t.rotation_deg)
    };

    // One simulated second: 30 frames of 2 ticks vs 120 frames alternating
    // 0 and 1 ticks. Both run exactly 60 fixed ticks.
    let thirty: Vec<u32> = core::iter::repeat_n(2, 30).collect();
    let one_twenty: Vec<u32> = (0..120).map(|i| i % 2).collect();
    assert_eq!(
        run(&thirty),
        run(&one_twenty),
        "the fixed-tick outcome must not depend on frame grouping"
    );
}

// Despawning a decomposed prop reaps its body, so it stops simulating
// (and colliding) once its entity is gone.
#[test]
fn despawning_a_prop_reaps_its_physics_body() {
    let id = AssetId(1);
    let mut world = World::new();
    let ball = spawn_prop(&mut world, id, [0.0, 5.0, 0.0], false);
    make_dynamic(&mut world, ball);

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());

    // Settle so the body is live and falling.
    for _ in 0..2 {
        physics.step(&mut world.context());
    }
    let before = physics.physics_body_count();

    // Despawn the ball (stand-in for the renderer) and step: PhysicsSystem
    // reaps the orphaned body.
    world.despawn(ball);
    physics.step(&mut world.context());
    let after = physics.physics_body_count();
    assert_eq!(after, before - 1, "the despawned prop's body was removed");

    // The sim keeps running cleanly with the body gone (no further removals).
    physics.step(&mut world.context());
    assert_eq!(
        physics.physics_body_count(),
        after,
        "no further bodies removed"
    );
}

// Picking up a carriable prop tags its entity with Held.
#[test]
fn pickup_sets_held_tag() {
    let id = AssetId(1);
    let mut world = World::new();
    let carriable = spawn_prop(&mut world, id, [0.0, 1.0, -2.0], true);
    make_dynamic(&mut world, carriable);
    world.push(interacting_camera([0.0, 1.0, 0.0]));

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());

    physics.step(&mut world.context());

    assert_eq!(
        world.context().query::<Held>().count(),
        1,
        "pickup inserts the Held tag on the entity"
    );
}

// A collider-bearing entity that appears after init (a runtime spawn) is
// adopted on the next step: it gets a body and falls like an authored one.
// The headroom is what reserves the body for it: the simulation is sized
// once at init and a spawn past the reservation is refused.
#[test]
fn runtime_spawned_prop_gets_a_body_and_falls() {
    let mut world = World::new();
    let mut physics = PhysicsSystem::new(&PhysicsConfig {
        spawn_headroom: 1,
        ..PhysicsConfig::default()
    });
    physics.init(&mut world.context());
    let baseline = physics.physics_body_count();

    let spawned = spawn_prop(&mut world, AssetId(7), [0.0, 5.0, 0.0], false);
    make_dynamic(&mut world, spawned);
    physics.step(&mut world.context());
    assert_eq!(
        physics.physics_body_count(),
        baseline + 1,
        "the spawned entity got a body on its first step"
    );
    for _ in 0..30 {
        physics.step(&mut world.context());
    }
    let y = world.get::<Transform>(spawned).unwrap().position[1];
    assert!(y < 4.5, "the spawned body falls (y = {y})");
    for _ in 0..300 {
        physics.step(&mut world.context());
    }
    let y = world.get::<Transform>(spawned).unwrap().position[1];
    assert!(
        (y - 0.5).abs() < 0.1,
        "the spawned ball rests on the flat floor (y = {y})"
    );
}

// A Transform written from outside physics (a behavior's `set_transform`,
// an editor drag) is adopted rather than overwritten: the body is moved
// there, at rest, and the pose written back is the one that was asked for.
#[test]
fn an_externally_written_transform_moves_a_tracked_prop() {
    let mut world = World::new();
    let ball = spawn_prop(&mut world, AssetId(1), [0.0, 5.0, 0.0], false);
    make_dynamic(&mut world, ball);
    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());
    for _ in 0..30 {
        physics.step(&mut world.context());
    }
    let fallen = world.get::<Transform>(ball).unwrap().position;
    assert!(fallen[1] < 4.5, "it was falling first (y = {})", fallen[1]);

    world
        .get_mut::<Transform>(ball)
        .expect("the prop has a transform")
        .position = [2.0, 12.0, -1.0];
    physics.step(&mut world.context());

    let landed = world.get::<Transform>(ball).unwrap().position;
    assert!((landed[0] - 2.0).abs() < 1.0e-4, "x = {}", landed[0]);
    assert!((landed[2] + 1.0).abs() < 1.0e-4, "z = {}", landed[2]);
    // A tick of fall from rest is about 0.006; the speed it had built up
    // would have cost twenty times that.
    assert!((landed[1] - 12.0).abs() < 0.02, "y = {}", landed[1]);
}

// Spawn, despawn, and respawn leave no bodies or colliders behind: a
// reaped body hands its slot back, so one body's worth of headroom covers
// any number of rounds.
#[test]
fn spawn_despawn_respawn_cycle_is_leak_free() {
    let mut world = World::new();
    let mut physics = PhysicsSystem::new(&PhysicsConfig {
        spawn_headroom: 1,
        ..PhysicsConfig::default()
    });
    physics.init(&mut world.context());
    let bodies = physics.physics_body_count();
    let colliders = physics.physics_collider_count();

    for round in 0..3 {
        let spawned = spawn_prop(&mut world, AssetId(100 + round), [0.0, 3.0, 0.0], false);
        make_dynamic(&mut world, spawned);
        physics.step(&mut world.context());
        assert_eq!(physics.physics_body_count(), bodies + 1, "round {round}");
        world.despawn(spawned);
        physics.step(&mut world.context());
        assert_eq!(
            physics.physics_body_count(),
            bodies,
            "round {round} reaped the body"
        );
        assert_eq!(
            physics.physics_collider_count(),
            colliders,
            "round {round} reaped the collider"
        );
    }
}

// A world whose shipped budget was counted from the same content it holds
// passes the init assert, and reserves the cap that budget implies.
#[test]
fn a_shipped_budget_matching_the_world_is_adopted() {
    let mut world = World::new();
    let entity = spawn_prop(&mut world, AssetId(1), [0.0, 3.0, 0.0], false);
    make_dynamic(&mut world, entity);
    world.push(controlled_camera());

    // The record cook would have written for this world: one dynamic prop,
    // the floor, the player capsule, and room for two spawns.
    let counts = scan_counts(&world.context());
    let budget = PhysicsBudget::derive(&counts, 2);
    assert_eq!(budget.dynamic, 1);
    assert_eq!(budget.kinematic, 1, "the first-person camera capsule");
    world.insert_resource(WorldPhysicsBudget(record_of(&budget)));

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());
    assert_eq!(physics.body_cap, budget.body_cap());
    assert_eq!(
        physics.physics_body_count(),
        budget.body_total() as usize,
        "init built exactly the bodies the budget reserved"
    );
}

// A world with no shipped budget (built in memory, as every test world is)
// reserves from what it holds plus the headroom its config authored, and
// caps spawns there. The cap used to be left open for such a world, on the
// grounds that nothing had counted its spawns; a fixed-capacity simulation
// cannot honor that -- a spawn past the reservation gets no body either
// way, and the cap is what turns a silently declined one into a refusal
// naming the knob to raise.
#[test]
fn a_world_without_a_shipped_budget_reserves_from_what_it_holds() {
    let mut world = World::new();
    let entity = spawn_prop(&mut world, AssetId(1), [0.0, 3.0, 0.0], false);
    make_dynamic(&mut world, entity);

    let mut physics = PhysicsSystem::new(&PhysicsConfig {
        spawn_headroom: 4,
        ..PhysicsConfig::default()
    });
    physics.init(&mut world.context());

    assert_eq!(
        physics.physics_body_count(),
        2,
        "the floor and the one authored prop"
    );
    assert_eq!(physics.body_cap, 2 + 4, "plus the authored headroom");
}

// Past the cap, a spawned prop gets no body and the world keeps stepping.
// The refusal happens once: the next tick's scan passes over the entity
// rather than re-refusing it.
#[test]
fn a_spawn_past_the_shipped_budget_is_refused() {
    let mut world = World::new();
    let authored = spawn_prop(&mut world, AssetId(1), [0.0, 3.0, 0.0], false);
    make_dynamic(&mut world, authored);

    // A budget with no headroom: the floor and the one authored prop.
    let counts = scan_counts(&world.context());
    let budget = PhysicsBudget::derive(&counts, 0);
    world.insert_resource(WorldPhysicsBudget(record_of(&budget)));

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());
    let full = physics.physics_body_count();
    assert_eq!(full, budget.body_cap() as usize, "the budget is spent");

    let spawned = spawn_prop(&mut world, AssetId(2), [0.0, 6.0, 0.0], false);
    make_dynamic(&mut world, spawned);
    physics.step(&mut world.context());
    assert_eq!(physics.physics_body_count(), full, "no body was built");
    assert!(physics.props.is_refused(spawned));

    // Stepping on keeps the refusal: the entity is skipped, not retried,
    // and the authored prop carries on simulating.
    for _ in 0..10 {
        physics.step(&mut world.context());
    }
    assert_eq!(physics.physics_body_count(), full);
    let refused_y = world.get::<Transform>(spawned).unwrap().position[1];
    assert_eq!(refused_y, 6.0, "a refused prop is not simulated at all");
    let live_y = world.get::<Transform>(authored).unwrap().position[1];
    assert!(live_y < 3.0, "the authored prop still falls");
}

// A rig capsule authored standing exactly on the floor stays there: it
// neither sinks tick after tick nor is lifted off it. The driver used to
// spawn one a fingernail above its authored position, because the
// controller ignored a hit a downward move started already touching and
// the capsule sank a little every frame; the controller now separates
// along the contact normal instead, so the authored position is the one
// that holds.
#[test]
fn a_rig_capsule_spawned_on_the_floor_neither_sinks_nor_rises() {
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut world = World::new();
    world.push(CharacterRig::new(
        SkinnedMeshHandle::new(1),
        crate::gfx::render_types::SkinnedIndex(0),
        identity,
        0.6,
        0.3,
    ));

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());
    let rig_y = |world: &mut World| {
        world
            .context()
            .query::<CharacterRig>()
            .next()
            .expect("the rig is there")
            .position[1]
    };
    assert_eq!(
        rig_y(&mut world),
        0.0,
        "the capsule spawns at its authored position, unlifted"
    );

    for _ in 0..30 {
        physics.step(&mut world.context());
    }
    let settled = rig_y(&mut world);
    for _ in 0..300 {
        physics.step(&mut world.context());
    }
    let held = rig_y(&mut world);

    assert!(
        settled.abs() < 0.01,
        "the capsule stayed on the floor (y = {settled})"
    );
    assert!(
        (held - settled).abs() < 1.0e-4,
        "and stopped moving ({settled} -> {held})"
    );
}

// A hard landing publishes one ContactEvent naming the prop; resting
// afterwards stays silent.
#[test]
fn contact_event_fires_on_impact_and_not_at_rest() {
    let id = AssetId(1);
    let mut world = World::new();
    let entity = spawn_prop(&mut world, id, [0.0, 5.0, 0.0], false);
    make_dynamic(&mut world, entity);

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());

    let mut cursor = EventCursor::default();
    let mut impacts: Vec<ContactEvent> = Vec::new();
    for _ in 0..120 {
        physics.step(&mut world.context());
        let ctx = world.context();
        if let Some(events) = ctx.events::<ContactEvent>() {
            impacts.extend(events.read(&mut cursor).copied());
        }
    }
    assert_eq!(impacts.len(), 1, "one landing, one event: {impacts:?}");
    let impact = impacts[0];
    assert_eq!(impact.a, entity);
    assert_eq!(impact.b, None, "the floor slab has no entity");
    assert!(
        impact.impulse > 3.0 && impact.impulse < 50.0,
        "impulse {} out of the plausible landing range",
        impact.impulse
    );

    // Settled: hundreds of resting ticks publish nothing further.
    for _ in 0..300 {
        physics.step(&mut world.context());
        let ctx = world.context();
        if let Some(events) = ctx.events::<ContactEvent>() {
            assert_eq!(
                events.read(&mut cursor).count(),
                0,
                "resting contact must not publish events"
            );
        }
    }
}

// A sensor region reports a prop crossing it, in and then out, naming the
// volume that saw it.
#[test]
fn a_trigger_volume_reports_a_prop_crossing_it() {
    let volume_id = AssetId(9);
    let mut world = World::new();
    world.context().push_identified(
        volume_id,
        TriggerVolume {
            position: [0.0, 3.0, 0.0],
            rotation_deg: [0.0; 3],
            collider: PropCollider {
                shape: PropColliderShape::Cuboid,
                half_extents: [1.0, 0.5, 1.0],
                ..Default::default()
            },
            detects: TriggerFilter::Props,
        },
    );
    let ball = spawn_prop(&mut world, AssetId(1), [0.0, 6.0, 0.0], false);
    make_dynamic(&mut world, ball);

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());

    let mut cursor = EventCursor::default();
    let mut crossings: Vec<VolumeEvent> = Vec::new();
    for _ in 0..180 {
        physics.step(&mut world.context());
        let ctx = world.context();
        if let Some(events) = ctx.events::<VolumeEvent>() {
            crossings.extend(events.read(&mut cursor).copied());
        }
    }
    assert_eq!(crossings.len(), 2, "in, then out: {crossings:?}");
    assert!(crossings.iter().all(|c| c.volume == volume_id));
    assert!(crossings[0].entered, "the ball entered first");
    assert!(!crossings[1].entered, "and left afterwards");
}

// A joint anchored to the world holds its body up: the hidden anchor body
// the driver mints for it is part of the reservation, and the prop hangs
// off it instead of falling.
#[test]
fn a_world_anchored_joint_holds_its_prop_up() {
    let bob_id = AssetId(1);
    let mut world = World::new();
    let bob = spawn_prop(&mut world, bob_id, [1.0, 4.0, 0.0], false);
    make_dynamic(&mut world, bob);
    world.push(PhysicsJoint {
        kind: crate::components::PhysicsJointKind::Spherical,
        body_a: Some(Ref::new(bob_id)),
        body_b: None,
        // The bob's own center hangs one unit from the anchor point.
        anchor_a: [-1.0, 0.0, 0.0],
        anchor_b: [0.0, 4.0, 0.0],
        ..Default::default()
    });

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());
    for _ in 0..240 {
        physics.step(&mut world.context());
    }

    let position = world.get::<Transform>(bob).unwrap().position;
    let reach =
        ((position[0]).powi(2) + (position[1] - 4.0).powi(2) + (position[2]).powi(2)).sqrt();
    assert!(
        (reach - 1.0).abs() < 0.05,
        "the bob hangs one unit from the anchor, at {position:?} ({reach})"
    );
    assert!(
        position[1] > 2.5,
        "and is held up rather than falling ({position:?})"
    );
}

// The config's no_collide pairs reach the built colliders: a prop on a
// layer that ignores `world` falls straight through the floor.
#[test]
fn no_collide_config_lets_a_layered_prop_fall_through_the_floor() {
    let config = PhysicsConfig {
        layers: vec!["ghost".to_string()],
        no_collide: vec![["ghost".to_string(), "world".to_string()]],
        ..PhysicsConfig::default()
    };

    let mut world = World::new();
    let entity = spawn_prop(&mut world, AssetId(1), [0.0, 2.0, 0.0], false);
    make_dynamic(&mut world, entity);
    world.get_mut::<Collider>(entity).unwrap().0.layer = "ghost".to_string();

    let mut physics = PhysicsSystem::new(&config);
    physics.init(&mut world.context());
    for _ in 0..240 {
        physics.step(&mut world.context());
    }
    let y = world.get::<Transform>(entity).unwrap().position[1];
    assert!(
        y < -10.0,
        "the ghost-layer prop fell through the floor (y = {y})"
    );
}

// Where the floor sits under (x, z), by dropping a ray onto it.
fn floor_height(physics: &PhysicsSystem, x: f32, z: f32) -> Option<f32> {
    physics
        .world
        .as_ref()?
        .raycast([x, 100.0, z], [0.0, -1.0, 0.0], 200.0, None, LayerMask::ALL)
        .map(|hit| hit.point[1])
}

#[test]
fn a_world_without_terrain_stands_on_a_level_slab() {
    let mut world = World::new();
    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());

    let a = floor_height(&physics, 0.0, 0.0).expect("the ray meets the floor");
    let b = floor_height(&physics, 12.0, -7.0).expect("the ray meets the floor");
    assert_eq!(a, b, "a slab is level");
    assert_eq!(a, 0.0);
}

// A terrain that cannot be read still leaves the world a floor to stand
// on rather than none at all.
#[test]
fn an_unreadable_terrain_falls_back_to_the_slab() {
    let mut world = World::new();
    world.push(crate::components::Terrain::default());
    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());
    assert_eq!(floor_height(&physics, 3.0, 4.0), Some(0.0));
}

// A capsule that jumps leaves the ground, then lands: the impulse is
// applied on the tick the request arrives, gravity brings it back, and
// touching down clears the downward velocity rather than letting it
// accumulate into the floor. The grounded state is published back to the
// RigidBody, which is what gates the next jump.
#[test]
fn a_grounded_player_jumps_and_lands() {
    let mut world = World::new();
    world.push(RigidBody {
        gravity_scale: 1.0,
        capsule_radius: 0.3,
        capsule_height: 1.8,
        jump_height: 1.0,
        ..RigidBody::default()
    });
    let mut camera = controlled_camera();
    camera.jump_requested = true;
    world.push(camera);

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());
    assert!(
        physics.player.as_ref().expect("a capsule").has_gravity,
        "a RigidBody in the world is what makes the capsule fall"
    );

    physics.step(&mut world.context());
    let launched = physics.player.as_ref().expect("a capsule").vy;
    assert!(
        launched > 0.0,
        "the jump did not lift the capsule: {launched}"
    );

    // Stop asking, and let it come back down.
    for camera in world.context().query_mut::<Camera3D>() {
        camera.jump_requested = false;
    }
    for _ in 0..240 {
        physics.step(&mut world.context());
    }

    let player = physics.player.as_ref().expect("a capsule");
    assert!(player.grounded, "the capsule never landed");
    assert_eq!(player.vy, 0.0, "landing clears the downward velocity");
    assert!(
        world
            .context()
            .query::<RigidBody>()
            .all(|body| body.is_grounded),
        "the grounded state that gates the next jump was not published"
    );
}

// Interacting twice picks the prop up and then throws it: the second
// interaction hands it back to dynamic simulation and drops the Held tag,
// so a carried prop cannot be left tagged after it has been released.
#[test]
fn interacting_twice_picks_up_then_throws_the_prop() {
    let id = AssetId(1);
    let mut world = World::new();
    let carriable = spawn_prop(&mut world, id, [0.0, 1.0, -2.0], true);
    make_dynamic(&mut world, carriable);
    world.push(interacting_camera([0.0, 1.0, 0.0]));

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());

    physics.step(&mut world.context());
    assert_eq!(world.context().query::<Held>().count(), 1, "picked up");
    assert!(physics.held.is_some());

    // The camera is still asking, so the next step is the drop.
    physics.step(&mut world.context());
    assert!(physics.held.is_none(), "the prop was not released");
    assert_eq!(
        world.context().query::<Held>().count(),
        0,
        "the Held tag outlived the throw"
    );
}

// Build a world holding one trigger volume with the given filter, plus a
// falling prop, and report the crossings `steps` steps produce.
fn crossings_through(detects: TriggerFilter, steps: usize) -> Vec<VolumeEvent> {
    let volume_id = AssetId(9);
    let mut world = World::new();
    world.context().push_identified(
        volume_id,
        TriggerVolume {
            position: [0.0, 3.0, 0.0],
            rotation_deg: [0.0; 3],
            collider: PropCollider {
                shape: PropColliderShape::Cuboid,
                half_extents: [1.0, 0.5, 1.0],
                ..Default::default()
            },
            detects,
        },
    );
    let ball = spawn_prop(&mut world, AssetId(1), [0.0, 6.0, 0.0], false);
    make_dynamic(&mut world, ball);

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());

    let mut cursor = EventCursor::default();
    let mut crossings = Vec::new();
    for _ in 0..steps {
        physics.step(&mut world.context());
        let ctx = world.context();
        if let Some(events) = ctx.events::<VolumeEvent>() {
            crossings.extend(events.read(&mut cursor).copied());
        }
    }
    crossings
}

// An `any` volume takes whatever crosses it, so the same prop that a
// `props` volume reports is reported here too.
#[test]
fn an_any_volume_reports_a_prop_crossing_it() {
    let crossings = crossings_through(TriggerFilter::Any, 180);
    assert_eq!(crossings.len(), 2, "in, then out: {crossings:?}");
}

// A camera nothing controls, moved from outside physics the way a camera
// track moves it, carries its capsule along: a `player` volume reports the
// camera arriving.
#[test]
fn a_camera_moved_from_outside_sets_off_a_player_volume() {
    let volume_id = AssetId(9);
    let mut world = World::new();
    world.context().push_identified(
        volume_id,
        TriggerVolume {
            position: [0.0, 5.0, -10.0],
            collider: PropCollider {
                shape: PropColliderShape::Cuboid,
                half_extents: [2.0; 3],
                ..Default::default()
            },
            detects: TriggerFilter::Player,
            ..Default::default()
        },
    );
    world.push(Camera3D {
        controller: None,
        position: [0.0, 5.0, 0.0],
        ..controlled_camera()
    });

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());

    let mut cursor = EventCursor::default();
    let mut entered = Vec::new();
    for step in 0..60 {
        for camera in world.context().query_mut::<Camera3D>() {
            camera.position[2] = -(step as f32) * 0.25;
        }
        physics.step(&mut world.context());
        let ctx = world.context();
        if let Some(events) = ctx.events::<VolumeEvent>() {
            entered.extend(
                events
                    .read(&mut cursor)
                    .filter(|e| e.entered)
                    .map(|e| e.volume),
            );
        }
    }
    assert_eq!(entered, vec![volume_id]);
}

// A prop authored asleep hangs where it was placed until a wake request
// names it, and falls from there.
#[test]
fn a_wake_request_drops_a_prop_authored_asleep() {
    let mut world = World::new();
    let ball = spawn_prop(&mut world, AssetId(1), [0.0, 6.0, 0.0], false);
    world.insert(
        ball,
        BodyDynamics {
            asleep: true,
            ..ball_dynamics()
        },
    );

    let mut physics = PhysicsSystem::new(&PhysicsConfig::default());
    physics.init(&mut world.context());
    let height = |world: &mut World| {
        world
            .context()
            .get::<Transform>(ball)
            .map(|t| t.position[1])
    };

    for _ in 0..60 {
        physics.step(&mut world.context());
    }
    assert_eq!(height(&mut world), Some(6.0), "it hung where it was placed");

    world
        .context()
        .events_mut::<WakeRequest>()
        .send(WakeRequest { target: ball });
    for _ in 0..30 {
        physics.step(&mut world.context());
    }
    assert!(height(&mut world).is_some_and(|y| y < 5.5), "it fell");
}

// A `player` volume classifies by body: a prop is not the player, so the
// same crossing that an `any` volume reports is filtered out here.
#[test]
fn a_player_volume_ignores_a_prop_crossing_it() {
    let crossings = crossings_through(TriggerFilter::Player, 180);
    assert!(
        crossings.is_empty(),
        "a prop is not the player: {crossings:?}"
    );
}

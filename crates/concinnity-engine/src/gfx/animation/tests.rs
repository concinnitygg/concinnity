// src/gfx/animation/tests.rs

use std::time::{Duration, Instant};

use super::resumed_origin;
use crate::assets::{AnimGraph, AnimParams, Animation};
use crate::ecs::SkinnedMeshHandle;
use crate::ecs::asset_id::{AssetId, intern};
use crate::ecs::{SystemAsset, World};

// Resuming after a pause must leave clip time `t = now - origin` exactly
// where it was when the pause began, so playback continues from the frozen
// pose with no jump, no matter how long the menu was open.
#[test]
fn resumed_origin_freezes_clip_time_across_pause() {
    let start = Instant::now();
    // Paused at t = 5s, menu held open for 30s of real time.
    let anchor = start + Duration::from_secs(5);
    let now = anchor + Duration::from_secs(30);

    let t_at_pause = (anchor - start).as_secs_f32();
    let new_origin = resumed_origin(start, anchor, now);
    let t_on_resume = (now - new_origin).as_secs_f32();

    assert!(
        (t_on_resume - t_at_pause).abs() < 1e-6,
        "clip time jumped across the pause: {t_at_pause} -> {t_on_resume}"
    );
}

// An `Animation` in the world implies the internal AnimationSystem: it is
// constructed by `World::start`, not declared as an asset.
#[test]
fn animation_component_spawns_internal_system() {
    let mut world = World::new_empty();
    world.add_component(Animation::default());
    world.start().unwrap();

    let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
    assert_eq!(names, ["AnimationSystem"]);
}

// No `Animation` means no AnimationSystem; the gate keys purely off world
// content.
#[test]
fn no_animation_no_internal_system() {
    let mut world = World::new_empty();
    world.start().unwrap();
    assert!(world.systems().is_empty());
}

// An `AnimGraph` alone also implies the system (a graph without clips is a
// build error, but the runtime gate must not depend on validation).
#[test]
fn anim_graph_component_spawns_internal_system() {
    let mut world = World::new_empty();
    world.add_component(AnimGraph::default());
    world.start().unwrap();

    let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
    assert_eq!(names, ["AnimationSystem"]);
}

// A clip targeting `hero`, named so graph states can reference it.
fn clip(name: &str, duration: f32) -> Animation {
    // Install the name resolver so the `"target":"hero"` reference deserializes.
    // Must not reset the interner: `intern` below accumulates ids across calls.
    crate::ecs::asset_id::ensure_name_resolver();
    let mut a: Animation = serde_json::from_value(serde_json::json!({
        "target": "hero",
        "duration": duration,
        "looping": true,
    }))
    .unwrap();
    a.asset_id = intern(name);
    a
}

// idle/run graph on `hero`, transitioning on `speed` in both directions with
// snap fades (duration 0) so state flips are visible after one step.
fn hero_graph() -> AnimGraph {
    let mut g: AnimGraph = serde_json::from_value(serde_json::json!({
        "target": "hero",
        "parameters": [{"name": "speed", "default": 0.0}],
        "initial": "idle",
        "states": [
            {"name": "idle", "clip": "idle_clip"},
            {"name": "run", "clip": "run_clip"}
        ],
        "transitions": [
            {"from": "idle", "to": "run",
             "conditions": [{"parameter": "speed", "op": "gt", "value": 0.5}]},
            {"from": "run", "to": "idle",
             "conditions": [{"parameter": "speed", "op": "le", "value": 0.5}]}
        ]
    }))
    .unwrap();
    g.asset_id = intern("hero_graph");
    g
}

fn graph_world() -> World {
    let mut world = World::new_empty();
    world.add_component(clip("idle_clip", 1.0));
    world.add_component(clip("run_clip", 0.8));
    world.add_component(hero_graph());
    world.start().unwrap();
    world
}

fn hero() -> AssetId {
    intern("hero")
}

// Match the system back out of the world for report assertions.
fn with_anim<R>(world: &mut World, f: impl FnOnce(&mut super::AnimationSystem) -> R) -> R {
    for system in world.systems_mut() {
        if let SystemAsset::AnimationSystem(anim) = system {
            return f(anim);
        }
    }
    panic!("AnimationSystem not constructed");
}

// Init publishes one `AnimParams` per graph, seeded with the declared
// defaults, and the graph starts in its initial state.
#[test]
fn graph_init_seeds_params_and_initial_state() {
    let mut world = graph_world();
    world.step();

    let params: Vec<&AnimParams> = world.query::<AnimParams>().collect();
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].target, hero());
    assert_eq!(params[0].values, vec![0.0]);

    let report = with_anim(&mut world, |anim| anim.graph_report(hero()).unwrap());
    assert_eq!(report.state, "idle");
    assert_eq!(report.params, vec![("speed".to_string(), 0.0)]);
    assert!(
        report.blend_weights.is_none(),
        "single-clip states report no blend weights"
    );
}

// A blendspace state's reported member weights track the parameter: parked
// on the first point at default, split across the bracketing pair mid-range.
#[test]
fn blendspace_weights_follow_the_parameter() {
    let target = intern("hero_blend");
    let mut world = World::new_empty();
    for (name, duration) in [("bl_idle", 1.0), ("bl_walk", 0.8), ("bl_run", 0.6)] {
        let mut a = clip(name, duration);
        a.target = Some(SkinnedMeshHandle(target.0));
        world.add_component(a);
    }
    let mut g: AnimGraph = serde_json::from_value(serde_json::json!({
        "target": "hero_blend",
        "parameters": [{"name": "speed", "default": 0.0}],
        "states": [
            {"name": "locomotion", "blend": {"kind": "blend1d", "parameter": "speed",
             "sync": true,
             "points": [
                 {"value": 0.0, "clip": "bl_idle"},
                 {"value": 1.6, "clip": "bl_walk"},
                 {"value": 5.0, "clip": "bl_run"}
             ]}}
        ]
    }))
    .unwrap();
    g.asset_id = intern("hero_blend_graph");
    world.add_component(g);
    world.start().unwrap();
    world.step();

    let report = with_anim(&mut world, |anim| anim.graph_report(target).unwrap());
    assert_eq!(report.state, "locomotion");
    assert_eq!(report.blend_weights, Some(vec![1.0, 0.0, 0.0]));

    // Half-way between the walk (1.6) and run (5.0) points.
    with_anim(&mut world, |anim| {
        anim.queue_param(target, "speed", 3.3).unwrap();
    });
    world.step();
    let w = with_anim(&mut world, |anim| anim.graph_report(target).unwrap())
        .blend_weights
        .unwrap();
    assert_eq!(w[0], 0.0);
    assert!(
        (w[1] - 0.5).abs() < 1e-4 && (w[2] - 0.5).abs() < 1e-4,
        "{w:?}"
    );
}

// Writing the `AnimParams` component (the gameplay surface) drives a
// transition on the next step; writing it back returns to the source state.
#[test]
fn graph_transitions_on_component_write() {
    let mut world = graph_world();
    world.step();

    for p in world.query_mut::<AnimParams>() {
        p.set(0, 2.0);
    }
    world.step();
    let report = with_anim(&mut world, |anim| anim.graph_report(hero()).unwrap());
    assert_eq!(report.state, "run");

    for p in world.query_mut::<AnimParams>() {
        p.set(0, 0.0);
    }
    world.step();
    let report = with_anim(&mut world, |anim| anim.graph_report(hero()).unwrap());
    assert_eq!(report.state, "idle");
}

// The `anim-param` path: a queued write lands in the component on the next
// step and the graph reacts to it; unknown names and flat targets fail.
#[test]
fn queued_param_writes_component_and_drives_graph() {
    let mut world = graph_world();
    world.step();

    with_anim(&mut world, |anim| {
        anim.queue_param(hero(), "speed", 3.0).unwrap();
        assert!(
            anim.queue_param(hero(), "nope", 1.0)
                .unwrap_err()
                .contains("no parameter"),
        );
    });
    world.step();

    let params: Vec<&AnimParams> = world.query::<AnimParams>().collect();
    assert_eq!(params[0].values, vec![3.0], "write landed in the component");
    let report = with_anim(&mut world, |anim| anim.graph_report(hero()).unwrap());
    assert_eq!(report.state, "run");
}

// Crossfade commands are the flat-bucket surface; a graph target must reject
// them (and point at anim-param), and a flat target must reject param
// commands (and point at anim-crossfade).
#[test]
fn mode_mismatched_commands_are_rejected() {
    let mut world = graph_world();
    world.step();
    with_anim(&mut world, |anim| {
        let err = anim
            .apply_crossfade(hero(), vec![1.0, 0.0], 0.0, 0.0)
            .unwrap_err();
        assert!(err.contains("graph-driven"));
        assert!(err.contains("anim-param"));
    });

    let mut flat_world = World::new_empty();
    let mut a = clip("solo_clip", 1.0);
    a.target = Some(SkinnedMeshHandle(intern("flat_hero").0));
    flat_world.add_component(a);
    flat_world.start().unwrap();
    flat_world.step();
    with_anim(&mut flat_world, |anim| {
        let target = intern("flat_hero");
        anim.apply_crossfade(target, vec![0.5], 0.0, 0.0).unwrap();
        let err = anim.queue_param(target, "speed", 1.0).unwrap_err();
        assert!(err.contains("anim-crossfade"));
        let err = anim.graph_report(target).unwrap_err();
        assert!(err.contains("no AnimGraph"));
    });
}

// A clip with a baked root track publishes per-frame `RootMotion` events
// carrying the mesh-local displacement; clips without one stay silent.
#[test]
fn root_motion_clip_publishes_displacement_events() {
    let target = intern("hero_rm");
    let mut world = World::new_empty();
    let mut a: Animation = serde_json::from_value(serde_json::json!({
        "target": "hero_rm",
        "duration": 1.0,
        "looping": true,
        "root_motion": true,
        "root_track": [
            {"time": 0.0, "translation": [0.0, 0.0, 0.0]},
            {"time": 1.0, "translation": [2.0, 0.0, 0.0]}
        ],
    }))
    .unwrap();
    a.asset_id = intern("hero_rm_walk");
    world.add_component(a);
    world.start().unwrap();

    world.step();
    std::thread::sleep(Duration::from_millis(5));
    world.step();

    let events = world
        .events::<crate::assets::RootMotion>()
        .expect("RootMotion queue exists");
    let mut cursor = crate::ecs::EventCursor::default();
    let motions = events.read(&mut cursor);
    assert!(!motions.is_empty(), "expected displacement events");
    let total: f32 = motions
        .iter()
        .filter(|m| m.target == target)
        .map(|m| m.delta[0])
        .sum();
    assert!(total > 0.0, "walk moves +X: {total}");
    assert!(
        motions
            .iter()
            .all(|m| m.delta[1] == 0.0 && m.delta[2] == 0.0)
    );
}

// The full rig chain: a skinned mesh with a capsule + a root-motion clip
// moves its CharacterRig through PhysicsSystem, staying grounded.
// Full IK chain: the graph authors an ik_chain on a three-joint leg whose
// foot hangs over a raised ledge (off to the side of the capsule, which
// stands on the flat floor). Probe rays go out, physics answers them, and
// the solve bends the leg so the foot rests on the ledge instead of
// clipping through it.
#[test]
fn ik_pins_the_foot_to_a_raised_ledge() {
    use crate::gfx::skinning::{Joint, JointPose, Skeleton};

    let target = intern("hero_ik");
    let mut world = World::new_empty();

    // A leg hanging from x = 0.6: hip at y = 2, knee at y = 1, foot at
    // y = 0 (bind). Named joints so the chain resolves.
    let joint = |name: &str, parent: Option<usize>, t: [f32; 3]| Joint {
        name: name.to_string(),
        parent,
        bind: JointPose {
            translation: t,
            ..JointPose::default()
        },
    };
    let skeleton = Skeleton::new(vec![
        joint("hip", None, [0.6, 2.0, 0.0]),
        joint("knee", Some(0), [0.0, -1.0, 0.0]),
        joint("foot", Some(1), [0.0, -1.0, 0.0]),
    ]);
    world.add_component(crate::assets::SkeletonPose::new(target, 0, skeleton));
    world.add_component(crate::assets::CharacterRig::new(
        target,
        0,
        crate::gfx::skinning::IDENTITY,
        0.5,
        0.3,
    ));

    // A constant clip (the bind pose) so the graph has something to play.
    let mut stand: Animation = serde_json::from_value(serde_json::json!({
        "target": "hero_ik",
        "duration": 1.0,
        "looping": true,
        "tracks": [{"joint": 0, "keyframes": [
            {"time": 0.0, "translation": [0.6, 2.0, 0.0]},
            {"time": 1.0, "translation": [0.6, 2.0, 0.0]}
        ]}],
    }))
    .unwrap();
    stand.asset_id = intern("hero_ik_stand");
    world.add_component(stand);

    let mut graph: AnimGraph = serde_json::from_value(serde_json::json!({
        "target": "hero_ik",
        "states": [{"name": "stand", "clip": "hero_ik_stand"}],
        "ik_chains": [{"joints": ["hip", "knee", "foot"], "pole": [0.0, 0.0, 1.0]}],
    }))
    .unwrap();
    graph.asset_id = intern("hero_ik_graph");
    world.add_component(graph);

    // Flat floor for the capsule; a ledge (top at y = 0.25) under the foot
    // only, clear of the capsule standing at the origin.
    world.add_component(crate::assets::PhysicsConfig::default());
    world.add_component(crate::assets::Prop {
        asset_id: intern("ledge"),
        position: [0.75, 0.1, 0.0],
        collider: Some(crate::assets::PropCollider {
            shape: "cuboid".to_string(),
            half_extents: [0.3, 0.15, 0.3],
            radius: 0.0,
            half_height: 0.0,
        }),
        ..Default::default()
    });
    world.start().unwrap();

    // Ray out -> physics answer -> solve; a few extra steps let the capsule
    // settle onto the floor.
    for _ in 0..8 {
        world.step();
        std::thread::sleep(Duration::from_millis(5));
    }

    let pose = world
        .query::<crate::assets::SkeletonPose>()
        .next()
        .expect("pose survives");
    let foot_mesh = {
        let m = pose.joint_matrices[2];
        let b = pose.skeleton.bind_position(2);
        [
            m[0][0] * b[0] + m[1][0] * b[1] + m[2][0] * b[2] + m[3][0],
            m[0][1] * b[0] + m[1][1] * b[1] + m[2][1] * b[2] + m[3][1],
            m[0][2] * b[0] + m[1][2] * b[1] + m[2][2] * b[2] + m[3][2],
        ]
    };
    let rig_y = world
        .query::<crate::assets::CharacterRig>()
        .next()
        .unwrap()
        .position[1];
    // The ledge top is at world 0.25; the foot's mesh-space height plus the
    // rig's world height must land there (the animated pose kept it at ~0).
    let foot_world_y = foot_mesh[1] + rig_y;
    assert!(
        (foot_world_y - 0.25).abs() < 0.03,
        "foot pinned to the ledge top: world y = {foot_world_y}"
    );
    // The foot stays put horizontally: pinning only lifts it.
    assert!((foot_mesh[0] - 0.6).abs() < 0.02, "{foot_mesh:?}");
}

#[test]
fn rig_capsule_follows_root_motion() {
    let target = intern("hero_rig");
    let mut world = World::new_empty();
    let mut a: Animation = serde_json::from_value(serde_json::json!({
        "target": "hero_rig",
        "duration": 1.0,
        "looping": true,
        "root_motion": true,
        "root_track": [
            {"time": 0.0, "translation": [0.0, 0.0, 0.0]},
            {"time": 1.0, "translation": [2.0, 0.0, 0.0]}
        ],
    }))
    .unwrap();
    a.asset_id = intern("hero_rig_walk");
    world.add_component(a);
    world.add_component(crate::assets::PhysicsConfig::default());
    // GraphicsSystem publishes rigs in a rendering world; this headless test
    // seeds one directly before start so PhysicsSystem::init sees it.
    world.add_component(crate::assets::CharacterRig::new(
        target,
        0,
        crate::gfx::skinning::IDENTITY,
        0.5,
        0.3,
    ));
    world.start().unwrap();

    for _ in 0..4 {
        world.step();
        std::thread::sleep(Duration::from_millis(5));
    }

    let rig = world
        .query::<crate::assets::CharacterRig>()
        .next()
        .expect("rig survives");
    assert!(
        rig.position[0] > 0.0,
        "capsule advanced along the walk: {:?}",
        rig.position
    );
    assert!(
        rig.moved,
        "render follow flag set (no GraphicsSystem to clear it)"
    );
    assert!(
        rig.position[1] > -0.2,
        "flat floor holds the capsule up: {:?}",
        rig.position
    );
}

// While a menu is open the animation step returns early, so graph clocks and
// transitions freeze with the pose.
#[test]
fn graph_freezes_while_menu_open() {
    let mut world = graph_world();
    world.step();

    world.insert_resource(crate::ecs::MenuActive(true));
    for p in world.query_mut::<AnimParams>() {
        p.set(0, 2.0);
    }
    world.step();
    let report = with_anim(&mut world, |anim| anim.graph_report(hero()).unwrap());
    assert_eq!(
        report.state, "idle",
        "paused step must not take transitions"
    );

    world.insert_resource(crate::ecs::MenuActive(false));
    world.step();
    let report = with_anim(&mut world, |anim| anim.graph_report(hero()).unwrap());
    assert_eq!(report.state, "run", "resumed step sees the parameter");
}

// A bare runtime clip of a given length, no tracks or root motion. Enough to
// re-seat a bucket slot via `apply_reloaded_clip`.
fn runtime_clip(duration: f32) -> crate::gfx::skinning::AnimationClip {
    crate::gfx::skinning::AnimationClip {
        duration,
        looping: true,
        tracks: Vec::new(),
        root: None,
    }
}

// A single flat clip targeting `target`, weight 1.
fn flat_clip(name: &str, target: AssetId) -> Animation {
    let mut a = clip(name, 1.0);
    a.target = Some(SkinnedMeshHandle(target.0));
    a
}

// A flat bucket accepts a re-imported clip into a valid slot and refuses a bad
// target or an out-of-range slot without mutating anything.
#[test]
fn apply_reloaded_clip_reseats_a_flat_slot_and_rejects_bad_targets() {
    let target = intern("flat_reload");
    let mut world = World::new_empty();
    world.add_component(flat_clip("fr_solo", target));
    world.start().unwrap();
    world.step();

    with_anim(&mut world, |anim| {
        assert!(
            anim.apply_reloaded_clip(target, 0, runtime_clip(3.0), 0.5),
            "a valid slot accepts the reload"
        );
        assert!(
            !anim.apply_reloaded_clip(intern("nobody"), 0, runtime_clip(1.0), 1.0),
            "an unknown target is refused"
        );
        assert!(
            !anim.apply_reloaded_clip(target, 9, runtime_clip(1.0), 1.0),
            "an out-of-range slot is refused"
        );
    });
}

// Reloading a clip a graph plays refreshes the state machine's compiled
// duration for that slot; the graph keeps running afterward.
#[test]
fn apply_reloaded_clip_refreshes_graph_clip_duration() {
    let mut world = graph_world();
    world.step();

    let applied = with_anim(&mut world, |anim| {
        anim.apply_reloaded_clip(hero(), 0, runtime_clip(2.5), 1.0)
    });
    assert!(applied, "the graph bucket's idle slot reloads");
    // The refresh path ran without disturbing the running graph.
    world.step();
    let report = with_anim(&mut world, |anim| anim.graph_report(hero()).unwrap());
    assert_eq!(report.state, "idle");
}

// With hot-reload off (no `cn debug`) and inline clips, the reload catalogue
// is empty: the getter returns an empty slice.
#[test]
fn reload_entries_is_empty_without_captured_sources() {
    let mut world = graph_world();
    world.step();
    let count = with_anim(&mut world, |anim| anim.reload_entries().len());
    assert_eq!(count, 0);
}

// The Debug impl summarizes the bucket and reload-catalogue counts rather than
// dumping their contents.
#[test]
fn debug_impl_summarizes_target_and_reload_counts() {
    let mut world = graph_world();
    world.step();
    let text = with_anim(&mut world, |anim| format!("{anim:?}"));
    assert!(text.contains("AnimationSystem"), "{text}");
    assert!(text.contains("targets: 1"), "{text}");
    assert!(text.contains("reload_entries: 0"), "{text}");
}

// A one-joint pose for `target`, used to observe the flat sampling arms.
fn single_joint_pose(target: AssetId) -> crate::assets::SkeletonPose {
    use crate::gfx::skinning::{Joint, JointPose, Skeleton};
    let skeleton = Skeleton::new(vec![Joint {
        name: "root".to_string(),
        parent: None,
        bind: JointPose::default(),
    }]);
    crate::assets::SkeletonPose::new(target, 0, skeleton)
}

// One flat clip drives the single-clip sampling arm: the pose gets one skinning
// matrix per joint, at full strength regardless of the clip's weight.
#[test]
fn flat_single_clip_samples_the_pose() {
    let target = intern("flat_single_pose");
    let mut world = World::new_empty();
    world.add_component(flat_clip("fs_solo", target));
    world.add_component(single_joint_pose(target));
    world.start().unwrap();
    world.step();

    let matrices = world
        .query::<crate::assets::SkeletonPose>()
        .next()
        .map(|p| p.joint_matrices.len())
        .unwrap();
    assert_eq!(matrices, 1, "one skinning matrix for the one joint");
}

// Two flat clips with a fade-in drive the startup ramp and the multi-clip
// weighted-blend arm: the fade transition anchors and advances on the first
// steps and the blended pose is written every frame.
#[test]
fn flat_fade_in_blends_multiple_clips_into_the_pose() {
    let target = intern("flat_blend_pose");
    let mut world = World::new_empty();
    // One clip requests a fade-in, so init builds a startup weight ramp.
    let mut faded = flat_clip("fb_a", target);
    faded.fade_in_secs = 0.5;
    world.add_component(faded);
    world.add_component(flat_clip("fb_b", target));
    world.add_component(single_joint_pose(target));
    world.start().unwrap();
    // First step anchors the fade ramp; second advances it further. Both run
    // the multi-clip blend arm and write the pose.
    world.step();
    world.step();

    let matrices = world
        .query::<crate::assets::SkeletonPose>()
        .next()
        .map(|p| p.joint_matrices.len())
        .unwrap();
    assert_eq!(matrices, 1, "the weighted blend produced a pose");
}

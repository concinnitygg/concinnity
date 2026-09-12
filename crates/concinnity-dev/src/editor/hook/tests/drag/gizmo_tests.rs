// src/editor/hook/tests/drag/gizmo_tests.rs
//
// The transform manipulator's drags (`hook/drag/gizmo.rs`): translate, rotate
// and scale over a single prop, the grid and angle snapping each applies and
// what Ctrl suspends, the same over a multi-member selection about its
// centroid, the mode keys, and a skinned mesh's position surviving the round
// trip. Each asserts one undo step per drag.

use concinnity_core::components::FrameInput;
use concinnity_core::components::InputKey;
use concinnity_core::components::Transform;
use concinnity_core::ecs::Entity;
use concinnity_core::ecs::World;
use concinnity_host::thread::asset_id;

use crate::debug_hook::DebugHook;

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::{
    SIDE_A, SIDE_B, click_at, click_at_mod, drag_input, drag_to, hook, pick_world, release_at,
    set_input, two_prop_rig, world_with_input,
};

use crate::editor::viewport::gizmo;
use crate::editor::viewport::snap;

// The full translate-gizmo loop: pick a prop, grab its X tip handle, drag
// right, release. The live Transform follows during the drag; release commits
// the moved position to the authored entry as ONE undo step, and Ctrl-class
// undo restores the original position.
#[test]
fn gizmo_drag_moves_the_prop_and_commits_one_undo_step() {
    asset_id::reset_interner();
    let id = asset_id::intern("box_near");
    // Down-left of the camera axis so the pick, the handles, and the drag all
    // land in screen regions no default panel covers.
    let start = [-6.11f32, -3.3, -5.0];
    let mut world = pick_world(
        [0.0; 3],
        vec![(
            id,
            [start[0] - 1.0, start[1] - 1.0, start[2] - 1.0],
            [start[0] + 1.0, start[1] + 1.0, start[2] + 1.0],
        )],
    );
    let entity = world.push(Transform {
        position: start,
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    });
    let mut by_name = std::collections::BTreeMap::new();
    by_name.insert(id, entity);
    world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
    for s in gizmo::sprites() {
        world.add_component(s);
    }

    let mut h = hook(vec![serde_json::json!({
        "name": "box_near", "type": "Prop", "args": { "position": start }
    })]);

    // Pick the prop (projects to ~[200, 600] for this camera).
    click_at(&mut world, &mut h, [200.0, 600.0]);
    assert_eq!(h.selection.active(), Some("box_near"));
    let layout = h
        .gizmo_layout(&world, [1280.0, 720.0])
        .expect("movable selection shows the gizmo");

    // Press the X tip handle: a drag starts, nothing re-picks.
    click_at(&mut world, &mut h, layout.tips[0]);
    assert!(h.gizmo_drag.is_some(), "the tip press starts a drag");
    assert!(!h.dirty, "no entry change until release");

    // Drag 50 px right: the live Transform follows along world X.
    set_input(
        &mut world,
        FrameInput {
            viewport: [1280.0, 720.0],
            mouse_x: layout.tips[0][0] + 50.0,
            mouse_y: layout.tips[0][1],
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    let live = world
        .get::<Transform>(entity)
        .expect("entity alive")
        .position;
    assert!(live[0] > start[0] + 0.3, "moved right: {}", live[0]);
    assert!((live[1] - start[1]).abs() < 1e-3, "Y untouched");
    assert!((live[2] - start[2]).abs() < 1e-3, "Z untouched");

    // Release: the entry commits, one undo step, form refreshed.
    set_input(
        &mut world,
        FrameInput {
            viewport: [1280.0, 720.0],
            mouse_x: layout.tips[0][0] + 50.0,
            mouse_y: layout.tips[0][1],
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert!(h.gizmo_drag.is_none(), "release ends the drag");
    assert!(h.dirty, "the move is an unsaved edit");
    let committed = h.entries[0]["args"]["position"][0].as_f64().unwrap();
    assert!(
        committed > f64::from(start[0]) + 0.3,
        "entry follows: {committed}"
    );

    // One undo step restores the pre-drag position.
    h.undo(&mut world);
    let restored = h.entries[0]["args"]["position"][0].as_f64().unwrap();
    assert!((restored - f64::from(start[0])).abs() < 1e-3, "{restored}");
    assert!(!h.can_undo(), "the whole drag was one step");
}

// Shared rig for the rotate / scale drag tests: a prop down-left of the
// camera axis (panel-free screen region), its live Transform entity, and the
// EntityByName map the gizmo resolves through.
fn gizmo_rig(start: [f32; 3]) -> (World, Entity, EditorHook) {
    asset_id::reset_interner();
    let id = asset_id::intern("box_near");
    let mut world = pick_world(
        [0.0; 3],
        vec![(
            id,
            [start[0] - 1.0, start[1] - 1.0, start[2] - 1.0],
            [start[0] + 1.0, start[1] + 1.0, start[2] + 1.0],
        )],
    );
    let entity = world.push(Transform {
        position: start,
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    });
    let mut by_name = std::collections::BTreeMap::new();
    by_name.insert(id, entity);
    world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
    for s in gizmo::sprites() {
        world.add_component(s);
    }
    let h = hook(vec![serde_json::json!({
        "name": "box_near", "type": "Prop", "args": { "position": start }
    })]);
    (world, entity, h)
}

// Rotate mode: turning the mouse a quarter circle around the origin rotates
// the prop 90 degrees about the grabbed axis, committed as one undo step.
#[test]
fn gizmo_rotate_drag_turns_the_prop() {
    let start = [-6.11f32, -3.3, -5.0];
    let (mut world, entity, mut h) = gizmo_rig(start);
    h.gizmo_mode = gizmo::GizmoMode::Rotate;

    click_at(&mut world, &mut h, [200.0, 600.0]);
    let layout = h.gizmo_layout(&world, [1280.0, 720.0]).expect("gizmo up");
    // Grab the X tip (70 px right of the origin: screen angle 0)...
    click_at(&mut world, &mut h, layout.tips[0]);
    assert!(h.gizmo_drag.is_some(), "rotate grab starts on the tip");
    // ...and swing the cursor to straight below the origin: +90 degrees of
    // screen angle. World X is perpendicular to the view (sign +1).
    set_input(
        &mut world,
        drag_input([layout.origin[0], layout.origin[1] + 70.0], true),
    );
    h.tick(&mut world);
    let live = world.get::<Transform>(entity).unwrap();
    assert!(
        (live.rotation_deg[0] - 90.0).abs() < 1.0,
        "quarter turn about X: {}",
        live.rotation_deg[0]
    );
    assert_eq!(live.position, start, "rotate leaves position alone");

    // Release commits rotation_deg; undo removes it again.
    set_input(
        &mut world,
        drag_input([layout.origin[0], layout.origin[1] + 70.0], false),
    );
    h.tick(&mut world);
    let committed = h.entries[0]["args"]["rotation_deg"][0].as_f64().unwrap();
    assert!((committed - 90.0).abs() < 1.0, "{committed}");
    h.undo(&mut world);
    assert!(
        h.entries[0]["args"].get("rotation_deg").is_none(),
        "one undo step restores the pre-drag entry"
    );
}

// Scale mode: dragging the X tip half its run further out scales X by ~1.5,
// leaving the other axes untouched.
#[test]
fn gizmo_scale_drag_stretches_one_axis() {
    let start = [-6.11f32, -3.3, -5.0];
    let (mut world, entity, mut h) = gizmo_rig(start);
    h.gizmo_mode = gizmo::GizmoMode::Scale;

    click_at(&mut world, &mut h, [200.0, 600.0]);
    let layout = h.gizmo_layout(&world, [1280.0, 720.0]).expect("gizmo up");
    click_at(&mut world, &mut h, layout.tips[0]);
    assert!(h.gizmo_drag.is_some());
    set_input(
        &mut world,
        drag_input([layout.tips[0][0] + 35.0, layout.tips[0][1]], true),
    );
    h.tick(&mut world);
    let live = world.get::<Transform>(entity).unwrap();
    assert!(
        live.scale[0] > 1.3 && live.scale[0] < 1.7,
        "X stretched ~1.5x: {}",
        live.scale[0]
    );
    assert_eq!(live.scale[1], 1.0, "Y untouched");
    assert_eq!(live.position, start, "scale leaves position alone");

    set_input(
        &mut world,
        drag_input([layout.tips[0][0] + 35.0, layout.tips[0][1]], false),
    );
    h.tick(&mut world);
    let committed = h.entries[0]["args"]["scale"][0].as_f64().unwrap();
    assert!(committed > 1.3 && committed < 1.7, "{committed}");
    h.undo(&mut world);
    assert!(h.entries[0]["args"].get("scale").is_none());
}

// How far `v` sits from the nearest multiple of `step`, in step units
// (transform math accumulates float error, so grid checks need a tolerance).
fn off_grid(v: f32, step: f32) -> f32 {
    (v / step - (v / step).round()).abs()
}

// With move snapping on, a translate drag lands on grid multiples of the step;
// holding Ctrl suspends the snap for the frame; release commits the snapped
// position.
#[test]
fn gizmo_translate_drag_snaps_to_the_grid_and_ctrl_suspends_it() {
    let start = [-6.11f32, -3.3, -5.0];
    let (mut world, entity, mut h) = gizmo_rig(start);
    h.snap.translate = snap::Snap {
        enabled: true,
        step: 0.25,
    };

    click_at(&mut world, &mut h, [200.0, 600.0]);
    let layout = h.gizmo_layout(&world, [1280.0, 720.0]).expect("gizmo up");
    click_at(&mut world, &mut h, layout.tips[0]);
    assert!(h.gizmo_drag.is_some());

    let target = [layout.tips[0][0] + 50.0, layout.tips[0][1]];
    set_input(&mut world, drag_input(target, true));
    h.tick(&mut world);
    let delta = world.get::<Transform>(entity).unwrap().position[0] - start[0];
    assert!(delta > 0.2, "moved right: {delta}");
    assert!(off_grid(delta, 0.25) < 1e-4, "on the grid: {delta}");

    let mut ctrl = drag_input(target, true);
    ctrl.ctrl = true;
    set_input(&mut world, ctrl);
    h.tick(&mut world);
    let free = world.get::<Transform>(entity).unwrap().position[0] - start[0];
    assert!(free > 0.2, "still follows the cursor: {free}");
    assert!(
        off_grid(free, 0.25) > 1e-3,
        "unsnapped while Ctrl is held: {free}"
    );

    // Releasing Ctrl re-snaps the preview; the commit is what is shown.
    set_input(&mut world, drag_input(target, true));
    h.tick(&mut world);
    set_input(&mut world, drag_input(target, false));
    h.tick(&mut world);
    let committed = h.entries[0]["args"]["position"][0].as_f64().unwrap() as f32 - start[0];
    assert!(
        off_grid(committed, 0.25) < 1e-3,
        "the commit is on the grid: {committed}"
    );
}

// With rotate snapping on, the applied angle rounds to the step: a ~60 degree
// swing lands on 45.
#[test]
fn gizmo_rotate_drag_snaps_the_applied_angle() {
    let start = [-6.11f32, -3.3, -5.0];
    let (mut world, entity, mut h) = gizmo_rig(start);
    h.gizmo_mode = gizmo::GizmoMode::Rotate;
    h.snap.rotate = snap::Snap {
        enabled: true,
        step: 45.0,
    };

    click_at(&mut world, &mut h, [200.0, 600.0]);
    let layout = h.gizmo_layout(&world, [1280.0, 720.0]).expect("gizmo up");
    click_at(&mut world, &mut h, layout.tips[0]);
    assert!(h.gizmo_drag.is_some());

    // Swing from the X tip (screen angle 0) to ~60 degrees of screen angle.
    let m = [layout.origin[0] + 35.0, layout.origin[1] + 60.6];
    set_input(&mut world, drag_input(m, true));
    h.tick(&mut world);
    let live = world.get::<Transform>(entity).unwrap();
    assert!(
        (live.rotation_deg[0] - 45.0).abs() < 1e-3,
        "snapped to 45: {}",
        live.rotation_deg[0]
    );

    set_input(&mut world, drag_input(m, false));
    h.tick(&mut world);
    let committed = h.entries[0]["args"]["rotation_deg"][0].as_f64().unwrap();
    assert!((committed - 45.0).abs() < 0.2, "{committed}");
}

// The gizmo anchors at the selection centroid and a translate drag moves
// every member by the same world delta, committed as ONE undo step.
#[test]
fn multi_translate_moves_all_members_as_one_undo_step() {
    let (mut world, e1, e2, mut h) = two_prop_rig(SIDE_A, SIDE_B, 1.0);
    click_at(&mut world, &mut h, [200.0, 600.0]);
    click_at_mod(&mut world, &mut h, [424.0, 598.0], true);

    let layout = h
        .gizmo_layout(&world, [1280.0, 720.0])
        .expect("multi selection shows the gizmo");
    // The anchor is the centroid: between the two boxes, ~[312, 598].
    assert!(
        (layout.origin[0] - 312.0).abs() < 3.0 && (layout.origin[1] - 598.0).abs() < 3.0,
        "centroid anchor, got {:?}",
        layout.origin
    );

    click_at(&mut world, &mut h, layout.tips[0]);
    assert!(h.gizmo_drag.is_some(), "the tip press starts a drag");
    drag_to(
        &mut world,
        &mut h,
        [layout.tips[0][0] + 50.0, layout.tips[0][1]],
    );
    let p1 = world.get::<Transform>(e1).unwrap().position;
    let p2 = world.get::<Transform>(e2).unwrap().position;
    let (d1, d2) = (p1[0] - SIDE_A[0], p2[0] - SIDE_B[0]);
    assert!(d1 > 0.3, "box_a moved right: {d1}");
    assert!((d1 - d2).abs() < 1e-3, "one shared delta: {d1} vs {d2}");
    assert_eq!(p1[1], SIDE_A[1], "Y untouched");

    release_at(
        &mut world,
        &mut h,
        [layout.tips[0][0] + 50.0, layout.tips[0][1]],
    );
    assert!(h.dirty, "the move is an unsaved edit");
    for (i, s) in [(0, SIDE_A), (1, SIDE_B)] {
        let committed = h.entries[i]["args"]["position"][0].as_f64().unwrap();
        assert!(
            committed > f64::from(s[0]) + 0.3,
            "entry {i} follows: {committed}"
        );
    }

    h.undo(&mut world);
    for (i, s) in [(0, SIDE_A), (1, SIDE_B)] {
        let restored = h.entries[i]["args"]["position"][0].as_f64().unwrap();
        assert!(
            (restored - f64::from(s[0])).abs() < 1e-3,
            "entry {i}: {restored}"
        );
    }
    assert!(!h.can_undo(), "the whole multi-drag was one step");
}

// Rotate orbits member positions about the centroid (and spins each member),
// the group behavior of every mainstream editor; both writes land in the
// entries and one undo restores them.
#[test]
fn multi_rotate_orbits_members_about_the_centroid() {
    // Stacked vertically: box_a above the centroid, box_b below, so an X-axis
    // turn swings them through Z.
    let s1 = [-6.11, -2.3, -5.0];
    let s2 = [-6.11, -4.3, -5.0];
    let (mut world, e1, e2, mut h) = two_prop_rig(s1, s2, 0.8);
    h.gizmo_mode = gizmo::GizmoMode::Rotate;
    click_at(&mut world, &mut h, [200.0, 526.0]);
    assert_eq!(h.selection.active(), Some("box_a"));
    click_at_mod(&mut world, &mut h, [200.0, 670.0], true);
    assert_eq!(h.selection.iter().count(), 2);

    let layout = h.gizmo_layout(&world, [1280.0, 720.0]).expect("gizmo up");
    // Grab the X tip and swing a quarter circle to straight below the origin:
    // +90 degrees about world X.
    click_at(&mut world, &mut h, layout.tips[0]);
    assert!(h.gizmo_drag.is_some(), "rotate grab starts on the tip");
    drag_to(
        &mut world,
        &mut h,
        [layout.origin[0], layout.origin[1] + 70.0],
    );

    let t1 = *world.get::<Transform>(e1).unwrap();
    let t2 = *world.get::<Transform>(e2).unwrap();
    assert!(
        (t1.rotation_deg[0] - 90.0).abs() < 1.0 && (t2.rotation_deg[0] - 90.0).abs() < 1.0,
        "both spin: {} / {}",
        t1.rotation_deg[0],
        t2.rotation_deg[0]
    );
    // +90 about X through the centroid [-6.11, -3.3, -5.0] carries the +Y
    // offset into +Z: box_a to z = -4, box_b to z = -6, both onto y = -3.3.
    assert!(
        (t1.position[1] + 3.3).abs() < 0.1 && (t1.position[2] + 4.0).abs() < 0.1,
        "box_a orbits: {:?}",
        t1.position
    );
    assert!(
        (t2.position[1] + 3.3).abs() < 0.1 && (t2.position[2] + 6.0).abs() < 0.1,
        "box_b orbits: {:?}",
        t2.position
    );

    release_at(
        &mut world,
        &mut h,
        [layout.origin[0], layout.origin[1] + 70.0],
    );
    let committed = h.entries[0]["args"]["rotation_deg"][0].as_f64().unwrap();
    assert!((committed - 90.0).abs() < 1.0, "{committed}");
    let z = h.entries[0]["args"]["position"][2].as_f64().unwrap();
    assert!((z + 4.0).abs() < 0.1, "the orbited position commits: {z}");

    h.undo(&mut world);
    for i in [0, 1] {
        assert!(
            h.entries[i]["args"].get("rotation_deg").is_none(),
            "one undo step restores entry {i}"
        );
    }
    let z = h.entries[0]["args"]["position"][2].as_f64().unwrap();
    assert!((z + 5.0).abs() < 1e-3, "position restored too: {z}");
    assert!(!h.can_undo(), "the whole multi-drag was one step");
}

// T / R / S switch the gizmo mode in edit mode, but never while typing.
#[test]
fn gizmo_mode_keys_switch_unless_typing() {
    let mut h = hook(Vec::new());
    let key = |h: &mut EditorHook, k: InputKey| {
        let mut world = world_with_input(FrameInput {
            viewport: [1280.0, 720.0],
            captured_key: Some(k),
            ..Default::default()
        });
        h.tick(&mut world);
    };
    key(&mut h, InputKey::R);
    assert_eq!(h.gizmo_mode, gizmo::GizmoMode::Rotate);
    key(&mut h, InputKey::S);
    assert_eq!(h.gizmo_mode, gizmo::GizmoMode::Scale);
    key(&mut h, InputKey::T);
    assert_eq!(h.gizmo_mode, gizmo::GizmoMode::Translate);

    // Shift+F toggles the fly camera through the same guard; plain F frames
    // the selection (a no-op with nothing selected) and leaves fly alone.
    let shift_f = |h: &mut EditorHook| {
        let mut world = world_with_input(FrameInput {
            viewport: [1280.0, 720.0],
            captured_key: Some(InputKey::F),
            shift: true,
            ..Default::default()
        });
        h.tick(&mut world);
    };
    key(&mut h, InputKey::F);
    assert!(!h.fly, "plain F frames instead of flying");
    shift_f(&mut h);
    assert!(h.fly, "Shift+F starts the fly camera");
    shift_f(&mut h);
    assert!(!h.fly, "Shift+F again stops it");

    // A focused text field keeps the keys for typing.
    h.story_focus = true;
    key(&mut h, InputKey::R);
    assert_eq!(h.gizmo_mode, gizmo::GizmoMode::Translate);
    shift_f(&mut h);
    assert!(!h.fly, "typing keeps F");
}

// A skinned mesh is movable like a prop once the engine indexes it: the pick
// selects it, the gizmo drag moves its live Transform, and the release
// commits the SkinnedMesh entry's position as one undo step.
#[test]
fn gizmo_drag_moves_a_skinned_mesh_and_commits_its_position() {
    asset_id::reset_interner();
    let id = asset_id::intern("body");
    let start = [-6.11f32, -3.3, -5.0];
    let mut world = pick_world(
        [0.0; 3],
        vec![(
            id,
            [start[0] - 1.0, start[1] - 1.0, start[2] - 1.0],
            [start[0] + 1.0, start[1] + 1.0, start[2] + 1.0],
        )],
    );
    let entity = world.push(Transform {
        position: start,
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    });
    let mut by_name = std::collections::BTreeMap::new();
    by_name.insert(id, entity);
    world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
    for s in gizmo::sprites() {
        world.add_component(s);
    }
    let mut h = hook(vec![serde_json::json!({
        "name": "body", "type": "SkinnedMesh",
        "args": { "source": "hero.glb", "position": start }
    })]);

    click_at(&mut world, &mut h, [200.0, 600.0]);
    assert_eq!(h.selection.active(), Some("body"));
    assert!(!h.form_open(), "selecting the body opens no form");
    let layout = h
        .gizmo_layout(&world, [1280.0, 720.0])
        .expect("a skinned mesh shows the gizmo");
    click_at(&mut world, &mut h, layout.tips[0]);
    assert!(h.gizmo_drag.is_some());
    drag_to(
        &mut world,
        &mut h,
        [layout.tips[0][0] + 50.0, layout.tips[0][1]],
    );
    let live = world.get::<Transform>(entity).unwrap().position;
    assert!(
        live[0] > start[0] + 0.3,
        "the live transform follows: {live:?}"
    );
    release_at(
        &mut world,
        &mut h,
        [layout.tips[0][0] + 50.0, layout.tips[0][1]],
    );
    let committed = h.entries[0]["args"]["position"][0].as_f64().unwrap();
    assert!(committed > f64::from(start[0]) + 0.3, "{committed}");
    h.undo(&mut world);
    assert!(!h.can_undo(), "one step");
}

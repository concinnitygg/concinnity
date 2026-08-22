// src/editor/hook/camera_tests.rs
//
// The editor's camera navigation drives: the Alt+drag tumble
// (`hook/orbit_drive.rs`), the eased glide that F-framing and a bookmark recall
// ride on (`hook/glide_drive.rs`), and the numbered pose slots
// (`hook/bookmarks.rs`). The pose math each of these calls is pure and tested
// beside it (`editor/orbit.rs`, `editor/framing.rs`); asserted here is what the
// drives own -- when a drag or glide starts, what it writes to the live camera,
// and what hands control back.

use super::*;
use crate::assets::{Camera3D, InputKey};
use crate::test_support::isolate_state_dir;

const VP: [f32; 2] = [1280.0, 720.0];

fn hook() -> EditorHook {
    EditorHook::new("unused.jsonl".to_string(), Vec::new())
}

// A world with a camera at `pos` facing -Z, and a PickIndex holding one unit
// box at the origin under the interned name "box".
fn camera_world(pos: [f32; 3]) -> (World, AssetId) {
    crate::ecs::asset_id::reset_interner();
    let id = crate::ecs::asset_id::intern("box");
    let mut world = World::new();
    world.add_component(Camera3D {
        position: pos,
        view_matrix: concinnity_core::gfx::camera::view_matrix(pos, 0.0, 0.0),
        fov_y_degrees: 90.0,
        near: 0.05,
        far: 200.0,
        yaw: 0.0,
        pitch: 0.0,
        desired_move: [0.0; 3],
        jump_requested: false,
        interact_requested: false,
        controller: None,
    });
    world.insert_resource(crate::ecs::PickIndex {
        entries: vec![crate::ecs::PickEntry {
            asset_id: id,
            bb_min: [-1.0, -1.0, -1.0],
            bb_max: [1.0, 1.0, 1.0],
        }],
    });
    (world, id)
}

fn camera(world: &World) -> &Camera3D {
    world.query::<Camera3D>().next().expect("camera")
}

// Run an armed glide to its end without waiting out its quarter second: rewind
// the start instant past the duration, then take the step that retires it.
fn finish_glide(h: &mut EditorHook, world: &mut World) {
    let glide = h.glide.as_mut().expect("a glide is armed");
    glide.start -= std::time::Duration::from_secs(1);
    h.drive_glide(
        &FrameInput {
            viewport: VP,
            ..Default::default()
        },
        world,
    );
}

// A press over the viewport (clear of the top bar) with Alt held.
fn alt_press(mouse: [f32; 2]) -> FrameInput {
    FrameInput {
        viewport: VP,
        mouse_x: mouse[0],
        mouse_y: mouse[1],
        alt: true,
        left_button_down: true,
        ..Default::default()
    }
}

fn drag_to(mouse: [f32; 2]) -> FrameInput {
    FrameInput {
        viewport: VP,
        mouse_x: mouse[0],
        mouse_y: mouse[1],
        left_button_down: true,
        ..Default::default()
    }
}

// An Alt+press over the viewport with something selected starts a tumble; the
// drag then circles the camera around the selection without moving the pivot.
#[test]
fn alt_drag_tumbles_the_camera_around_the_selection() {
    let (mut world, _) = camera_world([0.0, 0.0, 10.0]);
    let mut h = hook();
    h.selection.set(vec!["box".to_string()]);

    assert!(h.try_begin_orbit(&alt_press([600.0, 400.0]), VP, &world));
    assert!(h.orbit.is_some(), "the tumble is armed");

    let before = camera(&world).position;
    h.drive_orbit(&drag_to([700.0, 400.0]), &mut world);
    let after = camera(&world).position;

    assert_ne!(before, after, "the drag moved the camera");
    // The pivot is the selection's bounds center (the origin here), so the
    // orbit radius is preserved as the camera circles it.
    let radius = |p: [f32; 3]| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    assert!(
        (radius(before) - radius(after)).abs() < 0.01,
        "tumbling kept the distance to the pivot: {before:?} -> {after:?}"
    );
    // The view matrix is rewritten alongside the pose, never left stale.
    assert_eq!(
        camera(&world).view_matrix,
        concinnity_core::gfx::camera::view_matrix(after, camera(&world).yaw, camera(&world).pitch)
    );
}

// Releasing the button ends the tumble, so the next press starts a fresh one.
#[test]
fn releasing_the_button_ends_the_tumble() {
    let (mut world, _) = camera_world([0.0, 0.0, 10.0]);
    let mut h = hook();
    h.selection.set(vec!["box".to_string()]);
    assert!(h.try_begin_orbit(&alt_press([600.0, 400.0]), VP, &world));

    let released = FrameInput {
        viewport: VP,
        mouse_x: 700.0,
        mouse_y: 400.0,
        ..Default::default()
    };
    h.drive_orbit(&released, &mut world);
    assert!(h.orbit.is_none(), "the release disarmed the tumble");
}

// The tumble needs a pivot and a live viewport press: nothing selected, a press
// on the top bar, or a running simulation all decline it so the press routes on.
#[test]
fn a_tumble_is_declined_without_a_viewport_press_and_a_pivot() {
    let (world, _) = camera_world([0.0, 0.0, 10.0]);

    let mut empty = hook();
    assert!(
        !empty.try_begin_orbit(&alt_press([600.0, 400.0]), VP, &world),
        "nothing selected"
    );
    assert!(empty.orbit.is_none());

    let mut h = hook();
    h.selection.set(vec!["box".to_string()]);
    assert!(
        !h.try_begin_orbit(&alt_press([600.0, hud::BAR_H - 1.0]), VP, &world),
        "a press on the top bar is the bar's"
    );

    let mut playing = hook();
    playing.selection.set(vec!["box".to_string()]);
    playing.sim_toggle_play();
    assert!(
        !playing.try_begin_orbit(&alt_press([600.0, 400.0]), VP, &world),
        "play mode owns the camera"
    );
}

// A camera sitting exactly on the pivot has no orbit radius to tumble around.
#[test]
fn a_camera_on_the_pivot_declines_the_tumble() {
    let (world, _) = camera_world([0.0, 0.0, 0.0]);
    let mut h = hook();
    h.selection.set(vec!["box".to_string()]);
    assert!(!h.try_begin_orbit(&alt_press([600.0, 400.0]), VP, &world));
}

// F frames the selection: the glide is armed, and driving it to completion
// leaves the camera looking the same way from a distance that fits the bounds.
#[test]
fn framing_the_selection_glides_the_camera_to_fit_it() {
    let (mut world, _) = camera_world([0.0, 0.0, 100.0]);
    let mut h = hook();
    h.selection.set(vec!["box".to_string()]);
    let input = FrameInput {
        viewport: VP,
        ..Default::default()
    };

    h.frame_selection(input.viewport, &world);
    assert!(h.glide.is_some(), "F armed a glide");

    // A step short of the duration moves the camera without ending the glide.
    h.drive_glide(&input, &mut world);
    assert!(h.glide.is_some(), "the glide is still in flight");

    finish_glide(&mut h, &mut world);
    assert!(h.glide.is_none(), "the glide finished and cleared itself");

    let cam = camera(&world);
    assert_eq!((cam.yaw, cam.pitch), (0.0, 0.0), "the look direction held");
    let dist = (cam.position[2]).abs();
    assert!(
        dist < 100.0 && dist > 1.0,
        "the camera closed on the box without landing inside it, at z={dist}"
    );
}

// F with nothing selected has no bounds to fit, so no glide starts.
#[test]
fn framing_nothing_starts_no_glide() {
    let (world, _) = camera_world([0.0, 0.0, 10.0]);
    let mut h = hook();
    h.frame_selection(VP, &world);
    assert!(h.glide.is_none());
}

// Any deliberate navigation input hands control straight back, so a glide can
// never fight the user for the camera.
#[test]
fn steering_during_a_glide_hands_the_camera_back() {
    for steer in [
        FrameInput {
            forward: true,
            ..Default::default()
        },
        FrameInput {
            backward: true,
            ..Default::default()
        },
        FrameInput {
            left: true,
            ..Default::default()
        },
        FrameInput {
            right: true,
            ..Default::default()
        },
        FrameInput {
            mouse_dx: 3.0,
            ..Default::default()
        },
        FrameInput {
            mouse_dy: -3.0,
            ..Default::default()
        },
    ] {
        let (mut world, _) = camera_world([0.0, 0.0, 100.0]);
        let mut h = hook();
        h.selection.set(vec!["box".to_string()]);
        h.frame_selection(VP, &world);
        assert!(h.glide.is_some());

        h.drive_glide(&steer, &mut world);
        assert!(h.glide.is_none(), "{steer:?} did not end the glide");
        // The fly clock is dropped with it, so a fly re-entry cannot integrate
        // the glide's wall time as one step.
        assert!(h.fly_clock.is_none());
    }
}

// Only the digit keys address a bookmark slot, and they map in order.
#[test]
fn digit_keys_map_to_bookmark_slots() {
    let digits = [
        InputKey::Num1,
        InputKey::Num2,
        InputKey::Num3,
        InputKey::Num4,
        InputKey::Num5,
        InputKey::Num6,
        InputKey::Num7,
        InputKey::Num8,
        InputKey::Num9,
    ];
    for (i, key) in digits.into_iter().enumerate() {
        assert_eq!(bookmarks::slot_for(key), Some(i));
    }
    for key in [InputKey::A, InputKey::Num0, InputKey::Space] {
        assert_eq!(bookmarks::slot_for(key), None);
    }
}

// Ctrl+digit stores the live camera pose in a slot; the bare digit glides back
// to it from wherever the camera has since moved.
#[test]
fn a_saved_bookmark_recalls_the_pose_it_captured() {
    isolate_state_dir();
    let _guard = crate::test_support::lock();
    let (mut world, _) = camera_world([1.0, 2.0, 3.0]);
    let mut h = hook();

    h.save_bookmark(0, &world);
    assert!(h.bookmarks[0].is_some(), "the slot captured a pose");

    // Move the camera away, then recall.
    if let Some(cam) = world.query_mut::<Camera3D>().next() {
        cam.position = [50.0, 50.0, 50.0];
    }
    h.recall_bookmark(0, &world);
    assert!(h.glide.is_some(), "the recall armed a glide back");

    finish_glide(&mut h, &mut world);
    let p = camera(&world).position;
    for axis in 0..3 {
        assert!(
            (p[axis] - [1.0, 2.0, 3.0][axis]).abs() < 0.01,
            "the glide landed on the saved pose, got {p:?}"
        );
    }
}

// Recalling an empty slot does nothing rather than gliding to the origin.
#[test]
fn recalling_an_empty_slot_does_nothing() {
    let (world, _) = camera_world([1.0, 2.0, 3.0]);
    let mut h = hook();
    h.recall_bookmark(4, &world);
    assert!(h.glide.is_none());
}

// A recall cancels an in-flight tumble, so the two camera drives never both
// write the pose in one frame.
#[test]
fn a_recall_cancels_an_in_flight_tumble() {
    isolate_state_dir();
    let _guard = crate::test_support::lock();
    let (world, _) = camera_world([0.0, 0.0, 10.0]);
    let mut h = hook();
    h.selection.set(vec!["box".to_string()]);
    h.save_bookmark(2, &world);
    assert!(h.try_begin_orbit(&alt_press([600.0, 400.0]), VP, &world));

    h.recall_bookmark(2, &world);
    assert!(h.orbit.is_none(), "the tumble was dropped");
    assert!(h.glide.is_some());
}

// A billboard-backed selection member has no pick geometry, so its bounds come
// from the seeded Transform padded to a small box -- otherwise F would decline
// to frame a light or a camera.
#[test]
fn selection_bounds_fall_back_to_a_billboards_transform() {
    crate::ecs::asset_id::reset_interner();
    let id = crate::ecs::asset_id::intern("lamp");
    let mut world = World::new();
    let entity = world.push(crate::assets::Transform {
        position: [4.0, 5.0, 6.0],
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    });
    let mut by_name = std::collections::BTreeMap::new();
    by_name.insert(id, entity);
    world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
    world.insert_resource(crate::ecs::PickIndex::default());

    let mut h = hook();
    h.selection.set(vec!["lamp".to_string()]);
    let (mn, mx) = h
        .selection_bounds(&world)
        .expect("bounds from the transform");
    assert!(
        mn[0] < 4.0 && mx[0] > 4.0,
        "padded around x, got {mn:?}..{mx:?}"
    );
    assert!(mn[1] < 5.0 && mx[1] > 5.0);
    assert!(mn[2] < 6.0 && mx[2] > 6.0);
}

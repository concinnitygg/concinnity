// src/editor/hook/cinematic_tests.rs
//
// The start screen's attract camera, from the hook's side. The shot cycle, its
// poses, and its fade envelope are pure and tested beside them
// (`editor/worlds/cinematic.rs`); asserted here is the wiring -- when the cycle
// takes the preview's camera, what it refuses to take, when it hands the pose
// back, and that a shot never reaches the authored world.

use super::worlds_tests::{entry, open_project, world_with_name_field, write_world};
use super::*;
use crate::components::{Camera3D, CameraController, FollowController};
use framing::CameraPose;
use worlds::cinematic;

const VP: [f32; 2] = [1280.0, 720.0];
const AUTHORED: CameraPose = CameraPose {
    position: [0.0, 1.7, 12.0],
    yaw: 0.0,
    pitch: 0.0,
};

// A start-screen hook with no project behind it: everything under test here is
// driven directly, so the listing it would read is not needed.
fn start_hook() -> EditorHook {
    let mut h = EditorHook::new("unused.jsonl".to_string(), Vec::new());
    h.start_mode = true;
    h.worlds_open = true;
    h.viewport = VP;
    h
}

fn camera(controller: Option<CameraController>) -> Camera3D {
    Camera3D {
        fov_y_degrees: 60.0,
        near: 0.05,
        far: 200.0,
        view_matrix: concinnity_core::gfx::camera::view_matrix(
            AUTHORED.position,
            AUTHORED.yaw,
            AUTHORED.pitch,
        ),
        position: AUTHORED.position,
        yaw: AUTHORED.yaw,
        pitch: AUTHORED.pitch,
        desired_move: [0.0; 3],
        jump_requested: false,
        interact_requested: false,
        controller,
    }
}

// A previewed world: a camera at the authored pose, the fade sprite the
// injection provides, and a PickIndex holding one box (the renderable bounds
// the shots frame). `bounds` false leaves the index empty, which is the seeded
// empty scene.
fn preview_world(controller: Option<CameraController>, bounds: bool) -> World {
    crate::ecs::asset_id::reset_interner();
    let mut world = world_with_name_field();
    world.add_component(camera(controller));
    for id in std::iter::once(cinematic::FADE).chain(worlds::loading::all_sprite_ids()) {
        world.add_component(crate::components::Sprite {
            asset_id: id,
            ..Default::default()
        });
    }
    for id in worlds::loading::all_label_ids() {
        world.add_component(crate::components::TextLabel {
            asset_id: id,
            ..Default::default()
        });
    }
    let entries = match bounds {
        true => vec![crate::ecs::PickEntry {
            asset_id: crate::ecs::asset_id::intern("box"),
            bb_min: [-3.0, 0.0, -3.0],
            bb_max: [3.0, 2.0, 3.0],
        }],
        false => Vec::new(),
    };
    world.insert_resource(crate::ecs::PickIndex { entries });
    world
}

fn pose(world: &World) -> CameraPose {
    camera_pose::read(world).expect("the preview has a camera")
}

fn fade(world: &World) -> crate::components::Sprite {
    world
        .query::<crate::components::Sprite>()
        .find(|s| s.asset_id == cinematic::FADE)
        .cloned()
        .expect("the fade sprite is injected")
}

// Run one frame of the drive, with `dt` seconds behind it.
fn frame(h: &mut EditorHook, world: &mut World, dt: f32) {
    if h.cinematic_clock.is_some() {
        h.cinematic_clock =
            Some(std::time::Instant::now() - std::time::Duration::from_secs_f32(dt));
    }
    h.drive_cinematic(world);
    h.drive_cinematic_draw(world, VP, true);
    h.drive_loading_draw(world, true);
}

// Previewing a world with renderable bounds hands the view to the attract
// camera: it frames the bounds, and the screen opens on black.
#[test]
fn a_previewed_world_is_taken_by_the_attract_camera() {
    let mut h = start_hook();
    let mut world = preview_world(None, true);

    frame(&mut h, &mut world, 0.0);
    assert!(h.cinematic.is_some(), "the cycle is running");
    let opened = pose(&world);
    assert_ne!(opened, AUTHORED, "the shot placed the camera");
    // It stands off the bounds centre, looking down on it.
    assert!(opened.position[1] > 1.0 && opened.pitch < 0.0, "{opened:?}");
    let f = fade(&world);
    assert!(f.visible && f.tint[3] > 0.9, "the cycle opens on black");
    assert_eq!(f.tint[..3], [0.0, 0.0, 0.0], "which is black, not a dim");
    assert_eq!([f.width, f.height], VP, "covering the window");

    // The fade clears as the shot comes up, and the shot keeps moving.
    for _ in 0..20 {
        frame(&mut h, &mut world, 0.1);
    }
    assert_eq!(fade(&world).tint[3], 0.0, "the shot is up");
    assert_ne!(pose(&world), opened, "and the camera is moving");
}

// The pose is a presentation and nothing more: it never reaches the entries the
// screen would commit, and the preview never reads as edited.
#[test]
fn a_shot_never_reaches_the_authored_world() {
    let mut h = start_hook();
    h.entries = vec![entry("desk")];
    h.saved = h.entries.clone();
    let before = h.entries.clone();
    let bookmarks = h.bookmarks;
    let mut world = preview_world(None, true);

    for _ in 0..10 {
        frame(&mut h, &mut world, 0.1);
    }
    assert_eq!(h.entries, before);
    assert_eq!(h.saved, before);
    assert!(!h.dirty && !h.can_undo());
    assert_eq!(h.bookmarks, bookmarks, "and no bookmark slot moved");
}

// A world with nothing to frame (the seeded empty scene, or one whose props all
// collapsed) keeps its own camera and draws no fade.
#[test]
fn a_world_with_no_bounds_keeps_its_own_camera() {
    let mut h = start_hook();
    let mut world = preview_world(None, false);

    for _ in 0..5 {
        frame(&mut h, &mut world, 0.1);
    }
    assert!(h.cinematic.is_none(), "no cycle without bounds");
    assert_eq!(pose(&world), AUTHORED);
    assert!(!fade(&world).visible);
}

// A third-person camera is placed by the character it follows every step, so a
// pose written before that step would be gone by the time the frame drew.
#[test]
fn a_followed_camera_is_left_to_its_own_controller() {
    let mut h = start_hook();
    let controller = CameraController {
        follow: Some(FollowController::default()),
        ..Default::default()
    };
    let mut world = preview_world(Some(controller), true);

    frame(&mut h, &mut world, 0.0);
    assert!(h.cinematic.is_none());
    assert_eq!(pose(&world), AUTHORED);
    assert!(!fade(&world).visible);
}

// Both camera shapes the start screen meets are driven the same way, and both
// get their own pose back when the cycle ends.
#[test]
fn an_uncontrolled_and_a_free_fly_camera_are_both_taken_and_given_back() {
    for controller in [None, Some(CameraController::default())] {
        let mut h = start_hook();
        let mut world = preview_world(controller, true);
        for _ in 0..5 {
            frame(&mut h, &mut world, 0.1);
        }
        assert_ne!(pose(&world), AUTHORED, "the cycle took the camera");

        h.stop_cinematic(&mut world);
        assert_eq!(pose(&world), AUTHORED, "and handed it straight back");
        let cam = world.query::<Camera3D>().next().expect("the camera");
        assert_eq!(
            cam.view_matrix,
            concinnity_core::gfx::camera::view_matrix(
                AUTHORED.position,
                AUTHORED.yaw,
                AUTHORED.pitch
            ),
            "view matrix included"
        );
        assert!(h.cinematic.is_none() && h.cinematic_restore.is_none());
        h.drive_cinematic_draw(&mut world, VP, true);
        assert!(!fade(&world).visible, "and nothing black is left over");
    }
}

// The spin shot stands where the world's own camera stands: the pose it turns
// from is the one the preview came up on, not wherever the shot before it left
// the camera.
#[test]
fn the_spin_turns_at_the_worlds_own_camera() {
    let mut h = start_hook();
    let mut world = preview_world(None, true);
    // Run the opening shot out; the cycle hands over to the spin.
    while h.cinematic.map(|c| c.shot()) != Some(cinematic::Shot::Spin) {
        frame(&mut h, &mut world, 0.1);
    }
    for _ in 0..40 {
        frame(&mut h, &mut world, 0.1);
        let held = pose(&world);
        assert_eq!(held.position, AUTHORED.position, "it never leaves home");
        assert_eq!(held.pitch, AUTHORED.pitch, "nor changes the authored tilt");
    }
    assert_ne!(pose(&world).yaw, AUTHORED.yaw, "but it is turning");
}

// A world that loses its bounds mid-cycle (its props hidden, or a rebuild that
// emptied it) gets its camera back rather than keeping the shot that was on it.
#[test]
fn losing_the_bounds_mid_cycle_hands_the_camera_back() {
    let mut h = start_hook();
    let mut world = preview_world(None, true);
    for _ in 0..5 {
        frame(&mut h, &mut world, 0.1);
    }
    assert_ne!(pose(&world), AUTHORED);

    world.insert_resource(crate::ecs::PickIndex::default());
    frame(&mut h, &mut world, 0.1);
    assert!(h.cinematic.is_none() && h.cinematic_restore.is_none());
    assert_eq!(pose(&world), AUTHORED, "the world's own camera is back");
    assert!(!fade(&world).visible);
}

// A session editing a world never runs the cycle, whatever it inherited from
// the screen it opened from.
#[test]
fn a_session_never_runs_the_attract_camera() {
    let mut h = start_hook();
    let mut world = preview_world(None, true);
    frame(&mut h, &mut world, 0.0);
    assert!(h.cinematic.is_some());

    h.leave_start_screen();
    frame(&mut h, &mut world, 0.1);
    assert!(h.cinematic.is_none(), "the cycle stops with the screen");
    assert!(!fade(&world).visible);
    let held = pose(&world);
    frame(&mut h, &mut world, 0.1);
    assert_eq!(pose(&world), held, "and the session's camera is its own");
}

// A preview still owed its rebuild is showing the world on the way out: the
// cycle waits at black rather than framing it, and never holds that world's
// pose as the one to restore.
#[test]
fn a_pending_rebuild_holds_the_cycle_at_black() {
    let mut h = start_hook();
    let mut world = preview_world(None, true);
    h.rebuild_preview = true;

    for _ in 0..5 {
        frame(&mut h, &mut world, 0.1);
    }
    assert_eq!(pose(&world), AUTHORED, "the outgoing world is left alone");
    assert!(h.cinematic_restore.is_none());
    let cover = world
        .query::<crate::components::Sprite>()
        .find(|s| s.asset_id == worlds::loading::COVER)
        .cloned()
        .expect("the cover is injected");
    assert!(
        cover.visible,
        "and the screen holds behind the loading cover"
    );
    assert!(!fade(&world).visible, "which the fade stands down under");

    // The swap lands: the cycle takes the world that came up, from its start.
    h.rebuild_preview = false;
    frame(&mut h, &mut world, 0.1);
    assert_eq!(h.cinematic.map(|c| c.shot()), Some(cinematic::Shot::Orbit));
    assert_ne!(pose(&world), AUTHORED);
}

// Picking another row mid-fade restarts the cycle rather than resuming it: the
// new world opens on its first shot, and no black overlay is left standing.
#[test]
fn selecting_another_world_restarts_the_cycle() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let worlds_dir = dir.path().join("worlds");
    write_world(&worlds_dir, "arena", &[entry("crate_a")], 1_000);
    write_world(&worlds_dir, "lobby", &[entry("desk")], 3_000);

    let mut h = start_hook();
    h.refresh_worlds();
    let mut world = preview_world(None, true);
    // Run into the body of the first shot, well past its opening fade.
    for _ in 0..40 {
        frame(&mut h, &mut world, 0.1);
    }
    assert_eq!(fade(&world).tint[3], 0.0);
    let running = pose(&world);

    let i = h
        .worlds_rows
        .iter()
        .position(|r| r.name == "arena")
        .expect("the listing has it");
    h.apply_worlds_action(WorldsAction::Select(i), &mut world);
    assert!(h.cinematic.is_none(), "the cycle dropped with the world");
    assert!(h.rebuild_preview, "which is being rebuilt");

    // The rebuild is owed, so the loading cover stands over the preview area
    // and the world on its way out gets its own camera back, in case it stays.
    // The fade stands down under the cover: it is the same black, and it
    // leaves the sidebar out.
    frame(&mut h, &mut world, 0.1);
    assert!(h.loading_preview());
    assert!(!fade(&world).visible);
    assert_ne!(running, AUTHORED);
    assert_eq!(pose(&world), AUTHORED);

    // Once it lands the new preview opens on shot one, from black, and fades
    // in from there.
    h.rebuild_preview = false;
    frame(&mut h, &mut world, 0.1);
    assert_eq!(h.cinematic.map(|c| c.shot()), Some(cinematic::Shot::Orbit));
    assert_eq!(fade(&world).tint[3], 1.0, "the new cycle opens on black");
    frame(&mut h, &mut world, 0.1);
    let alpha = fade(&world).tint[3];
    assert!(alpha > 0.8 && alpha < 1.0, "fading in from black: {alpha}");

    crate::test_support::isolate_state_dir();
}

// Committing the preview ends the cycle before the session adopts the world, so
// the editor opens on the camera the world declared.
#[test]
fn opening_the_preview_hands_the_world_its_own_camera_back() {
    let _guard = crate::test_support::lock();
    let dir = concinnity_testing::TempTree::new();
    open_project(dir.path());
    let worlds_dir = dir.path().join("worlds");
    let lobby = write_world(&worlds_dir, "lobby", &[entry("desk")], 3_000);
    let path = lobby.to_string_lossy().into_owned();

    let mut h = start_hook();
    h.refresh_worlds();
    h.worlds_selected = Some(path.clone());
    h.worlds_preview = Some(path.clone());
    h.entries = vec![entry("desk")];
    h.world_entries = h.entries.clone();
    let mut world = preview_world(None, true);
    for _ in 0..30 {
        frame(&mut h, &mut world, 0.1);
    }
    assert_ne!(pose(&world), AUTHORED, "a shot is running");

    h.apply_worlds_action(WorldsAction::Open(0), &mut world);
    assert!(!h.start_mode, "the session owns the world now");
    assert_eq!(h.world_path, path);
    assert!(!h.rebuild_preview, "the running world was adopted as it is");
    assert_eq!(pose(&world), AUTHORED, "on its own camera");
    assert!(h.cinematic.is_none() && h.cinematic_restore.is_none());
    h.drive_cinematic_draw(&mut world, VP, true);
    assert!(!fade(&world).visible);

    crate::test_support::isolate_state_dir();
}

// The fade covers the previewed world and stops there: the sidebar listing over
// it draws above, so a transition never hides the list.
#[test]
fn the_fade_draws_under_the_sidebar() {
    let mut h = start_hook();
    let mut world = preview_world(None, true);
    frame(&mut h, &mut world, 0.0);

    let layers = h.compute_layers();
    let fade = layers[&cinematic::FADE];
    for id in EditorHook::panel_ids(PanelKey::Worlds) {
        assert!(
            layers[&id] > fade,
            "the sidebar draws over the fade: {id:?}"
        );
    }

    // With no cycle running the fade claims no layer at all.
    h.reset_cinematic();
    assert!(!h.compute_layers().contains_key(&cinematic::FADE));
}

// A rebuild owed while a shot holds the camera hands the world its own pose
// back first: a compile that fails leaves that world standing, and it must
// stand on its authored camera, not on the shot's.
#[test]
fn a_pending_rebuild_hands_the_pose_back_before_the_compile() {
    let mut h = start_hook();
    let mut world = preview_world(None, true);
    frame(&mut h, &mut world, 0.0);
    frame(&mut h, &mut world, 2.0);
    assert_ne!(pose(&world), AUTHORED, "a shot holds the camera");
    assert_eq!(h.cinematic_restore, Some(AUTHORED));

    h.restart_cinematic();
    h.rebuild_preview = true;
    frame(&mut h, &mut world, 0.5);
    assert_eq!(pose(&world), AUTHORED, "the world gets its camera back");
    assert!(h.cinematic_restore.is_none());
    assert!(
        h.cinematic.is_some(),
        "the cycle waits at its opening black"
    );

    // The compile failed: the same world stays, and the cycle takes it again
    // from its authored pose, not from wherever the last shot left it.
    h.rebuild_preview = false;
    frame(&mut h, &mut world, 0.0);
    assert_eq!(h.cinematic_restore, Some(AUTHORED));
}

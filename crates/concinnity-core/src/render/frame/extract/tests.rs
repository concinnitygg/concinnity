use crate::animation::pose_scratch::PoseScratch;
use crate::animation::proportions::ProportionLayer;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use super::*;
use crate::animation::skeleton;
use crate::components::{GlobalTransform, RenderHandle, SkeletonPose};
use crate::ecs::{SkinnedMeshHandle, World};
use crate::render::frame::snapshot::SceneOp;
use crate::render::overlay::OverlayFrame;

fn extract_once(fx: &mut FrameExtractor, world: &mut World) -> RenderSnapshot {
    let mut snap = RenderSnapshot::default();
    fx.extract(&mut world.context(), &mut snap, 0);
    snap
}

// A requested capture rides the next frame extracted, and only that one.
#[test]
fn a_capture_request_rides_the_next_frame_only() {
    let mut world = World::new();
    world.insert_resource(FrameCaptureRequest("shot.png".to_string()));
    let mut fx = FrameExtractor::default();
    assert_eq!(
        extract_once(&mut fx, &mut world).capture.as_deref(),
        Some("shot.png")
    );
    assert!(world.resource::<FrameCaptureRequest>().is_none());
    assert_eq!(extract_once(&mut fx, &mut world).capture, None);
}

// A started burst captures the frames after it, named for each frame.
#[test]
fn a_screenshot_burst_rides_the_frames_extracted_after_it() {
    let mut world = World::new();
    let mut fx = FrameExtractor::default();
    fx.start_screenshots(ScreenshotBurst::start("shots", "s".to_string(), 1));
    let mut snap = RenderSnapshot::default();
    fx.extract(&mut world.context(), &mut snap, 9);
    assert_eq!(snap.capture.as_deref(), Some("shots/s-frame9.png"));
    assert_eq!(extract_once(&mut fx, &mut world).capture, None);
}

// The sky reaches the frame twice: as the rows every cubemap sample is
// taken through, and as the rotated directional set the frame carries.
#[test]
fn extraction_carries_the_sky_into_the_frame_and_the_lights() {
    let mut world = World::new();
    {
        let mut ctx = world.context();
        ctx.components.push(
            DirectionalLight {
                direction: [0.0, 0.0, 1.0],
                ..Default::default()
            }
            .into(),
        );
        ctx.insert_resource(SkyOrientation::new([1.0, 0.0, 0.0], 90.0));
    }
    let mut fx = FrameExtractor::default();
    let snap = extract_once(&mut fx, &mut world);

    assert_ne!(snap.frame.sky_rot, SkyOrientation::IDENTITY_ROWS);
    match snap.frame.directional.map(|set| set.as_slice().to_vec()) {
        Some(lights) if lights.len() == 1 => assert!(
            lights[0].direction[1] > 0.99,
            "a quarter turn puts it overhead: {lights:?}"
        ),
        other => panic!("expected the turned set, got {other:?}"),
    }

    // The sky has not moved since, so nothing is restated.
    let snap = extract_once(&mut fx, &mut world);
    assert_eq!(snap.frame.directional, None);
}

fn translated(x: f32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, 0.0, 0.0, 1.0],
    ]
}

fn bare_pose(skinned_index: u32, joints: usize) -> SkeletonPose {
    SkeletonPose {
        mesh_id: SkinnedMeshHandle::new(0),
        skinned_index: SkinnedIndex(skinned_index),
        skeleton: skeleton::Skeleton::new(Vec::new()),
        joint_matrices: vec![translated(1.0); joints],
        morph_weights: Vec::new(),
        morph_base: Vec::new(),
        proportions: ProportionLayer::default(),
        updated: true,
        scratch: PoseScratch::default(),
    }
}

// A changed model matrix lands in the snapshot once per draw slot; an
// unchanged frame extracts nothing for the same entity.
#[test]
fn extraction_gathers_changed_models_and_gates_repeats() {
    let mut world = World::new();
    {
        let mut ctx = world.context();
        let e = ctx.components.spawn();
        ctx.insert(e, GlobalTransform(translated(2.0)));
        ctx.insert(
            e,
            RenderHandle {
                draws: [DrawIndex(3), DrawIndex(4)].into(),
            },
        );
    }
    let mut fx = FrameExtractor::default();
    let snap = extract_once(&mut fx, &mut world);
    assert_eq!(
        snap.models,
        vec![
            (DrawIndex(3), translated(2.0)),
            (DrawIndex(4), translated(2.0))
        ]
    );

    let snap = extract_once(&mut fx, &mut world);
    assert!(snap.models.is_empty(), "static frame re-extracts nothing");
}

// An updated pose is copied into the snapshot spans and its handshake flag
// cleared, so the next frame extracts nothing for it. Morph weights ride
// along only when present.
#[test]
fn extraction_copies_updated_poses_and_clears_the_flag() {
    let mut world = World::new();
    {
        let mut ctx = world.context();
        let e = ctx.components.spawn();
        let mut pose = bare_pose(5, 2);
        pose.morph_weights = vec![0.25, 0.75];
        ctx.insert(e, pose);
    }
    let mut fx = FrameExtractor::default();
    let snap = extract_once(&mut fx, &mut world);
    let poses: Vec<(SkinnedIndex, usize)> =
        snap.poses.iter().map(|(idx, m)| (idx, m.len())).collect();
    assert_eq!(poses, vec![(SkinnedIndex(5), 2)]);
    let morphs: Vec<(SkinnedIndex, Vec<f32>)> = snap
        .morphs
        .iter()
        .map(|(idx, w)| (idx, w.to_vec()))
        .collect();
    assert_eq!(morphs, vec![(SkinnedIndex(5), vec![0.25, 0.75])]);

    let snap = extract_once(&mut fx, &mut world);
    assert!(snap.poses.is_empty(), "consumed pose is not re-extracted");
    assert!(snap.morphs.is_empty());
}

// While a menu is open the skinned families are skipped entirely, and the
// frame scalars + overlay adoption reflect the menu state.
#[test]
fn extraction_skips_skinned_families_while_a_menu_is_open() {
    let mut world = World::new();
    {
        let mut ctx = world.context();
        let e = ctx.components.spawn();
        ctx.insert(e, bare_pose(0, 1));
        ctx.insert_resource(OverlayFrame {
            calls: Vec::new(),
            want_ui_cursor: true,
            menu_active: true,
            world_hidden: true,
        });
    }
    let mut fx = FrameExtractor::default();
    let snap = extract_once(&mut fx, &mut world);
    assert!(snap.poses.is_empty(), "menu freezes pose extraction");
    assert!(snap.frame.menu_active);
    assert!(snap.frame.world_hidden);
    assert!(snap.ui.cursor_hidden);
    let parked = world
        .resource::<OverlayFrame>()
        .expect("the slot stays parked after the take");
    assert!(
        parked.calls.is_empty() && !parked.menu_active && !parked.world_hidden,
        "the overlay build is consumed so a stale one is never redrawn"
    );

    // The pose is still flagged, so closing the menu extracts it.
    let snap = extract_once(&mut fx, &mut world);
    assert_eq!(snap.poses.iter().count(), 1);
}

fn camera_at(x: f32) -> Camera3D {
    Camera3D {
        fov_y_degrees: 70.0,
        near: 0.1,
        view_distance: None,
        view_matrix: translated(-x),
        position: [x, 0.0, 0.0],
        yaw: 0.0,
        pitch: 0.0,
        up: [0.0, 1.0, 0.0],
        desired_move: [0.0; 3],
        jump_requested: false,
        interact_requested: false,
        controller: None,
    }
}

// The camera-relative view the streaming drive published wins over the
// absolute Camera3D values; projection parameters come from the camera.
#[test]
fn extraction_prefers_the_camera_relative_view() {
    let mut world = World::new();
    {
        let mut ctx = world.context();
        let e = ctx.components.spawn();
        ctx.insert(
            e,
            Camera3D {
                fov_y_degrees: 90.0,
                view_distance: Some(500.0),
                view_matrix: translated(1.0),
                position: [1.0, 2.0, 3.0],
                ..camera_at(0.0)
            },
        );
        ctx.insert_resource(CameraRelativeView {
            view: translated(7.0),
            cam_pos: [7.0, 0.0, 0.0],
        });
    }
    let mut fx = FrameExtractor::default();
    let snap = extract_once(&mut fx, &mut world);
    assert_eq!(snap.frame.view, translated(7.0));
    assert_eq!(snap.frame.cam_pos, [7.0, 0.0, 0.0]);
    assert!((snap.frame.fov_y_radians - core::f32::consts::FRAC_PI_2).abs() < 1e-6);
    assert_eq!(snap.frame.near, 0.1);
    assert_eq!(snap.frame.view_distance, Some(500.0));
}

// A teleport, a settings change, a raised cut and a change of camera each
// reset the temporal history for one frame; a rebase of the render
// origin, which moves the camera-relative view but not the camera, does
// not.
#[test]
fn extraction_flags_a_history_reset_on_a_cut_or_a_settings_change() {
    let reset =
        |fx: &mut FrameExtractor, world: &mut World| extract_once(fx, world).frame.history_reset;
    let mut world = World::new();
    let cam = {
        let mut ctx = world.context();
        let e = ctx.components.spawn();
        ctx.insert(e, camera_at(0.0));
        e
    };
    let mut fx = FrameExtractor::default();
    assert!(!reset(&mut fx, &mut world).any());

    world.context().insert_resource(CameraRelativeView {
        view: translated(-0.5),
        cam_pos: [0.5, 0.0, 0.0],
    });
    assert!(!reset(&mut fx, &mut world).any());

    *world.context().get_mut::<Camera3D>(cam).unwrap() = camera_at(500.0);
    assert_eq!(reset(&mut fx, &mut world), HistoryResetCauses::CAMERA_CUT);
    assert!(!reset(&mut fx, &mut world).any());

    PendingHistoryReset::raise(&mut world.context(), HistoryResetCauses::SETTINGS_CHANGE);
    assert_eq!(
        reset(&mut fx, &mut world),
        HistoryResetCauses::SETTINGS_CHANGE
    );
    assert!(!reset(&mut fx, &mut world).any());

    // A raised cut lands with the camera's move, however small.
    PendingHistoryReset::raise(&mut world.context(), HistoryResetCauses::CAMERA_CUT);
    assert!(!reset(&mut fx, &mut world).any());
    *world.context().get_mut::<Camera3D>(cam).unwrap() = camera_at(500.5);
    assert_eq!(reset(&mut fx, &mut world), HistoryResetCauses::CAMERA_CUT);
    assert!(!reset(&mut fx, &mut world).any());

    // Drawing from another camera.
    {
        let mut ctx = world.context();
        ctx.components.despawn(cam);
        let other = ctx.components.spawn();
        ctx.insert(other, camera_at(500.5));
    }
    assert_eq!(reset(&mut fx, &mut world), HistoryResetCauses::CAMERA_CUT);
    assert!(!reset(&mut fx, &mut world).any());
}

// A dropped frame's causes are handed to the next frame extracted.
#[test]
fn a_reissued_reset_lands_on_the_next_frame() {
    let mut world = World::new();
    let mut fx = FrameExtractor::default();
    extract_once(&mut fx, &mut world);
    let dropped = FrameScalars {
        history_reset: HistoryResetCauses::SETTINGS_CHANGE,
        ..RenderSnapshot::default().frame
    };
    fx.reissue_history_reset(&dropped);
    assert_eq!(
        extract_once(&mut fx, &mut world).frame.history_reset,
        HistoryResetCauses::SETTINGS_CHANGE
    );
}

// An in-flight fade records its effects as scene ops instead of touching a
// backend; an idle flow records nothing.
#[test]
fn extraction_records_fade_effects_as_scene_ops() {
    let mut world = World::new();
    world.insert_resource(ActiveSceneFlow::new(Some(flow::SceneFlow {
        scenes: vec![AssetId(1), AssetId(2)],
        current: AssetId(1),
        fade: flow::FadePhase::ToBlack {
            started_at: 0.0,
            next: AssetId(2),
        },
    })));
    let mut fx = FrameExtractor::default();
    let snap = extract_once(&mut fx, &mut world);
    assert!(
        matches!(snap.scene_ops.first(), Some(SceneOp::SetFade(_))),
        "an in-flight fade records its fade value"
    );

    // Clear the fade: nothing is recorded on an idle flow.
    if let Some(slot) = world.resource_mut::<ActiveSceneFlow>() {
        slot.flow.as_mut().unwrap().fade = flow::FadePhase::None;
    }
    let snap = extract_once(&mut fx, &mut world);
    assert!(snap.scene_ops.is_empty());
}

// The editor's menu override resolves to submit intents; without it (a
// shipped runtime, no menu-mode world) both intents stay None.
#[test]
fn extraction_resolves_menu_override_into_ui_intents() {
    let mut world = World::new();
    let mut fx = FrameExtractor::default();
    let snap = extract_once(&mut fx, &mut world);
    assert_eq!(snap.ui.menu_mode, None);
    assert_eq!(snap.ui.camera_capture, None);

    world.insert_resource(MenuOverride(Some(false)));
    let snap = extract_once(&mut fx, &mut world);
    assert_eq!(snap.ui.menu_mode, Some(true));
    assert_eq!(
        snap.ui.camera_capture,
        Some(true),
        "no active menu screen, so the camera captures"
    );

    // Forced edit mode: the overlay build folded the override into an active
    // menu, so the backend enters menu mode with the cursor released.
    world.insert_resource(MenuOverride(Some(true)));
    world.insert_resource(menu_frame(true));
    let snap = extract_once(&mut fx, &mut world);
    assert_eq!(snap.ui.menu_mode, Some(true));
    assert_eq!(snap.ui.camera_capture, Some(false));
}

// The overlay build of a frame whose menu is `active`.
fn menu_frame(active: bool) -> OverlayFrame {
    OverlayFrame {
        calls: Vec::new(),
        want_ui_cursor: false,
        menu_active: active,
        world_hidden: false,
    }
}

// A menu-mode world releases the camera capture while its menu is open and
// takes it back once the menu closes; menu mode itself was set at init.
#[test]
fn a_menu_mode_world_releases_the_camera_behind_an_open_menu() {
    let mut world = World::new();
    let mut fx = FrameExtractor::default();
    fx.set_menu_mode(true);
    world.insert_resource(menu_frame(true));
    let snap = extract_once(&mut fx, &mut world);
    assert_eq!(snap.ui.camera_capture, Some(false));
    assert_eq!(snap.ui.menu_mode, None);

    world.insert_resource(menu_frame(false));
    assert_eq!(
        extract_once(&mut fx, &mut world).ui.camera_capture,
        Some(true)
    );
}

// Each frame's extract adopts the overlay draw list and hands the spent one
// back through `OverlayRecycle` for the next overlay build to reuse.
#[test]
fn extraction_hands_the_spent_overlay_list_back() {
    let mut world = World::new();
    let mut fx = FrameExtractor::default();
    for _ in 0..2 {
        extract_once(&mut fx, &mut world);
        assert!(world.resource::<OverlayRecycle>().is_some());
    }
}

// Without a camera-relative view the frame draws from the camera's own view
// and position.
#[test]
fn extraction_reads_the_camera_view_without_a_relative_view() {
    let mut world = World::new();
    let cam = {
        let mut ctx = world.context();
        let e = ctx.components.spawn();
        ctx.insert(e, camera_at(0.0));
        e
    };
    let mut fx = FrameExtractor::default();
    extract_once(&mut fx, &mut world);
    *world.context().get_mut::<Camera3D>(cam).unwrap() = Camera3D {
        position: [7.0, 8.0, 9.0],
        view_matrix: translated(-7.0),
        ..camera_at(0.0)
    };
    let snap = extract_once(&mut fx, &mut world);
    assert_eq!(snap.frame.view, translated(-7.0));
    assert_eq!(snap.frame.cam_pos, [7.0, 8.0, 9.0]);
}

// A Transform edit propagates into the entity's model: the first frame seeds
// its slot, static frames push nothing, and one move pushes exactly one
// update carrying the new matrix.
#[test]
fn a_transform_edit_pushes_one_model_update() {
    let mut world = World::new();
    let entity = {
        let mut ctx = world.context();
        let e = ctx.components.spawn();
        ctx.insert(e, Transform::default());
        ctx.insert(e, GlobalTransform(crate::transform::IDENTITY));
        ctx.insert(
            e,
            RenderHandle {
                draws: [DrawIndex(0)].into(),
            },
        );
        e
    };
    let mut fx = FrameExtractor::default();
    assert_eq!(extract_once(&mut fx, &mut world).models.len(), 1);
    for _ in 0..2 {
        assert!(extract_once(&mut fx, &mut world).models.is_empty());
    }

    world
        .context()
        .get_mut::<Transform>(entity)
        .unwrap()
        .position = [10.0, 20.0, 30.0];
    let snap = extract_once(&mut fx, &mut world);
    assert_eq!(snap.models.len(), 1, "one move, one update");
    let (slot, model) = snap.models[0];
    assert_eq!(slot, DrawIndex(0));
    assert_eq!([model[3][0], model[3][1], model[3][2]], [10.0, 20.0, 30.0]);
    assert!(extract_once(&mut fx, &mut world).models.is_empty());
}

// Editor-session hidden objects overwrite their queued models with the
// degenerate matrix and drop out of the published pick index; without the
// editor's resources nothing of the sort is extracted.
#[test]
fn extraction_applies_editor_hides_and_pick_index_only_when_opted_in() {
    let mut world = World::new();
    let entity = {
        let mut ctx = world.context();
        let e = ctx.components.spawn();
        ctx.insert(e, GlobalTransform(translated(2.0)));
        ctx.insert(
            e,
            RenderHandle {
                draws: [DrawIndex(9)].into(),
            },
        );
        e
    };
    // A shipped runtime: no candidates, so no pick index is published.
    let mut fx = FrameExtractor::default();
    let snap = extract_once(&mut fx, &mut world);
    assert_eq!(snap.models, vec![(DrawIndex(9), translated(2.0))]);
    assert!(world.resource::<PickIndex>().is_none());

    // The editor opted in: the index lists the candidate.
    fx.pick_candidates_mut().push(PickCandidate {
        asset_id: AssetId(42),
        entity,
        local_min: [-1.0; 3],
        local_max: [1.0; 3],
    });
    extract_once(&mut fx, &mut world);
    let index = world.resource::<PickIndex>().unwrap();
    assert_eq!(index.entries.len(), 1);
    let (slot, storage) = (
        core::ptr::from_ref::<PickIndex>(index),
        index.entries.as_ptr(),
    );
    extract_once(&mut fx, &mut world);
    let index = world.resource::<PickIndex>().unwrap();
    assert!(
        core::ptr::eq(slot, index) && storage == index.entries.as_ptr(),
        "the next frame refills the published index in place"
    );

    // It hid the asset: the queued model is overwritten in order and the
    // index excludes it.
    world.insert_resource(HiddenAssets([AssetId(42)].into_iter().collect()));
    {
        let mut ctx = world.context();
        if let Some(g) = ctx.get_mut::<GlobalTransform>(entity) {
            g.0 = translated(3.0);
        }
    }
    let snap = extract_once(&mut fx, &mut world);
    assert_eq!(
        snap.models,
        vec![
            (DrawIndex(9), translated(3.0)),
            (DrawIndex(9), HIDDEN_MODEL)
        ],
        "the hide overwrite follows the move so it wins on the backend"
    );
    let index = world.resource::<PickIndex>().unwrap();
    assert!(index.entries.is_empty(), "a hidden asset is not pickable");

    // Clearing the set restores both the real model and the index entry.
    world.insert_resource(HiddenAssets::default());
    let snap = extract_once(&mut fx, &mut world);
    assert_eq!(snap.models, vec![(DrawIndex(9), translated(3.0))]);
    assert_eq!(world.resource::<PickIndex>().unwrap().entries.len(), 1);
}

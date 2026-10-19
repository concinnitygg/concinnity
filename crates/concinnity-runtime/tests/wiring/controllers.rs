// The camera controllers under the engine's system table: the gates that pick
// a controller, and the paths that need the load-time decomposition and the
// other systems the table builds beside them.

use concinnity_core::components::{
    Camera3D, CameraController, FollowController, FrameInput, Interactable, KeyBinding,
    PhysicsConfig, Prop, PropCollider, PropColliderShape, Screen, ScreenCommand, Transform,
    UiAction,
};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{EventCursor, SkinnedMeshHandle, World};
use concinnity_host::thread::asset_id::intern;

use concinnity_runtime::SYSTEMS;

fn camera(controller: Option<CameraController>) -> Camera3D {
    Camera3D {
        fov_y_degrees: 75.0,
        near: 0.05,
        view_distance: None,
        view_matrix: [[0.0; 4]; 4],
        position: [0.0; 3],
        yaw: 0.0,
        pitch: 0.0,
        up: [0.0, 1.0, 0.0],
        desired_move: [0.0; 3],
        jump_requested: false,
        interact_requested: false,
        controller,
    }
}

fn system_names(world: &World) -> Vec<&'static str> {
    world.systems().iter().map(|s| s.name()).collect()
}

fn follow_camera() -> Camera3D {
    camera(Some(CameraController {
        follow: Some(FollowController {
            target: Some(SkinnedMeshHandle::new(intern("hero").0)),
            distance: 4.0,
            height: 1.5,
            ..FollowController::default()
        }),
        ..CameraController::default()
    }))
}

// A Camera3D whose `controller` is set spawns the internal controller.
#[test]
fn controlled_camera_spawns_internal_system() {
    let mut world = World::new();
    world.add_component(camera(Some(CameraController::default())));
    world.start(SYSTEMS).unwrap();
    assert_eq!(system_names(&world), ["Camera3DSystem"]);
}

// `controller: null` opts out: a cutscene camera gets no controller.
#[test]
fn uncontrolled_camera_has_no_system() {
    let mut world = World::new();
    world.add_component(camera(None));
    world.start(SYSTEMS).unwrap();
    assert!(world.systems().is_empty());
}

// A follow block selects the third-person controller instead.
#[test]
fn follow_controller_spawns_third_person_system() {
    let mut world = World::new();
    world.add_component(follow_camera());
    world.start(SYSTEMS).unwrap();
    let names = system_names(&world);
    assert!(names.contains(&"ThirdPersonSystem"), "{names:?}");
    assert!(!names.contains(&"Camera3DSystem"), "{names:?}");
}

// With both a controlled camera and a UiInputSystem (Screen + KeyBinding),
// both systems read the same per-frame FrameInput: Camera3DSystem runs first
// but does not consume it, so UiInputSystem still receives Escape and toggles
// the menu. (Regression: Camera3DSystem drained the input, starving the menu,
// so Escape did nothing over a captured camera.)
#[test]
fn camera_and_ui_share_frame_input() {
    let mut world = World::new();
    world.add_component(camera(Some(CameraController::default())));
    world.push_identified(
        AssetId(50),
        Screen {
            initial: false,
            fade_in_secs: 0.0,
            ..Default::default()
        },
    );
    world.add_component(KeyBinding {
        key: "Escape".to_string(),
        action: Some(UiAction::Screen(ScreenCommand::Toggle(AssetId(50)))),
        ..Default::default()
    });
    world.start(SYSTEMS).unwrap();

    let names = system_names(&world);
    assert!(names.contains(&"Camera3DSystem"));
    assert!(names.contains(&"UiInputSystem"));

    world.add_component(FrameInput {
        escape: true,
        ..Default::default()
    });
    world.step();

    let mut cursor = EventCursor::default();
    let cmd = world
        .events::<ScreenCommand>()
        .and_then(|e| e.read(&mut cursor).next().cloned());
    assert!(
        matches!(cmd, Some(ScreenCommand::Toggle(AssetId(50)))),
        "UiInputSystem must still process Escape when a camera is present"
    );
}

// Pressing interact rotates the target entity's Transform 45 degrees; the Prop
// column was drained at load, so the rotation can only land on the Transform.
#[test]
fn interact_rotates_transform() {
    let mut world = World::new();
    world.add_component(camera(Some(CameraController {
        free_fly: true,
        ..CameraController::default()
    })));
    // Two units ahead of a camera facing -Z: within reach and the facing cone.
    world.push_identified(
        AssetId(1),
        Prop {
            position: [0.0, 0.0, -2.0],
            interactable: true,
            ..Default::default()
        },
    );
    world.start(SYSTEMS).unwrap();
    world.add_component(FrameInput {
        interact: true,
        ..Default::default()
    });
    world.step();

    let transform_yaw = world
        .join2::<Interactable, Transform>()
        .map(|(_, _, t)| t.rotation_deg[1])
        .next()
        .expect("interactable entity has a Transform");
    assert_eq!(transform_yaw, 45.0, "interact rotates the Transform");
    assert_eq!(
        world.query::<Prop>().count(),
        0,
        "the Prop column is drained at load"
    );
}

// A wall between the character and the orbit camera pulls the camera in front
// of it: the occlusion probe (answered by PhysicsSystem) clamps the follow
// distance so the view is never cut.
#[test]
fn wall_behind_the_camera_pulls_it_in() {
    use concinnity_core::components::CharacterRig;
    use concinnity_core::ecs::FrameTime;
    use concinnity_core::gfx::render_types::SkinnedIndex;

    let mut world = World::new();
    world.add_component(follow_camera());
    world.add_component(CharacterRig::new(
        SkinnedMeshHandle::new(intern("hero").0),
        SkinnedIndex(0),
        concinnity_core::transform::IDENTITY,
        0.5,
        0.3,
    ));
    world.add_component(PhysicsConfig::default());
    // A wall crossing the camera's line at z = +2 (the camera orbits to z = +4
    // at yaw 0, the pivot sits at z = 0).
    world.push_identified(
        intern("wall"),
        Prop {
            position: [0.0, 1.5, 2.0],
            collider: Some(PropCollider {
                shape: PropColliderShape::Cuboid,
                half_extents: [3.0, 1.5, 0.2],
                radius: 0.0,
                half_height: 0.0,
                layer: String::new(),
            }),
            ..Default::default()
        },
    );
    world.start(SYSTEMS).unwrap();
    world.add_component(FrameInput::default());
    world.insert_resource(FrameTime {
        dt: 0.005,
        elapsed: 0.0,
    });
    for _ in 0..6 {
        world.step();
    }

    let camera = world.query::<Camera3D>().next().unwrap();
    assert!(
        camera.position[2] < 1.9,
        "camera pulled in front of the wall at z = 2: {:?}",
        camera.position
    );
    assert!(
        camera.position[2] > 0.3,
        "camera stays behind the pivot: {:?}",
        camera.position
    );
}

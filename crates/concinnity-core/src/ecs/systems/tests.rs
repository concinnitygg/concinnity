use super::*;
use crate::components::{
    Behavior, Camera3D, CameraController, CameraTrack, DebugHud, FollowController, FpsCounter,
    PhysicsConfig, RigidBody, Screen, SkyRotation, StatHud, Story, TextInput,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult};
use alloc::vec::Vec;

// Every core entry at once, for the gates that only a rendering host's table
// would otherwise reach.
const ALL: &SystemTable = &SystemTable {
    entries: Core::ALL,
    ..SystemTable::EMPTY
};

#[test]
fn both_tables_are_well_formed_and_keep_the_core_order() {
    assert_eq!(ALL.check_complete(Core::ALL), Ok(()));
    assert_eq!(HEADLESS_SYSTEMS.check(Core::ALL), Ok(()));
}

// A world with no content gates nothing in, so a headless run over it has
// nothing to do.
#[test]
fn an_empty_world_gates_no_systems() {
    let world = World::new();
    assert!(world.system_manifest(HEADLESS_SYSTEMS).is_empty());
    assert!(world.system_manifest(ALL).is_empty());
}

// A declared SkyRotation gates the sky in, ahead of every system that reads
// the orientation it publishes.
#[test]
fn a_sky_rotation_gates_the_sky_system_first() {
    let mut world = World::new();
    world.add_component(SkyRotation::default());
    world.add_component(Behavior::default());
    assert_eq!(
        world.system_manifest(HEADLESS_SYSTEMS),
        ["SkyRotationSystem", "BehaviorSystem"]
    );
}

// A camera track needs both halves: the path and the camera it drives.
// Either alone leaves the system out rather than half-building it.
#[test]
fn a_camera_track_gates_the_track_system_only_beside_a_camera() {
    let mut world = World::new();
    world.add_component(CameraTrack::default());
    assert!(world.system_manifest(HEADLESS_SYSTEMS).is_empty());

    let mut world = World::new();
    world.add_component(Camera3D::bake(Default::default()));
    assert!(
        !world
            .system_manifest(HEADLESS_SYSTEMS)
            .contains(&"CameraTrackSystem")
    );

    let mut world = World::new();
    world.add_component(CameraTrack::default());
    world.add_component(Camera3D::bake(Default::default()));
    world.add_component(PhysicsConfig::default());
    assert_eq!(
        world.system_manifest(HEADLESS_SYSTEMS),
        ["PhysicsSystem", "CameraTrackSystem"]
    );
}

// A declared track takes the camera off the input controller, so exactly one
// system writes the pose.
#[test]
fn a_camera_track_replaces_the_camera_controller() {
    let mut world = World::new();
    world.add_component(Camera3D::bake(Default::default()));
    world.add_component(PhysicsConfig::default());
    assert_eq!(
        world.system_manifest(HEADLESS_SYSTEMS),
        ["PhysicsSystem", "Camera3DSystem"]
    );
}

// The first controlled camera's `follow` block picks exactly one controller.
#[test]
fn camera_controller_gates_are_exclusive() {
    let mut fly_cam = Camera3D::bake(Default::default());
    fly_cam.controller = Some(CameraController::default());
    let mut fly = World::new();
    fly.add_component(fly_cam);
    assert_eq!(fly.system_manifest(HEADLESS_SYSTEMS), ["Camera3DSystem"]);

    let mut follow_cam = Camera3D::bake(Default::default());
    follow_cam.controller = Some(CameraController {
        follow: Some(FollowController::default()),
        ..Default::default()
    });
    let mut follow = World::new();
    follow.add_component(follow_cam);
    assert_eq!(
        follow.system_manifest(HEADLESS_SYSTEMS),
        ["ThirdPersonSystem"]
    );
}

// Physics content gates the simulation driver in, on its own and beside a
// behavior, in table order.
#[test]
fn physics_content_gates_the_physics_system() {
    let mut world = World::new();
    world.add_component(PhysicsConfig::default());
    assert_eq!(world.system_manifest(HEADLESS_SYSTEMS), ["PhysicsSystem"]);

    let mut world = World::new();
    world.add_component(RigidBody::default());
    assert_eq!(world.system_manifest(HEADLESS_SYSTEMS), ["PhysicsSystem"]);

    let mut world = World::new();
    world.add_component(Behavior::default());
    world.add_component(PhysicsConfig::default());
    assert_eq!(
        world.system_manifest(HEADLESS_SYSTEMS),
        ["BehaviorSystem", "PhysicsSystem"]
    );
}

// The HUD components each gate their system, in table order. The debug HUD
// is developer-only, but `cfg!(debug_assertions)` holds under test.
#[test]
fn hud_components_gate_in_table_order() {
    let mut world = World::new();
    world.add_component(FpsCounter::default());
    world.add_component(StatHud::default());
    world.add_component(DebugHud::default());
    assert_eq!(
        world.system_manifest(ALL),
        ["StatHudSystem", "DebugHudSystem", "FpsCounterSystem"]
    );
}

// Any HitRegion, Screen or KeyBinding gates the UI system in.
#[test]
fn ui_components_gate_the_ui_system() {
    let mut world = World::new();
    world.push_identified(AssetId(1), Screen::default());
    assert_eq!(world.system_manifest(ALL), ["UiInputSystem"]);
}

// A Story and a TextInput each gate their system.
#[test]
fn a_story_and_a_text_field_gate_their_systems() {
    let mut world = World::new();
    world.add_component(Story::default());
    world.add_component(TextInput::default());
    assert_eq!(
        world.system_manifest(ALL),
        ["StorySystem", "TextInputSystem"]
    );
}

// The render band stays out of a world no host published a rendered run
// mode for.
#[test]
fn the_render_band_needs_a_rendered_run() {
    use crate::render::render_mode::RenderMode;

    let mut world = World::new();
    assert!(world.system_manifest(ALL).is_empty());
    world.insert_resource(RenderMode::Rendered);
    assert_eq!(
        world.system_manifest(ALL),
        [
            "OverlaySystem",
            "InputSystem",
            "SpawnSystem",
            "SettingsSystem",
            "StreamingSystem"
        ]
    );
}

// The manifest reports exactly the systems `start()` builds, in order.
#[test]
fn system_manifest_matches_started_systems() {
    let mut world = World::new();
    world.add_component(StatHud::default());
    world.add_component(FpsCounter::default());
    world.add_component(Story::default());
    world.add_component(TextInput::default());

    let manifest = world.system_manifest(ALL);
    world.start(ALL).unwrap();
    let built: Vec<&str> = world
        .systems()
        .iter()
        .map(super::super::built_system::BuiltSystem::name)
        .collect();
    assert_eq!(manifest, built);
}

// A system registered on a world lands in the phase it named: after every
// table entry sharing that phase, and before every entry of a later one.
#[test]
fn a_registration_lands_in_the_phase_it_named() {
    #[derive(Debug)]
    struct Inert;

    impl System for Inert {
        fn step(&mut self, _ctx: &mut PipelineContext<'_>) -> StepResult {
            StepResult::Continue
        }
    }

    let mut world = World::new();
    world.add_component(SkyRotation::default());
    world.add_component(FpsCounter::default());
    world.add_system(Phase::Early, "MyEarly", Inert);
    world.add_system(Phase::Late, "MyLate", Inert);

    assert_eq!(
        world.system_manifest(ALL),
        ["SkyRotationSystem", "MyEarly", "FpsCounterSystem", "MyLate"]
    );
}

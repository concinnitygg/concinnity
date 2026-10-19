//! Gate builders for the system table (`define_systems!` in `registry`). Each
//! gate inspects the world and returns the constructed system when what it
//! gates on is there, or `None` to leave it out of the schedule. Most read the
//! world's components; the render band reads the run mode `Runtime::start`
//! resolved and published before any of them ran. `World::start` and `World::system_manifest` both run these same
//! gates, so what the manifest reports and what `start` builds cannot drift.
//!
//! Gates construct their system, so every system constructor must stay cheap
//! and side-effect-free: the manifest probe discards the value, and anything
//! heavy (device acquisition, payload reads) belongs in `System::init`.

use concinnity_core::components::AudioCue;
use concinnity_core::components::AudioEmitter;
use concinnity_core::components::Behavior;
use concinnity_core::components::DebugHud;
use concinnity_core::components::FpsCounter;
use concinnity_core::components::HitRegion;
use concinnity_core::components::KeyBinding;
use concinnity_core::components::LoadingOverlay;
use concinnity_core::components::PropBody;
use concinnity_core::components::Screen;
use concinnity_core::components::SkyRotation;
use concinnity_core::components::StatHud;
use concinnity_core::components::Story;
use concinnity_core::components::TextInput;
use concinnity_core::ecs::World;

// The render band's shared gate: the run mode `Runtime::start` resolved and
// published before building any system. A world reaching here without one
// never went through `Runtime::start` (the headless loop builds its own table),
// so the absent case is headless.
fn rendered(world: &World) -> bool {
    world
        .resource::<concinnity_core::render::render_mode::RenderMode>()
        .is_some_and(|m| m.renders())
}

// OverlaySystem: paired with GraphicsSystem (same gate) -- it shapes the
// overlay draw list graphics submits. Scheduled first so the menu state it
// publishes gates every later system this same tick.
pub(crate) fn overlay(world: &World) -> Option<concinnity_core::render::overlay::OverlaySystem> {
    rendered(world).then(concinnity_core::render::overlay::OverlaySystem::new)
}

// SkyRotationSystem: present whenever the world declares a `SkyRotation`.
// Scheduled ahead of the transform propagation and the draw list inside
// GraphicsSystem, so the orientation the pivot, the lights and the cubemap
// samples read is this tick's.
pub(crate) fn sky_rotation(world: &World) -> Option<concinnity_core::sky::SkyRotationSystem> {
    world
        .query::<SkyRotation>()
        .next()
        .map(concinnity_core::sky::SkyRotationSystem::new)
}

// BehaviorSystem: present whenever the world declares any `Behavior`.
// Scheduled before SpawnSystem / SettingsSystem / StorySystem / AudioSystem so
// the requests its firing bodies emit are drained the same tick. Built with
// what this host lends it: the job pool and a file-backed state store.
pub(crate) fn behavior(world: &World) -> Option<concinnity_core::behavior::BehaviorSystem> {
    world
        .query::<Behavior>()
        .next()
        .map(|_| crate::behavior::build(crate::ecs::state_tree(world)))
}

// SpawnSystem: paired with GraphicsSystem (same gate) -- its churn retires and
// clones the GPU draw slots graphics owns. Scheduled before it so a despawn is
// applied before the transform push and a spawn reuses slots freed this same
// frame.
pub(crate) fn spawn(world: &World) -> Option<concinnity_core::spawn::SpawnSystem> {
    rendered(world).then(concinnity_core::spawn::SpawnSystem::new)
}

// SettingsSystem: paired with GraphicsSystem (same gate). It applies the
// settings/scene command batches against the backend graphics owns and holds
// the settings state GraphicsSystem's init resolves. Scheduled just before
// GraphicsSystem so a change lands for this frame's submit.
pub(crate) fn settings(world: &World) -> Option<concinnity_core::settings::system::SettingsSystem> {
    rendered(world).then(concinnity_core::settings::system::SettingsSystem::new)
}

// SettingsWriterSystem: present wherever SettingsSystem is, writing the store
// back to disk when a change moves its revision.
pub(crate) fn settings_writer(world: &World) -> Option<crate::settings::SettingsWriterSystem> {
    rendered(world).then(crate::settings::SettingsWriterSystem::new)
}

// StreamingSystem: paired with GraphicsSystem (same gate) -- it drives the
// streaming pools and publishes the camera-relative screen graphics draws.
// Scheduled after the cameras and before GraphicsSystem so the screen it
// publishes is this tick's pose, a chunk world's screen rebase is ready for
// this frame's submit, and any texture/mesh upload lands before it.
pub(crate) fn streaming(world: &World) -> Option<crate::gfx::streaming::system::StreamingSystem> {
    rendered(world).then(crate::gfx::streaming::system::StreamingSystem::new)
}

// GraphicsSystem: present whenever the run resolved to a rendered one.
pub(crate) fn graphics(world: &World) -> Option<crate::gfx::system::GraphicsSystem> {
    rendered(world).then(|| crate::gfx::system::GraphicsSystem::new(crate::ecs::state_tree(world)))
}

// InputSystem: paired with GraphicsSystem (same gate) -- it publishes the input
// sampled from the window backend graphics drives. Scheduled at the top of the
// tick so the snapshot is fresh for every consumer below.
pub(crate) fn input(world: &World) -> Option<crate::input::system::InputSystem> {
    rendered(world).then(crate::input::system::InputSystem::new)
}

// StatHud: present whenever the world declares a `StatHud`; built from that
// component (the HUD's TextLabel refs).
pub(crate) fn stat_hud(world: &World) -> Option<concinnity_core::hud::stat_hud::StatHudSystem> {
    world
        .query::<StatHud>()
        .next()
        .cloned()
        .map(concinnity_core::hud::stat_hud::StatHudSystem::new)
}

// DebugHud: present whenever the world declares a `DebugHud`, but only in
// developer contexts. Blobs are profile-agnostic (the build injects a
// DebugHud into every rendering world), so the running binary is the one
// place its own profile is knowable: a debug build or a `cn debug` session
// activates the HUD, a release `cn run` leaves it inert.
pub(crate) fn debug_hud(world: &World) -> Option<concinnity_core::hud::debug_hud::DebugHudSystem> {
    let dev_loop = world
        .resource::<crate::app::run::LaunchRequest>()
        .is_some_and(|launch| launch.dev_loop);
    if !(cfg!(debug_assertions) || dev_loop) {
        return None;
    }
    world
        .query::<DebugHud>()
        .next()
        .cloned()
        .map(concinnity_core::hud::debug_hud::DebugHudSystem::new)
}

// LoadingOverlaySystem: present whenever the world declares a `LoadingOverlay`;
// built from that component (its screen + element refs).
pub(crate) fn loading_overlay(
    world: &World,
) -> Option<concinnity_core::hud::loading_overlay::LoadingOverlaySystem> {
    world
        .query::<LoadingOverlay>()
        .next()
        .cloned()
        .map(concinnity_core::hud::loading_overlay::LoadingOverlaySystem::new)
}

// PhysicsSystem: present whenever the world has physics content (see
// `PhysicsSystem::config_for`), stepping on the job pool.
pub(crate) fn physics(world: &World) -> Option<concinnity_core::physics::PhysicsSystem> {
    concinnity_core::physics::PhysicsSystem::config_for(world).map(crate::physics::build)
}

// PlanetSystem: present whenever the world declares a `Planet`. Scheduled
// before the world's logic, spawns, physics and the camera controllers, so a
// move of the simulated frame lands between two ticks.
pub(crate) fn planet(world: &World) -> Option<concinnity_core::planet::PlanetSystem> {
    concinnity_core::planet::PlanetSystem::for_world(world, crate::planet::start_ground_worker)
}

// The input controllers: the first controlled `Camera3D` picks one (see
// core's `controller`).
pub(crate) fn camera3d(
    world: &World,
) -> Option<concinnity_core::controller::camera::Camera3DSystem> {
    concinnity_core::controller::camera::Camera3DSystem::for_world(world)
}

pub(crate) fn third_person(
    world: &World,
) -> Option<concinnity_core::controller::third_person::ThirdPersonSystem> {
    concinnity_core::controller::third_person::ThirdPersonSystem::for_world(world)
}

// CameraTrackSystem: present when a world declares both a `CameraTrack` and the
// `Camera3D` it drives. Sits where the input controllers sit, after physics has
// settled the tick's poses and before the listener reads the camera.
pub(crate) fn camera_track(
    world: &World,
) -> Option<concinnity_core::camera_track::CameraTrackSystem> {
    concinnity_core::camera_track::CameraTrackSystem::for_world(world)
}

// FrameReportSystem: present whenever the run was armed as a benchmark; built
// from that run (the discard, the budget, the phases and the captures). Last in
// the Late band, so the frame it records is complete before it reads it.
pub(crate) fn frame_report(
    world: &World,
) -> Option<concinnity_core::frame_report::FrameReportSystem> {
    world
        .resource::<concinnity_core::benchmark::BenchmarkRun>()
        .map(|run| {
            concinnity_core::frame_report::FrameReportSystem::new(run, |text| print!("{text}"))
        })
}

// FpsCounter: present whenever the world declares an `FpsCounter`; built from
// that component (its optional TextLabel ref).
pub(crate) fn fps_counter(
    world: &World,
) -> Option<concinnity_core::hud::fps_counter::FpsCounterSystem> {
    world
        .query::<FpsCounter>()
        .next()
        .cloned()
        .map(concinnity_core::hud::fps_counter::FpsCounterSystem::new)
}

// AnimationSystem: present whenever the world declares any `Animation` or
// `AnimationGraph`. It drains both at init and writes `SkeletonPose` each
// frame. (A graph without clips is a build error, so the second check
// only matters for hand-assembled worlds.)
pub(crate) fn animation(world: &World) -> Option<concinnity_core::animation::AnimationSystem> {
    let system = concinnity_core::animation::AnimationSystem::for_world(world)?.with_fan_out(
        |poses, sample| concinnity_host::thread::jobs::pool().parallel_for(poses, sample),
    );
    let capture = world
        .resource::<crate::app::run::LaunchRequest>()
        .is_some_and(|launch| launch.dev_loop);
    Some(if capture {
        system.capturing_sources()
    } else {
        system
    })
}

// StorySystem: present whenever the world declares a `Story` (a compiled
// story graph). It runs before AudioSystem so its page-audio requests are
// heard the same tick, and before UiInputSystem like every other event
// producer (its screen commands apply next frame).
pub(crate) fn story(world: &World) -> Option<concinnity_core::story::StorySystem> {
    let system = concinnity_core::story::StorySystem::new(world.query::<Story>().next().cloned()?);
    Some(match crate::ecs::state_tree(world) {
        Some(tree) => {
            let transient = world
                .resource::<concinnity_core::ecs::TransientSaves>()
                .is_some_and(|t| t.0);
            system.with_store(Box::new(crate::story::FileStore::new(tree, transient)))
        }
        None => system,
    })
}

// AudioSystem: present whenever the world declares any `AudioEmitter`
// (positional sound), `AudioCue` (screen-triggered sound), `Story`
// (page-triggered sound), `Behavior` with a sound node, or `PropBody` with an
// impact sound. Its init opens an audio device, so a world with none of them
// stays silent and device-free.
pub(crate) fn audio(world: &World) -> Option<crate::audio::AudioSystem> {
    let needs = world.query::<AudioEmitter>().next().is_some()
        || world.query::<AudioCue>().next().is_some()
        || world
            .query::<Story>()
            .next()
            .is_some_and(Story::plays_audio)
        || world.query::<Behavior>().any(Behavior::plays_sound)
        || world.query::<PropBody>().any(|b| b.impact_clip.is_some());
    needs.then(crate::audio::AudioSystem::new)
}

// UiInputSystem: present whenever the world declares any `HitRegion`, `Screen`,
// or `KeyBinding`. It drains all three at init.
pub(crate) fn ui_input(world: &World) -> Option<concinnity_core::ui::UiInputSystem> {
    let needs = world.query::<HitRegion>().next().is_some()
        || world.query::<Screen>().next().is_some()
        || world.query::<KeyBinding>().next().is_some();
    needs.then(concinnity_core::ui::UiInputSystem::new)
}

// TextInputSystem: present whenever the world declares any `TextInput`. It
// edits the focused field in place from the frame's typed character and
// caret keys, so it runs after InputSystem publishes `FrameInput`.
pub(crate) fn text_input(
    world: &World,
) -> Option<concinnity_core::input::text_input::TextInputSystem> {
    world
        .query::<TextInput>()
        .next()
        .map(|_| concinnity_core::input::text_input::TextInputSystem::default())
}

#[cfg(test)]
mod tests {
    use crate::ecs::SYSTEMS;
    use concinnity_core::components::{
        AudioCue, AudioEmitter, Camera3D, CameraController, PhysicsConfig, RigidBody,
    };
    use concinnity_core::ecs::Ref;
    use concinnity_core::ecs::World;

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

    // A PhysicsConfig gates the internal physics system on.
    #[test]
    fn physics_config_spawns_internal_system() {
        let mut world = World::new();
        world.add_component(PhysicsConfig::default());
        world.start(SYSTEMS).unwrap();
        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["PhysicsSystem"]);
    }

    // A RigidBody (character capsule) gates physics on, even with no config.
    #[test]
    fn rigid_body_spawns_internal_system() {
        let mut world = World::new();
        world.add_component(RigidBody::default());
        world.start(SYSTEMS).unwrap();
        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["PhysicsSystem"]);
    }

    // No physics content (no PhysicsConfig / RigidBody / PropBody) → no system.
    #[test]
    fn no_physics_content_no_system() {
        let mut world = World::new();
        world.start(SYSTEMS).unwrap();
        assert!(world.systems().is_empty());
    }

    // PhysicsSystem runs before Camera3DSystem: it consumes the camera's
    // previous-frame movement intent.
    #[test]
    fn physics_runs_before_camera_controller() {
        let mut world = World::new();
        world.add_component(PhysicsConfig::default());
        world.add_component(controlled_camera());
        world.start(SYSTEMS).unwrap();
        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["PhysicsSystem", "Camera3DSystem"]);
    }

    // An `AudioEmitter` in the world spawns the internal AudioSystem; without
    // one, no audio device is opened.
    #[test]
    fn audio_emitter_spawns_internal_system() {
        let mut world = World::new();
        world.add_component(AudioEmitter::default());
        world.start(SYSTEMS).unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["AudioSystem"]);
    }

    // No audio content means no AudioSystem (no audio device is opened).
    #[test]
    fn no_audio_emitter_means_no_system() {
        let mut world = World::new();
        world.start(SYSTEMS).unwrap();
        assert!(world.systems().is_empty());
    }

    // An `AudioCue` alone (no emitter) also spawns the audio system: a UI-only
    // world can play screen-triggered audio.
    #[test]
    fn audio_cue_spawns_internal_system() {
        let mut world = World::new();
        world.add_component(AudioCue::default());
        world.start(SYSTEMS).unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert!(names.contains(&"AudioSystem"), "{names:?}");
    }

    // The full trigger chain: the initial screen's activation (announced by
    // UiInputSystem at init) reaches the audio system, which matches the
    // screen's cue on the first step. Playback itself needs a device and a
    // compiled payload, so the test observes the match counter.
    #[test]
    fn initial_view_fires_its_cue() {
        use concinnity_core::components::{AudioCue, Screen};
        use concinnity_core::ecs::AudioClipHandle;
        use concinnity_core::ecs::asset_id::AssetId;

        let mut world = World::new();
        let screen = AssetId(90);
        // The cue references its clip by handle. Matching (screen + clip present)
        // is independent of the clip payload, so no `AudioClipTable` is needed
        // here -- the counter observes the match, not playback.
        world.push_identified(
            screen,
            Screen {
                initial: true,
                fade_in_secs: 0.0,
                ..Default::default()
            },
        );
        world.add_component(AudioCue {
            screen: Some(Ref::new(screen)),
            clip: Some(AudioClipHandle::new(0)),
            ..Default::default()
        });
        world.start(SYSTEMS).unwrap();
        world.step();

        let matched = world
            .system::<crate::audio::AudioSystem>()
            .map(|a| a.cues_matched())
            .expect("world has an AudioSystem");
        assert_eq!(matched, 1, "the initial screen's cue should have matched");
    }
}

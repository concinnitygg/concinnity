//! Gate builders for the system table (`define_systems!` in `registry`). Each
//! gate inspects the world's content and returns the constructed system when
//! its gating components are present, or `None` to leave it out of the
//! schedule. `World::start` and `World::system_manifest` both run these same
//! gates, so what the manifest reports and what `start` builds cannot drift.
//!
//! Gates construct their system, so every system constructor must stay cheap
//! and side-effect-free: the manifest probe discards the value, and anything
//! heavy (device acquisition, payload reads) belongs in `System::init`.

use concinnity_core::components::Animation;
use concinnity_core::components::AnimationGraph;
use concinnity_core::components::AudioCue;
use concinnity_core::components::AudioEmitter;
use concinnity_core::components::Behavior;
use concinnity_core::components::Camera3D;
use concinnity_core::components::CameraController;
use concinnity_core::components::CameraTrack;
use concinnity_core::components::DebugHud;
use concinnity_core::components::FpsCounter;
use concinnity_core::components::FrameReport;
use concinnity_core::components::GraphicsConfig;
use concinnity_core::components::HitRegion;
use concinnity_core::components::KeyBinding;
use concinnity_core::components::LoadingOverlay;
use concinnity_core::components::PhysicsConfig;
use concinnity_core::components::PropBody;
use concinnity_core::components::RigidBody;
use concinnity_core::components::Screen;
use concinnity_core::components::SkyRotation;
use concinnity_core::components::StatHud;
use concinnity_core::components::Story;
use concinnity_core::components::TextInput;
use concinnity_core::components::TriggerVolume;
use concinnity_core::ecs::World;
use concinnity_core::resource::SkinnedMeshTable;

// OverlaySystem: paired with GraphicsSystem (same gate) -- it shapes the
// overlay draw list graphics submits. Scheduled first so the menu state it
// publishes gates every later system this same tick.
pub(crate) fn overlay(world: &World) -> Option<crate::gfx::overlay::OverlaySystem> {
    world
        .query::<GraphicsConfig>()
        .next()
        .map(|_| crate::gfx::overlay::OverlaySystem::new())
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
// clones the GPU draw slots graphics owns. Scheduled immediately before it so
// a despawn is applied before the transform push and a spawn reuses slots
// freed this same frame.
pub(crate) fn spawn(world: &World) -> Option<crate::spawn::SpawnSystem> {
    world
        .query::<GraphicsConfig>()
        .next()
        .map(|_| crate::spawn::SpawnSystem::new())
}

// SettingsSystem: paired with GraphicsSystem (same gate) -- it applies the
// settings/scene command batches against the backend graphics owns and holds
// the settings snapshot GraphicsSystem's init resolves. Scheduled just before
// GraphicsSystem so a change lands for this frame's submit.
pub(crate) fn settings(world: &World) -> Option<crate::gfx::settings::system::SettingsSystem> {
    world
        .query::<GraphicsConfig>()
        .next()
        .map(|_| crate::gfx::settings::system::SettingsSystem::new())
}

// StreamingSystem: paired with GraphicsSystem (same gate) -- it drives the
// streaming pools and publishes the camera-relative screen graphics draws.
// Scheduled immediately before GraphicsSystem so a chunk world's screen rebase is
// ready for this frame's submit and any texture/mesh upload lands before it.
pub(crate) fn streaming(world: &World) -> Option<crate::gfx::streaming::system::StreamingSystem> {
    world
        .query::<GraphicsConfig>()
        .next()
        .map(|_| crate::gfx::streaming::system::StreamingSystem::new())
}

// GraphicsSystem: present whenever the world declares a `GraphicsConfig`
// (the render marker).
pub(crate) fn graphics(world: &World) -> Option<crate::gfx::system::GraphicsSystem> {
    world
        .query::<GraphicsConfig>()
        .next()
        .map(|_| crate::gfx::system::GraphicsSystem::new(crate::ecs::state_tree(world)))
}

// InputSystem: paired with GraphicsSystem (same gate) -- it samples the window
// backend graphics drives. Scheduled immediately after it so the snapshot is
// taken right after the draw (the OS event pump on Metal runs inside
// draw_frame) and is fresh for every consumer below.
pub(crate) fn input(world: &World) -> Option<crate::input::system::InputSystem> {
    world
        .query::<GraphicsConfig>()
        .next()
        .map(|_| crate::input::system::InputSystem::new())
}

// StatHud: present whenever the world declares a `StatHud`; built from that
// component (the HUD's TextLabel refs).
pub(crate) fn stat_hud(world: &World) -> Option<crate::hud::stat_hud::StatHudSystem> {
    world
        .query::<StatHud>()
        .next()
        .cloned()
        .map(crate::hud::stat_hud::StatHudSystem::new)
}

// DebugHud: present whenever the world declares a `DebugHud`, but only in
// developer contexts. Blobs are profile-agnostic (the build injects a
// DebugHud into every rendering world), so the running binary is the one
// place its own profile is knowable: a debug build or a `cn debug` session
// activates the HUD, a release `cn run` leaves it inert.
pub(crate) fn debug_hud(world: &World) -> Option<crate::hud::debug_hud::DebugHudSystem> {
    if !(cfg!(debug_assertions) || crate::app::dev_flags::enabled()) {
        return None;
    }
    world
        .query::<DebugHud>()
        .next()
        .cloned()
        .map(crate::hud::debug_hud::DebugHudSystem::new)
}

// LoadingOverlaySystem: present whenever the world declares a `LoadingOverlay`;
// built from that component (its screen + element refs).
pub(crate) fn loading_overlay(
    world: &World,
) -> Option<crate::hud::loading_overlay::LoadingOverlaySystem> {
    world
        .query::<LoadingOverlay>()
        .next()
        .cloned()
        .map(crate::hud::loading_overlay::LoadingOverlaySystem::new)
}

// PhysicsSystem: present whenever the world has physics content, namely a
// `PhysicsConfig` (optional floor / terrain tuning), a `RigidBody` (character
// capsule), a `PropBody` (dynamic prop), or a `TriggerVolume` (sensor
// region). Reads the `PhysicsConfig` if present, otherwise a flat-floor
// default.
pub(crate) fn physics(world: &World) -> Option<concinnity_core::physics::PhysicsSystem> {
    let needs = world
        .query::<PhysicsConfig>()
        .next()
        .is_some()
        || world.query::<RigidBody>().next().is_some()
        || world.query::<PropBody>().next().is_some()
        || world
            .query::<TriggerVolume>()
            .next()
            .is_some()
        // A skinned mesh with a character capsule needs the rig drive
        // (the CharacterRig itself is published later, by GraphicsSystem
        // init, so gate on the baked resource data).
        || world
            .resource::<SkinnedMeshTable>()
            .is_some_and(|t| t.has_capsule());
    if !needs {
        return None;
    }
    // Cook injects the config into every shipped world with physics content,
    // so the fallback covers worlds built directly (tests, the editor's
    // in-memory path).
    let config = world
        .query::<PhysicsConfig>()
        .next()
        .cloned()
        .unwrap_or_default();
    Some(crate::physics::build(config))
}

// The first controlled `Camera3D` picks the controller flavor: no `follow`
// block selects this first-person / fly controller, a `follow` block selects
// the adjacent ThirdPersonSystem entry instead (a camera never gets both). A
// `controller: null` camera opts out entirely (cutscene cameras).
pub(crate) fn camera3d(world: &World) -> Option<crate::gfx::camera_controller::Camera3DSystem> {
    let ctrl = controlled_camera(world)?;
    ctrl.follow
        .is_none()
        .then(|| crate::gfx::camera_controller::Camera3DSystem::new(ctrl))
}

// Counterpart of `camera3d`: the first controlled camera declares a `follow`
// block, so the third-person controller drives it.
pub(crate) fn third_person(world: &World) -> Option<crate::gfx::third_person::ThirdPersonSystem> {
    let ctrl = controlled_camera(world)?;
    ctrl.follow
        .is_some()
        .then(|| crate::gfx::third_person::ThirdPersonSystem::new(&ctrl))
}

// A declared CameraTrack owns the camera, so neither input controller is
// built: two systems writing the same pose would fight for it every tick.
fn controlled_camera(world: &World) -> Option<CameraController> {
    if world.query::<CameraTrack>().next().is_some() {
        return None;
    }
    world.query::<Camera3D>().find_map(|c| c.controller.clone())
}

// CameraTrackSystem: present when a world declares both a `CameraTrack` and the
// `Camera3D` it drives. Sits where the input controllers sit, after physics has
// settled the tick's poses and before the listener reads the camera.
pub(crate) fn camera_track(
    world: &World,
) -> Option<concinnity_core::camera_track::CameraTrackSystem> {
    let track = world.query::<CameraTrack>().next()?;
    world.query::<Camera3D>().next()?;
    Some(concinnity_core::camera_track::CameraTrackSystem::new(track))
}

// FrameReportSystem: present whenever the world declares a `FrameReport`;
// built from that component (the discard, the budget, and whether the end of
// the camera track ends the run). Last in the Late band, so the frame it
// records is complete before it reads it.
pub(crate) fn frame_report(world: &World) -> Option<crate::frame_report::FrameReportSystem> {
    world
        .query::<FrameReport>()
        .next()
        .map(crate::frame_report::FrameReportSystem::new)
}

// FpsCounter: present whenever the world declares an `FpsCounter`; built from
// that component (its optional TextLabel ref).
pub(crate) fn fps_counter(world: &World) -> Option<crate::hud::fps_counter::FpsCounterSystem> {
    world
        .query::<FpsCounter>()
        .next()
        .cloned()
        .map(crate::hud::fps_counter::FpsCounterSystem::new)
}

// AnimationSystem: present whenever the world declares any `Animation` or
// `AnimationGraph`. It drains both at init and writes `SkeletonPose` each
// frame. (A graph without clips is a build error, so the second check
// only matters for hand-assembled worlds.)
pub(crate) fn animation(world: &World) -> Option<crate::gfx::animation::AnimationSystem> {
    let declared = world.query::<Animation>().next().is_some()
        || world.query::<AnimationGraph>().next().is_some();
    declared.then(crate::gfx::animation::AnimationSystem::new)
}

// StorySystem: present whenever the world declares a `Story` (a compiled
// story graph). It runs before AudioSystem so its page-audio requests are
// heard the same tick, and before UiInputSystem like every other event
// producer (its screen commands apply next frame).
pub(crate) fn story(world: &World) -> Option<crate::story::StorySystem> {
    world
        .query::<Story>()
        .next()
        .cloned()
        .map(|story| crate::story::StorySystem::new(story, crate::ecs::state_tree(world)))
}

// AudioSystem: present whenever the world declares any `AudioEmitter`
// (positional sound), `AudioCue` (screen-triggered sound), `Story`
// (page-triggered sound), or `Behavior` with a sound node. Its init opens
// an audio device, so a world with none of them stays silent and device-free.
pub(crate) fn audio(world: &World) -> Option<crate::audio::AudioSystem> {
    let needs = world.query::<AudioEmitter>().next().is_some()
        || world.query::<AudioCue>().next().is_some()
        || world.query::<Story>().next().is_some_and(|s| {
            s.nodes.iter().any(|n| {
                n.choice_music.is_some()
                    || !n.choice_sounds.is_empty()
                    || n.pages
                        .iter()
                        .any(|p| p.music.is_some() || !p.sounds.is_empty())
            })
        })
        || world.query::<Behavior>().any(Behavior::plays_sound);
    if !needs {
        return None;
    }
    // The persisted volumes live in the engine's settings store; resolve them
    // here and hand them to the system so the audio crate stays free of the
    // engine's `Settings` type.
    let audio = crate::config::Settings::load(crate::ecs::state_tree(world)).audio;
    Some(crate::audio::AudioSystem::new(crate::audio::AudioVolumes {
        master: audio.master_volume,
        music: audio.music_volume,
        sfx: audio.sfx_volume,
        voice: audio.voice_volume,
    }))
}

// UiInputSystem: present whenever the world declares any `HitRegion`, `Screen`,
// or `KeyBinding`. It drains all three at init.
pub(crate) fn ui_input(world: &World) -> Option<crate::ui::UiInputSystem> {
    let needs = world.query::<HitRegion>().next().is_some()
        || world.query::<Screen>().next().is_some()
        || world.query::<KeyBinding>().next().is_some();
    needs.then(crate::ui::UiInputSystem::new)
}

// TextInputSystem: present whenever the world declares any `TextInput`. It
// edits the focused field in place from the frame's typed character and
// caret keys, so it runs after GraphicsSystem deposits `FrameInput`.
pub(crate) fn text_input(world: &World) -> Option<crate::input::text_system::TextInputSystem> {
    world
        .query::<TextInput>()
        .next()
        .map(|_| crate::input::text_system::TextInputSystem::new())
}

#[cfg(test)]
mod tests {
    use crate::ecs::SYSTEMS;
    use concinnity_core::components::{
        AudioCue, AudioEmitter, Camera3D, CameraController, PhysicsConfig, RigidBody,
    };
    use concinnity_core::ecs::World;

    fn controlled_camera() -> Camera3D {
        Camera3D {
            fov_y_degrees: 75.0,
            near: 0.05,
            far: 200.0,
            view_matrix: [[0.0; 4]; 4],
            position: [0.0, 1.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
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
        use concinnity_host::thread::asset_id::AssetId;

        let mut world = World::new();
        let screen = AssetId(90);
        // The cue references its clip by handle. Matching (screen + clip present)
        // is independent of the clip payload, so no `AudioClipTable` is needed
        // here -- the counter observes the match, not playback.
        world.add_component(Screen {
            asset_id: screen,
            initial: true,
            fade_in_secs: 0.0,
            ..Default::default()
        });
        world.add_component(AudioCue {
            screen: Some(screen),
            clip: Some(AudioClipHandle(0)),
            ..Default::default()
        });
        world.start(SYSTEMS).unwrap();
        world.step();

        let matched = world
            .systems()
            .iter()
            .find_map(|s| s.downcast_ref::<crate::audio::AudioSystem>())
            .map(|a| a.cues_matched())
            .expect("world has an AudioSystem");
        assert_eq!(matched, 1, "the initial screen's cue should have matched");
    }
}

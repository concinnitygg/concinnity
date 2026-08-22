//! Gate builders for the system table (`define_systems!` in `registry`). Each
//! gate inspects the world's content and returns the constructed system when
//! its gating components are present, or `None` to leave it out of the
//! schedule. `World::build_internal_systems` and `World::system_manifest` both
//! run these same gates, so what the manifest reports and what `start()`
//! builds cannot drift.
//!
//! Gates construct their system, so every system constructor must stay cheap
//! and side-effect-free: the manifest probe discards the value, and anything
//! heavy (device acquisition, payload reads) belongs in `System::init`.

use crate::ecs::{SystemAsset, World};

/// One row of the system table: the schedule entry `define_systems!` generates
/// per system. Table order is run order.
pub struct SystemEntry {
    /// The `SystemAsset` variant name; the system's stable display name.
    pub name: &'static str,
    /// Human-readable gate condition, for docs and CLI reporting.
    pub present_when: &'static str,
    /// Constructs the system from world content when its gate holds.
    pub gate: fn(&World) -> Option<SystemAsset>,
    /// Systems (by variant name) that must run earlier in the tick than this
    /// one, and systems this one must precede. Validated against table order
    /// at schedule build: the table stays the one execution order, and an edge
    /// that contradicts it is a startup panic, not a silent reorder.
    pub after: &'static [&'static str],
    /// Systems this one must run before.
    pub before: &'static [&'static str],
}

// OverlaySystem: paired with GraphicsSystem (same gate) -- it shapes the
// overlay draw list graphics submits. Scheduled first so the menu state it
// publishes gates every later system this same tick.
pub(crate) fn overlay(world: &World) -> Option<SystemAsset> {
    world
        .query::<crate::assets::GraphicsConfig>()
        .next()
        .map(|_| crate::gfx::overlay::OverlaySystem::new().into())
}

// BehaviorSystem: present whenever the world declares any `Behavior`.
// Scheduled before SpawnSystem / SettingsSystem / StorySystem / AudioSystem so
// the requests its firing bodies emit are drained the same tick.
pub(crate) fn behavior(world: &World) -> Option<SystemAsset> {
    world
        .query::<crate::assets::Behavior>()
        .next()
        .map(|_| crate::behavior::BehaviorSystem::new().into())
}

// SpawnSystem: paired with GraphicsSystem (same gate) -- its churn retires and
// clones the GPU draw slots graphics owns. Scheduled immediately before it so
// a despawn is applied before the transform push and a spawn reuses slots
// freed this same frame.
pub(crate) fn spawn(world: &World) -> Option<SystemAsset> {
    world
        .query::<crate::assets::GraphicsConfig>()
        .next()
        .map(|_| crate::spawn::SpawnSystem::new().into())
}

// SettingsSystem: paired with GraphicsSystem (same gate) -- it applies the
// settings/scene command batches against the backend graphics owns and holds
// the settings snapshot GraphicsSystem's init resolves. Scheduled just before
// GraphicsSystem so a change lands for this frame's submit.
pub(crate) fn settings(world: &World) -> Option<SystemAsset> {
    world
        .query::<crate::assets::GraphicsConfig>()
        .next()
        .map(|_| crate::gfx::settings_system::SettingsSystem::new().into())
}

// StreamingSystem: paired with GraphicsSystem (same gate) -- it drives the
// streaming pools and publishes the camera-relative screen graphics draws.
// Scheduled immediately before GraphicsSystem so a chunk world's screen rebase is
// ready for this frame's submit and any texture/mesh upload lands before it.
pub(crate) fn streaming(world: &World) -> Option<SystemAsset> {
    world
        .query::<crate::assets::GraphicsConfig>()
        .next()
        .map(|_| crate::gfx::streaming_system::StreamingSystem::new().into())
}

// GraphicsSystem: present whenever the world declares a `GraphicsConfig`
// (the render marker).
pub(crate) fn graphics(world: &World) -> Option<SystemAsset> {
    world
        .query::<crate::assets::GraphicsConfig>()
        .next()
        .map(|_| crate::gfx::graphics_system::GraphicsSystem::new().into())
}

// InputSystem: paired with GraphicsSystem (same gate) -- it samples the window
// backend graphics drives. Scheduled immediately after it so the snapshot is
// taken right after the draw (the OS event pump on Metal runs inside
// draw_frame) and is fresh for every consumer below.
pub(crate) fn input(world: &World) -> Option<SystemAsset> {
    world
        .query::<crate::assets::GraphicsConfig>()
        .next()
        .map(|_| crate::gfx::input_system::InputSystem::new().into())
}

// StatHud: present whenever the world declares a `StatHud`; built from that
// component (the HUD's TextLabel refs).
pub(crate) fn stat_hud(world: &World) -> Option<SystemAsset> {
    world
        .query::<crate::assets::StatHud>()
        .next()
        .cloned()
        .map(|cfg| crate::hud::stat_hud::StatHudSystem::new(cfg).into())
}

// DebugHud: present whenever the world declares a `DebugHud`, but only in
// developer contexts. Blobs are profile-agnostic (the build injects a
// DebugHud into every rendering world), so the running binary is the one
// place its own profile is knowable: a debug build or a `cn debug` session
// activates the HUD, a release `cn run` leaves it inert.
pub(crate) fn debug_hud(world: &World) -> Option<SystemAsset> {
    if !(cfg!(debug_assertions) || crate::app::dev_flags::enabled()) {
        return None;
    }
    world
        .query::<crate::assets::DebugHud>()
        .next()
        .cloned()
        .map(|cfg| crate::hud::debug_hud::DebugHudSystem::new(cfg).into())
}

// LoadingOverlaySystem: present whenever the world declares a `LoadingOverlay`;
// built from that component (its screen + element refs).
pub(crate) fn loading_overlay(world: &World) -> Option<SystemAsset> {
    world
        .query::<crate::assets::LoadingOverlay>()
        .next()
        .cloned()
        .map(|cfg| crate::hud::loading_overlay::LoadingOverlaySystem::new(cfg).into())
}

// PhysicsSystem: present whenever the world has physics content, namely a
// `PhysicsConfig` (optional floor / terrain tuning), a `RigidBody` (character
// capsule), a `PropBody` (dynamic prop), or a `TriggerVolume` (sensor
// region). Reads the `PhysicsConfig` if present, otherwise a flat-floor
// default.
pub(crate) fn physics(world: &World) -> Option<SystemAsset> {
    let needs = world
        .query::<crate::assets::PhysicsConfig>()
        .next()
        .is_some()
        || world.query::<crate::assets::RigidBody>().next().is_some()
        || world.query::<crate::assets::PropBody>().next().is_some()
        || world
            .query::<crate::assets::TriggerVolume>()
            .next()
            .is_some()
        // A skinned mesh with a character capsule needs the rig drive
        // (the CharacterRig itself is published later, by GraphicsSystem
        // init, so gate on the baked resource data).
        || world
            .resources
            .get::<crate::resource::SkinnedMeshTable>()
            .is_some_and(|t| t.has_capsule());
    if !needs {
        return None;
    }
    let config = world
        .query::<crate::assets::PhysicsConfig>()
        .next()
        .cloned()
        .unwrap_or_default();
    Some(concinnity_physics::PhysicsSystem::new(config).into())
}

// The first controlled `Camera3D` picks the controller flavor: no `follow`
// block selects this first-person / fly controller, a `follow` block selects
// the adjacent ThirdPersonSystem entry instead (a camera never gets both). A
// `controller: null` camera opts out entirely (cutscene cameras).
pub(crate) fn camera3d(world: &World) -> Option<SystemAsset> {
    let ctrl = controlled_camera(world)?;
    ctrl.follow
        .is_none()
        .then(|| crate::gfx::camera_controller::Camera3DSystem::new(ctrl).into())
}

// Counterpart of `camera3d`: the first controlled camera declares a `follow`
// block, so the third-person controller drives it.
pub(crate) fn third_person(world: &World) -> Option<SystemAsset> {
    let ctrl = controlled_camera(world)?;
    ctrl.follow
        .is_some()
        .then(|| crate::gfx::third_person::ThirdPersonSystem::new(&ctrl).into())
}

fn controlled_camera(world: &World) -> Option<crate::assets::CameraController> {
    world
        .query::<crate::assets::Camera3D>()
        .find_map(|c| c.controller.clone())
}

// FpsCounter: present whenever the world declares an `FpsCounter`; built from
// that component (its optional TextLabel ref).
pub(crate) fn fps_counter(world: &World) -> Option<SystemAsset> {
    world
        .query::<crate::assets::FpsCounter>()
        .next()
        .cloned()
        .map(|cfg| crate::hud::fps_counter::FpsCounterSystem::new(cfg).into())
}

// AnimationSystem: present whenever the world declares any `Animation` or
// `AnimationGraph`. It drains both at init and writes `SkeletonPose` each
// frame. (A graph without clips is a build error, so the second check
// only matters for hand-assembled worlds.)
pub(crate) fn animation(world: &World) -> Option<SystemAsset> {
    let declared = world.query::<crate::assets::Animation>().next().is_some()
        || world
            .query::<crate::assets::AnimationGraph>()
            .next()
            .is_some();
    declared.then(|| crate::gfx::animation::AnimationSystem::new().into())
}

// StorySystem: present whenever the world declares a `Story` (a compiled
// story graph). It runs before AudioSystem so its page-audio requests are
// heard the same tick, and before UiInputSystem like every other event
// producer (its screen commands apply next frame).
pub(crate) fn story(world: &World) -> Option<SystemAsset> {
    world
        .query::<crate::assets::Story>()
        .next()
        .cloned()
        .map(|story| crate::story::StorySystem::new(story).into())
}

// AudioSystem: present whenever the world declares any `AudioEmitter`
// (positional sound), `AudioCue` (screen-triggered sound), `Story`
// (page-triggered sound), or `Behavior` with a sound node. Its init opens
// an audio device, so a world with none of them stays silent and device-free.
pub(crate) fn audio(world: &World) -> Option<SystemAsset> {
    let needs = world
        .query::<crate::assets::AudioEmitter>()
        .next()
        .is_some()
        || world.query::<crate::assets::AudioCue>().next().is_some()
        || world
            .query::<crate::assets::Story>()
            .next()
            .is_some_and(|s| {
                s.nodes.iter().any(|n| {
                    n.choice_music.is_some()
                        || !n.choice_sounds.is_empty()
                        || n.pages
                            .iter()
                            .any(|p| p.music.is_some() || !p.sounds.is_empty())
                })
            })
        || world
            .query::<crate::assets::Behavior>()
            .any(crate::assets::Behavior::plays_sound);
    if !needs {
        return None;
    }
    // The persisted volumes live in the engine's settings store; resolve them
    // here and hand them to the system so the audio crate stays free of the
    // engine's `Settings` type.
    let audio = crate::config::Settings::load().audio;
    Some(
        concinnity_audio::AudioSystem::new(concinnity_audio::AudioVolumes {
            master: audio.master_volume,
            music: audio.music_volume,
            sfx: audio.sfx_volume,
            voice: audio.voice_volume,
        })
        .into(),
    )
}

// UiInputSystem: present whenever the world declares any `HitRegion`, `Screen`,
// or `KeyBinding`. It drains all three at init.
pub(crate) fn ui_input(world: &World) -> Option<SystemAsset> {
    let needs = world.query::<crate::assets::HitRegion>().next().is_some()
        || world.query::<crate::assets::Screen>().next().is_some()
        || world.query::<crate::assets::KeyBinding>().next().is_some();
    needs.then(|| crate::ui::UiInputSystem::new().into())
}

// TextInputSystem: present whenever the world declares any `TextInput`. It
// edits the focused field in place from the frame's typed character and
// caret keys, so it runs after GraphicsSystem deposits `FrameInput`.
pub(crate) fn text_input(world: &World) -> Option<SystemAsset> {
    world
        .query::<crate::assets::TextInput>()
        .next()
        .map(|_| crate::text_input_system::TextInputSystem::new().into())
}

#[cfg(test)]
mod tests {
    use crate::assets::{Camera3D, CameraController, PhysicsConfig, RigidBody};
    use crate::ecs::World;

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
        world.start().unwrap();
        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["PhysicsSystem"]);
    }

    // A RigidBody (character capsule) gates physics on, even with no config.
    #[test]
    fn rigid_body_spawns_internal_system() {
        let mut world = World::new();
        world.add_component(RigidBody::default());
        world.start().unwrap();
        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["PhysicsSystem"]);
    }

    // No physics content (no PhysicsConfig / RigidBody / PropBody) → no system.
    #[test]
    fn no_physics_content_no_system() {
        let mut world = World::new();
        world.start().unwrap();
        assert!(world.systems().is_empty());
    }

    // PhysicsSystem runs before Camera3DSystem: it consumes the camera's
    // previous-frame movement intent.
    #[test]
    fn physics_runs_before_camera_controller() {
        let mut world = World::new();
        world.add_component(PhysicsConfig::default());
        world.add_component(controlled_camera());
        world.start().unwrap();
        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["PhysicsSystem", "Camera3DSystem"]);
    }

    // An `AudioEmitter` in the world spawns the internal AudioSystem; without
    // one, no audio device is opened.
    #[test]
    fn audio_emitter_spawns_internal_system() {
        let mut world = World::new();
        world.add_component(crate::assets::AudioEmitter::default());
        world.start().unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["AudioSystem"]);
    }

    // No audio content means no AudioSystem (no audio device is opened).
    #[test]
    fn no_audio_emitter_means_no_system() {
        let mut world = World::new();
        world.start().unwrap();
        assert!(world.systems().is_empty());
    }

    // An `AudioCue` alone (no emitter) also spawns the audio system: a UI-only
    // world can play screen-triggered audio.
    #[test]
    fn audio_cue_spawns_internal_system() {
        let mut world = World::new();
        world.add_component(crate::assets::AudioCue::default());
        world.start().unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert!(names.contains(&"AudioSystem"), "{names:?}");
    }

    // The full trigger chain: the initial screen's activation (announced by
    // UiInputSystem at init) reaches the audio system, which matches the
    // screen's cue on the first step. Playback itself needs a device and a
    // compiled payload, so the test observes the match counter.
    #[test]
    fn initial_view_fires_its_cue() {
        use crate::assets::{AudioCue, Screen};
        use crate::ecs::AudioClipHandle;
        use crate::ecs::asset_id::AssetId;

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
        world.start().unwrap();
        world.step();

        let matched = world
            .systems()
            .iter()
            .find_map(|s| match s {
                crate::ecs::SystemAsset::AudioSystem(a) => Some(a.cues_matched()),
                _ => None,
            })
            .expect("world has an AudioSystem");
        assert_eq!(matched, 1, "the initial screen's cue should have matched");
    }
}

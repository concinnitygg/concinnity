// src/ecs/schedule.rs
//
// Gate builders for the system table (`define_systems!` in `registry`). Each
// gate inspects the world's content and returns the constructed system when
// its gating components are present, or `None` to leave it out of the
// schedule. `World::build_internal_systems` and `World::system_manifest` both
// run these same gates, so what the manifest reports and what `start()`
// builds cannot drift.
//
// Gates construct their system, so every system constructor must stay cheap
// and side-effect-free: the manifest probe discards the value, and anything
// heavy (device acquisition, payload reads) belongs in `System::init`.

use crate::ecs::{SystemAsset, World};

// One row of the system table: the schedule entry `define_systems!` generates
// per system. Table order is run order.
pub struct SystemEntry {
    // The `SystemAsset` variant name; the system's stable display name.
    pub name: &'static str,
    // Human-readable gate condition, for docs and CLI reporting.
    pub present_when: &'static str,
    // Constructs the system from world content when its gate holds.
    pub gate: fn(&World) -> Option<SystemAsset>,
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
// streaming pools and publishes the camera-relative view graphics draws.
// Scheduled immediately before GraphicsSystem so a chunk world's view rebase is
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

// PhysicsSystem: present whenever the world has physics content, namely a
// `PhysicsConfig` (optional floor / terrain tuning), a `RigidBody` (character
// capsule), or a `PropBody` (dynamic prop). Reads the `PhysicsConfig` if
// present, otherwise a flat-floor default.
pub(crate) fn physics(world: &World) -> Option<SystemAsset> {
    let needs = world
        .query::<crate::assets::PhysicsConfig>()
        .next()
        .is_some()
        || world.query::<crate::assets::RigidBody>().next().is_some()
        || world.query::<crate::assets::PropBody>().next().is_some()
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
    Some(crate::physics::system::PhysicsSystem::new(config).into())
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
// `AnimGraph`. It drains both at init and writes `SkeletonPose` each
// frame. (A graph without clips is a build error, so the second check
// only matters for hand-assembled worlds.)
pub(crate) fn animation(world: &World) -> Option<SystemAsset> {
    let declared = world.query::<crate::assets::Animation>().next().is_some()
        || world.query::<crate::assets::AnimGraph>().next().is_some();
    declared.then(|| crate::gfx::animation::AnimationSystem::new().into())
}

// StorySystem: present whenever the world declares a `Story` (a compiled
// story graph). It runs before AudioSystem so its page-audio requests are
// heard the same tick, and before UiInputSystem like every other event
// producer (its view commands apply next frame).
pub(crate) fn story(world: &World) -> Option<SystemAsset> {
    world
        .query::<crate::assets::Story>()
        .next()
        .cloned()
        .map(|story| crate::story::StorySystem::new(story).into())
}

// AudioSystem: present whenever the world declares any `AudioEmitter`
// (positional sound), `AudioCue` (view-triggered sound), or `Story`
// (page-triggered sound). Its init opens an audio device, so a world with
// none of them stays silent and device-free.
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
            });
    needs.then(|| crate::audio::system::AudioSystem::new().into())
}

// UiInputSystem: present whenever the world declares any `HitRegion`, `View`,
// or `KeyBinding`. It drains all three at init.
pub(crate) fn ui_input(world: &World) -> Option<SystemAsset> {
    let needs = world.query::<crate::assets::HitRegion>().next().is_some()
        || world.query::<crate::assets::View>().next().is_some()
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

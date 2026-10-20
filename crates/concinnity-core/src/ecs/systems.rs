// This crate's systems as table entries, written once. A host's table lists
// these same entries among its own systems, so a core system's phase, edges,
// gate and description are one definition in every table. `HEADLESS_SYSTEMS`
// is the table for a world with no host beyond this crate.
//
// Edges here name only this crate's systems; a host's own entries declare
// their edges to these. A schedule ignores an edge to a system its table does
// not hold, so the headless table takes the same entries as a rendering host.

use alloc::boxed::Box;

use crate::animation::AnimationSystem;
use crate::audio::out::AudioOut;
use crate::audio::system::AudioSystem;
use crate::behavior::{BehaviorStore, BehaviorSystem};
use crate::camera_track::CameraTrackSystem;
use crate::controller::camera::Camera3DSystem;
use crate::controller::third_person::ThirdPersonSystem;
use crate::ecs::{Phase, System, SystemEntry, SystemTable, World, boxed};
use crate::fanout::{Fanout, Inline, Lent};
use crate::frame_report::FrameReportSystem;
use crate::hud::debug_hud::DebugHudSystem;
use crate::hud::fps_counter::FpsCounterSystem;
use crate::hud::loading_overlay::LoadingOverlaySystem;
use crate::hud::stat_hud::StatHudSystem;
use crate::input::system::{InputSystem, PadSource};
use crate::input::text_input::TextInputSystem;
use crate::launch::LaunchRequest;
use crate::physics::PhysicsSystem;
use crate::planet::{PlanetSystem, StartBuilder, inline_ground};
use crate::render::overlay::OverlaySystem;
use crate::render::streaming::pools::StreamingSystem;
use crate::settings::system::{SettingsStore, SettingsSystem};
use crate::sky::SkyRotationSystem;
use crate::spawn::SpawnSystem;
use crate::story::{StoryStore, StorySystem};
use crate::ui::UiInputSystem;

/// What a host lends this crate's systems when their gates build them: always
/// a pool (a host with no threads lends [`Inline`]), and seams that default to
/// none, so a system persists nothing and reports nowhere.
pub trait SystemHost: 'static {
    /// The thread pool the host lends.
    type Pool: Fanout;

    /// The host's pool, as each schedule mode picks it.
    const POOL: Lent<Self::Pool>;

    /// Starts a planet's ground builder.
    const GROUND: StartBuilder = inline_ground;

    /// Where a world's behavior state is kept, or `None` to persist nothing.
    fn behavior_store(_world: &World) -> Option<Box<dyn BehaviorStore>> {
        None
    }

    /// Opens the host's audio output, or `None` when it has no device.
    fn audio_out() -> Option<Box<dyn AudioOut>> {
        None
    }

    /// The host's gamepads, or `None` for a host with none.
    fn pads() -> Option<Box<dyn PadSource>> {
        None
    }

    /// Where the settings store is kept, or `None` to apply changes for the
    /// run only.
    fn settings_store(_world: &World) -> Option<Box<dyn SettingsStore>> {
        None
    }

    /// Where a world's story saves are kept, or `None` to save nothing.
    fn story_store(_world: &World) -> Option<Box<dyn StoryStore>> {
        None
    }

    /// Writes a benchmark's frame report.
    fn report(_text: &str) {}
}

/// The table entry of each of this crate's systems, over the seams `H` lends.
pub struct CoreSystems<H>(core::marker::PhantomData<H>);

impl<H: SystemHost> CoreSystems<H> {
    /// The overlay draw list and the menu state that gates the rest of the
    /// tick.
    pub const OVERLAY: SystemEntry = SystemEntry {
        name: "OverlaySystem",
        present_when: "the world runs with a window",
        phase: Phase::Early,
        gate: |world| boxed(OverlaySystem::for_world(world)),
        after: &[],
        before: &[
            "BehaviorSystem",
            "SpawnSystem",
            "PhysicsSystem",
            "AnimationSystem",
        ],
    };

    /// This tick's input, right after the overlay: the tick consumes the input
    /// sampled before it began, so everything downstream sees this tick's.
    pub const INPUT: SystemEntry = SystemEntry {
        name: "InputSystem",
        present_when: "the world runs with a window",
        phase: Phase::Early,
        gate: |world| {
            let system = InputSystem::for_world(world)?;
            boxed(Some(match H::pads() {
                Some(pads) => system.with_pads(pads),
                None => system,
            }))
        },
        after: &["OverlaySystem"],
        before: &["BehaviorSystem"],
    };

    /// The sky's orientation, ahead of everything that reads it.
    pub const SKY_ROTATION: SystemEntry = SystemEntry {
        name: "SkyRotationSystem",
        present_when: "the world declares a SkyRotation",
        phase: Phase::Early,
        gate: |world| boxed(SkyRotationSystem::for_world(world)),
        after: &["OverlaySystem"],
        before: &[],
    };

    /// A planet world's frame move, between two ticks.
    pub const PLANET: SystemEntry = SystemEntry {
        name: "PlanetSystem",
        present_when: "the world declares a Planet",
        phase: Phase::Early,
        gate: |world| boxed(PlanetSystem::for_world(world, H::GROUND)),
        after: &[],
        before: &[
            "BehaviorSystem",
            "SpawnSystem",
            "PhysicsSystem",
            "Camera3DSystem",
            "ThirdPersonSystem",
            "CameraTrackSystem",
        ],
    };

    /// The world's behaviors, whose requests drain later this tick.
    pub const BEHAVIOR: SystemEntry = SystemEntry {
        name: "BehaviorSystem",
        present_when: "the world declares any Behavior",
        phase: Phase::Logic,
        gate: behavior::<H>,
        after: &["OverlaySystem"],
        before: &["SpawnSystem", "SettingsSystem", "StorySystem"],
    };

    /// The simulation, ahead of the cameras that read its poses.
    pub const PHYSICS: SystemEntry = SystemEntry {
        name: "PhysicsSystem",
        present_when: "the world declares a PhysicsConfig, RigidBody, PropBody, or TriggerVolume, or a skinned mesh bakes a character capsule",
        phase: Phase::Late,
        gate: |world| boxed(PhysicsSystem::for_world(world).map(|s| s.with_pool(H::POOL))),
        after: &["OverlaySystem"],
        before: &["Camera3DSystem", "ThirdPersonSystem"],
    };

    /// The fly / walk camera controller.
    pub const CAMERA3D: SystemEntry = SystemEntry {
        name: "Camera3DSystem",
        present_when: "the first controlled Camera3D has no follow block",
        phase: Phase::Late,
        gate: |world| boxed(Camera3DSystem::for_world(world)),
        after: &["PhysicsSystem"],
        before: &[],
    };

    /// The follow camera controller.
    pub const THIRD_PERSON: SystemEntry = SystemEntry {
        name: "ThirdPersonSystem",
        present_when: "the first controlled Camera3D has a follow block",
        phase: Phase::Late,
        gate: |world| boxed(ThirdPersonSystem::for_world(world)),
        after: &["PhysicsSystem"],
        before: &[],
    };

    /// A declared camera path, in place of the input controllers.
    pub const CAMERA_TRACK: SystemEntry = SystemEntry {
        name: "CameraTrackSystem",
        present_when: "the world declares a CameraTrack and the Camera3D it drives",
        phase: Phase::Late,
        gate: |world| boxed(CameraTrackSystem::for_world(world)),
        after: &["PhysicsSystem"],
        before: &[],
    };

    /// Skeletal playback.
    pub const ANIMATION: SystemEntry = SystemEntry {
        name: "AnimationSystem",
        present_when: "the world declares any Animation or AnimationGraph",
        phase: Phase::Late,
        gate: animation::<H>,
        after: &["OverlaySystem"],
        before: &[],
    };

    /// The story graph walk.
    pub const STORY: SystemEntry = SystemEntry {
        name: "StorySystem",
        present_when: "the world declares a Story",
        phase: Phase::Late,
        gate: |world| {
            let system = StorySystem::for_world(world)?;
            boxed(Some(match H::story_store(world) {
                Some(store) => system.with_store(store),
                None => system,
            }))
        },
        after: &["BehaviorSystem"],
        before: &[],
    };

    /// Positional sound, screen cues, story and behavior plays and impact
    /// one-shots. The listener reads the camera, and sounds requested this tick
    /// are heard this tick.
    pub const AUDIO: SystemEntry = SystemEntry {
        name: "AudioSystem",
        present_when: "the world declares any AudioEmitter, AudioCue, a Story page/choice with audio, a Behavior with a sound node, or a PropBody with an impact sound",
        phase: Phase::Late,
        gate: |world| boxed(AudioSystem::for_world(world, H::audio_out)),
        after: &[
            "BehaviorSystem",
            "Camera3DSystem",
            "ThirdPersonSystem",
            "CameraTrackSystem",
            "StorySystem",
        ],
        before: &[],
    };

    /// Editing of the focused text field.
    pub const TEXT_INPUT: SystemEntry = SystemEntry {
        name: "TextInputSystem",
        present_when: "the world declares any TextInput",
        phase: Phase::Late,
        gate: |world| boxed(TextInputSystem::for_world(world)),
        after: &[],
        before: &[],
    };

    /// The stat HUD.
    pub const STAT_HUD: SystemEntry = SystemEntry {
        name: "StatHudSystem",
        present_when: "the world declares a StatHud",
        phase: Phase::Late,
        gate: |world| boxed(StatHudSystem::for_world(world)),
        after: &[],
        before: &[],
    };

    /// The debug HUD, in developer contexts only: a debug build or a
    /// developer session.
    pub const DEBUG_HUD: SystemEntry = SystemEntry {
        name: "DebugHudSystem",
        present_when: "the world declares a DebugHud and the build is a debug build or the run is a developer session",
        phase: Phase::Late,
        gate: |world| {
            let shown = cfg!(debug_assertions) || dev_loop(world);
            boxed(DebugHudSystem::for_world(world).filter(|_| shown))
        },
        after: &[],
        before: &[],
    };

    /// The frame-rate counter.
    pub const FPS_COUNTER: SystemEntry = SystemEntry {
        name: "FpsCounterSystem",
        present_when: "the world declares an FpsCounter",
        phase: Phase::Late,
        gate: |world| boxed(FpsCounterSystem::for_world(world)),
        after: &[],
        before: &[],
    };

    /// The spawn / despawn drain.
    pub const SPAWN: SystemEntry = SystemEntry {
        name: "SpawnSystem",
        present_when: "the world runs with a window",
        phase: Phase::PreRender,
        gate: |world| boxed(SpawnSystem::for_world(world)),
        after: &["BehaviorSystem"],
        before: &[],
    };

    /// The settings and scene command drain.
    pub const SETTINGS: SystemEntry = SystemEntry {
        name: "SettingsSystem",
        present_when: "the world runs with a window",
        phase: Phase::PreRender,
        gate: |world| {
            let system = SettingsSystem::for_world(world)?;
            boxed(Some(match H::settings_store(world) {
                Some(store) => system.with_store(store),
                None => system,
            }))
        },
        after: &[],
        before: &[],
    };

    /// The streaming pools, after the cameras: the camera-relative view it
    /// publishes is this tick's pose, and the loading overlay reads the
    /// residency it publishes.
    pub const STREAMING: SystemEntry = SystemEntry {
        name: "StreamingSystem",
        present_when: "the world runs with a window",
        phase: Phase::PreRender,
        gate: |world| boxed(StreamingSystem::for_world(world)),
        after: &["Camera3DSystem", "ThirdPersonSystem", "CameraTrackSystem"],
        before: &["LoadingOverlaySystem"],
    };

    /// The loading screen, ahead of the UI that applies its screen commands.
    pub const LOADING_OVERLAY: SystemEntry = SystemEntry {
        name: "LoadingOverlaySystem",
        present_when: "the world declares a LoadingOverlay",
        phase: Phase::PreRender,
        gate: |world| boxed(LoadingOverlaySystem::for_world(world)),
        after: &[],
        before: &["UiInputSystem"],
    };

    /// UI hit testing, navigation and screen commands.
    pub const UI_INPUT: SystemEntry = SystemEntry {
        name: "UiInputSystem",
        present_when: "the world declares any HitRegion, Screen, or KeyBinding",
        phase: Phase::PreRender,
        gate: |world| boxed(UiInputSystem::for_world(world)),
        after: &["LoadingOverlaySystem"],
        before: &[],
    };

    /// A benchmark's report, last: it records the frame that just went.
    pub const FRAME_REPORT: SystemEntry = SystemEntry {
        name: "FrameReportSystem",
        present_when: "the run is benchmarked",
        phase: Phase::PostRender,
        gate: |world| boxed(FrameReportSystem::for_world(world, H::report)),
        after: &["CameraTrackSystem"],
        before: &[],
    };

    /// Every entry above, in run order. A host's table keeps this relative
    /// order.
    pub const ALL: &'static [SystemEntry] = &[
        Self::OVERLAY,
        Self::INPUT,
        Self::SKY_ROTATION,
        Self::PLANET,
        Self::BEHAVIOR,
        Self::PHYSICS,
        Self::CAMERA3D,
        Self::THIRD_PERSON,
        Self::CAMERA_TRACK,
        Self::ANIMATION,
        Self::STORY,
        Self::AUDIO,
        Self::TEXT_INPUT,
        Self::STAT_HUD,
        Self::DEBUG_HUD,
        Self::FPS_COUNTER,
        Self::SPAWN,
        Self::SETTINGS,
        Self::STREAMING,
        Self::LOADING_OVERLAY,
        Self::UI_INPUT,
        Self::FRAME_REPORT,
    ];
}

// Whether the launch asked for a development session: it shows the debug HUD
// in any build and captures animation sources for reload.
fn dev_loop(world: &World) -> bool {
    world
        .resource::<LaunchRequest>()
        .is_some_and(|launch| launch.dev_loop)
}

fn behavior<H: SystemHost>(world: &World) -> Option<Box<dyn System>> {
    let mut system = BehaviorSystem::for_world(world)?.with_pool(H::POOL);
    if let Some(store) = H::behavior_store(world) {
        system = system.with_store(store);
    }
    boxed(Some(system))
}

fn animation<H: SystemHost>(world: &World) -> Option<Box<dyn System>> {
    let mut system = AnimationSystem::for_world(world)?.with_pool(H::POOL);
    if dev_loop(world) {
        system = system.capturing_sources();
    }
    boxed(Some(system))
}

// No host: every unit on the calling thread, every seam at its default.
struct Headless;

static INLINE: Inline = Inline;

impl SystemHost for Headless {
    type Pool = Inline;

    const POOL: Lent<Inline> = Lent::new(|_| &INLINE);
}

type Core = CoreSystems<Headless>;

/// The simulation systems a world runs with no host beyond this crate, in run
/// order. What [`App`](crate::App) starts a headless world against.
pub const HEADLESS_SYSTEMS: &SystemTable = &SystemTable {
    init_first: &[],
    entries: &[
        Core::SKY_ROTATION,
        Core::PLANET,
        Core::BEHAVIOR,
        Core::PHYSICS,
        Core::CAMERA3D,
        Core::THIRD_PERSON,
        Core::CAMERA_TRACK,
        Core::ANIMATION,
    ],
    // The headless tier renders nothing, so the completion pass leaves it with
    // the one default that is not a rendering concern: the PhysicsConfig its
    // simulation already runs on.
    complete_world: Some(crate::defaults::run),
    before_init: Some(crate::ecs::decompose::run),
    prepare_events: None,
};

#[cfg(test)]
mod tests;

// src/ecs/registry.rs
//
// The system table: the one place a system is registered and the one schedule
// document. `define_systems!` generates the runtime `SystemAsset` enum and the
// `SYSTEMS` manifest from it; table order is run order. Every system is
// internal: it has no declarable asset and carries no discriminant.
// `World::build_internal_systems` runs each entry's gate against the world's
// content and pushes the systems the gates return, in table order. To add a
// system: implement `System` on it, write its gate in `schedule`, and add one
// entry here in its run position.
//
// The order encodes the cross-system constraints:
//   * OverlaySystem first: it shapes the overlay draw list (from the HUD
//     content written last tick) and publishes the menu state (`MenuActive`)
//     that gates simulation, input, and the draw this same tick.
//   * SpawnSystem before GraphicsSystem: a despawned entity must be gone from
//     the transform push (it contributes nothing to any pass this frame), and
//     a spawn reuses draw slots freed this same frame.
//   * SettingsSystem before GraphicsSystem: a SettingCommand applied here lands
//     on the backend before this frame's submit (visible the same frame), and
//     a SceneCommand's jump primes the reel GraphicsSystem ticks below.
//   * GraphicsSystem right after them: makes payloads resident, uploads
//     transforms, and submits the frame (consuming the overlay build).
//   * InputSystem immediately after GraphicsSystem: on Metal the OS event pump
//     runs inside draw_frame, so sampling right after the draw snapshots the
//     freshest events. It deposits `FrameInput` (drained by no one -- the next
//     sample replaces it) for every consumer below.
//   * PhysicsSystem before the camera controllers: physics consumes the
//     camera's previous-frame `desired_move` (a one-frame-lagged resolution).
//   * Camera3DSystem / ThirdPersonSystem before AudioSystem: the audio
//     listener reads the camera. The two controller entries are mutually
//     exclusive (one `Camera3D.controller` picks the flavor).
//   * StorySystem before AudioSystem: its `PlayCue` page audio is heard the
//     same tick.
// Event-carried couplings (RootMotion, GroundProbes, SettingCommand) are
// order-robust thanks to the event store's two-frame retention.

use crate::ecs::{PipelineContext, StepResult, System, schedule};

crate::define_systems! {
    OverlaySystem => crate::gfx::overlay::OverlaySystem {
        gate: schedule::overlay,
        present_when: "the world declares a GraphicsConfig",
    },
    SpawnSystem => crate::spawn::SpawnSystem {
        gate: schedule::spawn,
        present_when: "the world declares a GraphicsConfig",
    },
    SettingsSystem => crate::gfx::settings_system::SettingsSystem {
        gate: schedule::settings,
        present_when: "the world declares a GraphicsConfig",
    },
    GraphicsSystem => crate::gfx::graphics_system::GraphicsSystem {
        gate: schedule::graphics,
        present_when: "the world declares a GraphicsConfig",
    },
    InputSystem => crate::gfx::input_system::InputSystem {
        gate: schedule::input,
        present_when: "the world declares a GraphicsConfig",
    },
    StatHud => crate::hud::stat_hud::StatHudSystem {
        gate: schedule::stat_hud,
        present_when: "the world declares a StatHud",
    },
    DebugHud => crate::hud::debug_hud::DebugHudSystem {
        gate: schedule::debug_hud,
        present_when: "the world declares a DebugHud AND the binary is a debug build or a `cn debug` session",
    },
    PhysicsSystem => crate::physics::system::PhysicsSystem {
        gate: schedule::physics,
        present_when: "the world declares a PhysicsConfig, RigidBody, or PropBody, or a skinned mesh bakes a character capsule",
    },
    Camera3DSystem => crate::gfx::camera_controller::Camera3DSystem {
        gate: schedule::camera3d,
        present_when: "the first controlled Camera3D has no follow block",
    },
    ThirdPersonSystem => crate::gfx::third_person::ThirdPersonSystem {
        gate: schedule::third_person,
        present_when: "the first controlled Camera3D has a follow block",
    },
    FpsCounter => crate::hud::fps_counter::FpsCounterSystem {
        gate: schedule::fps_counter,
        present_when: "the world declares an FpsCounter",
    },
    AnimationSystem => crate::gfx::animation::AnimationSystem {
        gate: schedule::animation,
        present_when: "the world declares any Animation or AnimGraph",
    },
    StorySystem => crate::story::StorySystem {
        gate: schedule::story,
        present_when: "the world declares a Story",
    },
    AudioSystem => crate::audio::system::AudioSystem {
        gate: schedule::audio,
        present_when: "the world declares any AudioEmitter, AudioCue, or a Story page/choice with audio",
    },
    UiInputSystem => crate::ui::UiInputSystem {
        gate: schedule::ui_input,
        present_when: "the world declares any HitRegion, View, or KeyBinding",
    },
    TextInputSystem => crate::text_input_system::TextInputSystem {
        gate: schedule::text_input,
        present_when: "the world declares any TextInput",
    },
}

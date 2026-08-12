// src/ecs/registry.rs
//
// The system table: the one place a system is registered and the one schedule
// document. `define_systems!` generates the runtime `SystemAsset` enum and the
// `SYSTEMS` manifest from it; table order is run order. Every system is
// internal: it has no declarable asset and carries no discriminant.
// `World::build_internal_systems` runs each entry's gate against the world's
// content and pushes the systems the gates return, in table order. To add a
// system: implement `System` on it, write its gate in `schedule`, and add one
// entry here in its run position with its ordering edges.
//
// `after`/`before` are the cross-system ordering constraints, validated
// against table order when the world's schedule is built (see `ecs::waves`).
// Each edge's rationale:
//   * Overlay first: publishes the menu state (`MenuActive`) that gates
//     simulation, input, and the draw this same tick.
//   * Behavior before Spawn/Settings/Story/Audio: the requests its firing
//     rules emit (spawn/despawn, scene, story, audio) drain the same tick.
//   * Spawn/Settings/Streaming before Graphics: despawns leave the transform
//     push, setting and streaming ops land before this frame's submit, and
//     `CameraRelativeView` is ready for the draw.
//   * Input after Graphics: on Metal the OS event pump runs inside
//     draw_frame, so sampling right after the draw snapshots the freshest
//     events (the mailbox deposit happens in Graphics' step).
//   * LoadingOverlay after Streaming (reads the residency status published
//     this tick) and before UiInput (its screen commands apply same tick).
//   * Physics before the camera controllers: physics consumes the camera's
//     previous-frame `desired_move` (a one-frame-lagged resolution).
//   * Cameras and Story before Audio: the listener reads the camera, and a
//     `PlayCue` page audio is heard the same tick.
// Event-carried couplings (RootMotion, GroundProbes, SettingCommand) are
// order-robust thanks to the event store's two-frame retention.

use crate::ecs::{PipelineContext, StepResult, System, schedule};

crate::define_systems! {
    OverlaySystem => crate::gfx::overlay::OverlaySystem {
        gate: schedule::overlay,
        present_when: "the world declares a GraphicsConfig",
        after: [],
        before: [BehaviorSystem, SpawnSystem, GraphicsSystem, InputSystem, PhysicsSystem, AnimationSystem],
    },
    BehaviorSystem => crate::behavior::BehaviorSystem {
        gate: schedule::behavior,
        present_when: "the world declares any Behavior",
        after: [OverlaySystem],
        before: [SpawnSystem, SettingsSystem, StorySystem, AudioSystem],
    },
    SpawnSystem => crate::spawn::SpawnSystem {
        gate: schedule::spawn,
        present_when: "the world declares a GraphicsConfig",
        after: [BehaviorSystem],
        before: [GraphicsSystem],
    },
    SettingsSystem => crate::gfx::settings_system::SettingsSystem {
        gate: schedule::settings,
        present_when: "the world declares a GraphicsConfig",
        after: [],
        before: [GraphicsSystem],
    },
    StreamingSystem => crate::gfx::streaming_system::StreamingSystem {
        gate: schedule::streaming,
        present_when: "the world declares a GraphicsConfig",
        after: [],
        before: [GraphicsSystem],
    },
    GraphicsSystem => crate::gfx::graphics_system::GraphicsSystem {
        gate: schedule::graphics,
        present_when: "the world declares a GraphicsConfig",
        after: [SpawnSystem, SettingsSystem, StreamingSystem],
        before: [InputSystem],
    },
    InputSystem => crate::gfx::input_system::InputSystem {
        gate: schedule::input,
        present_when: "the world declares a GraphicsConfig",
        after: [GraphicsSystem],
        before: [],
    },
    StatHud => crate::hud::stat_hud::StatHudSystem {
        gate: schedule::stat_hud,
        present_when: "the world declares a StatHud",
        after: [],
        before: [],
    },
    DebugHud => crate::hud::debug_hud::DebugHudSystem {
        gate: schedule::debug_hud,
        present_when: "the world declares a DebugHud AND the binary is a debug build or a `cn debug` session",
        after: [],
        before: [],
    },
    LoadingOverlaySystem => crate::hud::loading_overlay::LoadingOverlaySystem {
        gate: schedule::loading_overlay,
        present_when: "the world declares a LoadingOverlay",
        after: [StreamingSystem],
        before: [UiInputSystem],
    },
    PhysicsSystem => concinnity_physics::PhysicsSystem {
        gate: schedule::physics,
        present_when: "the world declares a PhysicsConfig, RigidBody, PropBody, or TriggerVolume, or a skinned mesh bakes a character capsule",
        after: [OverlaySystem],
        before: [Camera3DSystem, ThirdPersonSystem],
    },
    Camera3DSystem => crate::gfx::camera_controller::Camera3DSystem {
        gate: schedule::camera3d,
        present_when: "the first controlled Camera3D has no follow block",
        after: [PhysicsSystem],
        before: [AudioSystem],
    },
    ThirdPersonSystem => crate::gfx::third_person::ThirdPersonSystem {
        gate: schedule::third_person,
        present_when: "the first controlled Camera3D has a follow block",
        after: [PhysicsSystem],
        before: [AudioSystem],
    },
    FpsCounter => crate::hud::fps_counter::FpsCounterSystem {
        gate: schedule::fps_counter,
        present_when: "the world declares an FpsCounter",
        after: [],
        before: [],
    },
    AnimationSystem => crate::gfx::animation::AnimationSystem {
        gate: schedule::animation,
        present_when: "the world declares any Animation or AnimGraph",
        after: [OverlaySystem],
        before: [],
    },
    StorySystem => crate::story::StorySystem {
        gate: schedule::story,
        present_when: "the world declares a Story",
        after: [BehaviorSystem],
        before: [AudioSystem],
    },
    AudioSystem => concinnity_audio::AudioSystem {
        gate: schedule::audio,
        present_when: "the world declares any AudioEmitter, AudioCue, a Story page/choice with audio, or a Behavior with a sound node",
        after: [BehaviorSystem, Camera3DSystem, ThirdPersonSystem, StorySystem],
        before: [],
    },
    UiInputSystem => crate::ui::UiInputSystem {
        gate: schedule::ui_input,
        present_when: "the world declares any HitRegion, Screen, or KeyBinding",
        after: [LoadingOverlaySystem],
        before: [],
    },
    TextInputSystem => crate::text_input_system::TextInputSystem {
        gate: schedule::text_input,
        present_when: "the world declares any TextInput",
        after: [],
        before: [],
    },
}

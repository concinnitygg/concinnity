// src/ecs/registry.rs
//
// The system table: the one place a system is registered and the one schedule
// document. `define_systems!` generates the entries from it; table order is run
// order. Every system is internal: it has no declarable asset and carries no
// discriminant. `World::start` runs each entry's gate against the world's
// content and pushes the systems the gates return, in table order. To add a
// system: implement `System` on it, write its gate in `schedule`, and add one
// entry here in its run position with its ordering edges.
//
// `after`/`before` are the cross-system ordering constraints, validated
// against table order when the world's schedule is built.
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
// Event-carried couplings (RootMotionEvent, GroundProbes, SettingCommand) are
// order-robust thanks to the event store's two-frame retention.

use crate::ecs::{PipelineContext, access_ids, decompose, schedule};

// Runs once at world start, after the gates have built the systems and before
// their `init`.
fn before_init(ctx: &mut PipelineContext) {
    // Every context touch a stepping system makes is asserted against what it
    // declared; the hook is installed before the first step can happen.
    #[cfg(debug_assertions)]
    access_ids::install_hook();
    decompose::run(ctx);
}

crate::define_systems! {
    before_init: before_init,
    prepare_events: access_ids::ensure_event_queues,

    OverlaySystem => crate::gfx::overlay::OverlaySystem {
        gate: schedule::overlay,
        present_when: "the world declares a GraphicsConfig",
        after: [],
        before: [BehaviorSystem, SpawnSystem, GraphicsSystem, InputSystem, PhysicsSystem, AnimationSystem],
    },
    BehaviorSystem => concinnity_core::behavior::BehaviorSystem {
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
    PhysicsSystem => concinnity_core::physics::PhysicsSystem {
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
        present_when: "the world declares any Animation or AnimationGraph",
        after: [OverlaySystem],
        before: [],
    },
    StorySystem => crate::story::StorySystem {
        gate: schedule::story,
        present_when: "the world declares a Story",
        after: [BehaviorSystem],
        before: [AudioSystem],
    },
    AudioSystem => crate::audio::AudioSystem {
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

#[cfg(test)]
mod tests {
    use super::SYSTEMS;
    use crate::ecs::{ComponentAsset, World};

    // The table's entries, in run order.
    const ENTRIES: &[crate::ecs::SystemEntry] = SYSTEMS.entries;

    // The overlay HUD components each gate their internal system and build in
    // the fixed schedule order (StatHud, then DebugHud, then FpsCounter).
    // DebugHud is developer-only but `cfg!(debug_assertions)` holds under test.
    #[test]
    fn hud_components_spawn_in_schedule_order() {
        use crate::components::{DebugHud, FpsCounter, StatHud};

        let mut world = World::new();
        world.add_component(FpsCounter::default());
        world.add_component(StatHud::default());
        world.add_component(DebugHud::default());
        world.start(SYSTEMS).unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["StatHud", "DebugHud", "FpsCounter"]);
    }

    // The manifest reports exactly the systems `start()` builds, in the same
    // order, for a world gating several table entries. Audio is left ungated
    // so `start()` opens no device here.
    #[test]
    fn system_manifest_matches_started_systems() {
        use crate::components::{DebugHud, FpsCounter, StatHud, Story, TextInput};

        let mut world = World::new();
        world.add_component(StatHud::default());
        world.add_component(DebugHud::default());
        world.add_component(FpsCounter::default());
        world.add_component(Story::default());
        world.add_component(TextInput::default());

        let manifest = world.system_manifest(SYSTEMS);
        world.start(SYSTEMS).unwrap();
        let built: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(manifest, built);
    }

    // Manifest names come out in table order, and every name is a real table
    // entry (the manifest is a filtered view of `SYSTEMS`, nothing else).
    #[test]
    fn system_manifest_is_a_table_order_subset() {
        use crate::components::{FpsCounter, StatHud};

        let mut world = World::new();
        world.add_component(FpsCounter::default());
        world.add_component(StatHud::default());

        let table: Vec<&str> = ENTRIES.iter().map(|e| e.name).collect();
        let manifest = world.system_manifest(SYSTEMS);
        let mut cursor = table.iter();
        for name in &manifest {
            assert!(
                cursor.any(|t| t == name),
                "'{name}' out of table order or unknown: {manifest:?}"
            );
        }
    }

    // A GraphicsConfig world gates the whole render band, and StreamingSystem
    // runs immediately before GraphicsSystem so its `CameraRelativeView` is
    // ready for that frame's submit. (Manifest-only: gating a GraphicsConfig
    // never builds a GPU, unlike `start()`.)
    #[test]
    fn streaming_runs_immediately_before_graphics() {
        let mut world = World::new();
        world.add_component(crate::components::GraphicsConfig::default());
        let manifest = world.system_manifest(SYSTEMS);
        let s = manifest
            .iter()
            .position(|n| *n == "StreamingSystem")
            .expect("StreamingSystem present for a GraphicsConfig world");
        let g = manifest
            .iter()
            .position(|n| *n == "GraphicsSystem")
            .expect("GraphicsSystem present for a GraphicsConfig world");
        assert_eq!(
            g,
            s + 1,
            "StreamingSystem is directly before GraphicsSystem: {manifest:?}"
        );
    }

    // The two camera-controller entries are mutually exclusive: the first
    // controlled camera's `follow` block picks exactly one of them.
    #[test]
    fn camera_controller_gates_are_exclusive() {
        use crate::components::{Camera3D, CameraController, FollowController};

        let mut fly_cam = Camera3D::bake(Default::default());
        fly_cam.controller = Some(CameraController::default());
        let mut fly = World::new();
        fly.add_component(fly_cam);
        assert_eq!(fly.system_manifest(SYSTEMS), ["Camera3DSystem"]);

        let mut follow_cam = Camera3D::bake(Default::default());
        follow_cam.controller = Some(CameraController {
            follow: Some(FollowController::default()),
            ..Default::default()
        });
        let mut follow = World::new();
        follow.add_component(follow_cam);
        assert_eq!(follow.system_manifest(SYSTEMS), ["ThirdPersonSystem"]);
    }

    // An audio-gating component is visible in the manifest without a device:
    // the gate probe constructs the system, and device acquisition waits for
    // `System::init`.
    #[test]
    fn audio_gate_probes_without_a_device() {
        let mut world = World::new();
        world.add_component(crate::components::AudioEmitter::default());
        assert_eq!(world.system_manifest(SYSTEMS), ["AudioSystem"]);
    }

    // A Story gates the StorySystem. An empty-node story pulls in no audio
    // device (build_audio needs a page/choice cue), so this stays device-free.
    #[test]
    fn story_component_spawns_story_system() {
        let mut world = World::new();
        world.add_component(crate::components::Story::default());
        world.start(SYSTEMS).unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["StorySystem"]);
    }

    // A fresh world holds nothing; adding a component (through either the blob
    // path or the typed one) fills it, and `start()` is what gives it systems.
    #[test]
    fn empty_world_fills_from_components_then_systems() {
        use crate::components::{FpsCounter, TextLabel};

        let mut world = World::new();
        assert!(world.is_empty());
        assert_eq!(world.component_count(), 0);
        assert_eq!(world.system_count(), 0);

        world.add(ComponentAsset::from(TextLabel::default()));
        assert!(!world.is_empty());
        assert_eq!(world.component_count(), 1);

        world.add_component(FpsCounter::default());
        world.start(SYSTEMS).unwrap();
        assert_eq!(world.system_count(), 1, "the FpsCounter gate built one");
    }

    // Every declared edge agrees with table order. The table is the one
    // execution order, so an edge that contradicts it is a schedule-build panic
    // at world start; this catches it at the document instead.
    #[test]
    fn declared_edges_respect_table_order() {
        let position = |name: &str| {
            ENTRIES
                .iter()
                .position(|e| e.name == name)
                .expect("a known entry")
        };
        for (i, entry) in ENTRIES.iter().enumerate() {
            for after in entry.after {
                assert!(
                    position(after) < i,
                    "{} runs after {after}, but the table runs {after} later",
                    entry.name,
                );
            }
            for before in entry.before {
                assert!(
                    i < position(before),
                    "{} runs before {before}, but the table runs {before} earlier",
                    entry.name,
                );
            }
        }
    }

    // Every name in a declared edge is a real table entry: a typo would
    // silently drop the constraint.
    #[test]
    fn edge_names_exist() {
        for entry in ENTRIES {
            for name in entry.after.iter().chain(entry.before) {
                assert!(
                    ENTRIES.iter().any(|e| e.name == *name),
                    "{} names unknown system {name}",
                    entry.name
                );
            }
        }
    }

    // Every table entry carries a non-empty human-readable gate description.
    #[test]
    fn every_entry_documents_its_gate() {
        for entry in ENTRIES {
            assert!(
                !entry.present_when.is_empty(),
                "{} has no present_when",
                entry.name
            );
        }
    }
}

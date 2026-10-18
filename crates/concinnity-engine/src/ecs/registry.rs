//! The system table: the one place a system is registered and the one schedule
//! document. `define_systems!` generates the entries from it; table order is run
//! order. Every system is internal: it has no declarable asset and carries no
//! discriminant. `World::start` runs each entry's gate against the world's
//! content and pushes the systems the gates return, in table order. To add a
//! system: implement `System` on it, write its gate in `schedule`, and add one
//! entry here in its run position with its phase and its ordering edges.
//!
//! `phase` is the only thing this table owes the world outside it. Entries are
//! in phase order, and a system registered with `World::add_system` runs after
//! every entry sharing its phase, so an outside registration anchors to a phase
//! and never to an entry. The tick simulates first and submits its frame last,
//! so the frame shows everything the tick resolved. The five bands:
//!   * Early      -- the menu gate, this tick's input, the sky and a planet
//!     world's frame move.
//!   * Logic      -- the world's behaviors, whose requests drain later this tick.
//!   * Late       -- the simulation: physics, cameras, animation, story, audio.
//!   * PreRender  -- the request drains, streaming and the UI that reads them.
//!   * PostRender -- the frame itself, and the report that records it.
//!
//! `after`/`before` are the cross-system ordering constraints, validated
//! against table order when the world's schedule is built.
//! Each edge's rationale:
//!   * Overlay first: publishes the menu state (`MenuActive`) that gates
//!     simulation, input, and the draw this same tick.
//!   * Input right after the overlay: the tick consumes the input sampled
//!     before it began, so everything downstream sees this tick's input.
//!   * SkyRotation before Graphics: the orientation it publishes is what the
//!     frame's transform propagation, directional-light push and cubemap
//!     samples are taken at.
//!   * Behavior before Spawn/Settings/Story/Audio: the requests its firing
//!     rules emit (spawn/despawn, scene, story, audio) drain the same tick.
//!   * Planet before the world's logic, Spawn, Physics and the camera
//!     controllers: a move of a planet world's simulated frame lands before
//!     anything this tick places or moves, so a behavior's spawn request and
//!     its drain share one frame, and everything after it, the frame
//!     included, works in the new frame.
//!   * Physics before the camera controllers: physics consumes the camera's
//!     previous-frame `desired_move` (a one-frame-lagged resolution).
//!   * CameraTrack after Physics, before Audio: it sits where the input
//!     controllers sit, and replaces them when a world declares a track.
//!   * Cameras and Story before Audio: the listener reads the camera, and a
//!     `PlayCue` page audio is heard the same tick.
//!   * Streaming after the cameras: the camera-relative view it publishes is
//!     the pose this tick's frame draws.
//!   * Spawn/Settings/Streaming before Graphics: despawns leave the transform
//!     push, setting and streaming ops land before this frame's submit, and
//!     `CameraRelativeView` is ready for the draw.
//!   * SettingsWriter right after Settings and before UiInput and Graphics:
//!     both can stop the run, and a change made on the tick that stops must
//!     still be written.
//!   * LoadingOverlay after Streaming (reads the residency status published
//!     this tick) and before UiInput (its screen commands apply same tick).
//!   * Graphics last but for the report: the frame it submits shows the tick's
//!     input, simulation and cameras.
//!   * FrameReport last: it records the frame that just went, so everything
//!     it reads has to have run.
//!
//! Event-carried couplings (RootMotionEvent, GroundProbes, SettingCommand) are
//! order-robust thanks to the event store's two-frame retention.

use concinnity_core::ecs::PipelineContext;

use crate::ecs::{access_ids, decompose, host_resources, schedule};

// Runs once at world start, after the gates have built the systems and before
// their `init`. The engine defaults the world is completed with land earlier
// still (`complete_world` below), so a Prop one of them injects decomposes
// here like any other.
fn before_init(ctx: &mut PipelineContext) {
    // Every context touch a stepping system makes is asserted against what it
    // declared; the hook is installed before the first step can happen.
    #[cfg(debug_assertions)]
    access_ids::install_hook();
    host_resources::publish(ctx);
    decompose::run(ctx);
}

crate::define_systems! {
    complete_world: concinnity_core::defaults::run,
    before_init: before_init,
    prepare_events: access_ids::ensure_event_queues,
    // Graphics runs at the tail of the tick, but its init builds the window,
    // the backend and the payload indexes later inits read: it inits right
    // after the entries its own init reads.
    init_first: [
        OverlaySystem,
        SkyRotationSystem,
        BehaviorSystem,
        SpawnSystem,
        SettingsSystem,
        StreamingSystem,
        GraphicsSystem,
    ],

    OverlaySystem => concinnity_core::render::overlay::OverlaySystem {
        gate: schedule::overlay,
        present_when: "the world runs with a window",
        phase: Early,
        after: [],
        before: [BehaviorSystem, SpawnSystem, GraphicsSystem, InputSystem, PhysicsSystem, AnimationSystem],
    },
    InputSystem => crate::input::system::InputSystem {
        gate: schedule::input,
        present_when: "the world runs with a window",
        phase: Early,
        after: [OverlaySystem],
        before: [BehaviorSystem],
    },
    SkyRotationSystem => concinnity_core::sky::SkyRotationSystem {
        gate: schedule::sky_rotation,
        present_when: "the world declares a SkyRotation",
        phase: Early,
        after: [OverlaySystem],
        before: [GraphicsSystem],
    },
    PlanetSystem => crate::planet::PlanetSystem {
        gate: schedule::planet,
        present_when: "the world declares a Planet",
        phase: Early,
        after: [],
        before: [
            BehaviorSystem,
            SpawnSystem,
            PhysicsSystem,
            Camera3DSystem,
            ThirdPersonSystem,
            CameraTrackSystem,
        ],
    },
    BehaviorSystem => concinnity_core::behavior::BehaviorSystem {
        gate: schedule::behavior,
        present_when: "the world declares any Behavior",
        phase: Logic,
        after: [OverlaySystem],
        before: [SpawnSystem, SettingsSystem, StorySystem, AudioSystem],
    },
    PhysicsSystem => concinnity_core::physics::PhysicsSystem {
        gate: schedule::physics,
        present_when: "the world declares a PhysicsConfig, RigidBody, PropBody, or TriggerVolume, or a skinned mesh bakes a character capsule",
        phase: Late,
        after: [OverlaySystem],
        before: [Camera3DSystem, ThirdPersonSystem],
    },
    Camera3DSystem => concinnity_core::controller::camera::Camera3DSystem {
        gate: schedule::camera3d,
        present_when: "the first controlled Camera3D has no follow block",
        phase: Late,
        after: [PhysicsSystem],
        before: [AudioSystem],
    },
    ThirdPersonSystem => concinnity_core::controller::third_person::ThirdPersonSystem {
        gate: schedule::third_person,
        present_when: "the first controlled Camera3D has a follow block",
        phase: Late,
        after: [PhysicsSystem],
        before: [AudioSystem],
    },
    CameraTrackSystem => concinnity_core::camera_track::CameraTrackSystem {
        gate: schedule::camera_track,
        present_when: "the world declares a CameraTrack and the Camera3D it drives",
        phase: Late,
        after: [PhysicsSystem],
        before: [AudioSystem],
    },
    AnimationSystem => crate::animation::AnimationSystem {
        gate: schedule::animation,
        present_when: "the world declares any Animation or AnimationGraph",
        phase: Late,
        after: [OverlaySystem],
        before: [],
    },
    StorySystem => crate::story::StorySystem {
        gate: schedule::story,
        present_when: "the world declares a Story",
        phase: Late,
        after: [BehaviorSystem],
        before: [AudioSystem],
    },
    AudioSystem => crate::audio::AudioSystem {
        gate: schedule::audio,
        present_when: "the world declares any AudioEmitter, AudioCue, a Story page/choice with audio, or a Behavior with a sound node",
        phase: Late,
        after: [BehaviorSystem, Camera3DSystem, ThirdPersonSystem, StorySystem],
        before: [],
    },
    TextInputSystem => concinnity_core::input::text_input::TextInputSystem {
        gate: schedule::text_input,
        present_when: "the world declares any TextInput",
        phase: Late,
        after: [],
        before: [],
    },
    StatHud => concinnity_core::hud::stat_hud::StatHudSystem {
        gate: schedule::stat_hud,
        present_when: "the world declares a StatHud",
        phase: Late,
        after: [],
        before: [],
    },
    DebugHud => concinnity_core::hud::debug_hud::DebugHudSystem {
        gate: schedule::debug_hud,
        present_when: "the world declares a DebugHud AND the binary is a debug build or a `cn debug` session",
        phase: Late,
        after: [],
        before: [],
    },
    FpsCounter => concinnity_core::hud::fps_counter::FpsCounterSystem {
        gate: schedule::fps_counter,
        present_when: "the world declares an FpsCounter",
        phase: Late,
        after: [],
        before: [],
    },
    SpawnSystem => concinnity_core::spawn::SpawnSystem {
        gate: schedule::spawn,
        present_when: "the world runs with a window",
        phase: PreRender,
        after: [BehaviorSystem],
        before: [GraphicsSystem],
    },
    SettingsSystem => concinnity_core::settings::system::SettingsSystem {
        gate: schedule::settings,
        present_when: "the world runs with a window",
        phase: PreRender,
        after: [],
        before: [GraphicsSystem],
    },
    SettingsWriterSystem => crate::settings::SettingsWriterSystem {
        gate: schedule::settings_writer,
        present_when: "the world runs with a window",
        phase: PreRender,
        after: [SettingsSystem],
        before: [UiInputSystem, GraphicsSystem],
    },
    StreamingSystem => crate::gfx::streaming::system::StreamingSystem {
        gate: schedule::streaming,
        present_when: "the world runs with a window",
        phase: PreRender,
        after: [Camera3DSystem, ThirdPersonSystem, CameraTrackSystem],
        before: [GraphicsSystem],
    },
    LoadingOverlaySystem => concinnity_core::hud::loading_overlay::LoadingOverlaySystem {
        gate: schedule::loading_overlay,
        present_when: "the world declares a LoadingOverlay",
        phase: PreRender,
        after: [StreamingSystem],
        before: [UiInputSystem],
    },
    UiInputSystem => concinnity_core::ui::UiInputSystem {
        gate: schedule::ui_input,
        present_when: "the world declares any HitRegion, Screen, or KeyBinding",
        phase: PreRender,
        after: [LoadingOverlaySystem],
        before: [],
    },
    GraphicsSystem => crate::gfx::system::GraphicsSystem {
        gate: schedule::graphics,
        present_when: "the world runs with a window",
        phase: PostRender,
        after: [SpawnSystem, SettingsSystem, StreamingSystem],
        before: [],
    },
    FrameReportSystem => concinnity_core::frame_report::FrameReportSystem {
        gate: schedule::frame_report,
        present_when: "the run is benchmarked",
        phase: PostRender,
        after: [GraphicsSystem, CameraTrackSystem],
        before: [],
    },
}

#[cfg(test)]
mod tests {
    use super::SYSTEMS;
    use concinnity_core::components::AudioEmitter;
    use concinnity_core::components::Story;
    use concinnity_core::ecs::SystemEntry;
    use concinnity_core::ecs::{ComponentAsset, World};

    // The table's entries, in run order.
    const ENTRIES: &[SystemEntry] = SYSTEMS.entries;

    // The overlay HUD components each gate their internal system and build in
    // the fixed schedule order (StatHud, then DebugHud, then FpsCounter).
    // DebugHud is developer-only but `cfg!(debug_assertions)` holds under test.
    #[test]
    fn hud_components_spawn_in_schedule_order() {
        use concinnity_core::components::{DebugHud, FpsCounter, StatHud};

        let mut world = World::new();
        world.add_component(FpsCounter::default());
        world.add_component(StatHud::default());
        world.add_component(DebugHud::default());
        world.start(SYSTEMS).unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["StatHud", "DebugHud", "FpsCounter"]);
    }

    // Any HitRegion, Screen or KeyBinding gates the UI system in; a world with
    // none of them gets none.
    #[test]
    fn ui_components_gate_the_ui_system() {
        use concinnity_core::components::Screen;
        use concinnity_core::ecs::asset_id::AssetId;

        let mut world = World::new();
        world.start(SYSTEMS).unwrap();
        assert!(world.systems().is_empty());

        let mut world = World::new();
        world.push_identified(AssetId(1), Screen::default());
        world.start(SYSTEMS).unwrap();
        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["UiInputSystem"]);
    }

    // The manifest reports exactly the systems `start()` builds, in the same
    // order, for a world gating several table entries. Audio is left ungated
    // so `start()` opens no device here.
    #[test]
    fn system_manifest_matches_started_systems() {
        use concinnity_core::components::{DebugHud, FpsCounter, StatHud, Story, TextInput};

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

    // The table's completion pass runs from `start`, before the gates read the
    // world: a world with physics content gets the config its simulation runs
    // on, and the PhysicsSystem is built either way.
    #[test]
    fn start_completes_the_world_with_its_engine_defaults() {
        use concinnity_core::components::{EngineDefaults, PhysicsConfig, PropBody};

        let mut world = World::new();
        world.add_component(PropBody::default());
        world.start(SYSTEMS).unwrap();

        assert_eq!(world.query::<PhysicsConfig>().count(), 1);
        // The directive is consumed by the pass, so nothing holds one after.
        assert_eq!(world.query::<EngineDefaults>().count(), 0);
        let built: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(built, ["PhysicsSystem"]);
    }

    // A declared track takes the camera off the input controller, so exactly
    // one system writes the pose. Without the track the same world builds the
    // controller instead.
    #[test]
    fn a_camera_track_replaces_the_camera_controller() {
        use concinnity_core::components::{Camera3D, CameraTrack, PhysicsConfig};

        let mut world = World::new();
        world.add_component(Camera3D::bake(Default::default()));
        world.add_component(PhysicsConfig::default());
        assert_eq!(
            world.system_manifest(SYSTEMS),
            ["PhysicsSystem", "Camera3DSystem"]
        );

        let mut world = World::new();
        world.add_component(Camera3D::bake(Default::default()));
        world.add_component(CameraTrack::default());
        world.add_component(PhysicsConfig::default());
        assert_eq!(
            world.system_manifest(SYSTEMS),
            ["PhysicsSystem", "CameraTrackSystem"]
        );
    }

    // Manifest names come out in table order, and every name is a real table
    // entry (the manifest is a filtered view of `SYSTEMS`, nothing else).
    #[test]
    fn system_manifest_is_a_table_order_subset() {
        use concinnity_core::components::{FpsCounter, StatHud};

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

    // A planet world's frame moves before anything the tick places or moves:
    // a behavior's spawn request carries a world-space transform that its
    // drain must read in the frame it was made in.
    #[test]
    fn the_planet_frame_moves_before_the_worlds_logic() {
        let position = |name: &str| {
            ENTRIES
                .iter()
                .position(|e| e.name == name)
                .unwrap_or_else(|| panic!("{name} is a table entry"))
        };
        let planet = position("PlanetSystem");
        for later in [
            "BehaviorSystem",
            "SpawnSystem",
            "PhysicsSystem",
            "CameraTrackSystem",
        ] {
            assert!(planet < position(later), "PlanetSystem runs before {later}");
        }
    }

    // Every entry the table inits ahead of the rest is one of its own.
    #[test]
    fn init_first_names_exist() {
        for name in SYSTEMS.init_first {
            assert!(
                ENTRIES.iter().any(|e| e.name == *name),
                "{name} is a table entry"
            );
        }
    }

    // The frame is submitted after everything that moves what it draws: every
    // simulation entry runs before StreamingSystem, whose `CameraRelativeView`
    // is the pose the draw takes, and only the report follows GraphicsSystem.
    #[test]
    fn the_frame_follows_the_simulation() {
        use concinnity_core::ecs::Phase;
        let position = |name: &str| {
            ENTRIES
                .iter()
                .position(|e| e.name == name)
                .unwrap_or_else(|| panic!("{name} is a table entry"))
        };
        let streaming = position("StreamingSystem");
        let graphics = position("GraphicsSystem");
        for (i, entry) in ENTRIES.iter().enumerate() {
            if entry.phase <= Phase::Late {
                assert!(i < streaming, "{} runs before streaming", entry.name);
            }
        }
        assert!(streaming < graphics);
        let after: Vec<&str> = ENTRIES[graphics + 1..].iter().map(|e| e.name).collect();
        assert_eq!(after, ["FrameReportSystem"]);
    }

    // The two camera-controller entries are mutually exclusive: the first
    // controlled camera's `follow` block picks exactly one of them.
    #[test]
    fn camera_controller_gates_are_exclusive() {
        use concinnity_core::components::{Camera3D, CameraController, FollowController};

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
        world.add_component(AudioEmitter::default());
        assert_eq!(world.system_manifest(SYSTEMS), ["AudioSystem"]);
    }

    // A Story gates the StorySystem. An empty-node story pulls in no audio
    // device (build_audio needs a page/choice cue), so this stays device-free.
    #[test]
    fn story_component_spawns_story_system() {
        let mut world = World::new();
        world.add_component(Story::default());
        world.start(SYSTEMS).unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["StorySystem"]);
    }

    // A fresh world holds nothing; adding a component (through either the blob
    // path or the typed one) fills it, and `start()` is what gives it systems.
    #[test]
    fn empty_world_fills_from_components_then_systems() {
        use concinnity_core::components::{FpsCounter, TextLabel};

        let mut world = World::new();
        assert!(world.is_empty());
        assert_eq!(world.component_count(), 0);
        assert_eq!(world.system_count(), 0);

        world.add(ComponentAsset::from(TextLabel::default()), None);
        assert!(!world.is_empty());
        assert_eq!(world.component_count(), 1);

        world.add_component(FpsCounter::default());
        world.start(SYSTEMS).unwrap();
        assert_eq!(world.system_count(), 1, "the FpsCounter gate built one");
    }

    // The table is in phase order: the merge walks the phases once, so an entry
    // out of phase order would silently move the moment a system was registered
    // into the phase beside it.
    #[test]
    fn entries_are_in_phase_order() {
        for pair in ENTRIES.windows(2) {
            assert!(
                pair[0].phase <= pair[1].phase,
                "{} runs in {} and {} after it in {}",
                pair[0].name,
                pair[0].phase.as_str(),
                pair[1].name,
                pair[1].phase.as_str(),
            );
        }
    }

    // A system registered on a world lands in the phase it named, among the
    // engine's own: after every table entry sharing that phase, and before
    // every entry of a later one. Manifest-only, so no device is built.
    #[test]
    fn a_registration_lands_in_the_phase_it_named() {
        use concinnity_core::components::{FpsCounter, SkyRotation};
        use concinnity_core::ecs::{Phase, PipelineContext, StepResult, System};

        #[derive(Debug)]
        struct Inert;

        impl System for Inert {
            fn step(&mut self, _ctx: &mut PipelineContext) -> StepResult {
                StepResult::Continue
            }
        }

        let mut world = World::new();
        world.add_component(SkyRotation::default());
        world.add_component(FpsCounter::default());
        world.add_system(Phase::Early, "MyEarly", Inert);
        world.add_system(Phase::Late, "MyLate", Inert);

        assert_eq!(
            world.system_manifest(SYSTEMS),
            ["SkyRotationSystem", "MyEarly", "FpsCounter", "MyLate"]
        );
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

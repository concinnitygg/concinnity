use super::{Core, SYSTEMS};
use concinnity_core::audio::system::AudioSystem;
use concinnity_core::components::{
    AudioCue, AudioEmitter, EngineDefaults, FpsCounter, PhysicsConfig, PropBody, Screen, Story,
};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{AudioClipHandle, ComponentAsset, Phase, Ref, SystemEntry, World};

const ENTRIES: &[SystemEntry] = SYSTEMS.entries;

fn position(name: &str) -> usize {
    ENTRIES
        .iter()
        .position(|e| e.name == name)
        .unwrap_or_else(|| panic!("{name} is a table entry"))
}

fn started(world: &mut World) -> Vec<&'static str> {
    world.start(SYSTEMS).unwrap();
    world.systems().iter().map(|s| s.name()).collect()
}

// The table is well formed, holds every core entry, and keeps core's order.
#[test]
fn the_table_holds_every_core_system_in_core_order() {
    assert_eq!(SYSTEMS.check_complete(Core::ALL), Ok(()));
}

// The frame is submitted after everything that moves what it draws: every
// simulation entry runs before StreamingSystem, whose `CameraRelativeView` is
// the pose the draw takes, and only the report follows GraphicsSystem.
#[test]
fn the_frame_follows_the_simulation() {
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

// The completion pass runs from `start`, before the gates read the world: a
// world with physics content gets the config its simulation runs on.
#[test]
fn start_completes_the_world_with_its_engine_defaults() {
    let mut world = World::new();
    world.add_component(PropBody::default());
    assert_eq!(started(&mut world), ["PhysicsSystem"]);
    assert_eq!(world.query::<PhysicsConfig>().count(), 1);
    // The directive is consumed by the pass, so nothing holds one after.
    assert_eq!(world.query::<EngineDefaults>().count(), 0);
}

// A fresh world holds nothing; adding a component (through either the blob
// path or the typed one) fills it, and `start()` is what gives it systems.
#[test]
fn empty_world_fills_from_components_then_systems() {
    use concinnity_core::components::TextLabel;

    let mut world = World::new();
    assert!(world.is_empty());
    assert_eq!(world.system_count(), 0);
    world.add(ComponentAsset::from(TextLabel::default()), None);
    assert_eq!(world.component_count(), 1);
    world.add_component(FpsCounter::default());
    assert_eq!(started(&mut world), ["FpsCounterSystem"]);
}

// Each kind of sound content gates the audio system in, visible in the
// manifest without a device: the gate probe constructs the system, and device
// acquisition waits for `System::init`.
#[test]
fn sound_content_gates_the_audio_system_without_a_device() {
    let mut world = World::new();
    assert!(world.system_manifest(SYSTEMS).is_empty());
    world.add_component(AudioEmitter::default());
    assert_eq!(world.system_manifest(SYSTEMS), ["AudioSystem"]);

    let mut world = World::new();
    world.add_component(AudioCue::default());
    assert_eq!(world.system_manifest(SYSTEMS), ["AudioSystem"]);
}

// Impact sounds alone need the audio system; a silent body does not.
#[test]
fn an_impact_sound_gates_the_audio_system() {
    let mut silent = World::new();
    silent.add_component(PropBody::default());
    assert!(!silent.system_manifest(SYSTEMS).contains(&"AudioSystem"));

    let mut world = World::new();
    world.add_component(PropBody {
        impact_clip: Some(AudioClipHandle::new(0)),
        ..Default::default()
    });
    assert!(world.system_manifest(SYSTEMS).contains(&"AudioSystem"));
}

// An empty-node story pulls in no audio device: its gate needs a page or
// choice cue.
#[test]
fn a_silent_story_builds_no_audio_system() {
    let mut world = World::new();
    world.add_component(Story::default());
    assert_eq!(started(&mut world), ["StorySystem"]);
}

// The full trigger chain: the initial screen's activation (announced by
// UiInputSystem at init) reaches the audio system, which matches the screen's
// cue on the first step. Playback itself needs a device and a compiled
// payload, so the test observes the match counter.
#[test]
fn initial_view_fires_its_cue() {
    let mut world = World::new();
    let screen = AssetId(90);
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
        .system::<AudioSystem>()
        .map(|a| a.cues_matched())
        .expect("world has an AudioSystem");
    assert_eq!(matched, 1, "the initial screen's cue should have matched");
}

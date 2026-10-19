// These tests drive AudioSystem::init / step against a World over an
// in-memory blob with no output open, so no audio plays. They assert on the
// state the system tracks: cue bindings, queued-clip and match counters,
// emitter bindings, the occlusion probes and the requested mix.

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::{AudioSystem, EmitterBinding, EmitterId};
use crate::audio::occlusion::OcclusionSmoother;
use crate::audio::out::AudioOut;
use crate::components::{
    AudioBus, AudioCommand, AudioCue, AudioEmitter, AudioOcclusionProbe, AudioTarget, BodyDynamics,
    Camera3D, ContactEvent, CueKind, PlayCue, ScreenShown, Story, Transform,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{
    AudioClipHandle, PayloadLocator, ResourceKind, ResourceRecord, StepResult, System, World,
};
use crate::ecs::{PayloadStore, Ref};
use crate::error::PayloadError;
use crate::resource::AudioClipTable;
use crate::settings::StoredSettings;

fn no_output() -> Option<Box<dyn AudioOut>> {
    None
}

// One in-memory blob section every payload locator reads from.
struct OneSection(Vec<u8>);

impl PayloadStore for OneSection {
    fn read(&mut self, locator: &PayloadLocator) -> Result<&[u8], PayloadError> {
        let start = locator.offset as usize;
        let end = start + locator.len as usize;
        self.0.get(start..end).ok_or(PayloadError::OutOfBounds {
            index: locator.blob_index,
            offset: locator.offset,
            len: locator.len,
            section_len: self.0.len(),
        })
    }

    fn release(&mut self, _blob_index: u32) {}

    fn disk_backed(&self) -> bool {
        false
    }
}

// Accumulates one blob section serving every clip payload handed out, plus
// the audio-clip resource records, then seals into a world whose
// `AudioClipTable` is built from those records -- exactly as the runtime
// loads the blob's resource stream.
struct AudioClips {
    section: Vec<u8>,
    // The audio-clip resource records, in handle order (a clip added Nth is
    // handle N). Sealed into the world's `AudioClipTable`.
    clips: Vec<ResourceRecord>,
}

impl AudioClips {
    fn new() -> Self {
        Self {
            section: Vec::new(),
            clips: Vec::new(),
        }
    }

    // Add an audio clip whose payload is `bytes`, returning its handle (its
    // record order, which the table indexes by).
    fn clip(&mut self, bytes: &[u8]) -> AudioClipHandle {
        let handle = AudioClipHandle::new(self.clips.len() as u32);
        let offset = self.section.len() as u64;
        self.section.extend_from_slice(bytes);
        self.clips.push(ResourceRecord {
            resource_kind: ResourceKind::AudioClip,
            handle: handle.0,
            payload: Some(PayloadLocator {
                blob_index: 0,
                offset,
                len: bytes.len() as u64,
            }),
            data_bytes: Vec::new(),
        });
        handle
    }

    fn seal(mut self) -> World {
        let mut world = World::from_payloads(Box::new(OneSection(self.section)));
        world.insert_resource(AudioClipTable::from_records(&mut self.clips));
        world
    }
}

// init binds a screen-triggered cue with its routing defaults and hands
// its clip to the decode worker once, so firing the cue later never
// touches the blob.
#[test]
fn init_binds_cue_and_queues_its_clip() {
    let screen = AssetId(90);

    let mut clips = AudioClips::new();
    let clip = clips.clip(b"cue-clip-bytes");
    let mut world = clips.seal();
    world.push(AudioCue {
        screen: Some(Ref::new(screen)),
        clip: Some(clip),
        kind: CueKind::Music,
        volume: 0.7,
        ..Default::default()
    });

    let mut sys = AudioSystem::new(no_output);
    sys.init(&mut world.context());

    let bindings = sys.cues.get(&screen).expect("cue bound to its screen");
    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[0].clip, clip);
    assert_eq!(bindings[0].kind, CueKind::Music);
    assert!((bindings[0].volume - 0.7).abs() < 1.0e-6);
    assert_eq!(
        bindings[0].bus,
        AudioBus::Music,
        "music cue defaults to the music bus"
    );
    assert_eq!(bindings[0].priority, 0);
    assert_eq!(sys.clips_queued(), 1);
}

// An authored bus and priority override the kind-derived defaults.
#[test]
fn init_respects_authored_cue_bus_and_priority() {
    let screen = AssetId(90);

    let mut clips = AudioClips::new();
    let clip = clips.clip(b"line-bytes");
    let mut world = clips.seal();
    world.push(AudioCue {
        screen: Some(Ref::new(screen)),
        clip: Some(clip),
        kind: CueKind::Sound,
        bus: Some(AudioBus::Voice),
        priority: 4,
        ..Default::default()
    });

    let mut sys = AudioSystem::new(no_output);
    sys.init(&mut world.context());

    let bindings = sys.cues.get(&screen).expect("cue bound");
    assert_eq!(bindings[0].bus, AudioBus::Voice);
    assert_eq!(bindings[0].priority, 4);
}

// A cue missing its screen or its clip is ignored: nothing is bound and no
// clip is queued.
#[test]
fn init_ignores_cue_without_clip() {
    let mut world = AudioClips::new().seal();
    world.push(AudioCue {
        screen: Some(Ref::new(AssetId(90))),
        clip: None,
        ..Default::default()
    });

    let mut sys = AudioSystem::new(no_output);
    sys.init(&mut world.context());

    assert!(sys.cues.is_empty());
    assert_eq!(sys.clips_queued(), 0);
}

// A story queues every clip up front (it plays them by direct PlayCue
// request, not through screen-keyed cues), each clip exactly once even
// when a cue already queued it.
#[test]
fn init_queues_story_clips_without_duplicates() {
    let screen = AssetId(90);

    let mut clips = AudioClips::new();
    let cue_clip = clips.clip(b"cue-audio");
    let page_clip = clips.clip(b"story-page-audio");
    let mut world = clips.seal();
    world.push(AudioCue {
        screen: Some(Ref::new(screen)),
        clip: Some(cue_clip),
        ..Default::default()
    });
    world.push(Story::default());

    let mut sys = AudioSystem::new(no_output);
    sys.init(&mut world.context());

    let _ = page_clip;
    assert_eq!(sys.clips_queued(), 2, "both clips queued, neither twice");
}

// init binds every emitter (device or not), queues its clip, and
// attaches one occlusion probe to each emitter's own entity, aimed at it.
#[test]
fn init_binds_emitters_and_publishes_occlusion_probes() {
    let mut clips = AudioClips::new();
    let clip = clips.clip(b"loop-bytes");
    let mut world = clips.seal();
    let entity = world.push(AudioEmitter {
        clip: Some(clip),
        position: [3.0, 1.0, -2.0],
        ..Default::default()
    });

    let mut sys = AudioSystem::new(no_output);
    sys.init(&mut world.context());

    assert_eq!(sys.emitters.len(), 1);
    let binding = sys.emitters.get(&entity).expect("binding keyed by entity");
    assert_eq!(binding.position, [3.0, 1.0, -2.0]);
    assert_eq!(sys.clips_queued(), 1);

    let ctx = world.context();
    let probe = ctx
        .get::<AudioOcclusionProbe>(entity)
        .expect("probe rides the emitter's entity");
    assert_eq!(probe.to, [3.0, 1.0, -2.0]);
    assert_eq!(probe.blocked, None, "unanswered until physics steps");
    assert_eq!(ctx.query::<AudioOcclusionProbe>().count(), 1);
}

// A planet world's frame move carries a free-standing emitter, its
// component and its occlusion ray's end to where the move says.
#[test]
fn a_frame_move_carries_free_standing_emitters() {
    use crate::planet::{FrameRebases, Rebase};
    let mut clips = AudioClips::new();
    let clip = clips.clip(b"loop-bytes");
    let mut world = clips.seal();
    let entity = world.push(AudioEmitter {
        clip: Some(clip),
        position: [1_003.0, 2.0, -4.0],
        ..Default::default()
    });
    world.insert_resource(FrameRebases::default());
    let mut sys = AudioSystem::new(no_output);
    sys.init(&mut world.context());
    let rebase = Rebase {
        rotation: crate::math::quat_from_axis_angle([0.0, 0.0, 1.0], 0.02),
        translation: [-1_000.0, 20.0, 0.0],
    };
    let mut rebases = FrameRebases::default();
    rebases.push(rebase);
    world.insert_resource(rebases);
    sys.step(&mut world.context());

    let expected = rebase.apply_point([1_003.0, 2.0, -4.0]);
    assert_eq!(sys.emitters[&entity].position, expected);
    {
        let ctx = world.context();
        assert_eq!(ctx.get::<AudioEmitter>(entity).unwrap().position, expected);
        assert_eq!(ctx.get::<AudioOcclusionProbe>(entity).unwrap().to, expected);
    }
    sys.step(&mut world.context());
    assert_eq!(sys.emitters[&entity].position, expected, "carried once");
}

// An AudioEmitter appearing after init is adopted on the next step, and
// a binding whose entity despawns is reaped (probe and all).
#[test]
fn step_adopts_new_emitters_and_reaps_despawned_ones() {
    let mut clips = AudioClips::new();
    let clip = clips.clip(b"loop-bytes");
    let mut world = clips.seal();
    let first = world.push(AudioEmitter {
        clip: Some(clip),
        position: [1.0, 0.0, 0.0],
        ..Default::default()
    });

    let mut sys = AudioSystem::new(no_output);
    sys.init(&mut world.context());
    assert_eq!(sys.emitters.len(), 1);

    // A second emitter appears at runtime; the next step binds it and
    // attaches its probe.
    let second = {
        let ctx = world.context();
        ctx.components.push_typed(AudioEmitter {
            clip: Some(clip),
            position: [7.0, 0.0, 0.0],
            ..Default::default()
        })
    };
    assert_eq!(sys.step(&mut world.context()), StepResult::Continue);
    assert_eq!(sys.emitters.len(), 2);
    assert_eq!(
        sys.emitters.get(&second).map(|b| b.position),
        Some([7.0, 0.0, 0.0])
    );
    assert!(
        world.context().get::<AudioOcclusionProbe>(second).is_some(),
        "adopted emitter got its probe"
    );

    // The first emitter despawns; its binding is reaped and it is not
    // re-adopted.
    world.context().despawn(first);
    assert_eq!(sys.step(&mut world.context()), StepResult::Continue);
    assert_eq!(sys.emitters.len(), 1);
    assert!(sys.emitters.contains_key(&second), "survivor kept");
    assert_eq!(world.context().query::<AudioOcclusionProbe>().count(), 1);
}

// A contact whose body carries an impact clip plays it; bodies without
// dynamics or without a clip are silent. The counter is the observable.
#[test]
fn step_plays_impact_clips_from_contacts() {
    let mut clips = AudioClips::new();
    let clip = clips.clip(b"thud-bytes");
    let mut world = clips.seal();

    let (crate_entity, bare_entity) = {
        let mut ctx = world.context();
        let crate_entity = ctx.components.spawn();
        ctx.insert(
            crate_entity,
            BodyDynamics {
                impact_clip: Some(clip),
                ..Default::default()
            },
        );
        let bare_entity = ctx.components.spawn();
        ctx.insert(bare_entity, BodyDynamics::default());
        (crate_entity, bare_entity)
    };

    let mut sys = AudioSystem::new(no_output);
    sys.init(&mut world.context());
    assert_eq!(sys.clips_queued(), 1, "impact clip queued up front");

    {
        let mut ctx = world.context();
        let events = ctx.events_mut::<ContactEvent>();
        // Both sides resolve: only the crate has a clip.
        events.send(ContactEvent {
            a: crate_entity,
            b: Some(bare_entity),
            point: [0.0, 0.5, 0.0],
            normal: [0.0, 1.0, 0.0],
            impulse: 6.0,
        });
        // `a` has dynamics but no clip, and no `b`: silent.
        events.send(ContactEvent {
            a: bare_entity,
            b: None,
            point: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            impulse: 6.0,
        });
    }
    assert_eq!(sys.step(&mut world.context()), StepResult::Continue);
    assert_eq!(sys.impacts_played(), 1, "one clip-carrying side played");
}

// A blocked probe answer muffles gradually: the smoothed factor rises
// over several steps instead of slamming to 1, and the system refreshes
// the probe's ray endpoints each step.
#[test]
fn step_smooths_probe_answers_and_refreshes_endpoints() {
    let mut clips = AudioClips::new();
    let clip = clips.clip(b"loop-bytes");
    let mut world = clips.seal();
    world.push(AudioEmitter {
        clip: Some(clip),
        position: [5.0, 0.0, 0.0],
        ..Default::default()
    });

    let mut sys = AudioSystem::new(no_output);
    sys.init(&mut world.context());

    // Physics answered: the segment is blocked.
    {
        let mut ctx = world.context();
        for probe in ctx.query_mut::<AudioOcclusionProbe>() {
            probe.blocked = Some(true);
        }
    }
    let occlusion_of = |sys: &AudioSystem| {
        sys.emitters
            .values()
            .next()
            .expect("one binding")
            .occlusion
            .current()
    };
    assert_eq!(sys.step(&mut world.context()), StepResult::Continue);
    let after_one = occlusion_of(&sys);
    assert!(
        after_one > 0.0 && after_one < 0.5,
        "one tick moves partway: {after_one}"
    );
    for _ in 0..120 {
        sys.step(&mut world.context());
    }
    assert!(occlusion_of(&sys) > 0.9, "settles blocked");

    // The ray endpoints track the listener and emitter for next frame.
    let ctx = world.context();
    let probe = ctx.query::<AudioOcclusionProbe>().next().unwrap();
    assert_eq!(probe.from, [0.0; 3], "no camera: listener at origin");
    assert_eq!(probe.to, [5.0, 0.0, 0.0]);
}

// A shown screen fires each of its cues once, across both playback kinds; the
// output-independent match counter tracks the progress.
#[test]
fn step_fires_cued_view_across_both_kinds() {
    let screen = AssetId(90);

    let mut clips = AudioClips::new();
    let music_clip = clips.clip(b"music");
    let sound_clip = clips.clip(b"sound");
    let mut world = clips.seal();
    world.push(AudioCue {
        screen: Some(Ref::new(screen)),
        clip: Some(music_clip),
        kind: CueKind::Music,
        volume: 1.0,
        ..Default::default()
    });
    world.push(AudioCue {
        screen: Some(Ref::new(screen)),
        clip: Some(sound_clip),
        kind: CueKind::Sound,
        volume: 1.0,
        ..Default::default()
    });

    let mut sys = AudioSystem::new(no_output);
    sys.init(&mut world.context());
    assert_eq!(sys.cues.get(&screen).map(Vec::len), Some(2));

    {
        let mut ctx = world.context();
        ctx.events_mut::<ScreenShown>().send(ScreenShown { screen });
    }
    assert_eq!(sys.step(&mut world.context()), StepResult::Continue);
    assert_eq!(sys.cues_matched, 2, "both the screen's cues matched");

    // A screen with no cue leaves the counter untouched.
    {
        let mut ctx = world.context();
        ctx.events_mut::<ScreenShown>().send(ScreenShown {
            screen: AssetId(999),
        });
    }
    assert_eq!(sys.step(&mut world.context()), StepResult::Continue);
    assert_eq!(sys.cues_matched, 2);
}

// A direct PlayCue request (the story system's page audio) is played the
// same tick it is sent.
#[test]
fn step_plays_direct_play_cue_requests() {
    let mut clips = AudioClips::new();
    let clip = clips.clip(b"page-audio");
    let mut world = clips.seal();
    world.push(Story::default());

    let mut sys = AudioSystem::new(no_output);
    sys.init(&mut world.context());

    {
        let mut ctx = world.context();
        let events = ctx.events_mut::<PlayCue>();
        events.send(PlayCue {
            clip,
            kind: CueKind::Music,
            volume: 1.0,
            priority: 0,
        });
        events.send(PlayCue {
            clip,
            kind: CueKind::Sound,
            volume: 0.5,
            priority: 2,
        });
    }
    assert_eq!(sys.step(&mut world.context()), StepResult::Continue);
    assert_eq!(sys.cues_matched, 2, "both direct requests fired");
}

// A prop-bound emitter tracks its prop's Transform each step, and the
// listener rides the camera. Neither needs a device; the step drives the
// lookups and returns Continue.
#[test]
fn step_follows_prop_emitter_and_moves_listener() {
    let prop = AssetId(200);

    let mut world = AudioClips::new().seal();
    let entity = {
        let mut ctx = world.context();
        ctx.push(Camera3D::bake(Default::default()));
        let e = ctx.components.spawn();
        ctx.insert(
            e,
            Transform {
                position: [3.0, 4.0, 5.0],
                rotation_deg: [0.0; 3],
                scale: [1.0; 3],
            },
        );
        ctx.identify(e, prop);
        e
    };
    assert!(world.context().get::<Transform>(entity).is_some());

    let mut sys = AudioSystem::new(no_output);
    // A live emitter that follows the prop, seeded directly (a real
    // emitter needs a device, which the headless test has no access to).
    // Keyed by a spare entity standing in for the emitter's own.
    let emitter_entity = world.context().components.spawn();
    sys.emitters.insert(
        emitter_entity,
        EmitterBinding {
            id: Some(EmitterId(0)),
            follows: Some(prop),
            position: [0.0; 3],
            occlusion: OcclusionSmoother::default(),
        },
    );

    assert_eq!(sys.step(&mut world.context()), StepResult::Continue);
    assert_eq!(
        sys.emitters[&emitter_entity].position,
        [3.0, 4.0, 5.0],
        "binding tracked the prop"
    );
}

// A volume AudioCommand sent mid-tick is read AND applied by the audio
// system the same tick, so the new gain takes effect without a restart
// (the settings-menu volume rows). Default volumes at construction mean
// init leaves every stage at unity.
#[test]
fn audio_command_applies_volumes_live_per_target() {
    let mut world = AudioClips::new().seal();
    let mut sys = AudioSystem::new(no_output);
    sys.init(&mut world.context());
    // Init applied unity everywhere (no persisted volumes handed in).
    assert_eq!(sys.mixer.last_volume(AudioTarget::Master), 1.0);
    assert_eq!(sys.mixer.last_volume(AudioTarget::Voice), 1.0);

    // GraphicsSystem would send these when volume rows are cycled; the
    // audio system reads them this same tick.
    {
        let mut ctx = world.context();
        let events = ctx.events_mut::<AudioCommand>();
        events.send(AudioCommand {
            target: AudioTarget::Master,
            gain: 0.5,
        });
        events.send(AudioCommand {
            target: AudioTarget::Music,
            gain: 0.25,
        });
    }
    assert_eq!(sys.step(&mut world.context()), StepResult::Continue);
    assert_eq!(sys.mixer.last_volume(AudioTarget::Master), 0.5);
    assert_eq!(sys.mixer.last_volume(AudioTarget::Music), 0.25);
    assert_eq!(sys.mixer.last_volume(AudioTarget::Sfx), 1.0, "untouched");
}

// Several AudioCommands for one target sent in one tick (e.g. a rapid
// double-cycle) are all read in order; the last one sent wins.
#[test]
fn audio_command_last_write_wins_per_tick() {
    let mut world = AudioClips::new().seal();
    let mut sys = AudioSystem::new(no_output);
    sys.init(&mut world.context());

    {
        let mut ctx = world.context();
        let events = ctx.events_mut::<AudioCommand>();
        events.send(AudioCommand {
            target: AudioTarget::Master,
            gain: 0.5,
        });
        events.send(AudioCommand {
            target: AudioTarget::Master,
            gain: 0.25,
        });
    }
    assert_eq!(sys.step(&mut world.context()), StepResult::Continue);
    assert_eq!(sys.mixer.last_volume(AudioTarget::Master), 0.25);
}

// The stored settings-menu volumes are applied to every stage at init.
#[test]
fn init_applies_persisted_volumes() {
    let mut world = AudioClips::new().seal();
    let mut stored = StoredSettings::default();
    stored.audio.master_volume = Some(0.5);
    stored.audio.music_volume = Some(0.75);
    stored.audio.voice_volume = Some(0.25);
    world.insert_resource(stored);
    let mut sys = AudioSystem::new(no_output);
    sys.init(&mut world.context());
    assert_eq!(sys.mixer.last_volume(AudioTarget::Master), 0.5);
    assert_eq!(sys.mixer.last_volume(AudioTarget::Music), 0.75);
    assert_eq!(
        sys.mixer.last_volume(AudioTarget::Sfx),
        1.0,
        "unset = unity"
    );
    assert_eq!(sys.mixer.last_volume(AudioTarget::Voice), 0.25);
}

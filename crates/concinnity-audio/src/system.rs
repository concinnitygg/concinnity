// concinnity-audio/src/system.rs
//
// Audio playback: 3D positional emitters and screen-triggered cues. An internal
// system (not a declarable asset): the engine schedule constructs one whenever
// the world contains any `AudioEmitter` or `AudioCue`, so a world with neither
// never opens an audio device.

use std::collections::HashMap;

use crate::{AudioEngine, EmitterId};
use concinnity_core::assets::{
    AudioCommand, AudioCue, AudioEmitter, Behavior, Camera3D, CueKind, PlayCue, ScreenShown, Story,
    Transform,
};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{
    AudioClipHandle, EntityByName, EventCursor, PayloadLocator, PipelineContext, StepResult, System,
};
use concinnity_core::resource::AudioClipTable;

// Audio behavior. Constructed internally by `World::start` when the world
// declares any `AudioEmitter` or `AudioCue`; never a world-declared asset, so
// it carries no config.
pub struct AudioSystem {
    engine: AudioEngine,
    // The persisted master output volume (settings menu), applied to the main
    // mix track at init. `None` leaves output at unity. Resolved by the
    // engine's audio gate (which owns the settings store) and handed in at
    // construction, so this crate needs no dependency on the engine.
    master_volume: Option<f32>,
    // One entry per `AudioEmitter` that became a live engine emitter.
    emitters: Vec<EmitterBinding>,
    // Screen-triggered cues, keyed by the Screen whose activation fires them.
    cues: HashMap<AssetId, Vec<CueBinding>>,
    // Encoded clip payloads for the cue and story clips, keyed by the clip's
    // AudioClipHandle.
    cue_clip_bytes: HashMap<AudioClipHandle, Vec<u8>>,
    // Cursor into the Events<AudioCommand> queue (live master-volume changes).
    audio_cmd_cursor: EventCursor,
    // Cursor into the Events<ScreenShown> queue (cue triggers).
    view_shown_cursor: EventCursor,
    // Cursor into the Events<PlayCue> queue (direct play requests, e.g. the
    // story system's page audio).
    play_cue_cursor: EventCursor,
    // Cues that matched a shown screen so far; observable engine-independent
    // progress for headless tests (playback needs a device and a payload).
    cues_matched: usize,
}

// Links one engine emitter to the world data that positions it.
struct EmitterBinding {
    id: EmitterId,
    // The Prop this emitter follows each frame, if any.
    follows: Option<AssetId>,
}

// One AudioCue resolved to its clip and playback style.
struct CueBinding {
    clip: AudioClipHandle,
    kind: CueKind,
    volume: f32,
}

impl std::fmt::Debug for AudioSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioSystem")
            .field("engine", &self.engine)
            .field("emitters", &self.emitters.len())
            .field("cue_views", &self.cues.len())
            .finish()
    }
}

impl AudioSystem {
    // Fresh system with no device, live emitters, or cues. `master_volume` is
    // the persisted settings-menu master (`None` = unity), applied to the mix
    // in [`System::init`]. The output device is acquired and the emitters /
    // cues are bound from the world's components in `init`, so construction is
    // side-effect-free (required by the `World::system_manifest` gate probe).
    pub fn new(master_volume: Option<f32>) -> Self {
        Self {
            engine: AudioEngine::disabled(),
            master_volume,
            emitters: Vec::new(),
            cues: HashMap::new(),
            cue_clip_bytes: HashMap::new(),
            audio_cmd_cursor: EventCursor::default(),
            view_shown_cursor: EventCursor::default(),
            play_cue_cursor: EventCursor::default(),
            cues_matched: 0,
        }
    }

    // Number of cue bindings fired since init. Engine-independent progress the
    // schedule tests observe, since playback itself needs an output device.
    pub fn cues_matched(&self) -> usize {
        self.cues_matched
    }
}

impl System for AudioSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        // Acquire the output device (a disabled no-op engine when none is
        // available), deferred out of `new` so construction stays cheap.
        self.engine = AudioEngine::new();
        // Snapshot the emitters, then the clip payload locators indexed by
        // AudioClipHandle. The `AudioClipTable` resource is built from the blob's
        // resource stream, dense in handle order, so index N is the clip with
        // `AudioClipHandle(N)`. Collecting this owned Vec releases the resource
        // borrow before the `read_payload` calls below.
        let emitter_snaps: Vec<AudioEmitter> = ctx.query::<AudioEmitter>().cloned().collect();
        let clip_locators: Vec<Option<PayloadLocator>> = ctx
            .resource::<AudioClipTable>()
            .map(|table| table.0.iter().map(|e| e.payload.clone()).collect())
            .unwrap_or_default();

        // The persisted master volume (settings menu) scales every emitter via
        // the main mix track, so it can be changed live (see `step`). `None`
        // leaves output at unity. Clips play at their authored gain; the master
        // is a separate output-stage multiplier.
        self.engine
            .set_master_volume(self.master_volume.unwrap_or(1.0));

        for emitter in emitter_snaps {
            let Some(id) = self.engine.add_emitter(emitter.position) else {
                continue;
            };
            match emitter
                .clip
                .and_then(|clip| clip_locators.get(clip.index()).cloned().flatten())
            {
                Some(locator) => match ctx.read_payload(&locator) {
                    Ok(bytes) => {
                        self.engine
                            .play_clip(id, bytes, emitter.looping, emitter.volume);
                    }
                    Err(e) => tracing::warn!("AudioSystem: clip payload read failed: {e}"),
                },
                None => tracing::warn!(
                    "AudioSystem: emitter has no clip with a compiled payload, silent"
                ),
            }
            self.emitters.push(EmitterBinding {
                id,
                follows: emitter.prop,
            });
        }

        // Bind the screen-triggered cues and cache their clip payloads (keyed by
        // handle), so firing a cue never touches the blob mid-frame.
        let cue_snaps: Vec<AudioCue> = ctx.query::<AudioCue>().cloned().collect();
        for cue in cue_snaps {
            let (Some(screen), Some(clip)) = (cue.screen, cue.clip) else {
                tracing::warn!("AudioSystem: cue without a screen and a clip, ignored");
                continue;
            };
            if let std::collections::hash_map::Entry::Vacant(slot) = self.cue_clip_bytes.entry(clip)
            {
                match clip_locators.get(clip.index()).cloned().flatten() {
                    Some(locator) => match ctx.read_payload(&locator) {
                        Ok(bytes) => {
                            slot.insert(bytes.to_vec());
                        }
                        Err(e) => tracing::warn!("AudioSystem: cue payload read failed: {e}"),
                    },
                    None => {
                        tracing::warn!("AudioSystem: cue clip has no compiled payload, silent")
                    }
                }
            }
            self.cues.entry(screen).or_default().push(CueBinding {
                clip,
                kind: cue.kind,
                volume: cue.volume,
            });
        }

        // Stories and reactions play clips by direct PlayCue request rather
        // than through screen-keyed cues, so cache every clip payload up
        // front, keyed by handle.
        if ctx.query::<Story>().next().is_some()
            || ctx.query::<Behavior>().any(Behavior::plays_sound)
        {
            let uncached: Vec<(AudioClipHandle, PayloadLocator)> = clip_locators
                .iter()
                .enumerate()
                .filter_map(|(i, loc)| loc.clone().map(|l| (AudioClipHandle(i as u32), l)))
                .filter(|(handle, _)| !self.cue_clip_bytes.contains_key(handle))
                .collect();
            for (handle, locator) in uncached {
                match ctx.read_payload(&locator) {
                    Ok(bytes) => {
                        self.cue_clip_bytes.insert(handle, bytes.to_vec());
                    }
                    Err(e) => tracing::warn!("AudioSystem: story clip payload read failed: {e}"),
                }
            }
        }

        tracing::info!(
            "AudioSystem: {} emitter(s), {} cue screen(s), engine {}",
            self.emitters.len(),
            self.cues.len(),
            if self.engine.is_enabled() {
                "enabled"
            } else {
                "disabled"
            },
        );
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // Apply any live master-volume change sent this tick by GraphicsSystem,
        // which runs first. The last one this tick wins.
        if let Some(events) = ctx.events::<AudioCommand>() {
            for cmd in events.read(&mut self.audio_cmd_cursor) {
                self.engine.set_master_volume(cmd.master_volume);
            }
        }

        // Fire cues for screens shown since the last step. UiInputSystem runs
        // after this system, so a navigation is heard one tick later.
        if !self.cues.is_empty()
            && let Some(events) = ctx.events::<ScreenShown>()
        {
            let shown: Vec<AssetId> = events
                .read(&mut self.view_shown_cursor)
                .into_iter()
                .map(|e| e.screen)
                .collect();
            for screen in shown {
                let Some(bindings) = self.cues.get(&screen) else {
                    continue;
                };
                self.cues_matched += bindings.len();
                for cue in bindings {
                    let Some(bytes) = self.cue_clip_bytes.get(&cue.clip) else {
                        continue;
                    };
                    match cue.kind {
                        CueKind::Music => {
                            self.engine.play_music(cue.clip.0 as u64, bytes, cue.volume);
                        }
                        CueKind::Sound => {
                            self.engine.play_sound(bytes, cue.volume);
                        }
                    }
                }
            }
        }

        // Play direct requests (the story system's page audio). The story
        // system runs earlier in the schedule, so these are heard this tick.
        if let Some(events) = ctx.events::<PlayCue>() {
            let requests: Vec<PlayCue> = events
                .read(&mut self.play_cue_cursor)
                .into_iter()
                .copied()
                .collect();
            for cue in requests {
                self.cues_matched += 1;
                let Some(bytes) = self.cue_clip_bytes.get(&cue.clip) else {
                    continue;
                };
                match cue.kind {
                    CueKind::Music => {
                        self.engine.play_music(cue.clip.0 as u64, bytes, cue.volume);
                    }
                    CueKind::Sound => {
                        self.engine.play_sound(bytes, cue.volume);
                    }
                }
            }
        }

        // The listener rides the camera.
        if let Some((pos, yaw, pitch)) = ctx
            .query::<Camera3D>()
            .next()
            .map(|c| (c.position, c.yaw, c.pitch))
        {
            self.engine.set_listener(pos, yaw, pitch);
        }

        // Prop-bound emitters track their followed prop's current position, read
        // from its Transform via the name index.
        if self.emitters.iter().any(|b| b.follows.is_some()) {
            for binding in &self.emitters {
                if let Some(prop_id) = binding.follows
                    && let Some(entity) =
                        ctx.resource::<EntityByName>().and_then(|n| n.get(prop_id))
                    && let Some(t) = ctx.get::<Transform>(entity)
                {
                    self.engine.set_emitter_position(binding.id, t.position);
                }
            }
        }

        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    // These tests drive AudioSystem::init / step against a hand-built
    // PipelineContext and an in-memory blob, so no audio device is opened. They
    // assert on the engine-independent state the system tracks (its cue
    // bindings, cached payloads, and match counter) rather than on playback,
    // which needs a real device. Cue payloads are opaque to the caching path,
    // so any bytes serve. The gate/schedule tests (which need the engine's
    // `World`) live in the engine's `ecs/schedule.rs`.
    use super::{AudioSystem, EmitterBinding};
    use crate::EmitterId;
    use concinnity_core::assets::{
        AudioCommand, AudioCue, Camera3D, CueKind, PlayCue, ScreenShown, Story, Transform,
    };
    use concinnity_core::blob::BlobData;
    use concinnity_core::ecs::asset_id::AssetId;
    use concinnity_core::ecs::{
        AudioClipHandle, ComponentSlot, ComponentStorage, EntityByName, PayloadLocator,
        PipelineContext, ResourceKind, ResourceRecord, Resources, StepResult, System,
    };
    use concinnity_core::gfx::profile::FrameProfile;
    use concinnity_core::resource::AudioClipTable;

    // Accumulates audio components + one blob section serving every payload
    // locator handed out, plus the audio-clip resource records, then seals into a
    // context-owning world whose `AudioClipTable` is built from those records --
    // exactly as the runtime loads the blob's resource stream.
    struct AudioWorld {
        components: ComponentStorage,
        section: Vec<u8>,
        // The audio-clip resource records, in handle order (a clip added Nth is
        // handle N). Sealed into the world's `AudioClipTable`.
        clips: Vec<ResourceRecord>,
    }

    struct SealedAudio {
        components: ComponentStorage,
        blob: BlobData,
        profile: FrameProfile,
        resources: Resources,
    }

    impl AudioWorld {
        fn new() -> Self {
            Self {
                components: ComponentStorage::default(),
                section: Vec::new(),
                clips: Vec::new(),
            }
        }

        fn payload(&mut self, bytes: &[u8]) -> PayloadLocator {
            let offset = self.section.len() as u64;
            self.section.extend_from_slice(bytes);
            PayloadLocator {
                blob_index: 0,
                offset,
                len: bytes.len() as u64,
            }
        }

        fn push<C: ComponentSlot>(&mut self, c: C) {
            self.components.push_typed(c);
        }

        // Add an audio clip whose payload is `bytes`, returning its handle (its
        // record order, which the table indexes by).
        fn clip(&mut self, bytes: &[u8]) -> AudioClipHandle {
            let handle = AudioClipHandle(self.clips.len() as u32);
            let locator = self.payload(bytes);
            self.clips.push(ResourceRecord {
                resource_kind: ResourceKind::AudioClip as u8,
                handle: handle.0,
                payload: Some(locator),
                data_bytes: Vec::new(),
            });
            handle
        }

        fn seal(mut self) -> SealedAudio {
            let mut resources = Resources::new();
            resources.insert(AudioClipTable::from_records(&mut self.clips));
            SealedAudio {
                components: self.components,
                blob: BlobData::new(vec![Some(self.section)]),
                profile: FrameProfile::default(),
                resources,
            }
        }
    }

    impl SealedAudio {
        fn ctx(&mut self) -> PipelineContext<'_> {
            PipelineContext {
                components: &mut self.components,
                blob: &mut self.blob,
                profile: &mut self.profile,
                resources: &mut self.resources,
            }
        }
    }

    // init binds a screen-triggered cue and caches its clip payload up front, so
    // firing the cue later never touches the blob.
    #[test]
    fn init_binds_cue_and_caches_its_payload() {
        let screen = AssetId(90);
        let bytes = b"cue-clip-bytes";

        let mut w = AudioWorld::new();
        let clip = w.clip(bytes);
        w.push(AudioCue {
            screen: Some(screen),
            clip: Some(clip),
            kind: CueKind::Music,
            volume: 0.7,
            ..Default::default()
        });
        let mut sealed = w.seal();

        let mut sys = AudioSystem::new(None);
        sys.init(&mut sealed.ctx());

        let bindings = sys.cues.get(&screen).expect("cue bound to its screen");
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].clip, clip);
        assert_eq!(bindings[0].kind, CueKind::Music);
        assert!((bindings[0].volume - 0.7).abs() < 1.0e-6);
        assert_eq!(
            sys.cue_clip_bytes.get(&clip).map(Vec::as_slice),
            Some(&bytes[..])
        );
    }

    // A cue missing its screen or its clip is ignored: nothing is bound and no
    // payload is cached.
    #[test]
    fn init_ignores_cue_without_clip() {
        let mut w = AudioWorld::new();
        w.push(AudioCue {
            screen: Some(AssetId(90)),
            clip: None,
            ..Default::default()
        });
        let mut sealed = w.seal();

        let mut sys = AudioSystem::new(None);
        sys.init(&mut sealed.ctx());

        assert!(sys.cues.is_empty());
        assert!(sys.cue_clip_bytes.is_empty());
    }

    // A story caches every clip payload up front (it plays them by direct
    // PlayCue request, not through screen-keyed cues).
    #[test]
    fn init_caches_story_clip_payloads() {
        let bytes = b"story-page-audio";

        let mut w = AudioWorld::new();
        let clip = w.clip(bytes);
        w.push(Story::default());
        let mut sealed = w.seal();

        let mut sys = AudioSystem::new(None);
        sys.init(&mut sealed.ctx());

        assert!(sys.cues.is_empty(), "no cues declared");
        assert_eq!(
            sys.cue_clip_bytes.get(&clip).map(Vec::as_slice),
            Some(&bytes[..])
        );
    }

    // A shown screen fires each of its cues once, across both playback kinds; the
    // engine-independent match counter tracks the progress.
    #[test]
    fn step_fires_cued_view_across_both_kinds() {
        let screen = AssetId(90);

        let mut w = AudioWorld::new();
        let music_clip = w.clip(b"music");
        let sound_clip = w.clip(b"sound");
        w.push(AudioCue {
            screen: Some(screen),
            clip: Some(music_clip),
            kind: CueKind::Music,
            volume: 1.0,
            ..Default::default()
        });
        w.push(AudioCue {
            screen: Some(screen),
            clip: Some(sound_clip),
            kind: CueKind::Sound,
            volume: 1.0,
            ..Default::default()
        });
        let mut sealed = w.seal();

        let mut sys = AudioSystem::new(None);
        sys.init(&mut sealed.ctx());
        assert_eq!(sys.cues.get(&screen).map(Vec::len), Some(2));

        {
            let mut ctx = sealed.ctx();
            ctx.events_mut::<ScreenShown>().send(ScreenShown { screen });
        }
        assert_eq!(sys.step(&mut sealed.ctx()), StepResult::Continue);
        assert_eq!(sys.cues_matched, 2, "both the screen's cues matched");

        // A screen with no cue leaves the counter untouched.
        {
            let mut ctx = sealed.ctx();
            ctx.events_mut::<ScreenShown>().send(ScreenShown {
                screen: AssetId(999),
            });
        }
        assert_eq!(sys.step(&mut sealed.ctx()), StepResult::Continue);
        assert_eq!(sys.cues_matched, 2);
    }

    // A direct PlayCue request (the story system's page audio) is played the
    // same tick it is sent.
    #[test]
    fn step_plays_direct_play_cue_requests() {
        let mut w = AudioWorld::new();
        let clip = w.clip(b"page-audio");
        w.push(Story::default());
        let mut sealed = w.seal();

        let mut sys = AudioSystem::new(None);
        sys.init(&mut sealed.ctx());

        {
            let mut ctx = sealed.ctx();
            let events = ctx.events_mut::<PlayCue>();
            events.send(PlayCue {
                clip,
                kind: CueKind::Music,
                volume: 1.0,
            });
            events.send(PlayCue {
                clip,
                kind: CueKind::Sound,
                volume: 0.5,
            });
        }
        assert_eq!(sys.step(&mut sealed.ctx()), StepResult::Continue);
        assert_eq!(sys.cues_matched, 2, "both direct requests fired");
    }

    // A prop-bound emitter tracks its prop's Transform each step, and the
    // listener rides the camera. Neither needs a device; the step drives the
    // lookups and returns Continue.
    #[test]
    fn step_follows_prop_emitter_and_moves_listener() {
        let prop = AssetId(200);

        let mut sealed = AudioWorld::new().seal();
        let entity = {
            let mut ctx = sealed.ctx();
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
            let mut by_name = std::collections::BTreeMap::new();
            by_name.insert(prop, e);
            ctx.insert_resource(EntityByName(by_name));
            e
        };
        assert!(sealed.ctx().get::<Transform>(entity).is_some());

        let mut sys = AudioSystem::new(None);
        // A live emitter that follows the prop, seeded directly (a real
        // emitter needs a device, which the headless test has no access to).
        sys.emitters.push(EmitterBinding {
            id: EmitterId(0),
            follows: Some(prop),
        });

        assert_eq!(sys.step(&mut sealed.ctx()), StepResult::Continue);
    }

    // A master-volume AudioCommand sent mid-tick is read AND applied by the
    // audio system the same tick, so the new master takes effect without a
    // restart (the settings-menu master-volume row). `master_volume: None` at
    // construction means init leaves output at unity.
    #[test]
    fn audio_command_applies_master_volume_live() {
        let mut sealed = AudioWorld::new().seal();
        let mut sys = AudioSystem::new(None);
        sys.init(&mut sealed.ctx());
        // Init applied unity (no persisted master handed in).
        assert!((sys.engine.last_master_volume - 1.0).abs() < 1.0e-6);

        // GraphicsSystem would send this when the master-volume row is cycled;
        // the audio system reads it this same tick.
        {
            let mut ctx = sealed.ctx();
            ctx.events_mut::<AudioCommand>()
                .send(AudioCommand { master_volume: 0.5 });
        }
        assert_eq!(sys.step(&mut sealed.ctx()), StepResult::Continue);
        assert!(
            (sys.engine.last_master_volume - 0.5).abs() < 1.0e-6,
            "master volume should be applied live this tick"
        );
    }

    // Several AudioCommands sent in one tick (e.g. a rapid double-cycle) are
    // all read in order; the last one sent is applied last and wins.
    #[test]
    fn audio_command_last_write_wins_per_tick() {
        let mut sealed = AudioWorld::new().seal();
        let mut sys = AudioSystem::new(None);
        sys.init(&mut sealed.ctx());

        {
            let mut ctx = sealed.ctx();
            let events = ctx.events_mut::<AudioCommand>();
            events.send(AudioCommand { master_volume: 0.5 });
            events.send(AudioCommand {
                master_volume: 0.25,
            });
        }
        assert_eq!(sys.step(&mut sealed.ctx()), StepResult::Continue);
        assert!((sys.engine.last_master_volume - 0.25).abs() < 1.0e-6);
    }

    // A persisted master handed in at construction is applied to the mix at
    // init (the settings-menu master-volume, resolved by the engine's gate).
    #[test]
    fn init_applies_persisted_master_volume() {
        let mut sealed = AudioWorld::new().seal();
        let mut sys = AudioSystem::new(Some(0.5));
        sys.init(&mut sealed.ctx());
        assert!((sys.engine.last_master_volume - 0.5).abs() < 1.0e-6);
    }
}

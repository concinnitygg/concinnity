// concinnity-audio/src/system.rs
//
// Audio playback: 3D positional emitters and screen-triggered cues. An internal
// system (not a declarable asset): the engine schedule constructs one whenever
// the world contains any `AudioEmitter` or `AudioCue`, so a world with neither
// never opens an audio device.

use std::collections::HashSet;

use crate::occlusion::OcclusionSmoother;
use crate::{AudioEngine, AudioVolumes, EmitterId, EmitterParams};
use concinnity_core::assets::{
    AudioBus, AudioCommand, AudioCue, AudioEmitter, AudioOcclusionProbe, AudioTarget, Behavior,
    Camera3D, CueKind, PlayCue, ScreenShown, Story, Transform,
};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{
    AudioClipHandle, EntityByName, EventCursor, PayloadLocator, PipelineContext, SimTiming,
    StepResult, System,
};
use concinnity_core::resource::AudioClipTable;

// Audio behavior. Constructed internally by `World::start` when the world
// declares any `AudioEmitter` or `AudioCue`; never a world-declared asset, so
// it carries no config.
pub struct AudioSystem {
    engine: AudioEngine,
    // The persisted mix volumes (settings menu), applied at init. Resolved by
    // the engine's audio gate (which owns the settings store) and handed in
    // at construction, so this crate needs no dependency on the engine.
    volumes: AudioVolumes,
    // One entry per `AudioEmitter` in the world, in query order.
    emitters: Vec<EmitterBinding>,
    // Screen-triggered cues, keyed by the Screen whose activation fires them.
    cues: std::collections::HashMap<AssetId, Vec<CueBinding>>,
    // Last listener position handed to the engine; the occlusion probes ray
    // from it.
    listener_position: [f32; 3],
    // Cursor into the Events<AudioCommand> queue (live volume changes).
    audio_cmd_cursor: EventCursor,
    // Cursor into the Events<ScreenShown> queue (cue triggers).
    view_shown_cursor: EventCursor,
    // Cursor into the Events<PlayCue> queue (direct play requests, e.g. the
    // story system's page audio).
    play_cue_cursor: EventCursor,
    // Cues that matched a shown screen so far; observable engine-independent
    // progress for headless tests (playback needs a device and a payload).
    cues_matched: usize,
    // Distinct clips handed to the decode worker at init; the same kind of
    // engine-independent observable.
    clips_queued: usize,
}

// Links one engine emitter to the world data that positions it.
struct EmitterBinding {
    // `None` when the engine is disabled or kira's track limit was reached;
    // the binding still tracks position so probes stay uniform.
    id: Option<EmitterId>,
    // The Prop this emitter follows each frame, if any.
    follows: Option<AssetId>,
    // Current world position (authored, then updated from the followed prop).
    position: [f32; 3],
    // Smoothed occlusion state fed from this emitter's probe.
    occlusion: OcclusionSmoother,
}

// One AudioCue resolved to its clip and playback style.
struct CueBinding {
    clip: AudioClipHandle,
    kind: CueKind,
    volume: f32,
    bus: AudioBus,
    priority: i32,
}

// The bus a cue routes through when none is authored.
fn default_bus(kind: CueKind) -> AudioBus {
    match kind {
        CueKind::Music => AudioBus::Music,
        CueKind::Sound => AudioBus::Sfx,
    }
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
    // Fresh system with no device, live emitters, or cues. `volumes` holds
    // the persisted settings-menu mix (`None` per stage = unity), applied in
    // [`System::init`]. The output device is acquired and the emitters /
    // cues are bound from the world's components in `init`, so construction is
    // side-effect-free (required by the `World::system_manifest` gate probe).
    pub fn new(volumes: AudioVolumes) -> Self {
        Self {
            engine: AudioEngine::disabled(),
            volumes,
            emitters: Vec::new(),
            cues: std::collections::HashMap::new(),
            listener_position: [0.0; 3],
            audio_cmd_cursor: EventCursor::default(),
            view_shown_cursor: EventCursor::default(),
            play_cue_cursor: EventCursor::default(),
            cues_matched: 0,
            clips_queued: 0,
        }
    }

    // Number of cue bindings fired since init. Engine-independent progress the
    // schedule tests observe, since playback itself needs an output device.
    pub fn cues_matched(&self) -> usize {
        self.cues_matched
    }

    // Distinct clips handed to the decode worker at init; the same kind of
    // engine-independent observable.
    pub fn clips_queued(&self) -> usize {
        self.clips_queued
    }

    /// One-shot voices currently playing (or queued behind a clip decode),
    /// bounded by the engine's voice cap. Zero without an output device.
    pub fn playing_sounds(&self) -> usize {
        self.engine.playing_sounds()
    }

    // Read a clip's payload from the blob and hand it to the decode worker,
    // once per distinct clip. Returns false when the clip has no compiled
    // payload or the read failed (logged by the caller's context message).
    fn queue_clip(
        &mut self,
        ctx: &mut PipelineContext,
        locators: &[Option<PayloadLocator>],
        queued: &mut HashSet<AudioClipHandle>,
        clip: AudioClipHandle,
    ) -> bool {
        if !queued.insert(clip) {
            return true;
        }
        let Some(locator) = locators.get(clip.index()).cloned().flatten() else {
            queued.remove(&clip);
            return false;
        };
        match ctx.read_payload(&locator) {
            Ok(bytes) => {
                self.engine.queue_clip(clip.0 as u64, bytes.to_vec());
                self.clips_queued += 1;
                true
            }
            Err(e) => {
                tracing::warn!("AudioSystem: clip payload read failed: {e}");
                queued.remove(&clip);
                false
            }
        }
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

        // The persisted volumes (settings menu) scale the master mix and each
        // bus, so they can be changed live (see `step`). `None` leaves a
        // stage at unity. Clips play at their authored gain; these are
        // separate output-stage multipliers.
        self.engine
            .set_volume(AudioTarget::Master, self.volumes.master.unwrap_or(1.0));
        self.engine
            .set_volume(AudioTarget::Music, self.volumes.music.unwrap_or(1.0));
        self.engine
            .set_volume(AudioTarget::Sfx, self.volumes.sfx.unwrap_or(1.0));
        self.engine
            .set_volume(AudioTarget::Voice, self.volumes.voice.unwrap_or(1.0));

        let mut queued: HashSet<AudioClipHandle> = HashSet::new();

        for (index, emitter) in emitter_snaps.into_iter().enumerate() {
            let params = EmitterParams {
                min_distance: emitter.min_distance,
                max_distance: emitter.max_distance,
                rolloff: emitter.rolloff,
                bus: emitter.bus.unwrap_or(AudioBus::Sfx),
            };
            let id = self.engine.add_emitter(emitter.position, &params);
            match emitter.clip {
                Some(clip) => {
                    if self.queue_clip(ctx, &clip_locators, &mut queued, clip)
                        && let Some(id) = id
                    {
                        // Starts on the tick the decode lands.
                        self.engine.play_emitter_clip(
                            id,
                            clip.0 as u64,
                            emitter.looping,
                            emitter.volume,
                        );
                    }
                }
                None => tracing::warn!(
                    "AudioSystem: emitter has no clip with a compiled payload, silent"
                ),
            }
            // One occlusion probe per emitter; PhysicsSystem answers it each
            // frame when the world simulates physics.
            ctx.push(AudioOcclusionProbe {
                emitter: index as u32,
                from: [0.0; 3],
                to: emitter.position,
                blocked: None,
            });
            self.emitters.push(EmitterBinding {
                id,
                follows: emitter.prop,
                position: emitter.position,
                occlusion: OcclusionSmoother::new(),
            });
        }

        // Bind the screen-triggered cues and queue their clips for decode, so
        // firing a cue never touches the blob mid-frame.
        let cue_snaps: Vec<AudioCue> = ctx.query::<AudioCue>().cloned().collect();
        for cue in cue_snaps {
            let (Some(screen), Some(clip)) = (cue.screen, cue.clip) else {
                tracing::warn!("AudioSystem: cue without a screen and a clip, ignored");
                continue;
            };
            if !self.queue_clip(ctx, &clip_locators, &mut queued, clip) {
                tracing::warn!("AudioSystem: cue clip has no compiled payload, silent");
            }
            self.cues.entry(screen).or_default().push(CueBinding {
                clip,
                kind: cue.kind,
                volume: cue.volume,
                bus: cue.bus.unwrap_or(default_bus(cue.kind)),
                priority: cue.priority,
            });
        }

        // Stories and behaviors play clips by direct PlayCue request rather
        // than through screen-keyed cues, so queue every clip up front.
        if ctx.query::<Story>().next().is_some()
            || ctx.query::<Behavior>().any(Behavior::plays_sound)
        {
            let clips: Vec<AudioClipHandle> = clip_locators
                .iter()
                .enumerate()
                .filter_map(|(i, loc)| loc.as_ref().map(|_| AudioClipHandle(i as u32)))
                .collect();
            for clip in clips {
                if !self.queue_clip(ctx, &clip_locators, &mut queued, clip) {
                    tracing::warn!("AudioSystem: story clip payload read failed");
                }
            }
        }

        tracing::info!(
            "AudioSystem: {} emitter(s), {} cue screen(s), {} clip(s) decoding, engine {}",
            self.emitters.len(),
            self.cues.len(),
            self.clips_queued,
            if self.engine.is_enabled() {
                "enabled"
            } else {
                "disabled"
            },
        );
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // Drain finished decodes (starting deferred plays) and release the
        // voice slots of finished one-shots.
        self.engine.tick();

        // Apply any live volume change sent this tick by GraphicsSystem,
        // which runs first. The last one per target this tick wins.
        if let Some(events) = ctx.events::<AudioCommand>() {
            for cmd in events.read(&mut self.audio_cmd_cursor) {
                self.engine.set_volume(cmd.target, cmd.gain);
            }
        }

        // Fire cues for screens shown since the last step. UiInputSystem runs
        // after this system, so a navigation is heard one tick later.
        if !self.cues.is_empty()
            && let Some(events) = ctx.events::<ScreenShown>()
        {
            let shown: Vec<AssetId> = events
                .read(&mut self.view_shown_cursor)
                .map(|e| e.screen)
                .collect();
            for screen in shown {
                let Some(bindings) = self.cues.get(&screen) else {
                    continue;
                };
                self.cues_matched += bindings.len();
                for cue in bindings {
                    let key = cue.clip.0 as u64;
                    match cue.kind {
                        CueKind::Music => {
                            self.engine.play_music(key, cue.volume, cue.bus);
                        }
                        CueKind::Sound => {
                            self.engine
                                .play_sound(key, cue.volume, cue.bus, cue.priority);
                        }
                    }
                }
            }
        }

        // Play direct requests (the story system's page audio). The story
        // system runs earlier in the schedule, so these are heard this tick.
        if let Some(events) = ctx.events::<PlayCue>() {
            let requests: Vec<PlayCue> = events.read(&mut self.play_cue_cursor).copied().collect();
            for cue in requests {
                self.cues_matched += 1;
                let key = cue.clip.0 as u64;
                match cue.kind {
                    CueKind::Music => {
                        self.engine.play_music(key, cue.volume, AudioBus::Music);
                    }
                    CueKind::Sound => {
                        self.engine
                            .play_sound(key, cue.volume, AudioBus::Sfx, cue.priority);
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
            self.listener_position = pos;
            self.engine.set_listener(pos, yaw, pitch);
        }

        // Prop-bound emitters track their followed prop's current position,
        // read from its Transform via the name index.
        if self.emitters.iter().any(|b| b.follows.is_some()) {
            for binding in &mut self.emitters {
                if let Some(prop_id) = binding.follows
                    && let Some(entity) =
                        ctx.resource::<EntityByName>().and_then(|n| n.get(prop_id))
                    && let Some(t) = ctx.get::<Transform>(entity)
                {
                    binding.position = t.position;
                    if let Some(id) = binding.id {
                        self.engine.set_emitter_position(id, t.position);
                    }
                }
            }
        }

        // Feed each emitter's occlusion probe (physics answered last frame's
        // ray) through its smoother and into the mixer, and refresh the ray
        // endpoints for the next physics step. An unanswered probe (no
        // physics in the world) reads as clear.
        for probe in ctx.query_mut::<AudioOcclusionProbe>() {
            let Some(binding) = self.emitters.get_mut(probe.emitter as usize) else {
                continue;
            };
            probe.from = self.listener_position;
            probe.to = binding.position;
            let factor = binding
                .occlusion
                .step(probe.blocked.unwrap_or(false), SimTiming::TICK_DT);
            if let Some(id) = binding.id {
                self.engine.set_emitter_occlusion(id, factor);
            }
        }

        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    // These tests drive AudioSystem::init / step against a hand-built
    // PipelineContext and an in-memory blob, so no audio playback happens
    // (init may still probe for a device on a dev machine; every assertion
    // here is engine-independent). They assert on the state the system
    // tracks: cue bindings, queued-clip and match counters, emitter bindings,
    // and the occlusion probes. The gate/schedule tests (which need the
    // engine's `World`) live in the engine's `ecs/schedule.rs`.
    use super::{AudioSystem, EmitterBinding};
    use crate::occlusion::OcclusionSmoother;
    use crate::{AudioVolumes, EmitterId};
    use concinnity_core::assets::{
        AudioBus, AudioCommand, AudioCue, AudioEmitter, AudioOcclusionProbe, AudioTarget, Camera3D,
        CueKind, PlayCue, ScreenShown, Story, Transform,
    };
    use concinnity_core::ecs::asset_id::AssetId;
    use concinnity_core::ecs::{
        Arena, AudioClipHandle, ComponentSlot, ComponentStorage, EntityByName, FrameContext,
        PayloadLocator, PipelineContext, ResourceKind, ResourceRecord, Resources, StepResult,
        System,
    };
    use concinnity_core::gfx::profile::FrameProfile;
    use concinnity_core::resource::AudioClipTable;
    use concinnity_store::blob::BlobData;

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
        scratch: Arena,
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
                scratch: Arena::with_capacity(64 * 1024),
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
                frame: FrameContext::new(&self.scratch),
            }
        }
    }

    // init binds a screen-triggered cue with its routing defaults and hands
    // its clip to the decode worker once, so firing the cue later never
    // touches the blob.
    #[test]
    fn init_binds_cue_and_queues_its_clip() {
        let screen = AssetId(90);

        let mut w = AudioWorld::new();
        let clip = w.clip(b"cue-clip-bytes");
        w.push(AudioCue {
            screen: Some(screen),
            clip: Some(clip),
            kind: CueKind::Music,
            volume: 0.7,
            ..Default::default()
        });
        let mut sealed = w.seal();

        let mut sys = AudioSystem::new(AudioVolumes::default());
        sys.init(&mut sealed.ctx());

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

        let mut w = AudioWorld::new();
        let clip = w.clip(b"line-bytes");
        w.push(AudioCue {
            screen: Some(screen),
            clip: Some(clip),
            kind: CueKind::Sound,
            bus: Some(AudioBus::Voice),
            priority: 4,
            ..Default::default()
        });
        let mut sealed = w.seal();

        let mut sys = AudioSystem::new(AudioVolumes::default());
        sys.init(&mut sealed.ctx());

        let bindings = sys.cues.get(&screen).expect("cue bound");
        assert_eq!(bindings[0].bus, AudioBus::Voice);
        assert_eq!(bindings[0].priority, 4);
    }

    // A cue missing its screen or its clip is ignored: nothing is bound and no
    // clip is queued.
    #[test]
    fn init_ignores_cue_without_clip() {
        let mut w = AudioWorld::new();
        w.push(AudioCue {
            screen: Some(AssetId(90)),
            clip: None,
            ..Default::default()
        });
        let mut sealed = w.seal();

        let mut sys = AudioSystem::new(AudioVolumes::default());
        sys.init(&mut sealed.ctx());

        assert!(sys.cues.is_empty());
        assert_eq!(sys.clips_queued(), 0);
    }

    // A story queues every clip up front (it plays them by direct PlayCue
    // request, not through screen-keyed cues), each clip exactly once even
    // when a cue already queued it.
    #[test]
    fn init_queues_story_clips_without_duplicates() {
        let screen = AssetId(90);

        let mut w = AudioWorld::new();
        let cue_clip = w.clip(b"cue-audio");
        let page_clip = w.clip(b"story-page-audio");
        w.push(AudioCue {
            screen: Some(screen),
            clip: Some(cue_clip),
            ..Default::default()
        });
        w.push(Story::default());
        let mut sealed = w.seal();

        let mut sys = AudioSystem::new(AudioVolumes::default());
        sys.init(&mut sealed.ctx());

        let _ = page_clip;
        assert_eq!(sys.clips_queued(), 2, "both clips queued, neither twice");
    }

    // init binds every emitter (device or not), queues its clip, and
    // publishes one occlusion probe per emitter aimed at it.
    #[test]
    fn init_binds_emitters_and_publishes_occlusion_probes() {
        let mut w = AudioWorld::new();
        let clip = w.clip(b"loop-bytes");
        w.push(AudioEmitter {
            clip: Some(clip),
            position: [3.0, 1.0, -2.0],
            ..Default::default()
        });
        let mut sealed = w.seal();

        let mut sys = AudioSystem::new(AudioVolumes::default());
        sys.init(&mut sealed.ctx());

        assert_eq!(sys.emitters.len(), 1);
        assert_eq!(sys.emitters[0].position, [3.0, 1.0, -2.0]);
        assert_eq!(sys.clips_queued(), 1);

        let ctx = sealed.ctx();
        let probes: Vec<&AudioOcclusionProbe> = ctx.query::<AudioOcclusionProbe>().collect();
        assert_eq!(probes.len(), 1, "one probe per emitter");
        assert_eq!(probes[0].emitter, 0);
        assert_eq!(probes[0].to, [3.0, 1.0, -2.0]);
        assert_eq!(probes[0].blocked, None, "unanswered until physics steps");
    }

    // A blocked probe answer muffles gradually: the smoothed factor rises
    // over several steps instead of slamming to 1, and the system refreshes
    // the probe's ray endpoints each step.
    #[test]
    fn step_smooths_probe_answers_and_refreshes_endpoints() {
        let mut w = AudioWorld::new();
        let clip = w.clip(b"loop-bytes");
        w.push(AudioEmitter {
            clip: Some(clip),
            position: [5.0, 0.0, 0.0],
            ..Default::default()
        });
        let mut sealed = w.seal();

        let mut sys = AudioSystem::new(AudioVolumes::default());
        sys.init(&mut sealed.ctx());

        // Physics answered: the segment is blocked.
        {
            let mut ctx = sealed.ctx();
            for probe in ctx.query_mut::<AudioOcclusionProbe>() {
                probe.blocked = Some(true);
            }
        }
        assert_eq!(sys.step(&mut sealed.ctx()), StepResult::Continue);
        let after_one = sys.emitters[0].occlusion.current();
        assert!(
            after_one > 0.0 && after_one < 0.5,
            "one tick moves partway: {after_one}"
        );
        for _ in 0..120 {
            sys.step(&mut sealed.ctx());
        }
        assert!(sys.emitters[0].occlusion.current() > 0.9, "settles blocked");

        // The ray endpoints track the listener and emitter for next frame.
        let ctx = sealed.ctx();
        let probe = ctx.query::<AudioOcclusionProbe>().next().unwrap();
        assert_eq!(probe.from, [0.0; 3], "no camera: listener at origin");
        assert_eq!(probe.to, [5.0, 0.0, 0.0]);
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

        let mut sys = AudioSystem::new(AudioVolumes::default());
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

        let mut sys = AudioSystem::new(AudioVolumes::default());
        sys.init(&mut sealed.ctx());

        {
            let mut ctx = sealed.ctx();
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

        let mut sys = AudioSystem::new(AudioVolumes::default());
        // A live emitter that follows the prop, seeded directly (a real
        // emitter needs a device, which the headless test has no access to).
        sys.emitters.push(EmitterBinding {
            id: Some(EmitterId(0)),
            follows: Some(prop),
            position: [0.0; 3],
            occlusion: OcclusionSmoother::new(),
        });

        assert_eq!(sys.step(&mut sealed.ctx()), StepResult::Continue);
        assert_eq!(
            sys.emitters[0].position,
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
        let mut sealed = AudioWorld::new().seal();
        let mut sys = AudioSystem::new(AudioVolumes::default());
        sys.init(&mut sealed.ctx());
        // Init applied unity everywhere (no persisted volumes handed in).
        assert_eq!(sys.engine.last_volume(AudioTarget::Master), 1.0);
        assert_eq!(sys.engine.last_volume(AudioTarget::Voice), 1.0);

        // GraphicsSystem would send these when volume rows are cycled; the
        // audio system reads them this same tick.
        {
            let mut ctx = sealed.ctx();
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
        assert_eq!(sys.step(&mut sealed.ctx()), StepResult::Continue);
        assert_eq!(sys.engine.last_volume(AudioTarget::Master), 0.5);
        assert_eq!(sys.engine.last_volume(AudioTarget::Music), 0.25);
        assert_eq!(sys.engine.last_volume(AudioTarget::Sfx), 1.0, "untouched");
    }

    // Several AudioCommands for one target sent in one tick (e.g. a rapid
    // double-cycle) are all read in order; the last one sent wins.
    #[test]
    fn audio_command_last_write_wins_per_tick() {
        let mut sealed = AudioWorld::new().seal();
        let mut sys = AudioSystem::new(AudioVolumes::default());
        sys.init(&mut sealed.ctx());

        {
            let mut ctx = sealed.ctx();
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
        assert_eq!(sys.step(&mut sealed.ctx()), StepResult::Continue);
        assert_eq!(sys.engine.last_volume(AudioTarget::Master), 0.25);
    }

    // Persisted volumes handed in at construction are applied to every stage
    // at init (the settings-menu volumes, resolved by the engine's gate).
    #[test]
    fn init_applies_persisted_volumes() {
        let mut sealed = AudioWorld::new().seal();
        let mut sys = AudioSystem::new(AudioVolumes {
            master: Some(0.5),
            music: Some(0.75),
            sfx: None,
            voice: Some(0.25),
        });
        sys.init(&mut sealed.ctx());
        assert_eq!(sys.engine.last_volume(AudioTarget::Master), 0.5);
        assert_eq!(sys.engine.last_volume(AudioTarget::Music), 0.75);
        assert_eq!(
            sys.engine.last_volume(AudioTarget::Sfx),
            1.0,
            "unset = unity"
        );
        assert_eq!(sys.engine.last_volume(AudioTarget::Voice), 0.25);
    }
}

//! Audio playback: 3D positional emitters, screen-triggered cues, story and
//! behavior plays and impact one-shots, over the [`AudioOut`] a host lends. An
//! internal system (not a declarable asset), present only when the world has
//! something to play, so a world with nothing never opens an audio device.

use crate::math::SaturatingFrom;
use alloc::boxed::Box;
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};

use super::mixer::Mixer;
use super::occlusion::OcclusionSmoother;
use super::out::{Acoustics, AudioOut};
use super::{EmitterId, impact};
use crate::components::{
    AudioBus, AudioCommand, AudioCue, AudioEmitter, AudioOcclusionProbe, AudioTarget, Behavior,
    BodyDynamics, Camera3D, ContactEvent, CueKind, PlayCue, PropBody, ScreenShown, Story,
    Transform,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{
    AudioClipHandle, Entity, EventCursor, PayloadLocator, PipelineContext, Ref, SimTiming,
    StepResult, System, World,
};
use crate::planet::{FrameRebases, Rebase, RebaseCursor};
use crate::resource::AudioClipTable;
use crate::settings::StoredSettings;

/// Opens the host's audio output, or `None` when there is no device.
pub type OpenAudio = fn() -> Option<Box<dyn AudioOut>>;

/// Plays a world's sounds. Its init opens the output.
pub struct AudioSystem {
    open: OpenAudio,
    mixer: Mixer,
    // One binding per live `AudioEmitter`, keyed by its entity. Emitters that
    // appear after init (runtime adds) are adopted each step; bindings whose
    // entity died are reaped, mirroring the physics prop-body lifecycle.
    emitters: HashMap<Entity, EmitterBinding>,
    // Screen-triggered cues, keyed by the Screen whose activation fires them.
    cues: HashMap<AssetId, Vec<CueBinding>>,
    // Clip payload locators snapshotted at init, indexed by handle, so a
    // runtime-adopted emitter can still queue its clip.
    clip_locators: Vec<Option<PayloadLocator>>,
    // Clips already handed to the decode worker.
    queued: HashSet<AudioClipHandle>,
    // Last listener position handed to the mixer; the occlusion probes ray
    // from it.
    listener_position: [f32; 3],
    // Cursor into the Events<AudioCommand> queue (live volume changes).
    audio_cmd_cursor: EventCursor,
    // Cursor into the Events<ScreenShown> queue (cue triggers).
    view_shown_cursor: EventCursor,
    // Cursor into the Events<PlayCue> queue (direct play requests, e.g. the
    // story system's page audio).
    play_cue_cursor: EventCursor,
    // Cursor into the Events<ContactEvent> queue (impact one-shots).
    contact_cursor: EventCursor,
    // Cues that matched a shown screen so far; observable output-independent
    // progress for headless tests (playback needs a device and a payload).
    cues_matched: usize,
    // Distinct clips handed to the decode worker; the same kind of
    // output-independent observable.
    clips_queued: usize,
    // Contacts that resolved to an impact clip; same kind of observable.
    impacts_played: usize,
    // How far the emitters have caught up with the moves of a planet world's
    // simulated frame.
    rebases: RebaseCursor,
}

// Links one mixer emitter to the world data that positions it.
struct EmitterBinding {
    // `None` with no output open, or when the output had no room for a track;
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

impl core::fmt::Debug for AudioSystem {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AudioSystem")
            .field("mixer", &self.mixer)
            .field("emitters", &self.emitters.len())
            .field("cue_views", &self.cues.len())
            .finish_non_exhaustive()
    }
}

impl AudioSystem {
    /// The system for a world with anything to play: an `AudioEmitter`
    /// (positional sound), an `AudioCue` (screen-triggered sound), a `Story`
    /// with page audio, a `Behavior` with a sound node, or a `PropBody` with an
    /// impact sound. Its init opens the output with `open`, so a world with
    /// none of them stays silent and device-free.
    pub fn for_world(world: &World, open: OpenAudio) -> Option<Self> {
        let needs = world.query::<AudioEmitter>().next().is_some()
            || world.query::<AudioCue>().next().is_some()
            || world
                .query::<Story>()
                .next()
                .is_some_and(Story::plays_audio)
            || world.query::<Behavior>().any(Behavior::plays_sound)
            || world.query::<PropBody>().any(|b| b.impact_clip.is_some());
        needs.then(|| Self::new(open))
    }

    // Fresh system with no output, live emitters, or cues. The output is
    // opened and the emitters / cues are bound from the world's components in
    // `init`, so construction is side-effect-free (required by the
    // `World::system_manifest` gate probe).
    fn new(open: OpenAudio) -> Self {
        Self {
            open,
            mixer: Mixer::new(None),
            emitters: HashMap::new(),
            cues: HashMap::new(),
            clip_locators: Vec::new(),
            queued: HashSet::new(),
            listener_position: [0.0; 3],
            audio_cmd_cursor: EventCursor::default(),
            view_shown_cursor: EventCursor::default(),
            play_cue_cursor: EventCursor::default(),
            contact_cursor: EventCursor::default(),
            cues_matched: 0,
            clips_queued: 0,
            impacts_played: 0,
            rebases: RebaseCursor::default(),
        }
    }

    /// Cues fired since init, whether or not an output played them.
    pub fn cues_matched(&self) -> usize {
        self.cues_matched
    }

    // Distinct clips handed to the decode worker at init; the same kind of
    // output-independent observable.
    #[cfg(test)]
    pub(crate) fn clips_queued(&self) -> usize {
        self.clips_queued
    }

    // Contacts that resolved to an impact clip; the same kind of
    // output-independent observable as `cues_matched`.
    #[cfg(test)]
    pub(crate) fn impacts_played(&self) -> usize {
        self.impacts_played
    }

    // Read a clip's payload from the blob and hand it to the decode worker,
    // once per distinct clip. Returns false when the clip has no compiled
    // payload or the read failed (logged by the caller's context message).
    fn queue_clip(&mut self, ctx: &mut PipelineContext<'_>, clip: AudioClipHandle) -> bool {
        if !self.queued.insert(clip) {
            return true;
        }
        let Some(locator) = self.clip_locators.get(clip.index()).cloned().flatten() else {
            self.queued.remove(&clip);
            return false;
        };
        match ctx.read_payload(&locator) {
            Ok(bytes) => {
                self.mixer.queue_clip(u64::from(clip.0), bytes.to_vec());
                self.clips_queued += 1;
                true
            }
            Err(e) => {
                tracing::warn!("audio: clip payload read failed: {e}");
                self.queued.remove(&clip);
                false
            }
        }
    }

    // Bind one `AudioEmitter` living on `entity`: create its mixer emitter,
    // queue and start its clip, and attach its occlusion probe to the entity
    // (so the probe despawns with it). Shared by init and runtime adoption.
    fn bind_emitter(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        entity: Entity,
        emitter: &AudioEmitter,
    ) {
        let acoustics = Acoustics {
            bus: emitter.bus.unwrap_or(AudioBus::Sfx),
            distances: (emitter.min_distance, emitter.max_distance),
            rolloff: emitter.rolloff,
        };
        let id = self.mixer.add_emitter(emitter.position, acoustics);
        match emitter.clip {
            Some(clip) => {
                if self.queue_clip(ctx, clip)
                    && let Some(id) = id
                {
                    // Starts on the tick the decode lands.
                    self.mixer.play_emitter_clip(
                        id,
                        u64::from(clip.0),
                        emitter.looping,
                        emitter.volume,
                    );
                }
            }
            None => {
                tracing::warn!("audio: emitter has no clip with a compiled payload, silent");
            }
        }
        // Physics answers the probe each frame when the world
        // simulates physics.
        if ctx.get::<AudioOcclusionProbe>(entity).is_none() {
            ctx.insert(
                entity,
                AudioOcclusionProbe {
                    from: [0.0; 3],
                    to: emitter.position,
                    blocked: None,
                },
            );
        }
        self.emitters.insert(
            entity,
            EmitterBinding {
                id,
                follows: emitter.prop.map(Ref::id),
                position: emitter.position,
                occlusion: OcclusionSmoother::default(),
            },
        );
    }
}

impl AudioSystem {
    // Carry the free-standing emitters, and the components they were bound
    // from, into the frame `rebase` moves the world to.
    fn carry(&mut self, ctx: &mut PipelineContext<'_>, rebase: &Rebase) {
        for (&entity, binding) in &mut self.emitters {
            if binding.follows.is_some() {
                continue;
            }
            binding.position = rebase.apply_point(binding.position);
            if let Some(id) = binding.id {
                self.mixer.set_emitter_position(id, binding.position);
            }
            if let Some(emitter) = ctx.get_mut::<AudioEmitter>(entity) {
                emitter.position = binding.position;
            }
        }
    }
}

impl System for AudioSystem {
    fn init(&mut self, ctx: &mut PipelineContext<'_>) {
        // Open the output (an inert mixer when there is no device), deferred
        // out of `new` so construction stays cheap.
        self.mixer = Mixer::new((self.open)());
        self.rebases = RebaseCursor::at(ctx.resource::<FrameRebases>());
        // Snapshot the emitters, then the clip payload locators indexed by
        // AudioClipHandle. The `AudioClipTable` resource is built from the blob's
        // resource stream, dense in handle order, so index N is the clip with
        // `AudioClipHandle::new(N)`. Collecting this owned Vec releases the resource
        // borrow before the `read_payload` calls below.
        let emitter_snaps: Vec<(Entity, AudioEmitter)> = ctx
            .query_with_entity::<AudioEmitter>()
            .map(|(entity, e)| (entity, e.clone()))
            .collect();
        self.clip_locators = ctx
            .resource::<AudioClipTable>()
            .map(|table| table.0.iter().map(|e| e.payload.clone()).collect())
            .unwrap_or_default();

        // The stored settings-menu volumes scale the master mix and each bus,
        // so they can be changed live (see `step`). `None` leaves a stage at
        // unity. Clips play at their authored gain; these are separate
        // output-stage multipliers.
        let volumes = ctx
            .resource::<StoredSettings>()
            .map(|stored| stored.audio.clone())
            .unwrap_or_default();
        for (target, gain) in [
            (AudioTarget::Master, volumes.master_volume),
            (AudioTarget::Music, volumes.music_volume),
            (AudioTarget::Sfx, volumes.sfx_volume),
            (AudioTarget::Voice, volumes.voice_volume),
        ] {
            self.mixer.set_volume(target, gain.unwrap_or(1.0));
        }

        for (entity, emitter) in emitter_snaps {
            self.bind_emitter(ctx, entity, &emitter);
        }

        // Bind the screen-triggered cues and queue their clips for decode, so
        // firing a cue never touches the blob mid-frame.
        let cue_snaps: Vec<AudioCue> = ctx.query::<AudioCue>().cloned().collect();
        for cue in cue_snaps {
            let (Some(screen), Some(clip)) = (cue.screen, cue.clip) else {
                tracing::warn!("audio: cue without a screen and a clip, ignored");
                continue;
            };
            if !self.queue_clip(ctx, clip) {
                tracing::warn!("audio: cue clip has no compiled payload, silent");
            }
            self.cues.entry(screen.id()).or_default().push(CueBinding {
                clip,
                kind: cue.kind,
                volume: cue.volume,
                bus: cue.bus.unwrap_or_else(|| default_bus(cue.kind)),
                priority: cue.priority,
            });
        }

        // Stories and behaviors play clips by direct PlayCue request, and
        // physics contacts play whatever impact clip the colliding body
        // carries, so queue every remaining clip up front. Blob payloads are
        // freed after init, so a clip skipped here cannot be queued later.
        let plays_arbitrary_clips = ctx.query::<Story>().next().is_some()
            || ctx.query::<Behavior>().any(Behavior::plays_sound)
            || ctx.query::<BodyDynamics>().any(|d| d.impact_clip.is_some());
        if plays_arbitrary_clips {
            let clips: Vec<AudioClipHandle> = self
                .clip_locators
                .iter()
                .enumerate()
                .filter_map(|(i, loc)| {
                    loc.as_ref()
                        .map(|_| AudioClipHandle::new(u32::saturating_from(i)))
                })
                .collect();
            for clip in clips {
                if !self.queue_clip(ctx, clip) {
                    tracing::warn!("audio: clip payload read failed");
                }
            }
        }

        tracing::info!(
            "audio: {} emitter(s), {} cue screen(s), {} clip(s) decoding, output {}",
            self.emitters.len(),
            self.cues.len(),
            self.clips_queued,
            if self.mixer.is_enabled() {
                "enabled"
            } else {
                "disabled"
            },
        );
    }

    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        // Drain finished decodes (starting deferred plays) and release the
        // voice slots of finished one-shots.
        self.mixer.tick();

        // Reap bindings whose entity despawned (the spatial track stops and
        // unloads; the probe died with the entity), then adopt emitters that
        // appeared since init -- the same seen-set lifecycle the physics
        // system runs for prop bodies.
        let dead: Vec<Entity> = self
            .emitters
            .keys()
            .filter(|entity| !ctx.is_alive(**entity))
            .copied()
            .collect();
        for entity in dead {
            if let Some(binding) = self.emitters.remove(&entity)
                && let Some(id) = binding.id
            {
                self.mixer.remove_emitter(id);
            }
        }
        let adopted: Vec<(Entity, AudioEmitter)> = ctx
            .query_with_entity::<AudioEmitter>()
            .filter(|(entity, _)| !self.emitters.contains_key(entity))
            .map(|(entity, e)| (entity, e.clone()))
            .collect();
        for (entity, emitter) in adopted {
            self.bind_emitter(ctx, entity, &emitter);
        }

        // Apply any live volume change the settings menu sent. The last one
        // per target wins.
        if let Some(events) = ctx.events::<AudioCommand>() {
            for cmd in events.read(&mut self.audio_cmd_cursor) {
                self.mixer.set_volume(cmd.target, cmd.gain);
            }
        }

        // Fire cues for screens shown since the last step. UI input runs
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
                    let key = u64::from(cue.clip.0);
                    match cue.kind {
                        CueKind::Music => {
                            self.mixer.play_music(key, cue.volume, cue.bus);
                        }
                        CueKind::Sound => {
                            self.mixer
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
                let key = u64::from(cue.clip.0);
                match cue.kind {
                    CueKind::Music => {
                        self.mixer.play_music(key, cue.volume, AudioBus::Music);
                    }
                    CueKind::Sound => {
                        self.mixer
                            .play_sound(key, cue.volume, AudioBus::Sfx, cue.priority);
                    }
                }
            }
        }

        // Impact one-shots: a contact plays each colliding body's authored
        // impact clip at the contact point, scaled by the impulse. Physics
        // gates and debounces the events; the voice pool caps bursts. The
        // events copy into frame scratch to release the queue borrow before
        // the component reads below.
        let frame = ctx.frame;
        if let Some(events) = ctx.events::<ContactEvent>() {
            let contacts = frame.collect(events.read(&mut self.contact_cursor).copied());
            for &contact in &contacts {
                for entity in [Some(contact.a), contact.b].into_iter().flatten() {
                    let Some(dynamics) = ctx.get::<BodyDynamics>(entity) else {
                        continue;
                    };
                    let Some(clip) = dynamics.impact_clip else {
                        continue;
                    };
                    let gain = impact::gain(contact.impulse) * dynamics.impact_volume;
                    self.impacts_played += 1;
                    self.mixer
                        .play_sound_at(contact.point, u64::from(clip.0), gain, 0);
                }
            }
        }

        // A planet world's frame moved: every emitter that stands on its own
        // moves with it (a prop-bound one follows its prop below).
        if let Some(rebase) = self.rebases.take(ctx.resource::<FrameRebases>()) {
            self.carry(ctx, &rebase);
        }

        // The listener rides the camera.
        if let Some((pos, yaw, pitch)) = ctx
            .query::<Camera3D>()
            .next()
            .map(|c| (c.position, c.yaw, c.pitch))
        {
            self.listener_position = pos;
            self.mixer.set_listener(pos, yaw, pitch);
        }

        // Prop-bound emitters track their followed prop's current position,
        // read from its Transform via the name index.
        if self.emitters.values().any(|b| b.follows.is_some()) {
            for binding in self.emitters.values_mut() {
                if let Some(prop_id) = binding.follows
                    && let Some(entity) = ctx.entity_of(prop_id)
                    && let Some(t) = ctx.get::<Transform>(entity)
                {
                    binding.position = t.position;
                    if let Some(id) = binding.id {
                        self.mixer.set_emitter_position(id, t.position);
                    }
                }
            }
        }

        // Feed each emitter's occlusion probe (physics answered last frame's
        // ray) through its smoother and into the mixer, and refresh the ray
        // endpoints for the next physics step. An unanswered probe (no
        // physics in the world) reads as clear.
        for (entity, probe) in ctx.query_mut_with_entity::<AudioOcclusionProbe>() {
            let Some(binding) = self.emitters.get_mut(&entity) else {
                continue;
            };
            probe.from = self.listener_position;
            probe.to = binding.position;
            let factor = binding
                .occlusion
                .step(probe.blocked.unwrap_or(false), SimTiming::TICK_DT);
            if let Some(id) = binding.id {
                self.mixer.set_emitter_occlusion(id, factor);
            }
        }

        StepResult::Continue
    }
}

#[cfg(test)]
mod tests;

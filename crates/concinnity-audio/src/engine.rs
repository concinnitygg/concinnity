// concinnity-audio/src/engine.rs
//
// The kira-backed mixer behind the crate's facade. Owns one kira
// `AudioManager`, a listener, three mix buses (music / sfx / voice) under the
// main track, one spatial track per emitter, the one-shot voice pool, and the
// decode worker with its clip cache. Generic over the kira backend so tests
// drive the real mixer through kira's mock backend; production uses the
// default (cpal) backend.
//
// When no audio output device is available the engine is built in a disabled
// state and every method becomes a no-op. This keeps headless / CI runs
// (which may have no sound card) from failing.

use kira::backend::{Backend, DefaultBackend};
use kira::effect::EffectBuilder;
use kira::effect::filter::{FilterBuilder, FilterHandle};
use kira::listener::ListenerHandle;
use kira::sound::PlaybackState;
use kira::sound::static_sound::{StaticSoundData, StaticSoundHandle};
use kira::track::{SpatialTrackBuilder, SpatialTrackHandle, TrackBuilder, TrackHandle};
use kira::{AudioManager, AudioManagerSettings, Decibels, Tween};

use concinnity_core::components::{AudioBus, AudioTarget, Rolloff};

use crate::clips::{ClipState, ClipStore, PendingPlay};
use crate::decode::DecodeWorker;
use crate::voices::{Admission, VoiceSlots};
use crate::{occlusion, rolloff};

// Most one-shot sounds audible at once; see `VoiceSlots`.
const MAX_VOICES: usize = 32;

// Opaque handle to a spatial emitter inside an [`AudioEngine`]. Internal to
// the crate; the engine only ever touches `AudioSystem`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct EmitterId(pub(crate) usize);

// Authored acoustics for one emitter, resolved by the system from the
// AudioEmitter schema.
pub(crate) struct EmitterParams {
    pub(crate) min_distance: f32,
    pub max_distance: f32,
    pub(crate) rolloff: Rolloff,
    pub bus: AudioBus,
}

struct Emitter {
    track: SpatialTrackHandle,
    filter: FilterHandle,
    // Last occlusion factor applied, so an unchanged factor sends no
    // per-tick commands to the mixer.
    last_occlusion: f32,
}

struct Buses {
    music: TrackHandle,
    sfx: TrackHandle,
    voice: TrackHandle,
}

impl Buses {
    fn get_mut(&mut self, bus: AudioBus) -> &mut TrackHandle {
        match bus {
            AudioBus::Music => &mut self.music,
            AudioBus::Sfx => &mut self.sfx,
            AudioBus::Voice => &mut self.voice,
        }
    }
}

// The live kira state. Present only when an output device was acquired.
struct Active<B: Backend> {
    manager: AudioManager<B>,
    listener: ListenerHandle,
    buses: Buses,
    // One spatial track per emitter, indexed by `EmitterId` slot. A removed
    // emitter leaves `None` and its slot returns through `free_slots`, so
    // outstanding ids never shift onto another emitter.
    emitters: Vec<Option<Emitter>>,
    free_slots: Vec<usize>,
    // The looping music track, keyed by the caller's clip key so a replay of
    // the same clip keeps the track running instead of restarting it.
    music: Option<(u64, StaticSoundHandle)>,
    voices: VoiceSlots<StaticSoundHandle>,
    decoder: DecodeWorker,
    clips: ClipStore<StaticSoundData>,
}

// A 3D positional audio engine. Internal to the crate; driven by
// `AudioSystem`, which is the only type the engine constructs.
pub(crate) struct AudioEngine<B: Backend = DefaultBackend> {
    // `None` when no output device was available; the engine is then inert.
    active: Option<Active<B>>,
    // The last gain requested per target (master, music, sfx, voice; linear,
    // 1.0 = unity). Recorded even when the engine is disabled, so it reflects
    // the requested mix regardless of whether a device is present.
    last_volumes: [f32; 4],
}

impl<B: Backend> std::fmt::Debug for AudioEngine<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioEngine")
            .field("enabled", &self.active.is_some())
            .field(
                "emitters",
                &self.active.as_ref().map_or(0, |a| a.emitters.len()),
            )
            .field("volumes", &self.last_volumes)
            .finish()
    }
}

impl AudioEngine<DefaultBackend> {
    // Build the engine, acquiring the default output device. Returns a
    // disabled (no-op) engine when no device is available or kira fails to
    // start, so the caller never has to handle an error.
    pub(crate) fn new() -> Self {
        Self::start_or_disabled()
    }
}

impl<B: Backend> AudioEngine<B> {
    // An engine with no output device. Every method is a no-op. Also the
    // pre-init state of a constructed AudioSystem: acquiring the device waits
    // for `System::init`, so constructing the system has no side effects.
    pub(crate) fn disabled() -> Self {
        Self {
            active: None,
            last_volumes: [1.0; 4],
        }
    }

    fn start_or_disabled() -> Self
    where
        B::Settings: Default,
        B::Error: std::fmt::Debug,
    {
        match Self::try_start() {
            Ok(active) => Self {
                active: Some(active),
                last_volumes: [1.0; 4],
            },
            Err(e) => {
                tracing::warn!("AudioEngine disabled: {e}");
                Self::disabled()
            }
        }
    }

    fn try_start() -> Result<Active<B>, String>
    where
        B::Settings: Default,
        B::Error: std::fmt::Debug,
    {
        let mut manager = AudioManager::<B>::new(AudioManagerSettings::default())
            .map_err(|e| format!("audio manager init failed: {e:?}"))?;
        // The listener starts at the origin facing -Z (identity orientation);
        // AudioSystem moves it onto the camera on the first step.
        let listener = manager
            .add_listener(ORIGIN, IDENTITY_ORIENTATION)
            .map_err(|e| format!("listener init failed: {e}"))?;
        let mut bus = || {
            manager
                .add_sub_track(TrackBuilder::new())
                .map_err(|e| format!("mix bus init failed: {e}"))
        };
        let buses = Buses {
            music: bus()?,
            sfx: bus()?,
            voice: bus()?,
        };
        Ok(Active {
            manager,
            listener,
            buses,
            emitters: Vec::new(),
            free_slots: Vec::new(),
            music: None,
            voices: VoiceSlots::new(MAX_VOICES),
            decoder: DecodeWorker::spawn(),
            clips: ClipStore::new(),
        })
    }

    // Whether the engine acquired an output device.
    pub(crate) fn is_enabled(&self) -> bool {
        self.active.is_some()
    }

    // Hand an encoded clip (a whole ogg / wav / flac / ... file) to the
    // decode worker. A key already queued, decoded, or failed is ignored, so
    // callers may queue the same clip freely. No-op on a disabled engine.
    pub(crate) fn queue_clip(&mut self, key: u64, bytes: Vec<u8>) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        if active.clips.begin(key) && !active.decoder.send(key, bytes) {
            tracing::warn!("audio decode worker is gone; clip {key} will never play");
        }
    }

    // Drain finished decodes (starting any plays that waited on them) and
    // release the voice slots of finished one-shots. Call once per tick.
    pub(crate) fn tick(&mut self) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        active
            .voices
            .reap(|handle| handle.state() == PlaybackState::Stopped);
        for result in active.decoder.drain() {
            let flushed = active.clips.complete(result.key, result.decoded);
            for play in flushed {
                let Some(data) = active.clips.get(result.key).cloned() else {
                    continue;
                };
                active.play_now(play, &data);
            }
        }
    }

    // Add a spatial emitter at `position` with authored acoustics. Returns
    // `None` on a disabled engine or when kira's track limit is reached.
    pub(crate) fn add_emitter(
        &mut self,
        position: [f32; 3],
        params: &EmitterParams,
    ) -> Option<EmitterId> {
        let active = self.active.as_mut()?;
        let (min, max) = rolloff::clamp_distances(params.min_distance, params.max_distance);
        // The lowpass starts acoustically transparent; occlusion sweeps it.
        let (filter_effect, filter) = FilterBuilder::new()
            .cutoff(occlusion::cutoff_hz(0.0))
            .build();
        let builder = SpatialTrackBuilder::new()
            .distances((min, max))
            .attenuation_function(rolloff::attenuation(params.rolloff))
            .with_built_effect(filter_effect);
        let track = active
            .buses
            .get_mut(params.bus)
            .add_spatial_sub_track(&active.listener, vec3(position), builder)
            .map_err(|e| tracing::warn!("audio emitter add failed: {e}"))
            .ok()?;
        let emitter = Emitter {
            track,
            filter,
            last_occlusion: 0.0,
        };
        let id = match active.free_slots.pop() {
            Some(slot) => {
                active.emitters[slot] = Some(emitter);
                EmitterId(slot)
            }
            None => {
                active.emitters.push(Some(emitter));
                EmitterId(active.emitters.len() - 1)
            }
        };
        Some(id)
    }

    // Remove an emitter, stopping its playback (the spatial track unloads
    // when its handle drops). No-op on a disabled engine or an unknown or
    // already-removed id; the slot is recycled for a later emitter.
    pub(crate) fn remove_emitter(&mut self, id: EmitterId) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        if let Some(slot) = active.emitters.get_mut(id.0)
            && slot.take().is_some()
        {
            active.free_slots.push(id.0);
        }
    }

    // Start an emitter's clip. If the clip is still decoding, playback
    // begins on the tick the decode finishes. Returns false when the engine
    // is disabled or the clip failed to decode or was never queued.
    pub(crate) fn play_emitter_clip(
        &mut self,
        id: EmitterId,
        key: u64,
        looping: bool,
        gain: f32,
    ) -> bool {
        self.request_play(key, PendingPlay::Emitter { id, looping, gain })
    }

    // Play a looping music clip, replacing the current music. Requesting the
    // key that is already playing is a no-op, so navigation between screens
    // sharing a music cue never restarts the track. Returns false when the
    // engine is disabled or the clip failed to decode or was never queued.
    pub(crate) fn play_music(&mut self, key: u64, gain: f32, bus: AudioBus) -> bool {
        if let Some(active) = &self.active
            && let Some((current, _)) = &active.music
            && *current == key
        {
            return true;
        }
        self.request_play(key, PendingPlay::Music { key, gain, bus })
    }

    // Play a one-shot clip on a bus (UI feedback, story effects, dialogue).
    // The voice pool caps concurrent one-shots: a full pool silences its
    // oldest lowest-priority voice, or refuses the new sound if everything
    // playing outranks it. Returns false when refused, disabled, or the clip
    // failed to decode or was never queued.
    pub(crate) fn play_sound(&mut self, key: u64, gain: f32, bus: AudioBus, priority: i32) -> bool {
        self.request_play(
            key,
            PendingPlay::Sound {
                bus,
                gain,
                priority,
            },
        )
    }

    // Play a one-shot clip positioned in the world (impact sounds): a
    // transient spatial track on the sfx bus that unloads itself when the
    // sound finishes. Voice-pooled like `play_sound`.
    pub(crate) fn play_sound_at(
        &mut self,
        position: [f32; 3],
        key: u64,
        gain: f32,
        priority: i32,
    ) -> bool {
        self.request_play(
            key,
            PendingPlay::SoundAt {
                position,
                gain,
                priority,
            },
        )
    }

    // One-shot voices currently playing (or queued behind a decode).
    #[cfg(test)]
    pub(crate) fn playing_sounds(&self) -> usize {
        self.active.as_ref().map_or(0, |a| a.voices.len())
    }

    fn request_play(&mut self, key: u64, play: PendingPlay) -> bool {
        let Some(active) = self.active.as_mut() else {
            return false;
        };
        match active.clips.state(key) {
            Some(ClipState::Ready) => {
                let data = active.clips.get(key).cloned().expect("ready clip has data");
                active.play_now(play, &data)
            }
            Some(ClipState::InFlight) => active.clips.defer(key, play),
            Some(ClipState::Failed) => false,
            None => {
                tracing::warn!("audio clip {key} was never queued for decode");
                false
            }
        }
    }

    // Move an emitter. No-op on a disabled engine or an unknown id.
    pub(crate) fn set_emitter_position(&mut self, id: EmitterId, position: [f32; 3]) {
        if let Some(active) = self.active.as_mut()
            && let Some(emitter) = active.emitters.get_mut(id.0).and_then(Option::as_mut)
        {
            emitter.track.set_position(vec3(position), Tween::default());
        }
    }

    // Apply a smoothed occlusion factor in [0, 1] to an emitter: a volume
    // dip plus a lowpass sweep. No-op on a disabled engine, an unknown id,
    // or an unchanged factor.
    pub(crate) fn set_emitter_occlusion(&mut self, id: EmitterId, factor: f32) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        let Some(emitter) = active.emitters.get_mut(id.0).and_then(Option::as_mut) else {
            return;
        };
        if (factor - emitter.last_occlusion).abs() < 1.0e-3 {
            return;
        }
        emitter.last_occlusion = factor;
        emitter
            .track
            .set_volume(Decibels(occlusion::volume_db(factor)), Tween::default());
        emitter
            .filter
            .set_cutoff(occlusion::cutoff_hz(factor), Tween::default());
    }

    // Set a mix volume as a linear gain (1.0 = unchanged): the master output
    // or one of the buses under it. Applies to clips already playing as well
    // as future ones. No-op on a disabled engine. The short default tween
    // makes a live change click-free.
    pub(crate) fn set_volume(&mut self, target: AudioTarget, gain: f32) {
        self.last_volumes[target_index(target)] = gain;
        let Some(active) = self.active.as_mut() else {
            return;
        };
        let db = Decibels(gain_to_db(gain));
        match target {
            AudioTarget::Master => active.manager.main_track().set_volume(db, Tween::default()),
            AudioTarget::Music => active.buses.music.set_volume(db, Tween::default()),
            AudioTarget::Sfx => active.buses.sfx.set_volume(db, Tween::default()),
            AudioTarget::Voice => active.buses.voice.set_volume(db, Tween::default()),
        }
    }

    // The last gain requested for a target, whether or not a device applied it.
    #[cfg(test)]
    pub(crate) fn last_volume(&self, target: AudioTarget) -> f32 {
        self.last_volumes[target_index(target)]
    }

    // Update the listener pose from a camera position and yaw / pitch
    // (radians). No-op on a disabled engine.
    pub(crate) fn set_listener(&mut self, position: [f32; 3], yaw: f32, pitch: f32) {
        if let Some(active) = self.active.as_mut() {
            active
                .listener
                .set_position(vec3(position), Tween::default());
            active
                .listener
                .set_orientation(orientation_quat(yaw, pitch), Tween::default());
        }
    }
}

impl<B: Backend> Active<B> {
    // Start a decoded clip according to its play request. The decoded frames
    // are shared; per-play settings (gain, loop) ride a cheap settings copy.
    fn play_now(&mut self, play: PendingPlay, data: &StaticSoundData) -> bool {
        match play {
            PendingPlay::Emitter { id, looping, gain } => {
                let Some(emitter) = self.emitters.get_mut(id.0).and_then(Option::as_mut) else {
                    return false;
                };
                let mut data = data.clone().volume(Decibels(gain_to_db(gain)));
                if looping {
                    data = data.loop_region(..);
                }
                match emitter.track.play(data) {
                    Ok(_) => true,
                    Err(e) => {
                        tracing::warn!("audio clip play failed: {e}");
                        false
                    }
                }
            }
            PendingPlay::Music { key, gain, bus } => {
                if let Some((current, _)) = &self.music
                    && *current == key
                {
                    return true;
                }
                // The default tween fades the outgoing track briefly so the
                // swap is click-free.
                if let Some((_, mut handle)) = self.music.take() {
                    handle.stop(Tween::default());
                }
                let data = data
                    .clone()
                    .volume(Decibels(gain_to_db(gain)))
                    .loop_region(..);
                match self.buses.get_mut(bus).play(data) {
                    Ok(handle) => {
                        self.music = Some((key, handle));
                        true
                    }
                    Err(e) => {
                        tracing::warn!("music play failed: {e}");
                        false
                    }
                }
            }
            PendingPlay::Sound {
                bus,
                gain,
                priority,
            } => {
                match self.voices.make_room(priority) {
                    Admission::Available => {}
                    Admission::Steal(mut stolen) => stolen.stop(Tween::default()),
                    Admission::Refused => return false,
                }
                let data = data.clone().volume(Decibels(gain_to_db(gain)));
                match self.buses.get_mut(bus).play(data) {
                    Ok(handle) => {
                        self.voices.admit(priority, handle);
                        true
                    }
                    Err(e) => {
                        tracing::warn!("sound play failed: {e}");
                        false
                    }
                }
            }
            PendingPlay::SoundAt {
                position,
                gain,
                priority,
            } => {
                match self.voices.make_room(priority) {
                    Admission::Available => {}
                    Admission::Steal(mut stolen) => stolen.stop(Tween::default()),
                    Admission::Refused => return false,
                }
                // A transient spatial track: dropping its handle after the
                // play leaves it alive only until the sound finishes.
                let builder = SpatialTrackBuilder::new().persist_until_sounds_finish(true);
                let track = self.buses.get_mut(AudioBus::Sfx).add_spatial_sub_track(
                    &self.listener,
                    vec3(position),
                    builder,
                );
                let mut track = match track {
                    Ok(track) => track,
                    Err(e) => {
                        tracing::warn!("positioned sound track failed: {e}");
                        return false;
                    }
                };
                let data = data.clone().volume(Decibels(gain_to_db(gain)));
                match track.play(data) {
                    Ok(handle) => {
                        self.voices.admit(priority, handle);
                        true
                    }
                    Err(e) => {
                        tracing::warn!("positioned sound play failed: {e}");
                        false
                    }
                }
            }
        }
    }
}

fn target_index(target: AudioTarget) -> usize {
    match target {
        AudioTarget::Master => 0,
        AudioTarget::Music => 1,
        AudioTarget::Sfx => 2,
        AudioTarget::Voice => 3,
    }
}

// Listener spawn position.
const ORIGIN: mint::Vector3<f32> = mint::Vector3 {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};

// Unrotated listener orientation (faces -Z).
const IDENTITY_ORIENTATION: mint::Quaternion<f32> = mint::Quaternion { s: 1.0, v: ORIGIN };

fn vec3(p: [f32; 3]) -> mint::Vector3<f32> {
    mint::Vector3 {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

// Convert a linear gain multiplier to decibels. A gain of 1.0 maps to 0 dB;
// gains at or below ~-80 dB are clamped so silence does not yield `-inf`.
fn gain_to_db(gain: f32) -> f32 {
    20.0 * gain.max(1.0e-4).log10()
}

// Build the listener orientation quaternion from camera yaw / pitch (radians).
//
// An unrotated kira listener faces -Z with +X right and +Y up, which matches
// the engine's camera basis (yaw 0, pitch 0 looks toward -Z). The result is
// the yaw rotation about +Y composed with the pitch rotation about +X.
fn orientation_quat(yaw: f32, pitch: f32) -> mint::Quaternion<f32> {
    let hy = yaw * 0.5;
    let hp = -pitch * 0.5;
    let (sy, cy) = hy.sin_cos();
    let (sp, cp) = hp.sin_cos();
    mint::Quaternion {
        s: cy * cp,
        v: mint::Vector3 {
            x: cy * sp,
            y: sy * cp,
            z: -sy * sp,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_wav::pcm_wav_mono;
    use kira::backend::mock::MockBackend;

    // The real mixer on kira's mock backend: no device, fully headless.
    type MockEngine = AudioEngine<MockBackend>;

    fn mock_engine() -> MockEngine {
        let engine = MockEngine::start_or_disabled();
        assert!(engine.is_enabled(), "mock backend always starts");
        engine
    }

    impl MockEngine {
        // Let the mock renderer consume queued mixer commands and process a
        // chunk of audio, so handle-visible state (playback states, counts)
        // advances.
        fn pump(&mut self) {
            let active = self.active.as_mut().unwrap();
            active.manager.backend_mut().on_start_processing();
            active.manager.backend_mut().process();
        }

        fn clip_ready(&self, key: u64) -> bool {
            self.active
                .as_ref()
                .is_some_and(|a| a.clips.state(key) == Some(ClipState::Ready))
        }

        fn bus_sounds(&self, bus: AudioBus) -> usize {
            let active = self.active.as_ref().unwrap();
            match bus {
                AudioBus::Music => active.buses.music.num_sounds(),
                AudioBus::Sfx => active.buses.sfx.num_sounds(),
                AudioBus::Voice => active.buses.voice.num_sounds(),
            }
        }

        fn music_key(&self) -> Option<u64> {
            self.active
                .as_ref()
                .and_then(|a| a.music.as_ref())
                .map(|m| m.0)
        }

        // Tick until `key` decodes; the worker is a real thread, so poll.
        fn wait_ready(&mut self, key: u64) {
            for _ in 0..500 {
                self.tick();
                if self.clip_ready(key) {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            panic!("clip {key} never decoded");
        }
    }

    #[test]
    fn disabled_engine_is_inert() {
        let mut engine: AudioEngine<MockBackend> = AudioEngine::disabled();
        assert!(!engine.is_enabled());
        let params = EmitterParams {
            min_distance: 1.0,
            max_distance: 50.0,
            rolloff: Rolloff::Logarithmic,
            bus: AudioBus::Sfx,
        };
        assert!(engine.add_emitter([0.0; 3], &params).is_none());
        // None of these may panic on a disabled engine.
        engine.queue_clip(1, pcm_wav_mono(16));
        engine.tick();
        engine.set_emitter_position(EmitterId(0), [1.0, 2.0, 3.0]);
        engine.set_emitter_occlusion(EmitterId(0), 0.5);
        engine.set_listener([4.0, 5.0, 6.0], 1.0, 0.5);
        assert!(!engine.play_emitter_clip(EmitterId(0), 1, true, 1.0));
        assert!(!engine.play_music(1, 1.0, AudioBus::Music));
        assert!(!engine.play_sound(1, 1.0, AudioBus::Sfx, 0));
        assert_eq!(engine.playing_sounds(), 0);
        // A disabled engine still records requested volumes (it just cannot
        // apply them to a device).
        assert_eq!(engine.last_volume(AudioTarget::Master), 1.0);
        engine.set_volume(AudioTarget::Voice, 0.5);
        assert_eq!(engine.last_volume(AudioTarget::Voice), 0.5);
    }

    #[test]
    fn one_shots_route_to_their_bus() {
        let mut engine = mock_engine();
        engine.queue_clip(1, pcm_wav_mono(64));
        engine.wait_ready(1);
        assert!(engine.play_sound(1, 1.0, AudioBus::Voice, 0));
        assert!(engine.play_sound(1, 1.0, AudioBus::Sfx, 0));
        assert_eq!(engine.bus_sounds(AudioBus::Voice), 1);
        assert_eq!(engine.bus_sounds(AudioBus::Sfx), 1);
        assert_eq!(engine.bus_sounds(AudioBus::Music), 0);
        assert_eq!(engine.playing_sounds(), 2);
    }

    #[test]
    fn play_before_decode_completes_starts_on_a_later_tick() {
        let mut engine = mock_engine();
        engine.queue_clip(5, pcm_wav_mono(64));
        // Requested while the decode is still in flight: deferred, not lost.
        assert!(engine.play_sound(5, 1.0, AudioBus::Sfx, 0));
        assert_eq!(engine.bus_sounds(AudioBus::Sfx), 0, "not started yet");
        engine.wait_ready(5);
        assert_eq!(engine.bus_sounds(AudioBus::Sfx), 1, "flushed on decode");
        assert_eq!(engine.playing_sounds(), 1);
    }

    #[test]
    fn voice_cap_steals_low_and_refuses_outranked() {
        let mut engine = mock_engine();
        engine.queue_clip(1, pcm_wav_mono(4096));
        engine.wait_ready(1);
        for _ in 0..MAX_VOICES {
            assert!(engine.play_sound(1, 1.0, AudioBus::Sfx, 5));
        }
        assert_eq!(engine.playing_sounds(), MAX_VOICES);
        // An equal-priority sound steals the oldest; the pool stays capped.
        assert!(engine.play_sound(1, 1.0, AudioBus::Sfx, 5));
        assert_eq!(engine.playing_sounds(), MAX_VOICES);
        // An outranked sound is refused outright.
        assert!(!engine.play_sound(1, 1.0, AudioBus::Sfx, 1));
        assert_eq!(engine.playing_sounds(), MAX_VOICES);
    }

    #[test]
    fn finished_voices_are_reaped() {
        let mut engine = mock_engine();
        engine.queue_clip(1, pcm_wav_mono(64));
        engine.wait_ready(1);
        assert!(engine.play_sound(1, 1.0, AudioBus::Sfx, 0));
        assert_eq!(engine.playing_sounds(), 1);
        // The mock renderer runs at 1 Hz, so each pump processes 2 seconds:
        // a 64-frame clip finishes almost immediately.
        for _ in 0..500 {
            engine.pump();
            engine.tick();
            if engine.playing_sounds() == 0 {
                return;
            }
        }
        panic!("finished voice never reaped");
    }

    #[test]
    fn music_replays_are_seamless_and_swaps_replace() {
        let mut engine = mock_engine();
        engine.queue_clip(1, pcm_wav_mono(64));
        engine.queue_clip(2, pcm_wav_mono(64));
        engine.wait_ready(1);
        engine.wait_ready(2);
        assert!(engine.play_music(1, 1.0, AudioBus::Music));
        assert_eq!(engine.music_key(), Some(1));
        // Same key again: the running track is kept, not restarted.
        assert!(engine.play_music(1, 0.5, AudioBus::Music));
        assert_eq!(engine.bus_sounds(AudioBus::Music), 1);
        // A different key replaces the track.
        assert!(engine.play_music(2, 1.0, AudioBus::Music));
        assert_eq!(engine.music_key(), Some(2));
    }

    #[test]
    fn emitters_join_their_bus_with_authored_acoustics() {
        let mut engine = mock_engine();
        let params = EmitterParams {
            min_distance: 2.0,
            max_distance: 80.0,
            rolloff: Rolloff::Linear,
            bus: AudioBus::Sfx,
        };
        let id = engine
            .add_emitter([1.0, 0.0, 0.0], &params)
            .expect("emitter");
        let active = engine.active.as_ref().unwrap();
        assert_eq!(
            active.buses.sfx.num_sub_tracks(),
            1,
            "spatial track under sfx"
        );
        assert_eq!(active.buses.music.num_sub_tracks(), 0);

        engine.queue_clip(9, pcm_wav_mono(64));
        engine.wait_ready(9);
        assert!(engine.play_emitter_clip(id, 9, true, 0.8));
        engine.set_emitter_position(id, [2.0, 0.0, 0.0]);
        // Occlusion factors apply without panicking and dedupe repeats.
        engine.set_emitter_occlusion(id, 0.5);
        engine.set_emitter_occlusion(id, 0.5);
        engine.pump();
    }

    #[test]
    fn removed_emitters_free_their_slot_without_shifting_ids() {
        let mut engine = mock_engine();
        let params = EmitterParams {
            min_distance: 1.0,
            max_distance: 50.0,
            rolloff: Rolloff::Logarithmic,
            bus: AudioBus::Sfx,
        };
        let first = engine.add_emitter([1.0, 0.0, 0.0], &params).unwrap();
        let second = engine.add_emitter([2.0, 0.0, 0.0], &params).unwrap();
        assert_ne!(first, second);

        engine.remove_emitter(first);
        // Removing again is a no-op, and the survivor still answers.
        engine.remove_emitter(first);
        engine.set_emitter_position(second, [3.0, 0.0, 0.0]);
        engine.set_emitter_occlusion(second, 0.4);

        // A play aimed at the removed emitter is refused, not misrouted.
        engine.queue_clip(1, pcm_wav_mono(64));
        engine.wait_ready(1);
        assert!(!engine.play_emitter_clip(first, 1, true, 1.0));
        assert!(engine.play_emitter_clip(second, 1, true, 1.0));

        // The freed slot is recycled: the next emitter reuses `first`'s id.
        let third = engine.add_emitter([4.0, 0.0, 0.0], &params).unwrap();
        assert_eq!(third, first, "slot recycled");
        assert!(engine.play_emitter_clip(third, 1, true, 1.0));
    }

    #[test]
    fn positioned_one_shots_ride_transient_sfx_tracks_and_count_as_voices() {
        let mut engine = mock_engine();
        engine.queue_clip(1, pcm_wav_mono(4096));
        engine.wait_ready(1);
        assert!(engine.play_sound_at([2.0, 0.0, 1.0], 1, 0.8, 0));
        assert_eq!(engine.playing_sounds(), 1, "voice-pooled");
        let active = engine.active.as_ref().unwrap();
        assert_eq!(
            active.buses.sfx.num_sub_tracks(),
            1,
            "transient spatial track under sfx"
        );
        assert_eq!(active.buses.music.num_sub_tracks(), 0);
    }

    #[test]
    fn undecodable_clips_fail_without_wedging_the_engine() {
        let mut engine = mock_engine();
        engine.queue_clip(7, b"not an audio file".to_vec());
        // Deferred behind the decode, then dropped when it fails.
        assert!(engine.play_sound(7, 1.0, AudioBus::Sfx, 0));
        for _ in 0..500 {
            engine.tick();
            if let Some(active) = &engine.active
                && active.clips.state(7) == Some(ClipState::Failed)
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(
            !engine.play_sound(7, 1.0, AudioBus::Sfx, 0),
            "failed clip refuses plays"
        );
        assert_eq!(engine.playing_sounds(), 0);
        // The engine still works for other clips.
        engine.queue_clip(8, pcm_wav_mono(64));
        engine.wait_ready(8);
        assert!(engine.play_sound(8, 1.0, AudioBus::Sfx, 0));
    }

    #[test]
    fn identity_orientation_at_zero_yaw_pitch() {
        let q = orientation_quat(0.0, 0.0);
        assert!((q.s - 1.0).abs() < 1.0e-6);
        assert!(q.v.x.abs() < 1.0e-6 && q.v.y.abs() < 1.0e-6 && q.v.z.abs() < 1.0e-6);
    }

    #[test]
    fn orientation_quaternion_stays_unit_length() {
        for &(yaw, pitch) in &[(0.5, 0.3), (-1.2, 0.8), (3.0, -0.6), (-2.7, -1.1)] {
            let q = orientation_quat(yaw, pitch);
            let len = (q.s * q.s + q.v.x * q.v.x + q.v.y * q.v.y + q.v.z * q.v.z).sqrt();
            assert!(
                (len - 1.0).abs() < 1.0e-5,
                "not unit ({len}) at {yaw}/{pitch}"
            );
        }
    }

    #[test]
    fn gain_to_db_reference_points() {
        assert!(gain_to_db(1.0).abs() < 1.0e-4); // unity -> 0 dB
        assert!((gain_to_db(0.5) - (-6.0206)).abs() < 0.01); // half -> ~-6 dB
        assert!(gain_to_db(0.0) < -60.0); // silence clamped low, not -inf
        assert!(gain_to_db(0.0).is_finite());
    }
}

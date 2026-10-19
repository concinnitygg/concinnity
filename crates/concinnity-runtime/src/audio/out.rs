//! The kira-backed [`AudioOut`]: one kira `AudioManager`, a listener, three mix
//! buses (music / sfx / voice) under the main track, one spatial track per
//! emitter, the playing sounds, and the decode worker with the clips it
//! decoded. Generic over the kira backend so tests drive the real mixer through
//! kira's mock backend; production uses the default (cpal) backend.

use concinnity_core::audio::EmitterId;
use concinnity_core::audio::out::{Acoustics, AudioOut, PlayOn, VoiceId};
use concinnity_core::components::{AudioBus, AudioTarget};
use concinnity_core::math::Quat;
use hashbrown::HashMap;
use kira::backend::{Backend, DefaultBackend};
use kira::effect::EffectBuilder;
use kira::effect::filter::{FilterBuilder, FilterHandle};
use kira::listener::ListenerHandle;
use kira::sound::PlaybackState;
use kira::sound::static_sound::{StaticSoundData, StaticSoundHandle};
use kira::track::{SpatialTrackBuilder, SpatialTrackHandle, TrackBuilder, TrackHandle};
use kira::{AudioManager, AudioManagerSettings, Decibels, Tween};

use super::decode::DecodeWorker;
use super::rolloff;

/// The default output device's mixer, or `None` (with a warning) when there is
/// no device or kira fails to start, which leaves the world silent.
pub(crate) fn open() -> Option<Box<dyn AudioOut>> {
    match KiraOut::<DefaultBackend>::start() {
        Ok(out) => Some(Box::new(out)),
        Err(e) => {
            tracing::warn!("audio: output disabled: {e}");
            None
        }
    }
}

struct Emitter {
    track: SpatialTrackHandle,
    filter: FilterHandle,
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

pub(crate) struct KiraOut<B: Backend = DefaultBackend> {
    manager: AudioManager<B>,
    listener: ListenerHandle,
    buses: Buses,
    // One spatial track per emitter, indexed by `EmitterId`.
    emitters: Vec<Option<Emitter>>,
    // Handles of the bus and positioned voices, until they finish or stop.
    voices: HashMap<VoiceId, StaticSoundHandle>,
    next_voice: u64,
    // `None` when the OS refused the decode thread: clips then never play.
    decoder: Option<DecodeWorker>,
    clips: HashMap<u64, StaticSoundData>,
}

impl<B: Backend> std::fmt::Debug for KiraOut<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KiraOut")
            .field("emitters", &self.emitters.len())
            .field("voices", &self.voices.len())
            .finish_non_exhaustive()
    }
}

impl<B: Backend> KiraOut<B> {
    fn start() -> Result<Self, String>
    where
        B::Settings: Default,
        B::Error: std::fmt::Debug,
    {
        let mut manager = AudioManager::<B>::new(AudioManagerSettings::default())
            .map_err(|e| format!("audio manager init failed: {e:?}"))?;
        let listener = manager
            .add_listener(vec3([0.0; 3]), quat([0.0, 0.0, 0.0, 1.0]))
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
        Ok(Self {
            manager,
            listener,
            buses,
            emitters: Vec::new(),
            voices: HashMap::new(),
            next_voice: 0,
            decoder: DecodeWorker::spawn()
                .map_err(|e| {
                    tracing::warn!("audio decode worker spawn failed, clips stay silent: {e}");
                })
                .ok(),
            clips: HashMap::new(),
        })
    }

    fn emitter_mut(&mut self, id: EmitterId) -> Option<&mut Emitter> {
        self.emitters.get_mut(id.0).and_then(Option::as_mut)
    }

    // Play `data` on `on`, or say why it could not start.
    fn start_on(&mut self, on: PlayOn, data: StaticSoundData) -> Result<StaticSoundHandle, String> {
        match on {
            PlayOn::Emitter(id) => self
                .emitter_mut(id)
                .ok_or_else(|| format!("no emitter {}", id.0))?
                .track
                .play(data)
                .map_err(|e| e.to_string()),
            PlayOn::Bus(bus) => self
                .buses
                .get_mut(bus)
                .play(data)
                .map_err(|e| e.to_string()),
            PlayOn::At(position) => {
                // A transient spatial track: dropping its handle after the
                // play leaves it alive only until the sound finishes.
                let builder = SpatialTrackBuilder::new().persist_until_sounds_finish(true);
                let mut track = self
                    .buses
                    .sfx
                    .add_spatial_sub_track(&self.listener, vec3(position), builder)
                    .map_err(|e| e.to_string())?;
                track.play(data).map_err(|e| e.to_string())
            }
        }
    }
}

impl<B: Backend + Send> AudioOut for KiraOut<B> {
    fn decode(&mut self, key: u64, bytes: Vec<u8>) -> bool {
        self.decoder
            .as_mut()
            .is_some_and(|decoder| decoder.send(key, bytes))
    }

    fn decoded(&mut self) -> Option<(u64, Result<(), String>)> {
        let result = self.decoder.as_ref()?.try_recv()?;
        Some((
            result.key,
            result.decoded.map(|data| {
                self.clips.insert(result.key, data);
            }),
        ))
    }

    fn add_emitter(
        &mut self,
        id: EmitterId,
        position: [f32; 3],
        acoustics: &Acoustics,
        cutoff_hz: f64,
    ) -> bool {
        let (filter_effect, filter) = FilterBuilder::new().cutoff(cutoff_hz).build();
        let builder = SpatialTrackBuilder::new()
            .distances(acoustics.distances)
            .attenuation_function(rolloff::attenuation(acoustics.rolloff))
            .with_built_effect(filter_effect);
        let track = match self.buses.get_mut(acoustics.bus).add_spatial_sub_track(
            &self.listener,
            vec3(position),
            builder,
        ) {
            Ok(track) => track,
            Err(e) => {
                tracing::warn!("audio emitter add failed: {e}");
                return false;
            }
        };
        if self.emitters.len() <= id.0 {
            self.emitters.resize_with(id.0 + 1, || None);
        }
        self.emitters[id.0] = Some(Emitter { track, filter });
        true
    }

    fn remove_emitter(&mut self, id: EmitterId) {
        if let Some(slot) = self.emitters.get_mut(id.0) {
            *slot = None;
        }
    }

    fn move_emitter(&mut self, id: EmitterId, position: [f32; 3]) {
        if let Some(emitter) = self.emitter_mut(id) {
            emitter.track.set_position(vec3(position), Tween::default());
        }
    }

    fn muffle_emitter(&mut self, id: EmitterId, volume_db: f32, cutoff_hz: f64) {
        if let Some(emitter) = self.emitter_mut(id) {
            emitter
                .track
                .set_volume(Decibels(volume_db), Tween::default());
            emitter.filter.set_cutoff(cutoff_hz, Tween::default());
        }
    }

    fn play(&mut self, key: u64, on: PlayOn, gain_db: f32, looping: bool) -> Option<VoiceId> {
        // The decoded frames are shared; gain and loop ride a cheap copy.
        let mut data = self.clips.get(&key)?.clone().volume(Decibels(gain_db));
        if looping {
            data = data.loop_region(..);
        }
        let handle = self
            .start_on(on, data)
            .map_err(|e| tracing::warn!("audio clip {key} play failed: {e}"))
            .ok()?;
        let voice = VoiceId(self.next_voice);
        self.next_voice += 1;
        // An emitter's clip plays until its track closes and is never stopped
        // or queried, so only bus and positioned voices keep a handle.
        if !matches!(on, PlayOn::Emitter(_)) {
            self.voices.insert(voice, handle);
        }
        Some(voice)
    }

    fn stop(&mut self, voice: VoiceId) {
        if let Some(mut handle) = self.voices.remove(&voice) {
            handle.stop(Tween::default());
        }
    }

    fn finished(&mut self, voice: VoiceId) -> bool {
        let done = self
            .voices
            .get(&voice)
            .is_none_or(|handle| handle.state() == PlaybackState::Stopped);
        if done {
            self.voices.remove(&voice);
        }
        done
    }

    fn set_volume(&mut self, target: AudioTarget, db: f32) {
        let db = Decibels(db);
        match target {
            AudioTarget::Master => self.manager.main_track().set_volume(db, Tween::default()),
            AudioTarget::Music => self.buses.music.set_volume(db, Tween::default()),
            AudioTarget::Sfx => self.buses.sfx.set_volume(db, Tween::default()),
            AudioTarget::Voice => self.buses.voice.set_volume(db, Tween::default()),
        }
    }

    fn set_listener(&mut self, position: [f32; 3], orientation: Quat) {
        self.listener.set_position(vec3(position), Tween::default());
        self.listener
            .set_orientation(quat(orientation), Tween::default());
    }
}

fn vec3(p: [f32; 3]) -> mint::Vector3<f32> {
    mint::Vector3 {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

fn quat([x, y, z, w]: Quat) -> mint::Quaternion<f32> {
    mint::Quaternion {
        s: w,
        v: mint::Vector3 { x, y, z },
    }
}

#[cfg(test)]
mod tests;

//! What plays, where and how loud, over the host's [`AudioOut`]: clips wait on
//! their decode, one-shots share a capped voice pool, one music track plays at
//! a time, emitter ids are recycled slots, and occlusion and gains reach the
//! output in decibels. With no output open every call is a no-op that still
//! records the requested mix.

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::clips::{ClipState, ClipStore, PendingPlay};
use super::out::{Acoustics, AudioOut, PlayOn, VoiceId};
use super::voices::{Admission, VoiceSlots};
use super::{EmitterId, distance, occlusion};
use crate::components::{AudioBus, AudioTarget};
use concinnity_core::math::{self, Quat};

// Most one-shot sounds audible at once; see `VoiceSlots`.
const MAX_VOICES: usize = 32;

// An occlusion change smaller than this sends nothing to the output.
const OCCLUSION_EPSILON: f32 = 1.0e-3;

/// The audio state over one output.
pub struct Mixer<O: AudioOut + ?Sized = dyn AudioOut> {
    out: Option<Box<O>>,
    clips: ClipStore,
    voices: VoiceSlots<VoiceId>,
    // The last occlusion factor applied to each emitter, indexed by
    // `EmitterId`. A removed emitter leaves `None` and its slot returns
    // through `free_slots`, so outstanding ids never shift onto another one.
    emitters: Vec<Option<f32>>,
    free_slots: Vec<usize>,
    // The looping music, keyed by its clip so a replay of the same clip keeps
    // the track running instead of restarting it.
    music: Option<(u64, VoiceId)>,
    // The last gain requested per target (master, music, sfx, voice; linear,
    // 1.0 = unity), recorded whether or not an output applied it.
    last_volumes: [f32; 4],
}

impl<O: AudioOut + ?Sized> core::fmt::Debug for Mixer<O> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Mixer")
            .field("enabled", &self.out.is_some())
            .field("emitters", &self.emitters.len())
            .field("voices", &self.voices.len())
            .field("volumes", &self.last_volumes)
            .finish_non_exhaustive()
    }
}

impl<O: AudioOut + ?Sized> Mixer<O> {
    /// A mixer over `out`, or an inert one for `None`.
    pub fn new(out: Option<Box<O>>) -> Self {
        Self {
            out,
            clips: ClipStore::default(),
            voices: VoiceSlots::new(MAX_VOICES),
            emitters: Vec::new(),
            free_slots: Vec::new(),
            music: None,
            last_volumes: [1.0; 4],
        }
    }

    /// Whether an output is open.
    pub fn is_enabled(&self) -> bool {
        self.out.is_some()
    }

    /// Hand an encoded clip to the decoder. A key already queued, decoded or
    /// failed is ignored, so a caller may queue the same clip freely.
    pub fn queue_clip(&mut self, key: u64, bytes: Vec<u8>) {
        let Some(out) = self.out.as_mut() else {
            return;
        };
        if self.clips.begin(key) && !out.decode(key, bytes) {
            tracing::warn!("audio: the decoder is gone; clip {key} will never play");
        }
    }

    /// Release the voice slots of finished one-shots and start the plays that
    /// waited on a decode that finished. Call once per tick.
    pub fn tick(&mut self) {
        let Some(out) = self.out.as_mut() else {
            return;
        };
        self.voices.reap(|voice| out.finished(*voice));
        while let Some((key, result)) = self.out.as_mut().and_then(|out| out.decoded()) {
            for play in self.clips.complete(key, result) {
                self.play_now(key, play);
            }
        }
    }

    /// Add a spatial emitter at `position`, its distances sanitized and its
    /// lowpass open. `None` with no output, or when the output has no room for
    /// another track.
    pub fn add_emitter(&mut self, position: [f32; 3], acoustics: Acoustics) -> Option<EmitterId> {
        let out = self.out.as_mut()?;
        let id = if let Some(slot) = self.free_slots.pop() {
            EmitterId(slot)
        } else {
            self.emitters.push(None);
            EmitterId(self.emitters.len() - 1)
        };
        let (min, max) = acoustics.distances;
        let acoustics = Acoustics {
            distances: distance::clamp_distances(min, max),
            ..acoustics
        };
        if !out.add_emitter(id, position, &acoustics, occlusion::cutoff_hz(0.0)) {
            tracing::warn!("audio: emitter add failed, no room for another track");
            self.free_slots.push(id.0);
            return None;
        }
        self.emitters[id.0] = Some(0.0);
        Some(id)
    }

    /// Remove an emitter, stopping its playback; its slot is recycled. An
    /// unknown or already-removed id is ignored.
    pub fn remove_emitter(&mut self, id: EmitterId) {
        if let Some(slot) = self.emitters.get_mut(id.0)
            && slot.take().is_some()
        {
            self.free_slots.push(id.0);
            if let Some(out) = self.out.as_mut() {
                out.remove_emitter(id);
            }
        }
    }

    /// Start an emitter's clip; a clip still decoding starts on the tick its
    /// decode lands. `false` with no output, or when the clip failed or was
    /// never queued.
    pub fn play_emitter_clip(&mut self, id: EmitterId, key: u64, looping: bool, gain: f32) -> bool {
        self.request_play(key, PendingPlay::Emitter { id, looping, gain })
    }

    /// Play a looping music clip, replacing the current music. Asking for the
    /// clip already playing keeps it, so screens sharing a music cue never
    /// restart the track.
    pub fn play_music(&mut self, key: u64, gain: f32, bus: AudioBus) -> bool {
        if self.out.is_some() && self.music_is(key) {
            return true;
        }
        self.request_play(key, PendingPlay::Music { key, gain, bus })
    }

    /// Play a one-shot clip on a bus. The voice pool caps concurrent one-shots:
    /// a full pool silences its oldest lowest-priority voice, or refuses the
    /// new sound if everything playing outranks it.
    pub fn play_sound(&mut self, key: u64, gain: f32, bus: AudioBus, priority: i32) -> bool {
        self.request_play(
            key,
            PendingPlay::Sound {
                bus,
                gain,
                priority,
            },
        )
    }

    /// Play a one-shot clip at a world position, voice-pooled like
    /// [`play_sound`](Self::play_sound).
    pub fn play_sound_at(
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

    /// One-shot voices playing, or waiting on a decode.
    #[cfg(test)]
    fn playing_sounds(&self) -> usize {
        self.voices.len()
    }

    /// Move an emitter.
    pub fn set_emitter_position(&mut self, id: EmitterId, position: [f32; 3]) {
        if self.is_live(id)
            && let Some(out) = self.out.as_mut()
        {
            out.move_emitter(id, position);
        }
    }

    /// Apply a smoothed occlusion factor in `[0, 1]` to an emitter: a volume
    /// dip and a lowpass sweep. An unchanged factor sends nothing.
    pub fn set_emitter_occlusion(&mut self, id: EmitterId, factor: f32) {
        let (Some(out), Some(Some(last))) = (self.out.as_mut(), self.emitters.get_mut(id.0)) else {
            return;
        };
        if (factor - *last).abs() < OCCLUSION_EPSILON {
            return;
        }
        *last = factor;
        out.muffle_emitter(
            id,
            occlusion::volume_db(factor),
            occlusion::cutoff_hz(factor),
        );
    }

    /// Set a mix volume as a linear gain (1.0 = unchanged): the master output or
    /// one bus under it, for what plays now and later.
    pub fn set_volume(&mut self, target: AudioTarget, gain: f32) {
        self.last_volumes[target_index(target)] = gain;
        if let Some(out) = self.out.as_mut() {
            out.set_volume(target, gain_to_db(gain));
        }
    }

    /// The last gain requested for `target`, whether or not an output applied it.
    pub fn last_volume(&self, target: AudioTarget) -> f32 {
        self.last_volumes[target_index(target)]
    }

    /// Place the listener at a camera's position and yaw / pitch (radians).
    pub fn set_listener(&mut self, position: [f32; 3], yaw: f32, pitch: f32) {
        if let Some(out) = self.out.as_mut() {
            out.set_listener(position, orientation(yaw, pitch));
        }
    }

    fn is_live(&self, id: EmitterId) -> bool {
        matches!(self.emitters.get(id.0), Some(Some(_)))
    }

    fn music_is(&self, key: u64) -> bool {
        self.music.is_some_and(|(current, _)| current == key)
    }

    fn request_play(&mut self, key: u64, play: PendingPlay) -> bool {
        if self.out.is_none() {
            return false;
        }
        match self.clips.state(key) {
            Some(ClipState::Ready) => self.play_now(key, play),
            Some(ClipState::InFlight(_)) => self.clips.defer(key, play),
            Some(ClipState::Failed) => false,
            None => {
                tracing::warn!("audio: clip {key} was never queued for decode");
                false
            }
        }
    }

    // Make room in the voice pool for a one-shot of `priority`, silencing the
    // voice it steals. False when the pool refuses it.
    fn make_room(&mut self, priority: i32) -> bool {
        match self.voices.make_room(priority) {
            Admission::Available => true,
            Admission::Steal(stolen) => {
                if let Some(out) = self.out.as_mut() {
                    out.stop(stolen);
                }
                true
            }
            Admission::Refused => false,
        }
    }

    // Start decoded clip `key` the way `play` asks.
    fn play_now(&mut self, key: u64, play: PendingPlay) -> bool {
        match play {
            PendingPlay::Emitter { id, looping, gain } => {
                if !self.is_live(id) {
                    return false;
                }
                self.start(key, PlayOn::Emitter(id), gain, looping)
                    .is_some()
            }
            PendingPlay::Music { key, gain, bus } => {
                if self.music_is(key) {
                    return true;
                }
                if let Some((_, current)) = self.music.take()
                    && let Some(out) = self.out.as_mut()
                {
                    out.stop(current);
                }
                match self.start(key, PlayOn::Bus(bus), gain, true) {
                    Some(voice) => {
                        self.music = Some((key, voice));
                        true
                    }
                    None => false,
                }
            }
            PendingPlay::Sound {
                bus,
                gain,
                priority,
            } => self.start_one_shot(key, PlayOn::Bus(bus), gain, priority),
            PendingPlay::SoundAt {
                position,
                gain,
                priority,
            } => self.start_one_shot(key, PlayOn::At(position), gain, priority),
        }
    }

    fn start_one_shot(&mut self, key: u64, on: PlayOn, gain: f32, priority: i32) -> bool {
        if !self.make_room(priority) {
            return false;
        }
        match self.start(key, on, gain, false) {
            Some(voice) => {
                self.voices.admit(priority, voice);
                true
            }
            None => false,
        }
    }

    fn start(&mut self, key: u64, on: PlayOn, gain: f32, looping: bool) -> Option<VoiceId> {
        let voice = self.out.as_mut()?.play(key, on, gain_to_db(gain), looping);
        if voice.is_none() {
            tracing::warn!("audio: clip {key} failed to start");
        }
        voice
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

/// A linear gain in decibels: 1.0 is 0 dB, and gains at or below about -80 dB
/// clamp so silence does not read as `-inf`.
fn gain_to_db(gain: f32) -> f32 {
    20.0 * math::log10(gain.max(1.0e-4))
}

/// The listener orientation for a camera's yaw / pitch (radians). An unrotated
/// listener faces -Z with +X right and +Y up, the camera basis at yaw 0, pitch
/// 0; the result is the yaw rotation about +Y composed with the pitch rotation
/// about +X.
pub fn orientation(yaw: f32, pitch: f32) -> Quat {
    let (sy, cy) = math::sin_cos(yaw * 0.5);
    let (sp, cp) = math::sin_cos(-pitch * 0.5);
    [cy * sp, sy * cp, -sy * sp, cy * cp]
}

#[cfg(test)]
mod tests;

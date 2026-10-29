//! The seam between core's audio policy and the host's mixer: a host lends an
//! [`AudioOut`] that owns the device, the mix buses, the spatial emitter
//! tracks, the playing voices and the clip decoder. Everything it is asked to
//! do arrives already decided, in decibels and hertz.

use alloc::string::String;
use alloc::vec::Vec;

use super::EmitterId;
use crate::components::{AudioBus, AudioTarget, Rolloff};
use concinnity_core::math::Quat;

/// One playing sound inside the host's mixer, named by the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoiceId(pub u64);

/// Where a clip plays.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlayOn {
    /// On a spatial emitter's track.
    Emitter(EmitterId),
    /// Straight onto a mix bus.
    Bus(AudioBus),
    /// On a transient spatial track on the sfx bus, at a world position. The
    /// track unloads itself when the sound finishes.
    At([f32; 3]),
}

/// How a spatial emitter sounds with distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Acoustics {
    /// The bus its track joins.
    pub bus: AudioBus,
    /// Full volume inside the first distance, silent past the second.
    pub distances: (f32, f32),
    /// The volume curve between them.
    pub rolloff: Rolloff,
}

/// The host's mixer. Opened with its listener at the origin facing -Z and its
/// three buses (music, sfx, voice) under the master track.
pub trait AudioOut: core::fmt::Debug + Send {
    /// Queue a whole encoded clip (ogg, wav, flac, ...) for decode under
    /// `key`. `false` when the decoder is gone.
    fn decode(&mut self, key: u64, bytes: Vec<u8>) -> bool;

    /// One decode finished since the last call: its key, and why it failed if
    /// it did. Never blocks.
    fn decoded(&mut self) -> Option<(u64, Result<(), String>)>;

    /// Open emitter `id`'s spatial track at `position` behind a lowpass at
    /// `cutoff_hz`. `false` when the mixer has no room for another track.
    fn add_emitter(
        &mut self,
        id: EmitterId,
        position: [f32; 3],
        acoustics: &Acoustics,
        cutoff_hz: f64,
    ) -> bool;

    /// Close an emitter's track, stopping what plays on it.
    fn remove_emitter(&mut self, id: EmitterId);

    /// Move an emitter's track.
    fn move_emitter(&mut self, id: EmitterId, position: [f32; 3]);

    /// Lower an emitter's track by `volume_db` behind a lowpass at `cutoff_hz`.
    fn muffle_emitter(&mut self, id: EmitterId, volume_db: f32, cutoff_hz: f64);

    /// Start the decoded clip `key` on `on` at `gain_db`, looping or once. The
    /// voice it plays on, or `None` when it could not start.
    fn play(&mut self, key: u64, on: PlayOn, gain_db: f32, looping: bool) -> Option<VoiceId>;

    /// Stop a voice with a short click-free fade.
    fn stop(&mut self, voice: VoiceId);

    /// Whether a voice has finished playing; a finished voice is forgotten.
    fn finished(&mut self, voice: VoiceId) -> bool;

    /// Set the master track or one bus to `db`, fading briefly so a live change
    /// is click-free.
    fn set_volume(&mut self, target: AudioTarget, db: f32);

    /// Place the listener.
    fn set_listener(&mut self, position: [f32; 3], orientation: Quat);
}

/// A recording [`AudioOut`] for this crate's tests. Decodes finish on the next
/// [`AudioOut::decoded`] poll; bytes reading `bad` fail.
#[cfg(test)]
pub(crate) mod recording {
    use alloc::collections::{BTreeMap, VecDeque};
    use alloc::string::ToString;
    use hashbrown::HashSet;

    use super::*;

    #[derive(Debug, Default)]
    pub(crate) struct RecordingOut {
        pending: VecDeque<(u64, Result<(), String>)>,
        pub(crate) emitters: BTreeMap<usize, (AudioBus, [f32; 3])>,
        pub(crate) playing: Vec<(VoiceId, u64, PlayOn, bool)>,
        pub(crate) stopped: Vec<VoiceId>,
        pub(crate) finished: HashSet<VoiceId>,
        pub(crate) muffles: Vec<(EmitterId, f32)>,
        pub(crate) volumes: Vec<(AudioTarget, f32)>,
        pub(crate) listener: Option<([f32; 3], Quat)>,
        pub(crate) refuse_emitters: bool,
        next_voice: u64,
    }

    impl AudioOut for RecordingOut {
        fn decode(&mut self, key: u64, bytes: Vec<u8>) -> bool {
            let result = if bytes == b"bad" {
                Err("not audio".to_string())
            } else {
                Ok(())
            };
            self.pending.push_back((key, result));
            true
        }

        fn decoded(&mut self) -> Option<(u64, Result<(), String>)> {
            self.pending.pop_front()
        }

        fn add_emitter(
            &mut self,
            id: EmitterId,
            position: [f32; 3],
            acoustics: &Acoustics,
            _cutoff_hz: f64,
        ) -> bool {
            if self.refuse_emitters {
                return false;
            }
            self.emitters.insert(id.0, (acoustics.bus, position));
            true
        }

        fn remove_emitter(&mut self, id: EmitterId) {
            self.emitters.remove(&id.0);
        }

        fn move_emitter(&mut self, id: EmitterId, position: [f32; 3]) {
            if let Some(emitter) = self.emitters.get_mut(&id.0) {
                emitter.1 = position;
            }
        }

        fn muffle_emitter(&mut self, id: EmitterId, volume_db: f32, _cutoff_hz: f64) {
            self.muffles.push((id, volume_db));
        }

        fn play(&mut self, key: u64, on: PlayOn, _gain_db: f32, looping: bool) -> Option<VoiceId> {
            if let PlayOn::Emitter(id) = on
                && !self.emitters.contains_key(&id.0)
            {
                return None;
            }
            let voice = VoiceId(self.next_voice);
            self.next_voice += 1;
            self.playing.push((voice, key, on, looping));
            Some(voice)
        }

        fn stop(&mut self, voice: VoiceId) {
            self.stopped.push(voice);
        }

        fn finished(&mut self, voice: VoiceId) -> bool {
            self.finished.contains(&voice)
        }

        fn set_volume(&mut self, target: AudioTarget, db: f32) {
            self.volumes.push((target, db));
        }

        fn set_listener(&mut self, position: [f32; 3], orientation: Quat) {
            self.listener = Some((position, orientation));
        }
    }
}

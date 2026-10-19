//! Clip decode bookkeeping: which clips are in flight on the host's decoder,
//! which are ready, and the play requests waiting on each. A pure state machine,
//! so the request / complete / apply flow is testable without an audio device;
//! the decoded sound data stays in the host's output.

use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::HashMap;
use hashbrown::hash_map::Entry;

use super::EmitterId;
use crate::components::AudioBus;

/// A play request deferred until its clip finishes decoding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PendingPlay {
    /// A clip on an emitter's spatial track.
    Emitter {
        /// The emitter it plays on.
        id: EmitterId,
        /// Whether it loops.
        looping: bool,
        /// Its gain.
        gain: f32,
    },
    /// The music track.
    Music {
        /// The clip key.
        key: u64,
        /// Its gain.
        gain: f32,
        /// The bus it plays on.
        bus: AudioBus,
    },
    /// A non-spatial one-shot.
    Sound {
        /// The bus it plays on.
        bus: AudioBus,
        /// Its gain.
        gain: f32,
        /// Its voice priority.
        priority: i32,
    },
    /// A one-shot at a world position.
    SoundAt {
        /// Where it plays.
        position: [f32; 3],
        /// Its gain.
        gain: f32,
        /// Its voice priority.
        priority: i32,
    },
}

/// Decode progress of one clip.
#[derive(Debug, Clone, PartialEq)]
pub enum ClipState {
    /// On the decoder, with the play requests waiting on it in arrival order.
    InFlight(Vec<PendingPlay>),
    /// Decoded and playable.
    Ready,
    /// The decode failed.
    Failed,
}

/// Every clip handed to the decoder, keyed by the caller's clip key.
#[derive(Debug, Default)]
pub struct ClipStore {
    clips: HashMap<u64, ClipState>,
}

impl ClipStore {
    /// Register `key` as decoding. Returns false when the clip is already
    /// known (in any state), so a duplicate is never re-sent to the decoder.
    pub fn begin(&mut self, key: u64) -> bool {
        match self.clips.entry(key) {
            Entry::Occupied(_) => false,
            Entry::Vacant(slot) => {
                slot.insert(ClipState::InFlight(Vec::new()));
                true
            }
        }
    }

    /// The decode state of `key`, or `None` for a clip never registered.
    pub fn state(&self, key: u64) -> Option<&ClipState> {
        self.clips.get(&key)
    }

    /// Queue a play request behind `key`'s in-flight decode. Returns false
    /// when the clip is not in flight (unknown, ready, or failed).
    pub fn defer(&mut self, key: u64, play: PendingPlay) -> bool {
        match self.clips.get_mut(&key) {
            Some(ClipState::InFlight(pending)) => {
                pending.push(play);
                true
            }
            _ => false,
        }
    }

    /// Record a decode outcome, returning the play requests that were waiting
    /// on it (empty on failure, which is logged). A result for a clip that was
    /// not in flight is ignored.
    pub fn complete(&mut self, key: u64, result: Result<(), String>) -> Vec<PendingPlay> {
        let Some(state) = self.clips.get_mut(&key) else {
            return Vec::new();
        };
        let ClipState::InFlight(pending) = state else {
            return Vec::new();
        };
        let pending = core::mem::take(pending);
        match result {
            Ok(()) => {
                *state = ClipState::Ready;
                pending
            }
            Err(e) => {
                tracing::warn!("audio clip decode failed: {e}");
                *state = ClipState::Failed;
                Vec::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const SOUND: PendingPlay = PendingPlay::Sound {
        bus: AudioBus::Sfx,
        gain: 1.0,
        priority: 0,
    };

    #[test]
    fn duplicate_decode_requests_are_suppressed() {
        let mut store = ClipStore::default();
        assert!(store.begin(7), "first request registers");
        assert!(!store.begin(7), "second request suppressed");
        store.complete(7, Ok(()));
        assert!(!store.begin(7), "ready clip never re-decodes");
        assert_eq!(store.state(7), Some(&ClipState::Ready));
    }

    #[test]
    fn plays_deferred_in_flight_flush_on_completion_in_order() {
        let mut store = ClipStore::default();
        store.begin(1);
        assert!(store.defer(1, SOUND));
        let music = PendingPlay::Music {
            key: 1,
            gain: 0.5,
            bus: AudioBus::Music,
        };
        assert!(store.defer(1, music));

        let flushed = store.complete(1, Ok(()));
        assert_eq!(
            flushed,
            vec![SOUND, music],
            "requests flush in arrival order"
        );
        assert!(
            !store.defer(1, SOUND),
            "ready clip plays immediately, not deferred"
        );
        assert_eq!(store.state(1), Some(&ClipState::Ready));
    }

    #[test]
    fn failed_decode_drops_its_pending_plays() {
        let mut store = ClipStore::default();
        store.begin(3);
        store.defer(3, SOUND);
        let flushed = store.complete(3, Err("bad bytes".into()));
        assert!(flushed.is_empty());
        assert_eq!(store.state(3), Some(&ClipState::Failed));
        assert!(!store.defer(3, SOUND), "failed clip refuses new deferrals");
    }

    #[test]
    fn unknown_clips_are_inert() {
        let mut store = ClipStore::default();
        assert_eq!(store.state(99), None);
        assert!(!store.defer(99, SOUND));
        assert!(
            store.complete(99, Ok(())).is_empty(),
            "stray result ignored"
        );
        assert_eq!(store.state(99), None, "stray result does not register");
    }
}

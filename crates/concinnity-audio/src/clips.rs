// concinnity-audio/src/clips.rs
//
// Clip decode bookkeeping: which clips are in flight on the decode worker,
// which are ready, and the play requests waiting on each. Pure state machine,
// generic over the decoded payload so the request/complete/apply flow is
// testable without kira; the engine stores decoded kira sound data in it.

use std::collections::HashMap;

use concinnity_core::components::AudioBus;

use crate::EmitterId;

// A play request deferred until its clip finishes decoding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PendingPlay {
    Emitter {
        id: EmitterId,
        looping: bool,
        gain: f32,
    },
    Music {
        key: u64,
        gain: f32,
        bus: AudioBus,
    },
    Sound {
        bus: AudioBus,
        gain: f32,
        priority: i32,
    },
    SoundAt {
        position: [f32; 3],
        gain: f32,
        priority: i32,
    },
}

// Decode progress of one clip, without its payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClipState {
    InFlight,
    Ready,
    Failed,
}

enum ClipEntry<D> {
    InFlight { pending: Vec<PendingPlay> },
    Ready(D),
    Failed,
}

// Every clip handed to the decode worker, keyed by the caller's clip key.
pub(crate) struct ClipStore<D> {
    clips: HashMap<u64, ClipEntry<D>>,
}

impl<D> ClipStore<D> {
    pub(crate) fn new() -> Self {
        Self {
            clips: HashMap::new(),
        }
    }

    // Register `key` as decoding. Returns false when the clip is already
    // known (in any state), so a duplicate is never re-sent to the worker.
    pub(crate) fn begin(&mut self, key: u64) -> bool {
        match self.clips.entry(key) {
            std::collections::hash_map::Entry::Occupied(_) => false,
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(ClipEntry::InFlight {
                    pending: Vec::new(),
                });
                true
            }
        }
    }

    pub(crate) fn state(&self, key: u64) -> Option<ClipState> {
        self.clips.get(&key).map(|e| match e {
            ClipEntry::InFlight { .. } => ClipState::InFlight,
            ClipEntry::Ready(_) => ClipState::Ready,
            ClipEntry::Failed => ClipState::Failed,
        })
    }

    // The decoded payload, if `key` finished successfully.
    pub(crate) fn get(&self, key: u64) -> Option<&D> {
        match self.clips.get(&key) {
            Some(ClipEntry::Ready(data)) => Some(data),
            _ => None,
        }
    }

    // Queue a play request behind `key`'s in-flight decode. Returns false
    // when the clip is not in flight (unknown, ready, or failed).
    pub(crate) fn defer(&mut self, key: u64, play: PendingPlay) -> bool {
        match self.clips.get_mut(&key) {
            Some(ClipEntry::InFlight { pending }) => {
                pending.push(play);
                true
            }
            _ => false,
        }
    }

    // Record a decode outcome, returning the play requests that were waiting
    // on it (empty on failure; the caller logs). A result for a clip that was
    // never registered is ignored.
    pub(crate) fn complete(&mut self, key: u64, result: Result<D, String>) -> Vec<PendingPlay> {
        let Some(entry) = self.clips.get_mut(&key) else {
            return Vec::new();
        };
        let pending = match entry {
            ClipEntry::InFlight { pending } => std::mem::take(pending),
            _ => return Vec::new(),
        };
        match result {
            Ok(data) => {
                *entry = ClipEntry::Ready(data);
                pending
            }
            Err(e) => {
                tracing::warn!("audio clip decode failed: {e}");
                *entry = ClipEntry::Failed;
                Vec::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOUND: PendingPlay = PendingPlay::Sound {
        bus: AudioBus::Sfx,
        gain: 1.0,
        priority: 0,
    };

    #[test]
    fn duplicate_decode_requests_are_suppressed() {
        let mut store: ClipStore<u32> = ClipStore::new();
        assert!(store.begin(7), "first request registers");
        assert!(!store.begin(7), "second request suppressed");
        store.complete(7, Ok(42));
        assert!(!store.begin(7), "ready clip never re-decodes");
        assert_eq!(store.get(7), Some(&42));
    }

    #[test]
    fn plays_deferred_in_flight_flush_on_completion_in_order() {
        let mut store: ClipStore<u32> = ClipStore::new();
        store.begin(1);
        assert!(store.defer(1, SOUND));
        let music = PendingPlay::Music {
            key: 1,
            gain: 0.5,
            bus: AudioBus::Music,
        };
        assert!(store.defer(1, music));

        let flushed = store.complete(1, Ok(9));
        assert_eq!(
            flushed,
            vec![SOUND, music],
            "requests flush in arrival order"
        );
        assert!(
            !store.defer(1, SOUND),
            "ready clip plays immediately, not deferred"
        );
        assert_eq!(store.state(1), Some(ClipState::Ready));
    }

    #[test]
    fn failed_decode_drops_its_pending_plays() {
        let mut store: ClipStore<u32> = ClipStore::new();
        store.begin(3);
        store.defer(3, SOUND);
        let flushed = store.complete(3, Err("bad bytes".into()));
        assert!(flushed.is_empty());
        assert_eq!(store.state(3), Some(ClipState::Failed));
        assert_eq!(store.get(3), None);
        assert!(!store.defer(3, SOUND), "failed clip refuses new deferrals");
    }

    #[test]
    fn unknown_clips_are_inert() {
        let mut store: ClipStore<u32> = ClipStore::new();
        assert_eq!(store.state(99), None);
        assert!(!store.defer(99, SOUND));
        assert!(store.complete(99, Ok(1)).is_empty(), "stray result ignored");
        assert_eq!(store.state(99), None, "stray result does not register");
    }
}

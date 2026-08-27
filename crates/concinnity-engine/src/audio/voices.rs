// src/audio/voices.rs
//
// One-shot voice ledger: caps how many one-shot sounds play at once and
// decides which playing voice a new sound may silence. Pure bookkeeping,
// generic over the playback handle so the policy is testable without a
// device; the engine stores kira sound handles in it.

// Playing one-shot voices, capped at `cap`. Looping emitters and the single
// music track are not voices and never count against the cap.
pub(crate) struct VoiceSlots<H> {
    voices: Vec<Voice<H>>,
    cap: usize,
    next_seq: u64,
}

struct Voice<H> {
    handle: H,
    priority: i32,
    seq: u64,
}

// Outcome of asking for a slot for a new sound of a given priority.
pub(crate) enum Admission<H> {
    // A slot is free; play and `admit`.
    Available,
    // The pool was full; this voice was removed to make room. Stop its
    // playback, then play and `admit`.
    Steal(H),
    // Everything playing outranks the new sound; skip it.
    Refused,
}

impl<H> VoiceSlots<H> {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            voices: Vec::new(),
            cap,
            next_seq: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.voices.len()
    }

    // Drop every voice whose playback has finished.
    pub(crate) fn reap(&mut self, finished: impl Fn(&H) -> bool) {
        self.voices.retain(|v| !finished(&v.handle));
    }

    // Make room for a new sound of `priority`. When the pool is full, the
    // lowest-priority voice (oldest among equals) is stolen if the new sound
    // matches or outranks it; otherwise the new sound is refused.
    pub(crate) fn make_room(&mut self, priority: i32) -> Admission<H> {
        if self.voices.len() < self.cap {
            return Admission::Available;
        }
        let victim = self
            .voices
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| (v.priority, v.seq))
            .map(|(i, _)| i);
        match victim {
            Some(i) if self.voices[i].priority <= priority => {
                Admission::Steal(self.voices.swap_remove(i).handle)
            }
            _ => Admission::Refused,
        }
    }

    // Record a playing voice. Call only after `make_room` returned
    // `Available` or `Steal`.
    pub(crate) fn admit(&mut self, priority: i32, handle: H) {
        debug_assert!(self.voices.len() < self.cap);
        let seq = self.next_seq;
        self.next_seq += 1;
        self.voices.push(Voice {
            handle,
            priority,
            seq,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(cap: usize, priorities: &[i32]) -> VoiceSlots<u32> {
        let mut slots = VoiceSlots::new(cap);
        for (i, &p) in priorities.iter().enumerate() {
            assert!(matches!(slots.make_room(p), Admission::Available));
            slots.admit(p, i as u32);
        }
        slots
    }

    #[test]
    fn below_cap_always_has_room() {
        let mut slots = filled(4, &[0, 0, 0]);
        assert!(matches!(slots.make_room(i32::MIN), Admission::Available));
        assert_eq!(slots.len(), 3);
    }

    #[test]
    fn full_pool_steals_the_lowest_priority_voice() {
        let mut slots = filled(3, &[5, 1, 3]);
        match slots.make_room(2) {
            Admission::Steal(handle) => assert_eq!(handle, 1, "voice with priority 1 stolen"),
            _ => panic!("expected a steal"),
        }
        assert_eq!(slots.len(), 2);
    }

    #[test]
    fn equal_priorities_steal_the_oldest() {
        let mut slots = filled(3, &[2, 2, 2]);
        match slots.make_room(2) {
            Admission::Steal(handle) => assert_eq!(handle, 0, "first-admitted voice stolen"),
            _ => panic!("expected a steal"),
        }
    }

    #[test]
    fn outranked_sound_is_refused() {
        let mut slots = filled(2, &[4, 6]);
        assert!(matches!(slots.make_room(3), Admission::Refused));
        assert_eq!(slots.len(), 2, "refusal removes nothing");
    }

    #[test]
    fn reap_frees_slots_for_new_sounds() {
        let mut slots = filled(2, &[9, 9]);
        assert!(matches!(slots.make_room(0), Admission::Refused));
        slots.reap(|&h| h == 0);
        assert_eq!(slots.len(), 1);
        assert!(matches!(slots.make_room(0), Admission::Available));
    }

    #[test]
    fn zero_cap_refuses_everything() {
        let mut slots: VoiceSlots<u32> = VoiceSlots::new(0);
        assert!(matches!(slots.make_room(i32::MAX), Admission::Refused));
    }
}

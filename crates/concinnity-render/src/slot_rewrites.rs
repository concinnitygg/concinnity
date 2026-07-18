// src/slot_rewrites.rs
//
// Propagates a streamed texture-pool slot swap across per-frame descriptor
// copies without stalling the device. Backends that bake the texture pool
// into per-frame-in-flight descriptor sets (Vulkan) or heap regions (DirectX)
// cannot legally rewrite a descriptor while a command buffer referencing it
// is pending; instead a swap queues its slot here, and each `draw_frame`
// applies the queued slots to the copy owned by the frame slot it just
// fence-waited (which is therefore not referenced by any pending work). An
// entry retires once every frame-in-flight copy has been rewritten.

#[derive(Debug)]
pub struct SlotRewriteQueue {
    // (pool slot, applications remaining). A slot appears at most once.
    entries: Vec<(usize, usize)>,
    frames_in_flight: usize,
}

impl SlotRewriteQueue {
    pub fn new(frames_in_flight: usize) -> Self {
        Self {
            entries: Vec::new(),
            frames_in_flight: frames_in_flight.max(1),
        }
    }

    // Queue pool `slot` for rewriting in every per-frame copy. Re-queueing a
    // slot mid-propagation restarts its countdown, so every copy ends on the
    // view the caller swapped in last (the rewrite reads the pool at apply
    // time, not at queue time).
    pub fn queue(&mut self, slot: usize) {
        if let Some(entry) = self.entries.iter_mut().find(|(s, _)| *s == slot) {
            entry.1 = self.frames_in_flight;
        } else {
            self.entries.push((slot, self.frames_in_flight));
        }
    }

    // The slots whose descriptor must be rewritten in the frame copy about to
    // record. Call exactly once per frame, after the frame slot's fence wait.
    // Entries that have now covered every copy are dropped.
    pub fn begin_frame(&mut self) -> Vec<usize> {
        let slots: Vec<usize> = self.entries.iter().map(|(s, _)| *s).collect();
        for entry in &mut self.entries {
            entry.1 -= 1;
        }
        self.entries.retain(|(_, remaining)| *remaining > 0);
        slots
    }

    // Drop a queued slot: the caller rewrote every copy itself (e.g. a
    // fallback full rewrite under a device drain).
    pub fn remove(&mut self, slot: usize) {
        self.entries.retain(|(s, _)| *s != slot);
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slot_is_applied_once_per_frame_copy() {
        let mut q = SlotRewriteQueue::new(3);
        q.queue(7);
        assert_eq!(q.begin_frame(), vec![7]);
        assert_eq!(q.begin_frame(), vec![7]);
        assert_eq!(q.begin_frame(), vec![7]);
        assert!(q.begin_frame().is_empty());
        assert!(q.is_empty());
    }

    #[test]
    fn requeueing_mid_propagation_restarts_the_countdown() {
        let mut q = SlotRewriteQueue::new(3);
        q.queue(4);
        assert_eq!(q.begin_frame(), vec![4]);
        // A second swap of the same slot before propagation finished: the
        // remaining copies must still converge on the latest view.
        q.queue(4);
        assert_eq!(q.begin_frame(), vec![4]);
        assert_eq!(q.begin_frame(), vec![4]);
        assert_eq!(q.begin_frame(), vec![4]);
        assert!(q.begin_frame().is_empty());
    }

    #[test]
    fn independent_slots_propagate_independently() {
        let mut q = SlotRewriteQueue::new(2);
        q.queue(1);
        assert_eq!(q.begin_frame(), vec![1]);
        q.queue(2);
        let mut slots = q.begin_frame();
        slots.sort_unstable();
        assert_eq!(slots, vec![1, 2]);
        assert_eq!(q.begin_frame(), vec![2]);
        assert!(q.begin_frame().is_empty());
    }

    #[test]
    fn remove_drops_a_queued_slot() {
        let mut q = SlotRewriteQueue::new(3);
        q.queue(5);
        q.queue(6);
        q.remove(5);
        assert_eq!(q.begin_frame(), vec![6]);
    }

    #[test]
    fn zero_frames_in_flight_is_clamped_to_one() {
        let mut q = SlotRewriteQueue::new(0);
        q.queue(9);
        assert_eq!(q.begin_frame(), vec![9]);
        assert!(q.is_empty());
    }
}

// GraphicsSystem per-frame model-matrix push: a slot-indexed cache of the
// last matrix sent to the backend plus the frame's batched update list, so
// unchanged slots cost a 64-byte compare instead of a backend call and the
// backend trait is crossed once per frame instead of once per slot.

type Mat4 = [[f32; 4]; 4];

// Change gate + batch builder for one family of backend slots (static draw
// objects or skinned instances). Both buffers persist across frames so a
// steady-state frame allocates nothing.
#[derive(Default)]
pub(crate) struct ModelPushCache {
    // Last matrix pushed per slot; `None` for a slot never pushed.
    pushed: Vec<Option<Mat4>>,
    // The entries to send this frame, in push order (a slot pushed twice
    // keeps both entries; the backend applies them in order, so the last
    // write wins, matching the per-slot calls this replaced).
    batch: Vec<(u32, Mat4)>,
}

impl ModelPushCache {
    // Start a new frame's batch.
    pub(crate) fn begin(&mut self) {
        self.batch.clear();
    }

    // Queue `model` for `slot` unless it matches the last pushed value.
    pub(crate) fn push_changed(&mut self, slot: usize, model: Mat4) {
        if self.pushed.len() <= slot {
            self.pushed.resize(slot + 1, None);
        }
        if self.pushed[slot] == Some(model) {
            return;
        }
        self.pushed[slot] = Some(model);
        self.batch.push((slot as u32, model));
    }

    // Queue `model` for `slot` unconditionally (for callers with their own
    // exact change gate), still recording it as the last pushed value.
    pub(crate) fn push(&mut self, slot: usize, model: Mat4) {
        if self.pushed.len() <= slot {
            self.pushed.resize(slot + 1, None);
        }
        self.pushed[slot] = Some(model);
        self.batch.push((slot as u32, model));
    }

    // The frame's accumulated updates, ready for one backend call.
    pub(crate) fn batch(&self) -> &[(u32, Mat4)] {
        &self.batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translated(x: f32) -> Mat4 {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [x, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn first_push_always_lands_and_repeats_are_dropped() {
        let mut cache = ModelPushCache::default();
        cache.begin();
        cache.push_changed(3, translated(1.0));
        assert_eq!(cache.batch(), &[(3, translated(1.0))]);

        cache.begin();
        cache.push_changed(3, translated(1.0));
        assert!(cache.batch().is_empty(), "unchanged matrix is not re-sent");

        cache.begin();
        cache.push_changed(3, translated(2.0));
        assert_eq!(
            cache.batch(),
            &[(3, translated(2.0))],
            "a move is sent once"
        );
    }

    #[test]
    fn unconditional_push_updates_the_gate() {
        let mut cache = ModelPushCache::default();
        cache.begin();
        cache.push(0, translated(5.0));
        assert_eq!(cache.batch().len(), 1);

        // The recorded value gates the next conditional push.
        cache.begin();
        cache.push_changed(0, translated(5.0));
        assert!(cache.batch().is_empty());
        cache.push_changed(0, translated(6.0));
        assert_eq!(cache.batch().len(), 1);
    }

    #[test]
    fn double_push_of_one_slot_keeps_both_entries_in_order() {
        // The editor's hidden-object overwrite pushes a second value for a
        // slot in the same frame; both must reach the backend in order so
        // the overwrite wins.
        let mut cache = ModelPushCache::default();
        cache.begin();
        cache.push_changed(1, translated(1.0));
        cache.push_changed(1, translated(0.0));
        assert_eq!(cache.batch(), &[(1, translated(1.0)), (1, translated(0.0))]);
    }

    #[test]
    fn warm_buffers_are_reused_without_reallocating() {
        let mut cache = ModelPushCache::default();
        cache.begin();
        for slot in 0..16 {
            cache.push_changed(slot, translated(slot as f32));
        }
        let ptr = cache.batch.as_ptr();
        for pass in 0..3 {
            cache.begin();
            for slot in 0..16 {
                cache.push_changed(slot, translated(slot as f32 + pass as f32 + 1.0));
            }
            assert_eq!(cache.batch.as_ptr(), ptr, "batch buffer was reallocated");
        }
    }
}

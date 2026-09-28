// GraphicsSystem per-frame model-matrix change gate: a slot-indexed cache of
// the last matrix sent to the backend, so unchanged slots cost a 64-byte
// compare instead of a snapshot entry and the backend trait is crossed once
// per frame per family instead of once per slot. The frame's batch lives in
// the RenderSnapshot; this holds only the cross-frame dedupe state.

use std::marker::PhantomData;

use concinnity_core::gfx::render_types::{DrawIndex, SkinnedIndex};

type Mat4 = [[f32; 4]; 4];

// A backend slot family the gate can key: its position in the cache.
pub(crate) trait Slot: Copy {
    fn position(self) -> usize;
}

impl Slot for DrawIndex {
    fn position(self) -> usize {
        self.index()
    }
}

impl Slot for SkinnedIndex {
    fn position(self) -> usize {
        self.index()
    }
}

// Change gate for one family of backend slots (static draw objects or skinned
// instances). The `pushed` buffer persists across frames so a steady-state
// frame allocates nothing. The gate stays valid only while every batch it
// fills is submitted to the backend exactly once, in order.
pub(crate) struct ModelPushCache<I> {
    // Last matrix pushed per slot; `None` for a slot never pushed.
    pushed: Vec<Option<Mat4>>,
    family: PhantomData<I>,
}

impl<I> Default for ModelPushCache<I> {
    fn default() -> Self {
        Self {
            pushed: Vec::new(),
            family: PhantomData,
        }
    }
}

impl<I: Slot> ModelPushCache<I> {
    // Queue `model` for `slot` unless it matches the last pushed value. A slot
    // pushed twice keeps both entries; the backend applies them in order, so
    // the last write wins, matching the per-slot calls this replaced.
    pub(crate) fn push_changed(&mut self, batch: &mut Vec<(I, Mat4)>, slot: I, model: Mat4) {
        if *self.entry(slot) == Some(model) {
            return;
        }
        self.push(batch, slot, model);
    }

    // Queue `model` for `slot` unconditionally (for callers with their own
    // exact change gate), still recording it as the last pushed value.
    pub(crate) fn push(&mut self, batch: &mut Vec<(I, Mat4)>, slot: I, model: Mat4) {
        *self.entry(slot) = Some(model);
        batch.push((slot, model));
    }

    fn entry(&mut self, slot: I) -> &mut Option<Mat4> {
        let i = slot.position();
        if self.pushed.len() <= i {
            self.pushed.resize(i + 1, None);
        }
        &mut self.pushed[i]
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
        let mut cache = ModelPushCache::<DrawIndex>::default();
        let mut batch = Vec::new();
        cache.push_changed(&mut batch, DrawIndex(3), translated(1.0));
        assert_eq!(batch, vec![(DrawIndex(3), translated(1.0))]);

        batch.clear();
        cache.push_changed(&mut batch, DrawIndex(3), translated(1.0));
        assert!(batch.is_empty(), "unchanged matrix is not re-sent");

        cache.push_changed(&mut batch, DrawIndex(3), translated(2.0));
        assert_eq!(
            batch,
            vec![(DrawIndex(3), translated(2.0))],
            "a move is sent once"
        );
    }

    #[test]
    fn unconditional_push_updates_the_gate() {
        let mut cache = ModelPushCache::<DrawIndex>::default();
        let mut batch = Vec::new();
        cache.push(&mut batch, DrawIndex(0), translated(5.0));
        assert_eq!(batch.len(), 1);

        // The recorded value gates the next conditional push.
        batch.clear();
        cache.push_changed(&mut batch, DrawIndex(0), translated(5.0));
        assert!(batch.is_empty());
        cache.push_changed(&mut batch, DrawIndex(0), translated(6.0));
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn double_push_of_one_slot_keeps_both_entries_in_order() {
        // The editor's hidden-object overwrite pushes a second value for a
        // slot in the same frame; both must reach the backend in order so
        // the overwrite wins.
        let mut cache = ModelPushCache::<DrawIndex>::default();
        let mut batch = Vec::new();
        cache.push_changed(&mut batch, DrawIndex(1), translated(1.0));
        cache.push_changed(&mut batch, DrawIndex(1), translated(0.0));
        assert_eq!(
            batch,
            vec![
                (DrawIndex(1), translated(1.0)),
                (DrawIndex(1), translated(0.0))
            ]
        );
    }
}

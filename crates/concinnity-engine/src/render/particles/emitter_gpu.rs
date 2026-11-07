//! A backend's per-emitter GPU state, kept by the emitter id the scene's
//! [`ParticleSystem`](super::ParticleSystem) hands out.

use alloc::vec::Vec;

/// The GPU state `G` of each live emitter, by emitter id.
///
/// ```rust
/// # use concinnity_engine::render::particles::EmitterGpu;
/// let mut gpu = EmitterGpu::new();
/// gpu.set(2, "pool");
/// assert_eq!(gpu.get(2), Some(&"pool"));
/// assert_eq!(gpu.take(2), Some("pool"));
/// assert_eq!(gpu.get(2), None);
/// ```
pub struct EmitterGpu<G> {
    slots: Vec<Option<G>>,
}

impl<G> Default for EmitterGpu<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G> EmitterGpu<G> {
    /// No emitter's state.
    pub const fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Keep `gpu` as emitter `id`'s state, replacing any it had.
    pub fn set(&mut self, id: usize, gpu: G) {
        if self.slots.len() <= id {
            self.slots.resize_with(id + 1, || None);
        }
        self.slots[id] = Some(gpu);
    }

    /// Emitter `id`'s state, if it has one.
    pub fn get(&self, id: usize) -> Option<&G> {
        self.slots.get(id)?.as_ref()
    }

    /// Hand emitter `id`'s state back for the backend to release.
    pub fn take(&mut self, id: usize) -> Option<G> {
        self.slots.get_mut(id)?.take()
    }

    /// Every emitter's state, in id order.
    pub fn iter(&self) -> impl Iterator<Item = &G> {
        self.slots.iter().flatten()
    }

    /// Drop every emitter's state.
    pub fn clear(&mut self) {
        self.slots.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn states_are_kept_by_id_and_iterate_in_id_order() {
        let mut gpu = EmitterGpu::new();
        gpu.set(3, 'c');
        gpu.set(0, 'a');
        gpu.set(3, 'd');
        assert_eq!(gpu.iter().copied().collect::<Vec<_>>(), vec!['a', 'd']);
        assert_eq!(gpu.get(1), None);
        assert_eq!(gpu.get(9), None);
        assert_eq!(gpu.take(9), None);
        assert_eq!(gpu.take(0), Some('a'));
        assert_eq!(gpu.iter().copied().collect::<Vec<_>>(), vec!['d']);
        gpu.clear();
        assert_eq!(gpu.iter().count(), 0);
    }
}

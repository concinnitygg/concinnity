//! Assembly buffer for an overlay draw list: the calls built so far plus a
//! pool of spent vertex/index buffers. A frame's spent list is recycled back
//! in whole, so steady-state assembly reuses both the list and every call's
//! geometry allocations.

use crate::render_types::{TextDrawCall, TextVertex};

#[derive(Default)]
/// An overlay draw list under assembly, with its recycled geometry pool.
pub struct TextCallBuffer {
    /// Calls assembled so far, in draw order.
    pub calls: Vec<TextDrawCall>,
    // Spent geometry buffers (cleared, capacity intact) awaiting reuse.
    spare: Vec<(Vec<TextVertex>, Vec<u16>)>,
    // Index scratch for the layer sort, reused across frames.
    sort_order: Vec<u32>,
    sort_dest: Vec<u32>,
}

impl TextCallBuffer {
    /// Reclaim a spent draw list: each call's geometry feeds later builds and
    /// the drained list becomes the backing for the next `calls`.
    pub fn recycle(&mut self, mut spent: Vec<TextDrawCall>) {
        debug_assert!(self.calls.is_empty(), "recycling over an untaken build");
        for call in spent.drain(..) {
            let (mut vertices, mut indices) = (call.vertices, call.indices);
            vertices.clear();
            indices.clear();
            self.spare.push((vertices, indices));
        }
        self.calls = spent;
    }

    /// An empty vertex/index buffer pair, reusing spent capacity when any is
    /// pooled.
    pub fn geometry(&mut self) -> (Vec<TextVertex>, Vec<u16>) {
        self.spare.pop().unwrap_or_default()
    }

    /// Return a pair taken with [`TextCallBuffer::geometry`] that ended up
    /// unused, so its capacity stays pooled.
    pub fn park(&mut self, mut vertices: Vec<TextVertex>, mut indices: Vec<u16>) {
        vertices.clear();
        indices.clear();
        self.spare.push((vertices, indices));
    }

    /// Hand the assembled list to its consumer, leaving this buffer empty.
    pub fn take(&mut self) -> Vec<TextDrawCall> {
        std::mem::take(&mut self.calls)
    }

    /// Reorder the assembled calls by ascending draw layer, keeping same-layer
    /// calls in insertion order. Equivalent to a stable sort by layer, but the
    /// sort runs over indices in persistent scratch and the permutation is
    /// applied with swaps, so a steady-state frame allocates nothing. Skipped
    /// entirely when every call sits at layer 0.
    pub fn sort_by_layer(&mut self) {
        if self.calls.iter().all(|c| c.layer == 0) {
            return;
        }
        let calls = &mut self.calls;
        // `order[new] = old`: ties broken by index, so equal layers keep their
        // insertion order.
        let order = &mut self.sort_order;
        order.clear();
        order.extend(0..calls.len() as u32);
        order.sort_unstable_by_key(|&i| (calls[i as usize].layer, i));
        // Invert to `dest[old] = new`, then walk each swap cycle in place.
        let dest = &mut self.sort_dest;
        dest.clear();
        dest.resize(calls.len(), 0);
        for (new, &old) in order.iter().enumerate() {
            dest[old as usize] = new as u32;
        }
        for i in 0..calls.len() {
            while dest[i] as usize != i {
                let j = dest[i] as usize;
                calls.swap(i, j);
                dest.swap(i, j);
            }
        }
    }
}

// TextDrawCall carries no Debug; systems holding a buffer still derive it.
impl std::fmt::Debug for TextCallBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextCallBuffer")
            .field("calls", &self.calls.len())
            .field("spare", &self.spare.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(quads: usize) -> TextDrawCall {
        TextDrawCall {
            vertices: Vec::with_capacity(4 * quads),
            indices: Vec::with_capacity(6 * quads),
            atlas_slot: 0,
            clip_rect: None,
            layer: 0,
        }
    }

    #[test]
    fn recycle_reuses_the_list_and_its_geometry() {
        let mut buf = TextCallBuffer::default();
        let mut spent = vec![call(8), call(4)];
        spent[0].vertices.push(TextVertex {
            pos: [0.0, 0.0],
            uv: [0.0, 0.0],
            color: [0.0; 3],
            mode: 0.0,
        });
        let list_ptr = spent.as_ptr();
        let geom_cap = spent[1].vertices.capacity();

        buf.recycle(spent);
        assert_eq!(buf.calls.as_ptr(), list_ptr, "list backing reused");
        assert!(buf.calls.is_empty());

        // Pooled buffers come back cleared with their capacity intact.
        let (vertices, indices) = buf.geometry();
        assert!(vertices.is_empty() && indices.is_empty());
        assert!(vertices.capacity() >= geom_cap);
        let (second, _) = buf.geometry();
        assert!(second.is_empty(), "the pushed vertex was cleared");
        assert!(second.capacity() >= 32);
        // The pool is drained; further requests allocate fresh.
        assert_eq!(buf.geometry().0.capacity(), 0);
    }

    #[test]
    fn take_leaves_an_empty_buffer() {
        let mut buf = TextCallBuffer::default();
        buf.calls.push(call(1));
        assert_eq!(buf.take().len(), 1);
        assert!(buf.calls.is_empty());
    }

    // A call tagged through `atlas_slot` so a sort's reordering is observable.
    fn layered(layer: i32, tag: usize) -> TextDrawCall {
        TextDrawCall {
            layer,
            atlas_slot: tag,
            ..call(0)
        }
    }

    #[test]
    fn sort_by_layer_orders_ascending_and_keeps_same_layer_insertion_order() {
        let mut buf = TextCallBuffer::default();
        for (layer, tag) in [(2, 0), (0, 1), (1, 2), (0, 3), (2, 4)] {
            buf.calls.push(layered(layer, tag));
        }
        buf.sort_by_layer();
        let got: Vec<(i32, usize)> = buf.calls.iter().map(|c| (c.layer, c.atlas_slot)).collect();
        assert_eq!(got, [(0, 1), (0, 3), (1, 2), (2, 0), (2, 4)]);
    }

    #[test]
    fn sort_by_layer_leaves_an_unlayered_list_untouched() {
        let mut buf = TextCallBuffer::default();
        for tag in 0..4 {
            buf.calls.push(layered(0, tag));
        }
        buf.sort_by_layer();
        let tags: Vec<usize> = buf.calls.iter().map(|c| c.atlas_slot).collect();
        assert_eq!(tags, [0, 1, 2, 3]);
    }

    // The scratch is reused across sorts of different lengths, so a second sort
    // after a shorter frame still permutes correctly.
    #[test]
    fn sort_by_layer_is_correct_across_reuse() {
        let mut buf = TextCallBuffer::default();
        for (layer, tag) in [(3, 0), (1, 1), (2, 2)] {
            buf.calls.push(layered(layer, tag));
        }
        buf.sort_by_layer();
        buf.calls.clear();
        for (layer, tag) in [(5, 0), (4, 1)] {
            buf.calls.push(layered(layer, tag));
        }
        buf.sort_by_layer();
        let got: Vec<(i32, usize)> = buf.calls.iter().map(|c| (c.layer, c.atlas_slot)).collect();
        assert_eq!(got, [(4, 1), (5, 0)]);
    }
}

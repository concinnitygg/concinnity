// src/call_buffer.rs
//
// Assembly buffer for an overlay draw list: the calls built so far plus a
// pool of spent vertex/index buffers. A frame's spent list is recycled back
// in whole, so steady-state assembly reuses both the list and every call's
// geometry allocations.

use crate::render_types::{TextDrawCall, TextVertex};

#[derive(Default)]
pub struct TextCallBuffer {
    /// Calls assembled so far, in draw order.
    pub calls: Vec<TextDrawCall>,
    // Spent geometry buffers (cleared, capacity intact) awaiting reuse.
    spare: Vec<(Vec<TextVertex>, Vec<u16>)>,
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
}

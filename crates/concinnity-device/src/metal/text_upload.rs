// src/metal/text_upload.rs
//
// Persistent per-frame-slot upload buffer for the composite pass's transient
// HUD text geometry. Every label used to mint a vertex and an index buffer with
// `newBufferWithBytes` during the composite encode: two driver allocations per
// label per frame, each retained by the committed command buffer until the GPU
// retired the frame.
//
// Instead each frame-in-flight slot keeps one `StorageModeShared` buffer. The
// frame's whole text geometry is written into this frame's slot before the
// render graph runs, each label's vertex and index block at a rolling aligned
// offset, and the composite pass binds sub-ranges of that one buffer. A slot
// grows power-of-two on demand and is never shrunk, so steady state does zero
// allocation. The frames-in-flight fence guarantees frame `R - depth` retired
// before frame `R` reuses slot `R % depth`, so overwriting a slot never races
// an in-flight GPU read -- the same argument as `TransientRing` in
// `metal/transient.rs`.
//
// Mirrors `directx/upload_ring.rs` and `vulkan/upload_ring.rs`. Metal writes the
// frame's geometry up front rather than appending during the encode because the
// pass encoders run through `&MtlContext`, whose parallel-encode contract is
// read-only field access (see `metal/parallel_encoder.rs`).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::gfx::render_types::TextDrawCall;

use super::context::write_buffer_region;
use super::transient::grow_to;

// Sub-range alignment. 256 bytes satisfies the strictest offset rule a Metal
// buffer binding can face, and costs a HUD's worth of labels a few kilobytes.
const TEXT_UPLOAD_ALIGN: u64 = 256;

// Where one text call's vertex and index blocks sit in the frame's buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct TextRange {
    pub(super) vertex_offset: usize,
    pub(super) index_offset: usize,
}

fn align_up(offset: usize) -> usize {
    crate::gfx::fullscreen::align_up(offset as u64, TEXT_UPLOAD_ALIGN) as usize
}

// Lay out every call's vertex and index block back to back at aligned offsets,
// filling `out` with one range per call (in `text_calls` order, so the encoder
// indexes it alongside the calls), and return the total byte size. Never exceeds
// `fullscreen::text_upload_bytes` for the same calls and alignment.
fn plan_text_upload(text_calls: &[TextDrawCall], out: &mut Vec<TextRange>) -> usize {
    out.clear();
    out.reserve(text_calls.len());
    let mut cursor = 0usize;
    for call in text_calls {
        let vertex_offset = align_up(cursor);
        let index_offset =
            align_up(vertex_offset + std::mem::size_of_val(call.vertices.as_slice()));
        cursor = index_offset + std::mem::size_of_val(call.indices.as_slice());
        out.push(TextRange {
            vertex_offset,
            index_offset,
        });
    }
    cursor
}

// One shared-storage buffer per frame-in-flight slot, holding a whole frame's
// text geometry.
pub(super) struct TextUploadRing {
    slots: Vec<Option<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    // This frame's per-call block offsets, parallel to the frame's text calls.
    // Kept across frames so the per-frame layout reuses one heap allocation.
    ranges: Vec<TextRange>,
    // The slot [`Self::binding`] hands out, `None` until a frame with text has
    // been uploaded (and again on any frame that carries none).
    frame_slot: Option<usize>,
}

impl TextUploadRing {
    // `depth` is the frames-in-flight count; clamped to >= 1. Buffers are
    // allocated lazily on first use of each slot.
    pub(super) fn new(depth: usize) -> Self {
        Self {
            slots: (0..depth.max(1)).map(|_| None).collect(),
            ranges: Vec::new(),
            frame_slot: None,
        }
    }

    // Copy this frame's text geometry into `slot`'s buffer, growing it first if
    // the frame outgrew it. Call once per frame, after the frames-in-flight
    // fence has confirmed the GPU is done with this slot and before the frame's
    // composite pass encodes.
    pub(super) fn upload(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        slot: usize,
        text_calls: &[TextDrawCall],
    ) -> Result<(), String> {
        let Self {
            slots,
            ranges,
            frame_slot,
        } = self;
        *frame_slot = None;
        let total = plan_text_upload(text_calls, ranges);
        if total == 0 {
            return Ok(());
        }
        let idx = slot % slots.len();
        let have = slots[idx].as_ref().map_or(0, |buf| buf.length());
        if let Some(capacity) = grow_to(have, total) {
            slots[idx] = Some(
                device
                    .newBufferWithLength_options(capacity, MTLResourceOptions::StorageModeShared)
                    .ok_or("failed to allocate text upload buffer")?,
            );
        }
        let buffer = slots[idx]
            .as_ref()
            .expect("text ring slot was just ensured");
        for (call, range) in text_calls.iter().zip(ranges.iter()) {
            write_buffer_region(
                buffer,
                range.vertex_offset,
                bytemuck::cast_slice(&call.vertices),
            )?;
            write_buffer_region(
                buffer,
                range.index_offset,
                bytemuck::cast_slice(&call.indices),
            )?;
        }
        *frame_slot = Some(idx);
        Ok(())
    }

    // This frame's buffer plus one block range per text call, or `None` when the
    // frame uploaded no text geometry.
    pub(super) fn binding(&self) -> Option<(&ProtocolObject<dyn MTLBuffer>, &[TextRange])> {
        let buffer = self.slots[self.frame_slot?].as_ref()?;
        Some((buffer, &self.ranges))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_types::TextVertex;

    // A call carrying `glyphs` quads: 4 vertices + 6 indices each, the shape
    // `gfx::text::build_text_calls` emits.
    fn glyph_call(glyphs: usize) -> TextDrawCall {
        TextDrawCall {
            vertices: vec![
                TextVertex {
                    pos: [0.0; 2],
                    uv: [0.0; 2],
                    color: [0.0; 3],
                    mode: 0.0,
                };
                glyphs * 4
            ],
            indices: vec![0u16; glyphs * 6],
            atlas_slot: 0,
            clip_rect: None,
            layer: 0,
        }
    }

    #[test]
    fn planning_no_calls_needs_no_buffer() {
        let mut ranges = Vec::new();
        assert_eq!(plan_text_upload(&[], &mut ranges), 0);
        assert!(ranges.is_empty());
    }

    #[test]
    fn planning_emits_one_range_per_call() {
        let calls = [glyph_call(2), glyph_call(0), glyph_call(5)];
        let mut ranges = Vec::new();
        plan_text_upload(&calls, &mut ranges);
        assert_eq!(ranges.len(), calls.len());
    }

    #[test]
    fn planned_blocks_are_aligned_and_ordered() {
        let calls = [glyph_call(1), glyph_call(9), glyph_call(3)];
        let mut ranges = Vec::new();
        let total = plan_text_upload(&calls, &mut ranges);
        let align = TEXT_UPLOAD_ALIGN as usize;
        let mut prev_end = 0usize;
        for (call, range) in calls.iter().zip(ranges.iter()) {
            assert_eq!(range.vertex_offset % align, 0);
            assert_eq!(range.index_offset % align, 0);
            // Blocks never overlap the previous one, and stay inside the total.
            assert!(range.vertex_offset >= prev_end);
            let vertex_end = range.vertex_offset + std::mem::size_of_val(call.vertices.as_slice());
            assert!(range.index_offset >= vertex_end);
            prev_end = range.index_offset + std::mem::size_of_val(call.indices.as_slice());
            assert!(prev_end <= total);
        }
    }

    // The layout has to fit inside what the shared reservation helper reports,
    // which is what every other backend sizes its slot with.
    #[test]
    fn planned_total_fits_the_shared_reservation() {
        let calls = [glyph_call(4), glyph_call(1), glyph_call(120)];
        let mut ranges = Vec::new();
        let total = plan_text_upload(&calls, &mut ranges) as u64;
        let reserved = crate::gfx::fullscreen::text_upload_bytes(&calls, TEXT_UPLOAD_ALIGN);
        assert!(
            total <= reserved,
            "planned {total} exceeded reserved {reserved}"
        );
    }

    #[test]
    fn replanning_drops_the_previous_frames_ranges() {
        let mut ranges = Vec::new();
        plan_text_upload(&[glyph_call(2), glyph_call(2)], &mut ranges);
        assert_eq!(ranges.len(), 2);
        plan_text_upload(&[glyph_call(1)], &mut ranges);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].vertex_offset, 0);
    }
}

// src/directx/draw/text_upload.rs
//
// Sizing for the HUD text geometry the composite pass appends into its
// frame-slot upload buffer (see [`crate::directx::upload_ring::UploadRing`]).

use crate::directx::upload_ring::{UPLOAD_ALIGN, align_up};
use crate::gfx::render_types::{TextDrawCall, TextVertex};

// Total bytes a frame's text geometry consumes once each label's vertex and
// index blocks are aligned. Because every sub-allocation aligns its start up to
// `UPLOAD_ALIGN` and a prior aligned start plus an aligned size stays aligned,
// this sum is an exact upper bound on the slot cursor after all the pushes, so
// reserving it guarantees `push` never overflows mid-frame.
pub(super) fn text_calls_byte_size(text_calls: &[TextDrawCall]) -> u64 {
    text_calls
        .iter()
        .map(|c| {
            let v = (c.vertices.len() * std::mem::size_of::<TextVertex>()) as u64;
            let i = (c.indices.len() * std::mem::size_of::<u16>()) as u64;
            align_up(v, UPLOAD_ALIGN) + align_up(i, UPLOAD_ALIGN)
        })
        .sum()
}

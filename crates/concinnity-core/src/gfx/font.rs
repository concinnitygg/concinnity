//! Per-glyph metrics of a compiled font atlas: where a glyph sits in the atlas
//! texture and how the pen moves across it. Shared by the build-time rasteriser
//! that writes the payload, the decoder that reads it back, and the text layout
//! that turns it into quads, none of which owns the layout.

/// Per-glyph metrics stored in the compiled payload.
#[derive(Debug, Clone, Copy)]
pub struct GlyphMetrics {
    /// The Unicode code point this glyph renders.
    pub char_code: u32,
    /// Glyph's left edge in the atlas, in pixels.
    pub atlas_x: u16,
    /// Glyph's top edge in the atlas, in pixels.
    pub atlas_y: u16,
    /// Glyph width in the atlas, in pixels.
    pub atlas_w: u16,
    /// Glyph height in the atlas, in pixels.
    pub atlas_h: u16,
    /// Pen advance after this glyph, in pixels.
    pub advance_px: f32,
    /// Horizontal offset from the pen to the glyph's left edge.
    pub bearing_x: f32,
    /// Vertical offset from the baseline to the glyph's top edge.
    pub bearing_y: f32,
}

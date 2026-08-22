//! Runtime decode of the compiled font payload: a power-of-two RGBA atlas of
//! signed-distance-field glyphs plus their metrics, which GraphicsSystem uploads
//! and draws from. The rasteriser that produces the payload lives in the cook
//! crate; this half only reads bytes back off disk.
//!
//! Each atlas texel stores a normalised SDF value in [0, 1] where 0.5 = the glyph
//! outline. Values > 0.5 are inside; values < 0.5 are outside. The fragment shader
//! uses smoothstep + fwidth to reconstruct crisp, scale-independent alpha.

use crate::decode::{ByteReader, checked_product};

// Bytes each glyph's metrics occupy in the payload: char code, four atlas
// coordinates, advance, and two bearings.
const GLYPH_STRIDE: usize = 4 + 2 + 2 + 2 + 2 + 4 + 4 + 4;

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

// Decoded font payload: atlas width, atlas height, supersample factor,
// rasterisation size (px), RGBA atlas pixels, and per-glyph metrics.
pub(crate) type DecodedFont = (u32, u32, u32, u32, Vec<u8>, Vec<GlyphMetrics>);

/// Deserialise a font payload back into atlas dimensions, the atlas supersample
/// factor, the rasterisation size, RGBA pixels, and metrics.
pub fn deserialise(bytes: &[u8]) -> Result<DecodedFont, String> {
    let mut r = ByteReader::new(bytes, "font payload");
    let atlas_w = r.u32()?;
    let atlas_h = r.u32()?;
    let supersample = r.u32()?;
    let size_px = r.u32()?;

    // Dimensions come from the payload, so the atlas footprint is checked
    // rather than computed: a wrapped product would pass the length check
    // below and decode the metrics from the wrong offset.
    let pixel_bytes = checked_product("font atlas", &[atlas_w as usize, atlas_h as usize, 4])?;
    let rgba = r.take(pixel_bytes)?.to_vec();

    let glyph_count = r.u32()? as usize;
    let metrics_bytes = checked_product("font metrics", &[glyph_count, GLYPH_STRIDE])?;
    if r.remaining() < metrics_bytes {
        return Err(format!(
            "font payload truncated: need {} metric bytes, have {}",
            metrics_bytes,
            r.remaining()
        ));
    }
    let mut metrics = Vec::with_capacity(glyph_count);
    for _ in 0..glyph_count {
        metrics.push(GlyphMetrics {
            char_code: r.u32()?,
            atlas_x: r.u16()?,
            atlas_y: r.u16()?,
            atlas_w: r.u16()?,
            atlas_h: r.u16()?,
            advance_px: r.f32()?,
            bearing_x: r.f32()?,
            bearing_y: r.f32()?,
        });
    }
    Ok((atlas_w, atlas_h, supersample, size_px, rgba, metrics))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A payload in the wire format the cook crate emits.
    fn payload(atlas_w: u32, atlas_h: u32, glyphs: &[GlyphMetrics]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&atlas_w.to_le_bytes());
        out.extend_from_slice(&atlas_h.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&32u32.to_le_bytes());
        out.extend(std::iter::repeat_n(
            0xABu8,
            (atlas_w as usize) * (atlas_h as usize) * 4,
        ));
        out.extend_from_slice(&(glyphs.len() as u32).to_le_bytes());
        for m in glyphs {
            out.extend_from_slice(&m.char_code.to_le_bytes());
            out.extend_from_slice(&m.atlas_x.to_le_bytes());
            out.extend_from_slice(&m.atlas_y.to_le_bytes());
            out.extend_from_slice(&m.atlas_w.to_le_bytes());
            out.extend_from_slice(&m.atlas_h.to_le_bytes());
            out.extend_from_slice(&m.advance_px.to_le_bytes());
            out.extend_from_slice(&m.bearing_x.to_le_bytes());
            out.extend_from_slice(&m.bearing_y.to_le_bytes());
        }
        out
    }

    fn glyph(char_code: u32) -> GlyphMetrics {
        GlyphMetrics {
            char_code,
            atlas_x: 1,
            atlas_y: 2,
            atlas_w: 3,
            atlas_h: 4,
            advance_px: 5.5,
            bearing_x: 6.5,
            bearing_y: 7.5,
        }
    }

    // A header field written with the wrong endianness or offset would still
    // decode, so the round trip pins every field to its own value.
    #[test]
    fn decodes_a_well_formed_payload() {
        let glyphs = [glyph(b'A' as u32), glyph(b'B' as u32)];
        let bytes = payload(4, 2, &glyphs);
        let (w, h, ss, size_px, rgba, metrics) = deserialise(&bytes).unwrap();
        assert_eq!((w, h, ss, size_px), (4, 2, 2, 32));
        assert_eq!(rgba.len(), 4 * 2 * 4);
        assert!(rgba.iter().all(|b| *b == 0xAB));
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[0].char_code, b'A' as u32);
        assert_eq!(metrics[1].char_code, b'B' as u32);
        assert_eq!(metrics[1].atlas_x, 1);
        assert_eq!(metrics[1].advance_px, 5.5);
        assert_eq!(metrics[1].bearing_y, 7.5);
    }

    #[test]
    fn decodes_a_payload_with_no_glyphs() {
        let (_, _, _, _, _, metrics) = deserialise(&payload(1, 1, &[])).unwrap();
        assert!(metrics.is_empty());
    }

    #[test]
    fn rejects_a_payload_shorter_than_the_header() {
        for len in 0..16 {
            let bytes = vec![0u8; len];
            assert!(deserialise(&bytes).is_err(), "len {} decoded", len);
        }
    }

    #[test]
    fn rejects_a_payload_truncated_mid_atlas() {
        let mut bytes = payload(8, 8, &[glyph(b'x' as u32)]);
        bytes.truncate(16 + 40);
        assert!(deserialise(&bytes).is_err());
    }

    #[test]
    fn rejects_a_payload_truncated_before_the_glyph_count() {
        let full = payload(2, 2, &[]);
        let bytes = &full[..full.len() - 2];
        assert!(deserialise(bytes).is_err());
    }

    #[test]
    fn rejects_a_payload_truncated_mid_metrics() {
        let mut bytes = payload(2, 2, &[glyph(b'q' as u32), glyph(b'r' as u32)]);
        bytes.truncate(bytes.len() - 5);
        let err = deserialise(&bytes).unwrap_err();
        assert!(err.contains("truncated"), "{}", err);
    }

    // Dimensions large enough that `w * h * 4` wraps: the product must be
    // rejected rather than producing a small length that passes the bounds
    // check and decodes the metrics from inside the atlas.
    #[test]
    fn rejects_atlas_dimensions_that_overflow_the_footprint() {
        let mut bytes = u32::MAX.to_le_bytes().to_vec();
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        let err = deserialise(&bytes).unwrap_err();
        assert!(err.contains("overflow"), "{}", err);
    }

    // A plausible-looking but oversized atlas must not be trusted into an
    // allocation; it exceeds the buffer and has to be reported as truncated.
    #[test]
    fn rejects_an_atlas_larger_than_the_buffer() {
        let mut bytes = 4096u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&4096u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 64]);
        let err = deserialise(&bytes).unwrap_err();
        assert!(err.contains("unexpected end"), "{}", err);
    }

    // A glyph count near u32::MAX must be rejected on the declared size, not
    // by attempting the allocation it asks for.
    #[test]
    fn rejects_an_absurd_glyph_count() {
        let mut bytes = payload(1, 1, &[]);
        let count_at = bytes.len() - 4;
        bytes[count_at..].copy_from_slice(&u32::MAX.to_le_bytes());
        let err = deserialise(&bytes).unwrap_err();
        assert!(err.contains("truncated"), "{}", err);
    }
}

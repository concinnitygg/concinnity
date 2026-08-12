// Blob payload encoding for a compiled font atlas. The decoder lives in
// `concinnity_cpu::build::font`; the two must agree field for field.

use concinnity_cpu::build::font::GlyphMetrics;

pub(crate) fn serialise(
    atlas_w: u32,
    atlas_h: u32,
    supersample: u32,
    size_px: u32,
    rgba: &[u8],
    metrics: &[GlyphMetrics],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&atlas_w.to_le_bytes());
    out.extend_from_slice(&atlas_h.to_le_bytes());
    out.extend_from_slice(&supersample.to_le_bytes());
    // Rasterisation size (px). Intrinsic to the atlas -- the runtime reads it for
    // line-height / cap-height now that Font carries no drained `size_px` field.
    out.extend_from_slice(&size_px.to_le_bytes());
    out.extend_from_slice(rgba);
    out.extend_from_slice(&(metrics.len() as u32).to_le_bytes());
    for m in metrics {
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

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_cpu::build::font::deserialise;

    #[test]
    fn round_trip_serialise() {
        let metrics = vec![
            GlyphMetrics {
                char_code: b'A' as u32,
                atlas_x: 1,
                atlas_y: 1,
                atlas_w: 10,
                atlas_h: 12,
                advance_px: 11.5,
                bearing_x: 0.0,
                bearing_y: 12.0,
            },
            GlyphMetrics {
                char_code: b' ' as u32,
                atlas_x: 15,
                atlas_y: 1,
                atlas_w: 0,
                atlas_h: 0,
                advance_px: 6.0,
                bearing_x: 0.0,
                bearing_y: 0.0,
            },
        ];
        let rgba = vec![128u8; 64 * 64 * 4]; // 64x64 atlas
        let payload = serialise(64, 64, 2, 20, &rgba, &metrics);
        let (w, h, supersample, size_px, out_rgba, out_metrics) = deserialise(&payload).unwrap();
        assert_eq!(w, 64);
        assert_eq!(h, 64);
        assert_eq!(supersample, 2);
        assert_eq!(size_px, 20);
        assert_eq!(out_rgba, rgba);
        assert_eq!(out_metrics.len(), 2);
        assert_eq!(out_metrics[0].char_code, b'A' as u32);
        assert!((out_metrics[0].advance_px - 11.5).abs() < 1e-5);
        assert_eq!(out_metrics[1].char_code, b' ' as u32);
    }
}

// A compiled font payload read back as the face text lays out with.

use alloc::string::String;
use alloc::vec::Vec;

use super::{FontMetrics, LoadedFont, derive_cap_px};

/// A font atlas as the backend uploads it: width, height and RGBA pixels.
pub type FontAtlas = (u32, u32, Vec<u8>);

/// A font payload as the face text lays out with, drawn from atlas slot
/// `slot`, and the atlas that slot uploads.
///
/// # Errors
///
/// Returns an error if the payload does not deserialize (see
/// [`deserialize`](crate::bake::font::deserialize)).
pub fn load_face(slot: usize, bytes: &[u8]) -> Result<(LoadedFont, FontAtlas), String> {
    let (atlas_w, atlas_h, supersample, size_px, rgba, metrics) =
        crate::bake::font::deserialize(bytes)?;
    let metrics: FontMetrics = metrics.into_iter().map(|m| (m.char_code, m)).collect();
    let size_px = size_px as f32;
    let face = LoadedFont {
        atlas_slot: slot,
        cap_px: derive_cap_px(&metrics, size_px),
        metrics,
        atlas_w,
        atlas_h,
        size_px,
        supersample: supersample.max(1) as f32,
    };
    Ok((face, (atlas_w, atlas_h, rgba)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::font::metrics::GlyphMetrics;

    // The face keys its glyphs by code point, takes its slot, and reads the
    // payload's sizes; its atlas is the payload's pixels.
    #[test]
    fn a_payload_loads_as_a_face_and_its_atlas() {
        let glyph = GlyphMetrics {
            char_code: u32::from(b'H'),
            atlas_x: 1,
            atlas_y: 2,
            atlas_w: 3,
            atlas_h: 4,
            advance_px: 5.5,
            bearing_x: 6.5,
            bearing_y: 7.5,
        };
        let bytes = crate::bake::font::serialize(2, 2, 2, 32, &[0xAB; 16], &[glyph]);
        let (face, (w, h, rgba)) = load_face(3, &bytes).expect("loads");
        assert_eq!(face.atlas_slot, 3);
        assert_eq!((face.atlas_w, face.atlas_h, w, h), (2, 2, 2, 2));
        assert_eq!(face.size_px, 32.0);
        assert_eq!(face.supersample, 2.0);
        assert!(face.metrics.contains_key(&u32::from(b'H')));
        assert_eq!(rgba.len(), 16);
        assert!(load_face(0, &bytes[..3]).is_err());
    }
}

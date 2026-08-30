// The glyph atlas of the engine's bundled face, baked into the binary by
// `build.rs` at the native size text naming no Font lays out at.
//
// Two callers need a face with no compiled world data behind it: the startup
// error screen, which runs when loading that data is exactly what failed, and
// the renderer's fallback for any TextLabel or TextInput naming no Font, which
// is the engine's one answer to that whichever way the world was assembled.

use crate::ecs::FontHandle;
use crate::gfx::text::{LoadedFont, derive_cap_px};

const BAKED_ATLAS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/builtin_font.bin"));

// The decoded face: the metrics the text shaper lays glyphs out with, and the
// RGBA atlas the backend uploads into `atlas_slot`.
pub(crate) struct BuiltinFont {
    pub(crate) loaded: LoadedFont,
    pub(crate) atlas: (u32, u32, Vec<u8>),
}

// Decode the baked atlas for a face registered under `handle`, which is also
// its slot in the backend's text atlas pool (as every loaded font handle is).
// `None` means the embedded payload did not parse, which leaves the caller no
// way to draw text with it.
pub(crate) fn load(handle: FontHandle) -> Option<BuiltinFont> {
    let (atlas_w, atlas_h, supersample, size_px, rgba, metrics) =
        match concinnity_core::bake::font::deserialise(BAKED_ATLAS) {
            Ok(decoded) => decoded,
            Err(e) => {
                tracing::error!("built-in font atlas failed to decode: {e}");
                return None;
            }
        };

    let metrics: crate::gfx::text::FontMetrics =
        metrics.into_iter().map(|m| (m.char_code, m)).collect();
    let size_px = size_px as f32;
    Some(BuiltinFont {
        loaded: LoadedFont {
            atlas_slot: handle.0 as usize,
            cap_px: derive_cap_px(&metrics, size_px),
            metrics,
            atlas_w,
            atlas_h,
            size_px,
            supersample: supersample.max(1) as f32,
        },
        atlas: (atlas_w, atlas_h, rgba),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_baked_atlas_decodes_with_printable_ascii() {
        let font = load(FontHandle(0)).expect("the baked atlas decodes");
        let loaded = &font.loaded;

        // The size build.rs bakes at, which is what a font-less label draws at
        // before its own `scale`. Font-less text is documented at this size, so
        // a change here is a change to how every such label renders.
        assert_eq!(loaded.size_px, 24.0);
        assert!(loaded.cap_px > 0.0, "cap height derives from real metrics");
        // Every printable ASCII glyph, so any message the engine writes draws.
        assert_eq!(loaded.metrics.len(), (32u8..=126u8).count());
        for ch in [' ', 'A', 'z', '/', '.', '~'] {
            assert!(
                loaded.metrics.contains_key(&(ch as u32)),
                "{ch:?} is rasterised"
            );
        }

        // The atlas is sized to match its RGBA payload.
        let (w, h, rgba) = &font.atlas;
        assert_eq!(rgba.len(), (*w as usize) * (*h as usize) * 4);
    }

    // The handle the caller asks for is the atlas slot the face reports: the
    // error screen uploads it alone, while a world appends it after its own.
    #[test]
    fn the_face_takes_its_handle_as_its_atlas_slot() {
        assert_eq!(load(FontHandle(0)).expect("decodes").loaded.atlas_slot, 0);
        assert_eq!(load(FontHandle(3)).expect("decodes").loaded.atlas_slot, 3);
    }
}

// The glyph atlas the error screen draws with, baked into the binary by
// `build.rs`. The font a world would normally supply arrives through the blob,
// which is exactly what has failed by the time this screen runs, so the atlas
// has to be embedded rather than loaded.

use crate::ecs::FontHandle;
use crate::gfx::text::{LoadedFont, derive_cap_px};
use std::collections::HashMap;

const BAKED_ATLAS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/error_screen_font.bin"));

// The single atlas slot the embedded font occupies.
pub(super) const HANDLE: FontHandle = FontHandle(0);

// The embedded face, decoded into the two shapes the draw path needs: the
// metrics map `build_text_calls` lays glyphs out with, and the RGBA atlas the
// backend uploads at init.
pub(super) struct EmbeddedFont {
    pub(super) fonts: HashMap<FontHandle, LoadedFont>,
    pub(super) atlases: Vec<(u32, u32, Vec<u8>)>,
}

// Decode the baked atlas. `None` means the embedded payload did not parse,
// which leaves the caller no way to draw text and sends it back to the
// console-only path.
pub(super) fn load() -> Option<EmbeddedFont> {
    let (atlas_w, atlas_h, supersample, size_px, rgba, metrics) =
        match concinnity_cpu::build::font::deserialise(BAKED_ATLAS) {
            Ok(decoded) => decoded,
            Err(e) => {
                tracing::error!("error screen: embedded font atlas failed to decode: {e}");
                return None;
            }
        };

    let metrics: HashMap<u32, _> = metrics.into_iter().map(|m| (m.char_code, m)).collect();
    let size_px = size_px as f32;
    let mut fonts = HashMap::new();
    fonts.insert(
        HANDLE,
        LoadedFont {
            atlas_slot: 0,
            cap_px: derive_cap_px(&metrics, size_px),
            metrics,
            atlas_w,
            atlas_h,
            size_px,
            supersample: supersample.max(1) as f32,
        },
    );

    Some(EmbeddedFont {
        fonts,
        atlases: vec![(atlas_w, atlas_h, rgba)],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_atlas_decodes_with_printable_ascii() {
        let font = load().expect("the baked atlas decodes");
        let loaded = font.fonts.get(&HANDLE).expect("the handle is populated");

        assert!(loaded.size_px > 0.0);
        assert!(loaded.cap_px > 0.0, "cap height derives from real metrics");
        // Every printable ASCII glyph, so any message the engine writes draws.
        assert_eq!(loaded.metrics.len(), (32u8..=126u8).count());
        for ch in [' ', 'A', 'z', '/', '.', '~'] {
            assert!(
                loaded.metrics.contains_key(&(ch as u32)),
                "{ch:?} is rasterised"
            );
        }

        // One atlas, sized to match its RGBA payload.
        assert_eq!(font.atlases.len(), 1);
        let (w, h, rgba) = &font.atlases[0];
        assert_eq!(rgba.len(), (*w as usize) * (*h as usize) * 4);
    }

    // The atlas ships in every binary that links the engine, so a jump in its
    // size is a regression worth failing on rather than discovering in a
    // release build.
    #[test]
    fn embedded_atlas_stays_within_its_size_budget() {
        const BUDGET: usize = 3 * 1024 * 1024;
        assert!(
            BAKED_ATLAS.len() <= BUDGET,
            "baked atlas is {} bytes, over the {BUDGET}-byte budget; lower \
             ERROR_SCREEN_FONT_PX in build.rs",
            BAKED_ATLAS.len()
        );
    }
}

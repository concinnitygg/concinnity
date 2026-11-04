//! A fixed-width synthetic font for the overlay tests: every glyph 10px wide in
//! a 16px em and 12px tall, so measured boxes and built geometry are exact (a
//! one-line unpadded chip measures 10px per char wide, 12px tall, with a -2px
//! top inset).

use crate::bake::font::metrics;
use crate::ecs::FontHandle;
use crate::render::overlay::text;

/// The handle the test font is installed under.
pub(super) const FONT: FontHandle = FontHandle::new(0);

fn glyph(advance_px: f32) -> metrics::GlyphMetrics {
    metrics::GlyphMetrics {
        char_code: 0,
        atlas_x: 0,
        atlas_y: 0,
        atlas_w: 8,
        atlas_h: 12,
        advance_px,
        bearing_x: 0.0,
        bearing_y: 12.0,
    }
}

/// The font set holding the test font under [`FONT`], covering letters,
/// digits, `.` and space.
pub(super) fn loaded_fonts() -> text::FontSet {
    let metrics: text::FontMetrics = ('a'..='z')
        .chain('A'..='Z')
        .chain('0'..='9')
        .chain(['.', ' '])
        .map(|c| (c as u32, glyph(10.0)))
        .collect();
    let cap_px = text::derive_cap_px(&metrics, 16.0);
    let mut fonts = text::FontSet::default();
    fonts.insert(
        FONT,
        text::LoadedFont {
            atlas_slot: 0,
            cap_px,
            metrics,
            atlas_w: 128,
            atlas_h: 128,
            size_px: 16.0,
            supersample: 1.0,
        },
    );
    fonts
}

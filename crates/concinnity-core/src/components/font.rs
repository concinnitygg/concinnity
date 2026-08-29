// Font glyph-atlas schema.

use crate::ecs::PayloadLocator;
use crate::ecs::asset_id::AssetId;
use alloc::string::String;

/// Rasterises a TrueType font into a glyph atlas at build time.
///
/// Reference a Font by name from a [TextLabel](#textlabel). Declaring one is
/// optional: text naming no Font draws with the engine's built-in face at 24px,
/// and compiles no atlas at all. Declare a Font to pick the face, or to pick the
/// size the glyphs are rasterised at.
///
/// An empty `path` rasterises that same built-in face, which is how to get it at
/// a different `size_px`.
///
/// ```rust
/// # use concinnity_core::components::Font;
/// Font {
///     path: "assets/fonts/JetBrainsMono-Regular.ttf".into(),
///     size_px: 20,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Font {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Path to the TTF file, relative to the project root.
    pub path: String,
    /// Rasterisation size in pixels. Determines the rendered glyph height.
    pub size_px: u32,
    /// Filled by inject_locator after the build step packs the payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Default for Font {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            path: String::new(),
            size_px: 20,
            locator: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_font_rasterises_at_a_readable_body_size() {
        let f = Font::default();
        assert!(f.path.is_empty());
        assert_eq!(f.size_px, 20);
        assert_eq!(f.asset_id, AssetId::default());
        assert!(f.locator.is_none());
    }

    #[test]
    fn an_authored_atlas_size_parses_and_round_trips_through_postcard() {
        let f: Font = serde_json::from_str(r#"{"path":"fonts/body.ttf","size_px":48}"#).unwrap();
        assert_eq!(f.path, "fonts/body.ttf");
        assert_eq!(f.size_px, 48);

        let bytes = postcard::to_allocvec(&f).unwrap();
        let back: Font = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.path, "fonts/body.ttf");
        assert_eq!(back.size_px, 48);
        // The glyph atlas rides the blob, so the locator is injected at load.
        assert!(back.locator.is_none());
    }
}

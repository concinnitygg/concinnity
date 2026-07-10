// Font glyph-atlas schema.

use crate::{AssetId, PayloadLocator};
use alloc::string::String;

/// Rasterises a TrueType font into a glyph atlas at build time.
///
/// Reference a Font by name from a [TextLabel](#textlabel).
///
/// ```jsonl
/// {
///   "type": "Font",
///   "name": "fps_font",
///   "args": {
///     "path": "assets/fonts/JetBrainsMono-Regular.ttf",
///     "size_px": 20
///   }
/// }
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

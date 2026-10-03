// Font glyph-atlas schema.

use crate::ecs::PayloadLocator;
use alloc::string::String;

/// Rasterizes a TrueType font into a glyph atlas at build time.
///
/// Reference a Font by name from a [TextLabel](#textlabel). Declaring one is
/// optional: text naming no Font draws with the engine's built-in face at 24px,
/// and compiles no atlas at all. Declare a Font to pick the face, or to pick the
/// size the glyphs are rasterized at.
///
/// An empty `path` rasterizes that same built-in face, which is how to get it at
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
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Font {
    /// Path to the TTF file, relative to the project root.
    pub path: String,
    /// Rasterization size in pixels. Determines the rendered glyph height.
    #[asset(default = 20)]
    pub size_px: u32,
    /// Filled by inject_locator after the build step packs the payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

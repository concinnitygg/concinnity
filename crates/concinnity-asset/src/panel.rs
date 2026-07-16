// UI panel container schema.

use alloc::string::String;

/// A titled background container for grouping UI overlay elements.
///
/// `Panel` is a build-time shorthand: it expands into a filled, optionally
/// rounded background [Sprite](#sprite) and, when `title` is set, a
/// [TextLabel](#textlabel) heading inset from the top-left corner. Place other
/// overlay elements over it to frame a group (a settings card, a dialog body).
///
/// Like the other build-time UI shorthands, generated names are prefixed with
/// this asset's `name` (`<name>_bg`, `<name>_title`), so a panel named with a
/// screen prefix (`pause_card`) puts its children in that [Screen](#screen)
/// (`pause`) via the `<screen>_*` rule and they never clash with hand-authored
/// assets.
///
/// ```jsonl
/// {"name":"pause_card","type":"Panel","args":{"title":"Paused","x":440,"y":220,"width":400,"height":280}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Panel {
    /// Left edge of the panel in window pixels.
    pub x: f32,
    /// Top edge of the panel in window pixels.
    pub y: f32,
    /// Panel width in window pixels.
    pub width: f32,
    /// Panel height in window pixels.
    pub height: f32,
    /// RGBA fill of the background box, each channel in [0, 1].
    pub color: [f32; 4],
    /// Corner rounding radius of the background box, in panel pixels. 0 keeps
    /// sharp corners.
    pub corner_radius: f32,
    /// Heading text drawn at the top-left. Empty draws no heading.
    pub title: String,
    /// [Font](#font) for the title. Empty uses the built-in font.
    pub title_font: String,
    /// Linear-space RGB colour of the title text.
    pub title_color: [f32; 3],
    /// Scale applied to the title text.
    pub title_scale: f32,
    /// Inset of the title from the panel's top-left corner, in pixels.
    pub padding: f32,
}

impl Default for Panel {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 400.0,
            height: 300.0,
            color: [0.08, 0.09, 0.12, 0.96],
            corner_radius: 8.0,
            title: String::new(),
            title_font: String::new(),
            title_color: [0.95, 0.95, 0.97],
            title_scale: 1.0,
            padding: 16.0,
        }
    }
}

// Screen-space hit-region schema.

use crate::{AssetId, SpriteFit, de_opt_asset_ref};
use alloc::string::String;

/// A responsive invisible rectangular region in screen space.
///
/// When clicked, fires an `action`. When hovered, it optionally restyles a
/// referenced [TextLabel](#textlabel) (colour and/or scale).
///
/// The cursor must be free (not captured for camera control) for events to fire.
///
/// ```jsonl
/// {
///   "name": "btn_start",
///   "type": "HitRegion",
///   "args": {
///     "x": 430, "y": 330, "width": 220, "height": 40,
///     "label": "scene_menu_start",
///     "hover_color": [1.0, 0.85, 0.3],
///     "hover_scale": 1.08,
///     "action": "scene:scene_game"
///   }
/// }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct HitRegion {
    /// Left edge of the region in window pixels.
    pub x: f32,
    /// Top edge of the region in window pixels.
    pub y: f32,
    /// Width of the region in window pixels.
    pub width: f32,
    /// Height of the region in window pixels.
    pub height: f32,
    /// A [TextLabel](#textlabel) to style on hover. `None` = no label effect.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub label: Option<AssetId>,
    /// RGB colour applied to the label while hovered. `None` = no change.
    pub hover_color: Option<[f32; 3]>,
    /// Scale applied to the label while hovered. None = no change.
    pub hover_scale: Option<f32>,
    /// Action to fire on click. Recognised forms:
    /// `"scene:<name>"`, `"quit"`, `"screen:show:<name>"`, `"screen:hide"`,
    /// `"screen:toggle:<name>"`.
    pub action: String,
    /// The [Sprite](#sprite) a [Slider](#slider) drag region moves along its
    /// track. `None` for ordinary regions. Set automatically when a `Slider`
    /// expands; you don't set this directly.
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub drag_handle: Option<AssetId>,
    /// [Screen](#screen) this region belongs to. Resolved automatically from the
    /// naming convention (a region named `<screen>_*` belongs to screen
    /// `<screen>`); you don't set this directly. While a screen is active,
    /// only the top capturing screen's regions fire; with no screen active,
    /// only screen-less regions fire.
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub screen: Option<AssetId>,
    /// Whether this region is inert. A disabled region never hovers or fires.
    /// Set by the engine at runtime (e.g. a settings row whose feature the GPU
    /// cannot provide is disabled and grayed out); you don't set this directly.
    #[serde(default)]
    pub disabled: bool,
    /// When set, this region tracks its referenced [`label`](#hitregion): it
    /// follows the label's vertical position (so a menu the engine lays out at
    /// runtime keeps its buttons clickable) and is inert while the label's text
    /// is empty (so a hidden menu entry does not catch clicks). Requires
    /// `label`.
    #[serde(default)]
    pub follow_label: bool,
    /// How a screen-owned region maps from the reference canvas to the window
    /// when their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    /// `Bottom` keeps a region aligned with bottom-anchored furniture it
    /// covers. A region spanning the whole reference canvas always covers the
    /// full window regardless of `fit`.
    #[serde(default)]
    pub fit: SpriteFit,
}

impl Default for HitRegion {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 40.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: String::new(),
            drag_handle: None,
            screen: None,
            disabled: false,
            follow_label: false,
            fit: SpriteFit::Fit,
        }
    }
}

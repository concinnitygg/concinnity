// Screen-space hit-region schema.

use crate::components::SpriteFit;
use crate::ecs::asset_id::AssetId;
use crate::ecs::asset_id::de_opt_asset_ref;
use alloc::string::String;

/// A responsive invisible rectangular region in screen space.
///
/// When clicked, fires an `action`. When hovered, it optionally restyles a
/// referenced [TextLabel](#textlabel) (colour and/or scale).
///
/// The cursor must be free (not captured for camera control) for events to fire.
///
/// ```rust
/// # use concinnity_core::components::HitRegion;
/// HitRegion {
///     x: 430.0,
///     y: 330.0,
///     width: 220.0,
///     height: 40.0,
///     hover_color: Some([1.0, 0.85, 0.3]),
///     hover_scale: Some(1.08),
///     ..Default::default()
/// };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_region_is_an_enabled_button_sized_rectangle() {
        let h = HitRegion::default();
        assert_eq!((h.x, h.y), (0.0, 0.0));
        assert_eq!((h.width, h.height), (100.0, 40.0));
        assert!(!h.disabled);
        assert!(!h.follow_label);
        assert_eq!(h.fit, SpriteFit::Fit);
        assert!(h.action.is_empty());
        // Hover styling is opt-in: unset means "do not restyle on hover".
        assert_eq!(h.hover_color, None);
        assert_eq!(h.hover_scale, None);
        assert!(h.label.is_none());
        assert!(h.drag_handle.is_none());
        assert!(h.screen.is_none());
    }

    #[test]
    fn an_authored_region_parses_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let h: HitRegion = serde_json::from_str(
            r#"{"x":10,"y":20,"width":200,"height":48,"label":"play_label","action":"start",
                "hover_color":[1,0.85,0.3],"hover_scale":1.1,"drag_handle":"grip",
                "screen":"menu","disabled":true,"follow_label":true,"fit":"cover"}"#,
        )
        .unwrap();
        assert_eq!(h.label, Some(AssetId(10)));
        assert_eq!(h.drag_handle, Some(AssetId(4)));
        assert_eq!(h.screen, Some(AssetId(4)));
        assert_eq!(h.action, "start");
        assert_eq!(h.hover_scale, Some(1.1));
        assert_eq!(h.fit, SpriteFit::Cover);
        assert!(h.disabled);
        assert!(h.follow_label);

        let bytes = postcard::to_allocvec(&h).unwrap();
        let back: HitRegion = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.hover_color, Some([1.0, 0.85, 0.3]));
        assert_eq!((back.width, back.height), (200.0, 48.0));
        assert_eq!(back.label, Some(AssetId(10)));
        assert_eq!(back.fit, SpriteFit::Cover);
    }
}

// Screen-space sprite overlay schema.

use crate::components::Screen;
use crate::components::Vocabulary;
use crate::ecs::Ref;
use crate::ecs::TextureHandle;

/// Screen-space 2D rectangle drawn as a UI overlay each frame.
///
/// Sprites are pixel-anchored quads with an RGBA tint. They draw alongside
/// [TextLabel](#textlabel)s, ordered behind labels so text sits on top.
///
/// A sprite with a `texture` draws that image, multiplied by the tint (use a
/// white tint to show the image unchanged; the tint's alpha fades it).
/// Without one, the tint is drawn as a solid-colored rectangle.
///
/// ```rust
/// # use concinnity_core::components::Sprite;
/// Sprite {
///     x: 0.0,
///     y: 0.0,
///     width: 1280.0,
///     height: 720.0,
///     tint: [0.04, 0.06, 0.1, 1.0],
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
pub struct Sprite {
    /// Left edge in screen pixels from the window's top-left.
    pub x: f32,
    /// Top edge in screen pixels from the window's top-left.
    pub y: f32,
    /// Width in screen pixels.
    #[asset(default = 100.0)]
    pub width: f32,
    /// Height in screen pixels.
    #[asset(default = 100.0)]
    pub height: f32,
    /// [Texture](#texture) to draw, sampled over the sprite's rect and
    /// multiplied by `tint`. Omitted, the sprite is a solid `tint` fill.
    pub texture: Option<TextureHandle>,
    /// RGBA color the rectangle is filled with, each channel in [0, 1].
    #[asset(default = [1.0, 1.0, 1.0, 1.0])]
    pub tint: [f32; 4],
    /// When true, the sprite acts as an in-engine cursor: it is drawn on top of
    /// the other overlays as an arrow pointer tracking the mouse, with the
    /// pointer at the arrow's tip. `tint` is the arrow fill (a contrasting
    /// outline is added automatically) and `height` its size; `width` is
    /// ignored so the arrow keeps its shape. The system cursor is hidden while
    /// a visible `follow_cursor` sprite exists.
    pub follow_cursor: bool,
    /// When false the sprite is skipped each frame.
    #[asset(default = true)]
    pub visible: bool,
    /// [Screen](#screen) this sprite belongs to. `None` means the sprite is
    /// always visible (e.g. a scene background).
    #[serde(default)]
    pub screen: Option<Ref<Screen>>,
    /// How a screen-owned sprite maps from the reference canvas to the window
    /// when their aspect ratios differ.
    pub fit: SpriteFit,
    /// Corner rounding radius in the sprite's own pixel space. `0` keeps
    /// sharp corners; larger values round each corner with a quarter-circle
    /// arc (clamped to half the sprite's shorter side). The rounded edge is
    /// softly anti-aliased.
    pub corner_radius: f32,
    /// Border stroke width in the sprite's own pixel space, drawn just inside
    /// the sprite's outline and following its rounded corners. `0` draws no
    /// border; larger values paint a ring of that width in `border_color`
    /// (clamped to half the sprite's shorter side). An opaque fill is inset
    /// under the ring; a translucent fill keeps the whole rect, so the ring
    /// sits over its edge and whatever is behind still shows through.
    pub border_width: f32,
    /// RGBA color of the border stroke, each channel in [0, 1]. Ignored when
    /// `border_width` is `0`.
    #[asset(default = [0.0, 0.0, 0.0, 1.0])]
    pub border_color: [f32; 4],
}

/// How a screen-owned overlay element (a [Sprite](#sprite), [TextLabel](#textlabel),
/// or [HitRegion](#hitregion)) maps from the 1280x720 reference canvas to the
/// live window when their aspect ratios differ.
///
/// Screen-owned UI is authored against a fixed reference canvas and uniformly
/// scaled to the window at runtime.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default, Vocabulary,
)]
#[serde(rename_all = "lowercase")]
pub enum SpriteFit {
    /// The canvas fits inside the window, centered, leaving margins on the
    /// shorter axis. UI elements keep their proportions and stay fully
    /// visible.
    #[default]
    #[vocab("fit")]
    Fit,
    /// The canvas fills the window, centered, cropping the overflowing axis
    /// equally on both sides. Full-bleed stage imagery (scene backdrops,
    /// character portraits) reaches the window edges without distorting, and
    /// content anchored to a canvas edge stays flush with the window edge.
    #[vocab("cover")]
    Cover,
    /// The canvas keeps the `fit` scale (no cropping), but the whole overlay is
    /// shifted so the reference bottom edge lands on the window bottom edge.
    /// Bottom-anchored furniture (a visual-novel dialog box and its controls)
    /// hugs the window bottom at any aspect ratio instead of floating above a
    /// letterbox margin.
    #[vocab("bottom")]
    Bottom,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_names_parse_in_lowercase() {
        let f = |s: &str| serde_json::from_str::<SpriteFit>(s).unwrap();
        assert_eq!(f(r#""fit""#), SpriteFit::Fit);
        assert_eq!(f(r#""cover""#), SpriteFit::Cover);
        assert_eq!(f(r#""bottom""#), SpriteFit::Bottom);
        assert_eq!(
            serde_json::to_string(&SpriteFit::Bottom).unwrap(),
            r#""bottom""#
        );
    }
}

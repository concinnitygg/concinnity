// Screen-space sprite overlay schema.

use crate::ecs::TextureHandle;
use crate::ecs::asset_id::AssetId;
use crate::ecs::asset_id::de_opt_asset_ref;
use crate::ecs::de_opt_texture_handle;

/// Screen-space 2D rectangle drawn as a UI overlay each frame.
///
/// Sprites are pixel-anchored quads with an RGBA tint. They draw alongside
/// [TextLabel](#textlabel)s, ordered behind labels so text sits on top.
///
/// A sprite with a `texture` draws that image, multiplied by the tint (use a
/// white tint to show the image unchanged; the tint's alpha fades it).
/// Without one, the tint is drawn as a solid-coloured rectangle.
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Sprite {
    /// Assigned by the loader; not authored.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Left edge in screen pixels from the window's top-left.
    pub x: f32,
    /// Top edge in screen pixels from the window's top-left.
    pub y: f32,
    /// Width in screen pixels.
    pub width: f32,
    /// Height in screen pixels.
    pub height: f32,
    /// [Texture](#texture) to draw, sampled over the sprite's rect and
    /// multiplied by `tint`. Omitted, the sprite is a solid `tint` fill.
    #[serde(deserialize_with = "de_opt_texture_handle")]
    pub texture: Option<TextureHandle>,
    /// RGBA colour the rectangle is filled with, each channel in [0, 1].
    pub tint: [f32; 4],
    /// When true, the sprite acts as an in-engine cursor: it is drawn on top of
    /// the other overlays as an arrow pointer tracking the mouse, with the
    /// pointer at the arrow's tip. `tint` is the arrow fill (a contrasting
    /// outline is added automatically) and `height` its size; `width` is
    /// ignored so the arrow keeps its shape. The system cursor is hidden while
    /// a visible `follow_cursor` sprite exists.
    pub follow_cursor: bool,
    /// When false the sprite is skipped each frame.
    pub visible: bool,
    /// [Screen](#screen) this sprite belongs to. Resolved automatically from
    /// the naming convention (`<screen>_*`); you don't set this directly.
    /// `None` means the sprite is always visible (e.g. a scene background).
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub screen: Option<AssetId>,
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
    /// RGBA colour of the border stroke, each channel in [0, 1]. Ignored when
    /// `border_width` is `0`.
    pub border_color: [f32; 4],
}

/// How a screen-owned overlay element (a [Sprite](#sprite), [TextLabel](#textlabel),
/// or [HitRegion](#hitregion)) maps from the 1280x720 reference canvas to the
/// live window when their aspect ratios differ.
///
/// Screen-owned UI is authored against a fixed reference canvas and uniformly
/// scaled to the window at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SpriteFit {
    /// The canvas fits inside the window, centered, leaving margins on the
    /// shorter axis. UI elements keep their proportions and stay fully
    /// visible.
    #[default]
    Fit,
    /// The canvas fills the window, centered, cropping the overflowing axis
    /// equally on both sides. Full-bleed stage imagery (scene backdrops,
    /// character portraits) reaches the window edges without distorting, and
    /// content anchored to a canvas edge stays flush with the window edge.
    Cover,
    /// The canvas keeps the `fit` scale (no cropping), but the whole overlay is
    /// shifted so the reference bottom edge lands on the window bottom edge.
    /// Bottom-anchored furniture (a visual-novel dialog box and its controls)
    /// hugs the window bottom at any aspect ratio instead of floating above a
    /// letterbox margin.
    Bottom,
}

impl Default for Sprite {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            texture: None,
            tint: [1.0, 1.0, 1.0, 1.0],
            follow_cursor: false,
            visible: true,
            screen: None,
            fit: SpriteFit::Fit,
            corner_radius: 0.0,
            border_width: 0.0,
            border_color: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_sprite_is_a_visible_untinted_square_with_no_border() {
        let s = Sprite::default();
        assert_eq!((s.width, s.height), (100.0, 100.0));
        assert_eq!(s.tint, [1.0, 1.0, 1.0, 1.0]);
        assert!(s.visible);
        assert!(!s.follow_cursor);
        assert_eq!(s.fit, SpriteFit::Fit);
        assert_eq!(s.corner_radius, 0.0);
        // Zero width is what suppresses the border, so its colour is irrelevant.
        assert_eq!(s.border_width, 0.0);
        assert!(s.texture.is_none());
        assert!(s.screen.is_none());
        assert_eq!(SpriteFit::default(), SpriteFit::Fit);
    }

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

    #[test]
    fn a_cursor_sprite_parses_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let s: Sprite = serde_json::from_str(
            r#"{"x":10,"y":20,"width":32,"height":32,"texture":"tex_cursor",
                "tint":[1,1,1,0.8],"follow_cursor":true,"visible":false,"screen":"menu",
                "fit":"cover","corner_radius":4,"border_width":2,"border_color":[1,0,0,1]}"#,
        )
        .unwrap();
        assert_eq!(s.texture, Some(TextureHandle(10)));
        assert_eq!(s.screen, Some(AssetId(4)));
        assert!(s.follow_cursor);
        assert!(!s.visible);

        let bytes = postcard::to_allocvec(&s).unwrap();
        let back: Sprite = postcard::from_bytes(&bytes).unwrap();
        assert_eq!((back.x, back.y), (10.0, 20.0));
        assert_eq!(back.tint, [1.0, 1.0, 1.0, 0.8]);
        assert_eq!(back.fit, SpriteFit::Cover);
        assert_eq!(back.corner_radius, 4.0);
        assert_eq!(back.border_width, 2.0);
        assert_eq!(back.border_color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(back.asset_id, AssetId::default());
    }
}

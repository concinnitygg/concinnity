// Screen-space UI text-label schema.

use crate::{AssetId, FontHandle, SpriteFit, de_opt_asset_ref, de_opt_font_handle};
use alloc::string::String;

/// Horizontal alignment of a [TextLabel](#textlabel) relative to its `x`.
///
/// `Center` and `Right` measure the rendered text with the real font metrics
/// each frame, so a label stays visually centered (or right-aligned) at any
/// scale without the author estimating glyph widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TextAlign {
    /// `x` is the left edge of the text (the default).
    #[default]
    Left,
    /// `x` is the horizontal center of the text.
    Center,
    /// `x` is the right edge of the text.
    Right,
}

/// Screen-space text drawn as a UI overlay on top of the 3D scene each frame.
///
/// Text is laid out using the referenced [Font](#font). The `content` field can
/// be updated every frame (e.g. by an [FpsCounter](#fpscounter)).
///
/// A `\n` in `content` starts a new line. When `background` has an alpha > 0, a
/// box is filled behind the glyphs, extended outward by `padding` pixels,
/// useful for HUD chips.
///
/// ```rust
/// # use concinnity_asset::TextLabel;
/// TextLabel {
///     content: "FPS: --".into(),
///     x: 10.0,
///     y: 10.0,
///     color: [1.0, 1.0, 1.0],
///     scale: 1.0,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TextLabel {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The [Font](#font) asset to use for rendering.
    #[serde(deserialize_with = "de_opt_font_handle")]
    pub font: Option<FontHandle>,
    /// Text to display. Can be updated each frame.
    pub content: String,
    /// Horizontal position in pixels from the left edge of the window.
    pub x: f32,
    /// Vertical position in pixels from the top edge of the window.
    pub y: f32,
    /// Linear-space RGB text colour.
    pub color: [f32; 3],
    /// Uniform scale applied on top of the font's `size_px`. 1.0 = native size.
    pub scale: f32,
    /// When true, center the label in the viewport each frame; x and y are ignored.
    pub centered: bool,
    /// Horizontal alignment relative to `x` (measured with the real font
    /// metrics). Ignored when `centered` is set.
    pub align: TextAlign,
    /// How a screen-owned label maps from the reference canvas to the window when
    /// their aspect ratios differ (matches [Sprite](#sprite)'s `fit`). `Bottom`
    /// keeps a label flush with a bottom-anchored sprite it labels.
    pub fit: SpriteFit,
    /// RGBA fill of a box drawn behind the text. An alpha of 0 (the default)
    /// draws no box; any alpha > 0 draws the box at that opacity.
    pub background: [f32; 4],
    /// Pixels the background box extends past the text on every side. Only
    /// meaningful when `background` is visible.
    pub padding: f32,
    /// Width in the label's own pixels that text wraps within. `0` (the
    /// default) never wraps, so the text runs as far as it needs to. Any
    /// greater value breaks the content into lines at word boundaries, using
    /// the real font metrics, splitting a word only when it cannot fit a line
    /// on its own. Authored newlines are kept as breaks either way. Ignored
    /// when `centered` is set, since a centered label is sized to the viewport
    /// rather than to a container.
    pub wrap_width: f32,
    /// Most lines the label draws. `0` (the default) draws every line. When the
    /// text needs more than this, the last drawn line ends in an ellipsis, so
    /// text bounded by `wrap_width` is bounded in both directions and can never
    /// spill out of the box that holds it.
    pub max_lines: u32,
    /// When false, the label is hidden.
    pub visible: bool,
    /// [Screen](#screen) this label belongs to. Resolved automatically from
    /// the naming convention (`<screen>_*`); you don't set this directly.
    /// `None` means the label is always visible.
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub screen: Option<AssetId>,
}

impl Default for TextLabel {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            font: None,
            content: String::new(),
            x: 10.0,
            y: 10.0,
            color: [1.0, 1.0, 1.0],
            scale: 1.0,
            centered: false,
            align: TextAlign::Left,
            fit: SpriteFit::Fit,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            wrap_width: 0.0,
            max_lines: 0,
            visible: true,
            screen: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_label_draws_white_left_aligned_text_with_no_background() {
        let l = TextLabel::default();
        assert!(l.content.is_empty());
        assert_eq!((l.x, l.y), (10.0, 10.0));
        assert_eq!(l.color, [1.0, 1.0, 1.0]);
        assert_eq!(l.scale, 1.0);
        assert!(!l.centered);
        assert_eq!(l.align, TextAlign::Left);
        assert_eq!(l.fit, SpriteFit::Fit);
        // A fully transparent background is what suppresses the chip box.
        assert_eq!(l.background, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(l.padding, 0.0);
        // Zero means unbounded: no wrapping and no line cap.
        assert_eq!(l.wrap_width, 0.0);
        assert_eq!(l.max_lines, 0);
        assert!(l.visible);
        assert!(l.font.is_none());
        assert!(l.screen.is_none());
        assert_eq!(TextAlign::default(), TextAlign::Left);
    }

    #[test]
    fn alignment_names_parse_in_lowercase() {
        let a = |s: &str| serde_json::from_str::<TextAlign>(s).unwrap();
        assert_eq!(a(r#""left""#), TextAlign::Left);
        assert_eq!(a(r#""center""#), TextAlign::Center);
        assert_eq!(a(r#""right""#), TextAlign::Right);
        assert_eq!(
            serde_json::to_string(&TextAlign::Center).unwrap(),
            r#""center""#
        );
    }

    #[test]
    fn a_wrapped_chip_parses_and_round_trips_through_postcard() {
        crate::test_support::install_resolvers();
        let l: TextLabel = serde_json::from_str(
            r#"{"font":"body","content":"Hello there","x":20,"y":40,"color":[1,0.9,0.5],
                "scale":1.25,"centered":true,"align":"right","fit":"cover",
                "background":[0,0,0,0.6],"padding":6,"wrap_width":320,"max_lines":3,
                "visible":false,"screen":"menu"}"#,
        )
        .unwrap();
        assert_eq!(l.font, Some(FontHandle(4)));
        assert_eq!(l.screen, Some(AssetId(4)));
        assert_eq!(l.align, TextAlign::Right);
        assert!(l.centered);
        assert!(!l.visible);

        let bytes = postcard::to_allocvec(&l).unwrap();
        let back: TextLabel = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.content, "Hello there");
        assert_eq!(back.color, [1.0, 0.9, 0.5]);
        assert_eq!(back.scale, 1.25);
        assert_eq!(back.fit, SpriteFit::Cover);
        assert_eq!(back.background, [0.0, 0.0, 0.0, 0.6]);
        assert_eq!(back.padding, 6.0);
        assert_eq!(back.wrap_width, 320.0);
        assert_eq!(back.max_lines, 3);
        assert_eq!(back.asset_id, AssetId::default());
    }
}

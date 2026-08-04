// Editable single-line text-field schema.

use crate::{AssetId, FontHandle, SpriteFit, de_opt_asset_ref, de_opt_font_handle};
use alloc::string::String;

/// An editable single-line text field drawn as a UI overlay.
///
/// A filled rounded box showing the typed `content` (or a dimmer `placeholder`
/// while empty), plus a caret when the field holds keyboard focus. The engine
/// gives focus to the field the cursor clicks, appends the characters typed that
/// frame, and moves or edits at the caret with the arrow / Home / End /
/// Backspace / Delete keys. Read `content` back to use what the player typed;
/// set it to pre-fill the field.
///
/// Like other overlay elements it belongs to a [Screen](#screen) resolved from the
/// naming convention (`<screen>_*`), or is always shown when it has none.
///
/// ```jsonl
/// {
///   "type": "TextInput",
///   "name": "menu_playername",
///   "args": {
///     "font": "ui_font",
///     "placeholder": "Enter your name",
///     "x": 400, "y": 300, "width": 480, "height": 48,
///     "max_len": 24
///   }
/// }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TextInput {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The [Font](#font) used to render the field's text.
    #[serde(deserialize_with = "de_opt_font_handle")]
    pub font: Option<FontHandle>,
    /// The current text. Edited in place as the player types; set an initial
    /// value here to pre-fill the field.
    pub content: String,
    /// Dimmer prompt shown while `content` is empty and the field is unfocused.
    pub placeholder: String,
    /// Left edge in screen pixels from the window's top-left.
    pub x: f32,
    /// Top edge in screen pixels from the window's top-left.
    pub y: f32,
    /// Field width in screen pixels.
    pub width: f32,
    /// Field height in screen pixels.
    pub height: f32,
    /// Uniform scale applied on top of the font's `size_px`. 1.0 = native size.
    pub scale: f32,
    /// Linear-space RGB colour of the typed text.
    pub text_color: [f32; 3],
    /// Linear-space RGB colour of the placeholder prompt.
    pub placeholder_color: [f32; 3],
    /// RGBA fill of the field's background box, each channel in [0, 1].
    pub background: [f32; 4],
    /// Linear-space RGB colour of the caret bar.
    pub caret_color: [f32; 3],
    /// Corner rounding radius of the background box, in field pixels.
    pub corner_radius: f32,
    /// Inner horizontal inset from the box edge to the text, in pixels.
    pub padding: f32,
    /// Maximum number of characters accepted. 0 means no limit.
    pub max_len: u32,
    /// When false the field is skipped each frame and cannot take focus.
    pub visible: bool,
    /// How a screen-owned field maps from the reference canvas to the window when
    /// their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    pub fit: SpriteFit,
    /// [Screen](#screen) this field belongs to. Resolved automatically from
    /// the naming convention (`<screen>_*`); you don't set this directly.
    /// `None` means the field is always visible.
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub screen: Option<AssetId>,
    /// Runtime keyboard-focus flag, set by the engine while this is the active
    /// field. Not authored and not serialized to a blob.
    #[serde(skip)]
    pub focused: bool,
    /// Runtime inline-completion suffix, drawn in the placeholder colour after
    /// the typed content while the field holds focus. Set by whoever drives the
    /// field (e.g. an autocomplete); never edited by typing. Not authored and
    /// not serialized to a blob.
    #[serde(skip)]
    pub ghost: String,
    /// Runtime caret position as a character index into `content`. Not authored
    /// and not serialized to a blob.
    #[serde(skip)]
    pub caret: usize,
}

impl Default for TextInput {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            font: None,
            content: String::new(),
            placeholder: String::new(),
            x: 0.0,
            y: 0.0,
            width: 240.0,
            height: 40.0,
            scale: 1.0,
            text_color: [0.95, 0.95, 0.97],
            placeholder_color: [0.55, 0.55, 0.60],
            background: [0.10, 0.10, 0.13, 1.0],
            caret_color: [0.95, 0.95, 0.97],
            corner_radius: 4.0,
            padding: 8.0,
            max_len: 0,
            visible: true,
            fit: SpriteFit::Fit,
            screen: None,
            focused: false,
            ghost: String::new(),
            caret: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_field_is_visible_empty_and_unfocused() {
        let t = TextInput::default();
        assert!(t.content.is_empty());
        assert!(t.placeholder.is_empty());
        assert_eq!((t.width, t.height), (240.0, 40.0));
        assert_eq!(t.scale, 1.0);
        assert_eq!(t.corner_radius, 4.0);
        assert_eq!(t.padding, 8.0);
        // Zero means "no length limit", not "accepts nothing".
        assert_eq!(t.max_len, 0);
        assert!(t.visible);
        assert_eq!(t.fit, SpriteFit::Fit);
        // Edit state is runtime-only.
        assert!(!t.focused);
        assert!(t.ghost.is_empty());
        assert_eq!(t.caret, 0);
        assert!(t.font.is_none());
        assert!(t.screen.is_none());
    }

    #[test]
    fn edit_state_is_runtime_only_and_never_rides_the_wire() {
        crate::test_support::install_resolvers();
        let t: TextInput = serde_json::from_str(
            r#"{"font":"body","content":"hello","placeholder":"name","max_len":32,
                "screen":"menu","focused":true,"caret":5,"ghost":"world"}"#,
        )
        .unwrap();
        assert_eq!(t.font, Some(FontHandle(4)));
        assert_eq!(t.content, "hello");
        assert_eq!(t.screen, Some(AssetId(4)));
        // Focus, caret, and the completion ghost are skipped on the way in.
        assert!(!t.focused);
        assert_eq!(t.caret, 0);
        assert!(t.ghost.is_empty());

        let bytes = postcard::to_allocvec(&t).unwrap();
        let back: TextInput = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.content, "hello");
        assert_eq!(back.placeholder, "name");
        assert_eq!(back.max_len, 32);
        assert_eq!(back.font, Some(FontHandle(4)));
        assert!(!back.focused);
        assert_eq!(back.asset_id, AssetId::default());
    }

    #[test]
    fn an_authored_style_parses_and_round_trips_through_postcard() {
        let t: TextInput = serde_json::from_str(
            r#"{"x":12,"y":24,"width":300,"height":36,"scale":1.5,"text_color":[1,1,1],
                "placeholder_color":[0.4,0.4,0.4],"background":[0,0,0,1],
                "caret_color":[1,0,0],"corner_radius":0,"padding":4,"visible":false,
                "fit":"bottom"}"#,
        )
        .unwrap();
        let bytes = postcard::to_allocvec(&t).unwrap();
        let back: TextInput = postcard::from_bytes(&bytes).unwrap();
        assert_eq!((back.x, back.y), (12.0, 24.0));
        assert_eq!(back.scale, 1.5);
        assert_eq!(back.placeholder_color, [0.4, 0.4, 0.4]);
        assert_eq!(back.background, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(back.caret_color, [1.0, 0.0, 0.0]);
        assert_eq!(back.padding, 4.0);
        assert!(!back.visible);
        assert_eq!(back.fit, SpriteFit::Bottom);
    }
}

// Editable single-line text-field schema.

use crate::components::Screen;
use crate::components::SpriteFit;
use crate::ecs::FontHandle;
use crate::ecs::Ref;
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
/// Like other overlay elements it belongs to the [Screen](#screen) its
/// `screen` names, or is always shown when it names none.
///
/// ```rust
/// # use concinnity_core::components::TextInput;
/// TextInput {
///     placeholder: "Enter your name".into(),
///     x: 400.0,
///     y: 300.0,
///     width: 480.0,
///     height: 48.0,
///     max_len: 24,
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
pub struct TextInput {
    /// The [Font](#font) used to render the field's text. Unset draws with the
    /// engine's built-in face at its native 24px.
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
    #[asset(default = 240.0)]
    pub width: f32,
    /// Field height in screen pixels.
    #[asset(default = 40.0)]
    pub height: f32,
    /// Uniform scale applied on top of the font's `size_px` (24 for the
    /// built-in face). 1.0 = native size.
    #[asset(default = 1.0)]
    pub scale: f32,
    /// Linear-space RGB color of the typed text.
    #[asset(default = [0.95, 0.95, 0.97])]
    pub text_color: [f32; 3],
    /// Linear-space RGB color of the placeholder prompt.
    #[asset(default = [0.55, 0.55, 0.60])]
    pub placeholder_color: [f32; 3],
    /// RGBA fill of the field's background box, each channel in [0, 1].
    #[asset(default = [0.10, 0.10, 0.13, 1.0])]
    pub background: [f32; 4],
    /// Linear-space RGB color of the caret bar.
    #[asset(default = [0.95, 0.95, 0.97])]
    pub caret_color: [f32; 3],
    /// Corner rounding radius of the background box, in field pixels.
    #[asset(default = 4.0)]
    pub corner_radius: f32,
    /// Inner horizontal inset from the box edge to the text, in pixels.
    #[asset(default = 8.0)]
    pub padding: f32,
    /// Maximum number of characters accepted. 0 means no limit.
    pub max_len: u32,
    /// When false the field is skipped each frame and cannot take focus.
    #[asset(default = true)]
    pub visible: bool,
    /// How a screen-owned field maps from the reference canvas to the window when
    /// their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    pub fit: SpriteFit,
    /// [Screen](#screen) this field belongs to. `None` means the field is
    /// always visible.
    #[serde(default)]
    pub screen: Option<Ref<Screen>>,
    /// Runtime keyboard-focus flag, set by the engine while this is the active
    /// field. Not authored and not serialized to a blob.
    #[serde(skip)]
    pub focused: bool,
    /// Runtime inline-completion suffix, drawn in the placeholder color after
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_state_is_runtime_only_and_never_rides_the_wire() {
        let mut t: TextInput =
            serde_json::from_str(r#"{"content":"hello","focused":true,"caret":5,"ghost":"world"}"#)
                .unwrap();
        // Focus, caret, and the completion ghost are skipped on the way in.
        assert!(!t.focused);
        assert_eq!(t.caret, 0);
        assert!(t.ghost.is_empty());

        // And on the way out.
        t.focused = true;
        t.caret = 3;
        t.ghost = String::from("world");
        let back: TextInput = postcard::from_bytes(&postcard::to_allocvec(&t).unwrap()).unwrap();
        assert_eq!(back.content, "hello");
        assert!(!back.focused);
        assert_eq!(back.caret, 0);
        assert!(back.ghost.is_empty());
    }
}

// src/assets/text_input.rs

use crate::assets::SpriteFit;
use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, CompanionSpec, Component};

/// An editable single-line text field drawn as a UI overlay.
///
/// A filled rounded box showing the typed `content` (or a dimmer `placeholder`
/// while empty), plus a caret when the field holds keyboard focus. The engine
/// gives focus to the field the cursor clicks, appends the characters typed that
/// frame, and moves or edits at the caret with the arrow / Home / End /
/// Backspace / Delete keys. Read `content` back to use what the player typed;
/// set it to pre-fill the field.
///
/// Like other overlay elements it belongs to a [View](#view) resolved from the
/// naming convention (`<view>_*`), or is always shown when it has none.
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
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub font: Option<AssetId>,
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
    /// How a view-owned field maps from the reference canvas to the window when
    /// their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    pub fit: SpriteFit,
    /// [View](#view) this field belongs to. Resolved automatically from the
    /// naming convention (`<view>_*`); you don't set this directly. `None` means
    /// the field is always visible.
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub view: Option<AssetId>,
    /// Runtime keyboard-focus flag, set by the engine while this is the active
    /// field. Not authored and not serialized to a blob.
    #[serde(skip)]
    pub focused: bool,
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
            view: None,
            focused: false,
            caret: 0,
        }
    }
}

impl Component for TextInput {
    const NAME: &'static str = "TextInput";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn ref_fields() -> &'static [(&'static str, &'static str)] {
        &[("font", "Font"), ("view", "View")]
    }

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }

    fn companions(_args: &serde_json::Value, _world: &[serde_json::Value]) -> Vec<CompanionSpec> {
        vec![CompanionSpec {
            name: "GraphicsConfig",
            asset_type: "GraphicsConfig",
            args: serde_json::json!({}),
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_with_defaults() {
        let t: TextInput = serde_json::from_str("{}").unwrap();
        assert_eq!(t.content, "");
        assert_eq!(t.width, 240.0);
        assert_eq!(t.max_len, 0);
        assert!(t.visible);
        assert!(!t.focused);
        assert_eq!(t.caret, 0);
    }

    #[test]
    fn deserializes_with_fields() {
        let json = r#"{
            "placeholder": "Name", "content": "hi",
            "x": 10, "y": 20, "width": 300, "height": 48, "max_len": 24
        }"#;
        let t: TextInput = serde_json::from_str(json).unwrap();
        assert_eq!(t.placeholder, "Name");
        assert_eq!(t.content, "hi");
        assert_eq!(t.width, 300.0);
        assert_eq!(t.max_len, 24);
    }

    #[test]
    fn runtime_state_is_not_serialized() {
        // `focused` / `caret` / `asset_id` are runtime-only, so `args` (the
        // public schema) never carries them.
        let t = TextInput {
            focused: true,
            caret: 3,
            ..Default::default()
        };
        let v = serde_json::to_value(&t).unwrap();
        assert!(v.get("focused").is_none());
        assert!(v.get("caret").is_none());
        assert!(v.get("asset_id").is_none());
        assert!(v.is_object());
    }
}

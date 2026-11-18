// A multi-line text editor in a rect of the overlay.

use alloc::boxed::Box;
use alloc::string::String;

use crate::components::{Binding, Screen, SpriteFit, Vocabulary};
use crate::ecs::Ref;
use crate::presentation::text_edit::{Metrics, TextEditor};

use super::WidgetStyle;

/// The language a [`TextArea`](#textarea) colors its text as.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, Vocabulary,
)]
#[serde(rename_all = "lowercase")]
pub enum TextLanguage {
    /// Uncolored text.
    #[default]
    #[vocab("plain")]
    Plain,
    /// HLSL shader source.
    #[vocab("hlsl")]
    Hlsl,
    /// Markdown.
    #[vocab("markdown")]
    Markdown,
}

/// A multi-line text editor: line numbers in a gutter, a scrolling window of
/// the text, a caret and selection, and scrollbars.
///
/// A click focuses it and places the caret; a drag selects, a double click a
/// word, a triple click a line. While focused it takes the keyboard: typing,
/// the arrows (by word with Option on macOS, Control elsewhere), Home and End,
/// Page Up and Down, Tab and Shift+Tab to indent, undo and redo, select all,
/// and cut, copy, and paste. Escape or a click outside it gives the keyboard
/// back.
///
/// It starts holding `content`, or the text `text` names when set, and writes
/// every edit back to that binding; the app's systems can also set the
/// binding to replace the text. Each edit sends a `UiEvent` named by
/// `on_change`.
///
/// ```rust
/// # use concinnity_engine::components::{Binding, TextArea, TextLanguage};
/// let mut notes = TextArea::default();
/// notes.text = Some(Binding::State("notes".into()));
/// notes.language = TextLanguage::Markdown;
/// notes.width = 480.0;
/// notes.height = 320.0;
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
pub struct TextArea {
    /// The text it starts with when `text` is unset.
    pub content: String,
    /// The text it shows and edits: a `state:` or `var:` binding.
    pub text: Option<Binding>,
    /// The language its text is colored as.
    pub language: TextLanguage,
    /// Whether the text can be selected and copied but not edited.
    pub read_only: bool,
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    #[asset(default = 480.0)]
    pub width: f32,
    /// Height.
    #[asset(default = 240.0)]
    pub height: f32,
    /// The event an edit sends.
    pub on_change: String,
    /// How the editor draws. Its font is best a monospace one.
    pub style: WidgetStyle,
    /// When false the editor is not drawn and takes no input.
    #[asset(default = true)]
    pub visible: bool,
    /// How a screen-owned editor maps from the reference canvas to the window
    /// when their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    pub fit: SpriteFit,
    /// [Screen](#screen) the editor belongs to. Unset, it joins the world's
    /// HUD.
    pub screen: Option<Ref<Screen>>,
    /// Runtime: whether the editor holds the keyboard. Not authored.
    #[serde(skip)]
    pub focused: bool,
    /// Runtime: the text, caret, and history being edited. Not authored.
    #[serde(skip)]
    pub editor: Box<TextEditor>,
    /// Runtime: the grid the overlay measured the font to. Not authored.
    #[serde(skip)]
    pub metrics: Option<Metrics>,
    /// Runtime: the UI state and variable revisions the text was last read
    /// at, `None` before `editor` holds the text. Not authored.
    #[serde(skip)]
    pub read_at: Option<(Option<u64>, Option<u64>)>,
}

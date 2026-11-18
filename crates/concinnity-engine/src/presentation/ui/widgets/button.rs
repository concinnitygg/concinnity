// A captioned button that fires an action.

use alloc::string::String;

use crate::components::{Binding, Icon, Screen, SpriteFit, UiAction};
use crate::ecs::Ref;

use super::WidgetStyle;

/// A captioned button: a click fires its `action`.
///
/// A button with an `icon` draws the icon centered in place of its caption,
/// such as a toolbar's tool buttons. A button can show a boolean as pressed,
/// such as a toolbar toggle whose panel is open, by naming it in `active`. A
/// disabled button draws dimmed and takes no clicks.
///
/// ```rust
/// # use concinnity_engine::components::{Button, UiAction};
/// Button {
///     label: "Save".into(),
///     action: Some(UiAction::Emit("save".into())),
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
pub struct Button {
    /// The caption.
    pub label: String,
    /// A binding whose value the button shows in place of `label`, kept
    /// current as the value changes.
    pub text_binding: Option<Binding>,
    /// An icon drawn centered in place of the caption; empty draws the
    /// caption.
    pub icon: Icon,
    /// What a click fires. Unset fires nothing.
    pub action: Option<UiAction>,
    /// A boolean the button shows as pressed while true: a `setting:`,
    /// `state:`, or `var:` binding.
    pub active: Option<Binding>,
    /// Whether the button takes no clicks and draws dimmed.
    pub disabled: bool,
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width.
    #[asset(default = 96.0)]
    pub width: f32,
    /// Height.
    #[asset(default = 26.0)]
    pub height: f32,
    /// How the button draws. Its idle fill is the style's `background`; a
    /// fully transparent one draws only the caption until hovered.
    pub style: WidgetStyle,
    /// When false the button is not drawn and takes no input.
    #[asset(default = true)]
    pub visible: bool,
    /// How a screen-owned button maps from the reference canvas to the window
    /// when their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    pub fit: SpriteFit,
    /// [Screen](#screen) the button belongs to. Unset, it joins the world's
    /// HUD.
    pub screen: Option<Ref<Screen>>,
    /// Runtime: whether the cursor is over the button. Not authored.
    #[serde(skip)]
    pub hovered: bool,
}

impl Button {
    // The clickable rect.
    pub(crate) fn rect(&self) -> [f32; 4] {
        [self.x, self.y, self.width, self.height]
    }
}

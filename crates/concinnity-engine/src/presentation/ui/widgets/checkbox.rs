// A box that flips a boolean, with its caption.

use alloc::string::String;

use crate::components::{Binding, Screen, SpriteFit};
use crate::ecs::Ref;

use super::WidgetStyle;

/// A box that shows and flips a boolean, with a caption beside it and an
/// optional dimmer note under the caption.
///
/// A click anywhere on the row flips the boolean `checked` names and sends a
/// `UiEvent` named by `on_change` carrying the new value. A disabled box keeps
/// its value, draws dimmed, and takes no clicks; its note can say why.
///
/// ```rust
/// # use concinnity_engine::components::{Binding, Checkbox};
/// Checkbox {
///     label: "Show grid".into(),
///     checked: Some(Binding::State("grid".into())),
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
pub struct Checkbox {
    /// The caption beside the box.
    pub label: String,
    /// Dimmer text under the caption, such as why the box is disabled.
    pub note: String,
    /// The boolean the box shows and flips: a `setting:`, `state:`, or `var:`
    /// binding.
    pub checked: Option<Binding>,
    /// Whether the box takes no clicks and draws dimmed.
    pub disabled: bool,
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width of the clickable row.
    #[asset(default = 220.0)]
    pub width: f32,
    /// Height of the clickable row; the box is sized from it.
    #[asset(default = 26.0)]
    pub height: f32,
    /// The event a click sends.
    pub on_change: String,
    /// How the box draws.
    pub style: WidgetStyle,
    /// When false the box is not drawn and takes no input.
    #[asset(default = true)]
    pub visible: bool,
    /// How a screen-owned box maps from the reference canvas to the window when
    /// their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    pub fit: SpriteFit,
    /// [Screen](#screen) the box belongs to. Unset, it joins the world's HUD.
    pub screen: Option<Ref<Screen>>,
    /// Runtime: whether the cursor is over the row. Not authored.
    #[serde(skip)]
    pub hovered: bool,
}

// Size of a checkbox's box, at most.
const BOX_MAX: f32 = 18.0;

impl Checkbox {
    // The clickable row.
    pub(crate) fn rect(&self) -> [f32; 4] {
        [self.x, self.y, self.width, self.height]
    }

    // The box, at the row's left, centered on its caption line.
    pub(crate) fn box_rect(&self) -> [f32; 4] {
        let line = if self.note.is_empty() {
            self.height
        } else {
            self.height * 0.5
        };
        let size = (line * 0.7).min(BOX_MAX);
        [self.x, self.y + (line - size) * 0.5, size, size]
    }
}

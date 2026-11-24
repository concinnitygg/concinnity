// A dialog that takes over its screen until one of its buttons is pressed.

use alloc::string::String;
use alloc::vec::Vec;

use crate::components::{Binding, Screen, SpriteFit, UiAction};
use crate::ecs::Ref;

use super::WidgetStyle;

/// One button of a [Modal](#modal).
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct ModalButton {
    /// The button's text.
    pub label: String,
    /// What pressing the button fires, before the dialog closes.
    pub action: Option<UiAction>,
    /// Whether the button draws in the accent fill, as the one to press.
    pub primary: bool,
}

/// A dialog drawn centered over a dimmed [Screen](#screen) while the boolean
/// `open` names is true: a title, a wrapped message, and a row of buttons.
///
/// While it is open the rest of its screen takes no input. Pressing a button
/// fires its `action` and closes the dialog; Escape closes it without one.
/// Closing sets the `open` boolean false.
///
/// ```rust
/// # use concinnity_engine::components::{Binding, Modal, ModalButton, UiAction};
/// let mut dialog = Modal::default();
/// dialog.open = Some(Binding::State("confirm_quit".into()));
/// dialog.title = "Quit?".into();
/// dialog.message = "Unsaved changes will be lost.".into();
/// dialog.buttons = vec![
///     ModalButton { label: "Cancel".into(), ..Default::default() },
///     ModalButton {
///         label: "Quit".into(),
///         action: Some(UiAction::Quit),
///         primary: true,
///     },
/// ];
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
pub struct Modal {
    /// The boolean that shows the dialog while true: a `state:` or `var:`
    /// binding. Closing the dialog sets it false.
    pub open: Option<Binding>,
    /// The heading.
    pub title: String,
    /// The text under the heading, wrapped to the dialog's width. The app's
    /// systems may set it before opening the dialog.
    pub message: String,
    /// The buttons, left to right.
    pub buttons: Vec<ModalButton>,
    /// Width of the dialog.
    #[asset(default = 420.0)]
    pub width: f32,
    /// RGBA dim drawn over the rest of the screen while the dialog is open.
    #[asset(default = [0.0, 0.0, 0.0, 0.45])]
    pub dim: [f32; 4],
    /// How the dialog draws.
    pub style: WidgetStyle,
    /// How a screen-owned dialog maps from the reference canvas to the window
    /// when their aspect ratios differ (matches [Sprite](#sprite)'s `fit`).
    pub fit: SpriteFit,
    /// [Screen](#screen) the dialog opens over. Unset, it joins the world's HUD.
    pub screen: Option<Ref<Screen>>,
    /// Runtime index of the button under the cursor. Not authored.
    #[serde(skip)]
    pub hovered: Option<usize>,
    /// Runtime rectangles of the dialog and each button, as last drawn. Not
    /// authored.
    #[serde(skip)]
    pub drawn: ModalRects,
}

/// Where a [Modal](#modal)'s card and buttons were last drawn, for the clicks
/// over them.
#[derive(Debug, Clone, Default)]
pub struct ModalRects {
    /// The dialog's card.
    pub card: [f32; 4],
    /// Each button, left to right.
    pub buttons: Vec<[f32; 4]>,
}

// Padding inside a dialog's card.
pub(crate) const PADDING: f32 = 16.0;
// Height of a dialog's buttons.
pub(crate) const BUTTON_H: f32 = 30.0;
// Gap between a dialog's buttons, and between its title, message, and buttons.
pub(crate) const GAP: f32 = 10.0;

impl Modal {
    // Lay the dialog out centered in a screen of size `area`, its message
    // `message_h` tall at a title line of `line_h`.
    pub(crate) fn lay_out(&mut self, area: [f32; 2], line_h: f32, message_h: f32) {
        let w = self.width.min(area[0]);
        let message = if self.message.is_empty() {
            0.0
        } else {
            message_h + GAP
        };
        let h = PADDING + line_h + GAP + message + BUTTON_H + PADDING;
        let card = [(area[0] - w) * 0.5, (area[1] - h) * 0.5, w, h];
        self.drawn.card = card;
        self.drawn.buttons.clear();
        let n = self.buttons.len();
        if n == 0 {
            return;
        }
        let inner = w - 2.0 * PADDING;
        let bw = (inner - GAP * (n - 1) as f32) / n as f32;
        let y = card[1] + h - PADDING - BUTTON_H;
        for i in 0..n {
            let x = card[0] + PADDING + i as f32 * (bw + GAP);
            self.drawn.buttons.push([x, y, bw, BUTTON_H]);
        }
    }

    // The button under `(x, y)`, as last drawn.
    pub(crate) fn button_at(&self, x: f32, y: f32) -> Option<usize> {
        self.drawn
            .buttons
            .iter()
            .position(|r| super::point_in(x, y, *r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dialog_centers_and_shares_its_width_between_its_buttons() {
        let mut m = Modal {
            width: 400.0,
            message: "Sure?".into(),
            buttons: alloc::vec![ModalButton::default(), ModalButton::default()],
            ..Default::default()
        };
        m.lay_out([1280.0, 720.0], 16.0, 32.0);
        let card = m.drawn.card;
        let h = PADDING + 16.0 + GAP + 32.0 + GAP + BUTTON_H + PADDING;
        assert_eq!(card, [440.0, (720.0 - h) * 0.5, 400.0, h]);
        let [a, b] = [m.drawn.buttons[0], m.drawn.buttons[1]];
        assert_eq!(a[2], b[2]);
        assert_eq!(b[0] + b[2], card[0] + card[2] - PADDING, "flush right");
        assert_eq!(m.button_at(b[0] + 1.0, b[1] + 1.0), Some(1));
    }
}

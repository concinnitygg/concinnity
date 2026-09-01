// src/editor/hook/modal_drive.rs
//
// EditorHook: the confirmation dialog's open state, click routing, and draw.
// The geometry lives in `editor/modal.rs`. While open the dialog is
// screen-modal: every press and wheel is swallowed before any other routing
// (`tick` checks it first), and only a button press closes it -- a click-away
// is not a cancel, since the dialog guards destructive actions.

use super::*;

// An open confirmation dialog.
pub(super) struct ModalState {
    pub(super) message: String,
    pub(super) buttons: Vec<modal::Button>,
}

impl EditorHook {
    // Open the dialog. Buttons past the widget's pool are dropped.
    pub(super) fn open_modal(&mut self, message: &str, mut buttons: Vec<modal::Button>) {
        buttons.truncate(modal::MAX_BUTTONS);
        self.modal = Some(ModalState {
            message: message.to_string(),
            buttons,
        });
    }

    // Resolve a press while the dialog is open: a button runs its action and
    // closes the dialog; anywhere else -- the dialog's own chrome or the dimmed
    // screen behind it -- is swallowed. Returns whether the dialog was open
    // (the press is consumed either way).
    pub(super) fn route_modal_click(
        &mut self,
        input: &FrameInput,
        vp: [f32; 2],
        world: &mut World,
    ) -> bool {
        let Some(state) = &self.modal else {
            return false;
        };
        if let Some(i) = modal::hit_button(input.mouse_x, input.mouse_y, vp, state.buttons.len()) {
            let action = state.buttons[i].action.clone();
            self.modal = None;
            self.run_modal_action(action, world);
        }
        true
    }

    fn run_modal_action(&mut self, action: modal::Action, world: &mut World) {
        match action {
            modal::Action::Dismiss => {}
            modal::Action::Worlds(confirm) => self.apply_worlds_confirm(confirm, world),
        }
    }

    // Lay out (or hide) the dialog this frame.
    pub(super) fn drive_modal_draw(
        &self,
        world: &mut World,
        vp: [f32; 2],
        shown: bool,
        mouse: [f32; 2],
    ) {
        match (&self.modal, shown) {
            (Some(state), true) => modal::apply(world, vp, &state.message, &state.buttons, mouse),
            _ => modal::hide(world),
        }
    }
}

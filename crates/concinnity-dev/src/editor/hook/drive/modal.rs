//! EditorHook: the confirmation dialog's open state, click routing, and draw.
//! The geometry lives in `editor/modal.rs`. While open the dialog is
//! screen-modal: every press and wheel is swallowed before any other routing
//! (`tick` checks it first), and only a button press closes it -- a click-away
//! is not a cancel, since the dialog guards destructive actions.

use concinnity_core::components::FrameInput;
use concinnity_core::ecs::World;

use crate::editor::hook::EditorHook;
use crate::editor::modal::{self, Dialog};
use crate::editor::widget;
use crate::editor::widget_check::Check;

impl EditorHook {
    // Open the dialog. Buttons past the widget's pool are dropped.
    pub(in crate::editor::hook) fn open_modal(
        &mut self,
        message: &str,
        buttons: Vec<modal::Button>,
    ) {
        self.set_modal(message, buttons, false, None);
    }

    // The same, with a checkbox whose state the pressed button's action reads.
    pub(in crate::editor::hook) fn open_modal_with_check(
        &mut self,
        message: &str,
        buttons: Vec<modal::Button>,
        check: Check,
    ) {
        self.set_modal(message, buttons, false, Some(check));
    }

    // The same, with a name field to type into. The field is empty unless the
    // caller seeds it. A prompt owns the keyboard while it is open, so the
    // editor's shortcuts stand down.
    pub(in crate::editor::hook) fn open_prompt(
        &mut self,
        message: &str,
        buttons: Vec<modal::Button>,
    ) {
        self.set_modal(message, buttons, true, None);
    }

    fn set_modal(
        &mut self,
        message: &str,
        mut buttons: Vec<modal::Button>,
        field: bool,
        check: Option<Check>,
    ) {
        buttons.truncate(modal::MAX_BUTTONS);
        self.modal = Some(Dialog {
            message: message.to_string(),
            buttons,
            field,
            check,
        });
    }

    // Whether a prompt is open, so the shortcuts that would otherwise fire on
    // a keystroke stand down while it is being typed into.
    pub(in crate::editor::hook) fn prompting(&self) -> bool {
        self.modal.as_ref().is_some_and(|m| m.field)
    }

    // Resolve a press while the dialog is open: a button runs its action and
    // closes the dialog, and the checkbox flips; anywhere else -- the dialog's
    // own chrome or the dimmed screen behind it -- is swallowed. Returns
    // whether the dialog was open (the press is consumed either way).
    pub(in crate::editor::hook) fn route_modal_click(
        &mut self,
        input: &FrameInput,
        vp: [f32; 2],
        world: &mut World,
    ) -> bool {
        let Some(state) = &mut self.modal else {
            return false;
        };
        let c = state.controls();
        let (mx, my) = (input.mouse_x, input.mouse_y);
        if modal::hit_check(mx, my, vp, c) {
            if let Some(check) = &mut state.check {
                check.toggle();
            }
            return true;
        }
        if let Some(i) = modal::hit_button(mx, my, vp, state.buttons.len(), c) {
            let action = state.buttons[i].action.clone();
            let checked = state.check.as_ref().is_some_and(|k| k.on);
            // Read the field before the dialog closes: reopening it (a rejected
            // name) seeds a fresh one, so the typed text has to be taken here.
            let typed = widget::field_text(world, modal::NAME_INPUT);
            self.modal = None;
            widget::seed_field(world, modal::NAME_INPUT, "");
            self.run_modal_action(action, &typed, checked, world);
        }
        true
    }

    fn run_modal_action(
        &mut self,
        action: modal::Action,
        typed: &str,
        checked: bool,
        world: &mut World,
    ) {
        match action {
            modal::Action::Dismiss => {}
            modal::Action::Worlds(confirm) => self.apply_worlds_confirm(confirm, world),
            modal::Action::NameWorld => self.name_untitled_world(typed),
            modal::Action::LeaveShaderSource { save, then } => {
                self.answer_leave_shader_source(save, then, world)
            }
            modal::Action::DeleteShader(name) => self.delete_shader(&name, checked),
        }
    }

    // Lay out (or hide) the dialog this frame.
    pub(in crate::editor::hook) fn drive_modal_draw(
        &self,
        world: &mut World,
        vp: [f32; 2],
        shown: bool,
        mouse: [f32; 2],
    ) {
        match (&self.modal, shown) {
            (Some(dialog), true) => modal::place(world, vp, dialog, mouse),
            _ => modal::hide(world),
        }
    }
}

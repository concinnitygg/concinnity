//! `EditorHook`: the confirmation dialog's open state and what its controls
//! report. The dialog (`editor/modal.rs`) is a modal panel: while it is open
//! it takes every press and the wheel, so the hook routes none of them, and
//! only a button press closes it -- a click away is not a cancel, since the
//! dialog guards destructive actions.

use concinnity_engine::components::UiEvent;
use concinnity_engine::ecs::World;

use crate::editor::hook::EditorHook;
use crate::editor::modal::{self, Check, Dialog, DialogEvent};
use crate::editor::ui::BuiltUi;

impl EditorHook {
    // Open the dialog. Buttons past the dialog's few are dropped.
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

    // The same, with a name field to type into. A prompt owns the keyboard
    // while it is open, so the editor's shortcuts stand down.
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
        // A menu is a popup, which would draw over the dialog and offer
        // choices it shuts out, so the dialog closes them.
        self.display_menu_open = false;
        self.create_menu = None;
        self.modal = Some(Dialog {
            message: message.to_string(),
            buttons,
            field,
            text: String::new(),
            check,
            shown: false,
        });
    }

    // Whether a prompt is open, so the shortcuts that would otherwise fire on
    // a keystroke stand down while it is being typed into.
    pub(in crate::editor::hook) fn prompting(&self) -> bool {
        self.modal.as_ref().is_some_and(|m| m.field)
    }

    // Take in what was typed into the prompt's field, once the world's
    // dialog shows it.
    pub(in crate::editor::hook) fn observe_dialog(&mut self, world: &World) {
        let Some(dialog) = self.modal.as_mut().filter(|d| d.field && d.shown) else {
            return;
        };
        let typed = world
            .resource::<BuiltUi>()
            .and_then(|b| b.dialog.as_ref())
            .and_then(|built| modal::typed(world, built));
        if let Some(text) = typed
            && dialog.text != text
        {
            dialog.text = text.to_string();
        }
    }

    // Act on what the dialog reported; whether `event` was one.
    pub(in crate::editor::hook) fn on_dialog_event(
        &mut self,
        world: &mut World,
        event: &UiEvent,
    ) -> bool {
        match modal::event(event) {
            Some(DialogEvent::Press(i)) => {
                self.press_dialog_button(i, world);
                true
            }
            Some(DialogEvent::Check) => {
                if let Some(check) = self.modal.as_mut().and_then(|d| d.check.as_mut()) {
                    check.toggle();
                }
                true
            }
            None => false,
        }
    }

    // Press the open dialog's button `i`: it closes, and its action runs.
    pub(in crate::editor::hook) fn press_dialog_button(&mut self, i: usize, world: &mut World) {
        let Some(dialog) = self.modal.take() else {
            return;
        };
        let Some(button) = dialog.buttons.get(i) else {
            self.modal = Some(dialog);
            return;
        };
        let checked = dialog.check.as_ref().is_some_and(|k| k.on);
        self.run_modal_action(button.action.clone(), &dialog.text, checked, world);
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
                self.answer_leave_shader_source(save, then);
            }
            modal::Action::DeleteShader(name) => self.delete_shader(&name, checked),
            modal::Action::DeleteSdfField(path) => self.delete_sdf_field(&path, checked),
        }
    }
}

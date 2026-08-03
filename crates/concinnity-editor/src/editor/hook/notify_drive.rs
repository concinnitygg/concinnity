// src/editor/hook/notify_drive.rs
//
// EditorHook: the toast stack's per-frame drive and click routing. The queue
// and lifetime policy live in `editor/notify.rs`, the card geometry in
// `editor/toast_overlay.rs`; here the hook draws the live stack each frame and
// resolves presses (a card runs its action and dismisses, the overflow row
// opens the Console, where the full history lives).

use super::*;

impl EditorHook {
    pub(super) fn drive_toasts(
        &mut self,
        world: &mut World,
        vp: [f32; 2],
        shown: bool,
        mouse: [f32; 2],
    ) {
        if !shown || vp[0] <= 0.0 || self.notifier.is_empty() {
            // One hide pass after the last toast goes; zero work while idle.
            if !self.toasts_hidden {
                toast_overlay::hide(world);
                self.toasts_hidden = true;
            }
            return;
        }
        let stack = self.notifier.stack();
        if stack.cards.is_empty() {
            if !self.toasts_hidden {
                toast_overlay::hide(world);
                self.toasts_hidden = true;
            }
            return;
        }
        toast_overlay::apply(world, &stack, vp, mouse);
        self.toasts_hidden = false;
    }

    // Resolve a press over the toast stack; `false` when it misses (the caller
    // routes on). Toasts draw above every panel, so this runs first.
    pub(super) fn try_toast_press(
        &mut self,
        mx: f32,
        my: f32,
        vp: [f32; 2],
        world: &mut World,
    ) -> bool {
        if self.toasts_hidden || self.notifier.is_empty() {
            return false;
        }
        let stack = self.notifier.stack();
        match toast_overlay::hit(mx, my, vp, &stack) {
            Some(toast_overlay::Hit::Card(slot)) => {
                if let Some(action) = self.notifier.click_card(slot) {
                    self.run_toast_action(action, world);
                }
                true
            }
            // An operation card is not dismissible; it just absorbs the press
            // so nothing behind it is clicked through.
            Some(toast_overlay::Hit::Op) => true,
            Some(toast_overlay::Hit::Overflow) => {
                self.open_console_panel(world);
                true
            }
            None => false,
        }
    }

    fn run_toast_action(&mut self, action: notify::Action, world: &mut World) {
        match action {
            notify::Action::OpenConsole => self.open_console_panel(world),
            notify::Action::GoToBehaviorFault => {
                if !self.behavior_open {
                    self.behavior_open = true;
                    self.open_behavior(world);
                }
                self.focus_panel(PanelKey::Behavior);
                self.select_behavior_fault(world);
            }
        }
    }

    fn open_console_panel(&mut self, world: &mut World) {
        if self.console_open {
            self.focus_panel(PanelKey::Console);
        } else {
            self.toggle_console(world);
        }
    }

    // Toast a checker fault a commit introduced or changed. The panel banner
    // already shows it; the toast covers the panel being closed or buried.
    // Keyed on the message so per-keystroke re-checks of the same fault stay
    // quiet.
    pub(super) fn notify_behavior_fault(&mut self, prev_fault: Option<String>) {
        let Some(message) = self.behavior_status.as_ref().and_then(|s| s.error()) else {
            return;
        };
        if prev_fault.as_deref() == Some(message) {
            return;
        }
        self.notifier.push_with(
            notify::Level::Error,
            &format!("Behavior: {message}"),
            Some(notify::Action::GoToBehaviorFault),
        );
    }

    // The open behavior's current fault message, captured before a refresh so
    // `notify_behavior_fault` can tell a new fault from a persisting one.
    pub(super) fn behavior_fault_message(&self) -> Option<String> {
        self.behavior_status
            .as_ref()
            .and_then(|s| s.error())
            .map(String::from)
    }
}

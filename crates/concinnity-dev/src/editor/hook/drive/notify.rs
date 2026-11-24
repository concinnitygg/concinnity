//! `EditorHook`: the toast stack's per-frame drive and the clicks it reports.
//! The inbox and feed live in `editor/notify.rs`, the stack widget in
//! `editor/toasts.rs`; here the hook feeds what was pushed into the stack each
//! frame, replays it into a world the panels were just built into, and runs a
//! clicked card's action (the line counting the waiting cards opens the
//! Console, where the full history lives).

use std::time::Instant;

use concinnity_engine::components::{Toast, UiEvent};
use concinnity_engine::ecs::World;

use crate::editor::hook::EditorHook;
use crate::editor::notify;
use crate::editor::panels::registry::PanelKey;
use crate::editor::toasts::{self, ToastClick};
use crate::editor::ui::BuiltUi;

impl EditorHook {
    // Feed what was pushed since the last frame into the stack, shown while
    // the HUD is.
    pub(in crate::editor::hook) fn drive_toasts(&mut self, world: &mut World, shown: bool) {
        let mut out = std::mem::take(&mut self.toast_out);
        self.toasts.feed(&self.notifier, Instant::now(), &mut out);
        send_toasts(world, &mut out);
        self.toast_out = out;
        if let Some(id) = world.resource::<BuiltUi>().and_then(|b| b.toasts) {
            toasts::set_visible(world, id, shown);
        }
    }

    // Show every live toast again in a world that has none.
    pub(in crate::editor::hook) fn replay_toasts(&mut self, world: &mut World) {
        let mut out = std::mem::take(&mut self.toast_out);
        self.toasts.replay(Instant::now(), &mut out);
        send_toasts(world, &mut out);
        self.toast_out = out;
    }

    // Whether `(x, y)` lands on the toast stack, which takes the press itself.
    pub(in crate::editor::hook) fn over_toasts(
        world: &World,
        x: f32,
        y: f32,
        vp: [f32; 2],
    ) -> bool {
        world
            .resource::<BuiltUi>()
            .and_then(|b| b.toasts)
            .is_some_and(|id| toasts::covers(world, id, x, y, vp))
    }

    // Act on a click the stack reported; whether `event` was one.
    pub(in crate::editor::hook) fn on_toast_event(
        &mut self,
        world: &mut World,
        event: &UiEvent,
    ) -> bool {
        match toasts::click(event) {
            Some(ToastClick::Dismissed(key)) => {
                if let Some(action) = self.toasts.dismiss(&key) {
                    self.run_toast_action(action, world);
                }
                true
            }
            Some(ToastClick::More) => {
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
                if !self.open[PanelKey::Behavior] {
                    self.open[PanelKey::Behavior] = true;
                    self.open_behavior(world);
                }
                self.focus_panel(PanelKey::Behavior);
                self.select_behavior_fault(world);
            }
            // A name this world no longer resolves has nothing to select.
            notify::Action::SelectAsset(name)
                if self.entry_key_named(&name).is_none()
                    && concinnity_host::thread::asset_id::lookup(&name).is_none() =>
            {
                self.open_console_panel(world);
            }
            notify::Action::SelectAsset(name) => {
                self.select_named(&name);
                self.open_asset_form(&name);
                self.reveal_in_tree(&name);
                self.pick_last = None;
                self.frame_selection(self.viewport, world);
            }
        }
    }

    fn open_console_panel(&mut self, world: &mut World) {
        if self.open[PanelKey::Console] {
            self.focus_panel(PanelKey::Console);
        } else {
            self.toggle_console(world);
        }
    }

    // Toast a checker fault a commit introduced or changed. The panel banner
    // already shows it; the toast covers the panel being closed or buried.
    // Keyed on the message so per-keystroke re-checks of the same fault stay
    // quiet.
    pub(in crate::editor::hook) fn notify_behavior_fault(&self, prev_fault: Option<&str>) {
        let Some(message) = self.behavior.status.as_ref().and_then(|s| s.error()) else {
            return;
        };
        if prev_fault == Some(message) {
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
    pub(in crate::editor::hook) fn behavior_fault_message(&self) -> Option<String> {
        self.behavior
            .status
            .as_ref()
            .and_then(|s| s.error())
            .map(String::from)
    }
}

// Send the toasts in `out` to the world's stacks, leaving `out` empty.
fn send_toasts(world: &mut World, out: &mut Vec<Toast>) {
    if out.is_empty() {
        return;
    }
    let events = world.events_mut::<Toast>();
    for toast in out.drain(..) {
        events.send(toast);
    }
}

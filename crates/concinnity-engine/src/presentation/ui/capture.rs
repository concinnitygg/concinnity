// Key and gamepad-button rebind capture for a Controls-tab rebind row.

use alloc::string::String;

use crate::components::{FrameInput, SettingCommand, SettingOp};
use crate::ecs::{PipelineContext, StepResult};
use crate::settings::{SettingKey, SettingValues};

use super::UiInputSystem;

// Shown as a rebind row's value while it waits for the user to press a key.
pub(super) const REBIND_PROMPT: &str = "Press a key...";
const PAD_REBIND_PROMPT: &str = "Press a button...";

// An in-progress key rebind: a Controls-tab rebind row was clicked and is
// waiting for the user to press a key. The next key `FrameInput` reports binds
// it; Escape cancels and restores the row's previous value text.
#[derive(Debug)]
pub(super) struct Capture {
    // The rebind setting, whose shown value is the prompt while capturing.
    setting_key: SettingKey,
    // The setting's shown value before capture began, restored if the user
    // cancels.
    prev_text: String,
}

// Whether a rebind setting binds a gamepad button rather than a key.
fn captures_button(setting_key: SettingKey) -> bool {
    matches!(setting_key, SettingKey::PadRebind(_))
}

// Show `text` as `key`'s value.
fn show(ctx: &mut PipelineContext<'_>, key: SettingKey, text: &str) {
    if let Some(values) = ctx.resource_mut::<SettingValues>() {
        values.set_text(key, text);
    }
}

impl UiInputSystem {
    // A pending rebind consumes the whole frame: the next pressed key (or
    // gamepad button, for a `pad_*` row) binds it, Escape cancels, otherwise it
    // keeps waiting. Input of the other kind is ignored, so a stray key press
    // never lands in a button row.
    pub(super) fn step_rebind_capture(
        &mut self,
        input: &FrameInput,
        ctx: &mut PipelineContext<'_>,
    ) -> StepResult {
        let Some(cap) = self.capturing.as_ref() else {
            return StepResult::Continue;
        };
        let op = if captures_button(cap.setting_key) {
            input.captured_button.map(SettingOp::RebindButton)
        } else {
            input.fresh_keys().next().map(SettingOp::Rebind)
        };
        if input.escape {
            self.cancel_capture(ctx);
        } else if let Some(op) = op
            && let Some(cap) = self.capturing.take()
        {
            // The settings system shows the new binding when it applies the
            // command; the prompt shows until then.
            ctx.events_mut::<SettingCommand>().send(SettingCommand {
                setting: cap.setting_key,
                op,
                persist: true,
            });
        }
        StepResult::Continue
    }

    // Begin a rebind capture for a clicked rebind row: stash the setting's
    // shown value (to restore on cancel) and show the prompt for the input
    // kind the row captures.
    pub(super) fn begin_capture(&mut self, setting_key: SettingKey, ctx: &mut PipelineContext<'_>) {
        let prev_text = ctx
            .resource::<SettingValues>()
            .and_then(|v| v.text(setting_key))
            .map_or_default(String::from);
        let prompt = if captures_button(setting_key) {
            PAD_REBIND_PROMPT
        } else {
            REBIND_PROMPT
        };
        show(ctx, setting_key, prompt);
        self.capturing = Some(Capture {
            setting_key,
            prev_text,
        });
    }

    // Cancel a pending rebind capture, restoring the row's previous value text.
    pub(super) fn cancel_capture(&mut self, ctx: &mut PipelineContext<'_>) {
        if let Some(cap) = self.capturing.take() {
            show(ctx, cap.setting_key, &cap.prev_text);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::components::GamepadAction;
    use crate::input::keymap::Bindable;

    use super::*;

    #[test]
    fn pad_settings_capture_a_button() {
        assert!(captures_button(SettingKey::PadRebind(GamepadAction::Jump)));
        assert!(!captures_button(SettingKey::KeyRebind(Bindable::Forward)));
    }
}

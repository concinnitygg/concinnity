// Key and gamepad-button rebind capture for a Controls-tab rebind row.

use concinnity_core::components::{FrameInput, SettingCommand, SettingOp, TextLabel};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{PipelineContext, StepResult};
use concinnity_core::settings::SettingKey;

use super::UiInputSystem;

// Shown in a rebind row's value label while it waits for the user to press a key.
pub(super) const REBIND_PROMPT: &str = "Press a key...";
const PAD_REBIND_PROMPT: &str = "Press a button...";

// An in-progress key rebind: a Controls-tab rebind row was clicked and is
// waiting for the user to press a key. The next `FrameInput.captured_key` binds
// it; Escape cancels and restores the row's previous value text.
#[derive(Debug)]
pub(super) struct Capture {
    // The rebind setting.
    setting_key: SettingKey,
    // The value `TextLabel` showing the bound key (set to a prompt while
    // capturing; GraphicsSystem rewrites it after the bind).
    value_label: Option<AssetId>,
    // The label's text before capture began, restored if the user cancels.
    prev_text: String,
}

// Whether a rebind setting binds a gamepad button rather than a key.
fn captures_button(setting_key: SettingKey) -> bool {
    matches!(setting_key, SettingKey::PadRebind(_))
}

impl UiInputSystem {
    // A pending rebind consumes the whole frame: the next pressed key (or
    // gamepad button, for a `pad_*` row) binds it, Escape cancels, otherwise it
    // keeps waiting. Input of the other kind is ignored, so a stray key press
    // never lands in a button row.
    pub(super) fn step_rebind_capture(
        &mut self,
        input: &FrameInput,
        ctx: &mut PipelineContext,
    ) -> StepResult {
        let Some(cap) = self.capturing.as_ref() else {
            return StepResult::Continue;
        };
        let op = if captures_button(cap.setting_key) {
            input.captured_button.map(SettingOp::RebindButton)
        } else {
            input.captured_key.map(SettingOp::Rebind)
        };
        if input.escape {
            self.cancel_capture(ctx);
        } else if let Some(op) = op
            && let Some(cap) = self.capturing.take()
        {
            // GraphicsSystem rewrites the value label to the new binding when it
            // reads the command next tick; the prompt shows until then.
            ctx.events_mut::<SettingCommand>().send(SettingCommand {
                setting: cap.setting_key,
                op,
                value_label: cap.value_label,
                persist: true,
            });
        }
        StepResult::Continue
    }

    // Begin a rebind capture for a clicked rebind row: stash the value label's
    // current text (to restore on cancel) and show the prompt for the input
    // kind the row captures.
    pub(super) fn begin_capture(
        &mut self,
        setting_key: SettingKey,
        value_label: Option<AssetId>,
        ctx: &mut PipelineContext,
    ) {
        let prev_text = value_label
            .and_then(|id| {
                ctx.query::<TextLabel>()
                    .find(|l| l.asset_id == id)
                    .map(|l| l.content.clone())
            })
            .unwrap_or_default();
        if let Some(id) = value_label {
            let prompt = if captures_button(setting_key) {
                PAD_REBIND_PROMPT
            } else {
                REBIND_PROMPT
            };
            crate::ecs::by_asset_id::set_text(ctx, id, prompt);
        }
        self.capturing = Some(Capture {
            setting_key,
            value_label,
            prev_text,
        });
    }

    // Cancel a pending rebind capture, restoring the row's previous value text.
    pub(super) fn cancel_capture(&mut self, ctx: &mut PipelineContext) {
        if let Some(cap) = self.capturing.take()
            && let Some(id) = cap.value_label
        {
            crate::ecs::by_asset_id::set_text(ctx, id, &cap.prev_text);
        }
    }
}

#[cfg(test)]
mod tests {
    use concinnity_core::components::GamepadAction;
    use concinnity_core::input::keymap::Bindable;

    use super::*;

    #[test]
    fn pad_settings_capture_a_button() {
        assert!(captures_button(SettingKey::PadRebind(GamepadAction::Jump)));
        assert!(!captures_button(SettingKey::KeyRebind(Bindable::Forward)));
    }
}

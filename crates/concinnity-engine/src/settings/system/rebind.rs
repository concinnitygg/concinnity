// The Controls-tab rebind rows: bind an action to a captured key or gamepad
// button, swapping with whatever action held it, and relabel both rows.

use crate::components::{ControlsCommand, GamepadAction, GamepadButton, InputKey};
use crate::ecs::PipelineContext;
use crate::input::keymap::Bindable;
use crate::render::frame::ops::RenderOps;

use super::SettingsState;
use super::rows::set_label_content;
use crate::settings::StoredSettings;

impl SettingsState {
    // Bind the action to the captured key, swapping with whatever action held it,
    // push the map to the backend, and relabel the rebound row and any swap
    // victim's.
    pub(super) fn apply_key_rebind(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        ops: &mut RenderOps,
        cfg: &mut StoredSettings,
        action: Bindable,
        key: InputKey,
    ) -> bool {
        let victim = self.keymap.action_for_key(key).filter(|&a| a != action);
        self.keymap.rebind(action, key);
        let keymap = self.keymap;
        ops.record(move |backend| backend.set_keymap(keymap));
        cfg.controls.keymap = Some(self.keymap);
        for act in [Some(action), victim].into_iter().flatten() {
            if let Some(row) = self.rebind_rows.iter().find(|r| r.action == act) {
                set_label_content(ctx, row.value_id, self.keymap.get(act).display_name());
            }
        }
        true
    }

    // The gamepad counterpart of `apply_key_rebind`. The live map travels to
    // the input poll as a ControlsCommand, since the host polls the gamepad.
    pub(super) fn apply_pad_rebind(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        cfg: &mut StoredSettings,
        action: GamepadAction,
        button: GamepadButton,
    ) -> bool {
        let victim = self
            .gamepad_map
            .action_for_button(button)
            .filter(|&a| a != action);
        self.gamepad_map.rebind(action, button);
        ctx.events_mut::<ControlsCommand>().send(ControlsCommand {
            gamepad_map: Some(self.gamepad_map),
            ..Default::default()
        });
        cfg.controls.gamepad_map = Some(self.gamepad_map);
        for act in [Some(action), victim].into_iter().flatten() {
            if let Some(row) = self.pad_rebind_rows.iter().find(|r| r.action == act) {
                set_label_content(ctx, row.value_id, self.gamepad_map.get(act).display_name());
            }
        }
        true
    }
}

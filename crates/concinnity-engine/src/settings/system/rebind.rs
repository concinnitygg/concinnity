// The Controls-tab rebind rows: bind an action to a captured key or gamepad
// button, swapping with whatever action held it, and relabel both rows.

use concinnity_core::components::{
    ControlsCommand, GamepadAction, GamepadButton, InputKey, SettingCommand,
};
use concinnity_core::ecs::PipelineContext;
use concinnity_core::render::keymap;
use concinnity_core::render::ops::RenderOps;

use super::SettingsState;
use super::rows::set_label_content;
use crate::config::Settings;

impl SettingsState {
    // Bind the action to the captured key, swapping with whatever action held it,
    // push the map to the backend, and relabel the rebound row and any swap
    // victim's.
    pub(super) fn apply_key_rebind(
        &mut self,
        ctx: &mut PipelineContext,
        ops: &mut RenderOps,
        cfg: &mut Settings,
        cmd: &SettingCommand,
        key: InputKey,
    ) -> bool {
        let Some(action) = keymap::Bindable::from_setting_key(&cmd.setting) else {
            tracing::warn!("SettingsSystem: unknown rebind '{}'", cmd.setting);
            return false;
        };
        let victim = self.keymap.action_for_key(key).filter(|&a| a != action);
        self.keymap.rebind(action, key);
        let keymap = self.keymap;
        ops.record(move |backend| backend.set_keymap(&keymap));
        cfg.controls.keymap = Some(self.keymap);
        for act in [Some(action), victim].into_iter().flatten() {
            if let Some(row) = self.rebind_rows.iter().find(|r| r.action == act) {
                set_label_content(ctx, row.value_id, self.keymap.get(act).display_name());
            }
        }
        true
    }

    // The gamepad counterpart of `apply_key_rebind`. The live map travels to
    // InputSystem as a ControlsCommand, since the gamepad is polled engine-side.
    pub(super) fn apply_pad_rebind(
        &mut self,
        ctx: &mut PipelineContext,
        cfg: &mut Settings,
        cmd: &SettingCommand,
        button: GamepadButton,
    ) -> bool {
        let Some(action) = GamepadAction::from_setting_key(&cmd.setting) else {
            tracing::warn!("SettingsSystem: unknown gamepad rebind '{}'", cmd.setting);
            return false;
        };
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

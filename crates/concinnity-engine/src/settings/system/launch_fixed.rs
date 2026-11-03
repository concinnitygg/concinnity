// The settings rows a launch flag fixes for the whole run. Each shows the
// flag's value, grayed and inert, and the drain refuses to change it, so a
// flag's value never reaches the stored settings. A preset given at launch
// fixes the preset row and every row it governs, since a change to one would
// save a Custom entered from that preset.

use crate::ecs::PipelineContext;
use crate::settings::SettingKey;

use super::SettingsState;
use super::init::disable_rows;

impl SettingsState {
    // Whether a launch flag fixed `key` for this run.
    pub(super) fn fixed_by_launch(&self, key: SettingKey) -> bool {
        self.session.fixes(key)
    }

    // Gray out and disable every settings row a launch flag fixed for this
    // run. Its value label keeps the flag's value.
    pub(super) fn gate_launch_fixed_rows(&self, ctx: &mut PipelineContext<'_>) {
        disable_rows(ctx, |key| self.fixed_by_launch(key), |_| false);
    }
}

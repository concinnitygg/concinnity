//! Read-only views of the row bookkeeping captured at init, for tests that
//! drive graphics init and check what it handed the settings state.

use crate::ecs::asset_id::AssetId;

use super::SettingsState;

impl SettingsState {
    /// The perf-stats sub-rows' labels with their authored colors.
    pub fn perf_sub_row_labels(&self) -> &[(AssetId, [f32; 3])] {
        &self.perf_sub_row_labels
    }

    /// The resolution row's labels with their authored colors.
    pub fn resolution_row_labels(&self) -> &[(AssetId, [f32; 3])] {
        &self.resolution_row_labels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_state_has_no_captured_rows() {
        let state = SettingsState::for_tests();
        assert!(state.perf_sub_row_labels().is_empty());
        assert!(state.resolution_row_labels().is_empty());
    }
}

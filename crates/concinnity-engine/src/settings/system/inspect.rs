//! Read-only views of the row bookkeeping captured at init, for tests that
//! drive graphics init and check what it handed the settings state.

use alloc::vec::Vec;

use crate::ecs::asset_id::AssetId;
use crate::settings::SettingKey;

use super::SettingsState;

/// A slider row's key and track geometry: `(key, [track_x, track_w, handle_w])`.
pub type SliderRow = (SettingKey, [f32; 3]);

impl SettingsState {
    /// The value label captured for cycle row `key`.
    pub fn cycle_value_label(&self, key: SettingKey) -> Option<AssetId> {
        self.cycle_value_labels.get(&key).copied()
    }

    /// The captured slider rows, in capture order.
    pub fn slider_rows(&self) -> Vec<SliderRow> {
        self.sliders
            .iter()
            .map(|s| (s.key, [s.track_x, s.track_w, s.handle_w]))
            .collect()
    }

    /// How many key-rebind rows were captured.
    pub fn rebind_row_count(&self) -> usize {
        self.rebind_rows.len()
    }

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
        assert_eq!(state.cycle_value_label(SettingKey::Vsync), None);
        assert!(state.slider_rows().is_empty());
        assert_eq!(state.rebind_row_count(), 0);
        assert!(state.perf_sub_row_labels().is_empty());
        assert!(state.resolution_row_labels().is_empty());
    }
}

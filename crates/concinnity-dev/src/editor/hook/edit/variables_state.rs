//! `EditorHook`: the Variables panel's session state: the selected row of the
//! table, which of its two fields holds the keyboard, and the rows it shows.

use crate::editor::hook::edit::variables::VariablesData;

#[derive(Default)]
pub(in crate::editor::hook) struct VariablesState {
    pub(in crate::editor::hook) row: Option<usize>,
    pub(in crate::editor::hook) name_focus: bool,
    pub(in crate::editor::hook) value_focus: bool,
    // The rows, read at an entries and live-values revision, and a count
    // bumped each time they are read again.
    pub(in crate::editor::hook) shown: Option<(u64, u64, VariablesData)>,
    pub(in crate::editor::hook) revision: u64,
}

//! EditorHook: the Variables panel's session state: the selected row of the
//! table, the row window's scroll, and which of its two fields holds the
//! keyboard.

#[derive(Debug, Default)]
pub(in crate::editor::hook) struct VariablesState {
    pub(in crate::editor::hook) row: Option<usize>,
    pub(in crate::editor::hook) scroll: usize,
    pub(in crate::editor::hook) name_focus: bool,
    pub(in crate::editor::hook) value_focus: bool,
}

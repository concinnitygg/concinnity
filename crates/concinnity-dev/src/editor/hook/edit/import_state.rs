//! EditorHook: the Import panel's session state: whether the path field holds
//! keyboard focus, the list window scroll, and the last Add's outcome.

use crate::editor::panels::import_panel::ImportStatus;

#[derive(Debug, Default)]
pub(in crate::editor::hook) struct ImportState {
    pub(in crate::editor::hook) focus: bool,
    pub(in crate::editor::hook) scroll: usize,
    pub(in crate::editor::hook) status: Option<ImportStatus>,
}

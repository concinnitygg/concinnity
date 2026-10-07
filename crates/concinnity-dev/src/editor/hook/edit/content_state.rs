//! EditorHook: the Content panel's session state (the visual-asset thumbnail
//! grid): the grid's first visible row, the type-chip cycle position (0 = All,
//! i = `VISUAL_TYPES[i-1]`), whether its search field holds keyboard focus, and
//! an in-flight drag-out placement (`hook/drag/content.rs`), if any.

use crate::editor::hook::drag::content::ContentDrag;

#[derive(Default)]
pub(in crate::editor::hook) struct ContentState {
    pub(in crate::editor::hook) scroll: usize,
    pub(in crate::editor::hook) type_chip: usize,
    pub(in crate::editor::hook) search_focus: bool,
    pub(in crate::editor::hook) drag: Option<ContentDrag>,
}

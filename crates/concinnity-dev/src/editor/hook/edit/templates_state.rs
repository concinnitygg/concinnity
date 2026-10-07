//! EditorHook: the Templates panel's session state: the template whose detail
//! panel is open (an index into the templates registry; `None` means the
//! detail panel is closed) and the first visible row of its asset list.

#[derive(Debug, Default)]
pub(in crate::editor::hook) struct TemplatesState {
    pub(in crate::editor::hook) detail: Option<usize>,
    pub(in crate::editor::hook) detail_scroll: usize,
}

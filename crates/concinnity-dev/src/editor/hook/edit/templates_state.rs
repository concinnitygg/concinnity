//! `EditorHook`: the Templates panel's session state: the template whose detail
//! panel is open (an index into the templates registry; `None` means the
//! detail panel is closed).

#[derive(Debug, Default)]
pub(in crate::editor::hook) struct TemplatesState {
    pub(in crate::editor::hook) detail: Option<usize>,
}

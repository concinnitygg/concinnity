//! EditorHook: the Worlds panel's session state, beside its actions in
//! `worlds.rs`.

use crate::editor::worlds::WorldRow;

// The project's worlds as of the last refresh (the listing changes
// only when the panel acts on it, so it is not re-read every frame), the row
// window's scroll, the path of the row whose triple-dot menu is open, and why
// the last preview failed.
#[derive(Debug, Default)]
pub(in crate::editor::hook) struct WorldsState {
    pub(in crate::editor::hook) rows: Vec<WorldRow>,
    pub(in crate::editor::hook) scroll: usize,
    pub(in crate::editor::hook) menu: Option<String>,
    pub(in crate::editor::hook) status: Option<String>,
    // The path of the start screen's selected row, and of the world its
    // background preview was compiled from. They part company when the
    // previewed world is deleted (the background drops back to the seeded
    // empty scene) or its compile fails. Both `None` outside the start screen,
    // which has no selection model.
    pub(in crate::editor::hook) selected: Option<String>,
    pub(in crate::editor::hook) preview: Option<String>,
}

impl WorldsState {
    // Back to the listing's top, with no menu or stale failure: the panel
    // opens and closes this way.
    pub(in crate::editor::hook) fn reset_view(&mut self) {
        self.menu = None;
        self.status = None;
        self.scroll = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dirty() -> WorldsState {
        WorldsState {
            rows: vec![WorldRow {
                open: true,
                name: "a".into(),
                path: "a.jsonl".into(),
            }],
            scroll: 3,
            menu: Some("a.jsonl".into()),
            status: Some("failed".into()),
            selected: Some("a.jsonl".into()),
            preview: Some("b.jsonl".into()),
        }
    }

    fn assert_listing_kept(s: &WorldsState) {
        assert_eq!(s.rows.len(), 1);
        assert_eq!(s.selected.as_deref(), Some("a.jsonl"));
        assert_eq!(s.preview.as_deref(), Some("b.jsonl"));
    }

    #[test]
    fn reset_view_clears_menu_status_and_scroll() {
        let mut s = dirty();
        s.reset_view();
        assert_eq!((&s.menu, &s.status, s.scroll), (&None, &None, 0));
        assert_listing_kept(&s);
    }
}

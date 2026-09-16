//! EditorHook: the command palette's session state, beside its actions in
//! `palette.rs`.

use crate::editor::palette::{self, PaletteItem};

// Shown state, a one-frame focus blur after the Ctrl+K open, the query mirrored
// off its field once a frame, the item list built on open with the matches the
// query keeps, the highlighted match with its window scroll, and the labels of
// recent commits (session state, the empty query's launch list).
#[derive(Debug, Default)]
pub(in crate::editor::hook) struct PaletteState {
    pub(in crate::editor::hook) open: bool,
    pub(in crate::editor::hook) blur: bool,
    pub(in crate::editor::hook) query: String,
    pub(in crate::editor::hook) items: Vec<PaletteItem>,
    pub(in crate::editor::hook) matches: Vec<usize>,
    pub(in crate::editor::hook) pick: usize,
    pub(in crate::editor::hook) scroll: usize,
    pub(in crate::editor::hook) recent: Vec<String>,
}

impl PaletteState {
    // A changed query is a different list, so the highlight starts again at
    // its best answer.
    pub(in crate::editor::hook) fn rerank(&mut self) {
        self.matches = palette::matches(&self.items, &self.recent, &self.query);
        self.pick = 0;
        self.scroll = 0;
    }

    pub(in crate::editor::hook) fn note_recent(&mut self, label: String) {
        self.recent.retain(|l| l != &label);
        self.recent.insert(0, label);
        self.recent.truncate(palette::RECENT_CAP);
    }
}

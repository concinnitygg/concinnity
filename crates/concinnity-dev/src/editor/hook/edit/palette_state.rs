//! `EditorHook`: the command palette's session state, beside its actions in
//! `palette.rs`.

use crate::editor::palette::{self, PaletteItem};

// Palette state: the open blur, the mirrored query, items and matches, the
// highlight, recent commits, and change counters.
#[derive(Debug, Default)]
pub(in crate::editor::hook) struct PaletteState {
    pub(in crate::editor::hook) blur: bool,
    pub(in crate::editor::hook) query: String,
    pub(in crate::editor::hook) items: Vec<PaletteItem>,
    pub(in crate::editor::hook) matches: Vec<usize>,
    pub(in crate::editor::hook) pick: usize,
    pub(in crate::editor::hook) recent: Vec<String>,
    pub(in crate::editor::hook) revision: u64,
    pub(in crate::editor::hook) moved: u64,
}

impl PaletteState {
    // A changed query is a different list, so the highlight starts again at
    // its best answer.
    pub(in crate::editor::hook) fn rerank(&mut self) {
        self.matches = palette::matches(&self.items, &self.recent, &self.query);
        self.revision = self.revision.wrapping_add(1);
        self.highlight(0);
    }

    // Highlight match `at`, bringing it into view.
    pub(in crate::editor::hook) fn highlight(&mut self, at: usize) {
        self.pick = at;
        self.moved = self.moved.wrapping_add(1);
    }

    pub(in crate::editor::hook) fn note_recent(&mut self, label: String) {
        self.recent.retain(|l| l != &label);
        self.recent.insert(0, label);
        self.recent.truncate(palette::RECENT_CAP);
    }
}

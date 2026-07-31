// src/editor/hook/hide.rs
//
// EditorHook: hide-selected, isolate, and unhide-all. H adds the selection to
// the manual hide set (the same one the outliner eye edits); Shift+H toggles
// an isolate that keeps only the selection visible; Ctrl+H clears both. The
// composition rule lives in `editor/visibility.rs`.

use super::*;

impl EditorHook {
    // H: manually hide every selected asset.
    pub(super) fn hide_selected(&mut self) {
        let names: Vec<String> = self.selection.iter().map(str::to_string).collect();
        if names.is_empty() {
            return;
        }
        let n = names.len();
        self.hidden_assets.extend(names);
        self.console_sink.info(&format!("hid {n} selected"));
    }

    // Shift+H: isolate the selection (hide everything else), or leave an
    // active isolate.
    pub(super) fn toggle_isolate(&mut self) {
        if self.isolate.take().is_some() {
            self.console_sink.info("isolate off");
            return;
        }
        let keep: std::collections::BTreeSet<String> =
            self.selection.iter().map(str::to_string).collect();
        if keep.is_empty() {
            return;
        }
        self.console_sink
            .info(&format!("isolated {} selected", keep.len()));
        self.isolate = Some(keep);
    }

    // Ctrl+H: everything visible again (manual hides and isolate both).
    pub(super) fn unhide_all(&mut self) {
        let had = !self.hidden_assets.is_empty() || self.isolate.is_some();
        self.hidden_assets.clear();
        self.isolate = None;
        if had {
            self.console_sink.info("unhid all");
        }
    }

    // The per-name hide test billboards and other per-entry filters use.
    pub(super) fn name_hidden(&self, name: &str) -> bool {
        visibility::is_hidden(name, &self.hidden_assets, self.isolate.as_ref())
    }

    // The full effective hide set resolved to this world's dense ids, for the
    // per-frame `EditorHidden` publish. Names that no longer resolve (a
    // renamed or deleted asset) simply drop out until they return.
    pub(super) fn effective_hidden_ids(&self) -> std::collections::BTreeSet<AssetId> {
        if self.hidden_assets.is_empty() && self.isolate.is_none() {
            return std::collections::BTreeSet::new();
        }
        let all = self.entries.iter().filter_map(entry_name);
        let hidden = visibility::effective_hidden(&self.hidden_assets, self.isolate.as_ref(), all);
        let table = crate::ecs::asset_id::name_table();
        table
            .iter()
            .enumerate()
            .filter(|(_, n)| hidden.contains(n.as_str()))
            .map(|(i, _)| AssetId(i as u32))
            .collect()
    }
}

//! EditorHook: hide-selected, isolate, and unhide-all. H adds the selection to
//! the manual hide set (the same one the outliner eye edits); Shift+H toggles
//! an isolate that keeps only the selection visible; Ctrl+H clears both. The
//! composition rule lives in `editor/visibility.rs`.

use concinnity_core::ecs::asset_id::AssetId;

use super::EditorHook;
use crate::editor::asset_handle::AssetHandle;
use crate::editor::visibility;

impl EditorHook {
    // H: manually hide every selected asset.
    pub(super) fn hide_selected(&mut self) {
        let handles = self.selected();
        if handles.is_empty() {
            return;
        }
        let n = handles.len();
        self.hidden_assets.extend(handles);
        self.console_sink.info(&format!("hid {n} selected"));
    }

    // Shift+H: isolate the selection (hide everything else), or leave an
    // active isolate.
    pub(super) fn toggle_isolate(&mut self) {
        if self.isolate.take().is_some() {
            self.console_sink.info("isolate off");
            return;
        }
        let keep: std::collections::BTreeSet<AssetHandle> = self.selected().into_iter().collect();
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

    // The per-asset hide test billboards and other per-entry filters use.
    pub(super) fn handle_hidden(&self, handle: &AssetHandle) -> bool {
        visibility::is_hidden(handle, &self.hidden_assets, self.isolate.as_ref())
    }

    // The full effective hide set resolved to this world's dense ids, for the
    // per-frame `HiddenAssets` publish. An isolate hides every authored entry
    // outside it. Handles that no longer resolve (a deleted entry, an
    // expansion that stopped) simply drop out.
    pub(super) fn effective_hidden_ids(&self) -> std::collections::BTreeSet<AssetId> {
        if self.hidden_assets.is_empty() && self.isolate.is_none() {
            return std::collections::BTreeSet::new();
        }
        let all = (0..self.entries.len())
            .filter_map(|i| self.entries.key_at(i))
            .map(AssetHandle::Entry);
        let hidden = visibility::effective_hidden(&self.hidden_assets, self.isolate.as_ref(), all);
        hidden
            .iter()
            .filter_map(|h| self.handle_asset_id(h))
            .collect()
    }

    // The locked assets resolved to this world's dense ids, for the pick paths
    // that skip them.
    pub(super) fn locked_ids(&self) -> std::collections::BTreeSet<AssetId> {
        self.locked_assets
            .iter()
            .filter_map(|h| self.handle_asset_id(h))
            .collect()
    }
}

// src/editor/hook/expanded_tab.rs
//
// EditorHook: the Assets panel's Expanded tab. Owns when the cooked model is
// rebuilt (an expansion is far too costly per frame, so it is recomputed only
// when the tab is up and something changed), the group collapse state, and the
// "+" that copies a generated asset's entry into the config.

use super::*;

impl EditorHook {
    // Rebuild the Expanded model from the working entries if it is out of date
    // and the tab is showing. Called from the frame drive rather than from each
    // edit, so a burst of edits costs one expansion, not one each.
    pub(super) fn refresh_expanded_if_needed(&mut self) {
        if self.tab != Tab::Expanded || !self.expanded_stale || !self.panel_open {
            return;
        }
        self.expanded_stale = false;
        match self.expand_entries() {
            Ok(groups) => {
                self.expanded_groups = groups;
                self.expanded_status = None;
                // A group that no longer exists must not stay unfolded.
                let n = self.expanded_groups.len();
                self.expanded_open.retain(|&g| g < n);
            }
            Err(e) => {
                // A world mid-edit may not cook; the tab says so rather than
                // showing a stale list.
                self.expanded_groups.clear();
                self.expanded_open.clear();
                self.expanded_status = Some(e);
            }
        }
        self.clamp_expanded_scroll();
    }

    // Cook the working entries into an expanded world. The entries are the
    // in-memory edit state, so consumers reflect unsaved changes rather than
    // what is on disk. Shared with the Outliner's refresh.
    pub(super) fn cook_working_entries(
        &self,
    ) -> Result<concinnity_cook::world::LoadedWorld, String> {
        let content = crate::world::write_world_jsonl(&self.entries).map_err(|e| e.to_string())?;
        concinnity_cook::prepare_world(&content).map_err(|errs| {
            errs.first()
                .cloned()
                .unwrap_or_else(|| "the world does not build".to_string())
        })
    }

    // Cook the working entries and build the grouped model.
    fn expand_entries(&self) -> Result<Vec<ExpandedGroup>, String> {
        Ok(expanded::groups_from(&self.cook_working_entries()?))
    }

    // The tab's rows: every group header, plus the assets of the unfolded ones.
    pub(super) fn expanded_rows(&self) -> Vec<ExpandedRow> {
        expanded::rows(&self.expanded_groups, &self.expanded_open)
    }

    pub(super) fn switch_tab(&mut self, tab: Tab) {
        if self.tab == tab {
            return;
        }
        self.tab = tab;
        // The Config tab's transient overlays make no sense over the other body.
        self.combo = Combo::Closed;
        self.row_menu = None;
    }

    pub(super) fn toggle_expanded_group(&mut self, group: usize) {
        match self.expanded_open.iter().position(|&g| g == group) {
            Some(i) => {
                self.expanded_open.remove(i);
            }
            None => self.expanded_open.push(group),
        }
        self.clamp_expanded_scroll();
    }

    // Copy one generated asset's entry into the config, where it overrides what
    // the expansion produces (the cook side drops the generated copy in favour
    // of an authored one of the same name and type). Does not rebuild the
    // preview: the built world is unchanged by construction, since the entry is
    // exactly what the expansion would have emitted. Editing the copy from the
    // Config tab goes through the normal apply path and rebuilds then.
    pub(super) fn add_expanded(&mut self, group: usize, index: usize) {
        let Some(asset) = self
            .expanded_groups
            .get(group)
            .and_then(|g| g.assets.get(index))
        else {
            return;
        };
        let Some(entry) = asset.entry.clone() else {
            return;
        };
        if entry_name(&entry).is_some_and(|n| self.name_taken(n)) {
            return;
        }
        let name = asset.name.clone();
        self.entries.push(entry);
        self.dirty = true;
        // The asset is now a config line, so its row must gray out (and the
        // Outliner must relist it as authored).
        self.expanded_stale = true;
        self.outliner_stale = true;
        tracing::info!("editor: copied '{name}' into the config");
    }

    fn clamp_expanded_scroll(&mut self) {
        self.expanded_scroll = self.expanded_scroll.min(self.max_expanded_scroll());
    }

    fn max_expanded_scroll(&self) -> usize {
        self.expanded_rows().len().saturating_sub(panel::MAX_ROWS)
    }

    pub(super) fn scroll_expanded(&mut self, delta: f32) {
        let max = self.max_expanded_scroll();
        self.expanded_scroll = scroll_step(self.expanded_scroll, delta, max);
    }
}

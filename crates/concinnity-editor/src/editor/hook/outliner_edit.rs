// src/editor/hook/outliner_edit.rs
//
// EditorHook: the Outliner panel's drive. Owns when the cooked tree is
// rebuilt (an expansion is far too costly per frame, so it is recomputed only
// when the panel is up and something changed), the fold state, the search
// field, the editor-session hide / lock sets, and the two-way selection sync
// with the viewport (a row click drives the same selection set `hook/pick.rs`
// fills; a viewport pick unfolds and scrolls to its row).

use super::*;

impl EditorHook {
    // Rebuild the Outliner's grouped tree from the working entries if it is
    // out of date and the panel is showing. Called from the frame drive rather
    // than from each edit, so a burst of edits costs one expansion, not one
    // each.
    pub(super) fn refresh_outliner_if_needed(&mut self) {
        if !self.outliner_open || !self.outliner_stale {
            return;
        }
        self.outliner_stale = false;
        match self.cook_working_entries() {
            Ok(loaded) => {
                self.outliner_groups = outliner::groups_from(&loaded);
                self.outliner_status = None;
                // A group that no longer exists must not stay unfolded.
                let n = self.outliner_groups.len();
                self.outliner_unfolded.retain(|&g| g < n);
            }
            Err(e) => {
                // A world mid-edit may not cook; the panel says so rather than
                // showing a stale tree.
                self.outliner_groups.clear();
                self.outliner_unfolded.clear();
                self.outliner_status = Some(short_status(&e));
            }
        }
    }

    // The flattened tree under the live search filter (read back from the
    // engine-edited field).
    pub(super) fn outliner_rows(&self, world: &World) -> Vec<OutlinerRow> {
        let filter = widget::field_text(world, outliner_panel::FILTER_INPUT);
        outliner::rows(&self.outliner_groups, &self.outliner_unfolded, &filter)
    }

    pub(super) fn make_outliner_view<'a>(
        &'a self,
        rows: &'a [OutlinerRow],
        mouse: [f32; 2],
    ) -> OutlinerView<'a> {
        OutlinerView {
            rows,
            scroll: self.outliner_scroll,
            // Focus is asserted only while frontmost, matching the other
            // panels' guard against fighting for typed keys.
            focus: self.outliner_focus && self.panel_order.last() == Some(&PanelKey::Outliner),
            selection: &self.selection,
            hidden: &self.hidden_assets,
            locked: &self.locked_assets,
            status: self.outliner_status.as_deref(),
            total: self.outliner_groups.iter().map(|g| g.assets.len()).sum(),
            mouse,
        }
    }

    pub(super) fn scroll_outliner(&mut self, delta: f32, world: &World) {
        let max = self
            .outliner_rows(world)
            .len()
            .saturating_sub(outliner_panel::ROW_POOL);
        self.outliner_scroll = scroll_step(self.outliner_scroll, delta, max);
    }

    // Route a resolved Outliner click.
    pub(super) fn apply_outliner_action(&mut self, action: OutlinerAction, world: &mut World) {
        match action {
            OutlinerAction::FocusFilter => self.outliner_focus = true,
            OutlinerAction::ToggleGroup(group) => {
                match self.outliner_unfolded.iter().position(|&g| g == group) {
                    Some(i) => {
                        self.outliner_unfolded.remove(i);
                    }
                    None => self.outliner_unfolded.push(group),
                }
                let max = self
                    .outliner_rows(world)
                    .len()
                    .saturating_sub(outliner_panel::ROW_POOL);
                self.outliner_scroll = self.outliner_scroll.min(max);
            }
            // A row click mirrors a viewport pick: plain replaces the
            // selection and opens the asset's editing surface; shift toggles
            // membership, the form following the active member.
            OutlinerAction::Select(name) => {
                if self.shift_held {
                    if self.selection.toggle(name.clone()) {
                        self.focus_ui_on(&name, world);
                    } else {
                        self.follow_active(world);
                    }
                } else {
                    self.selection.replace(name.clone());
                    self.focus_ui_on(&name, world);
                }
                self.pick_last = None;
            }
            OutlinerAction::ToggleHide(name) => {
                if !self.hidden_assets.remove(&name) {
                    self.hidden_assets.insert(name);
                }
            }
            OutlinerAction::ToggleLock(name) => {
                if !self.locked_assets.remove(&name) {
                    self.locked_assets.insert(name);
                }
            }
            // A click on panel chrome blurs the search field.
            OutlinerAction::Consume => self.outliner_focus = false,
        }
    }

    // Enter blurs the search field (the filter applies live while typing).
    pub(super) fn outliner_keys(&mut self, _world: &mut World, input: &FrameInput) {
        if self.outliner_focus && input.captured_key == Some(crate::assets::Key::Enter) {
            self.outliner_focus = false;
        }
    }

    // Unfold the group holding `name` and scroll its row into the window, so a
    // viewport pick is always visible in the tree. A name the tree does not
    // list (a filtered-out match, a mid-edit cook failure) leaves it as-is.
    pub(super) fn reveal_outliner(&mut self, name: &str, world: &World) {
        if !self.outliner_open {
            return;
        }
        if let Some(group) = self
            .outliner_groups
            .iter()
            .position(|g| g.assets.iter().any(|a| a.name == name))
            && !self.outliner_unfolded.contains(&group)
        {
            self.outliner_unfolded.push(group);
        }
        let rows = self.outliner_rows(world);
        let Some(row) = rows
            .iter()
            .position(|r| matches!(r, OutlinerRow::Asset { name: n, .. } if n == name))
        else {
            return;
        };
        // Scroll only when the row is outside the visible window, keeping its
        // group header in view when it sits directly above.
        if row < self.outliner_scroll || row >= self.outliner_scroll + outliner_panel::ROW_POOL {
            let max = rows.len().saturating_sub(outliner_panel::ROW_POOL);
            self.outliner_scroll = row.saturating_sub(1).min(max);
        }
    }

    // The session hide set resolved to this world's dense ids, for the
    // per-frame `EditorHidden` publish. Names that no longer resolve (a
    // renamed or deleted asset) simply drop out until they return.
    pub(super) fn hidden_asset_ids(&self) -> std::collections::BTreeSet<AssetId> {
        if self.hidden_assets.is_empty() {
            return std::collections::BTreeSet::new();
        }
        let table = crate::ecs::asset_id::name_table();
        table
            .iter()
            .enumerate()
            .filter(|(_, n)| self.hidden_assets.contains(n.as_str()))
            .map(|(i, _)| AssetId(i as u32))
            .collect()
    }
}

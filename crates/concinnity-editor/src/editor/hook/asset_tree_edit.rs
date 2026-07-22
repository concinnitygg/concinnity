// src/editor/hook/asset_tree_edit.rs
//
// EditorHook: the Assets panel's drive. Owns when the cooked tree is rebuilt
// (an expansion is far too costly per frame, so it is recomputed only when the
// panel is up and something changed), the fold state, the search field, the "+"
// type picker, the editor-session hide / lock sets, and the two-way selection
// sync with the viewport (a row click drives the same selection set
// `hook/pick.rs` fills; a viewport pick unfolds and scrolls to its row).

use super::*;

impl EditorHook {
    // Rebuild the grouped tree from the working entries if it is out of date and
    // the panel is showing. Called from the frame drive rather than from each
    // edit, so a burst of edits costs one expansion, not one each.
    pub(super) fn refresh_tree_if_needed(&mut self) {
        if !self.panel_open || !self.tree_stale {
            return;
        }
        self.tree_stale = false;
        match self.cook_working_entries() {
            Ok(loaded) => {
                self.tree_groups = asset_tree::groups_from(&loaded);
                self.tree_status = None;
                // A group that no longer exists must not stay unfolded.
                let n = self.tree_groups.len();
                self.tree_unfolded.retain(|&g| g < n);
            }
            Err(e) => {
                // A world mid-edit may not cook; the panel says so rather than
                // showing a stale tree.
                self.tree_groups.clear();
                self.tree_unfolded.clear();
                self.tree_status = Some(short_status(&e));
            }
        }
    }

    // Cook the working entries into an expanded world. The entries are the
    // in-memory edit state, so consumers reflect unsaved changes rather than
    // what is on disk.
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

    // The flattened tree under the live search filter (read back from the
    // engine-edited field). Blank while the picker is open, since the field is
    // narrowing the picker's options rather than the tree.
    pub(super) fn tree_rows(&self, world: &World) -> Vec<TreeRow> {
        let filter = if self.picker_open {
            String::new()
        } else {
            widget::field_text(world, panel::SEARCH_INPUT)
        };
        asset_tree::rows(&self.tree_groups, &self.tree_unfolded, &filter)
    }

    // The "+" picker's option list, narrowed by the search field and sorted
    // ascending. `None` while the picker is closed.
    pub(super) fn picker_options(&self, world: &World) -> Option<Vec<String>> {
        if !self.picker_open {
            return None;
        }
        let filter = widget::field_text(world, panel::SEARCH_INPUT).to_lowercase();
        let mut opts: Vec<String> = panel::picker_types()
            .filter(|t| filter.is_empty() || t.to_lowercase().contains(&filter))
            .map(|t| t.to_string())
            .collect();
        opts.sort();
        Some(opts)
    }

    // The asset behind a resolved row click, if the tree still lists it.
    fn tree_asset(&self, group: usize, index: usize) -> Option<&asset_tree::TreeAsset> {
        self.tree_groups.get(group)?.assets.get(index)
    }

    pub(super) fn make_view<'a>(&'a self, d: &'a PanelData, mouse: [f32; 2]) -> PanelView<'a> {
        PanelView {
            rows: &d.rows,
            scroll: self.tree_scroll,
            // Focus is asserted only while frontmost, matching the other panels'
            // guard against fighting for typed keys.
            search_focus: self.search_focus && self.panel_order.last() == Some(&PanelKey::Assets),
            picker_options: d.picker_options.as_deref(),
            picker_scroll: self.picker_scroll,
            selection: &self.selection,
            hidden: &self.hidden_assets,
            locked: &self.locked_assets,
            row_menu: self.row_menu.as_deref(),
            total: self.tree_groups.iter().map(|g| g.assets.len()).sum(),
            status: self.tree_status.as_deref(),
            mouse,
        }
    }

    pub(super) fn scroll_tree(&mut self, delta: f32, world: &World) {
        if self.picker_open {
            let total = self
                .picker_options(world)
                .map_or(0, |o| o.len())
                .saturating_sub(panel::ROW_POOL);
            self.picker_scroll = scroll_step(self.picker_scroll, delta, total);
            return;
        }
        let max = self.tree_rows(world).len().saturating_sub(panel::ROW_POOL);
        self.tree_scroll = scroll_step(self.tree_scroll, delta, max);
        self.row_menu = None;
    }

    fn clamp_tree_scroll(&mut self, world: &World) {
        let max = self.tree_rows(world).len().saturating_sub(panel::ROW_POOL);
        self.tree_scroll = self.tree_scroll.min(max);
    }

    // Route a resolved Assets-panel click.
    pub(super) fn apply_panel(&mut self, action: PanelAction, world: &mut World) {
        match action {
            PanelAction::FocusSearch => {
                self.search_focus = true;
                self.row_menu = None;
            }
            PanelAction::TogglePicker => {
                if self.picker_open {
                    self.picker_open = false;
                } else {
                    // The field keeps whatever was typed: the picker simply
                    // narrows by the same text the tree was filtered by.
                    self.picker_open = true;
                    self.picker_scroll = 0;
                    self.row_menu = None;
                    self.search_focus = true;
                }
            }
            PanelAction::PickOption(i) => {
                let picked = self.picker_options(world).and_then(|o| o.get(i).cloned());
                if let Some(ty) = picked {
                    // A config singleton edits the world's existing instance if
                    // it has one, else adds it (edit-or-add); a multi-instance
                    // asset always adds a new one.
                    let existing = panel::is_singleton(&ty)
                        .then(|| {
                            self.entries
                                .iter()
                                .position(|e| entry_type(e) == Some(ty.as_str()))
                        })
                        .flatten();
                    let target = match existing {
                        Some(idx) => FormTarget::Entry(idx),
                        None => FormTarget::New,
                    };
                    self.picker_open = false;
                    self.open_form(world, ty, target);
                }
            }
            PanelAction::ToggleGroup(group) => {
                match self.tree_unfolded.iter().position(|&g| g == group) {
                    Some(i) => {
                        self.tree_unfolded.remove(i);
                    }
                    None => self.tree_unfolded.push(group),
                }
                self.row_menu = None;
                self.clamp_tree_scroll(world);
            }
            // A row click mirrors a viewport pick: plain replaces the selection
            // and opens the asset's editing surface; shift toggles membership,
            // the form following the active member.
            PanelAction::SelectRow(group, index) => {
                let Some(name) = self.tree_asset(group, index).map(|a| a.name.clone()) else {
                    return;
                };
                self.row_menu = None;
                if self.shift_held {
                    if self.selection.toggle(name.clone()) {
                        self.open_asset_form(&name, world);
                    } else {
                        self.follow_active(world);
                    }
                } else {
                    self.selection.replace(name.clone());
                    self.open_asset_form(&name, world);
                }
                self.pick_last = None;
            }
            PanelAction::ToggleHide(group, index) => {
                if let Some(name) = self.tree_asset(group, index).map(|a| a.name.clone())
                    && !self.hidden_assets.remove(&name)
                {
                    self.hidden_assets.insert(name);
                }
            }
            PanelAction::OpenRowMenu(group, index) => {
                self.row_menu = self.tree_asset(group, index).map(|a| a.name.clone());
            }
            // The lock is per-session, so it closes the menu without touching
            // the entries (unlike Delete, which is an authored edit).
            PanelAction::RowToggleLock => {
                if let Some(name) = self.row_menu.take()
                    && !self.locked_assets.remove(&name)
                {
                    self.locked_assets.insert(name);
                }
            }
            PanelAction::RowDelete => {
                if let Some(name) = self.row_menu.take() {
                    self.delete_entry_named(&name);
                }
                self.clamp_tree_scroll(world);
            }
            PanelAction::CloseOverlays => {
                self.picker_open = false;
                self.row_menu = None;
                self.search_focus = false;
            }
            PanelAction::Consume => {}
        }
    }

    // Open the editing surface for the asset called `name`: an authored line
    // edits in place, an asset the build generates opens a form seeded from what
    // the expansion produced (confirming it appends the line and so overrides
    // the expansion), and anything else has nothing to edit, so an open form is
    // closed rather than left pointing at the previous asset.
    pub(super) fn open_asset_form(&mut self, name: &str, world: &mut World) {
        if let Some(idx) = self
            .entries
            .iter()
            .position(|e| entry_name(e) == Some(name))
        {
            if let Some(ty) = self.entries.get(idx).and_then(entry_type).map(String::from) {
                self.open_form(world, ty, FormTarget::Entry(idx));
            }
            return;
        }
        let promote = self
            .tree_groups
            .iter()
            .flat_map(|g| &g.assets)
            .find(|a| a.name == name)
            .and_then(|a| a.promote.clone());
        match promote {
            Some(entry) => {
                if let Some(ty) = entry_type(&entry).map(String::from) {
                    self.open_form(world, ty, FormTarget::Promote(entry));
                }
            }
            None => self.close_form(),
        }
    }

    // Remove the authored line called `name`, if the world has one. Generated
    // assets have no line to delete: they are removed by editing whatever
    // produced them.
    fn delete_entry_named(&mut self, name: &str) {
        let Some(idx) = self
            .entries
            .iter()
            .position(|e| entry_name(e) == Some(name))
        else {
            return;
        };
        self.entries.remove(idx);
        self.mark_changed();
        // The open form indexes into `entries`: deleting the edited entry closes
        // it, and deleting an earlier one shifts it.
        match self.form_target {
            FormTarget::Entry(e) if e == idx => self.close_form(),
            FormTarget::Entry(e) if e > idx => self.form_target = FormTarget::Entry(e - 1),
            _ => {}
        }
    }

    // Enter blurs the search field (the filter applies live while typing).
    pub(super) fn tree_keys(&mut self, _world: &mut World, input: &FrameInput) {
        if self.search_focus && input.captured_key == Some(crate::assets::Key::Enter) {
            self.search_focus = false;
        }
    }

    // Unfold the group holding `name` and scroll its row into the window, so a
    // viewport pick is always visible in the tree. A name the tree does not list
    // (a filtered-out match, a mid-edit cook failure) leaves it as-is.
    pub(super) fn reveal_in_tree(&mut self, name: &str, world: &World) {
        if !self.panel_open {
            return;
        }
        if let Some(group) = self
            .tree_groups
            .iter()
            .position(|g| g.assets.iter().any(|a| a.name == name))
            && !self.tree_unfolded.contains(&group)
        {
            self.tree_unfolded.push(group);
        }
        let rows = self.tree_rows(world);
        let Some(row) = rows
            .iter()
            .position(|r| matches!(r, TreeRow::Asset { name: n, .. } if n == name))
        else {
            return;
        };
        // Scroll only when the row is outside the visible window, keeping its
        // group header in view when it sits directly above.
        if row < self.tree_scroll || row >= self.tree_scroll + panel::ROW_POOL {
            let max = rows.len().saturating_sub(panel::ROW_POOL);
            self.tree_scroll = row.saturating_sub(1).min(max);
        }
    }

    // The session hide set resolved to this world's dense ids, for the per-frame
    // `EditorHidden` publish. Names that no longer resolve (a renamed or deleted
    // asset) simply drop out until they return.
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

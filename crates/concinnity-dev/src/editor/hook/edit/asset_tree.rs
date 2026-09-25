//! EditorHook: the Assets panel's drive. Owns when the cooked tree is rebuilt
//! (an expansion is far too costly per frame, so it is recomputed only when the
//! panel is up and something changed), the fold state, the search field, the "+"
//! type picker, the editor-session hide / lock sets, and the two-way selection
//! sync with the viewport (a row click drives the same selection set
//! `hook/pick.rs` fills; a viewport pick unfolds and scrolls to its row).

use concinnity_core::components::FrameInput;
use concinnity_core::components::InputKey;
use concinnity_core::ecs::World;

use crate::editor::asset_handle::AssetHandle;
use crate::editor::entry_list::build_text;
use crate::editor::hook::{
    EditorHook, FormTarget, PanelData, entry_type, scroll_step, short_status,
};
use crate::editor::panels::asset_tree::{self, TreeRow};
use crate::editor::panels::assets_panel::{self, PanelAction, PanelView};
use crate::editor::panels::registry::PanelKey;
use crate::editor::selection::SelectedNames;
use crate::editor::widget;

impl EditorHook {
    // Rebuild the grouped tree from the working entries if it is out of date
    // and a consumer is showing (the Assets tree, the Content grid, or the
    // command palette). Called from the frame drive rather than from each
    // edit, so a burst of edits costs one expansion, not one each.
    pub(in crate::editor::hook) fn refresh_tree_if_needed(&mut self) {
        if !(self.panel_open || self.content_open || self.palette.open) || !self.tree_stale {
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
    ) -> Result<concinnity_cook::build_only::LoadedWorld, String> {
        Self::cook_entries(&self.entries)
    }

    // The expansion front half over an arbitrary entry list. Named apart from
    // the working-entry form because the live-edit path expands the list the
    // running world was built from, which a pending edit has already moved on
    // from.
    pub(in crate::editor::hook) fn cook_entries(
        entries: &[serde_json::Value],
    ) -> Result<concinnity_cook::build_only::LoadedWorld, String> {
        let content = build_text(entries).map_err(|e| e.to_string())?;
        concinnity_cook::prepare_world(&content, crate::project::assets_dir().as_deref()).map_err(
            |errs| {
                errs.first()
                    .cloned()
                    .unwrap_or_else(|| "the world does not build".to_string())
            },
        )
    }

    // The flattened tree under the live search filter (read back from the
    // engine-edited field). Blank while the picker is open, since the field is
    // narrowing the picker's options rather than the tree.
    pub(in crate::editor::hook) fn tree_rows(&self, world: &World) -> Vec<TreeRow> {
        let filter = if self.picker_open {
            String::new()
        } else {
            widget::field_text(world, assets_panel::SEARCH_INPUT)
        };
        asset_tree::rows(&self.tree_groups, &self.tree_unfolded, &filter)
    }

    // The "+" picker's option list, narrowed by the search field and sorted
    // ascending. `None` while the picker is closed.
    pub(in crate::editor::hook) fn picker_options(
        &self,
        world: &World,
    ) -> Option<Vec<assets_panel::PickerOption>> {
        if !self.picker_open {
            return None;
        }
        let filter = widget::field_text(world, assets_panel::SEARCH_INPUT).to_lowercase();
        let mut opts: Vec<assets_panel::PickerOption> = assets_panel::picker_types()
            .filter(|t| filter.is_empty() || t.to_lowercase().contains(&filter))
            .map(assets_panel::PickerOption::new)
            .collect();
        opts.sort_by(|a, b| a.name.cmp(&b.name));
        Some(opts)
    }

    // The asset behind a resolved row click, if the tree still lists it.
    fn tree_asset(&self, group: usize, index: usize) -> Option<&asset_tree::TreeAsset> {
        self.tree_groups.get(group)?.assets.get(index)
    }

    // The handle of the asset behind a resolved row click.
    fn tree_handle(&self, group: usize, index: usize) -> Option<AssetHandle> {
        Some(self.handle_for(&self.tree_asset(group, index)?.name))
    }

    pub(in crate::editor::hook) fn make_view<'a>(
        &'a self,
        d: &'a PanelData,
        selected: &'a SelectedNames,
        mouse: [f32; 2],
    ) -> PanelView<'a> {
        PanelView {
            rows: &d.rows,
            scroll: self.tree_scroll,
            // Focus is asserted only while frontmost, matching the other panels'
            // guard against fighting for typed keys.
            search_focus: self.search_focus && self.panel_order.last() == Some(&PanelKey::Assets),
            picker_options: d.picker_options.as_deref(),
            picker_scroll: self.picker_scroll,
            selected,
            hidden: &d.hidden,
            locked: &d.locked,
            row_menu: d.row_menu.as_deref(),
            total: self.tree_groups.iter().map(|g| g.assets.len()).sum(),
            status: self.tree_status.as_deref(),
            mouse,
        }
    }

    // The Assets panel's visible row count at its current (possibly resized)
    // height, for the scroll clamps.
    fn tree_rows_shown(&self) -> usize {
        assets_panel::visible_rows(self.effective_size(PanelKey::Assets)[1])
    }

    pub(in crate::editor::hook) fn scroll_tree(&mut self, delta: f32, world: &World) {
        if self.picker_open {
            let total = self
                .picker_options(world)
                .map_or(0, |o| o.len())
                .saturating_sub(self.tree_rows_shown());
            self.picker_scroll = scroll_step(self.picker_scroll, delta, total);
            return;
        }
        let max = self
            .tree_rows(world)
            .len()
            .saturating_sub(self.tree_rows_shown());
        self.tree_scroll = scroll_step(self.tree_scroll, delta, max);
        self.row_menu = None;
    }

    fn clamp_tree_scroll(&mut self, world: &World) {
        let max = self
            .tree_rows(world)
            .len()
            .saturating_sub(self.tree_rows_shown());
        self.tree_scroll = self.tree_scroll.min(max);
    }

    // Route a resolved Assets-panel click.
    pub(in crate::editor::hook) fn apply_panel(&mut self, action: PanelAction, world: &mut World) {
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
                let picked = self
                    .picker_options(world)
                    .and_then(|o| o.get(i).map(|p| p.name.clone()));
                if let Some(ty) = picked {
                    // A config singleton edits the world's existing instance if
                    // it has one, else adds it (edit-or-add); a multi-instance
                    // asset always adds a new one.
                    let existing = assets_panel::is_singleton(&ty)
                        .then(|| {
                            let idx = self
                                .entries
                                .iter()
                                .position(|e| entry_type(e) == Some(ty.as_str()))?;
                            self.entries.key_at(idx)
                        })
                        .flatten();
                    let target = match existing {
                        Some(key) => FormTarget::Entry(key),
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
                    if self.toggle_named(&name) {
                        self.open_asset_form(&name, world);
                    } else {
                        self.follow_active(world);
                    }
                } else {
                    self.select_named(&name);
                    self.open_asset_form(&name, world);
                }
                self.pick_last = None;
            }
            PanelAction::ToggleHide(group, index) => {
                if let Some(handle) = self.tree_handle(group, index)
                    && !self.hidden_assets.remove(&handle)
                {
                    self.hidden_assets.insert(handle);
                }
            }
            // The lock is per-session, so it flips the set without touching the
            // entries (unlike Delete, which is an authored edit).
            PanelAction::ToggleLock(group, index) => {
                if let Some(handle) = self.tree_handle(group, index)
                    && !self.locked_assets.remove(&handle)
                {
                    self.locked_assets.insert(handle);
                }
            }
            PanelAction::OpenRowMenu(group, index) => {
                self.row_menu = self.tree_handle(group, index);
            }
            // Generated assets have no line to delete: they are removed by
            // editing whatever produced them.
            PanelAction::RowDelete => {
                if let Some(idx) = self.row_menu.take().and_then(|h| self.handle_index(&h)) {
                    self.remove_entry_at(idx);
                }
                self.clamp_tree_scroll(world);
            }
            PanelAction::RowExport => {
                let name = self.row_menu.take().and_then(|h| self.handle_name(&h));
                if let Some(name) = name {
                    self.console_export(Some(name.as_str()), false);
                }
            }
            PanelAction::CloseOverlays => {
                self.picker_open = false;
                self.row_menu = None;
                self.search_focus = false;
            }
            PanelAction::Consume => {}
        }
    }

    // Open the editing surface for the asset called `name`. A template-derived
    // asset (generated or injected, patched or pristine) opens seeded from its
    // effective args with per-field override state; a plain authored line
    // edits in place; anything else has nothing to edit, so an open form is
    // closed rather than left pointing at the previous asset.
    pub(in crate::editor::hook) fn open_asset_form(&mut self, name: &str, world: &mut World) {
        if let Some(template) = self.form_template_for(name) {
            let target = match self.entry_key_named(name) {
                Some(key) => FormTarget::Entry(key),
                // Pristine: nothing authored yet. The carried entry is unused
                // (the seed comes from the template), but Promote keeps the
                // commit path honest about appending.
                None => FormTarget::Promote(serde_json::json!({
                    "type": template.0, "args": {"$id": name},
                })),
            };
            let (ty, template) = template;
            self.open_form_with(world, ty, target, Some(template));
            return;
        }
        if let Some(key) = self.entry_key_named(name) {
            if let Some(ty) = self
                .entries
                .by_key(key)
                .and_then(entry_type)
                .map(String::from)
            {
                self.open_form(world, ty, FormTarget::Entry(key));
            }
            return;
        }
        self.form.close();
    }

    // Enter blurs the search field (the filter applies live while typing).
    pub(in crate::editor::hook) fn tree_keys(&mut self, _world: &mut World, input: &FrameInput) {
        if self.search_focus && input.pressed_fresh(InputKey::Enter) {
            self.search_focus = false;
        }
    }

    // Unfold the group holding `name` and scroll its row into the window, so a
    // viewport pick is always visible in the tree. A name the tree does not list
    // (a filtered-out match, a mid-edit cook failure) leaves it as-is.
    pub(in crate::editor::hook) fn reveal_in_tree(&mut self, name: &str, world: &World) {
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
        if row < self.tree_scroll || row >= self.tree_scroll + self.tree_rows_shown() {
            let max = rows.len().saturating_sub(self.tree_rows_shown());
            self.tree_scroll = row.saturating_sub(1).min(max);
        }
    }
}

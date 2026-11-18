//! `EditorHook`: the Assets panel's drive. Owns when the cooked tree is rebuilt
//! (an expansion is far too costly per frame, so it is recomputed only when the
//! panel is up and something changed), the fold state, the search field, the "+"
//! type picker, the editor-session hide / lock sets, and the two-way selection
//! sync with the viewport (a row click drives the same selection set
//! `hook/pick.rs` fills; a viewport pick unfolds and scrolls to its row). The
//! rows and their list items are rebuilt only when what they show changes.

use std::collections::BTreeSet;

use concinnity_engine::components::{FrameInput, InputKey, TextField};
use concinnity_engine::ecs::World;

use super::assets_state::{DecorKey, RowsKey};
use crate::editor::asset_handle::AssetHandle;
use crate::editor::entry_list::build_text;
use crate::editor::hook::{EditorHook, FormTarget, entry_type_name, short_status};
use crate::editor::panels::asset_tree::{self, TreeRow};
use crate::editor::panels::assets_panel::{self, AssetsView, PanelAction, PickerOption, RowDecor};
use crate::editor::panels::registry::PanelKey;
use crate::editor::ui::{BuiltPanel, BuiltUi, field};

impl EditorHook {
    // Rebuild the grouped tree from the working entries if it is out of date
    // and a consumer is showing (the Assets tree, the Content grid, or the
    // command palette). Called from the frame drive rather than from each
    // edit, so a burst of edits costs one expansion, not one each.
    pub(in crate::editor::hook) fn refresh_tree_if_needed(&mut self) {
        if !(self.open[PanelKey::Assets]
            || self.open[PanelKey::Content]
            || self.open[PanelKey::Palette])
            || !self.assets.stale
        {
            return;
        }
        self.assets.stale = false;
        match self.cook_working_entries() {
            Ok(loaded) => {
                self.assets.groups = asset_tree::groups_from(&loaded);
                self.assets.tree_rev += 1;
                self.assets.status = None;
                // A group that no longer exists must not stay unfolded.
                let n = self.assets.groups.len();
                self.assets.unfolded.retain(|&g| g < n);
            }
            Err(e) => {
                // A world mid-edit may not cook; the panel says so rather than
                // showing a stale tree.
                self.assets.groups.clear();
                self.assets.tree_rev += 1;
                self.assets.unfolded.clear();
                self.assets.status = Some(short_status(&e));
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
                errs.diagnostics().first().map_or_else(
                    || "the world does not build".to_string(),
                    ToString::to_string,
                )
            },
        )
    }

    // The flattened tree under the search filter. Unfiltered while the
    // picker is open, since the field is narrowing the picker's options
    // rather than the tree.
    pub(in crate::editor::hook) fn tree_rows(&self) -> Vec<TreeRow> {
        let filter = if self.assets.picker_open {
            ""
        } else {
            self.assets.filter.as_str()
        };
        asset_tree::rows(&self.assets.groups, &self.assets.unfolded, filter)
    }

    // The "+" picker's option list, narrowed by the search field and sorted
    // ascending. `None` while the picker is closed.
    pub(in crate::editor::hook) fn picker_options(&self) -> Option<Vec<PickerOption>> {
        if !self.assets.picker_open {
            return None;
        }
        let filter = self.assets.filter.to_lowercase();
        let mut opts: Vec<PickerOption> = assets_panel::picker_types()
            .filter(|t| filter.is_empty() || t.to_lowercase().contains(&filter))
            .map(PickerOption::new)
            .collect();
        opts.sort_by(|a, b| a.name.cmp(&b.name));
        Some(opts)
    }

    // The asset behind a resolved row click, if the tree still lists it.
    fn tree_asset(&self, group: usize, index: usize) -> Option<&asset_tree::TreeAsset> {
        self.assets.groups.get(group)?.assets.get(index)
    }

    // The handle of the asset behind a resolved row click.
    fn tree_handle(&self, group: usize, index: usize) -> Option<AssetHandle> {
        Some(self.handle_for(&self.tree_asset(group, index)?.name))
    }

    // Bring the panel's rows, items, and picker options up to date with the
    // search text, the folds, the cooked tree, and the selection, rebuilding
    // only what changed.
    pub(in crate::editor::hook) fn prepare_assets(&mut self, world: &World, built: &BuiltPanel) {
        if let Some(field) =
            assets_panel::search_field(built).and_then(|id| world.get_by_id::<TextField>(id))
            && field.content != self.assets.filter
        {
            self.assets.filter.clone_from(&field.content);
        }
        self.assets.reveal_now = self.assets.reveal.take();
        let filter = (!self.assets.picker_open).then_some(self.assets.filter.as_str());
        let rows_changed = !self.assets.rows_key.as_ref().is_some_and(|k| {
            k.tree == self.assets.tree_rev
                && k.unfolded == self.assets.unfolded
                && k.filter.as_deref() == filter
        });
        if rows_changed {
            self.assets.rows = self.tree_rows();
            self.assets.menu_at = self.menu_row_index();
            self.assets.rows_key = Some(RowsKey {
                tree: self.assets.tree_rev,
                unfolded: self.assets.unfolded.clone(),
                filter: filter.map(str::to_string),
            });
        }
        let decor_changed = !self.assets.decor.as_ref().is_some_and(|d| {
            d.entries == self.entries.revision()
                && d.selection == self.selection
                && d.hidden == self.hidden_assets
                && d.locked == self.locked_assets
        });
        if rows_changed || decor_changed {
            self.assets.decor = Some(DecorKey {
                entries: self.entries.revision(),
                selection: self.selection.clone(),
                hidden: self.hidden_assets.clone(),
                locked: self.locked_assets.clone(),
            });
            let selected = self.selected_names();
            let names = |set: &BTreeSet<AssetHandle>| -> BTreeSet<String> {
                set.iter().filter_map(|h| self.handle_name(h)).collect()
            };
            let (hidden, locked) = (names(&self.hidden_assets), names(&self.locked_assets));
            self.assets.items = assets_panel::tree_items(&self.assets.rows, |name| RowDecor {
                selected: selected.contains(name),
                hidden: hidden.contains(name),
                locked: locked.contains(name),
            });
            self.assets.items_rev += 1;
        }
        let picker_filter = self
            .assets
            .picker_open
            .then_some(self.assets.filter.as_str());
        if picker_filter != self.assets.picker_filter.as_deref() {
            self.assets.picker = self.picker_options().unwrap_or_default();
            self.assets.picker_filter = picker_filter.map(str::to_string);
            self.assets.picker_rev += 1;
        }
    }

    // Empty the search field.
    pub(in crate::editor::hook) fn clear_assets_search(world: &mut World) {
        let field = world
            .resource::<BuiltUi>()
            .and_then(|b| b.panel(PanelKey::Assets))
            .and_then(assets_panel::search_field);
        field::set_text(world, field, "");
    }

    // Where the open row menu's asset sits among the rows.
    fn menu_row_index(&self) -> Option<usize> {
        let name = self.handle_name(self.assets.row_menu.as_ref()?)?;
        self.assets
            .rows
            .iter()
            .position(|r| matches!(r, TreeRow::Asset { name: n, .. } if *n == name))
    }

    // What the panel shows this frame.
    pub(in crate::editor::hook) fn make_assets_view(&self) -> AssetsView<'_> {
        let menu = self
            .assets
            .row_menu
            .is_some()
            .then_some(self.assets.menu_at)
            .flatten()
            .and_then(|i| self.assets.rows.get(i));
        AssetsView {
            tree_rev: self.assets.items_rev,
            tree: &self.assets.items,
            picker: self
                .assets
                .picker_open
                .then_some((self.assets.picker_rev, self.assets.picker.as_slice())),
            total: self.assets.groups.iter().map(|g| g.assets.len()).sum(),
            error: self.assets.status.as_deref(),
            // Focus is asserted only while frontmost, matching the other
            // panels' guard against fighting for typed keys.
            search_focus: self.assets.search_focus
                && self.panel_order.last() == Some(&PanelKey::Assets),
            menu,
            reveal: self.assets.reveal_now,
        }
    }

    // Take in a press that focused the search field or left it, and a row
    // menu a press off it closed.
    pub(in crate::editor::hook) fn observe_assets(&mut self, world: &World, built: &BuiltPanel) {
        let focused = assets_panel::search_focused(world, built);
        if focused != self.make_assets_view().search_focus {
            self.assets.search_focus = focused;
        }
        if self.assets.row_menu.is_some() && !assets_panel::menu_open(world) {
            self.assets.row_menu = None;
        }
    }

    // The tree row the open row menu belongs to.
    pub(in crate::editor::hook) fn menu_row(&self) -> Option<&TreeRow> {
        self.make_assets_view().menu
    }

    // Route a resolved Assets-panel click.
    pub(in crate::editor::hook) fn apply_panel(&mut self, action: PanelAction, world: &mut World) {
        match action {
            PanelAction::TogglePicker => {
                if self.assets.picker_open {
                    self.assets.picker_open = false;
                } else {
                    // The field keeps whatever was typed: the picker simply
                    // narrows by the same text the tree was filtered by.
                    self.assets.picker_open = true;
                    self.assets.row_menu = None;
                    self.assets.search_focus = true;
                }
            }
            PanelAction::PickOption(i) => {
                let picked = self
                    .picker_options()
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
                                .position(|e| entry_type_name(e) == Some(ty.as_str()))?;
                            self.entries.key_at(idx)
                        })
                        .flatten();
                    let target = match existing {
                        Some(key) => FormTarget::Entry(key),
                        None => FormTarget::New,
                    };
                    self.assets.picker_open = false;
                    self.open_form(world, ty, target);
                }
            }
            PanelAction::ToggleGroup(group) => {
                match self.assets.unfolded.iter().position(|&g| g == group) {
                    Some(i) => {
                        self.assets.unfolded.remove(i);
                    }
                    None => self.assets.unfolded.push(group),
                }
                self.assets.row_menu = None;
            }
            // A row click mirrors a viewport pick: plain replaces the selection
            // and opens the asset's editing surface; a modified one toggles
            // membership, the form following the active member.
            PanelAction::SelectRow(group, index) => {
                let Some(name) = self.tree_asset(group, index).map(|a| a.name.clone()) else {
                    return;
                };
                self.assets.row_menu = None;
                self.select_named(&name);
                self.open_asset_form(&name, world);
                self.pick_last = None;
            }
            PanelAction::ExtendRow(group, index) => {
                let Some(name) = self.tree_asset(group, index).map(|a| a.name.clone()) else {
                    return;
                };
                self.assets.row_menu = None;
                if self.toggle_named(&name) {
                    self.open_asset_form(&name, world);
                } else {
                    self.follow_active(world);
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
                self.assets.row_menu = self.tree_handle(group, index);
                self.assets.menu_at = self.assets.rows.iter().position(|r| {
                    matches!(r, TreeRow::Asset { group: g, index: i, .. } if (*g, *i) == (group, index))
                });
            }
            // Generated assets have no line to delete: they are removed by
            // editing whatever produced them.
            PanelAction::RowDelete => {
                if let Some(idx) = self
                    .assets
                    .row_menu
                    .take()
                    .and_then(|h| self.handle_index(&h))
                {
                    self.remove_entry_at(idx);
                }
            }
            PanelAction::RowExport => {
                let name = self
                    .assets
                    .row_menu
                    .take()
                    .and_then(|h| self.handle_name(&h));
                if let Some(name) = name {
                    self.console_export(Some(name.as_str()), false);
                }
            }
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
                .and_then(entry_type_name)
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
        if self.assets.search_focus && input.pressed_fresh(InputKey::Enter) {
            self.assets.search_focus = false;
        }
    }

    // Unfold the group holding `name` and scroll its row into view, so a
    // viewport pick is always visible in the tree. A name the tree does not
    // list (a filtered-out match, a mid-edit cook failure) leaves it as-is.
    pub(in crate::editor::hook) fn reveal_in_tree(&mut self, name: &str) {
        if !self.open[PanelKey::Assets] {
            return;
        }
        if let Some(group) = self
            .assets
            .groups
            .iter()
            .position(|g| g.assets.iter().any(|a| a.name == name))
            && !self.assets.unfolded.contains(&group)
        {
            self.assets.unfolded.push(group);
        }
        self.assets.reveal = self
            .tree_rows()
            .iter()
            .position(|r| matches!(r, TreeRow::Asset { name: n, .. } if n == name));
    }
}

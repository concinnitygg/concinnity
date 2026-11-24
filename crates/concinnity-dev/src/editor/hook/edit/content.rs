//! `EditorHook`: the Content panel's data assembly and actions. Items come from
//! the cooked asset tree (authored + expansion assets), narrowed to the visual
//! types, filtered by the panel's type button, and ranked by the search query;
//! cells bind baked thumbnails through `editor/thumbs.rs` and fall back to
//! typed icon glyphs. They are rebuilt only when what they show changes.

use std::sync::Arc;

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_engine::components::TextField;
use concinnity_engine::ecs::World;

use super::content_state::CellsKey;
use crate::editor::hook::EditorHook;
use crate::editor::panels::content_panel::{self, ContentAction, ContentView};
use crate::editor::panels::registry::PanelKey;
use crate::editor::ui::BuiltPanel;
use crate::editor::{filter, thumbs};

// The asset types the grid shows, the type button's cycle order. "All" leads.
pub(in crate::editor::hook) const VISUAL_TYPES: [RegisteredType; 7] = [
    RegisteredType::EnvironmentMap,
    RegisteredType::Material,
    RegisteredType::Mesh,
    RegisteredType::Model,
    RegisteredType::Prefab,
    RegisteredType::ProceduralMesh,
    RegisteredType::Texture,
];

impl EditorHook {
    // The type button's caption for the current cycle position.
    pub(in crate::editor::hook) fn content_type_caption(&self) -> &'static str {
        match self.content.type_chip {
            0 => "All",
            i => VISUAL_TYPES[(i - 1).min(VISUAL_TYPES.len() - 1)].as_str(),
        }
    }

    pub(in crate::editor::hook) fn cycle_content_type(&mut self) {
        self.content.type_chip = (self.content.type_chip + 1) % (VISUAL_TYPES.len() + 1);
    }

    // Every visual asset under the current type filter and search query, best
    // match first: `(name, type)` pairs over the cooked tree.
    pub(in crate::editor::hook) fn content_items(&self) -> Vec<(String, RegisteredType)> {
        let mut items: Vec<(String, RegisteredType)> = self
            .assets
            .groups
            .iter()
            .flat_map(|g| g.assets.iter())
            .filter(|a| VISUAL_TYPES.contains(&a.asset_type))
            .filter(|a| match self.content.type_chip {
                0 => true,
                i => a.asset_type == VISUAL_TYPES[i - 1],
            })
            .map(|a| (a.name.clone(), a.asset_type))
            .collect();
        items.sort_by(|a, b| (&a.0, a.1.as_str()).cmp(&(&b.0, b.1.as_str())));
        items.dedup();
        let query = &self.content.filter;
        if query.trim().is_empty() {
            return items;
        }
        filter::ranked(items.iter().map(|(n, t)| (n.as_str(), t.as_str())), query)
            .into_iter()
            .map(|i| items[i].clone())
            .collect()
    }

    // Bring the cells up to date with the search text, the type button, the
    // cooked tree, the thumbnails, and the selection.
    pub(in crate::editor::hook) fn prepare_content(&mut self, world: &World, built: &BuiltPanel) {
        if let Some(field) =
            content_panel::search_field(built).and_then(|id| world.get_by_id::<TextField>(id))
            && field.content != self.content.filter
        {
            self.content.filter.clone_from(&field.content);
        }
        self.refresh_content_cells();
    }

    // Rebuild the cells when what they show changed.
    pub(in crate::editor::hook) fn refresh_content_cells(&mut self) {
        let thumbs = thumbs::injected();
        let current = self.content.key.as_ref().is_some_and(|k| {
            k.tree == self.assets.tree_rev
                && k.type_chip == self.content.type_chip
                && k.filter == self.content.filter
                && Arc::ptr_eq(&k.thumbs, &thumbs)
                && k.entries == self.entries.revision()
                && k.selection == self.selection
        });
        if current {
            return;
        }
        self.content.key = Some(CellsKey {
            tree: self.assets.tree_rev,
            type_chip: self.content.type_chip,
            filter: self.content.filter.clone(),
            thumbs: Arc::clone(&thumbs),
            entries: self.entries.revision(),
            selection: self.selection.clone(),
        });
        self.content.items = self.content_items();
        let selected = self.selected_names();
        self.content.cells = self
            .content
            .items
            .iter()
            .map(|(name, ty)| {
                content_panel::cell(name, *ty, thumbs.get(name), selected.contains(name))
            })
            .collect();
        self.content.cells_rev += 1;
    }

    // What the panel shows this frame.
    pub(in crate::editor::hook) fn make_content_view(&self) -> ContentView<'_> {
        ContentView {
            cells_rev: self.content.cells_rev,
            cells: &self.content.cells,
            type_caption: self.content_type_caption(),
            search_focus: self.content.search_focus
                && self.panel_order.last() == Some(&PanelKey::Content),
        }
    }

    // Take in a press that focused the search field, or left it.
    pub(in crate::editor::hook) fn observe_content(&mut self, world: &World, built: &BuiltPanel) {
        let focused = content_panel::search_focused(world, built);
        if focused != self.make_content_view().search_focus {
            self.content.search_focus = focused;
        }
    }

    pub(in crate::editor::hook) fn apply_content_action(&mut self, action: ContentAction) {
        let item = |i: usize| self.content.items.get(i).cloned();
        match action {
            ContentAction::CycleType => {
                self.content.search_focus = false;
                self.cycle_content_type();
            }
            ContentAction::Select(i) => {
                if let Some((name, _)) = item(i) {
                    self.select_named(&name);
                }
            }
            ContentAction::Extend(i) => {
                if let Some((name, _)) = item(i) {
                    self.toggle_named(&name);
                }
            }
            ContentAction::Drag(i) => {
                if let Some((name, ty)) = item(i) {
                    self.start_content_drag(name, ty);
                }
            }
        }
    }
}

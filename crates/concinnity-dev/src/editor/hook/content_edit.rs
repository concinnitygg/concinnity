// src/editor/hook/content_edit.rs
//
// EditorHook: the Content panel's data assembly and click handling. Items come
// from the cooked asset tree (authored + expansion assets), narrowed to the
// visual types, filtered by the panel's type chip, and ranked by the search
// query; cells bind baked thumbnails through `editor/thumbs.rs` and fall back
// to typed icon chips.

use super::*;
use crate::editor::content_panel::{self, ContentAction};
use crate::editor::{filter, thumbs};

// The asset types the grid shows, the type-chip cycle order. "All" leads.
pub(super) const VISUAL_TYPES: [&str; 7] = [
    "EnvironmentMap",
    "Material",
    "Mesh",
    "Model",
    "Prefab",
    "ProceduralMesh",
    "Texture",
];

impl EditorHook {
    // The type chip's caption for the current cycle position.
    pub(super) fn content_type_caption(&self) -> &'static str {
        match self.content_type {
            0 => "All",
            i => VISUAL_TYPES[(i - 1).min(VISUAL_TYPES.len() - 1)],
        }
    }

    pub(super) fn cycle_content_type(&mut self) {
        self.content_type = (self.content_type + 1) % (VISUAL_TYPES.len() + 1);
        self.content_scroll = 0;
    }

    // Every visual asset under the current type filter and search query, best
    // match first: `(name, type)` pairs over the cooked tree.
    pub(super) fn content_items(&self, world: &World) -> Vec<(String, String)> {
        let mut items: Vec<(String, String)> = self
            .tree_groups
            .iter()
            .flat_map(|g| g.assets.iter())
            .filter(|a| VISUAL_TYPES.contains(&a.asset_type.as_str()))
            .filter(|a| match self.content_type {
                0 => true,
                i => a.asset_type == VISUAL_TYPES[i - 1],
            })
            .map(|a| (a.name.clone(), a.asset_type.clone()))
            .collect();
        items.sort();
        items.dedup();
        let query = widget::field_text(world, content_panel::SEARCH_INPUT);
        if query.trim().is_empty() {
            return items;
        }
        filter::ranked(items.iter().map(|(n, t)| (n.as_str(), t.as_str())), &query)
            .into_iter()
            .map(|i| items[i].clone())
            .collect()
    }

    // The visible cell window plus the filtered total, for the panel draw.
    pub(super) fn content_cells(&self, world: &World) -> (Vec<content_panel::CellView>, usize) {
        let items = self.content_items(world);
        let thumbs = thumbs::injected();
        let first = self.content_scroll * content_panel::COLS;
        let cells = items
            .iter()
            .skip(first)
            .take(content_panel::CELLS)
            .map(|(name, ty)| content_panel::CellView {
                thumb: thumbs.get(name),
                selected: self.selection.contains(name),
                name: name.clone(),
                asset_type: ty.clone(),
            })
            .collect();
        (cells, items.len())
    }

    pub(super) fn scroll_content(&mut self, delta: f32, world: &World) {
        let total = self.content_items(world).len();
        let rows = total.div_ceil(content_panel::COLS);
        let max = rows.saturating_sub(content_panel::GRID_ROWS);
        self.content_scroll = scroll_step(self.content_scroll, delta, max);
    }

    pub(super) fn apply_content_action(
        &mut self,
        action: ContentAction,
        world: &mut World,
        at: [f32; 2],
    ) {
        match action {
            ContentAction::FocusSearch => {
                self.content_search_focus = true;
                let text = widget::field_text(world, content_panel::SEARCH_INPUT);
                widget::focus_field_with(world, content_panel::SEARCH_INPUT, &text);
            }
            ContentAction::CycleType => {
                self.content_search_focus = false;
                self.cycle_content_type();
            }
            ContentAction::SelectCell(slot) => {
                self.content_search_focus = false;
                let items = self.content_items(world);
                let index = self.content_scroll * content_panel::COLS + slot;
                if let Some((name, ty)) = items.get(index) {
                    if self.shift_held {
                        self.selection.toggle(name.clone());
                    } else {
                        self.selection.replace(name.clone());
                        // A plain press may also become a drag-out placement.
                        self.arm_content_drag(name.clone(), ty.clone(), at);
                    }
                }
            }
            ContentAction::Consume => self.content_search_focus = false,
        }
    }
}

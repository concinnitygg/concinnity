//! EditorHook: duplicate the selection in place (Ctrl+D, /dup). Each selected
//! authored entry is cloned with all its args -- position included, so the copy
//! sits exactly on the original until it is dragged away -- under a unique
//! `$id`, or anonymous when the original is. The copies become the new
//! selection (ready to move), and the whole batch commits as ONE undo step.

use concinnity_cook::authoring::world::set_entry_id;

use super::{EditorHook, declared_id, entry_type};
use crate::editor::asset_handle::AssetHandle;
use crate::editor::panels::assets_panel;

impl EditorHook {
    // Duplicate every eligible selection member; the number of copies made.
    // Skipped: generated assets (no authored entry to clone) and singleton
    // types (a second instance is a cook error).
    pub(super) fn duplicate_selection(&mut self) -> usize {
        let mut copies = Vec::new();
        for handle in self.selected() {
            let Some(idx) = self.handle_index(&handle) else {
                continue;
            };
            let Some(ty) = entry_type(&self.entries[idx]) else {
                continue;
            };
            if assets_panel::is_singleton(ty) {
                continue;
            }
            // An anonymous entry's copy is anonymous too; a declared id is
            // made unique.
            let mut clone = self.entries[idx].clone();
            if let Some(base) = declared_id(&self.entries[idx]) {
                let new_name = self.unique_from(base);
                set_entry_id(&mut clone, &new_name);
            }
            copies.push(AssetHandle::Entry(self.entries.push(clone)));
        }
        if copies.is_empty() {
            return 0;
        }
        self.mark_changed();
        let made = copies.len();
        self.selection.set(copies);
        made
    }
}

//! EditorHook: duplicate the selection in place (Ctrl+D, /dup). Each selected
//! authored entry is cloned with all its args -- position included, so the copy
//! sits exactly on the original until it is dragged away -- under a unique
//! `$id`, or anonymous when the original is. A file the entry's type owns is
//! copied beside the original and the clone names the copy, so an edit to one
//! never reaches the other. The copies become the new selection (ready to
//! move), and the whole batch commits as ONE undo step; the copied files stay
//! when it is undone.

use concinnity_cook::authoring::world::set_entry_id;

use super::{EditorHook, declared_id, entry_type};
use crate::editor::asset_handle::AssetHandle;
use crate::editor::notify;
use crate::editor::owned_files::{self, Copied};
use crate::editor::panels::assets_panel;

impl EditorHook {
    pub(super) fn duplicate_selection(&mut self) -> usize {
        let handles = self.selected();
        self.duplicate(handles)
    }

    // Duplicate every eligible one of `handles`; the number of copies made.
    // Skipped: generated assets (no authored entry to clone), singleton types
    // (a second instance is a cook error), and a type at its world limit.
    pub(super) fn duplicate(&mut self, handles: Vec<AssetHandle>) -> usize {
        let assets = crate::project::assets_dir();
        let resolve =
            |d: &str| concinnity_host::store::source::resolve_source_path(d, assets.as_deref());
        let find = |name: &str| {
            assets
                .as_deref()
                .is_some_and(|dir| concinnity_host::store::source::find_in(dir, name).is_some())
        };
        let mut copies = Vec::new();
        let mut copied: Vec<Copied> = Vec::new();
        let mut refused = Vec::new();
        for handle in handles {
            let Some(idx) = self.handle_index(&handle) else {
                continue;
            };
            let Some(ty) = entry_type(&self.entries[idx]).map(str::to_string) else {
                continue;
            };
            if assets_panel::is_singleton(&ty) {
                continue;
            }
            if let Some(reason) = assets_panel::add_refused(&ty, &self.entries) {
                refused.push(reason);
                continue;
            }
            // An anonymous entry's copy is anonymous too; a declared id is
            // made unique.
            let mut clone = self.entries[idx].clone();
            if let Some(base) = declared_id(&self.entries[idx]) {
                let new_name = self.unique_from(base);
                set_entry_id(&mut clone, &new_name);
            }
            copied.extend(owned_files::copy_owned(&mut clone, resolve, find));
            copies.push(AssetHandle::Entry(self.entries.push(clone)));
        }
        if let Some(reason) = refused.first() {
            self.notifier
                .push(notify::Level::Error, &format!("Not duplicated: {reason}"));
        }
        if let Some(message) = owned_files::copied_message(&copied) {
            let failed = copied.iter().any(|c| matches!(c, Copied::Failed { .. }));
            match failed {
                true => self
                    .notifier
                    .error_with(&message, notify::Action::OpenConsole),
                false => self.notifier.success(&message),
            }
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

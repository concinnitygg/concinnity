// src/editor/hook/duplicate.rs
//
// EditorHook: duplicate the selection in place (Ctrl+D, /dup). Each selected
// authored entry is cloned with all its args -- position included, so the copy
// sits exactly on the original until it is dragged away -- under a unique
// name. The copies become the new selection (ready to move), and the whole
// batch commits as ONE undo step.

use super::*;

impl EditorHook {
    // Duplicate every eligible selection member; the number of copies made.
    // Skipped: generated assets (no authored entry to clone) and singleton
    // types (a second instance is a cook error).
    pub(super) fn duplicate_selection(&mut self) -> usize {
        let names: Vec<String> = self.selection.iter().map(String::from).collect();
        let mut copies = Vec::new();
        for name in &names {
            let Some(idx) = self
                .entries
                .iter()
                .position(|e| entry_name(e) == Some(name))
            else {
                continue;
            };
            let Some(ty) = entry_type(&self.entries[idx]) else {
                continue;
            };
            if panel::is_singleton(ty) {
                continue;
            }
            let mut clone = self.entries[idx].clone();
            let new_name = self.unique_from(name);
            if let Some(obj) = clone.as_object_mut() {
                obj.insert(
                    "name".to_string(),
                    serde_json::Value::String(new_name.clone()),
                );
            }
            self.entries.push(clone);
            copies.push(new_name);
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

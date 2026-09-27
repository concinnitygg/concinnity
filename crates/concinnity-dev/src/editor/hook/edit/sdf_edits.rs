//! EditorHook: deleting a distance field from the Shaders panel. Its menu's
//! Delete asks first, with the box that also deletes the file when nothing the
//! delete leaves still reads it. The delete removes every volume reading the
//! field that the editor can remove, as one undoable world edit, and only then
//! the file. A source panel showing the field closes first, asking over
//! unsaved edits.

use concinnity_cook::authoring::world::find_entry;

use super::shader_edits::{button, cancel};
use crate::editor::hook::{EditorHook, FormTarget};
use crate::editor::modal::Action;
use crate::editor::notify;
use crate::editor::panels::sdf_field_edit::{self, FieldDelete};
use crate::editor::panels::sdf_field_list::FieldDecl;
use crate::editor::panels::shader_edit::ShaderEdit;

impl EditorHook {
    // Field `i`'s Delete: say which volumes go, with the file box.
    pub(in crate::editor::hook) fn confirm_delete_field(&mut self, i: usize) {
        let Some(field) = self.declared_fields().into_iter().nth(i) else {
            return;
        };
        let delete = self.plan_field_delete(&field);
        if delete.removed.is_empty() {
            self.notifier.push(
                notify::Level::Error,
                &format!(
                    "Every volume reading {} is included from another file; delete them there",
                    field.declared
                ),
            );
            return;
        }
        let check = sdf_field_edit::files_check(&field, &delete);
        let action = Action::DeleteSdfField(field.path.clone());
        self.open_modal_with_check(
            &sdf_field_edit::delete_message(&field, &delete),
            vec![cancel(), button("Delete", true, action)],
            check,
        );
    }

    // The delete dialog's Delete, with its box's state as `files`.
    pub(in crate::editor::hook) fn delete_sdf_field(&mut self, path: &str, files: bool) {
        self.edit_shader(ShaderEdit::DeleteField {
            path: path.to_string(),
            files,
        });
    }

    fn plan_field_delete(&mut self, field: &FieldDecl) -> FieldDelete {
        let shaders = self.declared_shaders();
        let entries = &self.entries;
        let read_only =
            |name: &str| find_entry(entries, name).is_some_and(|i| entries.is_read_only(i));
        sdf_field_edit::plan(field, read_only, &shaders)
    }

    // Remove the volumes reading the field at `path` in one edit, then, with
    // `files` and nothing left reading it, the file. Undo restores the
    // volumes; a deleted file stays deleted.
    pub(in crate::editor::hook) fn apply_delete_field(&mut self, path: &str, files: bool) {
        let Some(field) = self.declared_fields().into_iter().find(|f| f.path == path) else {
            return;
        };
        let delete = self.plan_field_delete(&field);
        let mut at: Vec<usize> = delete
            .removed
            .iter()
            .filter_map(|name| find_entry(&self.entries, name))
            .collect();
        if at.is_empty() {
            return;
        }
        at.sort_unstable();
        for idx in at.into_iter().rev() {
            if self
                .entries
                .key_at(idx)
                .is_some_and(|key| self.form.target == FormTarget::Entry(key))
            {
                self.form.close();
            }
            self.entries.remove(idx);
        }
        if !self.commit() {
            return;
        }
        let file = (files && delete.readers.is_empty())
            .then(|| std::fs::remove_file(&field.path).map_err(|e| e.to_string()));
        let failed = matches!(file, Some(Err(_)));
        let message = sdf_field_edit::deleted_message(&field, &delete.removed, file);
        match failed {
            false => self.notifier.success(&message),
            true => self
                .notifier
                .error_with(&message, notify::Action::OpenConsole),
        }
    }
}

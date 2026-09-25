//! EditorHook: the Shaders panel's edits. "+ New Shader" asks for a name and
//! writes a starter fragment file; a heading's menu renames its Shader (its
//! `$id` and every reference to it, never its files) or deletes it (its entry,
//! the references to it, and, when asked, the files no other Shader reads);
//! "+ Add vertex file" writes a starter vertex file, and a vertex file's menu
//! stops declaring it. Each is one undoable world edit. An edit that takes away
//! the file the source panel shows closes it first, asking over unsaved edits.

use concinnity_cook::authoring::refs::retarget_references;
use concinnity_cook::authoring::world::{find_entry, set_entry_id};
use concinnity_core::components::ShaderStage;
use concinnity_core::ecs::World;
use std::path::{Path, PathBuf};

use crate::editor::hook::EditorHook;
use crate::editor::modal::{self, Action, Button};
use crate::editor::notify;
use crate::editor::panels::shader_edit::{self, ShaderEdit};
use crate::editor::panels::shader_list;
use crate::editor::panels::shader_source::{self, Leave, SourceKey, same_file};
use crate::editor::widget;

const SHADER: &str = "Shader";

fn button(label: &str, danger: bool, action: Action) -> Button {
    Button {
        label: label.to_string(),
        danger,
        action,
    }
}

fn cancel() -> Button {
    button("Cancel", false, Action::Dismiss)
}

impl EditorHook {
    // "+ New Shader": ask what to call it. `rejected` says why the last name
    // was turned down.
    pub(in crate::editor::hook) fn prompt_new_shader(&mut self, rejected: Option<String>) {
        if !self.can_add_shader() {
            return;
        }
        let message = rejected.unwrap_or_else(|| "Name the new Shader.".to_string());
        self.open_prompt(
            &message,
            vec![cancel(), button("Create", false, Action::NameShader)],
        );
    }

    fn can_add_shader(&mut self) -> bool {
        let can = shader_list::can_add_shader(shader_list::shader_count(&self.entries));
        if !can {
            self.notifier.push(
                notify::Level::Error,
                &format!("No Shader added: {}", shader_list::limit_reason()),
            );
        }
        can
    }

    // Add a Shader named `typed` (made unique) reading a starter fragment file
    // written under the project's assets, then open the file. The entry is a
    // world edit like any other; the file is written now, since the build
    // reads Shader files from disk.
    pub(in crate::editor::hook) fn create_shader(&mut self, typed: &str, world: &mut World) {
        let name = match shader_edit::check_name(typed) {
            Ok(name) => self.unique_from(&name),
            Err(reason) => {
                self.prompt_new_shader(Some(reason));
                widget::seed_field(world, modal::NAME_INPUT, typed);
                return;
            }
        };
        if !self.can_add_shader() {
            return;
        }
        let (path, declared) = new_shader_file(&shader_edit::file_stem(&name));
        if !self.write_starter(&path, shader_source::STARTER_SHADER) {
            return;
        }
        self.entries.push(serde_json::json!({
            "type": SHADER, "args": { "$id": name, "fragment": declared },
        }));
        self.mark_changed();
        self.notifier
            .success(&format!("Added Shader '{name}' ({declared})"));
        self.open_shader_file(SourceKey {
            shader: name,
            stage: ShaderStage::Fragment,
        });
    }

    // Write a new Shader file, or say why it could not be written.
    fn write_starter(&self, path: &Path, text: &str) -> bool {
        let written = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(path, text));
        if let Err(e) = &written {
            self.notifier.error_with(
                &format!("Could not write {}: {e}", path.display()),
                notify::Action::OpenConsole,
            );
        }
        written.is_ok()
    }

    // A heading's Rename: ask for the new name, seeded with the current one.
    pub(in crate::editor::hook) fn prompt_rename_shader(&mut self, i: usize, world: &mut World) {
        let Some(shader) = self.declared_shaders().into_iter().nth(i) else {
            return;
        };
        let message = format!("Rename the Shader '{}'.", shader.name);
        self.reprompt_rename(&shader.name, &message, &shader.name, world);
    }

    fn reprompt_rename(&mut self, name: &str, message: &str, seed: &str, world: &mut World) {
        self.open_prompt(
            message,
            vec![
                cancel(),
                button("Rename", false, Action::RenameShader(name.to_string())),
            ],
        );
        widget::seed_field(world, modal::NAME_INPUT, seed);
    }

    // Rename Shader `old` to `typed` (made unique against the other entries):
    // its `$id` and every reference to it, in one edit. Its files keep their
    // names, and a source panel open on it follows the new name.
    pub(in crate::editor::hook) fn rename_shader(
        &mut self,
        old: &str,
        typed: &str,
        world: &mut World,
    ) {
        let Some(idx) = find_entry(&self.entries, old) else {
            return;
        };
        let base = match shader_edit::check_name(typed) {
            Ok(base) => base,
            Err(reason) => return self.reprompt_rename(old, &reason, typed, world),
        };
        let Some(new) = self
            .entries
            .key_at(idx)
            .and_then(|key| self.finalize_rename(&base, key))
        else {
            return;
        };
        if new == old {
            return;
        }
        set_entry_id(&mut self.entries[idx], &new);
        let moved = self.retarget_shader(old, Some(&new));
        if !self.commit() {
            return;
        }
        if let Some(src) = self.shaders.source.as_mut()
            && src.key.shader == old
        {
            src.key.shader = new.clone();
        }
        let followers = match moved {
            0 => String::new(),
            1 => " and the 1 reference to it".to_string(),
            n => format!(" and the {n} references to it"),
        };
        self.notifier
            .success(&format!("Renamed Shader '{old}' to '{new}'{followers}"));
    }

    // Point every reference to Shader `from` at `to`, or drop it with `None`.
    fn retarget_shader(&mut self, from: &str, to: Option<&str>) -> usize {
        self.entries
            .iter_mut()
            .map(|e| retarget_references(e, SHADER, from, to))
            .sum()
    }

    // A heading's Delete: say what it does, with the box that also deletes
    // the files only this Shader reads.
    pub(in crate::editor::hook) fn confirm_delete_shader(&mut self, i: usize) {
        let shaders = self.declared_shaders();
        let Some(shader) = shaders.get(i) else {
            return;
        };
        let check = shader_edit::files_check(&shader_edit::files_of(&shaders, i));
        let delete = Action::DeleteShader(shader.name.clone());
        self.open_modal_with_check(
            &shader_edit::delete_message(&shaders, i),
            vec![cancel(), button("Delete", true, delete)],
            check,
        );
    }

    // The delete dialog's Delete, with its box's state as `files`.
    pub(in crate::editor::hook) fn delete_shader(&mut self, name: &str, files: bool) {
        self.edit_shader(ShaderEdit::Delete {
            name: name.to_string(),
            files,
        });
    }

    // A vertex file's Remove.
    pub(in crate::editor::hook) fn remove_vertex_file(&mut self, name: &str) {
        self.edit_shader(ShaderEdit::RemoveVertex(name.to_string()));
    }

    // Make `edit`, closing the source panel first when it shows a file the
    // edit takes away; Cancel on the unsaved-changes question abandons it.
    fn edit_shader(&mut self, edit: ShaderEdit) {
        match &self.shaders.source {
            Some(src) if edit.closes(&src.key) => self.leave_shader_source(Leave::Edit(edit)),
            _ => self.apply_shader_edit(edit),
        }
    }

    pub(in crate::editor::hook) fn apply_shader_edit(&mut self, edit: ShaderEdit) {
        match edit {
            ShaderEdit::RemoveVertex(name) => self.apply_remove_vertex(&name),
            ShaderEdit::Delete { name, files } => self.apply_delete(&name, files),
        }
    }

    // Remove Shader `name` and every reference to it in one edit (a Material
    // naming it falls back to the default), then, with `files`, delete the
    // files no other Shader reads. Undo restores the entry and the
    // references; deleted files stay deleted.
    fn apply_delete(&mut self, name: &str, files: bool) {
        let shaders = self.declared_shaders();
        let Some(i) = shaders.iter().position(|s| s.name == name) else {
            return;
        };
        let own = match files {
            true => shader_edit::files_of(&shaders, i).own,
            false => Vec::new(),
        };
        let Some(idx) = find_entry(&self.entries, name) else {
            return;
        };
        let before = self.entries.len();
        self.retarget_shader(name, None);
        self.remove_entry_at(idx);
        if self.entries.len() == before {
            return;
        }
        let (mut deleted, mut failed) = (Vec::new(), Vec::new());
        for path in own {
            match std::fs::remove_file(&path) {
                Ok(()) => deleted.push(path),
                Err(e) => failed.push(format!("{path}: {e}")),
            }
        }
        let message = shader_edit::deleted_message(name, &deleted, &failed);
        match failed.is_empty() {
            true => self.notifier.success(&message),
            false => self
                .notifier
                .error_with(&message, notify::Action::OpenConsole),
        }
    }

    // Stop declaring Shader `name`'s vertex file; the file stays on disk.
    fn apply_remove_vertex(&mut self, name: &str) {
        let Some(idx) = find_entry(&self.entries, name) else {
            return;
        };
        let stage = shader_source::stage_name(ShaderStage::Vertex);
        let Some(removed) = self.entries[idx]
            .get_mut("args")
            .and_then(|a| a.as_object_mut())
            .and_then(|a| a.remove(stage))
        else {
            return;
        };
        if self.commit() {
            let declared = removed.as_str().unwrap_or_default();
            self.notifier
                .success(&shader_edit::removed_vertex_message(name, declared));
        }
    }

    // "+ Add vertex file": write a starter vertex file that projects as the
    // engine does, declare it on declared Shader `i`, and open it.
    pub(in crate::editor::hook) fn add_vertex_file(&mut self, i: usize) {
        let Some(shader) = self.declared_shaders().into_iter().nth(i) else {
            return;
        };
        let Some(idx) = find_entry(&self.entries, &shader.name) else {
            return;
        };
        if shader.file(ShaderStage::Vertex).is_some() {
            return;
        }
        let (path, declared) = new_shader_file(&shader_edit::vertex_stem(&shader.name));
        // A read-only entry refuses the edit below, so it gets no file either.
        if !self.entries.is_read_only(idx)
            && !self.write_starter(&path, shader_source::STARTER_VERTEX)
        {
            return;
        }
        let stage = shader_source::stage_name(ShaderStage::Vertex);
        if let Some(args) = self.entries[idx]
            .get_mut("args")
            .and_then(|a| a.as_object_mut())
        {
            args.insert(stage.to_string(), serde_json::Value::String(declared));
        }
        if !self.commit() {
            return;
        }
        self.notifier
            .success(&format!("Added a vertex file to '{}'", shader.name));
        self.open_shader_file(SourceKey {
            shader: shader.name,
            stage: ShaderStage::Vertex,
        });
    }

    // After an undo or redo, a source panel whose Shader is no longer declared
    // follows the Shader that declares its file under another name: the rename
    // the jump undid or redid.
    pub(in crate::editor::hook) fn follow_shader_source(&mut self) {
        let Some(src) = &self.shaders.source else {
            return;
        };
        let (open, stage, path) = (src.key.shader.clone(), src.key.stage, src.path.clone());
        let shaders = self.declared_shaders();
        if shaders.iter().any(|s| s.name == open) {
            return;
        }
        let renamed = shaders
            .into_iter()
            .find(|s| s.file(stage).is_some_and(|f| same_file(&f.path, &path)));
        if let (Some(shader), Some(src)) = (renamed, self.shaders.source.as_mut()) {
            src.key.shader = shader.name;
        }
    }
}

// Where a new Shader file `stem` goes under the project's assets, numbered
// past files already there, and how a world line declares it.
fn new_shader_file(stem: &str) -> (PathBuf, String) {
    let dir = crate::project::assets_dir().unwrap_or_else(|| PathBuf::from("assets"));
    let path = shader_source::starter_path(&dir, stem, Path::exists);
    let root = std::env::current_dir().unwrap_or_default();
    let declared = shader_source::declared_form(&path, &root);
    (path, declared)
}

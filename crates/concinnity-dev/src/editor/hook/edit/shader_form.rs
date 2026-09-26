//! EditorHook: the Shader form's extras (`panels/shader_form.rs`) on the
//! generic add / edit form. Creating writes each file from its starter once
//! the entry edit is known to stand; editing adds or drops the vertex file,
//! renames with every reference following, and in both the Materials and the
//! world default ride the same undo step.

use concinnity_cook::authoring::world::entry_handles;
use concinnity_core::components::ShaderStage;

use super::shader_edits::new_shader_file;
use crate::editor::entry_list::EntryList;
use crate::editor::hook::EditorHook;
use crate::editor::hook::edits::references_note;
use crate::editor::hook::form_extras::{Committed, ExtrasCx, FormExtras, NewFile};
use crate::editor::hook::form_state::FormTarget;
use crate::editor::panels::form_extras::ExtraRow;
use crate::editor::panels::shader_edit::{file_stem, vertex_stem};
use crate::editor::panels::shader_form::{Mode, Paths, ShaderForm};
use crate::editor::panels::shader_source::{self, Leave, SourceKey};
use crate::editor::panels::shader_templates;

const SHADER: &str = "Shader";

#[derive(Debug)]
pub(in crate::editor::hook) struct ShaderExtras {
    form: ShaderForm,
    // The Shader's name when the form opened, for an edit.
    original: Option<String>,
}

impl ShaderExtras {
    pub(in crate::editor::hook) fn open(entries: &EntryList, target: &FormTarget) -> Self {
        let editing = target.entry().and_then(|k| entries.index_of(k));
        Self {
            form: ShaderForm::open(entries, editing),
            original: editing.and_then(|i| entry_handles(entries).swap_remove(i)),
        }
    }

    // The files the rows show and a commit writes, under `name`: a new
    // Shader's planned paths, or an edited one's declared paths with the vertex
    // file it would add.
    fn paths(&self, cx: &ExtrasCx) -> (Paths, Vec<NewFile>) {
        let edited = cx
            .target
            .entry()
            .and_then(|k| cx.entries.by_key(k))
            .and_then(|e| e.get("args"));
        let declared = |stage: ShaderStage| {
            edited
                .and_then(|a| a.get(shader_source::stage_name(stage)))
                .and_then(|v| v.as_str())
                .map(str::to_string)
        };
        let mut files = Vec::new();
        let mut plan = |stem: String, stage: ShaderStage, starter: usize| {
            let (path, declared) = new_shader_file(&stem);
            files.push(NewFile {
                path,
                text: shader_templates::text(stage, starter).to_string(),
            });
            declared
        };
        let fragment = match self.form.mode {
            Mode::New => plan(
                file_stem(cx.name),
                ShaderStage::Fragment,
                self.form.fragment_starter,
            ),
            Mode::Edit { .. } => declared(ShaderStage::Fragment).unwrap_or_default(),
        };
        let vertex = match (self.form.vertex, declared(ShaderStage::Vertex)) {
            (false, _) => None,
            (true, Some(existing)) if !self.form.is_new() => Some(existing),
            (true, _) => Some(plan(
                vertex_stem(cx.name),
                ShaderStage::Vertex,
                self.form.vertex_starter,
            )),
        };
        (Paths { fragment, vertex }, files)
    }

    fn removes_vertex(&self) -> bool {
        matches!(
            self.form.mode,
            Mode::Edit {
                had_vertex: true,
                ..
            }
        ) && !self.form.vertex
    }

    fn adds_vertex(&self) -> bool {
        matches!(
            self.form.mode,
            Mode::Edit {
                had_vertex: false,
                ..
            }
        ) && self.form.vertex
    }
}

// The Shaders `entries` declare, by name, in order.
fn shader_names(entries: &[serde_json::Value]) -> Vec<String> {
    entry_handles(entries)
        .into_iter()
        .zip(entries)
        .filter(|(_, e)| e.get("type").and_then(|t| t.as_str()) == Some(SHADER))
        .filter_map(|(h, _)| h)
        .collect()
}

impl FormExtras for ShaderExtras {
    fn hidden_fields(&self) -> &'static [&'static str] {
        &["fragment", "vertex"]
    }

    fn rows(&self, cx: &ExtrasCx) -> Vec<ExtraRow> {
        let (paths, _) = self.paths(cx);
        self.form.rows(&shader_names(cx.entries), &paths)
    }

    fn press(&mut self, id: usize) {
        self.form.press(id);
    }

    fn blocked(&self, cx: &ExtrasCx) -> Option<String> {
        let shaders = shader_names(cx.entries).len();
        self.form.blocked(cx.name, cx.name_taken(cx.name), shaders)
    }

    fn fill_args(
        &self,
        cx: &ExtrasCx,
        args: &mut serde_json::Map<String, serde_json::Value>,
    ) -> Vec<NewFile> {
        let (paths, files) = self.paths(cx);
        let stage = shader_source::stage_name;
        args.insert(
            stage(ShaderStage::Fragment).to_string(),
            paths.fragment.into(),
        );
        match paths.vertex {
            Some(vertex) => args.insert(stage(ShaderStage::Vertex).to_string(), vertex.into()),
            None => args.remove(stage(ShaderStage::Vertex)),
        };
        files
    }

    // Dropping the vertex file the source panel shows closes it first, asking
    // over unsaved edits; the answer confirms the form again.
    fn hold(&self, hook: &mut EditorHook) -> bool {
        let Some(original) = self.original.clone().filter(|_| self.removes_vertex()) else {
            return false;
        };
        let open = SourceKey {
            shader: original,
            stage: ShaderStage::Vertex,
        };
        let Some(src) = hook.shaders.source.as_ref().filter(|s| s.key == open) else {
            return false;
        };
        if !src.area.is_dirty() {
            hook.shaders.source = None;
            return false;
        }
        let prompt = shader_source::leave_prompt(src.file_name());
        hook.open_modal(&prompt, shader_source::leave_buttons(Leave::ConfirmForm));
        true
    }

    fn edit_entries(&mut self, entries: &mut EntryList, c: &Committed) {
        let Some(name) = c.name.as_deref() else {
            return;
        };
        self.form.assign_materials(entries, name);
        if let Some(idx) = entries.index_of(c.key) {
            self.form.place_default(entries, idx);
        }
    }

    fn after(&self, hook: &mut EditorHook, c: &Committed) {
        let Some(name) = c.name.clone() else {
            return;
        };
        let declared = |stage: ShaderStage| {
            hook.entries
                .by_key(c.key)
                .and_then(|e| {
                    e.get("args")?
                        .get(shader_source::stage_name(stage))?
                        .as_str()
                })
                .unwrap_or_default()
                .to_string()
        };
        let open = |stage| SourceKey {
            shader: name.clone(),
            stage,
        };
        if self.form.is_new() {
            let fragment = declared(ShaderStage::Fragment);
            hook.notifier
                .success(&format!("Added Shader '{name}' ({fragment})"));
            hook.open_shader_file(open(ShaderStage::Fragment));
            return;
        }
        if let Some(src) = hook.shaders.source.as_mut()
            && c.before.as_deref() == Some(src.key.shader.as_str())
        {
            src.key.shader = name.clone();
        }
        let mut message = match c.before.as_deref().filter(|b| *b != name) {
            Some(before) => format!("Renamed Shader '{before}' to '{name}'"),
            None => format!("Updated Shader '{name}'"),
        };
        message.push_str(&references_note(c.moved));
        if self.removes_vertex() {
            message.push_str("; its vertex file stays on disk");
        }
        hook.notifier.success(&message);
        if self.adds_vertex() {
            hook.open_shader_file(open(ShaderStage::Vertex));
        }
    }
}

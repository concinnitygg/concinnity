//! EditorHook: per-type extras on the add / edit form. A type whose form
//! needs more than its schema fields implements `FormExtras`: rows after the
//! fields, a reason confirming is unavailable, args it fills in itself, files
//! a commit writes, and edits to other entries that ride the same undo step.
//! `for_type` is the registry; a type without extras gets the plain form.

use std::path::PathBuf;

use concinnity_core::ecs::World;

use super::edit::shader_form::ShaderExtras;
use super::{EditorHook, FormTarget, declared_id};
use crate::editor::entry_list::{EntryId, EntryList};
use crate::editor::panels::form_extras::ExtraRow;
use crate::editor::panels::form_panel;
use crate::editor::widget;

// What the extras read about the open form.
pub(in crate::editor::hook) struct ExtrasCx<'a> {
    pub(in crate::editor::hook) entries: &'a EntryList,
    pub(in crate::editor::hook) target: &'a FormTarget,
    // The name heading's text, trimmed.
    pub(in crate::editor::hook) name: &'a str,
}

impl ExtrasCx<'_> {
    // Whether an entry other than the one the form edits declares `name`.
    pub(in crate::editor::hook) fn name_taken(&self, name: &str) -> bool {
        let own = self.target.entry().and_then(|k| self.entries.index_of(k));
        self.entries
            .iter()
            .enumerate()
            .any(|(i, e)| Some(i) != own && declared_id(e) == Some(name))
    }
}

// A file a commit writes, after its entry edit is known to stand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::editor::hook) struct NewFile {
    pub(in crate::editor::hook) path: PathBuf,
    pub(in crate::editor::hook) text: String,
}

// The entry a commit landed on, its `$id` before and after, and how many
// references a rename moved with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::editor::hook) struct Committed {
    pub(in crate::editor::hook) key: EntryId,
    pub(in crate::editor::hook) before: Option<String>,
    pub(in crate::editor::hook) name: Option<String>,
    pub(in crate::editor::hook) moved: usize,
}

pub(in crate::editor::hook) trait FormExtras:
    std::fmt::Debug + Send
{
    // Schema fields the extras present in their own way; the form leaves them
    // out.
    fn hidden_fields(&self) -> &'static [&'static str] {
        &[]
    }

    fn rows(&self, cx: &ExtrasCx) -> Vec<ExtraRow>;

    // A press on the row with `id`.
    fn press(&mut self, id: usize);

    // Why confirming is unavailable, when it is.
    fn blocked(&self, _cx: &ExtrasCx) -> Option<String> {
        None
    }

    // Fill in the args the extras decide, and name the files the commit writes.
    fn fill_args(
        &self,
        _cx: &ExtrasCx,
        _args: &mut serde_json::Map<String, serde_json::Value>,
    ) -> Vec<NewFile> {
        Vec::new()
    }

    // Whether the commit waits on a question the extras just asked; answering
    // it confirms the form again.
    fn hold(&self, _hook: &mut EditorHook) -> bool {
        false
    }

    // Edit other entries as part of the commit, on the list it landed in.
    fn edit_entries(&mut self, _entries: &mut EntryList, _c: &Committed) {}

    // Follow a commit that stood.
    fn after(&self, _hook: &mut EditorHook, _c: &Committed) {}
}

// The extras type `ty`'s form carries, for a form opened on `target`.
pub(in crate::editor::hook) fn for_type(
    ty: &str,
    entries: &EntryList,
    target: &FormTarget,
) -> Option<Box<dyn FormExtras>> {
    match (ty, target) {
        (_, FormTarget::Promote(_)) => None,
        ("Shader", _) => Some(Box::new(ShaderExtras::open(entries, target))),
        _ => None,
    }
}

impl EditorHook {
    // The open form's extras context, with the name heading's text.
    pub(in crate::editor::hook) fn extras_cx<'a>(&'a self, name: &'a str) -> ExtrasCx<'a> {
        ExtrasCx {
            entries: &self.entries,
            target: &self.form.target,
            name: name.trim(),
        }
    }

    // The open form's extra rows and why it cannot confirm, for this frame.
    pub(in crate::editor::hook) fn form_extras_data(
        &self,
        world: &World,
    ) -> (Vec<ExtraRow>, Option<String>) {
        let Some(extras) = &self.form.extras else {
            return (Vec::new(), None);
        };
        let typed = widget::field_text(world, form_panel::NAME_INPUT);
        let cx = self.extras_cx(&typed);
        (extras.rows(&cx), extras.blocked(&cx))
    }

    // Every row the form lists: its fields and its extra rows. The count does
    // not depend on the typed name, so the panel sizes without the world.
    pub(in crate::editor::hook) fn form_row_count(&self) -> usize {
        let extras = self
            .form
            .extras
            .as_ref()
            .map_or(0, |x| x.rows(&self.extras_cx("")).len());
        self.form.fields.len() + extras
    }

    // Commit a form with extras: the entry edit and the extras' edits as one
    // undo step, then the files, then the extras' follow-up. The form stays
    // open when the edit is refused or a file cannot be written.
    pub(in crate::editor::hook) fn commit_with_extras(
        &mut self,
        ty: &str,
        typed: &str,
        args: serde_json::Value,
        files: Vec<NewFile>,
    ) {
        let Some(mut extras) = self.form.extras.take() else {
            return;
        };
        if extras.hold(self) {
            self.form.extras = Some(extras);
            return;
        }
        let committed = self.commit_form_entry(ty, typed, args);
        extras.edit_entries(&mut self.entries, &committed);
        let stood = self.entries.changed_read_only(&self.baseline).is_none();
        let written = stood && self.write_new_files(&files);
        if !stood {
            // Refused: `mark_changed` puts the entries back and says why.
            self.mark_changed();
        } else if !written {
            self.entries = self.baseline.clone();
        }
        if !written {
            self.form.extras = Some(extras);
            return;
        }
        self.mark_changed();
        extras.after(self, &committed);
        self.form.close();
    }

    // The entry half of a commit: update the edited entry, or append a new
    // one, under the typed name.
    fn commit_form_entry(&mut self, ty: &str, typed: &str, args: serde_json::Value) -> Committed {
        match self.form.target.entry() {
            Some(key) => {
                let renamed = self.rename_entry(key, typed);
                if let Some(obj) = self.entries.by_key_mut(key).and_then(|e| e.as_object_mut()) {
                    obj.insert(
                        "args".to_string(),
                        with_optional_id(args, renamed.name.as_deref()),
                    );
                }
                Committed {
                    key,
                    before: renamed.before,
                    name: renamed.name,
                    moved: renamed.moved,
                }
            }
            None => {
                let name = self.finalize_name(typed);
                let key = self.entries.push(serde_json::json!({
                    "type": ty, "args": with_optional_id(args, name.as_deref()),
                }));
                Committed {
                    key,
                    before: None,
                    name,
                    moved: 0,
                }
            }
        }
    }

    // Write each new file, creating its directory; stop at the first that
    // fails, with the reason on the form.
    fn write_new_files(&mut self, files: &[NewFile]) -> bool {
        for file in files {
            let written = file
                .path
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|()| std::fs::write(&file.path, &file.text));
            if let Err(e) = written {
                self.form.error = Some(format!("Could not write {}: {e}", file.path.display()));
                return false;
            }
        }
        true
    }
}

// `args` declaring `id` as the `$id`, or anonymous when there is none.
pub(in crate::editor::hook) fn with_optional_id(
    args: serde_json::Value,
    id: Option<&str>,
) -> serde_json::Value {
    match id {
        Some(id) => concinnity_cook::authoring::world::args_with_id(args, id),
        None => args,
    }
}

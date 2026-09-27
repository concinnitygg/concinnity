//! EditorHook: an SdfVolume form's extras (`panels/sdf_form.rs`) on the generic
//! add / edit form. Creating writes the new field file from its starter once
//! the entry edit is known to stand, and opens it in the source panel; editing
//! shows which other volumes read the same file.

use std::path::{Path, PathBuf};

use crate::editor::entry_list::EntryList;
use crate::editor::hook::EditorHook;
use crate::editor::hook::edits::Renamed;
use crate::editor::hook::form_extras::{Committed, ExtrasCx, FormExtras, NewFile};
use crate::editor::hook::form_state::FormTarget;
use crate::editor::panels::form_extras::ExtraRow;
use crate::editor::panels::sdf_field_list::SDF_VOLUME;
use crate::editor::panels::sdf_form::SdfForm;
use crate::editor::panels::shader_edit::file_stem;
use crate::editor::panels::shader_source::{self, SourceKey};

#[derive(Debug)]
pub(in crate::editor::hook) struct SdfExtras {
    form: SdfForm,
}

impl SdfExtras {
    pub(in crate::editor::hook) fn open(entries: &EntryList, target: &FormTarget) -> Self {
        let editing = target.entry().and_then(|k| entries.index_of(k));
        let dir = crate::project::assets_dir();
        Self {
            form: SdfForm::open(entries, editing, |declared| {
                shader_source::resolve_field_path(declared, dir.as_deref())
            }),
        }
    }

    // The field file a new volume named `name` writes, and how it declares it.
    fn new_file(&self, name: &str) -> Option<(NewFile, String)> {
        let text = self.form.starter_text()?;
        let (path, declared) = new_field_file(&file_stem(name));
        Some((
            NewFile {
                path,
                text: text.to_string(),
            },
            declared,
        ))
    }
}

impl FormExtras for SdfExtras {
    fn hidden_fields(&self) -> &'static [&'static str] {
        self.form.hidden_fields()
    }

    fn rows(&self, cx: &ExtrasCx) -> Vec<ExtraRow> {
        let declared = self.new_file(cx.name).map(|(_, d)| d).unwrap_or_default();
        self.form.rows(&declared)
    }

    fn press(&mut self, id: usize) {
        self.form.press(id);
    }

    fn switch_type(&self) -> Option<&'static str> {
        self.form.switch_type()
    }

    fn blocked(&self, cx: &ExtrasCx) -> Option<String> {
        self.form.blocked(cx.name, cx.name_taken(cx.name))
    }

    fn fill_args(
        &self,
        cx: &ExtrasCx,
        args: &mut serde_json::Map<String, serde_json::Value>,
    ) -> Vec<NewFile> {
        let Some((file, declared)) = self.new_file(cx.name) else {
            return Vec::new();
        };
        args.insert("fragment_shader".to_string(), declared.into());
        args.insert("volumetric".to_string(), self.form.volumetric().into());
        vec![file]
    }

    fn after(&self, hook: &mut EditorHook, c: &Committed) {
        let Some(name) = c.name.clone() else {
            return;
        };
        if !self.form.is_new() {
            let renamed = Renamed {
                before: c.before.clone(),
                name: Some(name),
                moved: c.moved,
            };
            if let Some(message) = renamed.message(SDF_VOLUME) {
                hook.notifier.success(&message);
            }
            return;
        }
        let declared = hook
            .entries
            .by_key(c.key)
            .and_then(|e| e.get("args")?.get("fragment_shader")?.as_str())
            .unwrap_or_default()
            .to_string();
        hook.notifier
            .success(&format!("Added SDF field '{name}' ({declared})"));
        let path = hook.resolved_field(&declared);
        hook.open_shader_file(SourceKey::Field { path });
    }
}

// Where a new field file `stem` goes under the project's assets, numbered past
// files already there, and how an SdfVolume declares it: relative to the
// assets directory, which is where the build looks for it first.
fn new_field_file(stem: &str) -> (PathBuf, String) {
    let dir = crate::project::assets_dir().unwrap_or_else(|| PathBuf::from("assets"));
    let path = shader_source::starter_path(&dir, stem, Path::exists);
    let declared = shader_source::declared_form(&path, &dir);
    (path, declared)
}

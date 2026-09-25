//! The form-extras seam (`hook/form_extras.rs`) through a stub type's extras
//! on a PointLight form: hidden fields, rows after the fields, a press, the
//! reason the form cannot confirm, args the extras fill in, the file a commit
//! writes, and an edit to another entry riding the same undo step. A refused
//! edit keeps the form open and writes nothing.

use concinnity_cook::build_only::include::SourcedEntry;
use concinnity_core::ecs::World;
use std::path::PathBuf;

use super::fixtures::{entry, hook, set_field, world_with_fields};
use crate::editor::entry_list::EntryList;
use crate::editor::hook::EditorHook;
use crate::editor::hook::FormTarget;
use crate::editor::hook::form_extras::{Committed, ExtrasCx, FormExtras, NewFile};
use crate::editor::panels::form_extras::{ExtraControl, ExtraRow};
use crate::editor::panels::form_panel::{self, FormAction};

const FLAG: usize = 1;

#[derive(Debug)]
struct Stub {
    flag: bool,
    file: PathBuf,
}

impl FormExtras for Stub {
    fn hidden_fields(&self) -> &'static [&'static str] {
        &["range"]
    }

    fn rows(&self, cx: &ExtrasCx) -> Vec<ExtraRow> {
        vec![
            ExtraRow::label("Extra", Some(format!("for '{}'", cx.name))),
            ExtraRow {
                id: FLAG,
                caption: "flag".to_string(),
                indent: true,
                control: ExtraControl::Check {
                    on: self.flag,
                    enabled: true,
                },
                detail: None,
            },
        ]
    }

    fn press(&mut self, id: usize) {
        if id == FLAG {
            self.flag = !self.flag;
        }
    }

    fn blocked(&self, _cx: &ExtrasCx) -> Option<String> {
        (!self.flag).then(|| "tick the flag".to_string())
    }

    fn fill_args(
        &self,
        _cx: &ExtrasCx,
        args: &mut serde_json::Map<String, serde_json::Value>,
    ) -> Vec<NewFile> {
        args.insert("intensity".to_string(), 7.0.into());
        vec![NewFile {
            path: self.file.clone(),
            text: "written".to_string(),
        }]
    }

    fn edit_entries(&mut self, entries: &mut EntryList, c: &Committed) {
        assert!(
            entries.index_of(c.key).is_some(),
            "runs on the committed list"
        );
        for e in entries.iter_mut().filter(|e| e["args"]["$id"] == "other") {
            e["args"]["intensity"] = 3.0.into();
        }
    }

    fn after(&self, hook: &mut EditorHook, c: &Committed) {
        let name = c.name.clone().unwrap_or_default();
        hook.notifier.success(&format!("stub after {name}"));
    }
}

fn open_with_stub(h: &mut EditorHook, world: &mut World, file: PathBuf) {
    h.open_form(world, "PointLight".to_string(), FormTarget::New);
    h.form.extras = Some(Box::new(Stub { flag: false, file }));
    h.refresh_form(world);
    set_field(world, form_panel::NAME_INPUT, "lamp");
}

#[test]
fn a_types_extras_shape_the_form_and_ride_its_commit() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("made/by_stub.txt");
    let mut h = hook(vec![entry("other", "PointLight")]);
    let mut world = world_with_fields();
    open_with_stub(&mut h, &mut world, file.clone());

    assert!(h.form.fields.iter().all(|f| f.key != "range"), "hidden");
    assert_eq!(h.form_row_count(), h.form.fields.len() + 2);
    let (rows, blocked) = h.form_extras_data(&world);
    assert_eq!(rows[0].detail.as_deref(), Some("for 'lamp'"));
    assert_eq!(blocked.as_deref(), Some("tick the flag"));
    let data = h.panel_data(&world);
    let view = h.make_form_view(&data, [0.0, 0.0]);
    assert_eq!(view.extras.len(), 2);
    assert_eq!(view.blocked, Some("tick the flag"));

    h.apply_form(FormAction::Confirm, &mut world);
    assert_eq!(h.entries.len(), 1, "blocked: nothing added");
    assert_eq!(h.form.error.as_deref(), Some("tick the flag"));
    assert!(!file.exists());

    h.apply_form(FormAction::PressExtra(FLAG), &mut world);
    assert!(h.form.touched);
    assert_eq!(h.form_extras_data(&world).1, None);
    h.apply_form(FormAction::Confirm, &mut world);
    assert!(!h.form_open());
    assert_eq!(h.entries.len(), 2);
    assert_eq!(h.entries[1]["args"]["$id"], "lamp");
    assert_eq!(h.entries[1]["args"]["intensity"], 7.0);
    assert!(
        h.entries[1]["args"].get("range").is_some(),
        "kept at its default"
    );
    assert_eq!(h.entries[0]["args"]["intensity"], 3.0);
    assert_eq!(std::fs::read_to_string(&file).unwrap(), "written");
    let toasts: Vec<String> = h
        .notifier
        .stack()
        .cards
        .into_iter()
        .map(|c| c.message)
        .collect();
    assert!(
        toasts.contains(&"stub after lamp".to_string()),
        "{toasts:?}"
    );

    h.undo(&mut world);
    assert_eq!(h.entries.len(), 1);
    assert!(h.entries[0]["args"].get("intensity").is_none());
    assert!(!h.can_undo(), "one step");
    assert!(file.exists(), "the file stays");
}

// An extras edit that reaches an included entry is refused whole: the entries
// go back, the form stays open, and no file is written.
#[test]
fn a_refused_commit_keeps_the_form_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("by_stub.txt");
    let mut h = hook(Vec::new());
    h.entries = EntryList::with_includes(vec![SourcedEntry {
        entry: entry("other", "PointLight"),
        file: Some("lib.jsonl".into()),
    }]);
    h.baseline = h.entries.clone();
    let mut world = world_with_fields();
    open_with_stub(&mut h, &mut world, file.clone());
    h.apply_form(FormAction::PressExtra(FLAG), &mut world);
    h.apply_form(FormAction::Confirm, &mut world);
    assert_eq!(h.entries.len(), 1);
    assert!(h.entries[0]["args"].get("intensity").is_none());
    assert!(h.form_open() && h.form.extras.is_some());
    assert!(!file.exists());
    assert!(!h.can_undo());
}

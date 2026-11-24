//! `EditorHook`: the add / edit form's session state, beside its lifecycle in
//! `editing.rs`. The form's layout half (`panels/form_panel.rs`) stays
//! stateless; it draws from a view built over this.

use std::collections::HashSet;

use super::FormData;
use super::form_extras::FormExtras;
use crate::editor::entry_list::EntryId;
use crate::editor::panels::form::FormField;
use crate::editor::panels::form_panel::{FormFocus, ShownForm};
use crate::editor::panels::registry::PanelKey;

// The type of the open form doubles as the open signal: `selected_type` is
// `None` while the panel is closed.
#[derive(Debug)]
pub(in crate::editor::hook) struct FormState {
    pub(in crate::editor::hook) selected_type: Option<String>,
    // What confirming the open form commits to: a new asset, an existing line,
    // or the promotion of a generated asset.
    pub(in crate::editor::hook) target: FormTarget,
    // The editable arg fields of the open form (derived from the type's default
    // args). Empty while the form is closed.
    pub(in crate::editor::hook) fields: Vec<FormField>,
    // The text each field with a text control holds, parallel to `fields`
    // (empty for the rest), and the name the heading holds.
    pub(in crate::editor::hook) texts: Vec<String>,
    pub(in crate::editor::hook) name: String,
    // Counts the times the fields were derived, so the panel rebuilds its
    // rows only when their shape can have changed.
    pub(in crate::editor::hook) rows_rev: u64,
    // A row to scroll into view: asked for, and being shown this frame.
    pub(in crate::editor::hook) reveal: Option<usize>,
    pub(in crate::editor::hook) reveal_now: Option<usize>,
    // Counts the actions taken on the form, so its derived data is read again
    // only after one.
    pub(in crate::editor::hook) data_rev: u64,
    // The derived data as of the entries, fields, and actions it was read at,
    // and the name it was read with.
    pub(in crate::editor::hook) data: Option<(FormDataKey, String, FormData)>,
    // The rows built into the current world.
    pub(in crate::editor::hook) shown: Option<ShownForm>,
    // The row the form last jumped to, so the next jump moves on from it.
    pub(in crate::editor::hook) last_jump: usize,
    // Counts the forms opened, so rows rebuilt for the same form keep their
    // scroll.
    pub(in crate::editor::hook) opened: u64,
    // Which form input has keyboard focus.
    pub(in crate::editor::hook) focus: FormFocus,
    // A validation message from the last rejected Add, shown under the form.
    pub(in crate::editor::hook) error: Option<String>,
    // The form arg field whose value dropdown is open (a large enum / ref
    // set). `None` outside an open dropdown.
    pub(in crate::editor::hook) field_dropdown: Option<usize>,
    // The open form's working args tree: the fields are derived from it, and it is
    // mutated by add / remove (structure) and, on capture, by the controls. Empty
    // outside AddForm.
    pub(in crate::editor::hook) args: serde_json::Map<String, serde_json::Value>,
    // The template behind the open form when it edits a template-derived asset:
    // confirming writes the minimal patch against this baseline, and the rows
    // show per-field override state. `None` for plain authored / new assets.
    pub(in crate::editor::hook) template: Option<FormTemplate>,
    // The field whose override menu (Revert / Apply-to-template) is open, and
    // whether the header's entity-level menu is open.
    pub(in crate::editor::hook) override_menu: Option<usize>,
    pub(in crate::editor::hook) entity_menu_open: bool,
    // The paths of the form's non-color vector fields currently disclosed into
    // per-element leaves. Cleared when the form opens / closes.
    pub(in crate::editor::hook) vec_expanded: HashSet<String>,
    // The unapplied-edit marker behind the heading's "*".
    pub(in crate::editor::hook) touched: bool,
    // The type's rows and commit steps beyond its schema fields, if it has any.
    pub(in crate::editor::hook) extras: Option<Box<dyn FormExtras>>,
    // The panel the form was opened from; it shows while that panel does.
    pub(in crate::editor::hook) host: PanelKey,
}

impl Default for FormState {
    fn default() -> Self {
        Self {
            selected_type: None,
            target: FormTarget::New,
            fields: Vec::new(),
            texts: Vec::new(),
            name: String::new(),
            rows_rev: 0,
            reveal: None,
            reveal_now: None,
            data_rev: 0,
            data: None,
            shown: None,
            last_jump: 0,
            opened: 0,
            focus: FormFocus::Name,
            error: None,
            field_dropdown: None,
            args: serde_json::Map::new(),
            template: None,
            override_menu: None,
            entity_menu_open: false,
            vec_expanded: HashSet::new(),
            touched: false,
            extras: None,
            host: PanelKey::Assets,
        }
    }
}

impl FormState {
    // Close the form panel, discarding every piece of its transient state.
    pub(in crate::editor::hook) fn close(&mut self) {
        self.selected_type = None;
        self.touched = false;
        self.target = FormTarget::New;
        self.fields.clear();
        self.data = None;
        self.texts.clear();
        self.name.clear();
        self.rows_rev += 1;
        self.reveal = None;
        self.args = serde_json::Map::new();
        self.template = None;
        self.override_menu = None;
        self.entity_menu_open = false;
        self.vec_expanded.clear();
        self.extras = None;
        self.host = PanelKey::Assets;
        self.last_jump = 0;
        self.focus = FormFocus::Name;
        self.error = None;
        self.field_dropdown = None;
    }
}

// What the form's derived data was read at: the entries, the fields'
// derivation, the actions taken, and the unapplied-edit marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::editor::hook) struct FormDataKey {
    pub(in crate::editor::hook) entries: u64,
    pub(in crate::editor::hook) rows: u64,
    pub(in crate::editor::hook) actions: u64,
    pub(in crate::editor::hook) touched: bool,
}

// The template a form-edited asset derives from.
#[derive(Debug, Clone)]
pub(in crate::editor::hook) struct FormTemplate {
    pub(in crate::editor::hook) name: String,
    // Effective template args: the type's defaults with the generated args
    // merged over them, the baseline a field is inherited from.
    pub(in crate::editor::hook) baseline: serde_json::Map<String, serde_json::Value>,
    // The authored asset or injection pass that produced the asset.
    pub(in crate::editor::hook) generated_by: String,
}

// What confirming the open add / edit form commits to.
#[derive(Debug, Clone, PartialEq, Default)]
pub(in crate::editor::hook) enum FormTarget {
    // A new asset: confirming appends it under a unique name.
    #[default]
    New,
    // The authored entry with this session key: confirming updates that line
    // in place. A key rather than a position, so a line added or removed
    // elsewhere in the list cannot retarget the open form onto another entry.
    Entry(EntryId),
    // A generated asset with no world.jsonl line. Confirming appends its line,
    // which overrides the expansion; renaming adds a separate asset.
    Promote(serde_json::Value),
}

impl FormTarget {
    // The authored entry the form updates in place, if any.
    pub(in crate::editor::hook) fn entry(&self) -> Option<EntryId> {
        match self {
            Self::Entry(key) => Some(*key),
            _ => None,
        }
    }

    // Whether the form is editing an asset that already exists (in the world or
    // in the build), rather than adding a brand-new one.
    pub(in crate::editor::hook) fn is_edit(&self) -> bool {
        !matches!(self, Self::New)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::entry_list::EntryList;
    use crate::editor::panels::form::FieldKind;

    #[test]
    fn close_discards_every_field_of_the_open_form() {
        let mut args = serde_json::Map::new();
        args.insert("radius".into(), serde_json::json!(2.0));
        let mut s = FormState {
            selected_type: Some("Sphere".into()),
            target: FormTarget::Entry(
                EntryList::new(vec![serde_json::json!({})])
                    .key_at(0)
                    .unwrap(),
            ),
            fields: vec![FormField {
                key: "radius".into(),
                kind: FieldKind::Float,
                initial: "2.0".into(),
                boolval: false,
                variants: Vec::new(),
                variant_idx: 0,
                choices: Vec::new(),
                sparse: false,
                unset: false,
            }],
            last_jump: 3,
            texts: vec!["2.0".into()],
            name: "ball".into(),
            reveal: Some(4),
            focus: FormFocus::Field(1),
            error: Some("bad".into()),
            field_dropdown: Some(2),
            args,
            template: Some(FormTemplate {
                name: "t".into(),
                baseline: serde_json::Map::new(),
                generated_by: "g".into(),
            }),
            override_menu: Some(1),
            entity_menu_open: true,
            vec_expanded: HashSet::from(["pos".to_string()]),
            touched: true,
            extras: None,
            host: PanelKey::Shaders,
            ..FormState::default()
        };
        s.close();
        assert_eq!(s.selected_type, None);
        assert_eq!(s.target, FormTarget::New);
        assert!(s.fields.is_empty() && s.args.is_empty() && s.vec_expanded.is_empty());
        assert!(s.template.is_none());
        assert_eq!((s.override_menu, s.entity_menu_open), (None, false));
        assert_eq!((s.last_jump, s.focus), (0, FormFocus::Name));
        assert!(s.texts.is_empty() && s.name.is_empty());
        assert!(s.reveal.is_none() && s.data.is_none());
        assert_eq!(s.error, None);
        assert_eq!(s.field_dropdown, None);
        assert!(!s.touched);
        assert_eq!(s.host, PanelKey::Assets);
    }
}

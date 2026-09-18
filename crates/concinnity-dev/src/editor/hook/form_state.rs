//! EditorHook: the add / edit form's session state, beside its lifecycle in
//! `editing.rs`. The form's layout half (`panels/form_panel.rs`) stays
//! stateless; it draws from a view built over this.

use std::collections::HashSet;

use crate::editor::entry_list::EntryId;
use crate::editor::panels::form::FormField;
use crate::editor::panels::form_panel::FormFocus;

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
    // First visible field of the form's scroll window (its physical control pool is
    // fixed size, so a form wider than `form::FIELD_POOL` scrolls). Reset on open /
    // structural change.
    pub(in crate::editor::hook) scroll: usize,
    // Which form input has keyboard focus.
    pub(in crate::editor::hook) focus: FormFocus,
    // A validation message from the last rejected Add, shown under the form.
    pub(in crate::editor::hook) error: Option<String>,
    // The form arg field whose value dropdown is open (a large enum / ref set),
    // and its scroll offset. `None` outside an open dropdown.
    pub(in crate::editor::hook) field_dropdown: Option<usize>,
    pub(in crate::editor::hook) field_dropdown_scroll: usize,
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
}

impl Default for FormState {
    fn default() -> Self {
        Self {
            selected_type: None,
            target: FormTarget::New,
            fields: Vec::new(),
            scroll: 0,
            focus: FormFocus::Name,
            error: None,
            field_dropdown: None,
            field_dropdown_scroll: 0,
            args: serde_json::Map::new(),
            template: None,
            override_menu: None,
            entity_menu_open: false,
            vec_expanded: HashSet::new(),
            touched: false,
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
        self.args = serde_json::Map::new();
        self.template = None;
        self.override_menu = None;
        self.entity_menu_open = false;
        self.vec_expanded.clear();
        self.scroll = 0;
        self.focus = FormFocus::Name;
        self.error = None;
        self.field_dropdown = None;
        self.field_dropdown_scroll = 0;
    }
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
    // An asset the build generates, which has no world.jsonl line of its own.
    // The form is seeded from the entry the expansion produced, and confirming
    // appends that line -- which then overrides the expansion, since the cook
    // drops a generated asset in favour of an authored one of the same name and
    // type. Renaming it in the form instead leaves the generated asset in place
    // and adds a separate one, which is the honest reading of a rename.
    Promote(serde_json::Value),
}

impl FormTarget {
    // The authored entry the form updates in place, if any.
    pub(in crate::editor::hook) fn entry(&self) -> Option<EntryId> {
        match self {
            FormTarget::Entry(key) => Some(*key),
            _ => None,
        }
    }

    // Whether the form is editing an asset that already exists (in the world or
    // in the build), rather than adding a brand-new one.
    pub(in crate::editor::hook) fn is_edit(&self) -> bool {
        !matches!(self, FormTarget::New)
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
            }],
            scroll: 3,
            focus: FormFocus::Field(1),
            error: Some("bad".into()),
            field_dropdown: Some(2),
            field_dropdown_scroll: 5,
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
        };
        s.close();
        assert_eq!(s.selected_type, None);
        assert_eq!(s.target, FormTarget::New);
        assert!(s.fields.is_empty() && s.args.is_empty() && s.vec_expanded.is_empty());
        assert!(s.template.is_none());
        assert_eq!((s.override_menu, s.entity_menu_open), (None, false));
        assert_eq!((s.scroll, s.focus), (0, FormFocus::Name));
        assert_eq!(s.error, None);
        assert_eq!((s.field_dropdown, s.field_dropdown_scroll), (None, 0));
        assert!(!s.touched);
    }
}

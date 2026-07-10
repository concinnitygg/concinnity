// src/editor/hook/editing.rs
//
// EditorHook: the add / edit form lifecycle -- open, refresh from the working
// args, capture the live controls, and validate / commit on confirm.

use super::*;

impl EditorHook {
    // Open the combo in `flavour`, clearing and focusing the shared filter field.
    pub(super) fn open_combo(&mut self, flavour: Combo, world: &mut World) {
        self.combo = flavour;
        self.combo_scroll = 0;
        self.row_menu = None;
        widget::focus_field_with(world, panel::FILTER_INPUT, "");
    }

    // Open the add / edit form for `ty`: derive its editable arg fields from the
    // type's defaults (or the edited entry's current args), seed the name + each
    // text field, and focus the name. `editing` is `Some(idx)` for a rename +
    // arg-edit of an existing entry, `None` for a new asset.
    pub(super) fn open_form(&mut self, world: &mut World, ty: String, editing: Option<usize>) {
        let seed = editing
            .and_then(|idx| self.entries.get(idx))
            .and_then(|e| e.get("args"))
            .and_then(|v| v.as_object())
            .cloned();
        let name = match editing {
            Some(idx) => self
                .entries
                .get(idx)
                .and_then(entry_name)
                .unwrap_or_default()
                .to_string(),
            None => self.unique_name(&ty),
        };
        // The working args tree: type defaults with the edited entry merged over
        // them. Add / remove and the controls mutate it; the fields are derived from
        // it so a structural change (a grown / shrunk array) re-derives cleanly.
        self.form_args = form::working_args(&ty, seed.as_ref());
        self.form_focus = FormFocus::Name;
        self.form_error = None;
        self.selected_type = Some(ty);
        self.editing = editing;
        self.combo = Combo::Closed;
        self.row_menu = None;
        self.field_dropdown = None;
        self.field_dropdown_scroll = 0;
        self.form_scroll = 0;
        self.vec_expanded.clear();
        // A freshly opened form comes to the front (the click that opened it focused
        // the Assets panel; the form the user is now editing should sit on top).
        self.focus_panel(DragTarget::Edit);
        self.refresh_form(world);
        widget::focus_field_with(world, form_panel::NAME_INPUT, &name);
    }

    // Derive the form's fields from the current working args, fill each reference
    // field's options, and (re-)seed the text controls. Called on open and after a
    // structural change (array add / remove) re-shapes the field list.
    pub(super) fn refresh_form(&mut self, world: &mut World) {
        let Some(ty) = self.selected_type.clone() else {
            return;
        };
        self.form_fields = form::fields_for_with(&ty, Some(&self.form_args), &self.vec_expanded);
        // Clamp the scroll window to the (possibly changed) field count -- an array
        // shrink can leave `form_scroll` past the new last page.
        let max = self.form_fields.len().saturating_sub(form::FIELD_POOL);
        self.form_scroll = self.form_scroll.min(max);
        // Reference fields pick from the world's existing assets of their target
        // type. Resolve the option lists up front (reads `entries`) so the fill loop
        // does not borrow `self` twice.
        let ref_opts: Vec<(usize, Vec<String>)> = self
            .form_fields
            .iter()
            .enumerate()
            .filter_map(|(i, f)| match f.kind {
                form::FieldKind::Ref { target } => Some((i, names_of_type(&self.entries, target))),
                _ => None,
            })
            .collect();
        for (i, names) in ref_opts {
            form::set_ref_options(&mut self.form_fields[i], &names);
        }
        // Seed only the text controls inside the visible window (their pool is
        // slot-indexed). Bool (checkbox), Enum + Ref (cycle buttons), and Array (a
        // header) have no text input to seed.
        let scroll = self.form_scroll;
        for (j, field) in self.form_fields.iter().enumerate() {
            if !field.kind.has_text_input() {
                continue;
            }
            if let Some(r) = visible_slot(j, scroll) {
                widget::seed_field(world, form_panel::form_input(r), &field.initial);
            }
        }
    }

    // Capture the current control values into the working args, preserving its
    // structure (array lengths). Run before a structural change or commit so edits
    // are not lost when the fields re-derive.
    pub(super) fn capture_controls(&mut self, world: &World) {
        let Some(ty) = self.selected_type.clone() else {
            return;
        };
        let scroll = self.form_scroll;
        let texts: Vec<String> = self
            .form_fields
            .iter()
            .enumerate()
            .map(|(j, f)| {
                if !f.kind.has_text_input() {
                    // State lives in the field (boolval / variant_idx) or its child
                    // leaves (a disclosed vector), not a control of its own.
                    String::new()
                } else if let Some(r) = visible_slot(j, scroll) {
                    // In the window: read the live control.
                    widget::field_text(world, form_panel::form_input(r))
                } else {
                    // Off-window: feed its stored value back so `assemble` round-trips
                    // it unchanged rather than blanking it (an empty string would
                    // overwrite a text field).
                    form::current_text(&self.form_args, &f.key)
                }
            })
            .collect();
        self.form_args = form::assemble(&ty, Some(&self.form_args), &self.form_fields, &texts);
    }

    // Close the form panel, discarding its transient state (the browse row's
    // highlight clears with `editing`).
    pub(super) fn close_form(&mut self) {
        self.selected_type = None;
        self.editing = None;
        self.form_fields.clear();
        self.form_args = serde_json::Map::new();
        self.vec_expanded.clear();
        self.form_scroll = 0;
        self.form_focus = FormFocus::Name;
        self.form_error = None;
        self.field_dropdown = None;
        self.field_dropdown_scroll = 0;
    }

    // Route a resolved panel click. Field-focus transitions mutate the injected
    // `TextInput` components, so this needs the world.
    pub(super) fn apply_panel(&mut self, action: PanelAction, world: &mut World) {
        match action {
            PanelAction::TogglePicker => {
                if self.combo == Combo::Picker {
                    self.combo = Combo::Closed;
                } else {
                    self.open_combo(Combo::Picker, world);
                }
            }
            PanelAction::ToggleFilter => {
                if self.combo == Combo::Filter {
                    self.combo = Combo::Closed;
                } else {
                    self.open_combo(Combo::Filter, world);
                }
            }
            PanelAction::PickOption(i) => match self.combo {
                Combo::Filter => {
                    if let Some(o) = self.combo_options(world).get(i) {
                        self.type_filter = if o == panel::ALL_LABEL {
                            None
                        } else {
                            Some(o.clone())
                        };
                    }
                    self.list_scroll = 0;
                    self.combo = Combo::Closed;
                }
                Combo::Picker => {
                    if let Some(ty) = self.combo_options(world).get(i).cloned() {
                        // A config singleton edits the world's existing instance if
                        // it has one, else adds it (edit-or-add); a multi-instance
                        // asset always adds a new one.
                        let existing = panel::is_singleton(&ty)
                            .then(|| {
                                self.entries
                                    .iter()
                                    .position(|e| entry_type(e) == Some(ty.as_str()))
                            })
                            .flatten();
                        self.open_form(world, ty, existing);
                    }
                }
                Combo::Closed => {}
            },
            PanelAction::OpenEntry(idx) => {
                if let Some(ty) = self.entries.get(idx).and_then(entry_type).map(String::from) {
                    self.open_form(world, ty, Some(idx));
                }
            }
            PanelAction::OpenRowMenu(entry) => self.row_menu = Some(entry),
            PanelAction::RowDelete => {
                if let Some(idx) = self.row_menu
                    && idx < self.entries.len()
                {
                    self.entries.remove(idx);
                    self.mark_changed();
                    // The open form indexes into `entries`: deleting the edited
                    // entry closes it, and deleting an earlier one shifts it.
                    match self.editing {
                        Some(e) if e == idx => self.close_form(),
                        Some(e) if e > idx => self.editing = Some(e - 1),
                        _ => {}
                    }
                }
                self.row_menu = None;
                let max = self.list_rows().len().saturating_sub(panel::MAX_ROWS);
                self.list_scroll = self.list_scroll.min(max);
            }
            PanelAction::CloseOverlays => {
                self.combo = Combo::Closed;
                self.row_menu = None;
            }
            PanelAction::Consume => {}
        }
    }

    // Route a resolved form-panel click. Field-focus transitions mutate the
    // injected `TextInput` components, so this needs the world.
    pub(super) fn apply_form(&mut self, action: FormAction, world: &mut World) {
        match action {
            FormAction::FocusName => self.form_focus = FormFocus::Name,
            FormAction::FocusField(i) => self.form_focus = FormFocus::Field(i),
            FormAction::ToggleField(i) => {
                if let Some(f) = self.form_fields.get_mut(i) {
                    f.boolval = !f.boolval;
                }
                self.form_error = None;
            }
            FormAction::CycleField(i) => {
                if let Some(f) = self.form_fields.get_mut(i)
                    && !f.variants.is_empty()
                {
                    f.variant_idx = (f.variant_idx + 1) % f.variants.len();
                }
                self.form_error = None;
            }
            FormAction::OpenFieldDropdown(i) => {
                // Toggle: a second click on the open field's control closes it.
                self.field_dropdown = if self.field_dropdown == Some(i) {
                    None
                } else {
                    Some(i)
                };
                self.field_dropdown_scroll = 0;
                self.form_error = None;
            }
            FormAction::PickFieldOption(opt) => {
                if let Some(open) = self.field_dropdown
                    && let Some(f) = self.form_fields.get_mut(open)
                    && opt < f.variants.len()
                {
                    f.variant_idx = opt;
                }
                self.field_dropdown = None;
                self.form_error = None;
            }
            FormAction::AddArrayElement(j) => {
                self.capture_controls(world);
                if let (Some(ty), Some(path)) = (
                    self.selected_type.clone(),
                    self.form_fields.get(j).map(|f| f.key.clone()),
                ) {
                    form::add_array_elem(&ty, &mut self.form_args, &path);
                    self.form_focus = FormFocus::Name;
                    self.refresh_form(world);
                }
                self.form_error = None;
            }
            FormAction::RemoveArrayElement(j) => {
                self.capture_controls(world);
                if let Some(path) = self.form_fields.get(j).map(|f| f.key.clone()) {
                    form::remove_array_elem(&mut self.form_args, &path);
                    self.form_focus = FormFocus::Name;
                    self.refresh_form(world);
                }
                self.form_error = None;
            }
            FormAction::ToggleVecExpand(j) => {
                // Fold the live controls in first so an in-progress edit survives the
                // field list re-deriving with / without this vector's element leaves.
                self.capture_controls(world);
                if let Some(path) = self.form_fields.get(j).map(|f| f.key.clone()) {
                    if !self.vec_expanded.remove(&path) {
                        self.vec_expanded.insert(path);
                    }
                    self.form_focus = FormFocus::Name;
                    self.refresh_form(world);
                }
                self.form_error = None;
            }
            FormAction::Confirm => self.confirm_form(world),
            FormAction::Close => self.close_form(),
            FormAction::CloseOverlays => self.field_dropdown = None,
            FormAction::Consume => {}
        }
    }

    // Capture the form's controls into the working args, validate, and commit (add
    // a new entry or update the edited one). On a validation error the form stays
    // open with the message shown, so nothing invalid ever reaches world.jsonl.
    pub(super) fn confirm_form(&mut self, world: &mut World) {
        self.form_error = None;
        let Some(ty) = self.selected_type.clone() else {
            self.close_form();
            return;
        };
        let typed = widget::field_text(world, form_panel::NAME_INPUT);
        // Fold the live control values into the working args (which already holds the
        // structure: nested objects, array lengths), then validate the whole thing.
        self.capture_controls(world);
        let args = self.form_args.clone();
        if let Err(e) = form::validate(&ty, &args) {
            self.form_error = Some(short_error(&e));
            return;
        }
        let args_val = serde_json::Value::Object(args);
        match self.editing {
            Some(idx) => {
                let name = self.finalize_rename(&typed, idx, &ty);
                if let Some(obj) = self.entries.get_mut(idx).and_then(|e| e.as_object_mut()) {
                    obj.insert("name".to_string(), serde_json::Value::String(name));
                    obj.insert("args".to_string(), args_val);
                }
            }
            None => {
                let name = self.finalize_name(&typed, &ty);
                self.entries.push(serde_json::json!({
                    "name": name, "type": ty, "args": args_val,
                }));
            }
        }
        self.mark_changed();
        self.close_form();
    }
}

// src/editor/hook/editing.rs
//
// EditorHook: the add / edit form lifecycle -- open, refresh from the working
// args, capture the live controls, and validate / commit on confirm.

use super::*;

impl EditorHook {
    // Open the add / edit form for `ty`: derive its editable arg fields from the
    // type's defaults (or the edited asset's current args), seed the name + each
    // text field, and focus the name. `target` decides where confirming lands --
    // a fresh line, an existing one, or the promotion of a generated asset.
    pub(super) fn open_form(&mut self, world: &mut World, ty: String, target: FormTarget) {
        self.open_form_with(world, ty, target, None);
    }

    // As `open_form`, but seeded from `template`'s effective (merged) args
    // rather than the target entry's own: a template-derived asset's authored
    // line is a sparse patch, so seeding from it alone would show type
    // defaults where the template's values belong.
    pub(super) fn open_form_with(
        &mut self,
        world: &mut World,
        ty: String,
        target: FormTarget,
        template: Option<FormTemplate>,
    ) {
        // Cloned rather than borrowed: `unique_name` below reads `self` too.
        let existing: Option<serde_json::Value> = match &target {
            FormTarget::Entry(idx) => self.entries.get(*idx).cloned(),
            FormTarget::Promote(entry) => Some(entry.clone()),
            FormTarget::New => None,
        };
        let seed = match &template {
            Some(t) => {
                let patch = existing
                    .as_ref()
                    .and_then(|e| e.get("args"))
                    .cloned()
                    .unwrap_or(serde_json::json!({}));
                let effective = concinnity_cook::build_only::merge_args(
                    &serde_json::Value::Object(t.baseline.clone()),
                    &patch,
                );
                effective.as_object().cloned()
            }
            None => existing
                .as_ref()
                .and_then(|e| e.get("args"))
                .and_then(|v| v.as_object())
                .cloned(),
        };
        let name = match (&template, &existing) {
            (Some(t), _) => t.name.clone(),
            (None, Some(e)) => entry_name(e).unwrap_or_default().to_string(),
            (None, None) => self.unique_name(&ty),
        };
        self.form_template = template;
        self.form_touched = false;
        self.override_menu = None;
        self.entity_menu_open = false;
        // The working args tree: type defaults with the edited entry merged over
        // them. Add / remove and the controls mutate it; the fields are derived from
        // it so a structural change (a grown / shrunk array) re-derives cleanly.
        self.form_args = form::working_args(&ty, seed.as_ref());
        self.form_focus = FormFocus::Name;
        self.form_error = None;
        self.selected_type = Some(ty);
        self.form_target = target;
        self.picker_open = false;
        self.row_menu = None;
        self.field_dropdown = None;
        self.field_dropdown_scroll = 0;
        self.form_scroll = 0;
        self.vec_expanded.clear();
        // A freshly opened form comes to the front (the click that opened it focused
        // the Assets panel; the form the user is now editing should sit on top).
        self.focus_panel(PanelKey::Edit);
        self.refresh_form(world);
        widget::focus_field_with(world, form_panel::NAME_INPUT, &name);
    }

    // The form panel's visible field-slot count at its current (possibly resized)
    // height, for the seed / capture window and the scroll clamps.
    pub(super) fn form_window(&self) -> usize {
        form_panel::rows_for_height(self.effective_size(PanelKey::Edit)[1])
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
        let window = self.form_window();
        let max = self.form_fields.len().saturating_sub(window);
        self.form_scroll = self.form_scroll.min(max);
        // Reference fields pick from the world's existing assets of their target
        // type. Resolve the option lists up front (reads `entries` + the cooked
        // tree) so the fill loop does not borrow `self` twice.
        let ref_opts: Vec<(usize, Vec<String>)> = self
            .form_fields
            .iter()
            .enumerate()
            .filter_map(|(i, f)| match f.kind {
                form::FieldKind::Ref { target } => Some((i, self.ref_targets(target))),
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
            if let Some(r) = visible_slot(j, scroll, window) {
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
        let window = self.form_window();
        let texts: Vec<String> = self
            .form_fields
            .iter()
            .enumerate()
            .map(|(j, f)| {
                if !f.kind.has_text_input() {
                    // State lives in the field (boolval / variant_idx) or its child
                    // leaves (a disclosed vector), not a control of its own.
                    String::new()
                } else if let Some(r) = visible_slot(j, scroll, window) {
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

    // The names a reference field targeting `ty` can pick from: every asset of
    // that type in the expanded world, not just the authored lines. A promoted
    // asset's references point at generated assets, so an authored-only list
    // would offer no way to retarget one (`form::set_ref_options` keeps the
    // current value regardless, but could not offer its siblings).
    fn ref_targets(&self, ty: &str) -> Vec<String> {
        let mut names = names_of_type(&self.entries, ty);
        for asset in self.tree_groups.iter().flat_map(|g| &g.assets) {
            if asset.asset_type == ty && !names.iter().any(|n| n == &asset.name) {
                names.push(asset.name.clone());
            }
        }
        names
    }

    // Close the form panel, discarding its transient state.
    pub(super) fn close_form(&mut self) {
        self.selected_type = None;
        self.form_touched = false;
        self.form_target = FormTarget::New;
        self.form_fields.clear();
        self.form_args = serde_json::Map::new();
        self.form_template = None;
        self.override_menu = None;
        self.entity_menu_open = false;
        self.vec_expanded.clear();
        self.form_scroll = 0;
        self.form_focus = FormFocus::Name;
        self.form_error = None;
        self.field_dropdown = None;
        self.field_dropdown_scroll = 0;
    }

    // Route a resolved form-panel click. Field-focus transitions mutate the
    // injected `TextInput` components, so this needs the world.
    pub(super) fn apply_form(&mut self, action: FormAction, world: &mut World) {
        // Control edits (as opposed to focus moves / dropdown toggles) mark
        // the form's unapplied-edit state.
        if matches!(
            action,
            FormAction::ToggleField(_)
                | FormAction::CycleField(_)
                | FormAction::PickFieldOption(_)
                | FormAction::AddArrayElement(_)
                | FormAction::RemoveArrayElement(_)
        ) {
            self.form_touched = true;
        }
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
            FormAction::OpenOverrideMenu(i) => {
                self.override_menu = if self.override_menu == Some(i) {
                    None
                } else {
                    Some(i)
                };
                self.entity_menu_open = false;
                self.field_dropdown = None;
            }
            FormAction::PickOverrideOption(k) => self.pick_override_option(k, world),
            FormAction::OpenEntityMenu => {
                self.entity_menu_open = !self.entity_menu_open;
                self.override_menu = None;
                self.field_dropdown = None;
            }
            FormAction::PickEntityOption(k) => self.pick_entity_option(k, world),
            FormAction::JumpOverride => self.jump_to_override(world),
            FormAction::Confirm => self.confirm_form(world),
            FormAction::Close => self.close_form(),
            FormAction::CloseOverlays => {
                self.field_dropdown = None;
                self.override_menu = None;
                self.entity_menu_open = false;
            }
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
        if let Err(e) = form::validate(&ty, &typed, &args) {
            self.form_error = Some(short_status(&e));
            return;
        }
        let args_val = serde_json::Value::Object(args);
        // A template-derived asset commits the minimal patch against its
        // template baseline: only the fields that differ are authored, so
        // everything else keeps tracking the template. Its name is the link to
        // the template, so a rename is rejected rather than silently breaking it.
        if let Some(t) = self.form_template.clone() {
            if typed != t.name {
                self.form_error = Some("a template instance keeps its generated name".to_string());
                return;
            }
            let baseline = serde_json::Value::Object(t.baseline);
            let patch = overrides::minimal_patch(&baseline, &args_val);
            match (self.form_target.entry(), patch) {
                (Some(idx), Some(p)) => {
                    if let Some(obj) = self.entries.get_mut(idx).and_then(|e| e.as_object_mut()) {
                        obj.insert("args".to_string(), p);
                    }
                    self.mark_changed();
                }
                // Every field matches the template again: the patch line has
                // nothing left to say, so it goes away and the asset returns
                // to pristine. `remove_entry_at` records the undo step.
                (Some(idx), None) => self.remove_entry_at(idx),
                (None, Some(p)) => {
                    self.entries.push(serde_json::json!({
                        "name": t.name, "type": ty, "args": p,
                    }));
                    self.mark_changed();
                }
                // Nothing diverges and nothing is authored: nothing to commit.
                (None, None) => {}
            }
            self.close_form();
            return;
        }
        match self.form_target.entry() {
            Some(idx) => {
                let name = self.finalize_rename(&typed, idx, &ty);
                if let Some(obj) = self.entries.get_mut(idx).and_then(|e| e.as_object_mut()) {
                    obj.insert("name".to_string(), serde_json::Value::String(name));
                    obj.insert("args".to_string(), args_val);
                }
            }
            // A new asset, or the promotion of a generated one: both append. A
            // promotion keeps the generated name (nothing in `entries` holds it,
            // so `finalize_name` leaves it alone), and that identity is what
            // makes the new line override the expansion.
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

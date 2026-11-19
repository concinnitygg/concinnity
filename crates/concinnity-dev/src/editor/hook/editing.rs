//! `EditorHook`: the add / edit form lifecycle -- open, refresh from the working
//! args, capture the live controls, and validate / commit on confirm.

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_cook::authoring::world::{args_with_id, replace_args};
use concinnity_engine::ecs::World;

use super::form_extras::{self, with_optional_id};
use super::{EditorHook, FormTarget, FormTemplate, declared_id, names_of_types, short_status};
use crate::editor::overrides;
use crate::editor::panels::form;
use crate::editor::panels::form_panel::{FormAction, FormFocus};
use crate::editor::panels::registry::PanelKey;

impl EditorHook {
    // Open the add / edit form for `ty`: derive its editable arg fields from the
    // type's defaults (or the edited asset's current args), seed the name + each
    // text field, and focus the name. `target` decides where confirming lands --
    // a fresh line, an existing one, or the promotion of a generated asset.
    pub(super) fn open_form(&mut self, ty: String, target: FormTarget) {
        self.open_form_with(ty, target, None);
    }

    // As `open_form`, but seeded from `template`'s effective (merged) args
    // rather than the target entry's own: a template-derived asset's authored
    // line is a sparse patch, so seeding from it alone would show type
    // defaults where the template's values belong.
    pub(super) fn open_form_with(
        &mut self,
        ty: String,
        target: FormTarget,
        template: Option<FormTemplate>,
    ) {
        // Cloned rather than borrowed: `unique_name` below reads `self` too.
        let existing: Option<serde_json::Value> = match &target {
            FormTarget::Entry(key) => self.entries.by_key(*key).cloned(),
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
            (None, Some(e)) => declared_id(e).unwrap_or_default().to_string(),
            (None, None) => self.unique_name(&ty),
        };
        self.form.extras = match &template {
            Some(_) => None,
            None => form_extras::for_type(&ty, &self.entries, &target),
        };
        self.form.template = template;
        self.form.touched = false;
        self.form.override_menu = None;
        self.form.entity_menu_open = false;
        // The working args tree: type defaults with the edited entry merged over
        // them. Add / remove and the controls mutate it; the fields are derived from
        // it so a structural change (a grown / shrunk array) re-derives cleanly.
        self.form.args = form::working_args(&ty, seed.as_ref());
        self.form.focus = FormFocus::Name;
        self.form.error = None;
        self.form.selected_type = Some(ty);
        self.form.target = target;
        self.form.host = PanelKey::Assets;
        self.assets.picker_open = false;
        self.assets.row_menu = None;
        self.form.field_dropdown = None;
        self.form.last_jump = 0;
        self.form.opened += 1;
        self.form.vec_expanded.clear();
        // A freshly opened form comes to the front (the click that opened it focused
        // the Assets panel; the form the user is now editing should sit on top).
        self.focus_panel(PanelKey::Edit);
        self.refresh_form();
        self.form.name = name;
    }

    // Derive the form's fields from the current working args, fill each reference
    // field's options, and (re-)seed the field texts. Called on open and after a
    // structural change (array add / remove) re-shapes the field list.
    pub(super) fn refresh_form(&mut self) {
        let Some(ty) = self.form.selected_type.clone() else {
            return;
        };
        let mut fields = form::fields_for_with(&ty, Some(&self.form.args), &self.form.vec_expanded);
        if let Some(extras) = &self.form.extras {
            let hidden = extras.hidden_fields();
            fields.retain(|f| !hidden.iter().any(|h| form::is_at_or_under(&f.key, h)));
        }
        self.form.fields = fields;
        self.form.rows_rev += 1;
        // Reference fields pick from the world's existing assets of their target
        // type. Resolve the option lists up front (reads `entries` + the cooked
        // tree) so the fill loop does not borrow `self` twice.
        let ref_opts: Vec<(usize, Vec<String>)> = self
            .form
            .fields
            .iter()
            .enumerate()
            .filter_map(|(i, f)| match f.kind {
                form::FieldKind::Ref { targets } => Some((i, self.ref_options(targets))),
                _ => None,
            })
            .collect();
        for (i, names) in ref_opts {
            form::set_ref_options(&mut self.form.fields[i], &names);
        }
        // Action fields pick from what the world's Screens and Scenes let an
        // action do.
        if self
            .form
            .fields
            .iter()
            .any(|f| matches!(f.kind, form::FieldKind::Action { .. }))
        {
            let screens = self.ref_options(&["Screen"]);
            let scenes = self.ref_options(&["Scene"]);
            for field in &mut self.form.fields {
                form::set_action_options(field, &screens, &scenes);
            }
        }
        self.form.texts = self
            .form
            .fields
            .iter()
            .map(|f| {
                if f.kind.has_text_field() {
                    f.initial.clone()
                } else {
                    String::new()
                }
            })
            .collect();
    }

    // Capture the current control values into the working args, preserving its
    // structure (array lengths). Run before a structural change or commit so edits
    // are not lost when the fields re-derive.
    pub(super) fn capture_controls(&mut self) {
        let Some(ty) = self.form.selected_type.clone() else {
            return;
        };
        self.form.args = form::assemble(
            &ty,
            Some(&self.form.args),
            &self.form.fields,
            &self.form.texts,
        );
    }

    // The names a reference field targeting `targets` can pick from: every asset
    // of those types in the expanded world, not just the authored lines, and
    // every asset when the field names no type. A promoted asset's references
    // point at generated assets, so an authored-only list would offer no way to
    // retarget one (`form::set_ref_options` keeps the current value regardless,
    // but could not offer its siblings).
    fn ref_options(&self, targets: &[&str]) -> Vec<String> {
        let mut names = names_of_types(&self.entries, targets);
        let wanted: Vec<RegisteredType> = targets
            .iter()
            .filter_map(|t| RegisteredType::parse(t))
            .collect();
        for asset in self.assets.groups.iter().flat_map(|g| &g.assets) {
            let offered = wanted.is_empty() || wanted.contains(&asset.asset_type);
            if offered && !names.iter().any(|n| n == &asset.name) {
                names.push(asset.name.clone());
            }
        }
        names
    }

    // Route a resolved form-panel click. Field-focus transitions read the
    // form's `TextField`s, so this needs the world.
    pub(super) fn apply_form(&mut self, action: FormAction, world: &World) {
        self.form.data_rev += 1;
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
            self.form.touched = true;
        }
        match action {
            FormAction::ToggleField(i) => {
                if let Some(f) = self.form.fields.get_mut(i) {
                    f.boolval = !f.boolval;
                }
                self.form.error = None;
            }
            FormAction::CycleField(i) => {
                if let Some(f) = self.form.fields.get_mut(i)
                    && !f.variants.is_empty()
                {
                    f.variant_idx = (f.variant_idx + 1) % f.variants.len();
                }
                self.form.error = None;
            }
            FormAction::OpenFieldDropdown(i) => {
                // Toggle: a second click on the open field's control closes it.
                self.form.field_dropdown = if self.form.field_dropdown == Some(i) {
                    None
                } else {
                    Some(i)
                };
                self.form.error = None;
            }
            FormAction::PressExtra(id) => {
                let mut switch = None;
                if let Some(extras) = self.form.extras.as_mut() {
                    extras.press(id);
                    self.form.touched = true;
                    switch = extras.switch_type();
                }
                self.form.error = None;
                if let Some(ty) = switch {
                    self.switch_form_type(world, ty);
                }
            }
            FormAction::PickFieldOption(opt) => {
                if let Some(open) = self.form.field_dropdown
                    && let Some(f) = self.form.fields.get_mut(open)
                    && opt < f.variants.len()
                {
                    f.variant_idx = opt;
                }
                self.form.field_dropdown = None;
                self.form.error = None;
            }
            FormAction::AddArrayElement(j) => {
                self.capture_controls();
                if let (Some(ty), Some(path)) = (
                    self.form.selected_type.clone(),
                    self.form.fields.get(j).map(|f| f.key.clone()),
                ) {
                    form::add_array_elem(&ty, &mut self.form.args, &path);
                    self.form.focus = FormFocus::Name;
                    self.refresh_form();
                }
                self.form.error = None;
            }
            FormAction::RemoveArrayElement(j) => {
                self.capture_controls();
                if let Some(path) = self.form.fields.get(j).map(|f| f.key.clone()) {
                    form::remove_array_elem(&mut self.form.args, &path);
                    self.form.focus = FormFocus::Name;
                    self.refresh_form();
                }
                self.form.error = None;
            }
            FormAction::ToggleVecExpand(j) => {
                // Fold the live controls in first so an in-progress edit survives the
                // field list re-deriving with / without this vector's element leaves.
                self.capture_controls();
                if let Some(path) = self.form.fields.get(j).map(|f| f.key.clone()) {
                    if !self.form.vec_expanded.remove(&path) {
                        self.form.vec_expanded.insert(path);
                    }
                    self.form.focus = FormFocus::Name;
                    self.refresh_form();
                }
                self.form.error = None;
            }
            FormAction::OpenOverrideMenu(i) => {
                self.form.override_menu = if self.form.override_menu == Some(i) {
                    None
                } else {
                    Some(i)
                };
                self.form.entity_menu_open = false;
                self.form.field_dropdown = None;
            }
            FormAction::PickOverrideOption(k) => self.pick_override_option(k),
            FormAction::OpenEntityMenu => {
                self.form.entity_menu_open = !self.form.entity_menu_open;
                self.form.override_menu = None;
                self.form.field_dropdown = None;
            }
            FormAction::PickEntityOption(k) => self.pick_entity_option(k),
            FormAction::JumpOverride => self.jump_to_override(),
            FormAction::Confirm => self.confirm_form(),
            FormAction::Close => self.form.close(),
        }
    }

    // Close whichever menu the form has open.
    pub(super) fn close_form_menus(&mut self) {
        self.form.field_dropdown = None;
        self.form.override_menu = None;
        self.form.entity_menu_open = false;
        self.form.data_rev += 1;
    }

    // Capture the form's controls into the working args, validate, and commit (add
    // a new entry or update the edited one). On a validation error the form stays
    // open with the message shown, so nothing invalid ever reaches world.jsonl.
    pub(super) fn confirm_form(&mut self) {
        self.form.error = None;
        let Some(ty) = self.form.selected_type.clone() else {
            self.form.close();
            return;
        };
        let typed = self.form.name.clone();
        // Fold the live control values into the working args (which already holds the
        // structure: nested objects, array lengths), then validate the whole thing.
        self.capture_controls();
        let mut args = self.form.args.clone();
        let mut files = Vec::new();
        if let Some(extras) = &self.form.extras {
            let cx = self.extras_cx(&typed);
            if let Some(reason) = extras.blocked(&cx) {
                self.form.error = Some(reason);
                return;
            }
            files = extras.fill_args(&cx, &mut args);
        }
        if let Err(e) = form::validate(&ty, &typed, &args) {
            self.form.error = Some(short_status(&e));
            return;
        }
        let args_val = serde_json::Value::Object(args);
        if self.form.extras.is_some() {
            self.commit_with_extras(&ty, &typed, args_val, &files);
            return;
        }
        // A template-derived asset commits the minimal patch against its
        // template baseline: only the fields that differ are authored, so
        // everything else keeps tracking the template. Its name is the link to
        // the template, so a rename is rejected rather than silently breaking it.
        if let Some(t) = self.form.template.clone() {
            if typed != t.name {
                self.form.error = Some("a template instance keeps its generated name".to_string());
                return;
            }
            let baseline = serde_json::Value::Object(t.baseline);
            let patch = overrides::minimal_patch(&baseline, &args_val);
            match (self.form.target.entry(), patch) {
                (Some(key), Some(p)) => {
                    if let Some(entry) = self.entries.by_key_mut(key) {
                        replace_args(entry, p);
                    }
                    self.mark_changed();
                }
                // Every field matches the template again: the patch line has
                // nothing left to say, so it goes away and the asset returns
                // to pristine. `remove_entry_at` records the undo step.
                (Some(key), None) => {
                    if let Some(idx) = self.entries.index_of(key) {
                        self.remove_entry_at(idx);
                    }
                }
                (None, Some(p)) => {
                    self.entries.push(serde_json::json!({
                        "type": ty, "args": args_with_id(p, &t.name),
                    }));
                    self.mark_changed();
                }
                // Nothing diverges and nothing is authored: nothing to commit.
                (None, None) => {}
            }
            self.form.close();
            return;
        }
        if let Some(key) = self.form.target.entry() {
            let renamed = self.rename_entry(key, &typed);
            if let Some(obj) = self.entries.by_key_mut(key).and_then(|e| e.as_object_mut()) {
                obj.insert(
                    "args".to_string(),
                    with_optional_id(args_val, renamed.name.as_deref()),
                );
            }
            if let Some(message) = renamed.message(&ty) {
                self.notifier.success(&message);
            }
        } else {
            let id = self.finalize_name(&typed);
            self.entries.push(serde_json::json!({
                "type": ty, "args": with_optional_id(args_val, id.as_deref()),
            }));
        }
        self.mark_changed();
        self.form.close();
    }
}

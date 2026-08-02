// src/editor/hook/override_edit.rs
//
// EditorHook: the override loop for template-derived assets. Field state is
// derived from the committed patch (`editor/overrides.rs`); the actions here
// mutate the working entries -- remove a patch key (revert), write a value
// back into the authored Prefab definition (apply-to-template), or both in
// bulk -- each as a single undo step.

use super::*;
use crate::editor::overrides::prefab_map;

// One resolved option of the per-field or entity-level override menu.
pub(super) enum OverrideOption {
    // Remove the covered patch key, restoring the template value.
    Revert(String),
    // Write the instance value at the covered path into the prefab entry,
    // then drop the patch key (the template now carries the value).
    Apply(String),
    RevertAll,
    ApplyAll,
    // Strip patch fields equal to their template values (legacy full copies).
    Minimize,
    // Author the named preset-backed prefab definition as a world line.
    Materialize(String),
}

impl EditorHook {
    // The template info for `name`, rebuilding the index from the working
    // entries when an edit invalidated it. `None` when the asset is not
    // template-derived (or the world does not currently cook).
    pub(super) fn template_info(&mut self, name: &str) -> Option<overrides::TemplateInfo> {
        if self.template_index.is_none() {
            self.template_index = self
                .cook_working_entries()
                .ok()
                .map(|l| overrides::TemplateIndex::from_loaded(&l));
        }
        self.template_index.as_ref()?.get(name).cloned()
    }

    // The (type, template) pair the form needs to open `name` template-aware:
    // the baseline is the type's defaults with the generated args merged over.
    pub(super) fn form_template_for(&mut self, name: &str) -> Option<(String, FormTemplate)> {
        let info = self.template_info(name)?;
        let defaults = serde_json::Value::Object(form::base_args(&info.asset_type));
        let baseline = concinnity_cook::world::merge_args(&defaults, &info.baseline);
        Some((
            info.asset_type.clone(),
            FormTemplate {
                name: name.to_string(),
                baseline: baseline.as_object().cloned().unwrap_or_default(),
                generated_by: info.generated_by,
            },
        ))
    }

    // The committed patch behind the open form: the authored line's args, or
    // empty for a pristine asset. Live (un-applied) control edits are not a
    // patch yet, so the marks reflect what is actually authored.
    pub(super) fn committed_patch(&self) -> serde_json::Value {
        match self.form_target.entry() {
            Some(idx) => self
                .entries
                .get(idx)
                .and_then(|e| e.get("args"))
                .cloned()
                .unwrap_or(serde_json::json!({})),
            None => serde_json::json!({}),
        }
    }

    // Per-field override marks for the open form, parallel to `form_fields`.
    pub(super) fn form_override_marks(&self) -> Option<Vec<overrides::FieldOrigin>> {
        let t = self.form_template.as_ref()?;
        let template = serde_json::Value::Object(t.baseline.clone());
        let patch = self.committed_patch();
        Some(
            self.form_fields
                .iter()
                .map(|f| overrides::classify(&template, &patch, &f.key))
                .collect(),
        )
    }

    // The open per-field menu's options. Revert is always offered; Apply only
    // when the field maps into an authored Prefab entry.
    pub(super) fn override_menu_options(&self, field: usize) -> Vec<(OverrideOption, String)> {
        let Some(key) = self.form_fields.get(field).map(|f| f.key.clone()) else {
            return Vec::new();
        };
        let patch = self.committed_patch();
        let Some(covered) = overrides::covered_root(&patch, &key) else {
            return Vec::new();
        };
        let mut options = vec![(
            OverrideOption::Revert(covered.clone()),
            format!("Revert '{covered}'"),
        )];
        if let Some(t) = self.form_template.clone()
            && let Ok((slot, _)) = self.resolve_slot(&t, &covered)
        {
            let n = overrides::instance_count(&self.entries, &slot.def_name);
            options.push((
                OverrideOption::Apply(covered.clone()),
                format!(
                    "Apply to Prefab '{}' (updates {} instance{})",
                    slot.def_name,
                    n,
                    if n == 1 { "" } else { "s" }
                ),
            ));
        }
        options
    }

    // The entity-level menu's options (the "..." button in the form header).
    pub(super) fn entity_menu_options(&self) -> Vec<(OverrideOption, String)> {
        let Some(t) = self.form_template.clone() else {
            return Vec::new();
        };
        let patch = self.committed_patch();
        let has_patch = patch.as_object().is_some_and(|o| !o.is_empty());
        let mut options = Vec::new();
        if has_patch {
            options.push((
                OverrideOption::RevertAll,
                "Revert all overrides".to_string(),
            ));
        }
        if let Some(prefab_ref) = self.instance_prefab_ref(&t.generated_by) {
            let authored = self
                .entries
                .iter()
                .any(|e| entry_name(e) == Some(prefab_ref.as_str()) && is_prefab(e));
            if authored && has_patch {
                let n = overrides::instance_count(&self.entries, &prefab_ref);
                options.push((
                    OverrideOption::ApplyAll,
                    format!(
                        "Apply all to Prefab '{prefab_ref}' (updates {n} instance{})",
                        if n == 1 { "" } else { "s" }
                    ),
                ));
            }
            if !authored
                && !concinnity_cook::world::preset::load_preset_obj(&prefab_ref, "prefabs")
                    .is_null()
            {
                options.push((
                    OverrideOption::Materialize(prefab_ref.clone()),
                    format!("Materialize Prefab '{prefab_ref}' as authored"),
                ));
            }
        }
        if has_patch {
            options.push((OverrideOption::Minimize, "Minimize override".to_string()));
        }
        options
    }

    pub(super) fn pick_override_option(&mut self, k: usize, world: &mut World) {
        let Some(field) = self.override_menu.take() else {
            return;
        };
        let mut options = self.override_menu_options(field);
        if k < options.len() {
            self.run_override_option(options.remove(k).0, world);
        }
    }

    pub(super) fn pick_entity_option(&mut self, k: usize, world: &mut World) {
        if !self.entity_menu_open {
            return;
        }
        self.entity_menu_open = false;
        let mut options = self.entity_menu_options();
        if k < options.len() {
            self.run_override_option(options.remove(k).0, world);
        }
    }

    fn run_override_option(&mut self, option: OverrideOption, world: &mut World) {
        let Some(t) = self.form_template.clone() else {
            return;
        };
        let result = match option {
            OverrideOption::Revert(covered) => self.revert_covered(&t, &covered),
            OverrideOption::Apply(covered) => self.apply_covered(&t, &covered),
            OverrideOption::RevertAll => self.revert_all(&t),
            OverrideOption::ApplyAll => self.apply_all(&t),
            OverrideOption::Minimize => self.minimize_patch(&t),
            OverrideOption::Materialize(name) => self.materialize_prefab(&name),
        };
        let status = result.err().map(|e| short_status(&e));
        // Re-derive the form from the changed entries (same scroll, so the
        // reverted row stays under the cursor).
        let scroll = self.form_scroll;
        self.open_asset_form(&t.name, world);
        self.form_scroll = scroll.min(self.form_fields.len().saturating_sub(1));
        self.refresh_form(world);
        self.form_error = status;
    }

    // Remove the covered key from the instance's patch line; an emptied patch
    // removes the line, returning the asset to pristine.
    fn revert_covered(&mut self, t: &FormTemplate, covered: &str) -> Result<(), String> {
        let idx = self
            .patch_index(&t.name)
            .ok_or("nothing is overridden on this asset")?;
        let Some(args) = self.entries[idx].get_mut("args") else {
            return Err("the override line has no args".to_string());
        };
        if !overrides::remove_at_path(args, covered) {
            return Err(format!("'{covered}' is not overridden"));
        }
        self.drop_patch_if_empty_or_mark(idx);
        Ok(())
    }

    // Write the instance's value at `covered` into the Prefab entry that
    // generated it, then drop the patch key: the template now carries the
    // value, and every instance follows on the next expansion. One undo step.
    fn apply_covered(&mut self, t: &FormTemplate, covered: &str) -> Result<(), String> {
        let idx = self
            .patch_index(&t.name)
            .ok_or("nothing is overridden on this asset")?;
        let patch = self.committed_patch_of(idx);
        let value = overrides::value_at_path(&patch, covered)
            .cloned()
            .ok_or_else(|| format!("'{covered}' is not overridden"))?;
        let (slot, map) = self.resolve_slot(t, covered)?;
        self.write_template_field(&slot, map, covered, &value)?;
        let Some(args) = self.entries[idx].get_mut("args") else {
            return Err("the override line has no args".to_string());
        };
        overrides::remove_at_path(args, covered);
        self.drop_patch_if_empty_or_mark(idx);
        Ok(())
    }

    fn revert_all(&mut self, t: &FormTemplate) -> Result<(), String> {
        let idx = self
            .patch_index(&t.name)
            .ok_or("nothing is overridden on this asset")?;
        self.remove_entry_at(idx);
        Ok(())
    }

    // Apply every covered path that maps into the template; unmappable ones
    // stay authored and the caller reports them.
    fn apply_all(&mut self, t: &FormTemplate) -> Result<(), String> {
        let idx = self
            .patch_index(&t.name)
            .ok_or("nothing is overridden on this asset")?;
        let patch = self.committed_patch_of(idx);
        let mut kept: Vec<String> = Vec::new();
        for covered in overrides::patch_roots(&patch) {
            let value = match overrides::value_at_path(&patch, &covered) {
                Some(v) => v.clone(),
                None => continue,
            };
            match self
                .resolve_slot(t, &covered)
                .and_then(|(slot, map)| self.write_template_field(&slot, map, &covered, &value))
            {
                Ok(()) => {
                    if let Some(args) = self.entries[idx].get_mut("args") {
                        overrides::remove_at_path(args, &covered);
                    }
                }
                Err(_) => kept.push(covered),
            }
        }
        self.drop_patch_if_empty_or_mark(idx);
        if kept.is_empty() {
            Ok(())
        } else {
            Err(format!("kept as overrides: {}", kept.join(", ")))
        }
    }

    // Strip patch fields equal to their template values, so a legacy full
    // copy becomes the minimal patch (or disappears entirely).
    fn minimize_patch(&mut self, t: &FormTemplate) -> Result<(), String> {
        let idx = self
            .patch_index(&t.name)
            .ok_or("nothing is overridden on this asset")?;
        let baseline = serde_json::Value::Object(t.baseline.clone());
        let patch = self.committed_patch_of(idx);
        let effective = concinnity_cook::world::merge_args(&baseline, &patch);
        match overrides::minimal_patch(&baseline, &effective) {
            Some(p) => {
                if let Some(obj) = self.entries[idx].as_object_mut() {
                    obj.insert("args".to_string(), p);
                }
                self.mark_changed();
            }
            None => self.remove_entry_at(idx),
        }
        Ok(())
    }

    // Author a preset-backed prefab definition as a world line, so its entries
    // become editable and apply-to-template reaches them. The authored line
    // takes precedence over the preset in the expansion.
    fn materialize_prefab(&mut self, name: &str) -> Result<(), String> {
        if self
            .entries
            .iter()
            .any(|e| entry_name(e) == Some(name) && is_prefab(e))
        {
            return Ok(());
        }
        let preset = concinnity_cook::world::preset::load_preset_obj(name, "prefabs");
        if preset.is_null() {
            return Err(format!("no prefab preset named '{name}'"));
        }
        let args = preset.get("args").cloned().unwrap_or(serde_json::json!({}));
        self.entries
            .push(serde_json::json!({ "name": name, "type": "Prefab", "args": args }));
        self.mark_changed();
        Ok(())
    }

    // Cycle the form's scroll window to the next overridden field.
    pub(super) fn jump_to_override(&mut self, world: &mut World) {
        let Some(marks) = self.form_override_marks() else {
            return;
        };
        let marked: Vec<usize> = marks
            .iter()
            .enumerate()
            .filter(|(_, m)| **m != overrides::FieldOrigin::Inherited)
            .map(|(i, _)| i)
            .collect();
        if marked.is_empty() {
            return;
        }
        let next = marked
            .iter()
            .find(|&&i| i > self.form_scroll)
            .or(marked.first())
            .copied()
            .unwrap_or(0);
        let max = self.form_fields.len().saturating_sub(self.form_window());
        self.capture_controls(world);
        self.form_scroll = next.min(max);
        self.form_focus = FormFocus::Name;
        self.refresh_form(world);
    }

    // Helpers

    fn patch_index(&self, name: &str) -> Option<usize> {
        self.entries
            .iter()
            .position(|e| entry_name(e) == Some(name))
    }

    fn committed_patch_of(&self, idx: usize) -> serde_json::Value {
        self.entries[idx]
            .get("args")
            .cloned()
            .unwrap_or(serde_json::json!({}))
    }

    // Resolve where `covered` writes back: the prefab entry slot plus the
    // field mapping for the covered path's root arg.
    fn resolve_slot(
        &self,
        t: &FormTemplate,
        covered: &str,
    ) -> Result<(prefab_map::TemplateSlot, prefab_map::FieldMap), String> {
        let ty = self
            .selected_type
            .clone()
            .ok_or("no form type".to_string())?;
        let root = covered.split('.').next().unwrap_or(covered);
        let map = prefab_map::map_field(&ty, root)?;
        let slot = prefab_map::resolve(&self.entries, &t.generated_by, &t.name)?;
        Ok((slot, map))
    }

    fn write_template_field(
        &mut self,
        slot: &prefab_map::TemplateSlot,
        map: prefab_map::FieldMap,
        covered: &str,
        value: &serde_json::Value,
    ) -> Result<(), String> {
        let entry = self
            .entries
            .get_mut(slot.def_index)
            .and_then(|e| e.get_mut("args"))
            .and_then(|a| a.get_mut("props"))
            .and_then(|p| p.get_mut(slot.entry_index))
            .ok_or_else(|| format!("prefab '{}' changed underneath the form", slot.def_name))?;
        prefab_map::write_field(entry, slot, map, covered, value)
    }

    // After a patch mutation: an emptied patch line is removed outright
    // (`remove_entry_at` records the undo step), else the shrink itself is the
    // recorded edit. Either way, exactly one history step.
    fn drop_patch_if_empty_or_mark(&mut self, idx: usize) {
        let empty = self.entries[idx]
            .get("args")
            .and_then(|a| a.as_object())
            .is_none_or(|o| o.is_empty());
        if empty {
            self.remove_entry_at(idx);
        } else {
            self.mark_changed();
        }
    }

    fn instance_prefab_ref(&self, generated_by: &str) -> Option<String> {
        self.entries
            .iter()
            .find(|e| entry_name(e) == Some(generated_by))
            .and_then(|e| e.get("args"))
            .and_then(|a| a.get("prefab"))
            .and_then(|v| v.as_str())
            .filter(|r| !r.is_empty())
            .map(str::to_string)
    }
}

fn is_prefab(e: &serde_json::Value) -> bool {
    entry_type(e).is_some_and(|t| t.to_lowercase().replace('_', "") == "prefab")
}

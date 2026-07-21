// src/editor/hook/lighting_edit.rs
//
// EditorHook: the Lighting panel's actions. The panel is a curated view over
// the first world entry of each lighting section's asset type (`lighting.rs`);
// every commit flows through `form::assemble` + `form::validate`, the same
// path the add / edit form uses, so nothing invalid ever reaches the entries.
// Checkbox toggles commit immediately (parse-free, so the live preview updates
// on the click); text fields commit together on Apply, all-or-nothing across
// the touched assets.

use super::*;

impl EditorHook {
    // The `entries` index of the first entry of type `ty` (the lighting assets
    // are effectively singletons; extra instances are not surfaced here).
    pub(super) fn entry_index_of(&self, ty: &str) -> Option<usize> {
        self.entries.iter().position(|e| entry_type(e) == Some(ty))
    }

    // The authored args of entry `idx` (empty when the entry carries none).
    pub(super) fn entry_args(&self, idx: usize) -> serde_json::Map<String, serde_json::Value> {
        self.entries[idx]
            .get("args")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default()
    }

    // Which lighting sections have a backing entry in the world.
    pub(super) fn lighting_present(&self) -> Vec<bool> {
        lighting::SECTIONS
            .iter()
            .map(|s| self.entry_index_of(s.ty).is_some())
            .collect()
    }

    // The per-binding derived fields (`None` while a section's asset is absent),
    // indexed by the global binding index so the control pool stays stable as
    // sections come and go.
    pub(super) fn lighting_fields(&self) -> Vec<Option<FormField>> {
        let mut out = Vec::with_capacity(lighting::binding_count());
        for s in lighting::SECTIONS {
            match self.entry_index_of(s.ty) {
                Some(idx) => {
                    let args = self.entry_args(idx);
                    let fields = lighting::section_fields(s, Some(&args));
                    for (path, _) in s.fields {
                        out.push(fields.iter().find(|f| &f.key == path).cloned());
                    }
                }
                None => out.extend(std::iter::repeat_with(|| None).take(s.fields.len())),
            }
        }
        out
    }

    pub(super) fn lighting_data(&self) -> LightingData {
        LightingData {
            rows: lighting::rows(&self.lighting_present()),
            fields: self.lighting_fields(),
        }
    }

    pub(super) fn make_lighting_view<'a>(
        &'a self,
        d: &'a LightingData,
        mouse: [f32; 2],
    ) -> LightingView<'a> {
        LightingView {
            rows: &d.rows,
            fields: &d.fields,
            // Assert keyboard focus only while the panel is frontmost, so its
            // inputs never fight another panel's focused field for typed keys.
            focus: self
                .lighting_focus
                .filter(|_| self.panel_order.last() == Some(&PanelKey::Lighting)),
            status: self.lighting_status.as_deref(),
            mouse,
        }
    }

    // Seed every text control from the current entries: on open, after adding a
    // section's asset, and after a successful Apply. Deliberately NOT on every
    // external entry change, so in-progress edits are never clobbered mid-type.
    pub(super) fn seed_lighting(&mut self, world: &mut World) {
        for (b, field) in self.lighting_fields().iter().enumerate() {
            if let Some(f) = field
                && f.kind.has_text_input()
            {
                widget::seed_field(world, lighting_panel::input(b), &f.initial);
            }
        }
    }

    // Route a resolved Lighting-panel click.
    pub(super) fn apply_lighting_action(&mut self, action: LightingAction, world: &mut World) {
        match action {
            LightingAction::Focus(b) => self.lighting_focus = Some(b),
            LightingAction::Toggle(b) => self.toggle_lighting_bool(b),
            LightingAction::Add(s) => self.add_lighting_section(s, world),
            LightingAction::Apply => self.apply_lighting(world),
            // A click on panel chrome blurs the focused field.
            LightingAction::Consume => self.lighting_focus = None,
        }
    }

    // Capture every present section's text controls, coerce + validate per
    // asset, and commit all of them together; the first rejection shows on the
    // status line and commits nothing.
    pub(super) fn apply_lighting(&mut self, world: &mut World) {
        self.lighting_status = None;
        let mut staged = Vec::new();
        for (si, s) in lighting::SECTIONS.iter().enumerate() {
            let Some(idx) = self.entry_index_of(s.ty) else {
                continue;
            };
            let existing = self.entry_args(idx);
            let fields = lighting::section_fields(s, Some(&existing));
            let base = lighting::section_base(si);
            let texts: Vec<String> = fields
                .iter()
                .map(|f| {
                    let j = s.fields.iter().position(|(p, _)| *p == f.key).unwrap_or(0);
                    if f.kind.has_text_input() {
                        widget::field_text(world, lighting_panel::input(base + j))
                    } else {
                        String::new()
                    }
                })
                .collect();
            let args = form::assemble(s.ty, Some(&existing), &fields, &texts);
            let name = self.entries.get(idx).and_then(entry_name).unwrap_or(s.ty);
            if let Err(e) = form::validate(s.ty, name, &args) {
                self.lighting_status = Some(short_status(&format!("{}: {e}", s.title)));
                return;
            }
            staged.push((idx, args));
        }
        for (idx, args) in staged {
            if let Some(obj) = self.entries.get_mut(idx).and_then(|e| e.as_object_mut()) {
                obj.insert("args".to_string(), serde_json::Value::Object(args));
            }
        }
        self.mark_changed();
        self.seed_lighting(world);
    }

    // Flip a checkbox binding and commit it immediately. The section's text
    // fields are written back at their COMMITTED values (not the live controls),
    // so an in-progress typed edit stays pending in its control.
    pub(super) fn toggle_lighting_bool(&mut self, b: usize) {
        let (si, path, _) = lighting::binding(b);
        let s = &lighting::SECTIONS[si];
        let Some(idx) = self.entry_index_of(s.ty) else {
            return;
        };
        let existing = self.entry_args(idx);
        let mut fields = lighting::section_fields(s, Some(&existing));
        if let Some(f) = fields.iter_mut().find(|f| f.key == path) {
            f.boolval = !f.boolval;
        }
        let merged = form::working_args(s.ty, Some(&existing));
        let texts: Vec<String> = fields
            .iter()
            .map(|f| form::current_text(&merged, &f.key))
            .collect();
        let args = form::assemble(s.ty, Some(&existing), &fields, &texts);
        let name = self.entries.get(idx).and_then(entry_name).unwrap_or(s.ty);
        if let Err(e) = form::validate(s.ty, name, &args) {
            self.lighting_status = Some(short_status(&format!("{}: {e}", s.title)));
            return;
        }
        if let Some(obj) = self.entries.get_mut(idx).and_then(|e| e.as_object_mut()) {
            obj.insert("args".to_string(), serde_json::Value::Object(args));
        }
        self.mark_changed();
    }

    // Add the missing singleton asset behind section `si` with default args,
    // then seed its freshly-appeared controls.
    pub(super) fn add_lighting_section(&mut self, si: usize, world: &mut World) {
        let Some(s) = lighting::SECTIONS.get(si) else {
            return;
        };
        if self.entry_index_of(s.ty).is_some() {
            return;
        }
        let name = self.unique_name(s.ty);
        self.entries.push(serde_json::json!({
            "name": name, "type": s.ty, "args": {},
        }));
        self.mark_changed();
        self.seed_lighting(world);
    }
}

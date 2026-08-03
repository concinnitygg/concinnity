// src/editor/hook/browse.rs
//
// EditorHook: the Assets browse panel and header combo, plus the Templates
// panel's actions (including applying a template and its detail view).

use super::*;

impl EditorHook {
    // Apply every entry of engine-owned template `i`, skipping any whose name
    // already exists (so re-applying is idempotent). Marks dirty if anything was
    // added.
    pub(super) fn apply_template(&mut self, i: usize) {
        let Some(t) = concinnity_templates::TEMPLATES.get(i) else {
            return;
        };
        // The template's typed specs become world-line entries via the app bridge;
        // no JSON string is parsed here.
        let entries = crate::world_template_entries(t);
        let mut added = 0;
        for entry in entries {
            if entry_name(&entry).is_some_and(|n| self.name_taken(n)) {
                continue;
            }
            self.entries.push(entry);
            added += 1;
        }
        if added > 0 {
            self.mark_changed();
            tracing::info!("editor: applied template '{}' ({added} asset(s))", t.name);
        }
    }

    // The per-tick Assets-panel data: the flattened tree, the type picker's
    // options while it is open, and the form heading -- all derived from the
    // cooked model plus the live search field, then borrowed for both
    // hit-testing and layout.
    pub(super) fn panel_data(&self, world: &World) -> PanelData {
        let mut form_title = match (self.form_target.is_edit(), &self.selected_type) {
            (true, Some(t)) => format!("Edit {t}"),
            (false, Some(t)) => format!("New {t}"),
            _ => "New asset".to_string(),
        };
        if self.form_touched {
            form_title.push_str(" *");
        }
        PanelData {
            rows: self.tree_rows(world),
            picker_options: self.picker_options(world),
            form_title,
            form_overrides: self.form_overrides_data(),
        }
    }

    // The open form's override state (marks, summary count, any open menu's
    // option labels), owned for the tick. `None` outside a template-derived
    // form.
    fn form_overrides_data(&self) -> Option<FormOverridesData> {
        let marks = self.form_override_marks()?;
        let count = marks
            .iter()
            .filter(|m| **m != overrides::FieldOrigin::Inherited)
            .count();
        Some(FormOverridesData {
            field_menu: self.override_menu.map(|i| {
                let labels = self
                    .override_menu_options(i)
                    .into_iter()
                    .map(|(_, l)| l)
                    .collect();
                (i, labels)
            }),
            entity_menu: self.entity_menu_open.then(|| {
                self.entity_menu_options()
                    .into_iter()
                    .map(|(_, l)| l)
                    .collect()
            }),
            marks,
            count,
        })
    }

    pub(super) fn make_form_view<'a>(&'a self, d: &'a PanelData, mouse: [f32; 2]) -> FormView<'a> {
        FormView {
            title: &d.form_title,
            editing: self.form_target.is_edit(),
            form_fields: &self.form_fields,
            form_scroll: self.form_scroll,
            form_focus: self.form_focus,
            field_dropdown: self.field_dropdown,
            field_dropdown_scroll: self.field_dropdown_scroll,
            form_error: self.form_error.as_deref(),
            overrides: d
                .form_overrides
                .as_ref()
                .map(|ovr| form_panel::OverridesView {
                    marks: &ovr.marks,
                    count: ovr.count,
                    field_menu: ovr
                        .field_menu
                        .as_ref()
                        .map(|(i, opts)| (*i, opts.as_slice())),
                    entity_menu: ovr.entity_menu.as_deref(),
                }),
            mouse,
        }
    }

    // Action handling

    // Route a resolved top-bar click: SAVE persists to disk (the live preview is
    // already current), the View button toggles the View panel.
    pub(super) fn apply_top(&mut self, action: HudAction, world: &mut World) {
        match action {
            HudAction::Save => self.save(),
            HudAction::Undo => self.undo(world),
            HudAction::Redo => self.redo(world),
            HudAction::ToggleView => self.view_open = !self.view_open,
            HudAction::ToggleDisplay => self.toggle_display_menu(),
            HudAction::PlayPause => self.sim_toggle_play(),
            HudAction::Step => self.sim.step(),
            HudAction::Stop => self.sim_stop(),
            // A bar click that hit no chip: swallowed (the caller already
            // dismisses any open overlays).
            HudAction::Consume => {}
        }
    }

    // Toggle the whole assets UI (the tree panel plus any open edit form).
    // Hiding it KEEPS all that state -- panel positions, the open form, the
    // scroll offset, the fold state, the selection -- so toggling back restores
    // the same view. Only the transient picker / row-menu overlays are dropped.
    pub(super) fn toggle_assets(&mut self) {
        self.panel_open = !self.panel_open;
        self.picker_open = false;
        self.row_menu = None;
    }

    // Open (or re-target) the Template detail panel on template `i` (a preview of
    // the assets it would add, with an Apply button), bringing it to the front of
    // the focus stack. The Templates list stays open so another can be picked.
    pub(super) fn open_template_detail(&mut self, i: usize) {
        if i >= concinnity_templates::TEMPLATES.len() {
            return;
        }
        self.open_template = Some(i);
        self.template_list_scroll = 0;
        self.focus_panel(PanelKey::TemplateDetail);
    }

    // Close the Template detail panel (its state is transient; the Templates list
    // stays as it was).
    pub(super) fn close_template_detail(&mut self) {
        self.open_template = None;
        self.template_list_scroll = 0;
    }

    // Route a resolved Template-detail click: Apply layers the template's assets
    // (idempotently) then closes the panel; the "X" just closes it.
    pub(super) fn apply_template_detail(&mut self, action: TemplateAction) {
        match action {
            TemplateAction::Apply => {
                if let Some(i) = self.open_template {
                    self.apply_template(i);
                }
                self.close_template_detail();
            }
            TemplateAction::Close => self.close_template_detail(),
            TemplateAction::Consume => {}
        }
    }

    // The per-frame Template detail view data (title, description, grouped rows),
    // borrowed for both hit-testing and layout.
    pub(super) fn template_detail_data(&self, i: usize) -> TemplateDetailData {
        let (title, description) = concinnity_templates::TEMPLATES
            .get(i)
            .map(|t| (format!("Template {}", t.title), t.description.to_string()))
            .unwrap_or_default();
        TemplateDetailData {
            title,
            description,
            rows: self.template_rows(i),
        }
    }

    pub(super) fn make_template_view<'a>(
        &self,
        d: &'a TemplateDetailData,
        mouse: [f32; 2],
    ) -> TemplateView<'a> {
        TemplateView {
            title: &d.title,
            description: &d.description,
            rows: &d.rows,
            scroll: self.template_list_scroll,
            mouse,
        }
    }
}

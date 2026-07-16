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

    // -- Option lists (derived from the entries + the live filter field) --------

    // The distinct asset types present in the world, sorted.
    pub(super) fn distinct_types(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for e in &self.entries {
            if let Some(ty) = entry_type(e)
                && !out.iter().any(|s| s == ty)
            {
                out.push(ty.to_string());
            }
        }
        out.sort();
        out
    }

    // The browse list, grouped by type (the shared model): a sub-header row per
    // type matching the active filter, then an indented row per asset name
    // carrying its entry index (for the Delete / edit menu). Types + names sorted
    // alphabetically, identical to the Template panel's list.
    pub(super) fn list_rows(&self) -> Vec<ListRow> {
        super::asset_list::grouped_rows(&self.entries, self.type_filter.as_deref())
    }

    // The floating combo option list for the open flavour, narrowed by the typed
    // filter field. Empty when the combo is closed.
    pub(super) fn combo_options(&self, world: &World) -> Vec<String> {
        let filter = widget::field_text(world, panel::FILTER_INPUT).to_lowercase();
        let matches = |s: &str| filter.is_empty() || s.to_lowercase().contains(&filter);
        match self.combo {
            Combo::Filter => {
                let mut all = vec![panel::ALL_LABEL.to_string()];
                all.extend(self.distinct_types());
                all.into_iter().filter(|o| matches(o)).collect()
            }
            Combo::Picker => {
                let mut opts: Vec<String> = panel::picker_types()
                    .filter(|t| matches(t))
                    .map(|t| t.to_string())
                    .collect();
                // The picker lists the offered types alphabetically (ascending).
                opts.sort();
                opts
            }
            Combo::Closed => Vec::new(),
        }
    }

    pub(super) fn panel_data(&self, world: &World) -> PanelData {
        let combo_options = self.combo_options(world);
        let combo_selected = match self.combo {
            Combo::Filter => {
                let target = self
                    .type_filter
                    .clone()
                    .unwrap_or_else(|| panel::ALL_LABEL.to_string());
                combo_options.iter().position(|o| o == &target)
            }
            _ => None,
        };
        let form_title = match (&self.editing, &self.selected_type) {
            (Some(_), Some(t)) => format!("Edit {t}"),
            (None, Some(t)) => format!("New {t}"),
            _ => "New asset".to_string(),
        };
        PanelData {
            filter_label: self
                .type_filter
                .clone()
                .unwrap_or_else(|| panel::ALL_LABEL.to_string()),
            combo_options,
            combo_selected,
            list_rows: self.list_rows(),
            expanded_rows: self.expanded_rows(),
            form_title,
        }
    }

    pub(super) fn make_view<'a>(&'a self, d: &'a PanelData, mouse: [f32; 2]) -> PanelView<'a> {
        PanelView {
            tab: self.tab,
            expanded_rows: &d.expanded_rows,
            expanded_scroll: self.expanded_scroll,
            expanded_status: self.expanded_status.as_deref(),
            combo: self.combo,
            filter_label: &d.filter_label,
            combo_options: &d.combo_options,
            combo_selected: d.combo_selected,
            combo_scroll: self.combo_scroll,
            list_rows: &d.list_rows,
            list_scroll: self.list_scroll,
            row_menu: self.row_menu,
            // The entry whose form is open keeps its browse row highlighted.
            selected: self.editing,
            mouse,
        }
    }

    pub(super) fn make_form_view<'a>(&'a self, d: &'a PanelData, mouse: [f32; 2]) -> FormView<'a> {
        FormView {
            title: &d.form_title,
            editing: self.editing.is_some(),
            form_fields: &self.form_fields,
            form_scroll: self.form_scroll,
            form_focus: self.form_focus,
            field_dropdown: self.field_dropdown,
            field_dropdown_scroll: self.field_dropdown_scroll,
            form_error: self.form_error.as_deref(),
            mouse,
        }
    }

    // -- Action handling --------------------------------------------------------

    // Route a resolved top-bar click: SAVE persists to disk (the live preview is
    // already current), the View button toggles the View panel.
    pub(super) fn apply_top(&mut self, action: HudAction) {
        match action {
            HudAction::Save => self.save(),
            HudAction::ToggleView => self.view_open = !self.view_open,
        }
    }

    // Toggle the whole assets UI (browse panel + any open edit form + the browse
    // highlight). Hiding it KEEPS all that state -- panel positions, the open
    // form, the scroll offset, the selection -- so toggling back restores the same
    // view. Only the transient combo / row-menu overlays are dropped.
    pub(super) fn toggle_assets(&mut self) {
        self.panel_open = !self.panel_open;
        self.combo = Combo::Closed;
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

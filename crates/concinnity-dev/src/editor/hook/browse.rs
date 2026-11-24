//! `EditorHook`: the Assets browse panel and header combo, plus the Templates
//! panel's actions (including applying a template and its detail view).

use concinnity_engine::components::UiEvent;
use concinnity_engine::ecs::World;

use super::form_state::FormDataKey;
use super::{EditorHook, FormData, FormOverridesData, FormTarget, declared_id};
use crate::editor::entry_list::paths;
use crate::editor::hud::{self, HudAction};
use crate::editor::overrides;
use crate::editor::panels::form_panel::{self, FormMenu, FormView};
use crate::editor::panels::registry::PanelKey;
use crate::editor::ui::BuiltPanel;

impl EditorHook {
    // Apply every entry of engine-owned template `i`, skipping any whose name
    // already exists (so re-applying is idempotent). Marks dirty if anything was
    // added.
    pub(super) fn apply_template(&mut self, i: usize) {
        let Some(t) = concinnity_cook::authoring::template::TEMPLATES.get(i) else {
            return;
        };
        // The template's typed specs become world-line entries via the app bridge;
        // no JSON string is parsed here.
        let entries = crate::authoring::world_template_entries(t);
        let mut added = 0;
        for entry in entries {
            if declared_id(&entry).is_some_and(|n| self.name_taken(n)) {
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

    // The per-tick add / edit form data: its heading, override state, extra
    // rows, and why it cannot confirm.
    pub(super) fn form_data(&self) -> FormData {
        let mut form_title = match (self.form.target.is_edit(), &self.form.selected_type) {
            (true, Some(t)) => format!("Edit {t}"),
            (false, Some(t)) => format!("New {t}"),
            _ => "New asset".to_string(),
        };
        if let Some(file) = self.form_entry_file() {
            form_title = format!("{form_title} ({file})");
        }
        if self.form.touched {
            form_title.push_str(" *");
        }
        let (form_extras, form_blocked) = self.form_extras_data();
        FormData {
            title: form_title,
            overrides: self.form_overrides_data(),
            extras: form_extras,
            blocked: form_blocked,
        }
    }

    // The included file the open form's entry lives in, as the title shows it.
    // `None` for an entry of the world file and for a form on no entry.
    fn form_entry_file(&self) -> Option<String> {
        let FormTarget::Entry(key) = self.form.target else {
            return None;
        };
        let file = self.entries.included_from(self.entries.index_of(key)?)?;
        Some(paths::shown(file))
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
            field_menu: self.form.override_menu.map(|i| {
                let labels = self
                    .override_menu_options(i)
                    .into_iter()
                    .map(|(_, l)| l)
                    .collect();
                (i, labels)
            }),
            entity_menu: self.form.entity_menu_open.then(|| {
                self.entity_menu_options()
                    .into_iter()
                    .map(|(_, l)| l)
                    .collect()
            }),
            marks,
            count,
        })
    }

    // Bring the form's derived data and its rows up to date: the data when
    // the entries, the fields, the typed name, or an action changed it, and
    // the rows when the fields or the extra rows changed shape.
    pub(super) fn prepare_form(&mut self, world: &mut World, built: &mut BuiltPanel) {
        let key = FormDataKey {
            entries: self.entries.revision(),
            rows: self.form.rows_rev,
            actions: self.form.data_rev,
            touched: self.form.touched,
        };
        let current = self
            .form
            .data
            .as_ref()
            .is_some_and(|(k, name, _)| *k == key && *name == self.form.name);
        if !current {
            self.form.data = Some((key, self.form.name.clone(), self.form_data()));
        }
        self.form.reveal_now = self.form.reveal.take();
        let extras = self
            .form
            .data
            .as_ref()
            .map_or(&[][..], |(_, _, d)| &d.extras);
        let fresh = self
            .form
            .shown
            .as_ref()
            .is_some_and(|s| s.rev == self.form.rows_rev && s.extras_match(extras));
        if !fresh {
            let shown = form_panel::rebuild_rows(
                world,
                built,
                &self.form.fields,
                extras,
                (self.form.rows_rev, self.form.opened),
                self.form.shown.as_ref(),
            );
            self.form.shown = Some(shown);
        }
    }

    // The menu the form has open, and its options.
    fn form_menu<'a>(&'a self, data: &'a FormData) -> Option<(FormMenu, &'a [String])> {
        if let Some(i) = self.form.field_dropdown {
            let options = self.form.fields.get(i)?.variants.as_slice();
            return Some((FormMenu::Field(i), options));
        }
        let ovr = data.overrides.as_ref()?;
        if let Some((i, labels)) = &ovr.field_menu {
            return Some((FormMenu::Override(*i), labels.as_slice()));
        }
        ovr.entity_menu
            .as_deref()
            .map(|labels| (FormMenu::Entity, labels))
    }

    // Which menu the form has open.
    pub(super) fn open_form_menu(&self) -> Option<FormMenu> {
        if let Some(i) = self.form.field_dropdown {
            return Some(FormMenu::Field(i));
        }
        if let Some(i) = self.form.override_menu {
            return Some(FormMenu::Override(i));
        }
        self.form.entity_menu_open.then_some(FormMenu::Entity)
    }

    // What the form shows this frame.
    pub(super) fn make_form_view(&self) -> Option<FormView<'_>> {
        let (_, _, data) = self.form.data.as_ref()?;
        Some(FormView {
            title: &data.title,
            editing: self.form.target.is_edit(),
            fields: &self.form.fields,
            texts: &self.form.texts,
            name: &self.form.name,
            // Focus is asserted only while frontmost, matching the other
            // panels' guard against fighting for typed keys.
            focus: (self.panel_order.last() == Some(&PanelKey::Edit)).then_some(self.form.focus),
            error: self.form.error.as_deref(),
            blocked: data.blocked.as_deref(),
            marks: data
                .overrides
                .as_ref()
                .map(|o| (o.marks.as_slice(), o.count)),
            extras: &data.extras,
            menu: self.form_menu(data),
            reveal: self.form.reveal_now,
        })
    }

    // Take in what the user typed into the form, the field they focused, and
    // a menu a press off it closed.
    pub(super) fn observe_form(&mut self, world: &World, built: &BuiltPanel) {
        let Some(shown) = &self.form.shown else {
            return;
        };
        let (_, focus) = form_panel::observe(
            world,
            built,
            shown,
            &mut self.form.name,
            &mut self.form.texts,
        );
        if let Some(focus) = focus {
            self.form.focus = focus;
        }
        if self.open_form_menu().is_some() && !form_panel::menu_open(world) {
            self.close_form_menus();
        }
    }

    // Action handling

    // Route a resolved top-bar click: SAVE persists to disk (the live preview is
    // already current), the View button toggles the View panel.
    pub(super) fn apply_top(&mut self, action: HudAction) {
        match action {
            HudAction::Save => self.save(),
            HudAction::Undo => self.undo(),
            HudAction::Redo => self.redo(),
            HudAction::ToggleView => self.open[PanelKey::View] = !self.open[PanelKey::View],
            HudAction::ToggleDisplay => self.toggle_display_menu(),
            HudAction::PlayPause => self.sim_toggle_play(),
            HudAction::Step => self.sim.step(),
            HudAction::Stop => self.sim_stop(),
        }
    }

    // A top-bar chip's click; whether `event` was one.
    pub(super) fn on_bar_event(&mut self, event: &UiEvent) -> bool {
        let Some(action) = hud::action(event) else {
            return false;
        };
        self.apply_top(action);
        self.assets.picker_open = false;
        self.assets.row_menu = None;
        true
    }

    // Open (or re-target) the Template detail panel on template `i` (a preview of
    // the assets it would add, with an Apply button), bringing it to the front of
    // the focus stack. The Templates list stays open so another can be picked.
    pub(super) fn open_template_detail(&mut self, i: usize) {
        if i >= concinnity_cook::authoring::template::TEMPLATES.len() {
            return;
        }
        self.templates.detail = Some(i);
        self.focus_panel(PanelKey::TemplateDetail);
    }

    // Close the Template detail panel (its state is transient; the Templates list
    // stays as it was).
    pub(super) fn close_template_detail(&mut self) {
        self.templates.detail = None;
    }

    // Layer the open template's assets (idempotently), then close its detail
    // panel.
    pub(super) fn apply_open_template(&mut self) {
        if let Some(i) = self.templates.detail {
            self.apply_template(i);
        }
        self.close_template_detail();
    }
}

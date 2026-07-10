// src/editor/hook/layout.rs
//
// EditorHook: the floating panels' focus stack, published draw layers, and
// each panel's on-screen origin.

use super::*;

impl EditorHook {
    // Bring a panel to the front of the focus stack (drawn on top, first to be
    // clicked). A no-op if it is already frontmost.
    pub(super) fn focus_panel(&mut self, target: DragTarget) {
        self.panel_order.retain(|&p| p != target);
        self.panel_order.push(target);
    }

    // Every injected element id of a panel, for the `HudLayers` layer map.
    pub(super) fn panel_ids(target: DragTarget) -> Vec<AssetId> {
        match target {
            DragTarget::Assets => panel::all_sprite_ids()
                .into_iter()
                .chain(panel::all_label_ids())
                .chain(panel::all_field_ids())
                .collect(),
            DragTarget::Edit => form_panel::all_sprite_ids()
                .into_iter()
                .chain(form_panel::all_label_ids())
                .chain(form_panel::all_field_ids())
                .collect(),
            DragTarget::Preview => preview::all_sprite_ids()
                .into_iter()
                .chain(preview::all_label_ids())
                .collect(),
            DragTarget::View => view::all_sprite_ids()
                .into_iter()
                .chain(view::all_label_ids())
                .collect(),
            DragTarget::Templates => templates::all_sprite_ids()
                .into_iter()
                .chain(templates::all_label_ids())
                .collect(),
            DragTarget::TemplateDetail => template_panel::all_sprite_ids()
                .into_iter()
                .chain(template_panel::all_label_ids())
                .collect(),
        }
    }

    // The per-frame HUD draw layers: each panel at its focus-stack rank (1..=3,
    // higher = more front), the top bar pinned above them all.
    pub(super) fn compute_layers(&self) -> std::collections::HashMap<AssetId, i32> {
        let mut layers = std::collections::HashMap::new();
        for (rank, &target) in self.panel_order.iter().enumerate() {
            let layer = rank as i32 + 1;
            for id in Self::panel_ids(target) {
                layers.insert(id, layer);
            }
        }
        for id in hud::all_ids() {
            layers.insert(id, TOP_BAR_LAYER);
        }
        layers
    }

    // Publish the draw layers for the renderer's overlay sort, so a dragged /
    // clicked panel occludes the rest instead of its text bleeding through their
    // backgrounds.
    pub(super) fn publish_layers(&self, world: &mut World) {
        world.insert_resource(HudLayers(self.compute_layers()));
    }

    // Whether the add / edit form panel is open.
    pub(super) fn form_open(&self) -> bool {
        self.selected_type.is_some()
    }

    // The Assets panel's top-left for this frame: the dragged position (or the
    // default anchor below the top bar), clamped so the whole panel stays on
    // screen even after a window resize.
    pub(super) fn panel_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        let pos = self.panel_pos.unwrap_or(panel::default_origin(vp[0]));
        widget::clamp_origin(pos, panel::size(), vp)
    }

    // The edit-form panel's top-left for this frame, clamped at its current
    // height (the field area tracks the open type's field count).
    pub(super) fn edit_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        let pos = self.edit_pos.unwrap_or(form_panel::default_origin(vp[0]));
        widget::clamp_origin(pos, form_panel::size(self.form_fields.len()), vp)
    }

    // The Preview panel's top-left for this frame, clamped like the others.
    pub(super) fn preview_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        let pos = self.preview_pos.unwrap_or(preview::default_origin());
        widget::clamp_origin(pos, preview::size(), vp)
    }

    // The View panel's top-left for this frame, clamped like the others.
    pub(super) fn view_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        let pos = self.view_pos.unwrap_or(view::default_origin());
        widget::clamp_origin(pos, view::size(), vp)
    }

    // The Templates panel's top-left for this frame, clamped like the others.
    pub(super) fn templates_origin(&self, vp: [f32; 2]) -> [f32; 2] {
        let pos = self
            .templates_pos
            .unwrap_or(templates::default_origin(vp[0]));
        widget::clamp_origin(pos, templates::size(), vp)
    }

    // The Template detail panel's top-left for this frame, clamped at its current
    // height (the list area tracks the open template's asset-row count).
    pub(super) fn template_detail_origin(&self, i: usize, vp: [f32; 2]) -> [f32; 2] {
        let n = self.template_rows(i).len();
        let pos = self
            .template_detail_pos
            .unwrap_or(template_panel::default_origin(vp[0]));
        widget::clamp_origin(pos, template_panel::size(n), vp)
    }

    // The world-line entries of template `i` (its typed specs via the app bridge).
    pub(super) fn template_entries(&self, i: usize) -> Vec<serde_json::Value> {
        concinnity_templates::TEMPLATES
            .get(i)
            .map(concinnity_app::world_template_entries)
            .unwrap_or_default()
    }

    // Template `i`'s assets as the shared grouped rows (types + names alphabetical,
    // identical to the Assets panel's list).
    pub(super) fn template_rows(&self, i: usize) -> Vec<ListRow> {
        super::asset_list::grouped_rows(&self.template_entries(i), None)
    }

    // Which panels are currently shown, so the View panel's checkboxes reflect it.
    pub(super) fn view_state(&self) -> ViewState {
        ViewState {
            assets: self.panel_open,
            preview: self.preview_open,
            templates: self.templates_open,
        }
    }
}

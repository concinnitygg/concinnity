// src/editor/hook/layout.rs
//
// EditorHook: the floating panels' focus stack, published draw layers, and
// each panel's on-screen origin -- all derived from the panel registry.

use super::*;

impl EditorHook {
    // Bring a panel to the front of the focus stack (drawn on top, first to be
    // clicked). A no-op if it is already frontmost.
    pub(super) fn focus_panel(&mut self, key: PanelKey) {
        self.panel_order.retain(|&p| p != key);
        self.panel_order.push(key);
    }

    // Every injected element id of a panel, for the `HudLayers` layer map.
    pub(super) fn panel_ids(key: PanelKey) -> Vec<AssetId> {
        let p = registry::panel(key);
        p.sprite_ids()
            .into_iter()
            .chain(p.label_ids())
            .chain(p.field_ids().into_iter().map(|(id, _)| id))
            .collect()
    }

    // The per-frame HUD draw layers: each panel at its focus-stack rank (higher
    // = more front), the top bar pinned above them all.
    pub(super) fn compute_layers(&self) -> std::collections::BTreeMap<AssetId, i32> {
        let mut layers = std::collections::BTreeMap::new();
        for (rank, &key) in self.panel_order.iter().enumerate() {
            let layer = rank as i32 + 1;
            for id in Self::panel_ids(key) {
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

    // A panel's top-left for this frame: the dragged position (or its default
    // anchor), clamped at its current footprint so the whole panel stays on
    // screen even after a window resize.
    pub(super) fn origin(&self, key: PanelKey, vp: [f32; 2]) -> [f32; 2] {
        let p = registry::panel(key);
        let pos = self.positions[key.index()].unwrap_or_else(|| p.default_origin(vp));
        widget::clamp_origin(pos, p.size(self), vp, hud::BAR_H)
    }

    // The world-line entries of template `i` (its typed specs via the app bridge).
    pub(super) fn template_entries(&self, i: usize) -> Vec<serde_json::Value> {
        concinnity_templates::TEMPLATES
            .get(i)
            .map(crate::world_template_entries)
            .unwrap_or_default()
    }

    // Template `i`'s assets as the shared grouped rows (types + names alphabetical,
    // identical to the Assets panel's list).
    pub(super) fn template_rows(&self, i: usize) -> Vec<ListRow> {
        super::asset_list::grouped_rows(&self.template_entries(i), None)
    }

    // The View panel's toggle rows: one checkbox per registered panel that opts
    // in (`Panel::view_row`), reflecting its shown state.
    pub(super) fn view_rows(&self) -> Vec<Row> {
        registry::view_toggles()
            .map(|p| Row::checkbox(p.view_row().unwrap_or(""), p.is_open(self)))
            .collect()
    }

    // Flip the panel behind View-panel toggle row `i`.
    pub(super) fn toggle_view_row(&mut self, i: usize, world: &mut World) {
        if let Some(p) = registry::view_toggles().nth(i) {
            p.toggle(self, world);
        }
    }
}

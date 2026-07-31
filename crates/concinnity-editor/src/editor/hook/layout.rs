// src/editor/hook/layout.rs
//
// EditorHook: the floating panels' focus stack, published draw layers, and
// each panel's on-screen origin -- all derived from the panel registry.

use super::*;

// Layers each panel's rank claims, so an overlay can draw above its own panel
// without reaching the one in front of it.
const PANEL_LAYER_SPAN: i32 = 10;
// Where an open overlay sits inside its panel's band.
const OVERLAY_LAYER: i32 = 5;

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
    // = more front), the top bar pinned above them all. A panel's rank is a band
    // rather than a single layer so its overlays can sit above it.
    pub(super) fn compute_layers(&self) -> std::collections::BTreeMap<AssetId, i32> {
        let mut layers = std::collections::BTreeMap::new();
        for (rank, &key) in self.panel_order.iter().enumerate() {
            // Ranks are spaced so each panel owns a band of layers: its own
            // elements at the bottom of the band and any open overlay above
            // them, while the panel in front still clears the whole band.
            let layer = (rank as i32 + 1) * PANEL_LAYER_SPAN;
            for id in Self::panel_ids(key) {
                layers.insert(id, layer);
            }
            for id in registry::panel(key).overlay_ids(self) {
                layers.insert(id, layer + OVERLAY_LAYER);
            }
        }
        for id in hud::all_ids() {
            layers.insert(id, TOP_BAR_LAYER);
        }
        // The create and Display menus are modal while open, so they clear
        // everything -- panels and top bar included (their ids are hidden
        // otherwise).
        for id in create_menu::all_sprite_ids()
            .into_iter()
            .chain(create_menu::all_label_ids())
            .chain(view_menu::all_sprite_ids())
            .chain(view_menu::all_label_ids())
        {
            layers.insert(id, TOP_BAR_LAYER + PANEL_LAYER_SPAN);
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
        widget::clamp_origin(pos, self.effective_size(key), vp, hud::BAR_H)
    }

    // A panel's content-derived default size, which is also its minimum: the user
    // can grow a panel past this but never below it.
    pub(super) fn default_size(&self, key: PanelKey) -> [f32; 2] {
        registry::panel(key).size(self)
    }

    // A panel's size this frame: the user's override clamped into
    // `[default, max]`, so content that outgrows the override still wins and the
    // panel never exceeds its content pool (a non-resizable panel, never
    // overridden, is always its default).
    pub(super) fn effective_size(&self, key: PanelKey) -> [f32; 2] {
        let d = self.default_size(key);
        match self.sizes[key.index()] {
            Some(o) => {
                let max = registry::panel(key).max_size(self);
                [
                    o[0].clamp(d[0], max[0].max(d[0])),
                    o[1].clamp(d[1], max[1].max(d[1])),
                ]
            }
            None => d,
        }
    }

    // The edges the pointer grabs on resizable panel `key` (at origin `o`, size
    // `s`), with any axis the panel has locked (its max equals its default,
    // e.g. a width-only panel) filtered out so it offers no grab there. `None`
    // when no live edge is under the pointer.
    pub(super) fn resize_edges(
        &self,
        key: PanelKey,
        o: [f32; 2],
        s: [f32; 2],
        mx: f32,
        my: f32,
    ) -> Option<resize::Edges> {
        let e = resize::hit_test(o, s, resize::BORDER, mx, my)?;
        let d = self.default_size(key);
        let max = registry::panel(key).max_size(self);
        let e = resize::Edges {
            left: e.left && max[0] > d[0],
            right: e.right && max[0] > d[0],
            top: e.top && max[1] > d[1],
            bottom: e.bottom && max[1] > d[1],
        };
        e.any().then_some(e)
    }

    // The resize cursor for the pointer at `mouse`: the shape of the frontmost
    // resizable open panel whose edge / corner it grabs, or the arrow when it is
    // over no such edge. Front-to-back, so a panel's body blocks the panels
    // behind it; the close button keeps priority over a top-corner grab.
    pub(super) fn hover_cursor(&self, vp: [f32; 2], mouse: [f32; 2]) -> CursorShape {
        let (mx, my) = (mouse[0], mouse[1]);
        for &key in self.panel_order.iter().rev() {
            let p = registry::panel(key);
            if !p.resizable() || !p.is_open(self) {
                continue;
            }
            let o = self.origin(key, vp);
            let s = self.effective_size(key);
            let title = [o[0], o[1], s[0], widget::TITLE_H];
            if point_in(mx, my, widget::close_rect(title)) {
                return CursorShape::Default;
            }
            if let Some(edges) = self.resize_edges(key, o, s, mx, my) {
                return resize::cursor_shape(edges);
            }
            if point_in(mx, my, [o[0], o[1], s[0], s[1]]) {
                return CursorShape::Default;
            }
        }
        CursorShape::Default
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

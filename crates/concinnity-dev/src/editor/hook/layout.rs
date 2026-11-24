//! `EditorHook`: the floating panels' focus stack, which sets the ranks the
//! engine stacks them by, the resize cursor over their edges, and each panel's
//! on-screen origin -- all derived from the panel registry.

use concinnity_engine::components::{Layout, ResizeEdges};
use concinnity_engine::ecs::World;
use concinnity_engine::protocol::CursorShape;

use super::EditorHook;
use crate::editor::hud;
use crate::editor::panels::asset_list::{self, ListRow};
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::ui::BuiltUi;
use crate::editor::ui::rect::{clamp_origin, point_in};

impl EditorHook {
    // Bring a panel to the front of the focus stack (drawn on top, first to be
    // clicked). A no-op if it is already frontmost.
    pub(super) fn focus_panel(&mut self, key: PanelKey) {
        self.panel_order.retain(|&p| p != key);
        self.panel_order.push(key);
    }

    // Panel `key`'s own open flag, which `Panel::is_open` reads by default.
    pub(crate) fn open_flag(&self, key: PanelKey) -> bool {
        self.open[key]
    }

    pub(crate) fn set_open_flag(&mut self, key: PanelKey, open: bool) {
        self.open[key] = open;
    }

    // Whether the add / edit form panel is open.
    pub(super) fn form_open(&self) -> bool {
        self.form.selected_type.is_some()
    }

    // Whether panel `key` shows and routes this frame. The start screen is the
    // whole session while it is up, so every other panel is suppressed rather
    // than torn down: its state stands, and it comes back the moment a world
    // is open.
    pub(super) fn panel_shown(&self, key: PanelKey) -> bool {
        registry::is_open(key, self) && (!self.start_mode || key == PanelKey::Worlds)
    }

    // A panel's top-left this frame: its dragged position or default anchor,
    // clamped to stay on screen. The start screen docks to the left edge, an
    // anchor the registry cannot express.
    pub(super) fn origin(&self, key: PanelKey, vp: [f32; 2]) -> [f32; 2] {
        let (anchor, top) = match (self.start_mode, key) {
            (true, PanelKey::Worlds) => (self.worlds_layout().default_origin(), 0.0),
            _ => (
                self.positions[key].unwrap_or_else(|| registry::panel(key).default_origin(vp)),
                hud::BAR_H,
            ),
        };
        clamp_origin(anchor, self.effective_size(key), vp, top)
    }

    // A panel's content-derived default size, which is also its minimum: the user
    // can grow a panel past this but never below it.
    pub(super) fn default_size(&self, key: PanelKey) -> [f32; 2] {
        registry::panel(key).size(self)
    }

    // A panel's size this frame: the user's override, never below the default,
    // so content that outgrows the override still wins (a panel never sized is
    // always its default).
    pub(super) fn effective_size(&self, key: PanelKey) -> [f32; 2] {
        let d = self.default_size(key);
        match self.sizes[key] {
            Some(o) => [o[0].max(d[0]), o[1].max(d[1])],
            None => d,
        }
    }

    // The resize cursor for the pointer at `mouse`: the edges a panel is being
    // sized by while a drag holds them, else those of the frontmost shown
    // panel's edge under the pointer, else the arrow. Front-to-back, so a
    // panel's body blocks the panels behind it.
    pub(super) fn hover_cursor(&self, world: &World, mouse: [f32; 2]) -> CursorShape {
        // The open confirmation dialog swallows the pointer, so no panel edge
        // offers a resize grab under it.
        if self.modal.is_some() {
            return CursorShape::Default;
        }
        let Some(built) = world.resource::<BuiltUi>() else {
            return CursorShape::Default;
        };
        let roots = || {
            self.panel_order.iter().rev().filter_map(|&key| {
                built
                    .panel(key)
                    .and_then(|p| world.get_by_id::<Layout>(p.root))
                    .filter(|l| l.visible)
            })
        };
        if let Some(edges) = roots().find_map(|l| l.resizing) {
            return resize_cursor(edges);
        }
        for layout in roots() {
            if let Some(edges) = layout.resize_grip(mouse[0], mouse[1]) {
                return resize_cursor(edges);
            }
            if layout.rect.is_some_and(|r| point_in(mouse[0], mouse[1], r)) {
                return CursorShape::Default;
            }
        }
        CursorShape::Default
    }

    // The world-line entries of template `i` (its typed specs via the app bridge).
    pub(super) fn template_entries(i: usize) -> Vec<serde_json::Value> {
        concinnity_cook::authoring::template::TEMPLATES
            .get(i)
            .map(crate::authoring::world_template_entries)
            .unwrap_or_default()
    }

    // Template `i`'s assets as the shared grouped rows (types + names alphabetical,
    // identical to the Assets panel's list).
    pub(super) fn template_rows(i: usize) -> Vec<ListRow> {
        asset_list::grouped_rows(&Self::template_entries(i), None)
    }

    // Flip the panel behind View-panel toggle row `i`.
    pub(super) fn toggle_view_row(&mut self, i: usize, world: &mut World) {
        if let Some(key) = registry::view_toggles().nth(i) {
            registry::toggle(key, self, world);
        }
    }
}

// The cursor for sizing a panel by `edges`.
fn resize_cursor(edges: ResizeEdges) -> CursorShape {
    match (edges.right, edges.bottom) {
        (true, true) => CursorShape::ResizeNWSE,
        (true, false) => CursorShape::ResizeEW,
        _ => CursorShape::ResizeNS,
    }
}

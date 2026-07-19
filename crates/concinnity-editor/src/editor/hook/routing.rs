// src/editor/hook/routing.rs
//
// EditorHook: per-frame pointer routing -- the scroll steps behind each panel's
// wheel region, title-bar dragging, and click hit-testing across the top bar and
// the registered panels (front-to-back).

use super::*;

impl EditorHook {
    // Scroll the Assets panel's browse list, or the combo option list while a
    // combo is open.
    pub(super) fn scroll_list(&mut self, delta: f32, world: &mut World) {
        if self.combo == Combo::Closed {
            let max = self.list_rows().len().saturating_sub(panel::MAX_ROWS);
            self.list_scroll = scroll_step(self.list_scroll, delta, max);
            self.row_menu = None;
        } else {
            let max = self
                .combo_options(world)
                .len()
                .saturating_sub(panel::MAX_ROWS);
            self.combo_scroll = scroll_step(self.combo_scroll, delta, max);
        }
    }

    // Scroll the edit form: an open value dropdown scrolls its own option list;
    // otherwise the field window moves (folding the visible controls into the
    // working args first, so no in-progress edit is lost).
    pub(super) fn scroll_form(&mut self, delta: f32, world: &mut World) {
        if let Some(open) = self.field_dropdown {
            let total = self.form_fields.get(open).map_or(0, |f| f.variants.len());
            let max = total.saturating_sub(form_panel::MAX_DROP_ROWS);
            self.field_dropdown_scroll = scroll_step(self.field_dropdown_scroll, delta, max);
        } else {
            // The same capture / refresh cycle an array add / remove uses.
            let max = self.form_fields.len().saturating_sub(form::FIELD_POOL);
            let next = scroll_step(self.form_scroll, delta, max);
            if next == self.form_scroll {
                return;
            }
            self.capture_controls(world);
            self.form_scroll = next;
            self.form_focus = FormFocus::Name;
            self.refresh_form(world);
        }
    }

    // Scroll the Template detail panel's asset list.
    pub(super) fn scroll_template_list(&mut self, delta: f32) {
        if let Some(i) = self.open_template {
            let max = self
                .template_rows(i)
                .len()
                .saturating_sub(super::asset_list::MAX_ROWS);
            self.template_list_scroll = scroll_step(self.template_list_scroll, delta, max);
        }
    }

    pub(super) fn hud_state(&self) -> HudState {
        HudState {
            dirty: self.dirty,
            undo: self.can_undo(),
            redo: self.can_redo(),
            view_open: self.view_open,
            visible: self.hud_visible,
        }
    }

    // While a title-bar drag is active, follow the cursor (clamped fully on
    // screen at the panel's current footprint); releasing the button ends the
    // drag.
    pub(super) fn drive_drag(&mut self, input: &FrameInput, vp: [f32; 2]) {
        let Some(drag) = self.drag else {
            return;
        };
        if !input.left_button_down {
            self.drag = None;
            return;
        }
        let pos = [input.mouse_x - drag.grab[0], input.mouse_y - drag.grab[1]];
        let size = registry::panel(drag.key).size(self);
        self.positions[drag.key.index()] = Some(widget::clamp_origin(pos, size, vp));
    }

    // Route a press: the top bar first (it draws over the panels), then the panels
    // front-to-back so the frontmost claims a press in an overlap. Whichever panel
    // claims it comes to the front. A press on a panel's title bar starts a drag.
    pub(super) fn route_click(&mut self, input: &FrameInput, vp: [f32; 2], world: &mut World) {
        let (mx, my) = (input.mouse_x, input.mouse_y);
        if let Some(a) = hud::hit_test(mx, my, true, self.hud_state(), vp[0]) {
            // SAVE only writes to disk now; it neither rebuilds nor re-injects the
            // world, so an open form is left intact (no blank-field risk).
            self.apply_top(a, world);
            self.combo = Combo::Closed;
            self.row_menu = None;
            return;
        }
        // Front-to-back: the frontmost shown panel to claim the press handles it and
        // rises to the front (so clicking an exposed sliver of a buried panel brings
        // it forward). `panel_order`'s tail is frontmost.
        let front_to_back: Vec<PanelKey> = self.panel_order.iter().rev().copied().collect();
        for key in front_to_back {
            if self.try_panel_press(key, mx, my, vp, world) {
                return;
            }
        }
        // Nothing claimed the press: offer it to the 3D view as a pick. Only
        // edit mode gets here (`left_click` stays false while the world holds
        // the cursor in play mode).
        self.click_world(input, world);
    }

    // Try to resolve a press against the panel registered at `key`: `false` when
    // it is hidden or the press misses it (the caller tries the next panel back).
    // A hit brings the panel to the front; the close button closes it, a
    // title-bar press starts a drag, and a body press resolves through the
    // panel's own hit test.
    pub(super) fn try_panel_press(
        &mut self,
        key: PanelKey,
        mx: f32,
        my: f32,
        vp: [f32; 2],
        world: &mut World,
    ) -> bool {
        let p = registry::panel(key);
        if !p.is_open(self) {
            return false;
        }
        let o = self.origin(key, vp);
        let title = [o[0], o[1], p.size(self)[0], widget::TITLE_H];
        // The X is checked before the title bar so it never starts a drag instead.
        if point_in(mx, my, widget::close_rect(title)) {
            self.focus_panel(key);
            p.close(self, world);
            return true;
        }
        if point_in(mx, my, title) {
            self.focus_panel(key);
            self.drag = Some(Drag {
                key,
                grab: [mx - o[0], my - o[1]],
            });
            return true;
        }
        if p.press(self, world, mx, my, o) {
            self.focus_panel(key);
            return true;
        }
        false
    }
}

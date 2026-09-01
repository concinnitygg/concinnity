// src/editor/hook/routing.rs
//
// EditorHook: per-frame pointer routing -- the scroll steps behind each panel's
// wheel region, title-bar dragging, and click hit-testing across the top bar and
// the registered panels (front-to-back).

use super::*;

impl EditorHook {
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
            let max = self.form_fields.len().saturating_sub(self.form_window());
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
            let window =
                template_panel::rows_for_height(self.effective_size(PanelKey::TemplateDetail)[1]);
            let max = self.template_rows(i).len().saturating_sub(window);
            self.template_list_scroll = scroll_step(self.template_list_scroll, delta, max);
        }
    }

    pub(super) fn hud_state(&self) -> HudState {
        HudState {
            dirty: self.dirty,
            undo: self.can_undo(),
            redo: self.can_redo(),
            view_open: self.view_open,
            display_open: self.display_menu_open,
            sim: self.sim.state,
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
        let size = self.effective_size(drag.key);
        self.positions[drag.key.index()] = Some(widget::clamp_origin(pos, size, vp, hud::BAR_H));
    }

    // While an edge / corner resize is active, grow the panel from the press
    // anchor (clamped at its minimum and fully on screen); releasing the button
    // ends the resize.
    pub(super) fn drive_resize(&mut self, input: &FrameInput, vp: [f32; 2]) {
        let Some(r) = self.resize else {
            return;
        };
        if !input.left_button_down {
            self.resize = None;
            return;
        }
        let min = self.default_size(r.key);
        let max = registry::panel(r.key).max_size(self);
        let (o, s) = resize::apply(&r, [input.mouse_x, input.mouse_y], min, max, vp, hud::BAR_H);
        self.positions[r.key.index()] = Some(o);
        self.sizes[r.key.index()] = Some(s);
    }

    // Route a press: the top bar first (it draws over the panels), then the panels
    // front-to-back so the frontmost claims a press in an overlap. Whichever panel
    // claims it comes to the front. A press on a panel's title bar starts a drag.
    pub(super) fn route_click(&mut self, input: &FrameInput, vp: [f32; 2], world: &mut World) {
        let (mx, my) = (input.mouse_x, input.mouse_y);
        // The toast stack draws above everything, so it claims its presses
        // first (rect-guarded: a miss falls straight through).
        if self.try_toast_press(mx, my, vp, world) {
            return;
        }
        if let Some(a) = hud::hit_test(mx, my, true, self.hud_state(), vp[0]) {
            // SAVE only writes world.jsonl; it neither rebuilds nor re-injects
            // the world, so an open form is left intact (no blank-field risk).
            self.apply_top(a, world);
            self.picker_open = false;
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
        // Nothing claimed the press: the gizmo's tip handles get it first
        // (grabbing a handle must not re-pick the object behind it), then the
        // billboard icons (which arbitrate against the mesh AABBs by camera
        // distance), then the 3D view as a pick. Only edit mode gets here
        // (`left_click` stays false while the world holds the cursor in play
        // mode).
        if self.try_gizmo_press(input, vp, world) {
            return;
        }
        if self.try_billboard_press(input, vp, world) {
            return;
        }
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
        let s = self.effective_size(key);
        let title = [o[0], o[1], s[0], widget::TITLE_H];
        // The X is checked before the title bar so it never starts a drag instead.
        if point_in(mx, my, widget::close_rect(title)) {
            self.focus_panel(key);
            p.close(self, world);
            return true;
        }
        // An edge / corner press starts a resize, checked after the close button
        // (which sits in a corner) but before the title-bar drag and the body.
        if p.resizable()
            && let Some(edges) = self.resize_edges(key, o, s, mx, my)
        {
            self.focus_panel(key);
            self.resize = Some(resize::Resize {
                key,
                edges,
                grab_mouse: [mx, my],
                start_origin: o,
                start_size: s,
            });
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

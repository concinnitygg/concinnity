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

    // Everything a frame's input drives while a world is open: the
    // in-flight drags, the press and wheel routing across the top bar and
    // the panels, and the editor's keyboard shortcuts. The start screen
    // takes `drive_start_input` instead, which routes nothing else.
    pub(super) fn drive_session_input(
        &mut self,
        input: &FrameInput,
        vp: [f32; 2],
        world: &mut World,
    ) {
        // An active title-bar drag follows the cursor; a fresh press (only
        // when no drag is running -- the press that starts one must not
        // also resolve to a control) routes to the bar and the panels.
        self.drive_drag(input, vp);
        // An active edge / corner resize follows the cursor the same way.
        self.drive_resize(input, vp);
        // A chart pan does too, so the canvas tracks the cursor.
        self.drive_behavior_pan(input);
        // An in-flight gizmo drag follows the cursor, cancels on
        // Escape, and commits on release, before any new press routes.
        if self.gizmo_drag.is_some() {
            self.drive_gizmo_drag(input, vp, world);
        }
        // A shape slider drag likewise: follow + live preview,
        // cancel, or commit once.
        if self.shape_drag.is_some() {
            self.drive_shape_drag(input, world);
        }
        // An in-flight marquee likewise: follow, cancel, or select.
        if self.marquee.is_some() {
            self.drive_marquee(input, vp, world);
        }
        // An in-flight orbit tumble follows the cursor and ends on
        // release.
        if self.orbit.is_some() {
            self.drive_orbit(input, world);
        }
        // A drag-out placement from the Content panel: ghost follow,
        // cancel, or commit.
        if self.content_drag.is_some() {
            self.drive_content_drag(input, vp, world);
        }
        if input.left_click
            && self.drag.is_none()
            && self.resize.is_none()
            && self.gizmo_drag.is_none()
            && self.shape_drag.is_none()
            && self.marquee.is_none()
            && self.content_drag.is_none()
            && self.orbit.is_none()
        {
            // The confirmation dialog swallows every press first.
            // Behind it, an open Display or create menu is modal: it
            // takes the press (a row acts, anything else dismisses)
            // before normal routing. An Alt+press over the viewport
            // starts an orbit tumble instead of a pick.
            let claimed = self.route_modal_click(input, vp, world)
                || self.route_display_menu_click(input, vp)
                || self.route_create_menu_click(input, vp)
                || self.route_palette_dismiss(input, vp)
                || (input.alt && self.try_begin_orbit(input, vp, world));
            if !claimed {
                self.route_click(input, vp, world);
            }
        }
        // An unclaimed viewport right press opens the create menu at
        // the cursor, under the same not-mid-gesture guards; the
        // confirmation dialog swallows it like every other press.
        if input.right_click
            && self.modal.is_none()
            && self.drag.is_none()
            && self.resize.is_none()
            && self.gizmo_drag.is_none()
            && self.shape_drag.is_none()
            && self.marquee.is_none()
            && self.content_drag.is_none()
            && self.orbit.is_none()
        {
            self.open_create_menu(input, vp, world);
        }
        // T / R / S pick the gizmo's mode (translate / rotate /
        // scale), under the same guards as the history shortcuts.
        if !input.ctrl
            && !self.sim.playing()
            && !self.text_focus_active()
            && self.gizmo_drag.is_none()
        {
            match input.captured_key {
                Some(crate::components::InputKey::T) => {
                    self.gizmo_mode = gizmo::GizmoMode::Translate;
                }
                Some(crate::components::InputKey::R) => self.gizmo_mode = gizmo::GizmoMode::Rotate,
                Some(crate::components::InputKey::S) => self.gizmo_mode = gizmo::GizmoMode::Scale,
                // F frames the selection; Shift+F keeps the old fly
                // toggle one modifier away.
                Some(crate::components::InputKey::F) => {
                    if input.shift {
                        self.toggle_fly();
                    } else {
                        self.frame_selection(input.viewport, world);
                    }
                }
                // H hides the selection; Shift+H isolates it.
                Some(crate::components::InputKey::H) => {
                    if input.shift {
                        self.toggle_isolate();
                    } else {
                        self.hide_selected();
                    }
                }
                // 1..9 glide back to a saved camera bookmark.
                Some(key) => {
                    if let Some(slot) = bookmarks::slot_for(key) {
                        self.recall_bookmark(slot, world);
                    }
                }
                None => {}
            }
        }
        // Backtick toggles the console. The flag cleared here is the
        // one-frame focus blur a backtick open sets, so the text
        // system never types that backtick into the command line.
        self.console_blur = false;
        self.drive_console_toggle(input, world);
        // Ctrl+K toggles the palette, with the same one-frame blur.
        self.palette_blur = false;
        self.drive_palette_toggle(input, world);
        // Ctrl+Z / Ctrl+Y step the entry list through the history,
        // unless the world owns the keyboard (play mode), a text
        // field does (its own editing keys must win), or a gizmo drag
        // is mid-flight (its commit has not landed yet). Ctrl+D
        // duplicates the selection and Ctrl+Down drops it to the
        // floor, under the same guards; a frontmost Behavior panel
        // keeps its own Ctrl+D (row duplicate, via frame_keys below).
        if input.ctrl
            && !self.sim.playing()
            && !self.text_focus_active()
            && self.gizmo_drag.is_none()
            && self.shape_drag.is_none()
        {
            match input.captured_key {
                Some(crate::components::InputKey::Z) => self.undo(world),
                Some(crate::components::InputKey::Y) => self.redo(world),
                Some(crate::components::InputKey::D)
                    if self.frontmost_open_panel() != Some(PanelKey::Behavior) =>
                {
                    self.duplicate_selection();
                }
                Some(crate::components::InputKey::Down) => {
                    self.drop_selection_to_floor(world);
                }
                // Ctrl+H makes everything visible again.
                Some(crate::components::InputKey::H) => self.unhide_all(),
                // Ctrl+1..9 save the camera pose to a bookmark.
                Some(key) => {
                    if let Some(slot) = bookmarks::slot_for(key) {
                        self.save_bookmark(slot, world);
                    }
                }
                None => {}
            }
        }
        // The transport shortcuts, live in every state (pausing a
        // running world is their whole point).
        self.sim_keys(input);
        // Per-frame editing keys go to the frontmost open panel.
        if let Some(key) = self.frontmost_open_panel() {
            registry::panel(key).frame_keys(self, world, input);
        }
        // Wheel routing: the frontmost scrollable panel under the cursor
        // takes the wheel. An open confirmation dialog swallows it with
        // the rest of the pointer. An open value dropdown is modal and
        // can extend past the form panel, so it scrolls the form from
        // anywhere while open.
        if self.modal.is_none() && input.scroll_delta.abs() > 0.5 {
            let (mx, my) = (input.mouse_x, input.mouse_y);
            let form_shown = registry::panel(PanelKey::Edit).is_open(self);
            if form_shown && self.field_dropdown.is_some() {
                self.scroll_form(input.scroll_delta, world);
            } else {
                let front_to_back: Vec<PanelKey> = self.panel_order.iter().rev().copied().collect();
                for key in front_to_back {
                    let p = registry::panel(key);
                    if !self.panel_shown(key) {
                        continue;
                    }
                    let o = self.origin(key, vp);
                    if p.wheel_over(self, world, mx, my, o) {
                        p.scroll(self, world, input.scroll_delta);
                        break;
                    }
                }
            }
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
            // The start screen is the whole window: the bar it would sit under
            // is not drawn, and never resolves a press.
            visible: self.hud_visible && !self.start_mode,
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
        if !self.panel_shown(key) {
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

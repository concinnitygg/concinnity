// src/editor/hook/routing.rs
//
// EditorHook: per-frame pointer routing -- wheel scroll dispatch, title-bar
// dragging, and click hit-testing across the top bar and panels.

use super::*;

impl EditorHook {
    // Move the targeted region's scroll offset in the wheel direction. The tick
    // picks the target from the cursor position (both panels can be open).
    pub(super) fn scroll(&mut self, delta: f32, target: ScrollTarget, world: &mut World) {
        match target {
            ScrollTarget::List if self.combo == Combo::Closed => {
                let max = self.list_rows().len().saturating_sub(panel::MAX_ROWS);
                self.list_scroll = scroll_step(self.list_scroll, delta, max);
                self.row_menu = None;
            }
            ScrollTarget::List => {
                let max = self
                    .combo_options(world)
                    .len()
                    .saturating_sub(panel::MAX_ROWS);
                self.combo_scroll = scroll_step(self.combo_scroll, delta, max);
            }
            ScrollTarget::Form => {
                if let Some(open) = self.field_dropdown {
                    // An open value dropdown scrolls its own option list.
                    let total = self.form_fields.get(open).map_or(0, |f| f.variants.len());
                    let max = total.saturating_sub(form_panel::MAX_DROP_ROWS);
                    self.field_dropdown_scroll =
                        scroll_step(self.field_dropdown_scroll, delta, max);
                } else {
                    // Scroll the field window: fold the visible controls into the
                    // working args, move the window, then re-seed the newly visible
                    // slots. The same capture / refresh cycle an array add / remove
                    // uses, so no in-progress edit is lost as the window moves.
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
            ScrollTarget::TemplateList => {
                if let Some(i) = self.open_template {
                    let max = self
                        .template_rows(i)
                        .len()
                        .saturating_sub(super::asset_list::MAX_ROWS);
                    self.template_list_scroll = scroll_step(self.template_list_scroll, delta, max);
                }
            }
        }
    }

    pub(super) fn hud_state(&self) -> HudState {
        HudState {
            dirty: self.dirty,
            view_open: self.view_open,
            visible: self.hud_visible,
        }
    }

    // While a title-bar drag is active, follow the cursor (clamped fully on
    // screen); releasing the button ends the drag.
    pub(super) fn drive_drag(&mut self, input: &FrameInput, vp: [f32; 2]) {
        let Some(drag) = self.drag else {
            return;
        };
        if !input.left_button_down {
            self.drag = None;
            return;
        }
        let pos = [input.mouse_x - drag.grab[0], input.mouse_y - drag.grab[1]];
        match drag.target {
            DragTarget::Assets => {
                self.panel_pos = Some(widget::clamp_origin(pos, panel::size(), vp));
            }
            DragTarget::Edit => {
                let size = form_panel::size(self.form_fields.len());
                self.edit_pos = Some(widget::clamp_origin(pos, size, vp));
            }
            DragTarget::Preview => {
                self.preview_pos = Some(widget::clamp_origin(pos, preview::size(), vp));
            }
            DragTarget::View => {
                self.view_pos = Some(widget::clamp_origin(pos, view::size(), vp));
            }
            DragTarget::Templates => {
                self.templates_pos = Some(widget::clamp_origin(pos, templates::size(), vp));
            }
            DragTarget::TemplateDetail => {
                let n = self
                    .open_template
                    .map_or(1, |i| self.template_rows(i).len());
                let size = template_panel::size(n);
                self.template_detail_pos = Some(widget::clamp_origin(pos, size, vp));
            }
        }
    }

    // Route a press: the top bar first (it draws over the panels), then the panels
    // front-to-back so the frontmost claims a press in an overlap. Whichever panel
    // claims it comes to the front. A press on a panel's title bar starts a drag.
    pub(super) fn route_click(&mut self, input: &FrameInput, vp: [f32; 2], world: &mut World) {
        let (mx, my) = (input.mouse_x, input.mouse_y);
        if let Some(a) = hud::hit_test(mx, my, true, self.dirty, vp[0]) {
            // SAVE only writes to disk now; it neither rebuilds nor re-injects the
            // world, so an open form is left intact (no blank-field risk).
            self.apply_top(a);
            self.combo = Combo::Closed;
            self.row_menu = None;
            return;
        }
        // Front-to-back: the frontmost shown panel to claim the press handles it and
        // rises to the front (so clicking an exposed sliver of a buried panel brings
        // it forward). `panel_order`'s tail is frontmost.
        let front_to_back: Vec<DragTarget> = self.panel_order.iter().rev().copied().collect();
        for target in front_to_back {
            if self.try_panel_press(target, mx, my, vp, world) {
                return;
            }
        }
    }

    // Try to resolve a press against panel `target`: `false` when it is hidden or
    // the press misses it (the caller tries the next panel back). A hit brings the
    // panel to the front; a title-bar press starts a drag, a body press resolves a
    // control.
    pub(super) fn try_panel_press(
        &mut self,
        target: DragTarget,
        mx: f32,
        my: f32,
        vp: [f32; 2],
        world: &mut World,
    ) -> bool {
        match target {
            DragTarget::Preview => {
                if !self.preview_open {
                    return false;
                }
                let pv = self.preview_origin(vp);
                if !point_in(mx, my, preview::panel_rect(pv)) {
                    return false;
                }
                self.focus_panel(DragTarget::Preview);
                if point_in(mx, my, preview::close_rect(pv)) {
                    self.preview_open = false;
                } else if point_in(mx, my, preview::title_rect(pv)) {
                    self.drag = Some(Drag {
                        target,
                        grab: [mx - pv[0], my - pv[1]],
                    });
                } else if let Some(PreviewAction::ToggleCapture) = preview::hit_test(mx, my, pv) {
                    self.world_capture = !self.world_capture;
                }
                true
            }
            DragTarget::Edit => {
                // The form is part of the assets UI: interactive only while the
                // browse panel is open.
                if !(self.form_open() && self.panel_open) {
                    return false;
                }
                let fo = self.edit_origin(vp);
                // The X in the title bar closes the form; checked before the
                // title-bar drag so it never starts a drag instead.
                if point_in(mx, my, form_panel::close_rect(fo)) {
                    self.focus_panel(DragTarget::Edit);
                    self.apply_form(FormAction::Close, world);
                    return true;
                }
                if point_in(mx, my, form_panel::title_rect(fo)) {
                    self.focus_panel(DragTarget::Edit);
                    self.drag = Some(Drag {
                        target,
                        grab: [mx - fo[0], my - fo[1]],
                    });
                    return true;
                }
                let action = {
                    let data = self.panel_data(world);
                    let view = self.make_form_view(&data, [mx, my]);
                    form_panel::hit_test(&view, mx, my, fo)
                };
                if let Some(fa) = action {
                    self.focus_panel(DragTarget::Edit);
                    self.apply_form(fa, world);
                    return true;
                }
                false
            }
            DragTarget::Assets => {
                if !self.panel_open {
                    return false;
                }
                let po = self.panel_origin(vp);
                // The X in the title bar closes the Assets panel (state kept, like a
                // View-checkbox untick); checked before the title-bar drag.
                if point_in(mx, my, panel::close_rect(po)) {
                    self.focus_panel(DragTarget::Assets);
                    self.panel_open = false;
                    self.combo = Combo::Closed;
                    self.row_menu = None;
                    return true;
                }
                if point_in(mx, my, panel::title_rect(po)) {
                    self.focus_panel(DragTarget::Assets);
                    self.drag = Some(Drag {
                        target,
                        grab: [mx - po[0], my - po[1]],
                    });
                    return true;
                }
                let action = {
                    let data = self.panel_data(world);
                    let view = self.make_view(&data, [mx, my]);
                    panel::hit_test(&view, mx, my, po)
                };
                if let Some(pa) = action {
                    self.focus_panel(DragTarget::Assets);
                    self.apply_panel(pa, world);
                    return true;
                }
                false
            }
            DragTarget::View => {
                if !self.view_open {
                    return false;
                }
                let vo = self.view_origin(vp);
                if !point_in(mx, my, view::panel_rect(vo)) {
                    return false;
                }
                self.focus_panel(DragTarget::View);
                if point_in(mx, my, view::close_rect(vo)) {
                    self.view_open = false;
                } else if point_in(mx, my, view::title_rect(vo)) {
                    self.drag = Some(Drag {
                        target,
                        grab: [mx - vo[0], my - vo[1]],
                    });
                } else if let Some(a) = view::hit_test(mx, my, vo) {
                    self.apply_view(a);
                }
                true
            }
            DragTarget::Templates => {
                if !self.templates_open {
                    return false;
                }
                let to = self.templates_origin(vp);
                if !point_in(mx, my, templates::panel_rect(to)) {
                    return false;
                }
                self.focus_panel(DragTarget::Templates);
                if point_in(mx, my, templates::close_rect(to)) {
                    self.templates_open = false;
                } else if point_in(mx, my, templates::title_rect(to)) {
                    self.drag = Some(Drag {
                        target,
                        grab: [mx - to[0], my - to[1]],
                    });
                } else if let Some(a) = templates::hit_test(mx, my, to) {
                    self.apply_templates(a);
                }
                true
            }
            DragTarget::TemplateDetail => {
                // The detail panel is part of the Templates UI: interactive only
                // while the Templates list is open and a template is picked.
                let Some(i) = self.open_template.filter(|_| self.templates_open) else {
                    return false;
                };
                let to = self.template_detail_origin(i, vp);
                // The X in the title bar closes the detail; checked before the
                // title-bar drag so it never starts a drag instead.
                if point_in(mx, my, template_panel::close_rect(to)) {
                    self.focus_panel(DragTarget::TemplateDetail);
                    self.close_template_detail();
                    return true;
                }
                if point_in(mx, my, template_panel::title_rect(to)) {
                    self.focus_panel(DragTarget::TemplateDetail);
                    self.drag = Some(Drag {
                        target,
                        grab: [mx - to[0], my - to[1]],
                    });
                    return true;
                }
                let action = {
                    let data = self.template_detail_data(i);
                    let view = self.make_template_view(&data, [mx, my]);
                    template_panel::hit_test(&view, mx, my, to)
                };
                if let Some(a) = action {
                    self.focus_panel(DragTarget::TemplateDetail);
                    self.apply_template_detail(a);
                    return true;
                }
                false
            }
        }
    }
}

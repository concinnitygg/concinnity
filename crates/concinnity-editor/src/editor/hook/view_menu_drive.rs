// src/editor/hook/view_menu_drive.rs
//
// EditorHook: the Display menu's open state, click routing, and draw. The
// menu's geometry lives in `editor/view_menu.rs`; the selections it edits
// (view mode, show flags, the billboard toggle) are published to the renderer
// each tick as the `ViewOverrides` resource.

use super::*;

impl EditorHook {
    // The top bar's Display chip.
    pub(super) fn toggle_display_menu(&mut self) {
        self.display_menu_open = !self.display_menu_open;
        if self.display_menu_open {
            self.create_menu = None;
        }
    }

    // An open Display menu takes the press: a row acts (the menu stays up so
    // several toggles land in one visit), anywhere else dismisses. A press on
    // the top bar is left to the bar's own hit test, whose Display chip
    // toggles the menu.
    pub(super) fn route_display_menu_click(&mut self, input: &FrameInput, vp: [f32; 2]) -> bool {
        if !self.display_menu_open {
            return false;
        }
        let (mx, my) = (input.mouse_x, input.mouse_y);
        if my <= hud::BAR_H {
            return false;
        }
        if let Some(i) = view_menu::hit_row(mx, my, vp[0]) {
            match view_menu::rows()[i] {
                view_menu::MenuRow::Mode(m) => self.view_mode = m,
                view_menu::MenuRow::ShowHeading => {}
                view_menu::MenuRow::Flag(f, _) => {
                    self.show_flags = self.show_flags.toggled(f);
                }
                view_menu::MenuRow::Billboards => self.show_billboards = !self.show_billboards,
            }
            return true;
        }
        if view_menu::over(mx, my, vp[0]) {
            return true;
        }
        self.display_menu_open = false;
        true
    }

    pub(super) fn drive_display_menu_draw(
        &self,
        world: &mut World,
        vp: [f32; 2],
        shown: bool,
        mouse: [f32; 2],
    ) {
        if !shown || !self.display_menu_open || self.sim.playing() {
            view_menu::hide(world);
            return;
        }
        view_menu::apply(
            world,
            vp[0],
            view_menu::MenuState {
                mode: self.view_mode,
                show: self.show_flags,
                billboards: self.show_billboards,
            },
            mouse,
        );
    }
}

//! `EditorHook`: the Display menu's open state and picks. The menu itself is
//! an engine `Menu` (`editor/view_menu.rs`); the selections it edits (view
//! mode, show flags, the billboard toggle, the extent outlines) are published
//! to the renderer each tick as the `ViewOverrides` resource.

use concinnity_engine::components::UiEvent;
use concinnity_engine::ecs::World;

use crate::editor::hook::EditorHook;
use crate::editor::ui::BuiltUi;
use crate::editor::view_menu::{self, MenuRow};

impl EditorHook {
    // The top bar's Display chip.
    pub(in crate::editor::hook) fn toggle_display_menu(&mut self) {
        self.display_menu_open = !self.display_menu_open;
        if self.display_menu_open {
            self.create_menu = None;
        }
    }

    // Apply one Display menu row, picked from the menu or the palette.
    pub(in crate::editor::hook) fn apply_display_row(&mut self, row: MenuRow) {
        match row {
            MenuRow::Mode(m) => self.view_mode = m,
            MenuRow::Heading(_) => {}
            MenuRow::Flag(f, _) => self.show_flags = self.show_flags.toggled(f),
            MenuRow::Billboards => self.show_billboards = !self.show_billboards,
            MenuRow::Extent(c, _) => self.extent_show = self.extent_show.toggled(c),
        }
    }

    // A pick from the menu; whether `event` was one.
    pub(in crate::editor::hook) fn on_display_menu_event(&mut self, event: &UiEvent) -> bool {
        let Some(row) = view_menu::pick(event) else {
            return false;
        };
        self.apply_display_row(row);
        true
    }

    // Take in a close the menu made itself: a click away or Escape. A world
    // the menu was just built into has shown it open nowhere yet.
    pub(in crate::editor::hook) fn observe_display_menu(&mut self, world: &World) {
        let fresh = world.resource::<BuiltUi>().is_some_and(|b| b.fresh);
        if !fresh && self.display_menu_open && !view_menu::is_open(world) {
            self.display_menu_open = false;
        }
    }

    // Bring the menu in line with the selections; it closes while the HUD is
    // hidden or the world runs.
    pub(in crate::editor::hook) fn drive_display_menu(
        &mut self,
        world: &mut World,
        vp: [f32; 2],
        shown: bool,
    ) {
        if !shown || self.sim.playing() {
            self.display_menu_open = false;
        }
        let Some(id) = world.resource::<BuiltUi>().and_then(|b| b.display_menu) else {
            return;
        };
        let state = view_menu::MenuState {
            open: self.display_menu_open,
            mode: self.view_mode,
            show: self.show_flags,
            billboards: self.show_billboards,
            extents: self.extent_show,
        };
        view_menu::sync(world, id, &state, vp[0]);
    }
}

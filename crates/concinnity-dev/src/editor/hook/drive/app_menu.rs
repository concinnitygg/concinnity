//! EditorHook: the macOS menu bar. The bar is installed once, for a session
//! that really has a window, and from then on this is the frame's two-way
//! exchange with it: choices made in the menu are drained and applied, and the
//! panel state they edit is pushed back so the checkmarks follow.
//!
//! The View button in the top bar stays the way it is on every platform; the
//! View menu is a second way into the same panel toggles, sharing their one
//! apply (`toggle_view_row`).

use concinnity_core::ecs::World;

use crate::editor::hook::EditorHook;

impl EditorHook {
    /// Install the session's menu bar. Called only from the `cn editor` run
    /// path, so a hook built for a test never reaches AppKit.
    #[cfg(target_os = "macos")]
    pub(crate) fn with_app_menu(self) -> Self {
        crate::editor::app_menu::install(self.panel_marks());
        self
    }

    #[cfg(not(target_os = "macos"))]
    pub(crate) fn with_app_menu(self) -> Self {
        self
    }

    #[cfg(target_os = "macos")]
    pub(in crate::editor::hook) fn drive_app_menu(&mut self, world: &mut World) {
        use crate::editor::app_menu::{self, MenuCommand};

        for command in app_menu::take_chosen() {
            match command {
                MenuCommand::PanelToggle(i) => self.toggle_view_row(i, world),
                // Cancels the same token CTRL+C and the debug `shutdown` call
                // cancel, so the session leaves by its one clean exit.
                MenuCommand::Quit => {
                    if let Some(shutdown) = &self.shutdown {
                        shutdown.cancel();
                    }
                }
            }
        }
        app_menu::sync(self.panel_marks());
    }

    #[cfg(not(target_os = "macos"))]
    pub(in crate::editor::hook) fn drive_app_menu(&mut self, _world: &mut World) {}

    // Which of the View panel's panels are open, for the menu bar's checkmarks.
    #[cfg(target_os = "macos")]
    fn panel_marks(&self) -> crate::editor::app_menu::PanelMarks {
        use crate::editor::panels::registry;

        crate::editor::app_menu::PanelMarks::from_open(
            registry::view_toggles().map(|p| p.is_open(self)),
        )
    }
}

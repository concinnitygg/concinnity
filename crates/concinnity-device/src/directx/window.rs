//! `WindowControl` for `DxContext`: the window lives on the shared Win32 layer,
//! which every query and hint here reaches through.

use concinnity_engine::components;
use concinnity_engine::input::snapshot::InputSnapshot;
use concinnity_engine::render::backend::WindowControl;
use concinnity_engine::window::clipboard::Clipboard;
use concinnity_engine::window::display_mode;

use super::context::DxContext;
use crate::win32::window;
use crate::win32::window::{frame_tick, take_input_snapshot};

impl WindowControl for DxContext {
    fn window_closed(&mut self) -> bool {
        // Message pump + cursor window-exit / fullscreen-confinement refresh +
        // the Resolution-mode reconcile, shared with the Vulkan Windows window.
        // Partial field borrow (not `win_mut`) so `fullscreen_display` can be
        // borrowed disjointly in the same call.
        frame_tick(
            self.hw
                .win_state
                .as_mut()
                .expect("DxContext window state present"),
            &mut self.hw.fullscreen_display,
        )
    }

    fn take_input(&mut self) -> InputSnapshot {
        take_input_snapshot(self.win_mut())
    }

    fn request_cursor_capture(&mut self) {
        // Arm click-to-capture instead of grabbing a possibly unfocused window;
        // the first left-click in the content area grabs the cursor.
        self.win_mut().recapture_on_click = true;
    }

    // The drawable pixel size, the space both the UI shader and WM_MOUSEMOVE
    // use.
    fn logical_size(&self) -> (f32, f32) {
        (
            self.targets.extent.output_width as f32,
            self.targets.extent.output_height as f32,
        )
    }

    // Hide or show the OS cursor for an in-engine UI cursor (e.g. a MainMenu),
    // without engaging camera capture. Edge-triggered in the helper, so calling
    // it every frame with the same value is cheap.
    fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        window::set_ui_cursor_hidden(self.win_mut(), hidden);
    }

    // Whether the real cursor has left the window so the renderer should stop
    // drawing the in-engine UI cursor (windowed / borderless). Recomputed each
    // frame by `update_ui_cursor_confinement` in `window_closed`; false while
    // captured or in fullscreen (which confines the cursor instead).
    fn cursor_outside_window(&self) -> bool {
        self.win().cursor_outside_window
    }

    fn window_minimized(&self) -> bool {
        window::is_minimized(self.win().hwnd)
    }

    // A togglable menu coexists with a captured camera; see
    // `WindowControl::set_menu_mode`. The wnd_proc reads this flag to route
    // Escape to the ECS and suppress click-to-recapture.
    fn set_menu_mode(&mut self, on: bool) {
        self.win_mut().menu_mode = on;
    }

    // Edge-triggered capture: capture for camera control, release while a menu
    // is open. GraphicsSystem calls this each frame in menu mode. Unlike the
    // startup `request_cursor_capture` (which arms click-to-capture), closing the menu
    // recaptures immediately so the camera resumes without an extra click.
    fn set_camera_capture(&mut self, capture: bool) {
        if capture == self.win().cursor_captured {
            return;
        }
        if capture {
            let hwnd = self.win().hwnd;
            window::capture_cursor(hwnd, self.win_mut());
        } else {
            window::release_cursor(self.win_mut());
        }
    }

    // Only the sync interval changes. Tearing needs the swapchain created with
    // vsync off; otherwise interval 0 presents uncapped without it.
    fn set_vsync(&mut self, on: bool) {
        self.swapchain.present_sync_interval = u32::from(on);
    }

    // Switch window mode / resize at runtime (windowed / borderless / fullscreen
    // and content-size presets). The Win32 work lives in `window.rs`; the resize
    // path picks up the resulting WM_SIZE.
    fn set_window_mode(&mut self, mode: components::WindowMode) {
        window::set_window_mode(self.win_mut(), mode);
    }

    fn set_window_size(&mut self, width: u32, height: u32) {
        window::set_window_size(self.win(), width, height);
    }

    // The display modes (resolution + refresh rate) of the monitor the window
    // sits on, feeding the Resolution settings row (the caller dedups + sorts).
    fn display_modes(&self) -> Vec<display_mode::DisplayMode> {
        crate::win32::display_mode::enumerate(self.win().hwnd)
    }

    // The mode the window's monitor is currently running (what the Resolution
    // row shows before the user ever picks one).
    fn current_display_mode(&self) -> Option<display_mode::DisplayMode> {
        crate::win32::display_mode::current(self.win().hwnd)
    }

    // Remember the display mode to hold while the window is in fullscreen.
    // Applied by the per-frame reconcile in `window_closed` (which also
    // restores the desktop mode on leaving fullscreen), so a choice made in
    // any window mode takes effect when fullscreen is (or becomes) active.
    fn set_display_mode(&mut self, mode: display_mode::DisplayMode) {
        self.hw.fullscreen_display.set_desired(mode);
    }

    fn clipboard(&mut self) -> Option<&mut dyn Clipboard> {
        Some(self.win_mut())
    }

    fn shared_clipboard(&self) -> Option<Box<dyn Clipboard + Send + Sync>> {
        Some(Box::new(crate::win32::clipboard::WindowClipboard::new(
            self.win().hwnd,
        )))
    }
}

//! `WindowControl` for `MtlContext`: the window lives on the shared `AppKit`
//! layer, which every query and hint here reaches through.

use concinnity_engine::components::WindowMode;
use concinnity_engine::input::snapshot::InputSnapshot;
use concinnity_engine::presentation::window::clipboard::Clipboard;
use concinnity_engine::presentation::window::display_mode::{DisplayMode, DisplayRefresh};
use concinnity_engine::render::backend::WindowControl;

use super::context::MtlContext;

impl WindowControl for MtlContext {
    fn window_closed(&mut self) -> bool {
        self.window().appkit.closed()
    }

    fn take_input(&mut self) -> InputSnapshot {
        self.window_mut().appkit.take_input()
    }

    fn request_cursor_capture(&mut self) {
        self.window_mut().appkit.capture_cursor();
    }

    fn logical_size(&self) -> (f32, f32) {
        self.window().appkit.logical_size()
    }

    fn top_content_inset(&self) -> f32 {
        self.window().appkit.top_content_inset()
    }

    fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        self.window_mut().appkit.set_ui_cursor_hidden(hidden);
    }

    fn cursor_outside_window(&self) -> bool {
        self.window().appkit.cursor_outside_window()
    }

    fn window_minimized(&self) -> bool {
        self.window().appkit.window_minimized()
    }

    fn set_menu_mode(&mut self, on: bool) {
        self.window_mut().appkit.set_menu_mode(on);
    }

    fn set_camera_capture(&mut self, capture: bool) {
        self.window_mut().appkit.set_camera_capture(capture);
    }

    // Turn display sync on or off through the view's backing CAMetalLayer.
    // Setting displaySyncEnabled is an idempotent property write (no swapchain
    // rebuild on Metal), so a redundant call is cheap. Vulkan reaches the same
    // end by rebuilding the swapchain with a different present mode, so this
    // does not live on the shared window layer.
    fn set_vsync(&mut self, on: bool) {
        super::init::set_display_sync(&self.window().view, on);
    }

    fn set_window_mode(&mut self, mode: WindowMode) {
        self.window_mut().appkit.set_window_mode(mode);
    }

    fn set_window_size(&mut self, width: u32, height: u32) {
        self.window_mut().appkit.set_window_size(width, height);
    }

    fn display_modes(&self) -> Vec<DisplayMode> {
        self.window().appkit.display_modes()
    }

    fn current_display_mode(&self) -> Option<DisplayMode> {
        self.window().appkit.current_display_mode()
    }

    fn display_refresh(&self) -> Option<DisplayRefresh> {
        self.window().appkit.display_refresh()
    }

    fn set_display_mode(&mut self, mode: DisplayMode) {
        self.window_mut().appkit.set_display_mode(mode);
    }

    fn clipboard(&mut self) -> Option<&mut dyn Clipboard> {
        Some(self.window_mut().appkit.clipboard())
    }
}

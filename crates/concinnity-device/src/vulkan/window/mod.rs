// The platform window VkContext owns: the shared native Win32 layer on Windows
// and the shared native AppKit layer on macOS (one window/input implementation
// with the DirectX and Metal backends respectively), GLFW on the desktop Unix
// tier. The gate matches the manifest's, so a target with no window layer
// fails naming this alias rather than naming a missing crate. `WindowControl`
// for `VkContext` is implemented here, over whichever window that is.

use concinnity_engine::components::WindowMode;
use concinnity_engine::input::snapshot::InputSnapshot;
use concinnity_engine::presentation::window::clipboard::Clipboard;
use concinnity_engine::presentation::window::display_mode::{DisplayMode, DisplayRefresh};
use concinnity_engine::render::backend::WindowControl;

use super::context::VkContext;

#[cfg(target_os = "macos")]
mod appkit;
#[cfg(all(unix, not(target_vendor = "apple"), not(target_os = "android")))]
mod glfw;
#[cfg(all(unix, not(target_vendor = "apple"), not(target_os = "android")))]
mod glfw_clipboard;
#[cfg(target_os = "windows")]
mod win32;

#[cfg(target_os = "macos")]
pub(super) use self::appkit::AppKitVkWindow as PlatformWindow;
#[cfg(all(unix, not(target_vendor = "apple"), not(target_os = "android")))]
pub(super) use self::glfw::GlfwWindow as PlatformWindow;
#[cfg(target_os = "windows")]
pub(super) use self::win32::Win32Window as PlatformWindow;

impl WindowControl for VkContext {
    fn window_closed(&mut self) -> bool {
        self.window_mut().poll()
    }

    fn take_input(&mut self) -> InputSnapshot {
        // Both platform windows snapshot straight into the shared InputSnapshot.
        self.window_mut().take_input()
    }

    fn request_cursor_capture(&mut self) {
        self.window_mut().request_cursor_capture();
    }

    // Live window size for overlay (view-owned UI) scaling and cursor
    // hit-testing, in the window's logical units. Read from the platform window
    // rather than the swapchain extent so the overlay space matches the units
    // `poll()` reports the cursor in on every platform: points on macOS, client
    // pixels on Windows, window coordinates on Linux. Where the framebuffer is
    // larger (a retina drawable, a scaled Wayland surface) the difference is
    // absorbed by the UI shader's divide to NDC, and only the text scissor
    // converts back to pixels.
    fn logical_size(&self) -> (f32, f32) {
        self.window().logical_size()
    }

    // The window chrome overlapping the top of the frame, in the same units.
    // Non-zero only on the macOS window, whose content view runs under a
    // transparent title bar.
    fn top_content_inset(&self) -> f32 {
        self.window().top_content_inset()
    }

    // Hide or show the OS cursor for an in-engine UI cursor (e.g. a MainMenu),
    // without engaging camera capture. Edge-triggered in the window helper.
    fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        self.window_mut().set_ui_cursor_hidden(hidden);
    }

    // Whether the real cursor has left the window so the renderer should stop
    // drawing the in-engine UI cursor (windowed / borderless). Recomputed each
    // `poll` (in `window_closed`); false while captured or in fullscreen (which
    // confines the cursor instead).
    fn cursor_outside_window(&self) -> bool {
        self.window().cursor_outside_window()
    }

    fn window_minimized(&self) -> bool {
        self.window().window_minimized()
    }

    // A togglable menu coexists with a captured camera; see
    // `WindowControl::set_menu_mode`.
    fn set_menu_mode(&mut self, on: bool) {
        self.window_mut().set_menu_mode(on);
    }

    // Drive cursor capture from the menu state each frame: capture for camera
    // control, release while a menu is open. Edge-triggered in the window.
    fn set_camera_capture(&mut self, capture: bool) {
        self.window_mut().set_camera_capture(capture);
    }

    // Turn display sync (vsync) on or off at runtime. The present mode is fixed
    // at swapchain creation (FIFO for vsync, MAILBOX/IMMEDIATE for uncapped), so
    // a change recreates the swapchain, which re-selects the mode from
    // `self.hw.vsync`. Edge-triggered: a redundant call (a swapchain rebuild is
    // expensive) is skipped.
    fn set_vsync(&mut self, on: bool) {
        if on == self.hw.vsync {
            return;
        }
        self.hw.vsync = on;
        if let Err(e) = self.rebuild_swapchain() {
            tracing::warn!("set_vsync: rebuild_swapchain failed: {}", e);
        }
    }

    // Switch window mode / resize at runtime. The GLFW work lives in window.rs;
    // the framebuffer-size change drives a swapchain rebuild via the present
    // path's OUT_OF_DATE handling.
    fn set_window_mode(&mut self, mode: WindowMode) {
        self.window_mut().set_window_mode(mode);
    }

    fn set_window_size(&mut self, width: u32, height: u32) {
        self.window_mut().set_window_size(width, height);
    }

    // The display modes feeding the Resolution settings row; enumeration,
    // the fullscreen mode hold, and the desktop-mode restore all live in
    // window.rs (GLFW owns the video-mode switching).
    fn display_modes(&self) -> Vec<DisplayMode> {
        self.window().display_modes()
    }

    fn current_display_mode(&self) -> Option<DisplayMode> {
        self.window().current_display_mode()
    }

    fn display_refresh(&self) -> Option<DisplayRefresh> {
        self.window().display_refresh()
    }

    fn set_display_mode(&mut self, mode: DisplayMode) {
        self.window_mut().set_display_mode(mode);
    }

    fn clipboard(&mut self) -> Option<&mut dyn Clipboard> {
        Some(self.window_mut().clipboard())
    }
}

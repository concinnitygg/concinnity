// The Win32 window every HWND-rendering backend presents to: the window state
// the `wnd_proc` feeds and the fullscreen display-mode hold, answering
// `WindowControl` for DirectX and Vulkan alike.

use concinnity_engine::components::WindowMode;
use concinnity_engine::input::snapshot::InputSnapshot;
use concinnity_engine::render::backend::WindowControl;
use concinnity_engine::window::clipboard::Clipboard;
use concinnity_engine::window::display_mode::DisplayMode;

use super::display_mode::{self, FullscreenDisplayMode};
use super::window::{self, WindowState, create_window, frame_tick, take_input_snapshot};

pub(crate) struct Win32Window {
    state: Box<WindowState>,
    // The user's chosen fullscreen display mode, held on the monitor while the
    // window is fullscreen and restored on exit / drop; reconciled once per
    // frame by `frame_tick` in `window_closed`.
    fullscreen_display: FullscreenDisplayMode,
}

impl Win32Window {
    // Create the window, always windowed; a caller applies any other mode after.
    pub(crate) fn create(
        title: &str,
        width: u32,
        height: u32,
        title_bar: bool,
    ) -> Result<Self, String> {
        let (_hwnd, state) = create_window(title, width, height, title_bar)?;
        Ok(Self {
            state,
            fullscreen_display: FullscreenDisplayMode::new(),
        })
    }

    // The window state the `wnd_proc` writes: the HWND, its client size, the
    // cursor and keyboard state.
    pub(crate) fn state(&self) -> &WindowState {
        &self.state
    }

    // The swapchain-facing surface size in pixels (the client area, tracked via
    // WM_SIZE).
    pub(crate) fn framebuffer_size(&self) -> (i32, i32) {
        (self.state.width, self.state.height)
    }
}

impl Drop for Win32Window {
    // Restore cursor clip and visibility so the OS is not left in a bad state.
    fn drop(&mut self) {
        window::release_cursor(&mut self.state);
    }
}

impl WindowControl for Win32Window {
    // Drain the message pump, refresh the cursor window-exit / confinement
    // state, and reconcile the fullscreen display mode.
    fn window_closed(&mut self) -> bool {
        frame_tick(&mut self.state, &mut self.fullscreen_display)
    }

    fn take_input(&mut self) -> InputSnapshot {
        take_input_snapshot(&mut self.state)
    }

    // Arm click-to-capture rather than grabbing the cursor immediately: a
    // freshly spawned window may not be focused. The first content click
    // captures.
    fn request_cursor_capture(&mut self) {
        self.state.recapture_on_click = true;
    }

    // Windows reports WM_MOUSEMOVE in the same client pixels the swapchain is
    // sized to, so logical units are framebuffer pixels here.
    fn logical_size(&self) -> (f32, f32) {
        (self.state.width as f32, self.state.height as f32)
    }

    fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        window::set_ui_cursor_hidden(&mut self.state, hidden);
    }

    // Recomputed each `window_closed`; false while captured or in fullscreen,
    // which confines the cursor instead.
    fn cursor_outside_window(&self) -> bool {
        self.state.cursor_outside_window
    }

    fn window_minimized(&self) -> bool {
        window::is_minimized(self.state.hwnd)
    }

    // The wnd_proc reads this flag to route Escape to the ECS and suppress
    // click-to-recapture.
    fn set_menu_mode(&mut self, on: bool) {
        self.state.menu_mode = on;
    }

    // Edge-triggered: unlike `request_cursor_capture`, closing a menu
    // recaptures immediately so the camera resumes without an extra click.
    fn set_camera_capture(&mut self, capture: bool) {
        if capture == self.state.cursor_captured {
            return;
        }
        if capture {
            let hwnd = self.state.hwnd;
            window::capture_cursor(hwnd, &mut self.state);
        } else {
            window::release_cursor(&mut self.state);
        }
    }

    fn set_window_mode(&mut self, mode: WindowMode) {
        window::set_window_mode(&mut self.state, mode);
    }

    fn set_window_size(&mut self, width: u32, height: u32) {
        window::set_window_size(&self.state, width, height);
    }

    fn display_modes(&self) -> Vec<DisplayMode> {
        display_mode::enumerate(self.state.hwnd)
    }

    fn current_display_mode(&self) -> Option<DisplayMode> {
        display_mode::current(self.state.hwnd)
    }

    // Applied by the per-frame reconcile in `window_closed`, which also restores
    // the desktop mode on leaving fullscreen.
    fn set_display_mode(&mut self, mode: DisplayMode) {
        self.fullscreen_display.set_desired(mode);
    }

    fn clipboard(&mut self) -> Option<&mut dyn Clipboard> {
        Some(&mut *self.state)
    }

    fn shared_clipboard(&self) -> Option<Box<dyn Clipboard + Send + Sync>> {
        Some(Box::new(super::clipboard::WindowClipboard::new(
            self.state.hwnd,
        )))
    }
}

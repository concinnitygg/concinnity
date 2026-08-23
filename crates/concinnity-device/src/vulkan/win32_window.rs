// src/vulkan/win32_window.rs
//
// The Vulkan backend's Windows window: a thin adapter over the shared native
// Win32 layer (crate::win32) the DirectX backend also uses, so the two
// HWND-rendering backends share one window/input/display-mode implementation
// with identical behavior (wnd_proc, raw-input camera deltas, cursor
// capture/confinement, window modes, Resolution-row mode switching). GLFW
// (window.rs) remains the windowing layer on Linux only; the surface is
// created directly through VK_KHR_win32_surface.

use ash::vk;

use crate::assets::WindowMode;
use crate::gfx::display_mode::DisplayMode;
use crate::gfx::input::RenderInput;
use crate::gfx::keymap::KeyMap;
use crate::win32::display_mode::{self, FullscreenDisplayMode};
use crate::win32::window::{
    WindowState, create_window, do_capture_cursor, do_release_cursor, do_set_ui_cursor_hidden,
    do_set_window_mode, do_set_window_size, frame_tick, take_input_snapshot,
};

pub(crate) struct Win32Window {
    win_state: Box<WindowState>,
    // The user's chosen fullscreen display mode, held on the monitor while the
    // window is fullscreen and restored on exit / drop; reconciled once per
    // frame by `frame_tick` in `poll`.
    fullscreen_display: FullscreenDisplayMode,
}

impl Win32Window {
    // Create the window. Always created windowed (like DirectX); a
    // non-default creation mode is applied immediately after. (The engine
    // currently creates Windowed here and applies the world / persisted mode
    // through GraphicsSystem's init, the same flow as DirectX.)
    pub(crate) fn new(
        title: &str,
        width: u32,
        height: u32,
        mode: &WindowMode,
        _resizable: bool,
        title_bar: bool,
    ) -> Result<Self, String> {
        let (_hwnd, win_state) = create_window(title, width, height, title_bar)?;
        let mut this = Self {
            win_state,
            fullscreen_display: FullscreenDisplayMode::new(),
        };
        if !matches!(mode, WindowMode::Windowed) {
            this.set_window_mode(*mode);
        }
        Ok(this)
    }

    // Drain the message pump, refresh the cursor window-exit / confinement
    // state, and reconcile the fullscreen display mode. Returns true when the
    // window was closed. Called once per frame by `VkContext::window_closed`.
    pub(crate) fn poll(&mut self) -> bool {
        frame_tick(&mut self.win_state, &mut self.fullscreen_display)
    }

    // Snapshot of the accumulated input since the last call.
    pub(crate) fn take_input(&mut self) -> RenderInput {
        take_input_snapshot(&mut self.win_state)
    }

    // Arm click-to-capture rather than grabbing the cursor immediately: a
    // freshly spawned window may not be focused, and grabbing before the user
    // interacts is jarring. The first content click captures (the same flow
    // as DirectX and as GLFW's focus-gated engage on Linux).
    pub(crate) fn capture_cursor(&mut self) {
        self.win_state.recapture_on_click = true;
    }

    // Symmetric with `capture_cursor`; reached only through
    // `set_camera_capture` today, kept so the cursor API stays complete.
    #[expect(
        dead_code,
        reason = "symmetric with capture_cursor; reached only through set_camera_capture today"
    )]
    pub(crate) fn release_cursor(&mut self) {
        do_release_cursor(&mut self.win_state);
    }

    // Hide or show the OS cursor for an in-engine UI cursor (e.g. a MainMenu),
    // without engaging camera capture. Edge-triggered in the helper.
    pub(crate) fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        do_set_ui_cursor_hidden(&mut self.win_state, hidden);
    }

    // A togglable menu coexists with a captured camera; see
    // `RenderBackend::set_menu_mode`. The wnd_proc reads this flag to route
    // Escape to the ECS and suppress click-to-recapture.
    pub(crate) fn set_menu_mode(&mut self, on: bool) {
        self.win_state.menu_mode = on;
    }

    // Edge-triggered capture: capture for camera control, release while a
    // menu is open. Unlike the startup `capture_cursor` (which arms
    // click-to-capture), closing the menu recaptures immediately so the
    // camera resumes without an extra click.
    pub(crate) fn set_camera_capture(&mut self, capture: bool) {
        if capture == self.win_state.cursor_captured {
            return;
        }
        if capture {
            let hwnd = self.win_state.hwnd;
            do_capture_cursor(hwnd, &mut self.win_state);
        } else {
            do_release_cursor(&mut self.win_state);
        }
    }

    // Whether the real cursor has left the window so the renderer should stop
    // drawing the in-engine UI cursor (windowed / borderless). Recomputed each
    // `poll`; false while captured or in fullscreen (which confines instead).
    pub(crate) fn cursor_outside_window(&self) -> bool {
        self.win_state.cursor_outside_window
    }

    // Replace the runtime movement key map; takes effect on the next message.
    pub(crate) fn set_keymap(&mut self, keymap: &KeyMap) {
        self.win_state.key.set_keymap(keymap);
    }

    pub(crate) fn set_window_mode(&mut self, mode: WindowMode) {
        do_set_window_mode(&mut self.win_state, mode);
    }

    pub(crate) fn set_window_size(&mut self, width: u32, height: u32) {
        do_set_window_size(&mut self.win_state, width, height);
    }

    // The display modes of the window's monitor, feeding the Resolution
    // settings row (the caller dedups + sorts). Enumerated live, unlike the
    // GLFW window's creation-time cache (no &mut constraint here).
    pub(crate) fn display_modes(&self) -> Vec<DisplayMode> {
        display_mode::enumerate(self.win_state.hwnd)
    }

    // The mode the window's monitor is currently running (what the Resolution
    // row shows before the user ever picks one).
    pub(crate) fn current_display_mode(&self) -> Option<DisplayMode> {
        display_mode::current(self.win_state.hwnd)
    }

    // Remember the display mode to hold while fullscreen. Applied by the
    // per-frame reconcile in `poll` (which also restores the desktop mode on
    // leaving fullscreen), so a choice made in any window mode takes effect
    // when fullscreen is (or becomes) active.
    pub(crate) fn set_display_mode(&mut self, mode: DisplayMode) {
        self.fullscreen_display.set_desired(mode);
    }

    // The swapchain-facing surface size in pixels (the client area, tracked
    // via WM_SIZE). Named for parity with the GLFW window's
    // `framebuffer_size`; on Windows client pixels are framebuffer pixels.
    pub(crate) fn framebuffer_size(&self) -> (i32, i32) {
        (self.win_state.width, self.win_state.height)
    }

    // The overlay coordinate space. Windows reports WM_MOUSEMOVE in the same
    // client pixels the swapchain is sized to, so logical units are framebuffer
    // pixels here and this equals `framebuffer_size`.
    pub(crate) fn logical_size(&self) -> (f32, f32) {
        (self.win_state.width as f32, self.win_state.height as f32)
    }

    // Create the presentation surface for this window via
    // VK_KHR_win32_surface.
    pub(crate) fn create_surface(
        &mut self,
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<vk::SurfaceKHR, String> {
        // SAFETY: passing None asks for the handle of the current process image, which is always
        // valid.
        let hinstance = unsafe { windows::Win32::System::LibraryLoader::GetModuleHandleW(None) }
            .map_err(|e| format!("GetModuleHandleW: {e}"))?;
        let info = vk::Win32SurfaceCreateInfoKHR::default()
            .hinstance(hinstance.0 as isize)
            .hwnd(self.win_state.hwnd.0 as isize);
        let loader = ash::khr::win32_surface::Instance::new(entry, instance);
        // SAFETY: `info` borrows the module handle and HWND for the call; both name live Win32
        // objects owned by this window.
        unsafe { loader.create_win32_surface(&info, None) }
            .map_err(|e| format!("vkCreateWin32SurfaceKHR: {e}"))
    }

    // Vulkan instance extensions required for surface creation on Windows.
    pub(crate) fn required_instance_extensions(&self) -> Vec<String> {
        [ash::khr::surface::NAME, ash::khr::win32_surface::NAME]
            .into_iter()
            .map(|n| n.to_str().unwrap_or_default().to_string())
            .collect()
    }
}

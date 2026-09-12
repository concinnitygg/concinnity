//! The window a backend renders into, as the settings menu sees it.
//!
//! Vsync, window mode and size, the display-mode list, the overlay's logical
//! size, and the cursor state the UI drives. Every method here is about the
//! surface and its input, never about the scene drawn on it.

use crate::render::keymap::KeyMap;
use alloc::vec::Vec;

/// The presentation surface and its input: vsync, window mode and size, the
/// display modes the target supports, and the cursor.
///
/// All defaulted. A backend on a fixed surface (an embedded view, a headless
/// target) reports the conservative values and ignores the mutators, which is
/// what lets the settings menu ask any backend these questions.
pub trait WindowControl {
    /// The overlay coordinate space: the window's content size in logical,
    /// DPI-independent units (points on macOS, client pixels on Windows, window
    /// coordinates on Linux). Every backend reports the cursor in these same
    /// units, so UI hit-testing, text layout, and the overlay shader's divide to
    /// NDC all share one space regardless of the backing scale. A backend
    /// converts to attachment pixels only where a pixel rect is unavoidable,
    /// through `fullscreen::clip_rect_to_scissor`.
    ///
    /// Default `(0.0, 0.0)` for a headless backend with no window.
    fn logical_size(&self) -> (f32, f32) {
        (0.0, 0.0)
    }

    /// Height of the window chrome overlapping the top of the render surface,
    /// in the logical units `logical_size` reports. Non-zero only where the
    /// content view runs under a transparent title bar (macOS), which leaves
    /// the OS window buttons floating over the frame's top-left corner. UI that
    /// must stay clear of them starts below this; the frame itself still covers
    /// the whole window.
    ///
    /// Default `0.0`: a window whose content already begins below its chrome.
    fn top_content_inset(&self) -> f32 {
        0.0
    }

    /// Show or hide the OS cursor for an in-engine UI cursor (e.g. a MainMenu),
    /// independent of camera capture. Edge-triggered by the backend, so calling
    /// it every frame with the same value is cheap. Default no-op: a backend
    /// without a free-mode cursor hide leaves the system cursor visible.
    fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        let _ = hidden;
    }

    /// Whether the real cursor has left the window, so an in-engine UI cursor
    /// should stop drawing (windowed / borderless). The backend confines the
    /// cursor to the active screen while in fullscreen, so it reports `false`
    /// there. Default `false` (inside): a backend without window-bounds tracking
    /// always draws the in-engine cursor.
    fn cursor_outside_window(&self) -> bool {
        false
    }

    /// Tell the backend a togglable menu (a Screen toggled by an Escape KeyBinding)
    /// coexists with a captured camera. In this mode Escape routes to the ECS
    /// (so the menu shows/hides) instead of releasing the cursor inline, and a
    /// click never recaptures the cursor (it fires a UI action). Set once at
    /// setup. Default no-op: a backend without dynamic capture keeps the static
    /// behavior.
    fn set_menu_mode(&mut self, on: bool) {
        let _ = on;
    }

    /// Drive cursor capture from the menu state each frame: capture for camera
    /// control, release while a menu is open. Edge-triggered by the backend.
    /// Default no-op: the startup capture decision stands.
    fn set_camera_capture(&mut self, capture: bool) {
        let _ = capture;
    }

    /// Turn display sync (vsync) on or off at runtime, applied to presentation.
    /// Edge-triggered by the backend, so calling it with the unchanged value is
    /// cheap. Default no-op: a backend that only honors vsync at init ignores
    /// runtime changes.
    fn set_vsync(&mut self, on: bool) {
        let _ = on;
    }

    /// Switch the window between windowed / borderless / fullscreen at runtime.
    /// The change flows through the backend's normal resize path (no GPU rebuild
    /// beyond the resize it triggers). Default no-op for backends without a
    /// window (embedded / preview) or that don't yet implement it.
    fn set_window_mode(&mut self, mode: crate::components::WindowMode) {
        let _ = mode;
    }

    /// Resize the window's content area at runtime (meaningful in windowed mode).
    /// Drives the same resize path as a user-dragged resize. Default no-op for
    /// backends without a window or that don't yet implement it.
    fn set_window_size(&mut self, width: u32, height: u32) {
        let _ = (width, height);
    }

    /// The display modes (pixel resolution + refresh rate) the display this
    /// backend renders to supports, unshaped (the caller dedups + sorts).
    /// Default empty: a backend that cannot enumerate (or has no window) makes
    /// the Resolution row fall back to the static preset list.
    fn display_modes(&self) -> Vec<crate::render::display_mode::DisplayMode> {
        Vec::new()
    }

    /// The mode the display is currently running, if the backend can read it.
    /// Shown by the Resolution row when the user has never chosen a mode (the
    /// display keeps its desktop mode until one is chosen). Default `None`.
    fn current_display_mode(&self) -> Option<crate::render::display_mode::DisplayMode> {
        None
    }

    /// Select the display mode to hold while the window is in fullscreen. The
    /// backend applies it whenever the window is (or becomes) fullscreen and
    /// restores the display's original mode when the window leaves fullscreen
    /// or shuts down; outside fullscreen the choice is only remembered. Default
    /// no-op: a backend without mode switching leaves the display alone.
    fn set_display_mode(&mut self, mode: crate::render::display_mode::DisplayMode) {
        let _ = mode;
    }

    /// Push the gameplay movement key map. The backend resolves each canonical
    /// `InputKey` to its native key code and decodes physical key events through the
    /// map (instead of hardcoded keys), so a settings-menu rebind takes effect on
    /// the next key event. Pushed once after the backend is built and again on
    /// each rebind. Default no-op: a backend without keymap decode keeps its
    /// built-in defaults.
    fn set_keymap(&mut self, keymap: &KeyMap) {
        let _ = keymap;
    }
}

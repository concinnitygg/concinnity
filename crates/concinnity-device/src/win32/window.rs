// src/win32/window.rs
//
// Win32 window creation, the window proc, cursor capture/release, and the
// message pump, shared by the DirectX backend and the Vulkan backend's
// Windows window (vulkan/win32_window.rs).
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::{
    ClientToScreen, GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromWindow,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{SetFocus, VK_ESCAPE, VK_MENU};
use windows::Win32::UI::WindowsAndMessaging::*;

use crate::components::WindowMode;

use super::chrome::windowed_style;
use super::input::*;

//  Parked window for editor live-swap reuse
//
// The `cn editor` live world-swap drops the old render backend and builds a new
// one on the same thread (see the client `run_init` `PendingBackend` fallback).
// A full rebuild would otherwise call `create_window` again and pop a brand-new
// OS window every edit, leaking the old one (nothing ever `DestroyWindow`s it).
// Instead the outgoing `WindowState` parks its HWND here on drop, and the next
// `create_window` adopts it -- so the reload keeps the exact same window (no
// flash, no reposition), matching the Metal backend's window-preserving reload.
//
// Thread-local because the whole swap runs on the render/main thread; a shipped
// game never drops its backend mid-run, so this is inert outside the editor.
thread_local! {
    static PARKED_WINDOW: std::cell::Cell<Option<HWND>> = const { std::cell::Cell::new(None) };
}

fn park_window(hwnd: HWND) {
    PARKED_WINDOW.with(|p| p.set(Some(hwnd)));
}

fn take_parked_window() -> Option<HWND> {
    PARKED_WINDOW.with(|p| p.take())
}

//  Window proc state (thread-local)

// Because Win32 window procs are global C callbacks, we stash the mutable input
// state as a raw pointer in the window's GWLP_USERDATA slot so the proc can
// reach it without unsafe global statics.

pub(crate) struct WindowState {
    // The window this state belongs to. Stored so the DxContext cursor methods
    // (which only hold the WindowState) can reach the client rect for a
    // menu-driven recapture; the wnd_proc gets the same handle as a parameter.
    pub(crate) hwnd: HWND,
    pub(crate) key: KeyState,
    pub(crate) mouse_dx: f32,
    pub(crate) mouse_dy: f32,
    pub(crate) mouse_x: f32,
    pub(crate) mouse_y: f32,
    pub(crate) left_click_pending: bool,
    // True while the left button is held with the cursor free (a UI drag, e.g.
    // a settings Slider handle). Set on WM_LBUTTONDOWN, cleared on WM_LBUTTONUP;
    // unlike `left_click_pending` it persists across take_input() so a drag can
    // track the cursor. Mirrors the Metal `left_button_down` signal.
    pub(crate) left_button_down: bool,
    // Set on WM_RBUTTONDOWN with the cursor free; a one-shot cleared by
    // take_input(). Mirrors the Metal `right_click_pulse` signal.
    pub(crate) right_click_pending: bool,
    // Accumulated vertical scroll-wheel delta since the last take_input(), in
    // scroll_delta units (WM_MOUSEWHEEL notches scaled via
    // `wheel_notches_to_scroll_delta`). Reset each take_input().
    pub(crate) scroll_delta: f32,
    pub(crate) cursor_captured: bool,
    // Set when the cursor is released via Escape so the next left-click in the
    // content area recaptures it instead of firing a UI click.
    pub(crate) recapture_on_click: bool,
    // Whether the OS cursor is currently hidden for an in-engine UI cursor
    // (e.g. a MainMenu). Tracked so `set_ui_cursor_hidden` only flips Win32's
    // ShowCursor display count on a transition, keeping it balanced against
    // capture's own hide/show.
    pub(crate) ui_cursor_hidden: bool,
    // A togglable menu coexists with a captured camera (a MainMenu over a
    // Camera3D world). When set, Escape routes to the ECS and clicks never
    // recapture; GraphicsSystem drives capture from the active menu instead.
    pub(crate) menu_mode: bool,
    // The current window mode. Tracked so the per-frame cursor confinement
    // knows whether to confine (Fullscreen) or hide the in-engine arrow on
    // leave (Windowed / Borderless). The window is always created windowed;
    // `do_set_window_mode` keeps this in sync as the settings menu cycles it.
    pub(crate) window_mode: WindowMode,
    // The world's authored `Window.title_bar`. Held because `do_set_window_mode`
    // rebuilds the style every time the settings menu cycles back to Windowed
    // and has to reinstate the authored chrome, not a standard caption.
    pub(crate) title_bar: bool,
    // Whether the real cursor has left the window content area while the cursor
    // is free (windowed / borderless). Recomputed each frame by
    // `update_ui_cursor_confinement`; the renderer hides the in-engine cursor
    // when set. False while captured or in fullscreen (which confines instead).
    pub(crate) cursor_outside_window: bool,
    // Whether the per-frame confinement currently holds a `ClipCursor` clip for
    // a fullscreen menu (distinct from capture's own clip). Tracked so the clip
    // is released exactly once when the confining condition ends (mode change or
    // capture engaging), keeping it balanced against capture's clip.
    pub(crate) menu_clip_active: bool,
    pub(crate) closed: bool,
    pub(crate) width: i32,
    pub(crate) height: i32,
}

impl Drop for WindowState {
    fn drop(&mut self) {
        // Park this HWND for the next `create_window` to adopt instead of
        // destroying it, so the editor's live world-swap keeps the same OS
        // window (see the module note on `PARKED_WINDOW`). Clear the userdata
        // pointer first so a stray message dispatched between park and adopt
        // cannot dereference this `WindowState` after it frees; the whole swap is
        // synchronous, so in practice no message runs in that gap. If nothing
        // adopts it (e.g. the process is exiting, or a rebuild failed), the HWND
        // stays parked until the next `create_window` or is reclaimed by the OS
        // at process exit -- never presented to, so it is harmless.
        // SAFETY: `self.hwnd` is this window's live handle, and the call only stores an integer in
        // a per-window slot.
        unsafe { SetWindowLongPtrW(self.hwnd, GWLP_USERDATA, 0) };
        park_window(self.hwnd);
    }
}

// Cursor capture/release helpers shared by the wnd_proc and the backends'
// context methods. Both callers need them, and the wnd_proc cannot reach the
// context because it only has the WindowState pointer stored in GWLP_USERDATA.
pub(crate) fn do_capture_cursor(hwnd: HWND, state: &mut WindowState) {
    // Capture only while the window is foreground. Engaging from the
    // background half-works: RIDEV_INPUTSINK raw input flows (the camera
    // moves) but the OS ignores a background ClipCursor and the keyboard
    // stays with the foreground app, leaving a "captured" window whose
    // cursor still wanders. Declining (without setting cursor_captured)
    // makes the callers converge instead: the menu-driven
    // set_camera_capture retries every frame until focus arrives, and the
    // click-to-capture path is always foreground by the time the button
    // message is delivered.
    // SAFETY: a read of the foreground window handle; it borrows nothing.
    if unsafe { GetForegroundWindow() } != hwnd {
        return;
    }
    state.cursor_captured = true;
    state.recapture_on_click = false;
    // SAFETY: adjusts this thread's cursor display count; it borrows nothing.
    unsafe { ShowCursor(false) };
    let mut rect = windows::Win32::Foundation::RECT::default();
    // SAFETY: `hwnd` is this window's live handle, and `rect` is a live local the call fills.
    if unsafe { GetClientRect(hwnd, &mut rect) }.is_ok() {
        let mut tl = POINT {
            x: rect.left,
            y: rect.top,
        };
        let mut br = POINT {
            x: rect.right,
            y: rect.bottom,
        };
        // SAFETY: `hwnd` is this window's live handle, and both points are live locals the calls
        // convert in place.
        unsafe {
            let _ = ClientToScreen(hwnd, &mut tl);
            let _ = ClientToScreen(hwnd, &mut br);
        }
        let screen_rect = windows::Win32::Foundation::RECT {
            left: tl.x,
            top: tl.y,
            right: br.x,
            bottom: br.y,
        };
        // SAFETY: `screen_rect` is a live local the call only reads.
        let _ = unsafe { ClipCursor(Some(&screen_rect)) };
    }
    // Discard any spurious deltas accumulated before capture.
    state.mouse_dx = 0.0;
    state.mouse_dy = 0.0;
}

pub(crate) fn do_release_cursor(state: &mut WindowState) {
    if !state.cursor_captured {
        return;
    }
    state.cursor_captured = false;
    state.recapture_on_click = true;
    // SAFETY: neither call borrows state: one drops the clip, the other adjusts this thread's
    // cursor display count.
    unsafe {
        let _ = ClipCursor(None);
        ShowCursor(true);
    }
}

// Hide or show the OS cursor for an in-engine UI cursor (e.g. a MainMenu),
// without engaging camera capture. Edge-triggered on `ui_cursor_hidden`: Win32
// keeps a per-thread cursor display count, so we flip ShowCursor only on a
// transition to keep it balanced against capture's own hide/show.
pub(crate) fn do_set_ui_cursor_hidden(state: &mut WindowState, hidden: bool) {
    if hidden == state.ui_cursor_hidden {
        return;
    }
    state.ui_cursor_hidden = hidden;
    // SAFETY: adjusts this thread's cursor display count; it borrows nothing.
    unsafe { ShowCursor(!hidden) };
}

// The window's client area in screen coordinates (top-left origin, y down),
// or None if the rect query fails. Shared by the confinement's content-area
// test and its fullscreen `ClipCursor` bounds.
fn client_screen_rect(hwnd: HWND) -> Option<RECT> {
    let mut rect = RECT::default();
    // SAFETY: `hwnd` is this window's live handle, and `rect` is a live local the call fills.
    if unsafe { GetClientRect(hwnd, &mut rect) }.is_err() {
        return None;
    }
    let mut tl = POINT {
        x: rect.left,
        y: rect.top,
    };
    let mut br = POINT {
        x: rect.right,
        y: rect.bottom,
    };
    // SAFETY: `hwnd` is this window's live handle, and both points are live locals the calls
    // convert in place.
    unsafe {
        if ClientToScreen(hwnd, &mut tl).as_bool() && ClientToScreen(hwnd, &mut br).as_bool() {
            Some(RECT {
                left: tl.x,
                top: tl.y,
                right: br.x,
                bottom: br.y,
            })
        } else {
            None
        }
    }
}

// Per-frame bookkeeping for an in-engine UI cursor (a menu), mirroring the
// Metal `update_ui_cursor_confinement`: report whether the real cursor has left
// the window content area so the renderer can stop drawing the in-engine cursor
// in windowed / borderless modes, and confine the cursor to the window while in
// fullscreen so it cannot stray onto another display. A no-op while the cursor
// is captured (a gameplay camera owns the pointer). Called each frame after the
// message pump; `cursor_outside_window` is read by GraphicsSystem the same frame.
pub(crate) fn update_ui_cursor_confinement(state: &mut WindowState) {
    // While captured the pointer is already clipped + hidden for the camera, so
    // there is no in-engine arrow to hide and nothing to confine here. Capture
    // owns the clip now (`do_capture_cursor` set its own), so just relinquish our
    // flag without releasing -- releasing would undo capture's clip.
    if state.cursor_captured {
        state.menu_clip_active = false;
        state.cursor_outside_window = false;
        return;
    }
    let mut cursor = POINT::default();
    let (Ok(()), Some(rect)) = (
        // SAFETY: `cursor` is a live local the call fills.
        unsafe { GetCursorPos(&mut cursor) },
        client_screen_rect(state.hwnd),
    ) else {
        release_menu_clip(state);
        state.cursor_outside_window = false;
        return;
    };
    if matches!(state.window_mode, WindowMode::Fullscreen) {
        // Confine the cursor to the fullscreen window so a menu pointer cannot
        // wander onto another monitor -- but ONLY while this window is the
        // foreground window. ClipCursor is a global per-desktop resource: Windows
        // drops our clip when the window is deactivated, and the render loop keeps
        // ticking while backgrounded (run_loop_default never blocks on the message
        // pump), so re-asserting the clip unconditionally would yank the cursor
        // away from whatever app the user Alt+Tabbed to. When not foreground we
        // just clear our flag: Windows already released the clip, and issuing our
        // own ClipCursor(None) from the background could stomp the foreground app's
        // clip. It is a hard OS confine (no visible snap-back), re-applied each
        // frame while foreground; released once when the condition ends
        // (see `release_menu_clip`).
        // SAFETY: a read of the foreground window handle; it borrows nothing.
        if unsafe { GetForegroundWindow() } == state.hwnd {
            // SAFETY: `rect` is a live local the call only reads.
            let _ = unsafe { ClipCursor(Some(&rect)) };
            state.menu_clip_active = true;
        } else {
            state.menu_clip_active = false;
        }
        state.cursor_outside_window = false;
        return;
    }
    // Windowed / borderless: the in-engine cursor shows only while the real
    // cursor is over the content area. Drop any fullscreen menu clip first (the
    // mode may have just changed out of fullscreen).
    release_menu_clip(state);
    let inside = cursor.x >= rect.left
        && cursor.x < rect.right
        && cursor.y >= rect.top
        && cursor.y < rect.bottom;
    state.cursor_outside_window = !inside;
}

// Release the fullscreen-menu `ClipCursor` clip if the confinement holds one.
// Edge-triggered so it is balanced against capture's own clip and never frees a
// clip we do not own.
fn release_menu_clip(state: &mut WindowState) {
    if state.menu_clip_active {
        state.menu_clip_active = false;
        // SAFETY: the call releases this thread's clip and borrows nothing.
        let _ = unsafe { ClipCursor(None) };
    }
}

// NOTE: the window-mode / window-size helpers below mirror the Metal
// implementation (`metal/input.rs`) for cross-backend parity but are written on
// macOS and have NOT been built or run on Windows; verify the exact `windows`
// crate signatures and the borderless/resize behavior on a Windows host.
//
// Switch the window between windowed / borderless / fullscreen by swapping the
// window style and repositioning. Borderless and fullscreen both map to a
// borderless window covering the current monitor: exclusive DXGI fullscreen
// (SetFullscreenState) is deliberately avoided -- it is documented as fraught
// with alt-tab, multi-display, and resolution-change issues. SetWindowPos fires
// WM_SIZE, which the resize path turns into a ResizeBuffers.
pub(crate) fn do_set_window_mode(state: &mut WindowState, mode: WindowMode) {
    let hwnd = state.hwnd;
    // Record the mode so the per-frame cursor confinement can tell fullscreen
    // (confine) from windowed / borderless (hide the arrow on leave).
    state.window_mode = mode;
    // SAFETY: `hwnd` is this window's live handle, and every rect and style these calls read or
    // fill is a live local.
    unsafe {
        match mode {
            WindowMode::Windowed => {
                let style = windowed_style(state.title_bar);
                SetWindowLongPtrW(hwnd, GWL_STYLE, style.0 as isize);
                let w = state.width.max(640);
                let h = state.height.max(480);
                let mut rect = RECT {
                    left: 0,
                    top: 0,
                    right: w,
                    bottom: h,
                };
                let _ = AdjustWindowRect(&mut rect, style, false);
                let _ = SetWindowPos(
                    hwnd,
                    None,
                    80,
                    80,
                    rect.right - rect.left,
                    rect.bottom - rect.top,
                    SWP_FRAMECHANGED | SWP_NOZORDER,
                );
            }
            WindowMode::Borderless | WindowMode::Fullscreen => {
                SetWindowLongPtrW(hwnd, GWL_STYLE, (WS_POPUP | WS_VISIBLE).0 as isize);
                if let Some(rect) = monitor_rect(hwnd) {
                    let _ = SetWindowPos(
                        hwnd,
                        None,
                        rect.left,
                        rect.top,
                        rect.right - rect.left,
                        rect.bottom - rect.top,
                        SWP_FRAMECHANGED | SWP_NOZORDER,
                    );
                }
            }
        }
        let _ = ShowWindow(hwnd, SW_SHOW);
    }
}

// Resize the window's content area (windowed mode only). AdjustWindowRect
// converts the desired client size to the full window rect; WM_SIZE then drives
// ResizeBuffers.
pub(crate) fn do_set_window_size(state: &mut WindowState, width: u32, height: u32) {
    let hwnd = state.hwnd;
    // SAFETY: `hwnd` is this window's live handle, and `rect` is a live local `AdjustWindowRect`
    // fills before `SetWindowPos` reads it.
    unsafe {
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        let _ = AdjustWindowRect(&mut rect, windowed_style(state.title_bar), false);
        let _ = SetWindowPos(
            hwnd,
            None,
            0,
            0,
            rect.right - rect.left,
            rect.bottom - rect.top,
            SWP_NOMOVE | SWP_NOZORDER,
        );
    }
}

// Work-area-inclusive bounds of the monitor the window is mostly on.
fn monitor_rect(hwnd: HWND) -> Option<RECT> {
    // SAFETY: `hwnd` is this window's live handle, `mon` comes from the query on the line above,
    // and `info` is a live local whose own `cbSize` tells the call how much to fill.
    unsafe {
        let mon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
        let mut info = MONITORINFO {
            cbSize: std::mem::size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if GetMonitorInfoW(mon, &mut info).as_bool() {
            Some(info.rcMonitor)
        } else {
            None
        }
    }
}

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: `install_window_state` stored the address of this window's boxed `WindowState`, whose
    // heap allocation outlives the window, and `WindowState::drop` clears the slot before the HWND
    // is parked, so a non-null pointer here always addresses a live state. Messages reach this
    // procedure only from `pump_messages` on the thread that owns the window, so the `&mut` is not
    // aliased by another dispatch, and every handle and local the arms below name is live for the
    // call.
    unsafe {
        let state_ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut WindowState;
        if state_ptr.is_null() {
            return DefWindowProcW(hwnd, msg, wparam, lparam);
        }
        let state = &mut *state_ptr;

        match msg {
            WM_DESTROY | WM_CLOSE => {
                state.closed = true;
                PostQuitMessage(0);
                LRESULT(0)
            }
            WM_SIZE => {
                state.width = (lparam.0 & 0xFFFF) as i32;
                state.height = ((lparam.0 >> 16) & 0xFFFF) as i32;
                LRESULT(0)
            }
            WM_KEYDOWN => {
                let vk = vk_from_wparam(wparam.0);
                // In menu mode (a MainMenu over a captured camera) Escape always
                // pulses so UiInputSystem can toggle the menu and GraphicsSystem
                // drives capture from there. Otherwise: a captured-cursor world
                // releases the cursor (the safe exit; the window stays in front
                // and a click recaptures), and a free-cursor world pulses for
                // UiInputSystem. Same split as `metal/input.rs`.
                if vk == VK_ESCAPE {
                    if state.menu_mode || !state.cursor_captured {
                        state.key.on_escape_uncaptured();
                    } else {
                        do_release_cursor(state);
                    }
                }
                state.key.on_key_down(vk);
                LRESULT(0)
            }
            WM_KEYUP => {
                state.key.on_key_up(vk_from_wparam(wparam.0));
                LRESULT(0)
            }
            WM_SYSKEYDOWN | WM_SYSKEYUP => {
                // Alt arrives here rather than through WM_KEYDOWN/WM_KEYUP.
                // Track the modifier, then swallow it: DefWindowProc treats a
                // bare Alt press/release as window-menu activation and enters a
                // modal loop that stalls the message pump (and so the render
                // loop) until the next click. Every other system key still goes
                // to DefWindowProc, so Alt+F4 and Alt+Enter are unaffected.
                let vk = vk_from_wparam(wparam.0);
                state.key.on_sys_key(vk, msg == WM_SYSKEYDOWN);
                if vk == VK_MENU {
                    LRESULT(0)
                } else {
                    DefWindowProcW(hwnd, msg, wparam, lparam)
                }
            }
            WM_CHAR => {
                // `TranslateMessage` (in `pump_messages`) synthesises WM_CHAR
                // from WM_KEYDOWN with the layout / Shift / dead-key resolution
                // already applied, so wParam is the final UTF-16 code unit. Feed
                // printable glyphs to text-input fields; `on_char` filters out the
                // control codes (Backspace, Enter, Escape, ...). Lone surrogate
                // halves (non-BMP input) yield `None` and are dropped, matching
                // the one-codepoint-per-frame contract.
                if let Some(c) = char::from_u32(wparam.0 as u32) {
                    state.key.on_char(c);
                }
                LRESULT(0)
            }
            WM_KILLFOCUS => {
                // Free the cursor when the window loses focus so Alt+Tab works.
                // Arm click-to-recapture so an explicit click back into the
                // window re-captures (menu-mode worlds instead re-capture via
                // the per-frame set_camera_capture once the window is
                // foreground again).
                if state.cursor_captured {
                    state.cursor_captured = false;
                    state.recapture_on_click = true;
                    let _ = ClipCursor(None);
                    ShowCursor(true);
                }
                // Alt+Tab consumes the Alt release, so drop the held modifiers
                // rather than leaving them set for the rest of the session.
                state.key.on_focus_lost();
                LRESULT(0)
            }
            WM_MOUSEMOVE => {
                let x = (lparam.0 & 0xFFFF) as i16 as f32;
                let y = ((lparam.0 >> 16) & 0xFFFF) as i16 as f32;
                if !state.cursor_captured {
                    // Track the cursor position for UI hit-testing only. Camera-look
                    // deltas come solely from raw input while the cursor is captured
                    // (WM_INPUT below), mirroring metal/input.rs (which accumulates
                    // mouse_dx only when captured) and the GLFW path. Accumulating an
                    // absolute-position delta here yanked the camera on the first
                    // move: mouse_x/mouse_y start at 0, so the first delta was the
                    // full cursor coordinate -- a ~90-degree yaw plus a pitch slammed
                    // to the floor at scene start (gameplay input is gated on the
                    // menu, not on capture, so it reached the camera controller).
                    state.mouse_x = x;
                    state.mouse_y = y;
                }
                LRESULT(0)
            }
            WM_INPUT => {
                // Raw input for captured-cursor delta. The packet is read
                // straight into a `RAWINPUT` rather than a `Vec<u8>` that is
                // then cast: a byte vector carries no alignment guarantee, and
                // only mouse devices are registered, so a `RAWINPUT`-sized
                // destination always holds the whole packet.
                if state.cursor_captured {
                    let mut raw = windows::Win32::UI::Input::RAWINPUT::default();
                    let mut size =
                        std::mem::size_of::<windows::Win32::UI::Input::RAWINPUT>() as u32;
                    let copied = windows::Win32::UI::Input::GetRawInputData(
                        windows::Win32::UI::Input::HRAWINPUT(lparam.0 as _),
                        windows::Win32::UI::Input::RID_INPUT,
                        Some(&mut raw as *mut _ as *mut std::ffi::c_void),
                        &mut size,
                        std::mem::size_of::<windows::Win32::UI::Input::RAWINPUTHEADER>() as u32,
                    );
                    if copied != u32::MAX
                        && raw.header.dwType == windows::Win32::UI::Input::RIM_TYPEMOUSE.0
                    {
                        state.mouse_dx += raw.data.mouse.lLastX as f32;
                        state.mouse_dy += raw.data.mouse.lLastY as f32;
                    }
                }
                LRESULT(0)
            }
            WM_LBUTTONDOWN => {
                if !state.cursor_captured {
                    // In menu mode a click fires a UI action; capture is driven
                    // by the active menu, not by clicking (mirrors metal/input.rs).
                    if !state.menu_mode && state.recapture_on_click {
                        do_capture_cursor(hwnd, state);
                    } else {
                        state.left_click_pending = true;
                        // Begin a held-button (UI drag) gesture.
                        state.left_button_down = true;
                    }
                }
                LRESULT(0)
            }
            WM_RBUTTONDOWN => {
                // A right press is only a UI signal (context menus); it never
                // captures or recaptures the cursor, unlike WM_LBUTTONDOWN.
                if !state.cursor_captured {
                    state.right_click_pending = true;
                }
                LRESULT(0)
            }
            WM_LBUTTONUP => {
                // End any held-button (drag) gesture. Always cleared, even if the
                // press began while captured, so the flag can never stick across a
                // capture transition. Mirrors metal/input.rs.
                state.left_button_down = false;
                LRESULT(0)
            }
            WM_MOUSEWHEEL => {
                // Accumulate the wheel delta for scrollable UI while the cursor is
                // free. The high word of wParam is a signed multiple of
                // WHEEL_DELTA (120) per notch, positive when rotated away from the
                // user; normalise to notches and convert to a scroll_delta
                // increment (matching the Metal sign convention).
                if !state.cursor_captured {
                    let raw = (wparam.0 >> 16) as i16 as f32;
                    let notches = raw / WHEEL_DELTA as f32;
                    state.scroll_delta += crate::gfx::input::wheel_notches_to_scroll_delta(notches);
                }
                LRESULT(0)
            }
            _ => DefWindowProcW(hwnd, msg, wparam, lparam),
        }
    }
}

//  DxContext

//  Win32 helpers

// Build a fresh `WindowState` for `hwnd` at its default (uncaptured, windowed)
// state. Shared by fresh window creation and parked-window adoption.
fn fresh_window_state(hwnd: HWND, width: i32, height: i32, title_bar: bool) -> Box<WindowState> {
    Box::new(WindowState {
        hwnd,
        title_bar,
        key: KeyState::default(),
        mouse_dx: 0.0,
        mouse_dy: 0.0,
        mouse_x: 0.0,
        mouse_y: 0.0,
        left_click_pending: false,
        left_button_down: false,
        right_click_pending: false,
        scroll_delta: 0.0,
        cursor_captured: false,
        recapture_on_click: false,
        ui_cursor_hidden: false,
        menu_mode: false,
        // The window is (re)adopted as a standard titled window; a persisted
        // Borderless / Fullscreen choice is applied later via set_window_mode.
        window_mode: WindowMode::Windowed,
        cursor_outside_window: false,
        menu_clip_active: false,
        closed: false,
        width,
        height,
    })
}

// Register raw mouse input for `hwnd` and point the wnd_proc at `win_state` via
// GWLP_USERDATA. The Box's heap allocation is stable across moves of the Box
// itself, so the stashed pointer stays valid for the window's lifetime. Shared
// by fresh creation and parked-window adoption (re-registering raw input on an
// already-registered HWND just re-sets the target, which is harmless).
fn install_window_state(hwnd: HWND, win_state: &mut WindowState) {
    let rid = windows::Win32::UI::Input::RAWINPUTDEVICE {
        usUsagePage: 0x01,
        usUsage: 0x02, // mouse
        dwFlags: windows::Win32::UI::Input::RIDEV_INPUTSINK,
        hwndTarget: hwnd,
    };
    // SAFETY: `rid` is a live local, and the element size passed alongside it is that local's own
    // type.
    let _ = unsafe {
        windows::Win32::UI::Input::RegisterRawInputDevices(
            &[rid],
            std::mem::size_of::<windows::Win32::UI::Input::RAWINPUTDEVICE>() as u32,
        )
    };
    // SAFETY: the stored address is that of a `WindowState` held in a `Box` whose heap allocation
    // outlives the window, so `wnd_proc` can dereference it for as long as the window delivers
    // messages.
    unsafe { SetWindowLongPtrW(hwnd, GWLP_USERDATA, win_state as *mut WindowState as isize) };
}

// Adopt a window parked by a prior `WindowState::drop` (the editor live-swap
// path): reuse the existing HWND with a fresh `WindowState` instead of creating
// a new OS window. Its actual client size is queried so the rebuilt swapchain
// matches a window the user may have resized. The window is already shown +
// focused, so no ShowWindow / SetForegroundWindow is needed (and skipping them
// avoids a flicker).
fn adopt_parked_window(hwnd: HWND, title_bar: bool) -> (HWND, Box<WindowState>) {
    let (width, height) = {
        let mut rect = RECT::default();
        // SAFETY: `hwnd` is the parked window's live handle, and `rect` is a live local the call
        // fills.
        if unsafe { GetClientRect(hwnd, &mut rect) }.is_ok() {
            (
                (rect.right - rect.left).max(1),
                (rect.bottom - rect.top).max(1),
            )
        } else {
            (1, 1)
        }
    };
    let mut win_state = fresh_window_state(hwnd, width, height, title_bar);
    install_window_state(hwnd, &mut win_state);
    (hwnd, win_state)
}

pub(crate) fn create_window(
    title: &str,
    width: u32,
    height: u32,
    title_bar: bool,
) -> Result<(HWND, Box<WindowState>), String> {
    // Reuse a window parked by a prior backend's drop (editor live-swap) rather
    // than popping a new one, so a world reload keeps the same OS window.
    if let Some(hwnd) = take_parked_window() {
        let (hwnd, state) = adopt_parked_window(hwnd, title_bar);
        // The adopted window carries the outgoing world's style, and the state
        // it is adopted into is Windowed. Restyle in place -- SWP_NOMOVE |
        // SWP_NOSIZE keeps the reposition-free reuse parking exists for -- so a
        // reload that flipped `title_bar` (or parked while borderless) lands on
        // the style the state claims.
        // SAFETY: `hwnd` is the parked window's live handle, and the restyle borrows nothing else.
        unsafe {
            SetWindowLongPtrW(hwnd, GWL_STYLE, windowed_style(title_bar).0 as isize);
            let _ = SetWindowPos(
                hwnd,
                None,
                0,
                0,
                0,
                0,
                SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER,
            );
        }
        return Ok((hwnd, state));
    }

    let class_name: Vec<u16> = "ConcinnityWindow\0".encode_utf16().collect();
    let title_wide: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();

    // SAFETY: a query for this process's own module handle; it borrows nothing.
    let hinstance = unsafe { windows::Win32::System::LibraryLoader::GetModuleHandleW(None) }
        .map_err(|e| format!("GetModuleHandle: {e}"))?;

    let wc = WNDCLASSEXW {
        cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(wnd_proc),
        hInstance: hinstance.into(),
        lpszClassName: windows::core::PCWSTR(class_name.as_ptr()),
        // SAFETY: loads a built-in system cursor named by a constant; it borrows nothing.
        hCursor: unsafe { LoadCursorW(None, IDC_ARROW).unwrap_or_default() },
        ..Default::default()
    };
    // SAFETY: `wc` and the NUL-terminated class-name buffer it points at are live for the call,
    // which copies the name into the process's class atom table.
    unsafe { RegisterClassExW(&wc) };

    let style = windowed_style(title_bar);
    let mut rect = windows::Win32::Foundation::RECT {
        left: 0,
        top: 0,
        right: width as i32,
        bottom: height as i32,
    };
    // SAFETY: `rect` is a live local the call adjusts in place.
    unsafe { AdjustWindowRect(&mut rect, style, false) }.ok();

    // SAFETY: the window class was registered above, and the NUL-terminated class-name and title
    // buffers outlive the call.
    let hwnd = unsafe {
        CreateWindowExW(
            WINDOW_EX_STYLE::default(),
            windows::core::PCWSTR(class_name.as_ptr()),
            windows::core::PCWSTR(title_wide.as_ptr()),
            style,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(hinstance.into()),
            None,
        )
    }
    .map_err(|e| format!("CreateWindowExW: {e}"))?;

    // SAFETY: `hwnd` is the window just created and still owned here; none of these calls borrow
    // anything else.
    unsafe {
        let _ = ShowWindow(hwnd, SW_SHOW);
        // Explicitly take foreground + keyboard focus. SW_SHOW alone can
        // leave the launching terminal focused (the OS foreground lock),
        // which used to start the engine keyboard-less: the camera moved
        // (raw input is INPUTSINK) while keystrokes went to the old app.
        // Best-effort: when Windows denies the foreground switch this
        // no-ops and the first click into the window focuses + captures.
        let _ = SetForegroundWindow(hwnd);
        let _ = SetFocus(Some(hwnd));
    };

    let mut win_state = fresh_window_state(hwnd, width as i32, height as i32, title_bar);

    // Register for raw mouse input (the captured-cursor camera delta arrives
    // via WM_INPUT, not WM_MOUSEMOVE) and point the wnd_proc at the state.
    install_window_state(hwnd, &mut win_state);

    Ok((hwnd, win_state))
}

// Per-frame window service shared by both backends, called once at the top of
// each frame (`window_closed`): drain the message pump, refresh the in-engine
// cursor's window-exit / fullscreen-confinement state (read by GraphicsSystem
// later the same frame), and converge the monitor on the chosen Resolution
// mode (held while the window is in borderless fullscreen, restored
// otherwise). When the monitor mode just changed under a monitor-covering
// window (a fullscreen switch, or a restore on the way out to borderless),
// re-cover the monitor's new bounds; a windowed window keeps its own rect.
// Returns whether the window was closed.
pub(crate) fn frame_tick(
    state: &mut WindowState,
    display: &mut super::display_mode::FullscreenDisplayMode,
) -> bool {
    pump_messages();
    update_ui_cursor_confinement(state);
    let mode = state.window_mode;
    let fullscreen = matches!(mode, WindowMode::Fullscreen);
    if display.reconcile(state.hwnd, fullscreen) && !matches!(mode, WindowMode::Windowed) {
        do_set_window_mode(state, mode);
    }
    state.closed
}

// Drain the accumulated input into a `RenderInput` snapshot, shared by both
// backends' `take_input`. The mouse delta, pending click, and scroll are
// one-shot (reset here); the held-button flag persists until WM_LBUTTONUP and
// the keyboard one-shots are reset inside `KeyState::take`.
pub(crate) fn take_input_snapshot(state: &mut WindowState) -> crate::gfx::input::RenderInput {
    let dx = state.mouse_dx;
    let dy = state.mouse_dy;
    let mx = state.mouse_x;
    let my = state.mouse_y;
    let lc = state.left_click_pending;
    let lbd = state.left_button_down;
    let rc = state.right_click_pending;
    let scroll = state.scroll_delta;
    state.mouse_dx = 0.0;
    state.mouse_dy = 0.0;
    state.left_click_pending = false;
    state.right_click_pending = false;
    state.scroll_delta = 0.0;
    state.key.take(MouseSnapshot {
        dx,
        dy,
        x: mx,
        y: my,
        left_click: lc,
        left_button_down: lbd,
        right_click: rc,
        scroll_delta: scroll,
    })
}

pub(crate) fn pump_messages() {
    let mut msg = MSG::default();
    // SAFETY: `msg` is a live local the pump fills.
    while unsafe { PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE) }.as_bool() {
        // SAFETY: `msg` was just filled by `PeekMessageW` and is live for both calls.
        // `DispatchMessageW` re-enters `wnd_proc`, which reaches its own window's state through the
        // per-window slot rather than through anything borrowed here.
        unsafe {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The parked-window handoff (`WindowState::drop` parks; the next
    // `create_window` adopts) is what keeps the editor live-swap on one OS
    // window instead of popping a new one each edit. Exercise the pure slot
    // logic; the HWND is a fake handle since no real window is created here.
    #[test]
    fn park_then_take_round_trips_and_is_consumed_once() {
        // Clear any residue from a reused test thread first.
        let _ = take_parked_window();
        assert!(take_parked_window().is_none(), "empty when nothing parked");

        let fake = HWND(0x1234 as *mut core::ffi::c_void);
        park_window(fake);
        let taken = take_parked_window();
        assert_eq!(
            taken.map(|h| h.0 as usize),
            Some(0x1234),
            "the parked HWND is handed back to the next create_window"
        );
        assert!(
            take_parked_window().is_none(),
            "take consumes the parked window (a second create_window builds fresh)"
        );
    }

    #[test]
    fn parking_overwrites_the_previous_slot() {
        let _ = take_parked_window();
        park_window(HWND(0xAAAA as *mut core::ffi::c_void));
        park_window(HWND(0xBBBB as *mut core::ffi::c_void));
        assert_eq!(
            take_parked_window().map(|h| h.0 as usize),
            Some(0xBBBB),
            "only the most-recently parked window is adopted"
        );
    }
}

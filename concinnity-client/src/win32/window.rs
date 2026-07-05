// src/win32/window.rs
//
// Win32 window creation, the window proc, cursor capture/release, and the
// message pump, shared by the DirectX backend and the Vulkan backend's
// Windows window (vulkan/win32_window.rs).
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::{
    ClientToScreen, GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromWindow,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{SetFocus, VK_ESCAPE};
use windows::Win32::UI::WindowsAndMessaging::*;

use crate::assets::WindowMode;

use super::input::*;

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
    if unsafe { GetForegroundWindow() } != hwnd {
        return;
    }
    state.cursor_captured = true;
    state.recapture_on_click = false;
    unsafe { ShowCursor(false) };
    let mut rect = windows::Win32::Foundation::RECT::default();
    if unsafe { GetClientRect(hwnd, &mut rect) }.is_ok() {
        let mut tl = POINT {
            x: rect.left,
            y: rect.top,
        };
        let mut br = POINT {
            x: rect.right,
            y: rect.bottom,
        };
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
    unsafe { ShowCursor(!hidden) };
}

// The window's client area in screen coordinates (top-left origin, y down),
// or None if the rect query fails. Shared by the confinement's content-area
// test and its fullscreen `ClipCursor` bounds.
fn client_screen_rect(hwnd: HWND) -> Option<RECT> {
    let mut rect = RECT::default();
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
        if unsafe { GetForegroundWindow() } == state.hwnd {
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
    unsafe {
        match mode {
            WindowMode::Windowed => {
                SetWindowLongPtrW(hwnd, GWL_STYLE, WS_OVERLAPPEDWINDOW.0 as isize);
                let w = state.width.max(640);
                let h = state.height.max(480);
                let mut rect = RECT {
                    left: 0,
                    top: 0,
                    right: w,
                    bottom: h,
                };
                let _ = AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false);
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
// ResizeBuffers. Unverified on Windows (see the note above).
pub(crate) fn do_set_window_size(state: &mut WindowState, width: u32, height: u32) {
    let hwnd = state.hwnd;
    unsafe {
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        let _ = AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false);
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
                // Raw input for captured-cursor delta.
                if state.cursor_captured {
                    let mut size: u32 = 0;
                    windows::Win32::UI::Input::GetRawInputData(
                        windows::Win32::UI::Input::HRAWINPUT(lparam.0 as _),
                        windows::Win32::UI::Input::RID_INPUT,
                        None,
                        &mut size,
                        std::mem::size_of::<windows::Win32::UI::Input::RAWINPUTHEADER>() as u32,
                    );
                    if size > 0 {
                        let mut buf = vec![0u8; size as usize];
                        windows::Win32::UI::Input::GetRawInputData(
                            windows::Win32::UI::Input::HRAWINPUT(lparam.0 as _),
                            windows::Win32::UI::Input::RID_INPUT,
                            Some(buf.as_mut_ptr() as *mut std::ffi::c_void),
                            &mut size,
                            std::mem::size_of::<windows::Win32::UI::Input::RAWINPUTHEADER>() as u32,
                        );
                        let raw = &*(buf.as_ptr() as *const windows::Win32::UI::Input::RAWINPUT);
                        if raw.header.dwType == windows::Win32::UI::Input::RIM_TYPEMOUSE.0 {
                            state.mouse_dx += raw.data.mouse.lLastX as f32;
                            state.mouse_dy += raw.data.mouse.lLastY as f32;
                        }
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

pub(crate) fn create_window(
    title: &str,
    width: u32,
    height: u32,
) -> Result<(HWND, Box<WindowState>), String> {
    let class_name: Vec<u16> = "ConcinnityWindow\0".encode_utf16().collect();
    let title_wide: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();

    let hinstance = unsafe { windows::Win32::System::LibraryLoader::GetModuleHandleW(None) }
        .map_err(|e| format!("GetModuleHandle: {e}"))?;

    let wc = WNDCLASSEXW {
        cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(wnd_proc),
        hInstance: hinstance.into(),
        lpszClassName: windows::core::PCWSTR(class_name.as_ptr()),
        hCursor: unsafe { LoadCursorW(None, IDC_ARROW).unwrap_or_default() },
        ..Default::default()
    };
    unsafe { RegisterClassExW(&wc) };

    let style = WS_OVERLAPPEDWINDOW;
    let mut rect = windows::Win32::Foundation::RECT {
        left: 0,
        top: 0,
        right: width as i32,
        bottom: height as i32,
    };
    unsafe { AdjustWindowRect(&mut rect, style, false) }.ok();

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

    let mut win_state = Box::new(WindowState {
        hwnd,
        key: KeyState::default(),
        mouse_dx: 0.0,
        mouse_dy: 0.0,
        mouse_x: 0.0,
        mouse_y: 0.0,
        left_click_pending: false,
        left_button_down: false,
        scroll_delta: 0.0,
        cursor_captured: false,
        recapture_on_click: false,
        ui_cursor_hidden: false,
        menu_mode: false,
        // The window is created as a standard titled window; a persisted
        // Borderless / Fullscreen choice is applied later via set_window_mode.
        window_mode: WindowMode::Windowed,
        cursor_outside_window: false,
        menu_clip_active: false,
        closed: false,
        width: width as i32,
        height: height as i32,
    });

    // Register for raw mouse input (the captured-cursor camera delta arrives
    // via WM_INPUT, not WM_MOUSEMOVE).
    let rid = windows::Win32::UI::Input::RAWINPUTDEVICE {
        usUsagePage: 0x01,
        usUsage: 0x02, // mouse
        dwFlags: windows::Win32::UI::Input::RIDEV_INPUTSINK,
        hwndTarget: hwnd,
    };
    let _ = unsafe {
        windows::Win32::UI::Input::RegisterRawInputDevices(
            &[rid],
            std::mem::size_of::<windows::Win32::UI::Input::RAWINPUTDEVICE>() as u32,
        )
    };

    // Stash the state pointer in GWLP_USERDATA so the wnd_proc can reach it.
    // The Box's heap allocation is stable across moves of the Box itself, so
    // the pointer stays valid for the window's lifetime.
    unsafe {
        SetWindowLongPtrW(
            hwnd,
            GWLP_USERDATA,
            &mut *win_state as *mut WindowState as isize,
        )
    };

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
    let scroll = state.scroll_delta;
    state.mouse_dx = 0.0;
    state.mouse_dy = 0.0;
    state.left_click_pending = false;
    state.scroll_delta = 0.0;
    state.key.take(MouseSnapshot {
        dx,
        dy,
        x: mx,
        y: my,
        left_click: lc,
        left_button_down: lbd,
        scroll_delta: scroll,
    })
}

pub(crate) fn pump_messages() {
    let mut msg = MSG::default();
    while unsafe { PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE) }.as_bool() {
        unsafe {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

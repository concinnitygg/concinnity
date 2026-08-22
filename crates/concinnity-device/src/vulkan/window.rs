// src/vulkan/window.rs
//
// GLFW window and input for the Vulkan backend on Linux. (On Windows the
// backend uses the shared native Win32 layer instead -- see win32_window.rs;
// this module is compiled only off-Windows.)
//
// Input design mirrors metal.rs: events accumulate into an InputState between
// poll() calls; GraphicsSystem drains the state each step via take_input()
// and deposits it as a FrameInput component for Camera3DSystem to consume.
//
// Cursor capture is enabled by GraphicsSystem::init() when a Camera3D
// component is present. GLFW's CursorDisabled mode delivers raw relative
// deltas directly via CursorPos events, so no manual warping is needed.

use crate::assets::{InputKey, WindowMode};
use crate::gfx::display_mode::DisplayMode;
use crate::gfx::keymap::KeyMap;

// The previously-duplicated InputState collapsed into the shared
// crate::gfx::input::RenderInput; this alias keeps the historical name.
pub use crate::gfx::input::RenderInput as InputState;

// Owns the GLFW library handle, the window, and the event receiver.
//
// Created once by GraphicsSystem during init(); polled every step().
// All GLFW calls must happen on the thread that created this struct -- the
// world loop guarantees single-threaded system execution.
pub(crate) struct GlfwWindow {
    pub glfw: glfw::Glfw,
    pub window: glfw::PWindow,
    events: glfw::GlfwReceiver<(f64, glfw::WindowEvent)>,
    // last cursor position, used to compute deltas when not in raw mode
    last_cursor: Option<(f64, f64)>,
    input: InputState,
    cursor_captured: bool,
    // Whether the OS cursor is hidden for an in-engine UI cursor (e.g. a
    // MainMenu) while not captured. GLFW's cursor mode is a single enum, so the
    // effective mode is computed from both this and `cursor_captured` (see
    // `apply_cursor_mode`); tracked so `set_ui_cursor_hidden` only re-applies on
    // a transition.
    ui_cursor_hidden: bool,
    // A togglable menu coexists with a captured camera (a MainMenu over a
    // Camera3D world). When set, Escape routes to the ECS instead of releasing
    // the cursor inline; GraphicsSystem drives capture from the active menu.
    menu_mode: bool,
    // The current window mode. Tracked so the per-frame cursor confinement knows
    // whether to confine (Fullscreen) or hide the in-engine arrow on leave
    // (Windowed / Borderless). Seeded from the creation mode and kept in sync by
    // `set_window_mode`.
    window_mode: WindowMode,
    // Whether the real cursor has left the window content area while the cursor
    // is free (windowed / borderless). Recomputed each poll by
    // `update_ui_cursor_confinement`; the renderer hides the in-engine cursor
    // when set. False while captured or in fullscreen (which confines instead).
    cursor_outside_window: bool,
    // The runtime movement key map. The key event arm decodes through it instead
    // of hardcoded keys, so a settings-menu rebind takes effect immediately.
    // Defaults to W/S/A/D/Shift/Space/E. (GLFW delivers Shift as Left/Right Shift
    // key events, so it is just another key here -- no separate modifier path.)
    keymap: KeyMap,
    // The primary monitor's video modes + desktop mode, enumerated once at
    // creation: the &self getters cannot take the &mut Glfw a live query
    // needs, and the hardware mode list does not change mid-session.
    display_modes: Vec<DisplayMode>,
    desktop_mode: Option<DisplayMode>,
    // The user's chosen fullscreen display mode (snapped to a real video
    // mode). Fullscreen switches the monitor to it via set_monitor; GLFW
    // itself restores the desktop mode when the window leaves fullscreen or
    // is destroyed, so no explicit hold/restore machinery is needed here
    // (unlike DirectX / Metal).
    desired_mode: Option<DisplayMode>,
}

// The (width, height, refresh Hz) of a GLFW video mode. GLFW reports 0 for an
// unknown refresh rate (e.g. Wayland), which is already DisplayMode's
// "unknown" convention.
fn display_mode_of(v: &glfw::VidMode) -> DisplayMode {
    DisplayMode {
        width: v.width,
        height: v.height,
        refresh_hz: v.refresh_rate,
    }
}

// SAFETY: GlfwWindow is only ever used on the thread that created it.
unsafe impl Send for GlfwWindow {}

// Resolve the GLFW cursor mode from the two independent intents. A captured
// camera locks + hides the cursor (Disabled, raw deltas) and takes precedence;
// otherwise an in-engine UI cursor hides it but keeps it freely positioned
// (Hidden); with neither, the OS cursor is shown (Normal).
fn resolve_cursor_mode(captured: bool, ui_cursor_hidden: bool) -> glfw::CursorMode {
    if captured {
        glfw::CursorMode::Disabled
    } else if ui_cursor_hidden {
        glfw::CursorMode::Hidden
    } else {
        glfw::CursorMode::Normal
    }
}

// Apply a key transition to whichever gameplay actions are bound to `key`.
// `pressed` is the held state (movement / sprint follow it) and, on a press,
// fires the one-shot actions (jump / interact). Mirrors the Metal / DirectX
// `apply_binding`.
fn apply_binding(input: &mut InputState, km: KeyMap, key: InputKey, pressed: bool) {
    if km.forward == key {
        input.forward = pressed;
    }
    if km.backward == key {
        input.backward = pressed;
    }
    if km.left == key {
        input.left = pressed;
    }
    if km.right == key {
        input.right = pressed;
    }
    if km.sprint == key {
        input.sprint = pressed;
    }
    if pressed {
        if km.jump == key {
            input.jump = true;
        }
        if km.interact == key {
            input.interact = true;
        }
    }
}

// Map a GLFW key to a canonical `InputKey`, or `None` for a key the engine does not
// bind (function keys, Escape, Ctrl/Alt, keypad, etc.). Left/Right Shift both map
// to `InputKey::Shift`: GLFW delivers them as ordinary key events.
fn key_from_glfw(key: glfw::InputKey) -> Option<InputKey> {
    use glfw::InputKey as G;
    Some(match key {
        G::A => InputKey::A,
        G::B => InputKey::B,
        G::C => InputKey::C,
        G::D => InputKey::D,
        G::E => InputKey::E,
        G::F => InputKey::F,
        G::G => InputKey::G,
        G::H => InputKey::H,
        G::I => InputKey::I,
        G::J => InputKey::J,
        G::K => InputKey::K,
        G::L => InputKey::L,
        G::M => InputKey::M,
        G::N => InputKey::N,
        G::O => InputKey::O,
        G::P => InputKey::P,
        G::Q => InputKey::Q,
        G::R => InputKey::R,
        G::S => InputKey::S,
        G::T => InputKey::T,
        G::U => InputKey::U,
        G::V => InputKey::V,
        G::W => InputKey::W,
        G::X => InputKey::X,
        G::Y => InputKey::Y,
        G::Z => InputKey::Z,
        G::Num0 => InputKey::Num0,
        G::Num1 => InputKey::Num1,
        G::Num2 => InputKey::Num2,
        G::Num3 => InputKey::Num3,
        G::Num4 => InputKey::Num4,
        G::Num5 => InputKey::Num5,
        G::Num6 => InputKey::Num6,
        G::Num7 => InputKey::Num7,
        G::Num8 => InputKey::Num8,
        G::Num9 => InputKey::Num9,
        G::Space => InputKey::Space,
        G::Tab => InputKey::Tab,
        G::Enter => InputKey::Enter,
        G::Backspace => InputKey::Backspace,
        G::Delete => InputKey::Delete,
        G::LeftShift | G::RightShift => InputKey::Shift,
        G::Up => InputKey::Up,
        G::Down => InputKey::Down,
        G::Left => InputKey::Left,
        G::Right => InputKey::Right,
        G::Minus => InputKey::Minus,
        G::Equal => InputKey::Equals,
        G::LeftBracket => InputKey::LeftBracket,
        G::RightBracket => InputKey::RightBracket,
        G::Backslash => InputKey::Backslash,
        G::Semicolon => InputKey::Semicolon,
        G::Apostrophe => InputKey::Quote,
        G::Comma => InputKey::Comma,
        G::Period => InputKey::Period,
        G::Slash => InputKey::Slash,
        G::GraveAccent => InputKey::Backtick,
        _ => return None,
    })
}

impl GlfwWindow {
    // create a new glfw window with no opengl context (vulkan surface mode)
    pub(crate) fn new(
        title: &str,
        width: u32,
        height: u32,
        mode: &WindowMode,
        resizable: bool,
        title_bar: bool,
    ) -> Result<Self, String> {
        let mut glfw = glfw::init(glfw::fail_on_errors).map_err(|e| format!("glfw init: {e}"))?;

        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
        glfw.window_hint(glfw::WindowHint::Resizable(resizable));
        // An undecorated window loses the whole frame, close button included:
        // X11/Wayland draw their controls in the title bar, unlike macOS where
        // the traffic lights float over the content and survive. The Borderless
        // arm below re-hints this for its own creation.
        glfw.window_hint(glfw::WindowHint::Decorated(title_bar));

        // The Resolution row's mode list + the desktop mode. Queried before
        // the window exists so a fullscreen creation never reports its own
        // (possibly switched) mode as the desktop mode.
        let (display_modes, desktop_mode) = glfw.with_primary_monitor(|_, monitor| match monitor {
            Some(m) => (
                m.get_video_modes().iter().map(display_mode_of).collect(),
                m.get_video_mode().map(|v| display_mode_of(&v)),
            ),
            None => (Vec::new(), None),
        });

        let (mut window, events) = match mode {
            WindowMode::Windowed => glfw
                .create_window(width, height, title, glfw::WindowMode::Windowed)
                .ok_or_else(|| "Failed to create GLFW window (windowed)".to_string())?,

            WindowMode::Fullscreen => glfw.with_primary_monitor(|glfw, monitor| {
                let monitor = monitor.ok_or("No primary monitor")?;
                glfw.create_window(width, height, title, glfw::WindowMode::FullScreen(monitor))
                    .ok_or_else(|| "Failed to create GLFW window (fullscreen)".to_string())
            })?,

            WindowMode::Borderless => glfw.with_primary_monitor(|glfw, monitor| {
                let monitor = monitor.ok_or("No primary monitor")?;
                let vid_mode = monitor
                    .get_video_mode()
                    .ok_or("Could not query primary monitor video mode")?;
                glfw.window_hint(glfw::WindowHint::Decorated(false));
                glfw.create_window(
                    vid_mode.width,
                    vid_mode.height,
                    title,
                    glfw::WindowMode::Windowed, // borderless = undecorated windowed
                )
                .ok_or_else(|| "Failed to create GLFW window (borderless)".to_string())
            })?,
        };

        window.set_close_polling(true);
        window.set_key_polling(true);
        // Char events deliver the layout- / modifier-resolved printable glyph for
        // text-input fields (the editor's name / filter / arg fields). Without
        // this GLFW never queues Char events, so the `Char` arm in `poll()` never
        // runs and text fields cannot be typed into.
        window.set_char_polling(true);
        window.set_cursor_pos_polling(true);
        // Mouse-button events drive UI clicks (e.g. a MainMenu HitRegion).
        // Without this GLFW never queues Button events, so the `MouseButton`
        // arm in `poll()` never runs and clicks are silently dropped.
        window.set_mouse_button_polling(true);
        // Scroll events drive scrollable UI (e.g. the settings panel). Without
        // this GLFW never queues Scroll events, so the `Scroll` arm in `poll()`
        // never runs and the wheel is silently dropped.
        window.set_scroll_polling(true);
        window.set_framebuffer_size_polling(true);

        Ok(Self {
            glfw,
            window,
            events,
            last_cursor: None,
            input: InputState::default(),
            cursor_captured: false,
            ui_cursor_hidden: false,
            menu_mode: false,
            window_mode: *mode,
            cursor_outside_window: false,
            keymap: KeyMap::default(),
            display_modes,
            desktop_mode,
            desired_mode: None,
        })
    }

    // Push the cursor mode resolved from the two independent intents onto the
    // window. Centralised because GLFW exposes one mode enum where Metal /
    // DirectX keep two independent ref-counts.
    fn apply_cursor_mode(&mut self) {
        self.window.set_cursor_mode(resolve_cursor_mode(
            self.cursor_captured,
            self.ui_cursor_hidden,
        ));
    }

    // Hide the cursor and begin delivering relative mouse deltas via CursorPos
    // events. Should be called once after the window is shown, when a
    // Camera3D component is present.
    pub(crate) fn capture_cursor(&mut self) {
        self.cursor_captured = true;
        self.apply_cursor_mode();
        // enable raw mouse motion if the platform supports it -- bypasses
        // pointer acceleration for more direct 1:1 feel
        if self.glfw.supports_raw_motion() {
            self.window.set_raw_mouse_motion(true);
        }
        self.last_cursor = None;
    }

    // Show the cursor and stop accumulating relative deltas; symmetric with
    // `capture_cursor`. Driven by `set_camera_capture` in menu mode.
    pub(crate) fn release_cursor(&mut self) {
        if !self.cursor_captured {
            return;
        }
        self.cursor_captured = false;
        self.apply_cursor_mode();
    }

    // Hide or show the OS cursor for an in-engine UI cursor (e.g. a MainMenu),
    // without engaging camera capture. Edge-triggered: re-applies the combined
    // cursor mode only on a transition.
    pub(crate) fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        if hidden == self.ui_cursor_hidden {
            return;
        }
        self.ui_cursor_hidden = hidden;
        self.apply_cursor_mode();
    }

    // A togglable menu coexists with a captured camera; see
    // `RenderBackend::set_menu_mode`. The poll loop reads this to route Escape
    // to the ECS instead of releasing the cursor inline.
    pub(crate) fn set_menu_mode(&mut self, on: bool) {
        self.menu_mode = on;
    }

    // Edge-triggered capture: capture for camera control, release while a menu
    // is open. GraphicsSystem calls this each frame in menu mode.
    pub(crate) fn set_camera_capture(&mut self, capture: bool) {
        if capture == self.cursor_captured {
            return;
        }
        if capture {
            self.capture_cursor();
        } else {
            self.release_cursor();
        }
    }

    // Whether the real cursor has left the window so the renderer should stop
    // drawing the in-engine UI cursor (windowed / borderless). Recomputed each
    // `poll`; false while captured or in fullscreen (which confines instead).
    pub(crate) fn cursor_outside_window(&self) -> bool {
        self.cursor_outside_window
    }

    // Per-poll bookkeeping for an in-engine UI cursor (a menu), mirroring the
    // Metal `update_ui_cursor_confinement`: report whether the real cursor has
    // left the window content area so the renderer can stop drawing the in-engine
    // cursor in windowed / borderless modes, and confine the cursor to the window
    // while in fullscreen so it cannot stray onto another display. A no-op while
    // the cursor is captured (a gameplay camera owns the pointer, in GLFW's
    // Disabled mode).
    fn update_ui_cursor_confinement(&mut self) {
        if self.cursor_captured {
            // GLFW's Disabled mode owns the OS cursor; nothing to confine.
            self.cursor_outside_window = false;
            return;
        }
        if matches!(self.window_mode, WindowMode::Fullscreen) {
            // Confine the cursor to the fullscreen window so a menu pointer cannot
            // wander onto another monitor.
            self.confine_fullscreen();
            self.cursor_outside_window = false;
            return;
        }
        // Windowed / borderless: the in-engine cursor shows only while the real
        // cursor is over the content area. GLFW_HOVERED is the OS-tracked signal
        // for that, so no manual bounds test is needed.
        self.cursor_outside_window = !self.window.is_hovered();
    }

    // Best-effort fullscreen confine: no clip-to-window mode exists for a free
    // cursor under GLFW, so warp the cursor back to the content bounds each
    // poll (produces a slight snap-back at a multi-monitor edge). Gated on
    // input focus: GLFW's set_cursor_pos no-ops without focus anyway. A
    // single-display fullscreen is already OS-confined, so the clamp never
    // fires there.
    fn confine_fullscreen(&mut self) {
        let (w, h) = self.window.get_size();
        if self.window.is_focused() && w > 0 && h > 0 {
            let (cx, cy) = self.window.get_cursor_pos();
            let clamped_x = cx.clamp(0.0, (w - 1) as f64);
            let clamped_y = cy.clamp(0.0, (h - 1) as f64);
            if clamped_x != cx || clamped_y != cy {
                self.window.set_cursor_pos(clamped_x, clamped_y);
            }
        }
    }

    // NOTE: the window mode/size methods below mirror the Metal implementation
    // for cross-backend parity but were written on macOS and have NOT been built
    // or run on Linux/Windows. Verify the glfw crate API (`set_monitor`,
    // `set_decorated`, `set_size`) and surface survival across a monitor change
    // (Wayland may invalidate the Vulkan surface -- the present path already
    // rebuilds the swapchain on ERROR_OUT_OF_DATE_KHR, which should cover it).
    //
    // Switch windowed / borderless / fullscreen. The framebuffer-size change
    // makes the next present return OUT_OF_DATE, which rebuilds the swapchain.
    pub(crate) fn set_window_mode(&mut self, mode: WindowMode) {
        // Leaving a mode-switched fullscreen: GLFW restores the desktop mode
        // as part of the switch away, so a borderless cover must size to the
        // cached desktop mode, not the (still-switched) current video mode.
        let leaving_switched_fullscreen =
            matches!(self.window_mode, WindowMode::Fullscreen) && self.desired_mode.is_some();
        // Record the mode so the per-frame cursor confinement can tell fullscreen
        // (confine) from windowed / borderless (hide the arrow on leave).
        self.window_mode = mode;
        // Disjoint field borrows so the with_primary_monitor closure can drive
        // the window while glfw is borrowed for the monitor lookup.
        let desired_mode = self.desired_mode;
        let desktop_mode = self.desktop_mode;
        let glfw = &mut self.glfw;
        let window = &mut self.window;
        match mode {
            WindowMode::Windowed => {
                window.set_decorated(true);
                let (x, y) = window.get_pos();
                let (w, h) = window.get_size();
                window.set_monitor(
                    glfw::WindowMode::Windowed,
                    x.max(0),
                    y.max(0),
                    w.max(640) as u32,
                    h.max(480) as u32,
                    None,
                );
            }
            WindowMode::Borderless => {
                // Undecorated windowed at the monitor's video-mode size.
                window.set_decorated(false);
                glfw.with_primary_monitor(|_, monitor| {
                    if let Some(m) = monitor {
                        let size = if leaving_switched_fullscreen {
                            desktop_mode.map(|d| (d.width, d.height))
                        } else {
                            None
                        }
                        .or_else(|| m.get_video_mode().map(|vid| (vid.width, vid.height)));
                        if let Some((w, h)) = size {
                            window.set_monitor(glfw::WindowMode::Windowed, 0, 0, w, h, None);
                        }
                    }
                });
            }
            WindowMode::Fullscreen => {
                window.set_decorated(true);
                glfw.with_primary_monitor(|_, monitor| {
                    if let Some(m) = monitor {
                        // The chosen Resolution mode (already snapped to a
                        // real video mode), else the monitor's current one.
                        // GLFW holds the switched mode while fullscreen and
                        // restores the desktop mode on leaving it.
                        let target = desired_mode
                            .map(|d| (d.width, d.height, d.refresh_hz))
                            .or_else(|| {
                                m.get_video_mode()
                                    .map(|vid| (vid.width, vid.height, vid.refresh_rate))
                            });
                        if let Some((w, h, hz)) = target {
                            window.set_monitor(
                                glfw::WindowMode::FullScreen(m),
                                0,
                                0,
                                w,
                                h,
                                (hz != 0).then_some(hz),
                            );
                        }
                    }
                });
            }
        }
    }

    // The display modes of the primary monitor (enumerated at creation),
    // feeding the Resolution settings row; the caller dedups + sorts.
    pub(crate) fn display_modes(&self) -> Vec<DisplayMode> {
        self.display_modes.clone()
    }

    // The desktop mode of the primary monitor (captured at creation, before
    // any fullscreen switch): what the Resolution row shows before the user
    // ever picks a mode.
    pub(crate) fn current_display_mode(&self) -> Option<DisplayMode> {
        self.desktop_mode
    }

    // Remember the display mode to hold while fullscreen, snapped to a video
    // mode the monitor actually has (a stale persisted choice from another
    // display is ignored with a warning, mirroring Metal / DirectX). While
    // fullscreen it applies immediately by re-entering fullscreen at the new
    // mode; otherwise the next switch to Fullscreen picks it up.
    pub(crate) fn set_display_mode(&mut self, mode: DisplayMode) {
        let Some(idx) = crate::gfx::display_mode::best_native_index(&self.display_modes, mode)
        else {
            tracing::warn!(
                "display has no {}x{} mode; keeping the current mode",
                mode.width,
                mode.height
            );
            return;
        };
        let snapped = self.display_modes[idx];
        if self.desired_mode == Some(snapped) {
            return;
        }
        self.desired_mode = Some(snapped);
        if matches!(self.window_mode, WindowMode::Fullscreen) {
            self.set_window_mode(WindowMode::Fullscreen);
        }
    }

    // Resize the window (windowed mode only; GraphicsSystem gates this). The
    // framebuffer-size change triggers a swapchain rebuild via OUT_OF_DATE.
    pub(crate) fn set_window_size(&mut self, width: u32, height: u32) {
        self.window.set_size(width as i32, height as i32);
    }

    // Replace the runtime movement key map. `poll` decodes key events through
    // it, so a settings-menu rebind takes effect immediately.
    pub(crate) fn set_keymap(&mut self, keymap: &KeyMap) {
        self.keymap = *keymap;
    }

    // Drain all pending GLFW events, update input state, and return true if
    // the window should close. InputKey state is tracked as a running bitmask;
    // cursor deltas are accumulated so no delta is lost between poll calls.
    pub(crate) fn poll(&mut self) -> bool {
        self.glfw.poll_events();
        let mut should_close = self.window.should_close();

        for (_, event) in glfw::flush_messages(&self.events) {
            match event {
                glfw::WindowEvent::Close => {
                    should_close = true;
                }
                glfw::WindowEvent::InputKey(glfw::InputKey::Escape, _, glfw::Action::Press, _) => {
                    // In menu mode (a MainMenu over a captured camera) Escape
                    // always pulses so UiInputSystem can toggle the menu and
                    // GraphicsSystem drives capture from there. Otherwise a
                    // captured-cursor world releases the cursor (matching
                    // Metal / DirectX) and a free-cursor world pulses for
                    // UiInputSystem. The release is a direct field write rather
                    // than `release_cursor()` because `self.events` is borrowed
                    // by this loop; this branch is reached only with no UI
                    // cursor, so plain Normal is the correct combined mode.
                    if self.menu_mode || !self.cursor_captured {
                        self.input.escape = true;
                    } else {
                        self.window.set_cursor_mode(glfw::CursorMode::Normal);
                        self.cursor_captured = false;
                    }
                }
                glfw::WindowEvent::InputKey(glfw::InputKey::F1, _, glfw::Action::Press, _) => {
                    // F1 toggles the in-engine profiler HUD. Pulse-only
                    // (cleared by `take_input`).
                    self.input.hud_toggle = true;
                }
                glfw::WindowEvent::InputKey(key, _, action, _) => {
                    // Held Alt modifier (the editor's Alt+drag orbit). GLFW
                    // delivers both Alt keys as ordinary key events and
                    // `key_from_glfw` maps neither, so track it here rather than
                    // through the key map.
                    if matches!(key, glfw::InputKey::LeftAlt | glfw::InputKey::RightAlt)
                        && action != glfw::Action::Repeat
                    {
                        self.input.alt = action == glfw::Action::Press;
                    }
                    // Decode through the runtime key map (GLFW delivers Shift as
                    // Left/Right Shift key events, so it is handled like any other
                    // key -- no separate modifier path, matching DirectX).
                    if action != glfw::Action::Repeat
                        && let Some(canon) = key_from_glfw(key)
                    {
                        let pressed = action == glfw::Action::Press;
                        if pressed {
                            self.input.captured_key = Some(canon);
                        }
                        apply_binding(&mut self.input, self.keymap, canon, pressed);
                    }
                }
                glfw::WindowEvent::Char(c) => {
                    // The layout- / modifier-resolved printable glyph for
                    // text-input fields, one codepoint per event (matching
                    // `captured_key`). GLFW reports only text input here (no
                    // control / navigation keys), but filter defensively so the
                    // contract matches the Win32 / Metal paths. Editing keys
                    // (Backspace / Delete / Left / Right) ride `captured_key`.
                    if !c.is_control() {
                        self.input.typed_char = Some(c);
                    }
                }
                glfw::WindowEvent::CursorPos(x, y) => {
                    if self.cursor_captured {
                        if let Some((lx, ly)) = self.last_cursor {
                            self.input.mouse_dx += (x - lx) as f32;
                            self.input.mouse_dy += (y - ly) as f32;
                        }
                        self.last_cursor = Some((x, y));
                    } else {
                        // GLFW CursorPos has origin top-left with Y increasing
                        // downward -- matches TextLabel coords directly. It
                        // arrives in window coordinates, which are the overlay
                        // space (`logical_size`), so it needs no scaling: a
                        // hi-DPI framebuffer larger than the window is absorbed
                        // by the overlay's divide to NDC.
                        self.input.mouse_x = x as f32;
                        self.input.mouse_y = y as f32;
                    }
                }
                glfw::WindowEvent::MouseButton(
                    glfw::MouseButton::Button1,
                    glfw::Action::Press,
                    _,
                ) if !self.cursor_captured => {
                    self.input.left_click = true;
                    // Begin a held-button (UI drag) gesture.
                    self.input.left_button_down = true;
                }
                glfw::WindowEvent::MouseButton(
                    glfw::MouseButton::Button2,
                    glfw::Action::Press,
                    _,
                ) if !self.cursor_captured => {
                    // A right press is only a UI signal (context menus); it never
                    // begins a drag or recaptures the cursor. Mirrors metal / win32.
                    self.input.right_click = true;
                }
                glfw::WindowEvent::MouseButton(
                    glfw::MouseButton::Button1,
                    glfw::Action::Release,
                    _,
                ) => {
                    // End any held-button (drag) gesture. Always cleared, even if
                    // the press began while captured, so the flag can never stick
                    // across a capture transition. Mirrors metal / directx.
                    self.input.left_button_down = false;
                }
                glfw::WindowEvent::Scroll(_, yoffset) if !self.cursor_captured => {
                    // Accumulate the wheel delta for scrollable UI while the
                    // cursor is free. GLFW yoffset is in notches, positive when
                    // rotated away from the user; convert to a scroll_delta
                    // increment (matching the Metal sign convention).
                    self.input.scroll_delta +=
                        crate::gfx::input::wheel_notches_to_scroll_delta(yoffset as f32);
                }
                _ => {}
            }
        }

        // After draining this poll's events, refresh the in-engine cursor's
        // window-exit / fullscreen-confinement state (mirrors the tail of Metal's
        // `pump_ns_events`). GraphicsSystem reads `cursor_outside_window` later
        // this same frame.
        self.update_ui_cursor_confinement();

        should_close
    }

    // Return a snapshot of the current input state. Held-key flags
    // (forward/backward/left/right/sprint) and the absolute cursor position
    // persist -- they only change on a GLFW InputKey/CursorPos event, and GLFW
    // sends no events for a key that is simply held down (the first repeat
    // event lags the press by ~0.5 s). Resetting them here, as a blanket
    // `mem::take` once did, dropped held movement between events and made
    // the camera stutter for that gap. Only the momentary one-shot inputs
    // (interact/jump/left_click) and the per-call accumulated mouse delta
    // are cleared.
    pub(crate) fn take_input(&mut self) -> InputState {
        let snapshot = self.input;
        self.input.interact = false;
        self.input.jump = false;
        self.input.left_click = false;
        self.input.right_click = false;
        self.input.hud_toggle = false;
        self.input.escape = false;
        self.input.captured_key = None;
        // One-shot like `captured_key`: the printable glyph is consumed once.
        self.input.typed_char = None;
        self.input.mouse_dx = 0.0;
        self.input.mouse_dy = 0.0;
        // Accumulated like the mouse delta; the held-button flag persists until
        // its release event.
        self.input.scroll_delta = 0.0;
        snapshot
    }

    // The framebuffer size in pixels, the extent the swapchain is sized to.
    pub(crate) fn framebuffer_size(&self) -> (i32, i32) {
        self.window.get_framebuffer_size()
    }

    // The overlay coordinate space: GLFW window coordinates, the same units
    // `CursorPos` reports the cursor in. Equal to `framebuffer_size` on an
    // unscaled surface; smaller by the content scale on hi-DPI Wayland.
    pub(crate) fn logical_size(&self) -> (f32, f32) {
        let (w, h) = self.window.get_size();
        (w as f32, h as f32)
    }

    // Create the presentation surface for this window (GLFW picks the
    // platform's surface extension). `_entry` keeps the signature shared with
    // the Win32 window, which loads VK_KHR_win32_surface through it.
    pub(crate) fn create_surface(
        &mut self,
        _entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<ash::vk::SurfaceKHR, String> {
        use ash::vk::Handle;
        let mut raw_surface: usize = 0;
        // SAFETY: `instance.handle()` is the live Vulkan instance and `raw_surface` is a live local
        // GLFW writes the surface handle into; the allocator argument is null.
        let result = unsafe {
            self.window.create_window_surface(
                instance.handle().as_raw() as usize as *mut _,
                std::ptr::null(),
                &mut raw_surface as *mut usize as *mut *mut _,
            )
        };
        if result != 0 {
            Err(format!(
                "glfwCreateWindowSurface failed: VkResult({result})"
            ))
        } else {
            Ok(ash::vk::SurfaceKHR::from_raw(raw_surface as u64))
        }
    }

    // vulkan instance extensions required for surface creation on this platform
    pub(crate) fn required_instance_extensions(&self) -> Vec<String> {
        self.glfw
            .get_required_instance_extensions()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_mode_prefers_capture_then_ui_then_normal() {
        // Capture wins regardless of the UI-cursor intent (camera control needs
        // the locked, raw-delta Disabled mode).
        assert_eq!(resolve_cursor_mode(true, false), glfw::CursorMode::Disabled);
        assert_eq!(resolve_cursor_mode(true, true), glfw::CursorMode::Disabled);
        // Not captured but a UI cursor is shown: hide the OS cursor while
        // keeping it freely positioned.
        assert_eq!(resolve_cursor_mode(false, true), glfw::CursorMode::Hidden);
        // Neither: the OS cursor is visible.
        assert_eq!(resolve_cursor_mode(false, false), glfw::CursorMode::Normal);
    }

    #[test]
    fn editing_keys_decode_for_captured_key() {
        // Backspace and forward-delete decode so text fields can edit; they ride
        // `captured_key`, not `typed_char` (mirrors metal / win32). Printable
        // glyphs arrive separately via WindowEvent::Char.
        use glfw::InputKey as G;
        assert_eq!(key_from_glfw(G::Backspace), Some(InputKey::Backspace));
        assert_eq!(key_from_glfw(G::Delete), Some(InputKey::Delete));
        assert_eq!(key_from_glfw(G::Left), Some(InputKey::Left));
        assert_eq!(key_from_glfw(G::Right), Some(InputKey::Right));
    }
}

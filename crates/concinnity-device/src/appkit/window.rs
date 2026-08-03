// src/appkit/window.rs
//
// The AppKit window + input state every macOS backend owns, extracted from what
// used to be a block of `MtlContext` fields plus `metal/input.rs`. It works
// entirely through `NSView`, never the concrete view subclass, so the Metal
// backend can hand it its `MTKView` (kept for drawable acquisition) and the
// Vulkan backend a plain `CAMetalLayer`-backed view, and both get one
// window/input/display-mode implementation. Mirrors `win32/window.rs`.

#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2_app_kit::{
    NSApplication, NSCursor, NSEvent, NSEventMask, NSEventModifierFlags, NSEventType, NSScreen,
    NSView, NSWindow, NSWindowStyleMask, NSWindowTitleVisibility,
};
use objc2_foundation::{NSDate, NSPoint, NSSize};

use crate::assets::{Key, WindowMode};
use crate::gfx::display_mode::DisplayMode;
use crate::gfx::keymap::KeyMap;

use super::chrome::{apply_title_bar, set_window_buttons_hidden, windowed_style_mask};
use super::display_mode::{self, FullscreenDisplayMode};
use super::input::{InputState, KeyState, key_from_mac, printable_char};

unsafe extern "C" {
    // Moves the OS cursor without generating a mouse-moved event.
    fn CGWarpMouseCursorPosition(new_cursor_position: NSPoint) -> i32;
    // When connected=false (0), decouples cursor position from hardware mouse
    // movement so deltaX/deltaY in NSEvents are pure hardware deltas with no
    // warp feedback. Part of CoreGraphics (CGRemoteOperation.h).
    fn CGAssociateMouseAndMouseCursorPosition(connected: i32) -> i32;
}

// The window, view, and input state shared by the macOS backends. The backend
// keeps whatever it additionally needs to present (Metal: the `MTKView` this
// view upcasts from; Vulkan: the surface built from the view's CAMetalLayer).
pub(crate) struct AppKitWindow {
    // None in embedded mode (no separate NSWindow is created).
    window: Option<Retained<NSWindow>>,
    // The rendered-into view. Held as `NSView` so the concrete subclass stays a
    // backend concern; every operation here is inherited from NSView.
    view: Retained<NSView>,
    // The world's authored `Window.title_bar`. Held because `set_window_mode`
    // restyles the window every time the settings menu cycles back to Windowed
    // and has to reinstate the authored chrome, not a standard title bar.
    title_bar: bool,
    window_closed: bool,
    // Whether the frame loop should pump NSEvents and honour cursor capture.
    // True for windowed mode and for the blocking-in-view play path; false
    // for the preview (which lets the host own input dispatch).
    pump_events: bool,
    cursor_captured: bool,
    // Set when the cursor is released via Escape so a subsequent left-click
    // recaptures it rather than firing a UI click event.
    recapture_on_click: bool,
    // Whether the OS cursor is currently hidden for an in-engine UI cursor
    // (e.g. a MainMenu). Tracked so `set_ui_cursor_hidden` only calls the
    // ref-counted NSCursor hide/unhide on a transition, not every frame.
    ui_cursor_hidden: bool,
    // A togglable menu coexists with a captured camera (a MainMenu over a
    // Camera3D world). When set, Escape routes to the ECS and clicks never
    // recapture; GraphicsSystem drives capture from the active menu instead.
    menu_mode: bool,
    // Authoritative native-fullscreen state, kept in sync by `window_delegate`
    // (the NSWindow `FullScreen` style-mask bit lags the animated transition).
    // Read by `set_window_mode` / `set_window_size`; an `AtomicBool` because the
    // delegate stores into it from AppKit's notification callbacks. Always
    // false in embedded mode (no NSWindow to go fullscreen).
    fullscreen: std::sync::Arc<std::sync::atomic::AtomicBool>,
    // NSWindowDelegate that tracks the fullscreen transition. None in embedded
    // mode. Retained here because NSWindow holds its delegate as a zeroing weak
    // reference, so dropping this would detach the delegate; the field is never
    // read directly (the delegate communicates through `fullscreen`).
    #[allow(
        dead_code,
        reason = "retained only to keep NSWindow's weak delegate reference alive"
    )]
    window_delegate: Option<Retained<super::window_delegate::WindowDelegate>>,
    // Holds the display to the user's chosen mode while the window is in
    // native fullscreen; restores the desktop mode on exit / drop. Reconciled
    // once per frame by the backend's draw path.
    fullscreen_display: FullscreenDisplayMode,
    keys: KeyState,
    // The runtime movement key map (canonical action -> key). `handle_key`
    // decodes physical events through this instead of hardcoded keys, so a
    // settings-menu rebind takes effect immediately. Defaults to W/S/A/D/Shift/
    // Space/E; GraphicsSystem pushes any persisted override via `set_keymap`.
    keymap: KeyMap,
}

// What the backend hands over at construction, after it has created (or been
// given) the window and the view it renders into.
pub(crate) struct AppKitWindowParts {
    pub window: Option<Retained<NSWindow>>,
    pub view: Retained<NSView>,
    pub title_bar: bool,
    pub pump_events: bool,
    pub fullscreen: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub window_delegate: Option<Retained<super::window_delegate::WindowDelegate>>,
}

// The window-side handles a live world reload transplants onto the rebuilt
// context, so a save reuses the window instead of spawning a new one. The view
// is not carried: the backend owns the concrete subclass and supplies its own.
#[cfg(backend_metal)]
pub(crate) struct WindowHandles {
    pub window: Option<Retained<NSWindow>>,
    pub pump_events: bool,
    pub fullscreen: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub window_delegate: Option<Retained<super::window_delegate::WindowDelegate>>,
}

impl AppKitWindow {
    pub(crate) fn new(parts: AppKitWindowParts) -> Self {
        let AppKitWindowParts {
            window,
            view,
            title_bar,
            pump_events,
            fullscreen,
            window_delegate,
        } = parts;
        Self {
            window,
            view,
            title_bar,
            window_closed: false,
            pump_events,
            cursor_captured: false,
            recapture_on_click: false,
            ui_cursor_hidden: false,
            menu_mode: false,
            fullscreen,
            window_delegate,
            fullscreen_display: FullscreenDisplayMode::new(),
            keys: KeyState::default(),
            keymap: KeyMap::default(),
        }
    }

    // The view the backend renders into, for the presentation resources it owns
    // (Metal's drawable, Vulkan's surface) and for size queries.
    #[cfg(backend_vk)] // Metal reaches its MTKView directly
    pub(crate) fn view(&self) -> &NSView {
        &self.view
    }

    // The engine-created NSWindow, or None in embedded mode.
    pub(crate) fn window(&self) -> Option<&NSWindow> {
        self.window.as_deref()
    }

    // The window / delegate handles a live world reload transplants onto the
    // rebuilt context, so a save reuses the window instead of spawning a new
    // one. The caller supplies the view (it owns the concrete subclass).
    #[cfg(backend_metal)] // Vulkan carries its window through `VkReuse`
    pub(crate) fn handles_for_reuse(&self) -> WindowHandles {
        WindowHandles {
            window: self.window.clone(),
            pump_events: self.pump_events,
            fullscreen: std::sync::Arc::clone(&self.fullscreen),
            window_delegate: self.window_delegate.clone(),
        }
    }

    // Carry over the live state a fresh build resets but a world reload must
    // keep. The keymap is re-pushed by GraphicsSystem immediately, but the
    // fullscreen display-mode hold is not, so a fullscreen editor would lose its
    // mode-restore state. NSCursor's hide count and the CGAssociate coupling are
    // process-global and survive teardown, so the flags tracking them must come
    // across too or a reload leaks a hide and strands the OS cursor.
    #[cfg(backend_metal)] // Vulkan carries its window through `VkReuse`
    pub(crate) fn adopt_live_state(&mut self, prev: &mut AppKitWindow) {
        self.fullscreen_display =
            std::mem::replace(&mut prev.fullscreen_display, FullscreenDisplayMode::new());
        self.keymap = prev.keymap;
        self.ui_cursor_hidden = prev.ui_cursor_hidden;
        self.cursor_captured = prev.cursor_captured;
    }

    // Whether the frame loop should pump NSEvents and honour cursor capture.
    #[cfg(backend_metal)] // the Vulkan adapter always pumps
    pub(crate) fn pump_events(&self) -> bool {
        self.pump_events
    }

    // Whether native fullscreen is active, as tracked by the window delegate.
    pub(crate) fn is_fullscreen(&self) -> bool {
        self.fullscreen.load(std::sync::atomic::Ordering::Relaxed)
    }

    // The overlay coordinate space on macOS: the view's size in points, the same
    // units `cursor_in_content` reports the cursor in. Both macOS backends read
    // it from here, so the drawable's backing scale never leaks into UI space
    // (see `RenderBackend::logical_size`).
    pub(crate) fn logical_size(&self) -> (f32, f32) {
        let s = self.view.bounds().size;
        (s.width as f32, s.height as f32)
    }

    // Whether a window-close event has been seen.
    pub(crate) fn closed(&self) -> bool {
        self.window_closed
    }

    // Hold the display at the chosen fullscreen mode, or restore the desktop
    // mode when fullscreen ends. Driven once per frame by the backend.
    pub(crate) fn reconcile_display_mode(&mut self) {
        let fullscreen = self.is_fullscreen();
        self.fullscreen_display
            .reconcile(self.window.as_deref(), fullscreen);
    }

    // The NSWindow currently hosting the renderer. In windowed mode this is
    // the NSWindow we created; in embedded mode (preview tab, or the
    // play-in-view path where the host owns the window) it is the view's
    // host. Returns None only when the view isn't yet in a window
    // (transient: during init the parent hasn't been set yet).
    fn host_window(&self) -> Option<Retained<NSWindow>> {
        if let Some(ref w) = self.window {
            return Some(w.clone());
        }
        self.view.window()
    }

    // Hide the cursor and begin accumulating relative mouse deltas. No-op
    // for the preview tab (pump_events=false), where the cursor must remain
    // usable for a host UI's tab bar and sidebar controls. Also a no-op when
    // no host window is yet attached.
    pub(crate) fn capture_cursor(&mut self) {
        if !self.pump_events {
            return;
        }
        if self.host_window().is_none() {
            return;
        }
        NSCursor::hide();
        // Decouple cursor position from hardware movement so deltaX/deltaY are
        // pure hardware deltas and the OS cursor stays frozen where the user
        // last left it. release_cursor reads that frozen position back, so the
        // menu cursor reappears there instead of snapping on the first move.
        unsafe { CGAssociateMouseAndMouseCursorPosition(0) };
        // Drop any deltas already accumulated before capture, and arm a
        // one-shot discard so the first motion event pumped after capture
        // (which may have been queued during init, before the OS settled
        // into raw-delta mode) doesn't snap the camera.
        self.keys.mouse_dx = 0.0;
        self.keys.mouse_dy = 0.0;
        self.keys.discard_next_motion = true;
        self.cursor_captured = true;
        self.recapture_on_click = false;
    }

    // Hide or show the OS cursor for an in-engine UI cursor (e.g. a MainMenu),
    // without engaging camera capture. Edge-triggered: NSCursor hide/unhide are
    // ref-counted, so we only toggle on a state change. No-op for the preview
    // tab (pump_events=false), which must keep the system cursor usable.
    pub(crate) fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        if !self.pump_events || hidden == self.ui_cursor_hidden {
            return;
        }
        self.ui_cursor_hidden = hidden;
        if hidden {
            NSCursor::hide();
        } else {
            NSCursor::unhide();
        }
    }

    // Whether the real cursor has left the window so the renderer should stop
    // drawing the in-engine UI cursor. Recomputed each frame by
    // `update_ui_cursor_confinement`.
    pub(crate) fn cursor_outside_window(&self) -> bool {
        self.keys.cursor_outside_window
    }

    // Per-frame bookkeeping for an in-engine UI cursor (a menu): report whether
    // the real cursor has left the window so the renderer can stop drawing the
    // cursor in windowed / borderless modes, and confine the cursor to the
    // active screen while in fullscreen so it cannot stray onto another display.
    // A no-op while the cursor is captured (a gameplay camera owns the pointer)
    // or with no host window (embedded preview).
    fn update_ui_cursor_confinement(&mut self, mtm: objc2::MainThreadMarker) {
        if self.cursor_captured {
            self.keys.cursor_outside_window = false;
            return;
        }
        let Some(window) = self.host_window() else {
            self.keys.cursor_outside_window = false;
            return;
        };
        let Some(screen) = window.screen() else {
            self.keys.cursor_outside_window = false;
            return;
        };
        // Global cursor position (AppKit screen coordinates, origin bottom-left
        // of the primary display, y up).
        let cursor = NSEvent::mouseLocation();
        if self.is_fullscreen() {
            // Confine to the fullscreen display: if the cursor strayed onto
            // another monitor, warp it back just inside the edge. A
            // single-display fullscreen is already confined by the OS, so this
            // never fires there.
            let sf = screen.frame();
            let (min_x, max_x) = (sf.origin.x, sf.origin.x + sf.size.width);
            let (min_y, max_y) = (sf.origin.y, sf.origin.y + sf.size.height);
            // Only warp when actually off the screen, so a cursor resting near
            // the edge is never nudged (an unconditional clamp would warp a
            // valid sub-pixel position in the last row/column every frame).
            let outside =
                cursor.x < min_x || cursor.x >= max_x || cursor.y < min_y || cursor.y >= max_y;
            if outside {
                let cx = cursor.x.clamp(min_x, max_x - 1.0);
                let cy = cursor.y.clamp(min_y, max_y - 1.0);
                // CGWarpMouseCursorPosition takes the global display coordinate
                // space (origin top-left of the primary display, y down); flip Y
                // about the PRIMARY display height (screens[0], the (0,0)-origin
                // screen), which is correct for any monitor arrangement -- not
                // just when the window is on the main display.
                let screens = NSScreen::screens(mtm);
                let primary_h = if screens.count() > 0 {
                    screens.objectAtIndex(0).frame().size.height
                } else {
                    sf.size.height
                };
                let warp = NSPoint::new(cx, primary_h - cy);
                unsafe { CGWarpMouseCursorPosition(warp) };
            }
            self.keys.cursor_outside_window = false;
            return;
        }
        // Windowed / borderless: the in-engine cursor shows only while the real
        // cursor is over the content area.
        let content = window.contentRectForFrameRect(window.frame());
        let inside = cursor.x >= content.origin.x
            && cursor.x < content.origin.x + content.size.width
            && cursor.y >= content.origin.y
            && cursor.y < content.origin.y + content.size.height;
        self.keys.cursor_outside_window = !inside;
    }

    // Show the cursor and stop accumulating mouse deltas.
    pub(crate) fn release_cursor(&mut self) {
        if !self.cursor_captured {
            return;
        }
        self.cursor_captured = false;
        self.recapture_on_click = true;
        unsafe { CGAssociateMouseAndMouseCursorPosition(1) };
        NSCursor::unhide();
        // Seed the tracked UI cursor from the OS cursor's real position (frozen
        // at the pre-capture location while decoupled). Without this the tracked
        // position is stale from before capture, so the first mouse move after a
        // menu opens snaps the in-engine cursor to wherever the OS cursor sits.
        let (mx, my) = self.cursor_in_content();
        self.keys.mouse_x = mx;
        self.keys.mouse_y = my;
    }

    // A togglable menu coexists with a captured camera; see
    // `RenderBackend::set_menu_mode`.
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

    // Switch the engine-created window between windowed / borderless /
    // fullscreen. Only `self.window` is touched: in embedded mode (the preview
    // tab or a host-owned window) this is a no-op so we never restyle a host
    // window. The change flows through the backend's per-frame resize
    // detection, so no render targets are rebuilt here.
    pub(crate) fn set_window_mode(&mut self, mode: WindowMode) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        // Read the fullscreen state from the flag the NSWindowDelegate keeps in
        // sync (it flips at the start of the animation via
        // windowWillEnter/ExitFullScreen). This does not lag the way the
        // style-mask bit does, so stepping the Window Mode row faster than the
        // ~1s native-fullscreen animation no longer toggles the wrong way.
        let is_fullscreen = self.fullscreen.load(std::sync::atomic::Ordering::Relaxed);
        // Record the intended fullscreen state synchronously so a second step
        // issued before the delegate callback lands still decides correctly;
        // the delegate's did-callbacks reconcile this with reality at the end
        // of the transition (and capture OS-driven toggles like the green
        // traffic-light button).
        self.fullscreen.store(
            matches!(mode, WindowMode::Fullscreen),
            std::sync::atomic::Ordering::Relaxed,
        );
        match mode {
            WindowMode::Windowed => {
                if is_fullscreen {
                    window.toggleFullScreen(None);
                }
                window.setStyleMask(windowed_style_mask(self.title_bar));
                apply_title_bar(window, self.title_bar);
            }
            WindowMode::Borderless => {
                if is_fullscreen {
                    window.toggleFullScreen(None);
                }
                // Keep the window key-window-eligible (a pure Borderless,
                // non-panel window cannot become key, which kills keyboard
                // input): a Titled + full-size-content window with a
                // transparent, hidden title bar and hidden traffic-light
                // buttons reads as borderless but still receives key events.
                window.setStyleMask(
                    NSWindowStyleMask::Titled
                        | NSWindowStyleMask::Closable
                        | NSWindowStyleMask::Resizable
                        | NSWindowStyleMask::FullSizeContentView,
                );
                window.setTitlebarAppearsTransparent(true);
                window.setTitleVisibility(NSWindowTitleVisibility::Hidden);
                set_window_buttons_hidden(window, true);
                // Borderless covers the window's current display.
                if let Some(screen) = window.screen() {
                    window.setFrame_display(screen.frame(), true);
                }
            }
            WindowMode::Fullscreen => {
                // Native fullscreen animates from a windowed window and keeps
                // its authored chrome: macOS hides the title bar while
                // fullscreen regardless, and preserving the style here means an
                // OS-driven exit (the green button, which set_window_mode never
                // sees) lands back on the style the world asked for rather than
                // reinstating a title bar it turned off.
                window.setStyleMask(windowed_style_mask(self.title_bar));
                apply_title_bar(window, self.title_bar);
                if !is_fullscreen {
                    window.toggleFullScreen(None);
                }
            }
        }
        // Re-acquire key + front so keyboard input keeps flowing after a restyle.
        window.makeKeyAndOrderFront(None);
    }

    // The display modes (pixel resolution + refresh rate) of the display the
    // engine window sits on. Empty in embedded mode: the host owns the window,
    // so the engine never switches its display and the Resolution row falls
    // back to the windowed presets.
    pub(crate) fn display_modes(&self) -> Vec<DisplayMode> {
        if self.window.is_none() {
            return Vec::new();
        }
        display_mode::enumerate(self.window.as_deref())
    }

    // The mode the engine window's display is currently running (what the
    // Resolution row shows before the user ever picks one). None in embedded
    // mode, matching display_modes.
    pub(crate) fn current_display_mode(&self) -> Option<DisplayMode> {
        let window = self.window.as_deref()?;
        display_mode::current(Some(window))
    }

    // Remember the display mode to hold while the window is in native
    // fullscreen. Applied by the per-frame `reconcile_display_mode` (which also
    // restores the desktop mode on leaving fullscreen), so a choice made in
    // any window mode takes effect when fullscreen is (or becomes) active.
    pub(crate) fn set_display_mode(&mut self, mode: DisplayMode) {
        self.fullscreen_display.set_desired(mode);
    }

    // Resize the engine-created window's content area (windowed mode only).
    // No-op in embedded mode or while in native fullscreen.
    pub(crate) fn set_window_size(&mut self, width: u32, height: u32) {
        // Resizing the content area is meaningless while in native fullscreen;
        // read the delegate-tracked flag (not the lagging style-mask bit).
        if self.is_fullscreen() {
            return;
        }
        let Some(window) = self.window.as_ref() else {
            return;
        };
        window.setContentSize(NSSize::new(width as f64, height as f64));
    }

    // Snapshot the current input state for this frame.
    // Key booleans reflect what is held right now; mouse deltas are cleared
    // after being read so they don't accumulate across frames.
    // `interact` and `jump` are true for exactly one frame per key press then cleared.
    pub(crate) fn take_input(&mut self) -> InputState {
        let snapshot = InputState {
            forward: self.keys.forward,
            backward: self.keys.backward,
            left: self.keys.left,
            right: self.keys.right,
            sprint: self.keys.sprint,
            interact: self.keys.interact_pulse,
            jump: self.keys.jump_pulse,
            mouse_dx: self.keys.mouse_dx,
            mouse_dy: self.keys.mouse_dy,
            scroll_delta: self.keys.scroll_delta,
            mouse_x: self.keys.mouse_x,
            mouse_y: self.keys.mouse_y,
            left_click: self.keys.left_click_pulse,
            // Held state: read but not cleared here (cleared on LeftMouseUp).
            left_button_down: self.keys.left_button_down,
            right_click: self.keys.right_click_pulse,
            hud_toggle: self.keys.hud_toggle_pulse,
            escape: self.keys.escape_pulse,
            ctrl: self.keys.control_down,
            alt: self.keys.alt_down,
            cmd: self.keys.command_down,
            captured_key: self.keys.captured_key,
            typed_char: self.keys.typed_char,
        };
        self.keys.interact_pulse = false;
        self.keys.jump_pulse = false;
        self.keys.mouse_dx = 0.0;
        self.keys.mouse_dy = 0.0;
        self.keys.scroll_delta = 0.0;
        self.keys.left_click_pulse = false;
        self.keys.right_click_pulse = false;
        self.keys.hud_toggle_pulse = false;
        self.keys.escape_pulse = false;
        self.keys.captured_key = None;
        self.keys.typed_char = None;
        snapshot
    }

    // Dequeue all pending NSEvents and update input state. Sets the closed flag
    // on a window-will-close application event. Key events update the persistent
    // key state; mouse moved events accumulate deltas if the cursor is captured.
    pub(crate) fn pump_ns_events(&mut self, mtm: objc2::MainThreadMarker) {
        let ns_app = NSApplication::sharedApplication(mtm);
        loop {
            let event = ns_app.nextEventMatchingMask_untilDate_inMode_dequeue(
                NSEventMask::Any,
                Some(&NSDate::distantPast()),
                objc2_foundation::ns_string!("kCFRunLoopDefaultMode"),
                true,
            );
            let event = match event {
                Some(e) => e,
                None => break,
            };

            match event.r#type() {
                NSEventType::KeyDown => self.handle_key(&event, true),
                NSEventType::KeyUp => self.handle_key(&event, false),
                NSEventType::FlagsChanged => {
                    // Fires immediately when a modifier key is pressed or
                    // released, independent of any other key event. Shift is a
                    // pure modifier on macOS (no KeyDown/KeyUp), so it is decoded
                    // here: drive any action bound to Shift (sprint by default)
                    // and fire the rebind-capture pulse on its rising edge.
                    let flags = event.modifierFlags();
                    let shift = flags.contains(NSEventModifierFlags::Shift);
                    let edge_down = shift && !self.keys.shift_down;
                    self.keys.shift_down = shift;
                    if edge_down {
                        self.keys.captured_key = Some(Key::Shift);
                    }
                    self.apply_binding(Key::Shift, shift, edge_down);
                    // Control is a held modifier too (a story's Ctrl fast-forward
                    // reads it each frame); track it like Shift but drive no
                    // gameplay binding.
                    self.keys.control_down = flags.contains(NSEventModifierFlags::Control);
                    // Option/Alt is tracked the same way; like Control it
                    // drives no gameplay binding.
                    self.keys.alt_down = flags.contains(NSEventModifierFlags::Option);
                    // Command likewise, the modifier macOS shortcuts are built
                    // on.
                    self.keys.command_down = flags.contains(NSEventModifierFlags::Command);
                }
                NSEventType::MouseMoved | NSEventType::LeftMouseDragged => {
                    if self.cursor_captured {
                        // CGAssociateMouseAndMouseCursorPosition(false) is active while captured,
                        // so deltaX/deltaY are pure hardware deltas with no warp
                        // feedback. No per-event warp needed.
                        if self.keys.discard_next_motion {
                            self.keys.discard_next_motion = false;
                        } else {
                            self.keys.mouse_dx += event.deltaX() as f32;
                            self.keys.mouse_dy += event.deltaY() as f32;
                        }
                    } else {
                        // Track the absolute cursor position for UI hit-testing and
                        // the in-engine pointer, in view points with a top-left
                        // origin (see cursor_in_content: sourced from the global
                        // cursor position so a fullscreen menu-bar reveal cannot
                        // fling the pointer off screen).
                        let (mx, my) = self.cursor_in_content();
                        self.keys.mouse_x = mx;
                        self.keys.mouse_y = my;
                    }
                }
                NSEventType::LeftMouseDown => {
                    if !self.cursor_captured {
                        // In menu mode a click fires a UI action; capture is
                        // driven by the active menu, not by clicking.
                        if !self.menu_mode
                            && self.recapture_on_click
                            && self.in_content_area(&event)
                        {
                            self.capture_cursor();
                        } else {
                            self.keys.left_click_pulse = true;
                            self.keys.left_button_down = true;
                        }
                    }
                    ns_app.sendEvent(&event);
                }
                NSEventType::RightMouseDown => {
                    // A right press is only a UI signal (context menus); it never
                    // captures or recaptures the cursor, unlike LeftMouseDown.
                    if !self.cursor_captured {
                        self.keys.right_click_pulse = true;
                    }
                    ns_app.sendEvent(&event);
                }
                NSEventType::LeftMouseUp => {
                    // End any held-button state (drag release). Always cleared,
                    // even if the down began while captured, so the flag can
                    // never stick across a capture transition.
                    self.keys.left_button_down = false;
                    ns_app.sendEvent(&event);
                }
                NSEventType::ScrollWheel => {
                    // Accumulate the wheel delta for scrollable UI while the
                    // cursor is free. scrollingDeltaY is positive when scrolling
                    // up (away from the user); negate so positive moves a panel's
                    // content up (matching FrameInput.scroll_delta's convention).
                    if !self.cursor_captured {
                        self.keys.scroll_delta -= event.scrollingDeltaY() as f32;
                    }
                    ns_app.sendEvent(&event);
                }
                NSEventType::ApplicationDefined => {
                    self.window_closed = true;
                }
                _ => {
                    ns_app.sendEvent(&event);
                }
            }
        }
        // After draining this frame's events, refresh the in-engine cursor's
        // window-exit / fullscreen-confinement state.
        self.update_ui_cursor_confinement(mtm);
    }

    // The live cursor position in view points with a top-left origin, for UI
    // hit-testing and the in-engine pointer. The Y flip is about the live
    // `view.bounds()` height -- the exact view the renderer draws the overlay
    // + cursor against (`logical_size`) -- and the conversion goes through that
    // view (not `window.contentView()`, which in embedded play-in-view mode is
    // the host's content view rather than our subview), so pointer and draw
    // share one reference in every window mode.
    fn cursor_in_content(&self) -> (f32, f32) {
        // Source the pointer from the GLOBAL cursor position, NOT from a mouse
        // event's `locationInWindow`. When macOS auto-reveals the menu bar over a
        // native-fullscreen window it shrinks the window and delivers the move
        // events relative to a transient system window, so `locationInWindow`
        // collapses to a bogus value (measured: cursor pinned at the physical
        // screen top -> loc.y jumps from ~1060 to 64 in a 1084-tall window,
        // flinging the pointer to the bottom). `NSEvent::mouseLocation()` stays
        // correct throughout, so convert THAT through our own window + view.
        let glob = NSEvent::mouseLocation();
        let Some(window) = self.host_window() else {
            return (glob.x as f32, 0.0);
        };
        let win_pt = window.convertPointFromScreen(glob);
        let p = self.view.convertPoint_fromView(win_pt, None);
        let h = self.view.bounds().size.height;
        (p.x as f32, (h - p.y) as f32)
    }

    // Returns true when the event's click position is inside the view's
    // drawable area (below the title bar). Title-bar clicks (traffic lights,
    // drag area) land above the view and return false so they don't trigger
    // cursor recapture. Uses the view's own coordinate system + bounds (the
    // same reference as cursor_in_content) rather than
    // `contentRectForFrameRect(frame)`, which diverges during a fullscreen
    // title-bar reveal.
    fn in_content_area(&self, event: &NSEvent) -> bool {
        let loc = event.locationInWindow();
        let p = self.view.convertPoint_fromView(loc, None);
        p.y >= 0.0 && p.y < self.view.bounds().size.height
    }

    // Replace the runtime movement key map. `handle_key` decodes events through
    // it, so a settings-menu rebind takes effect on the next key event.
    pub(crate) fn set_keymap(&mut self, keymap: &KeyMap) {
        self.keymap = *keymap;
    }

    // Apply a key transition to whichever gameplay actions are bound to `key`.
    // `down` is the held state (movement / sprint follow it); `fire_pulse` fires
    // the one-shot actions (jump / interact). For a keyboard event the press
    // edge is the KeyDown, so both come from `pressed`; for the Shift modifier
    // the pulse fires only on the rising edge (FlagsChanged can re-fire while
    // Shift stays held if another modifier changes).
    fn apply_binding(&mut self, key: Key, down: bool, fire_pulse: bool) {
        let km = self.keymap;
        if km.forward == key {
            self.keys.forward = down;
        }
        if km.backward == key {
            self.keys.backward = down;
        }
        if km.left == key {
            self.keys.left = down;
        }
        if km.right == key {
            self.keys.right = down;
        }
        if km.sprint == key {
            self.keys.sprint = down;
        }
        if fire_pulse {
            if km.jump == key {
                self.keys.jump_pulse = true;
            }
            if km.interact == key {
                self.keys.interact_pulse = true;
            }
        }
    }

    // Update the persistent key state from a key event. Escape and F1 are fixed
    // (not rebindable); every other key is decoded to a canonical `Key` and
    // routed through the runtime key map. Sprint's default (Shift) is a pure
    // modifier and is handled in the FlagsChanged arm, not here.
    fn handle_key(&mut self, event: &NSEvent, pressed: bool) {
        let kc = event.keyCode();
        // Fixed keys.
        match kc {
            53 if pressed => {
                // Escape. In menu mode (a MainMenu over a captured camera) it
                // always pulses so UiInputSystem can toggle the menu and
                // GraphicsSystem drives capture from there. Otherwise: a
                // captured-cursor world releases the cursor (the safe exit), and
                // a free-cursor world pulses for UiInputSystem.
                if self.menu_mode || !self.cursor_captured {
                    self.keys.escape_pulse = true;
                } else {
                    self.release_cursor();
                }
            }
            122 if pressed => self.keys.hud_toggle_pulse = true, // F1: stat HUD.
            _ => {}
        }
        // Read from the event rather than the tracked flag so the decisions
        // below hold whatever order the queue delivered FlagsChanged in.
        let command = event
            .modifierFlags()
            .contains(NSEventModifierFlags::Command);
        // Rebindable keys, decoded through the runtime key map.
        if let Some(key) = key_from_mac(kc) {
            if pressed {
                self.keys.captured_key = Some(key);
            }
            // A Command chord is a shortcut, so its press drives no gameplay
            // action (Cmd+W is not "walk forward"). A release always binds:
            // macOS withholds the key-up of a Command chord, so a key held
            // before Command went down must still be able to let go.
            if !(pressed && command) {
                self.apply_binding(key, pressed, pressed);
            }
        }
        // Printable text input: the OS-resolved glyph for this press (correct
        // casing / shifted symbols / dead keys), for text-input fields. Editing
        // and navigation keys resolve to control glyphs and are filtered out,
        // as is a Command chord, which carries a glyph but means a shortcut.
        if pressed
            && !command
            && let Some(c) = printable_char(event)
        {
            self.keys.typed_char = Some(c);
        }
    }
}

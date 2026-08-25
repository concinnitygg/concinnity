//! Backend-agnostic input snapshot returned by RenderBackend::take_input.
//! Each backend was previously carrying its own structurally-identical
//! InputState; this single type replaces those duplicates.

/// Accumulated input state since the last poll. Drained and reset every
/// frame by GraphicsSystem and converted into a FrameInput component for
/// Camera3DSystem to consume.
#[derive(Default, Debug, Clone, Copy)]
pub struct RenderInput {
    /// Forward movement key held.
    pub forward: bool,
    /// Backward movement key held.
    pub backward: bool,
    /// Left strafe key held.
    pub left: bool,
    /// Right strafe key held.
    pub right: bool,
    /// Sprint key held.
    pub sprint: bool,
    /// True for exactly one frame per interact-key press.
    pub interact: bool,
    /// True for exactly one frame per jump-key press.
    pub jump: bool,
    /// True while the Control key is held. A UI modifier (a story fast-forwards
    /// its dialogue while it is down); not gated by menu state, like `escape` and
    /// `captured_key`. Wired on Metal; DirectX / Vulkan set it from their key
    /// callbacks.
    pub ctrl: bool,
    /// True while the Option/Alt key is held. A UI modifier (the editor's orbit
    /// drag); not gated by menu state, like `ctrl`. Wired on Metal; DirectX /
    /// Vulkan set it from their key callbacks.
    pub alt: bool,
    /// True while the platform's command modifier is held: the Command key on
    /// macOS, where it is the idiomatic modifier for an application shortcut.
    /// Windows and Linux leave this false and keep Ctrl as their shortcut
    /// modifier, because the Super key there belongs to the desktop shell.
    pub cmd: bool,
    /// Accumulated mouse delta since the last take_input() call.
    pub mouse_dx: f32,
    /// Accumulated vertical mouse delta since the last take_input().
    pub mouse_dy: f32,
    /// Accumulated vertical scroll-wheel delta since the last take_input().
    /// Only delivered while the cursor is free.
    pub scroll_delta: f32,
    /// Absolute cursor position in window pixels (origin top-left).
    /// Only meaningful when the cursor is not captured.
    pub mouse_x: f32,
    /// Absolute cursor y in window pixels, origin top-left.
    pub mouse_y: f32,
    /// True for exactly one frame when the left mouse button is pressed
    /// while the cursor is not captured.
    pub left_click: bool,
    /// True while the left mouse button is held (cursor not captured). Persists
    /// across frames until release so a UI drag can track the cursor.
    pub left_button_down: bool,
    /// True for exactly one frame when the right mouse button is pressed
    /// while the cursor is not captured. Wired on Metal; DirectX / Vulkan set
    /// it from their mouse callbacks.
    pub right_click: bool,
    /// True for exactly one frame when the HUD-toggle key is pressed (F1).
    pub hud_toggle: bool,
    /// True for exactly one frame when Escape is pressed while the cursor is
    /// not captured. (In captured-cursor worlds Escape continues to release
    /// the cursor, as before, and this pulse stays false.)
    pub escape: bool,
    /// The canonical key pressed this poll, for the settings-menu rebind
    /// capture, or `None`. A one-frame pulse, surfaced regardless of menu /
    /// capture state. Wired on Metal; DirectX / Vulkan set it from their key
    /// callbacks.
    pub captured_key: Option<crate::components::InputKey>,
    /// The printable character produced by this poll's key press (with the OS's
    /// shift / dead-key / layout handling applied), for text-input fields, or
    /// `None`. A one-frame pulse like `captured_key`, ungated by menu / capture
    /// state. Editing keys (Backspace / Delete / arrows) are not here: those
    /// arrive via `captured_key`. Wired on Metal; DirectX / Vulkan set it from
    /// their WM_CHAR / char callback when built on Windows / Linux.
    pub typed_char: Option<char>,
}

/// One frame's sampled window input, taken beside the backend right after the
/// draw (whose event pump produced it) and consumed by the input system. In
/// serial execution it is deposited and consumed within the same tick; the
/// pipelined driver ships it across the thread boundary instead.
#[derive(Default, Debug, Clone, Copy)]
pub struct InputPacket {
    /// The raw sampled input state.
    pub raw: RenderInput,
    /// Whether the cursor has left the window.
    pub cursor_outside_window: bool,
    /// Logical window size, for UI hit-testing and overlay layout.
    pub viewport: (f32, f32),
}

impl InputPacket {
    /// Sample the backend's accumulated input, cursor containment, and
    /// logical size into one packet. Called right after the draw so the
    /// frame's event pump is reflected.
    pub fn sample(backend: &mut dyn crate::backend::RenderBackend) -> Self {
        Self {
            raw: backend.take_input(),
            cursor_outside_window: backend.cursor_outside_window(),
            viewport: backend.logical_size(),
        }
    }

    /// Fold a newer packet onto this one without losing edges: one-frame
    /// pulses OR together, deltas accumulate, positions and held states take
    /// the newer value. Only needed when a consumer misses a frame (startup,
    /// a stall); steady state is one packet per tick.
    pub fn merge_from(&mut self, newer: InputPacket) {
        let old = self.raw;
        let mut raw = newer.raw;
        raw.interact |= old.interact;
        raw.jump |= old.jump;
        raw.left_click |= old.left_click;
        raw.right_click |= old.right_click;
        raw.hud_toggle |= old.hud_toggle;
        raw.escape |= old.escape;
        raw.mouse_dx += old.mouse_dx;
        raw.mouse_dy += old.mouse_dy;
        raw.scroll_delta += old.scroll_delta;
        raw.captured_key = raw.captured_key.or(old.captured_key);
        raw.typed_char = raw.typed_char.or(old.typed_char);
        self.raw = raw;
        self.cursor_outside_window = newer.cursor_outside_window;
        self.viewport = newer.viewport;
    }
}

// Scroll units emitted per physical wheel notch on backends whose wheel events
// arrive as discrete notches (DirectX WM_MOUSEWHEEL, GLFW Scroll). macOS reports
// precise (often large) scroll deltas directly, so Metal feeds scrollingDeltaY
// raw and does not use this. The shared UI multiplies scroll_delta by its own
// WHEEL_SCROLL_SPEED (see ui.rs), so this is scroll-delta units per notch.
// Consumed by DirectX (WM_MOUSEWHEEL) and Vulkan (GLFW Scroll); dead on a Metal
// build, which feeds scrollingDeltaY raw.
pub(crate) const WHEEL_NOTCH_SCROLL_UNITS: f32 = 20.0;

/// Convert a signed wheel rotation in notches (positive = rotated away from the
/// user, i.e. scroll up) into an additive scroll_delta increment. Negated so a
/// positive scroll_delta scrolls a panel's content up, matching
/// FrameInput.scroll_delta's convention (see ui.rs and metal/input.rs).
pub fn wheel_notches_to_scroll_delta(notches: f32) -> f32 {
    -notches * WHEEL_NOTCH_SCROLL_UNITS
}

#[cfg(test)]
mod tests {
    use super::*;

    // A missed consume must not lose edges: pulses OR, deltas accumulate,
    // positions and held state take the newer value, and an earlier one-frame
    // Option pulse survives a newer empty poll.
    #[test]
    fn packet_merge_keeps_pulses_and_accumulates_deltas() {
        let mut pending = InputPacket {
            raw: RenderInput {
                jump: true,
                left_click: true,
                mouse_dx: 2.0,
                scroll_delta: 1.0,
                mouse_x: 10.0,
                left_button_down: true,
                typed_char: Some('a'),
                ..Default::default()
            },
            cursor_outside_window: true,
            viewport: (100.0, 100.0),
        };
        pending.merge_from(InputPacket {
            raw: RenderInput {
                interact: true,
                mouse_dx: 3.0,
                scroll_delta: -0.5,
                mouse_x: 42.0,
                left_button_down: false,
                ..Default::default()
            },
            cursor_outside_window: false,
            viewport: (200.0, 150.0),
        });
        assert!(pending.raw.jump, "the earlier pulse survives");
        assert!(pending.raw.left_click);
        assert!(pending.raw.interact, "the newer pulse is present");
        assert_eq!(pending.raw.mouse_dx, 5.0, "deltas accumulate");
        assert_eq!(pending.raw.scroll_delta, 0.5);
        assert_eq!(pending.raw.mouse_x, 42.0, "position takes the newer value");
        assert!(
            !pending.raw.left_button_down,
            "held state takes the newer value"
        );
        assert_eq!(
            pending.raw.typed_char,
            Some('a'),
            "the typed pulse survives"
        );
        assert!(!pending.cursor_outside_window);
        assert_eq!(pending.viewport, (200.0, 150.0));
    }

    #[test]
    fn wheel_notch_sign_and_scale() {
        // Rotating the wheel away from the user (positive notches, "scroll up")
        // yields a negative scroll_delta so a panel's content moves down,
        // revealing the top.
        assert!(wheel_notches_to_scroll_delta(1.0) < 0.0);
        // Rotating toward the user ("scroll down") yields a positive
        // scroll_delta so the content moves up, revealing lower rows.
        assert!(wheel_notches_to_scroll_delta(-1.0) > 0.0);
        // The increment scales linearly with the number of notches.
        assert_eq!(
            wheel_notches_to_scroll_delta(-2.0),
            2.0 * wheel_notches_to_scroll_delta(-1.0)
        );
    }
}

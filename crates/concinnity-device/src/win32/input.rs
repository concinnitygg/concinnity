// Keyboard input state for the shared Win32 window layer (DirectX + the
// Vulkan Windows window). Win32 keyboard/mouse events are processed in the
// wnd_proc (window.rs); this struct is the snapshot consumed by
// GraphicsSystem each tick.

use crate::assets::Key;
use crate::gfx::keymap::KeyMap;
use windows::Win32::UI::Input::KeyboardAndMouse::*;

// The previously-duplicated InputState collapsed into the shared
// crate::gfx::input::RenderInput; this alias keeps the historical name.
pub use crate::gfx::input::RenderInput as InputState;

// One frame's accumulated mouse input, owned by `WindowState` and handed to
// [`KeyState::take`] so the drained snapshot carries pointer motion, position,
// button, and scroll state alongside the keyboard one-shots.
#[derive(Clone, Copy)]
pub(crate) struct MouseSnapshot {
    pub dx: f32,
    pub dy: f32,
    pub x: f32,
    pub y: f32,
    pub left_click: bool,
    pub left_button_down: bool,
    pub right_click: bool,
    pub scroll_delta: f32,
}

// Per-key pressed state tracked across Win32 WM_KEYDOWN / WM_KEYUP messages.
#[derive(Default)]
pub(crate) struct KeyState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub sprint: bool,
    // Held Control modifier (a story's Ctrl fast-forward reads it each frame).
    // Windows delivers Ctrl as an ordinary WM_KEYDOWN/WM_KEYUP carrying
    // VK_CONTROL, so it is tracked here like a held key rather than through a
    // separate modifier path (unlike macOS FlagsChanged); it drives no gameplay
    // binding. Not a one-shot, so `take` reads without resetting it.
    pub ctrl: bool,
    // Held Alt modifier (the editor's Alt+drag orbit reads it each frame).
    // Windows routes Alt through WM_SYSKEYDOWN / WM_SYSKEYUP carrying VK_MENU
    // rather than the ordinary key messages, so it is tracked from those (see
    // `on_sys_key`) and cleared on focus loss so Alt+Tab does not leave it
    // stuck down. Not a one-shot, so `take` reads without resetting it.
    pub alt: bool,
    // One-shot flags: set on down, cleared after take_input() reads them.
    pub interact_pending: bool,
    pub jump_pending: bool,
    // One-shot: set on F1-down, cleared by `take`. Drives the `StatHud`
    // system's F1 toggle so the in-engine profiler overlay can be flipped
    // at runtime.
    pub hud_toggle_pending: bool,
    // One-shot: set on Escape-down when the cursor is *not* captured.
    // (When the cursor is captured the wnd_proc routes Escape through
    // `do_release_cursor` instead, matching the Metal backend.)
    pub escape_pending: bool,
    // One-shot: the canonical key pressed since the last `take`, for the
    // settings-menu rebind capture. Set on any mapped key-down; reset by `take`.
    pub captured_key: Option<Key>,
    // One-shot: the printable glyph typed since the last `take`, for text-input
    // fields (the editor's name/filter/arg fields). Filled from WM_CHAR, which
    // Windows resolves for the active layout, Shift, and dead keys, so casing
    // and shifted symbols are already correct. One codepoint per frame (fast
    // typing / IME multi-codepoint frames drop extras, matching `captured_key`);
    // reset by `take`. Editing / navigation keys produce no WM_CHAR and instead
    // ride `captured_key` (Backspace / Delete / Left / Right in `key_from_vk`).
    pub typed_char: Option<char>,
    // The runtime movement key map. `on_key_down` / `on_key_up` decode events
    // through it instead of hardcoded keys, so a settings-menu rebind takes
    // effect immediately. Defaults to W/S/A/D/Shift/Space/E. (Windows delivers
    // Shift as an ordinary WM_KEYDOWN, so it is just another key here -- no
    // separate modifier path is needed, unlike macOS.)
    pub keymap: KeyMap,
}

impl KeyState {
    // Replace the runtime movement key map.
    pub(crate) fn set_keymap(&mut self, keymap: &KeyMap) {
        self.keymap = *keymap;
    }

    // Apply a key transition to whichever gameplay actions are bound to `key`.
    // `down` is the held state (movement / sprint follow it); `fire_pulse`
    // fires the one-shot actions (jump / interact) on a press.
    fn apply_binding(&mut self, key: Key, down: bool, fire_pulse: bool) {
        let km = self.keymap;
        if km.forward == key {
            self.forward = down;
        }
        if km.backward == key {
            self.backward = down;
        }
        if km.left == key {
            self.left = down;
        }
        if km.right == key {
            self.right = down;
        }
        if km.sprint == key {
            self.sprint = down;
        }
        if fire_pulse {
            if km.jump == key {
                self.jump_pending = true;
            }
            if km.interact == key {
                self.interact_pending = true;
            }
        }
    }

    // Update held/pending flags from a WM_KEYDOWN message. F1 stays fixed (the
    // stat-HUD toggle); every other key routes through the key map.
    pub(crate) fn on_key_down(&mut self, vk: VIRTUAL_KEY) {
        if vk == VK_F1 {
            self.hud_toggle_pending = true;
        }
        if vk == VK_CONTROL {
            self.ctrl = true;
        }
        if let Some(key) = key_from_vk(vk) {
            self.captured_key = Some(key);
            self.apply_binding(key, true, true);
        }
    }

    // Note an Escape press while the cursor is *not* captured. The wnd_proc
    // keeps swallowing Escape into `do_release_cursor` while captured, so
    // this is only called for the "menu / UI" case; mirrors the Metal
    // `escape_pulse` rule.
    pub(crate) fn on_escape_uncaptured(&mut self) {
        self.escape_pending = true;
    }

    // Track the held Alt modifier from a WM_SYSKEYDOWN / WM_SYSKEYUP message.
    // Only VK_MENU is read: the other system-key presses (Alt+F4, Alt+Enter)
    // stay with DefWindowProc.
    pub(crate) fn on_sys_key(&mut self, vk: VIRTUAL_KEY, down: bool) {
        if vk == VK_MENU {
            self.alt = down;
        }
    }

    // Drop the held modifiers when the window loses focus. Alt+Tab consumes the
    // Alt release, so without this the flag would stay set for the rest of the
    // session.
    pub(crate) fn on_focus_lost(&mut self) {
        self.ctrl = false;
        self.alt = false;
    }

    // Update held flags from a WM_KEYUP message.
    pub(crate) fn on_key_up(&mut self, vk: VIRTUAL_KEY) {
        if vk == VK_CONTROL {
            self.ctrl = false;
        }
        if let Some(key) = key_from_vk(vk) {
            self.apply_binding(key, false, false);
        }
    }

    // Record a printable glyph from a WM_CHAR message. `TranslateMessage`
    // synthesises WM_CHAR from WM_KEYDOWN after layout / modifier resolution, so
    // `c` is the final text character. Control characters (Backspace, Tab,
    // Enter, Escape, delete) are filtered out -- those editing / navigation keys
    // ride `captured_key` via `key_from_vk` instead.
    pub(crate) fn on_char(&mut self, c: char) {
        if is_printable_glyph(c) {
            self.typed_char = Some(c);
        }
    }

    // Drain into an InputState snapshot, resetting one-shot flags. The mouse
    // fields (deltas, position, click, held-button, scroll) are owned by
    // `WindowState` and passed in; the keyboard one-shots tracked here are reset.
    pub(crate) fn take(&mut self, mouse: MouseSnapshot) -> InputState {
        let MouseSnapshot {
            dx: mouse_dx,
            dy: mouse_dy,
            x: mouse_x,
            y: mouse_y,
            left_click,
            left_button_down,
            right_click,
            scroll_delta,
        } = mouse;
        let s = InputState {
            forward: self.forward,
            backward: self.backward,
            left: self.left,
            right: self.right,
            sprint: self.sprint,
            interact: self.interact_pending,
            jump: self.jump_pending,
            mouse_dx,
            mouse_dy,
            scroll_delta,
            mouse_x,
            mouse_y,
            left_click,
            left_button_down,
            right_click,
            hud_toggle: self.hud_toggle_pending,
            escape: self.escape_pending,
            // Held Control modifier (VK_CONTROL), tracked across key-down/up; a
            // story's Ctrl fast-forward reads it. Held state, not a one-shot, so
            // it is not reset below.
            ctrl: self.ctrl,
            // Held Alt modifier (VK_MENU via the system-key messages), tracked
            // across down/up; the editor's orbit drag reads it. Held state, not
            // a one-shot, so it is not reset below.
            alt: self.alt,
            // Windows shortcuts are built on Ctrl, and the Windows key belongs
            // to the shell (Win+K opens a system panel), so nothing here claims
            // it.
            cmd: false,
            captured_key: self.captured_key,
            // Printable text input from WM_CHAR (text-input fields read it). A
            // one-shot like `captured_key`, reset below.
            typed_char: self.typed_char,
        };
        self.interact_pending = false;
        self.jump_pending = false;
        self.hud_toggle_pending = false;
        self.escape_pending = false;
        self.captured_key = None;
        self.typed_char = None;
        s
    }
}

// Translate a WM_KEYDOWN/WM_KEYUP wParam into a VIRTUAL_KEY.
pub(crate) fn vk_from_wparam(wparam: usize) -> VIRTUAL_KEY {
    VIRTUAL_KEY(wparam as u16)
}

// Whether a character from WM_CHAR is a printable glyph suitable for a text
// field, i.e. not a control character. WM_CHAR delivers actual text, so unlike
// macOS there is no private-use function-key range to exclude; the editing /
// navigation keys arrive as their control codes (0x08 Backspace, 0x1B Escape,
// etc.) and are rejected here so they route through `captured_key` instead.
fn is_printable_glyph(c: char) -> bool {
    !c.is_control()
}

// Map a Win32 virtual key to a canonical `Key`, or `None` for a key the engine
// does not bind (function keys, Escape, Ctrl/Alt, etc.). Shift is mapped: unlike
// macOS, Windows delivers it as an ordinary key-down.
fn key_from_vk(vk: VIRTUAL_KEY) -> Option<Key> {
    Some(match vk {
        VK_A => Key::A,
        VK_B => Key::B,
        VK_C => Key::C,
        VK_D => Key::D,
        VK_E => Key::E,
        VK_F => Key::F,
        VK_G => Key::G,
        VK_H => Key::H,
        VK_I => Key::I,
        VK_J => Key::J,
        VK_K => Key::K,
        VK_L => Key::L,
        VK_M => Key::M,
        VK_N => Key::N,
        VK_O => Key::O,
        VK_P => Key::P,
        VK_Q => Key::Q,
        VK_R => Key::R,
        VK_S => Key::S,
        VK_T => Key::T,
        VK_U => Key::U,
        VK_V => Key::V,
        VK_W => Key::W,
        VK_X => Key::X,
        VK_Y => Key::Y,
        VK_Z => Key::Z,
        VK_0 => Key::Num0,
        VK_1 => Key::Num1,
        VK_2 => Key::Num2,
        VK_3 => Key::Num3,
        VK_4 => Key::Num4,
        VK_5 => Key::Num5,
        VK_6 => Key::Num6,
        VK_7 => Key::Num7,
        VK_8 => Key::Num8,
        VK_9 => Key::Num9,
        VK_SPACE => Key::Space,
        VK_TAB => Key::Tab,
        VK_RETURN => Key::Enter,
        VK_BACK => Key::Backspace,
        VK_DELETE => Key::Delete,
        VK_SHIFT => Key::Shift,
        VK_LEFT => Key::Left,
        VK_RIGHT => Key::Right,
        VK_UP => Key::Up,
        VK_DOWN => Key::Down,
        VK_OEM_MINUS => Key::Minus,
        VK_OEM_PLUS => Key::Equals,
        VK_OEM_4 => Key::LeftBracket,
        VK_OEM_6 => Key::RightBracket,
        VK_OEM_5 => Key::Backslash,
        VK_OEM_1 => Key::Semicolon,
        VK_OEM_7 => Key::Quote,
        VK_OEM_COMMA => Key::Comma,
        VK_OEM_PERIOD => Key::Period,
        VK_OEM_2 => Key::Slash,
        VK_OEM_3 => Key::Backtick,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(ks: &mut KeyState) -> InputState {
        ks.take(MouseSnapshot {
            dx: 0.0,
            dy: 0.0,
            x: 0.0,
            y: 0.0,
            left_click: false,
            left_button_down: false,
            right_click: false,
            scroll_delta: 0.0,
        })
    }

    #[test]
    fn control_key_tracks_held_ctrl_modifier() {
        // Regression: `ctrl` used to be hardcoded `false` in `take`, so a story's
        // Ctrl fast-forward never fired on Windows (worked on Metal only).
        let mut ks = KeyState::default();
        assert!(!snapshot(&mut ks).ctrl, "ctrl starts released");
        ks.on_key_down(VK_CONTROL);
        assert!(snapshot(&mut ks).ctrl, "ctrl held after VK_CONTROL down");
        // Held modifier, not a one-shot: still set on a later frame with no new
        // event (so a sustained hold keeps fast-forwarding).
        assert!(snapshot(&mut ks).ctrl, "ctrl stays held across frames");
        ks.on_key_up(VK_CONTROL);
        assert!(!snapshot(&mut ks).ctrl, "ctrl released after VK_CONTROL up");
    }

    #[test]
    fn typed_char_carries_a_printable_glyph_once() {
        // Regression: `typed_char` used to be hardcoded `None` in `take`, so
        // text-input fields (the editor's name / filter / arg fields) could not
        // be typed into on Windows (worked on Metal only).
        let mut ks = KeyState::default();
        assert_eq!(snapshot(&mut ks).typed_char, None, "starts empty");
        ks.on_char('A');
        assert_eq!(
            snapshot(&mut ks).typed_char,
            Some('A'),
            "printable glyph surfaces in the snapshot"
        );
        // One-shot: cleared by `take`, so a held key does not re-insert on a
        // frame with no new WM_CHAR.
        assert_eq!(snapshot(&mut ks).typed_char, None, "cleared after take");
    }

    #[test]
    fn control_chars_do_not_become_typed_chars() {
        // Backspace / Enter / Escape / Tab arrive as WM_CHAR control codes; they
        // must be filtered so they route through `captured_key` (editing keys),
        // not inserted as text.
        let mut ks = KeyState::default();
        for c in ['\u{08}', '\r', '\n', '\u{1b}', '\t', '\u{7f}'] {
            ks.on_char(c);
        }
        assert_eq!(
            snapshot(&mut ks).typed_char,
            None,
            "control chars are not printable glyphs"
        );
    }

    #[test]
    fn is_printable_glyph_accepts_text_rejects_controls() {
        assert!(is_printable_glyph('a'));
        assert!(is_printable_glyph('Z'));
        assert!(is_printable_glyph('9'));
        assert!(is_printable_glyph(' '));
        assert!(is_printable_glyph('é'));
        assert!(!is_printable_glyph('\u{08}')); // Backspace
        assert!(!is_printable_glyph('\u{7f}')); // Delete
        assert!(!is_printable_glyph('\n'));
    }

    #[test]
    fn editing_keys_decode_for_captured_key() {
        // Backspace and forward-delete decode so text fields can edit; they ride
        // `captured_key`, not `typed_char` (mirrors metal/input.rs).
        assert_eq!(key_from_vk(VK_BACK), Some(Key::Backspace));
        assert_eq!(key_from_vk(VK_DELETE), Some(Key::Delete));
        assert_eq!(key_from_vk(VK_LEFT), Some(Key::Left));
        assert_eq!(key_from_vk(VK_RIGHT), Some(Key::Right));
        // A key-down surfaces the editing key on `captured_key`.
        let mut ks = KeyState::default();
        ks.on_key_down(VK_BACK);
        assert_eq!(snapshot(&mut ks).captured_key, Some(Key::Backspace));
    }
}

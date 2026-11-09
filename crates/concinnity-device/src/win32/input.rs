// Keyboard input state for the shared Win32 window layer (DirectX + the
// Vulkan Windows window). Win32 keyboard/mouse events are processed in the
// wnd_proc (window.rs); this struct is the snapshot consumed by
// GraphicsSystem each tick.

use concinnity_engine::components::{InputKey, KeyEvent, KeyMods, KeyPress};
use concinnity_engine::input::held_keys::HeldKeys;
use concinnity_engine::input::snapshot::InputSnapshot;
use windows::Win32::UI::Input::KeyboardAndMouse::*;

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
    // The keys held down, from WM_KEYDOWN to WM_KEYUP. Control is held from
    // its own messages, and Alt from the system-key ones (see `on_sys_key`);
    // both and Shift are dropped on focus loss so Alt+Tab leaves none stuck.
    pub held: HeldKeys,
    // One-shot: set on F1-down, cleared by `take`. Drives the `StatHud`
    // system's F1 toggle so the in-engine profiler overlay can be flipped
    // at runtime.
    pub hud_toggle_pending: bool,
    // One-shot: set on Escape-down when the cursor is *not* captured.
    // (When the cursor is captured the wnd_proc routes Escape through
    // `release_cursor` instead, matching the Metal backend.)
    pub escape_pending: bool,
    // Every key press (auto-repeats included) and printable glyph since the
    // last `take`, in arrival order; drained by `take`. Glyphs come from
    // WM_CHAR, which Windows resolves for the active layout, Shift, and dead
    // keys; editing / navigation keys produce no WM_CHAR and travel as presses.
    pub events: Vec<KeyEvent>,
    // The high half of a surrogate pair WM_CHAR delivered, waiting for its low
    // half so a character outside the Basic Multilingual Plane types whole.
    pub high_surrogate: Option<u16>,
}

impl KeyState {
    // The modifiers held right now, recorded on each queued press.
    fn mods(&self) -> KeyMods {
        KeyMods {
            shift: self.held.contains(InputKey::Shift),
            ctrl: self.held.contains(InputKey::Control),
            alt: self.held.contains(InputKey::Alt),
            cmd: false,
        }
    }

    // Update held/pending flags from a WM_KEYDOWN message. `repeat` is set for
    // the auto-repeat of a key already held (lParam bit 30). F1 stays fixed
    // (the stat-HUD toggle); every other key is held until its WM_KEYUP.
    pub(crate) fn on_key_down(&mut self, vk: VIRTUAL_KEY, repeat: bool) {
        if vk == VK_F1 {
            self.hud_toggle_pending = true;
        }
        if let Some(key) = key_from_vk(vk) {
            self.held.set(key, true);
            self.events.push(KeyEvent::Press(KeyPress {
                key,
                mods: self.mods(),
                repeat,
            }));
        }
    }

    // Note an Escape press while the cursor is *not* captured. The wnd_proc
    // keeps swallowing Escape into `release_cursor` while captured, so
    // this is only called for the "menu / UI" case; mirrors the Metal
    // `escape_pulse` rule.
    pub(crate) fn on_escape_uncaptured(&mut self) {
        self.escape_pending = true;
    }

    // Track the held Alt modifier from a WM_SYSKEYDOWN / WM_SYSKEYUP message.
    // Only VK_MENU is read: the other system-key presses (Alt+F4, Alt+Enter)
    // stay with DefWindowProc.
    pub(crate) fn on_sys_key(&mut self, vk: VIRTUAL_KEY, down: bool) {
        if vk != VK_MENU {
            return;
        }
        let fresh = down && !self.held.contains(InputKey::Alt);
        self.held.set(InputKey::Alt, down);
        if fresh {
            self.events.push(KeyEvent::Press(KeyPress {
                key: InputKey::Alt,
                mods: self.mods(),
                repeat: false,
            }));
        }
    }

    // Drop the held modifiers when the window loses focus. Alt+Tab consumes the
    // Alt release, so without this the flag would stay set for the rest of the
    // session.
    pub(crate) fn on_focus_lost(&mut self) {
        for key in [InputKey::Control, InputKey::Alt, InputKey::Shift] {
            self.held.set(key, false);
        }
    }

    // Update held flags from a WM_KEYUP message.
    pub(crate) fn on_key_up(&mut self, vk: VIRTUAL_KEY) {
        if let Some(key) = key_from_vk(vk) {
            self.held.set(key, false);
        }
    }

    // Record one UTF-16 code unit from a WM_CHAR message. `TranslateMessage`
    // synthesizes WM_CHAR from WM_KEYDOWN after layout / modifier resolution, so
    // the units are the final text; a surrogate pair arrives as two messages and
    // is joined here. Control characters (Backspace, Tab, Enter, Escape,
    // delete) are filtered out: those keys travel as presses instead.
    pub(crate) fn on_char(&mut self, unit: u16) {
        let pair = match self.high_surrogate.take() {
            Some(high) => [high, unit],
            None if (0xD800..0xDC00).contains(&unit) => {
                self.high_surrogate = Some(unit);
                return;
            }
            None => [unit, 0],
        };
        let len = if pair[1] == 0 { 1 } else { 2 };
        for c in char::decode_utf16(pair[..len].iter().copied()).flatten() {
            if is_printable_glyph(c) {
                self.events.push(KeyEvent::Text(c));
            }
        }
    }

    // Drain into an InputSnapshot, resetting one-shot flags. The mouse
    // fields (deltas, position, click, held-button, scroll) are owned by
    // `WindowState` and passed in; the keyboard one-shots tracked here are reset.
    pub(crate) fn take(&mut self, mouse: MouseSnapshot) -> InputSnapshot {
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
        let s = InputSnapshot {
            held: self.held,
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
            // Windows shortcuts are built on Ctrl, and the Windows key belongs
            // to the shell (Win+K opens a system panel), so nothing here claims
            // it.
            cmd: false,
            key_events: std::mem::take(&mut self.events),
        };
        self.hud_toggle_pending = false;
        self.escape_pending = false;
        s
    }
}

// The low 16 bits of a message parameter, as Win32's `LOWORD`.
pub(crate) fn lo_word(param: usize) -> u16 {
    let [a, b, ..] = param.to_le_bytes();
    u16::from_le_bytes([a, b])
}

// Bits 16..32 of a message parameter, as Win32's `HIWORD`.
pub(crate) fn hi_word(param: usize) -> u16 {
    let [_, _, a, b, ..] = param.to_le_bytes();
    u16::from_le_bytes([a, b])
}

// Translate a WM_KEYDOWN/WM_KEYUP wParam into a VIRTUAL_KEY.
pub(crate) fn vk_from_wparam(wparam: usize) -> VIRTUAL_KEY {
    VIRTUAL_KEY(lo_word(wparam))
}

// Whether a character from WM_CHAR is a printable glyph suitable for a text
// field, i.e. not a control character. WM_CHAR delivers actual text, so unlike
// macOS there is no private-use function-key range to exclude; the editing /
// navigation keys arrive as their control codes (0x08 Backspace, 0x1B Escape,
// etc.) and are rejected here so they travel as presses instead.
fn is_printable_glyph(c: char) -> bool {
    !c.is_control()
}

// Map a Win32 virtual key to a canonical `InputKey`, or `None` for a key the engine
// does not bind (function keys, Escape, Ctrl/Alt, etc.). Shift is mapped: unlike
// macOS, Windows delivers it as an ordinary key-down.
fn key_from_vk(vk: VIRTUAL_KEY) -> Option<InputKey> {
    Some(match vk {
        VK_A => InputKey::A,
        VK_B => InputKey::B,
        VK_C => InputKey::C,
        VK_D => InputKey::D,
        VK_E => InputKey::E,
        VK_F => InputKey::F,
        VK_G => InputKey::G,
        VK_H => InputKey::H,
        VK_I => InputKey::I,
        VK_J => InputKey::J,
        VK_K => InputKey::K,
        VK_L => InputKey::L,
        VK_M => InputKey::M,
        VK_N => InputKey::N,
        VK_O => InputKey::O,
        VK_P => InputKey::P,
        VK_Q => InputKey::Q,
        VK_R => InputKey::R,
        VK_S => InputKey::S,
        VK_T => InputKey::T,
        VK_U => InputKey::U,
        VK_V => InputKey::V,
        VK_W => InputKey::W,
        VK_X => InputKey::X,
        VK_Y => InputKey::Y,
        VK_Z => InputKey::Z,
        VK_0 => InputKey::Num0,
        VK_1 => InputKey::Num1,
        VK_2 => InputKey::Num2,
        VK_3 => InputKey::Num3,
        VK_4 => InputKey::Num4,
        VK_5 => InputKey::Num5,
        VK_6 => InputKey::Num6,
        VK_7 => InputKey::Num7,
        VK_8 => InputKey::Num8,
        VK_9 => InputKey::Num9,
        VK_SPACE => InputKey::Space,
        VK_TAB => InputKey::Tab,
        VK_RETURN => InputKey::Enter,
        VK_BACK => InputKey::Backspace,
        VK_DELETE => InputKey::Delete,
        VK_SHIFT => InputKey::Shift,
        VK_CONTROL => InputKey::Control,
        VK_LEFT => InputKey::Left,
        VK_RIGHT => InputKey::Right,
        VK_UP => InputKey::Up,
        VK_DOWN => InputKey::Down,
        VK_HOME => InputKey::Home,
        VK_END => InputKey::End,
        VK_PRIOR => InputKey::PageUp,
        VK_NEXT => InputKey::PageDown,
        VK_OEM_MINUS => InputKey::Minus,
        VK_OEM_PLUS => InputKey::Equals,
        VK_OEM_4 => InputKey::LeftBracket,
        VK_OEM_6 => InputKey::RightBracket,
        VK_OEM_5 => InputKey::Backslash,
        VK_OEM_1 => InputKey::Semicolon,
        VK_OEM_7 => InputKey::Quote,
        VK_OEM_COMMA => InputKey::Comma,
        VK_OEM_PERIOD => InputKey::Period,
        VK_OEM_2 => InputKey::Slash,
        VK_OEM_3 => InputKey::Backtick,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_split_a_message_parameter_like_loword_and_hiword() {
        let param = 0xDEAD_FFFE_usize;
        assert_eq!(lo_word(param), 0xFFFE);
        assert_eq!(hi_word(param), 0xDEAD);
        // A signed mouse coordinate survives the round trip through the word.
        assert_eq!(lo_word((-2_isize).cast_unsigned()).cast_signed(), -2);
        assert_eq!(hi_word(0x0078_0000), 120);
    }

    fn snapshot(ks: &mut KeyState) -> InputSnapshot {
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
        // Control is held from its own key messages, so a story's Ctrl
        // fast-forward works on Windows as it does elsewhere.
        let mut ks = KeyState::default();
        assert!(
            !snapshot(&mut ks).held.contains(InputKey::Control),
            "ctrl starts released"
        );
        ks.on_key_down(VK_CONTROL, false);
        assert!(
            snapshot(&mut ks).held.contains(InputKey::Control),
            "ctrl held after VK_CONTROL down"
        );
        // Held modifier, not a one-shot: still set on a later frame with no new
        // event (so a sustained hold keeps fast-forwarding).
        assert!(
            snapshot(&mut ks).held.contains(InputKey::Control),
            "ctrl stays held across frames"
        );
        ks.on_key_up(VK_CONTROL);
        assert!(
            !snapshot(&mut ks).held.contains(InputKey::Control),
            "ctrl released after VK_CONTROL up"
        );
    }

    // Control and Alt queue a press as they go down, once, so an action or a
    // rebind capture bound to them sees them like any other key.
    #[test]
    fn modifiers_queue_a_press_as_they_go_down() {
        let pressed = |snap: &InputSnapshot| -> Vec<InputKey> {
            snap.key_events
                .iter()
                .filter_map(|e| match e {
                    KeyEvent::Press(p) if !p.repeat => Some(p.key),
                    _ => None,
                })
                .collect()
        };
        let mut ks = KeyState::default();
        ks.on_key_down(VK_CONTROL, false);
        ks.on_sys_key(VK_MENU, true);
        ks.on_sys_key(VK_MENU, true);
        let snap = snapshot(&mut ks);
        assert_eq!(pressed(&snap), [InputKey::Control, InputKey::Alt]);
        assert!(snap.held.contains(InputKey::Alt));
        ks.on_sys_key(VK_MENU, false);
        assert!(!snapshot(&mut ks).held.contains(InputKey::Alt));
    }

    fn text(ks: &mut KeyState) -> String {
        snapshot(ks)
            .key_events
            .iter()
            .filter_map(|e| match e {
                KeyEvent::Text(c) => Some(*c),
                KeyEvent::Press(_) => None,
            })
            .collect()
    }

    fn unit(c: char) -> u16 {
        let mut buf = [0u16; 2];
        c.encode_utf16(&mut buf)[0]
    }

    #[test]
    fn typed_glyphs_queue_in_order_and_drain_once() {
        // Regression: typing used to keep one glyph per frame, so fast typing
        // at a low frame rate dropped keys.
        let mut ks = KeyState::default();
        assert_eq!(text(&mut ks), "", "starts empty");
        for c in "Abc".chars() {
            ks.on_char(unit(c));
        }
        assert_eq!(text(&mut ks), "Abc", "every glyph of the frame survives");
        assert_eq!(text(&mut ks), "", "drained by take");
    }

    #[test]
    fn surrogate_pairs_join_into_one_char() {
        let mut ks = KeyState::default();
        let mut buf = [0u16; 2];
        for &u in '\u{1F600}'.encode_utf16(&mut buf).iter() {
            ks.on_char(u);
        }
        assert_eq!(text(&mut ks), "\u{1F600}");
    }

    #[test]
    fn control_chars_do_not_become_text() {
        // Backspace / Enter / Escape / Tab arrive as WM_CHAR control codes; they
        // must be filtered so they travel as presses, not inserted as text.
        let mut ks = KeyState::default();
        for c in ['\u{08}', '\r', '\n', '\u{1b}', '\t', '\u{7f}'] {
            ks.on_char(unit(c));
        }
        assert_eq!(text(&mut ks), "", "control chars are not printable glyphs");
    }

    #[test]
    fn presses_record_modifiers_and_repeats_in_order() {
        let mut ks = KeyState::default();
        ks.on_key_down(VK_CONTROL, false);
        ks.on_key_down(VK_Z, false);
        ks.on_key_up(VK_CONTROL);
        ks.on_key_down(VK_LEFT, false);
        ks.on_key_down(VK_LEFT, true);
        let presses: Vec<KeyPress> = snapshot(&mut ks)
            .key_events
            .iter()
            .filter_map(|e| match e {
                KeyEvent::Press(p) => Some(*p),
                KeyEvent::Text(_) => None,
            })
            .collect();
        let keys: Vec<InputKey> = presses.iter().map(|p| p.key).collect();
        assert_eq!(
            keys,
            [
                InputKey::Control,
                InputKey::Z,
                InputKey::Left,
                InputKey::Left
            ]
        );
        assert!(presses[1].mods.ctrl, "Ctrl held for the Z");
        assert!(!presses[2].mods.ctrl, "released before the Left");
        assert!(!presses[2].repeat && presses[3].repeat);
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
    fn editing_keys_decode_as_presses() {
        // The editing and navigation keys decode so text fields can edit; they
        // travel as presses, never as text (mirrors appkit/input.rs).
        assert_eq!(key_from_vk(VK_BACK), Some(InputKey::Backspace));
        assert_eq!(key_from_vk(VK_DELETE), Some(InputKey::Delete));
        assert_eq!(key_from_vk(VK_LEFT), Some(InputKey::Left));
        assert_eq!(key_from_vk(VK_RIGHT), Some(InputKey::Right));
        assert_eq!(key_from_vk(VK_HOME), Some(InputKey::Home));
        assert_eq!(key_from_vk(VK_END), Some(InputKey::End));
        assert_eq!(key_from_vk(VK_PRIOR), Some(InputKey::PageUp));
        assert_eq!(key_from_vk(VK_NEXT), Some(InputKey::PageDown));
        let mut ks = KeyState::default();
        ks.on_key_down(VK_BACK, false);
        assert_eq!(
            snapshot(&mut ks).key_events,
            [KeyEvent::Press(KeyPress::new(
                InputKey::Backspace,
                KeyMods::NONE
            ))]
        );
    }
}

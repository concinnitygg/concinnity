//! InputKey-event decoding and the persistent per-frame input state for the `AppKit`
//! window layer. The pieces here are pure (no window, no view, no Objective-C
//! state beyond reading an `NSEvent`), so they are unit-testable and shared by
//! every backend that renders into an `NSView`; `window.rs` owns the event pump
//! that drives them. Mirrors `win32/input.rs`.

use concinnity_engine::components::{InputKey, KeyEvent, KeyMods, KeyPress};
use concinnity_engine::input::held_keys::HeldKeys;
use objc2_app_kit::{NSEvent, NSEventModifierFlags};

// Persistent key state tracked across frames. Held keys are set on KeyDown
// and cleared on KeyUp; they are never reset between frames so that held keys
// remain active even when no repeat event arrives (avoiding the OS key-repeat
// delay gap). Mouse deltas are accumulated here and cleared by take_input().
// Pulse fields are set on KeyDown and cleared after one take_input() call so
// callers see exactly one true frame per press.
#[derive(Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "an input snapshot: one flag per action or key"
)]
pub(super) struct KeyState {
    pub(super) held: HeldKeys,
    pub(super) mouse_dx: f32,
    pub(super) mouse_dy: f32,
    // Accumulated vertical scroll-wheel delta since the last take_input();
    // cleared by take_input() like the mouse deltas. Used by scrollable UI.
    pub(super) scroll_delta: f32,
    // Absolute cursor position in window-content pixels (origin top-left).
    pub(super) mouse_x: f32,
    pub(super) mouse_y: f32,
    // Pulse: set on left-mouse-down when cursor is free; cleared by take_input().
    pub(super) left_click_pulse: bool,
    // Held: set on left-mouse-down and cleared on left-mouse-up (cursor free).
    // Unlike the pulse it persists across frames, so a UI drag (slider) can
    // track the cursor for the whole press. NOT cleared by take_input().
    pub(super) left_button_down: bool,
    // Pulse: set on right-mouse-down when cursor is free; cleared by take_input().
    pub(super) right_click_pulse: bool,
    // Pulse: set on F1 key-down; cleared by take_input().
    pub(super) hud_toggle_pulse: bool,
    // Pulse: set on Escape key-down when the cursor is not captured;
    // cleared by take_input(). When the cursor is captured Escape continues
    // to call release_cursor() instead.
    pub(super) escape_pulse: bool,
    // Every key press (repeats included) and printable character since the
    // last take_input(), in arrival order; drained by take_input(). Not gated
    // by capture / menu state, so a rebind row and text fields see every key.
    // Control glyphs (Backspace, Enter, Escape, arrows) never become text:
    // those travel as presses.
    pub(super) events: Vec<KeyEvent>,
    // Whether Command is currently held, tracked from FlagsChanged like
    // Shift. Surfaced as a held modifier (the editor's palette shortcut); a
    // Command chord also suppresses text and gameplay bindings in `handle_key`.
    pub(super) command_down: bool,
    // Set by capture_cursor(); the next mouse-motion event after capture
    // has its delta discarded so queued pre-capture events (which were
    // produced before CGAssociateMouseAndMouseCursorPosition(0) took
    // effect, often during init) can't snap the camera.
    pub(super) discard_next_motion: bool,
    // Whether the real cursor has left the window client area while the cursor
    // is free (windowed / borderless). Recomputed each frame by
    // `update_ui_cursor_confinement`; the renderer hides the in-engine cursor
    // when set. False while captured or in fullscreen (which confines instead).
    pub(super) cursor_outside_window: bool,
}

// Whether a character is a printable glyph suitable for a text field, i.e. not a
// control character and not in the NSFunctionKey private-use range
// (0xF700-0xF8FF: arrows, F-keys, Home/End, and the like).
pub(super) fn is_printable_glyph(c: char) -> bool {
    !c.is_control() && !('\u{F700}'..='\u{F8FF}').contains(&c)
}

// The printable glyph a key event produces, or None for a control / navigation
// key (Backspace, Enter, Escape, arrows, etc.). macOS puts the layout- and
// modifier-resolved characters on the event, so casing and shifted symbols are
// already correct.
pub(super) fn printable_char(event: &NSEvent) -> Option<char> {
    let text = event.characters()?.to_string();
    let c = text.chars().next()?;
    is_printable_glyph(c).then_some(c)
}

// The modifiers a key event carries, read from the event itself so they hold
// whatever order the queue delivered FlagsChanged in.
pub(super) fn event_mods(flags: NSEventModifierFlags) -> KeyMods {
    KeyMods {
        shift: flags.contains(NSEventModifierFlags::Shift),
        ctrl: flags.contains(NSEventModifierFlags::Control),
        alt: flags.contains(NSEventModifierFlags::Option),
        cmd: flags.contains(NSEventModifierFlags::Command),
    }
}

// Map a macOS virtual key code to a canonical `InputKey`, or `None` for a key the
// engine does not bind (modifiers other than Shift, function keys, Escape, etc.).
// The codes are hardware-independent (the same on every Mac keyboard). Shift is
// deliberately absent: it arrives via FlagsChanged, not a key code.
pub(super) fn key_from_mac(kc: u16) -> Option<InputKey> {
    Some(match kc {
        0 => InputKey::A,
        11 => InputKey::B,
        8 => InputKey::C,
        2 => InputKey::D,
        14 => InputKey::E,
        3 => InputKey::F,
        5 => InputKey::G,
        4 => InputKey::H,
        34 => InputKey::I,
        38 => InputKey::J,
        40 => InputKey::K,
        37 => InputKey::L,
        46 => InputKey::M,
        45 => InputKey::N,
        31 => InputKey::O,
        35 => InputKey::P,
        12 => InputKey::Q,
        15 => InputKey::R,
        1 => InputKey::S,
        17 => InputKey::T,
        32 => InputKey::U,
        9 => InputKey::V,
        13 => InputKey::W,
        7 => InputKey::X,
        16 => InputKey::Y,
        6 => InputKey::Z,
        29 => InputKey::Num0,
        18 => InputKey::Num1,
        19 => InputKey::Num2,
        20 => InputKey::Num3,
        21 => InputKey::Num4,
        23 => InputKey::Num5,
        22 => InputKey::Num6,
        26 => InputKey::Num7,
        28 => InputKey::Num8,
        25 => InputKey::Num9,
        49 => InputKey::Space,
        48 => InputKey::Tab,
        36 => InputKey::Enter,
        51 => InputKey::Backspace,
        117 => InputKey::Delete,
        123 => InputKey::Left,
        124 => InputKey::Right,
        125 => InputKey::Down,
        126 => InputKey::Up,
        115 => InputKey::Home,
        119 => InputKey::End,
        116 => InputKey::PageUp,
        121 => InputKey::PageDown,
        27 => InputKey::Minus,
        24 => InputKey::Equals,
        33 => InputKey::LeftBracket,
        30 => InputKey::RightBracket,
        42 => InputKey::Backslash,
        41 => InputKey::Semicolon,
        39 => InputKey::Quote,
        43 => InputKey::Comma,
        47 => InputKey::Period,
        44 => InputKey::Slash,
        50 => InputKey::Backtick,
        _ => return None,
    })
}

impl KeyState {
    // Take the modifier keys a FlagsChanged event reports as held: each one
    // that went down queues a press, so an action or a rebind capture bound to
    // it sees it the way it sees any other key.
    pub(super) fn modifiers_changed(&mut self, flags: NSEventModifierFlags) {
        const MODIFIERS: [(InputKey, NSEventModifierFlags); 3] = [
            (InputKey::Shift, NSEventModifierFlags::Shift),
            (InputKey::Control, NSEventModifierFlags::Control),
            (InputKey::Alt, NSEventModifierFlags::Option),
        ];
        for (key, flag) in MODIFIERS {
            let down = flags.contains(flag);
            if down && !self.held.contains(key) {
                self.events
                    .push(KeyEvent::Press(KeyPress::new(key, event_mods(flags))));
            }
            self.held.set(key, down);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each modifier that goes down queues one press and is held until it is
    // let go; one already down queues nothing more.
    #[test]
    fn modifiers_queue_a_press_as_they_go_down() {
        let presses = |ks: &KeyState| -> Vec<InputKey> {
            ks.events
                .iter()
                .filter_map(|e| match e {
                    KeyEvent::Press(p) => Some(p.key),
                    KeyEvent::Text(_) => None,
                })
                .collect()
        };
        let mut ks = KeyState::default();
        ks.modifiers_changed(NSEventModifierFlags::Control);
        ks.modifiers_changed(NSEventModifierFlags::Control | NSEventModifierFlags::Option);
        assert_eq!(presses(&ks), [InputKey::Control, InputKey::Alt]);
        assert!(ks.held.contains(InputKey::Control) && ks.held.contains(InputKey::Alt));
        ks.modifiers_changed(NSEventModifierFlags::empty());
        assert!(!ks.held.contains(InputKey::Control) && !ks.held.contains(InputKey::Alt));
        assert_eq!(presses(&ks).len(), 2);
    }

    #[test]
    fn key_from_mac_covers_the_defaults() {
        // The default bindings must decode, so a fresh world keeps moving.
        assert_eq!(key_from_mac(13), Some(InputKey::W));
        assert_eq!(key_from_mac(0), Some(InputKey::A));
        assert_eq!(key_from_mac(1), Some(InputKey::S));
        assert_eq!(key_from_mac(2), Some(InputKey::D));
        assert_eq!(key_from_mac(49), Some(InputKey::Space));
        assert_eq!(key_from_mac(14), Some(InputKey::E));
        // Escape / F1 stay fixed (no canonical mapping).
        assert_eq!(key_from_mac(53), None);
        assert_eq!(key_from_mac(122), None);
    }

    #[test]
    fn editing_keys_decode() {
        // The editing and navigation keys decode so text fields can edit; they
        // travel as presses, never as text.
        assert_eq!(key_from_mac(51), Some(InputKey::Backspace));
        assert_eq!(key_from_mac(117), Some(InputKey::Delete));
        assert_eq!(key_from_mac(115), Some(InputKey::Home));
        assert_eq!(key_from_mac(119), Some(InputKey::End));
        assert_eq!(key_from_mac(116), Some(InputKey::PageUp));
        assert_eq!(key_from_mac(121), Some(InputKey::PageDown));
    }

    #[test]
    fn event_mods_read_each_flag() {
        assert_eq!(event_mods(NSEventModifierFlags::empty()), KeyMods::NONE);
        assert_eq!(event_mods(NSEventModifierFlags::Command), KeyMods::CMD);
        let both = event_mods(NSEventModifierFlags::Option | NSEventModifierFlags::Shift);
        assert!(both.alt && both.shift && !both.ctrl && !both.cmd);
    }

    #[test]
    fn printable_glyph_filter() {
        // Real glyphs (including space) type; control and function keys don't.
        assert!(is_printable_glyph('a'));
        assert!(is_printable_glyph('Z'));
        assert!(is_printable_glyph(' '));
        assert!(is_printable_glyph('/'));
        assert!(!is_printable_glyph('\u{8}')); // Backspace
        assert!(!is_printable_glyph('\r')); // Enter
        assert!(!is_printable_glyph('\u{7f}')); // Delete
        assert!(!is_printable_glyph('\u{F702}')); // Left arrow (NSFunctionKey range)
    }
}

use crate::components::InputKey;

/// The modifier keys held at the moment of a [KeyPress](#keypress).
///
/// `cmd` is the Command key on macOS and is never set on Windows or Linux,
/// where application shortcuts are built on `ctrl` instead.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KeyMods {
    /// Shift was held.
    pub shift: bool,
    /// Control was held.
    pub ctrl: bool,
    /// Alt (Option on macOS) was held.
    pub alt: bool,
    /// Command (macOS only) was held.
    pub cmd: bool,
}

impl KeyMods {
    /// No modifier held.
    pub const NONE: KeyMods = KeyMods {
        shift: false,
        ctrl: false,
        alt: false,
        cmd: false,
    };

    /// Only Shift held.
    pub const SHIFT: KeyMods = KeyMods {
        shift: true,
        ..KeyMods::NONE
    };

    /// Only Control held.
    pub const CTRL: KeyMods = KeyMods {
        ctrl: true,
        ..KeyMods::NONE
    };

    /// Only Alt held.
    pub const ALT: KeyMods = KeyMods {
        alt: true,
        ..KeyMods::NONE
    };

    /// Only Command held.
    pub const CMD: KeyMods = KeyMods {
        cmd: true,
        ..KeyMods::NONE
    };

    /// These modifiers with Shift added.
    pub const fn with_shift(self) -> KeyMods {
        KeyMods {
            shift: true,
            ..self
        }
    }
}

/// One key press, carried by [FrameInput](#frameinput) in its
/// [key_events](#structfield.key_events).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KeyPress {
    /// The key pressed.
    pub key: InputKey,
    /// The modifiers held when it was pressed.
    pub mods: KeyMods,
    /// True when the operating system generated this press by auto-repeat
    /// while the key is held, rather than from a fresh press.
    pub repeat: bool,
}

impl KeyPress {
    /// A fresh press of `key` with `mods` held.
    pub const fn new(key: InputKey, mods: KeyMods) -> KeyPress {
        KeyPress {
            key,
            mods,
            repeat: false,
        }
    }
}

/// One keyboard event of a frame, in the order the window received it.
///
/// A printable key produces both a [Press](#variant.Press) and, right after
/// it, the [Text](#variant.Text) it typed. Editing and navigation keys
/// (Backspace, Delete, Enter, Tab, the arrows) produce only the press, and so
/// does a shortcut chord (Ctrl or Command held).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KeyEvent {
    /// A key went down, or repeated while held.
    Press(KeyPress),
    /// A printable character was typed, with the keyboard layout, Shift, and
    /// dead keys already applied by the operating system.
    Text(char),
}

impl KeyEvent {
    /// A fresh press of `key` with no modifier held.
    pub const fn press(key: InputKey) -> KeyEvent {
        KeyEvent::Press(KeyPress::new(key, KeyMods::NONE))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // (shift, ctrl, alt, cmd)
    fn flags(m: KeyMods) -> [bool; 4] {
        [m.shift, m.ctrl, m.alt, m.cmd]
    }

    #[test]
    fn mods_constants_set_one_modifier_each() {
        assert_eq!(KeyMods::default(), KeyMods::NONE);
        assert_eq!(flags(KeyMods::SHIFT), [true, false, false, false]);
        assert_eq!(flags(KeyMods::CTRL), [false, true, false, false]);
        assert_eq!(flags(KeyMods::ALT), [false, false, true, false]);
        assert_eq!(flags(KeyMods::CMD), [false, false, false, true]);
        assert_eq!(flags(KeyMods::CMD.with_shift()), [true, false, false, true]);
    }

    #[test]
    fn a_new_press_is_not_a_repeat() {
        let p = KeyPress::new(InputKey::A, KeyMods::CTRL);
        assert_eq!(p.key, InputKey::A);
        assert_eq!(p.mods, KeyMods::CTRL);
        assert!(!p.repeat);
        assert_eq!(
            KeyEvent::press(InputKey::B),
            KeyEvent::Press(KeyPress::new(InputKey::B, KeyMods::NONE))
        );
    }
}

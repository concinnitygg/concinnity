// src/assets/input_key.rs

/// A canonical, backend-agnostic keyboard key.
///
/// Each rendering backend maps its native key codes (macOS NSEvent key codes,
/// Windows virtual keys, GLFW keys) to and from this enum, so a key binding can
/// be stored and shown the same way everywhere. Unit variants serialize to
/// their name, so a persisted binding survives a build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Key {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Num0,
    Num1,
    Num2,
    Num3,
    Num4,
    Num5,
    Num6,
    Num7,
    Num8,
    Num9,
    Space,
    Tab,
    Enter,
    Shift,
    Control,
    Alt,
    Up,
    Down,
    Left,
    Right,
    Minus,
    Equals,
    LeftBracket,
    RightBracket,
    Backslash,
    Semicolon,
    Quote,
    Comma,
    Period,
    Slash,
    Backtick,
}

impl Key {
    /// The canonical variant name, matching the serialized form and how a
    /// [KeyBinding](#keybinding) stores its `key` (e.g. `"W"`, `"Space"`,
    /// `"Enter"`, `"Control"`). Unlike [display_name](#method.display_name)
    /// this is the exact enum-variant spelling, so it round-trips with serde.
    pub fn name(self) -> &'static str {
        match self {
            Key::A => "A",
            Key::B => "B",
            Key::C => "C",
            Key::D => "D",
            Key::E => "E",
            Key::F => "F",
            Key::G => "G",
            Key::H => "H",
            Key::I => "I",
            Key::J => "J",
            Key::K => "K",
            Key::L => "L",
            Key::M => "M",
            Key::N => "N",
            Key::O => "O",
            Key::P => "P",
            Key::Q => "Q",
            Key::R => "R",
            Key::S => "S",
            Key::T => "T",
            Key::U => "U",
            Key::V => "V",
            Key::W => "W",
            Key::X => "X",
            Key::Y => "Y",
            Key::Z => "Z",
            Key::Num0 => "Num0",
            Key::Num1 => "Num1",
            Key::Num2 => "Num2",
            Key::Num3 => "Num3",
            Key::Num4 => "Num4",
            Key::Num5 => "Num5",
            Key::Num6 => "Num6",
            Key::Num7 => "Num7",
            Key::Num8 => "Num8",
            Key::Num9 => "Num9",
            Key::Space => "Space",
            Key::Tab => "Tab",
            Key::Enter => "Enter",
            Key::Shift => "Shift",
            Key::Control => "Control",
            Key::Alt => "Alt",
            Key::Up => "Up",
            Key::Down => "Down",
            Key::Left => "Left",
            Key::Right => "Right",
            Key::Minus => "Minus",
            Key::Equals => "Equals",
            Key::LeftBracket => "LeftBracket",
            Key::RightBracket => "RightBracket",
            Key::Backslash => "Backslash",
            Key::Semicolon => "Semicolon",
            Key::Quote => "Quote",
            Key::Comma => "Comma",
            Key::Period => "Period",
            Key::Slash => "Slash",
            Key::Backtick => "Backtick",
        }
    }

    /// A short label for the settings menu (e.g. `"W"`, `"Space"`, `"Shift"`).
    pub fn display_name(self) -> &'static str {
        match self {
            Key::A => "A",
            Key::B => "B",
            Key::C => "C",
            Key::D => "D",
            Key::E => "E",
            Key::F => "F",
            Key::G => "G",
            Key::H => "H",
            Key::I => "I",
            Key::J => "J",
            Key::K => "K",
            Key::L => "L",
            Key::M => "M",
            Key::N => "N",
            Key::O => "O",
            Key::P => "P",
            Key::Q => "Q",
            Key::R => "R",
            Key::S => "S",
            Key::T => "T",
            Key::U => "U",
            Key::V => "V",
            Key::W => "W",
            Key::X => "X",
            Key::Y => "Y",
            Key::Z => "Z",
            Key::Num0 => "0",
            Key::Num1 => "1",
            Key::Num2 => "2",
            Key::Num3 => "3",
            Key::Num4 => "4",
            Key::Num5 => "5",
            Key::Num6 => "6",
            Key::Num7 => "7",
            Key::Num8 => "8",
            Key::Num9 => "9",
            Key::Space => "Space",
            Key::Tab => "Tab",
            Key::Enter => "Enter",
            Key::Shift => "Shift",
            Key::Control => "Ctrl",
            Key::Alt => "Alt",
            Key::Up => "Up",
            Key::Down => "Down",
            Key::Left => "Left",
            Key::Right => "Right",
            Key::Minus => "-",
            Key::Equals => "=",
            Key::LeftBracket => "[",
            Key::RightBracket => "]",
            Key::Backslash => "\\",
            Key::Semicolon => ";",
            Key::Quote => "'",
            Key::Comma => ",",
            Key::Period => ".",
            Key::Slash => "/",
            Key::Backtick => "`",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_to_variant_name() {
        // A unit variant serializes to its name, so a persisted binding is
        // readable and stable across builds.
        let json = serde_json::to_string(&Key::W).unwrap();
        assert_eq!(json, "\"W\"");
        let back: Key = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Key::W);
    }

    #[test]
    fn display_names_are_short() {
        assert_eq!(Key::W.display_name(), "W");
        assert_eq!(Key::Space.display_name(), "Space");
        assert_eq!(Key::Shift.display_name(), "Shift");
        assert_eq!(Key::Num1.display_name(), "1");
    }

    #[test]
    fn name_matches_serialization() {
        // `name` must equal the serialized variant name so it can be compared
        // against a KeyBinding's stored `key` string (and parsed back).
        for key in [
            Key::A,
            Key::Space,
            Key::Enter,
            Key::Control,
            Key::Shift,
            Key::Num0,
            Key::Num9,
            Key::Up,
            Key::Slash,
            Key::Backtick,
        ] {
            assert_eq!(
                serde_json::to_string(&key).unwrap(),
                format!("\"{}\"", key.name()),
                "name() must match serde for {key:?}",
            );
        }
    }

    // Every declared key, so both match tables are exercised end to end.
    const ALL_KEYS: [Key; 57] = [
        Key::A,
        Key::B,
        Key::C,
        Key::D,
        Key::E,
        Key::F,
        Key::G,
        Key::H,
        Key::I,
        Key::J,
        Key::K,
        Key::L,
        Key::M,
        Key::N,
        Key::O,
        Key::P,
        Key::Q,
        Key::R,
        Key::S,
        Key::T,
        Key::U,
        Key::V,
        Key::W,
        Key::X,
        Key::Y,
        Key::Z,
        Key::Num0,
        Key::Num1,
        Key::Num2,
        Key::Num3,
        Key::Num4,
        Key::Num5,
        Key::Num6,
        Key::Num7,
        Key::Num8,
        Key::Num9,
        Key::Space,
        Key::Tab,
        Key::Enter,
        Key::Shift,
        Key::Control,
        Key::Alt,
        Key::Up,
        Key::Down,
        Key::Left,
        Key::Right,
        Key::Minus,
        Key::Equals,
        Key::LeftBracket,
        Key::RightBracket,
        Key::Backslash,
        Key::Semicolon,
        Key::Quote,
        Key::Comma,
        Key::Period,
        Key::Slash,
        Key::Backtick,
    ];

    #[test]
    fn all_variants_cover_name_and_display() {
        // For every variant: name() and display_name() are non-empty, name()
        // equals the serde spelling, and the binding round-trips. This walks
        // both full match statements, not just a hand-picked sample.
        for key in ALL_KEYS {
            assert!(!key.name().is_empty(), "name empty for {key:?}");
            assert!(!key.display_name().is_empty(), "display empty for {key:?}");
            let json = serde_json::to_string(&key).unwrap();
            assert_eq!(
                json,
                format!("\"{}\"", key.name()),
                "serde vs name for {key:?}"
            );
            let back: Key = serde_json::from_str(&json).unwrap();
            assert_eq!(back, key, "round trip for {key:?}");
        }
    }

    #[test]
    fn variant_names_are_unique() {
        // Names double as persisted identifiers, so no two variants may share
        // one (this also guards ALL_KEYS against an accidental duplicate).
        let mut seen = std::collections::HashSet::new();
        for key in ALL_KEYS {
            assert!(seen.insert(key.name()), "duplicate name {}", key.name());
        }
    }
}

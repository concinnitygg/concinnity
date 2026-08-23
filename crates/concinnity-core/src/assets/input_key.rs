// src/assets/input_key.rs

// Declare a key enum from one table, so the variant list, the serde spelling,
// the short label, and the exhaustive `ALL` array cannot drift apart. Each
// entry is `Variant` (its label is the variant name) or `Variant => "label"`
// when the settings menu shows something shorter.
// The settings-menu label for one table entry: the override when given, the
// variant name otherwise.

macro_rules! key_label {
    ($variant:ident) => {
        stringify!($variant)
    };
    ($variant:ident => $label:literal) => {
        $label
    };
}

macro_rules! define_keys {
    ($($variant:ident $(=> $label:literal)?),* $(,)?) => {
        /// A canonical, backend-agnostic keyboard key.
        ///
        /// Each rendering backend maps its native key codes (macOS NSEvent key
        /// codes, Windows virtual keys, GLFW keys) to and from this enum, so a
        /// key binding can be stored and shown the same way everywhere. Unit
        /// variants serialize to their name, so a persisted binding survives a
        /// build.
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
        )]
        // Each variant is one key name; the vocabulary is described above
        // rather than restated per variant.
        #[expect(missing_docs, reason = "each variant is one key name; the vocabulary is documented on the enum")]
        pub enum InputKey {
            $($variant),*
        }

        impl InputKey {
            /// Every declared key, in declaration order.
            pub const ALL: &'static [InputKey] = &[$(InputKey::$variant),*];

            /// The canonical variant name, matching the serialized form and how
            /// a [KeyBinding](#keybinding) stores its `key` (e.g. `"W"`,
            /// `"Space"`, `"Enter"`, `"Control"`). Unlike
            /// [display_name](#method.display_name) this is the exact
            /// enum-variant spelling, so it round-trips with serde.
            pub fn name(self) -> &'static str {
                match self {
                    $(InputKey::$variant => stringify!($variant)),*
                }
            }

            /// A short label for the settings menu (e.g. `"W"`, `"Space"`,
            /// `"Ctrl"`). Defaults to [name](#method.name) unless the key
            /// declared a shorter one.
            pub fn display_name(self) -> &'static str {
                match self {
                    $(InputKey::$variant => key_label!($variant $(=> $label)?)),*
                }
            }
        }
    };
}

define_keys! {
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Num0 => "0",
    Num1 => "1",
    Num2 => "2",
    Num3 => "3",
    Num4 => "4",
    Num5 => "5",
    Num6 => "6",
    Num7 => "7",
    Num8 => "8",
    Num9 => "9",
    Space,
    Tab,
    Enter,
    Backspace => "Bksp",
    Delete => "Del",
    Shift,
    Control => "Ctrl",
    Alt,
    Up,
    Down,
    Left,
    Right,
    Minus => "-",
    Equals => "=",
    LeftBracket => "[",
    RightBracket => "]",
    Backslash => "\\",
    Semicolon => ";",
    Quote => "'",
    Comma => ",",
    Period => ".",
    Slash => "/",
    Backtick => "`",
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn serializes_to_variant_name() {
        // A unit variant serializes to its name, so a persisted binding is
        // readable and stable across builds.
        let json = serde_json::to_string(&InputKey::W).unwrap();
        assert_eq!(json, "\"W\"");
        let back: InputKey = serde_json::from_str(&json).unwrap();
        assert_eq!(back, InputKey::W);
    }

    #[test]
    fn display_names_are_short() {
        assert_eq!(InputKey::W.display_name(), "W");
        assert_eq!(InputKey::Space.display_name(), "Space");
        assert_eq!(InputKey::Shift.display_name(), "Shift");
        assert_eq!(InputKey::Num1.display_name(), "1");
        assert_eq!(InputKey::Backspace.display_name(), "Bksp");
        assert_eq!(InputKey::Control.display_name(), "Ctrl");
        assert_eq!(InputKey::Minus.display_name(), "-");
    }

    #[test]
    fn all_variants_cover_name_and_display() {
        // For every variant: name() and display_name() are non-empty, name()
        // equals the serde spelling, and the binding round-trips. This walks
        // both full match statements, not just a hand-picked sample.
        for &key in InputKey::ALL {
            assert!(!key.name().is_empty(), "name empty for {key:?}");
            assert!(!key.display_name().is_empty(), "display empty for {key:?}");
            let json = serde_json::to_string(&key).unwrap();
            assert_eq!(
                json,
                format!("\"{}\"", key.name()),
                "serde vs name for {key:?}"
            );
            let back: InputKey = serde_json::from_str(&json).unwrap();
            assert_eq!(back, key, "round trip for {key:?}");
        }
    }

    #[test]
    fn variant_names_are_unique() {
        // Names double as persisted identifiers, so no two variants may share
        // one.
        let mut seen = alloc::collections::BTreeSet::new();
        for &key in InputKey::ALL {
            assert!(seen.insert(key.name()), "duplicate name {}", key.name());
        }
    }
}

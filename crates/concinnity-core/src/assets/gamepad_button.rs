// src/assets/gamepad_button.rs

// Declare the button enum from one table, so the variant list, the serde
// spelling, the short label, and the exhaustive `ALL` slice cannot drift
// apart. Each entry is `Variant`, or `Variant => "label"` when the settings
// menu shows something shorter.
macro_rules! button_label {
    ($variant:ident) => {
        stringify!($variant)
    };
    ($variant:ident => $label:literal) => {
        $label
    };
}

macro_rules! define_buttons {
    ($($variant:ident $(=> $label:literal)?),* $(,)?) => {
        /// A canonical, vendor-neutral gamepad button.
        ///
        /// Face buttons are named by position (`South` is the bottom face
        /// button: Xbox A, PlayStation Cross) so a persisted binding reads the
        /// same for every controller brand. Unit variants serialize to their
        /// name, so a persisted binding survives a build, like [Key](#key).
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
        )]
        pub enum GamepadButton {
            $($variant),*
        }

        impl GamepadButton {
            /// Every declared button, in a stable display order.
            pub const ALL: &'static [GamepadButton] = &[$(GamepadButton::$variant),*];

            /// The canonical variant name, matching the serialized form (e.g.
            /// `"South"`, `"LeftShoulder"`). Like [Key::name](#method.name)
            /// this is the exact enum-variant spelling, so it round-trips with
            /// serde.
            pub fn name(self) -> &'static str {
                match self {
                    $(GamepadButton::$variant => stringify!($variant)),*
                }
            }

            /// A short label for the settings menu (e.g. `"South"`, `"LB"`,
            /// `"L3"`). Defaults to [name](#method.name) unless the button
            /// declared a shorter one.
            pub fn display_name(self) -> &'static str {
                match self {
                    $(GamepadButton::$variant => button_label!($variant $(=> $label)?)),*
                }
            }
        }
    };
}

define_buttons! {
    South,
    East,
    West,
    North,
    LeftShoulder => "LB",
    RightShoulder => "RB",
    LeftTrigger => "LT",
    RightTrigger => "RT",
    LeftStick => "L3",
    RightStick => "R3",
    DpadUp => "D-Up",
    DpadDown => "D-Down",
    DpadLeft => "D-Left",
    DpadRight => "D-Right",
    Start,
    Select,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_to_variant_name() {
        let json = serde_json::to_string(&GamepadButton::South).unwrap();
        assert_eq!(json, "\"South\"");
        let back: GamepadButton = serde_json::from_str(&json).unwrap();
        assert_eq!(back, GamepadButton::South);
    }

    #[test]
    fn all_variants_cover_name_and_display() {
        // For every variant: name() and display_name() are non-empty, name()
        // equals the serde spelling, and the binding round-trips.
        for &button in GamepadButton::ALL {
            assert!(!button.name().is_empty(), "name empty for {button:?}");
            assert!(
                !button.display_name().is_empty(),
                "display empty for {button:?}"
            );
            let json = serde_json::to_string(&button).unwrap();
            assert_eq!(
                json,
                format!("\"{}\"", button.name()),
                "serde vs name for {button:?}"
            );
            let back: GamepadButton = serde_json::from_str(&json).unwrap();
            assert_eq!(back, button, "round trip for {button:?}");
        }
    }

    #[test]
    fn variant_names_are_unique() {
        // Names double as persisted identifiers, so no two variants may share
        // one (this also guards ALL against an accidental duplicate).
        let mut seen = alloc::collections::BTreeSet::new();
        for &button in GamepadButton::ALL {
            assert!(
                seen.insert(button.name()),
                "duplicate name {}",
                button.name()
            );
        }
        assert_eq!(seen.len(), GamepadButton::ALL.len());
    }
}

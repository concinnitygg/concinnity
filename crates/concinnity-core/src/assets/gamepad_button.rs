// src/assets/gamepad_button.rs

/// A canonical, vendor-neutral gamepad button.
///
/// Face buttons are named by position (`South` is the bottom face button:
/// Xbox A, PlayStation Cross) so a persisted binding reads the same for every
/// controller brand. Unit variants serialize to their name, so a persisted
/// binding survives a build, like [Key](#key).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GamepadButton {
    South,
    East,
    West,
    North,
    LeftShoulder,
    RightShoulder,
    LeftTrigger,
    RightTrigger,
    LeftStick,
    RightStick,
    DpadUp,
    DpadDown,
    DpadLeft,
    DpadRight,
    Start,
    Select,
}

impl GamepadButton {
    /// Every declared button, in a stable display order.
    pub const ALL: [GamepadButton; 16] = [
        GamepadButton::South,
        GamepadButton::East,
        GamepadButton::West,
        GamepadButton::North,
        GamepadButton::LeftShoulder,
        GamepadButton::RightShoulder,
        GamepadButton::LeftTrigger,
        GamepadButton::RightTrigger,
        GamepadButton::LeftStick,
        GamepadButton::RightStick,
        GamepadButton::DpadUp,
        GamepadButton::DpadDown,
        GamepadButton::DpadLeft,
        GamepadButton::DpadRight,
        GamepadButton::Start,
        GamepadButton::Select,
    ];

    /// The canonical variant name, matching the serialized form (e.g. `"South"`,
    /// `"LeftShoulder"`). Like [Key::name](#method.name) this is the exact
    /// enum-variant spelling, so it round-trips with serde.
    pub fn name(self) -> &'static str {
        match self {
            GamepadButton::South => "South",
            GamepadButton::East => "East",
            GamepadButton::West => "West",
            GamepadButton::North => "North",
            GamepadButton::LeftShoulder => "LeftShoulder",
            GamepadButton::RightShoulder => "RightShoulder",
            GamepadButton::LeftTrigger => "LeftTrigger",
            GamepadButton::RightTrigger => "RightTrigger",
            GamepadButton::LeftStick => "LeftStick",
            GamepadButton::RightStick => "RightStick",
            GamepadButton::DpadUp => "DpadUp",
            GamepadButton::DpadDown => "DpadDown",
            GamepadButton::DpadLeft => "DpadLeft",
            GamepadButton::DpadRight => "DpadRight",
            GamepadButton::Start => "Start",
            GamepadButton::Select => "Select",
        }
    }

    /// A short label for the settings menu (e.g. `"South"`, `"LB"`, `"L3"`).
    pub fn display_name(self) -> &'static str {
        match self {
            GamepadButton::South => "South",
            GamepadButton::East => "East",
            GamepadButton::West => "West",
            GamepadButton::North => "North",
            GamepadButton::LeftShoulder => "LB",
            GamepadButton::RightShoulder => "RB",
            GamepadButton::LeftTrigger => "LT",
            GamepadButton::RightTrigger => "RT",
            GamepadButton::LeftStick => "L3",
            GamepadButton::RightStick => "R3",
            GamepadButton::DpadUp => "D-Up",
            GamepadButton::DpadDown => "D-Down",
            GamepadButton::DpadLeft => "D-Left",
            GamepadButton::DpadRight => "D-Right",
            GamepadButton::Start => "Start",
            GamepadButton::Select => "Select",
        }
    }
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
        for button in GamepadButton::ALL {
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
        for button in GamepadButton::ALL {
            assert!(
                seen.insert(button.name()),
                "duplicate name {}",
                button.name()
            );
        }
        assert_eq!(seen.len(), GamepadButton::ALL.len());
    }
}

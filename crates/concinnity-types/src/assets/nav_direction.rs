// src/assets/nav_direction.rs

/// A directional UI-navigation pulse, carried by
/// [FrameInput::nav](#structfield.nav) for one frame.
///
/// Produced from d-pad presses (with hold auto-repeat) and deliberate
/// left-stick deflections; menu focus movement consumes it while a screen is
/// active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NavDirection {
    Up,
    Down,
    Left,
    Right,
}

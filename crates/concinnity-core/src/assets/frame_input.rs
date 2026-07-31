// src/assets/frame_input.rs

/// Per-frame keyboard and mouse input state.
///
/// One `FrameInput` is updated each frame from the window's keyboard and mouse
/// state and read by camera and UI behavior. It is maintained automatically and
/// is never saved with the world.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct FrameInput {
    /// True while the move-forward key (W) is held.
    pub forward: bool,
    /// True while the move-backward key (S) is held.
    pub backward: bool,
    /// True while the strafe-left key (A) is held.
    pub left: bool,
    /// True while the strafe-right key (D) is held.
    pub right: bool,
    /// True while the sprint key (Shift) is held.
    pub sprint: bool,
    /// True for exactly one frame when the interact key (E) is pressed.
    pub interact: bool,
    /// True for exactly one frame when the jump key (Space) is pressed.
    pub jump: bool,
    /// True while the Control key is held. A modifier used by UI (e.g. a story
    /// fast-forwards its dialogue while it is down). Like [escape](#structfield.escape)
    /// and [captured_key](#structfield.captured_key) it is not frozen while a
    /// menu is open.
    pub ctrl: bool,
    /// True while the Shift key is held. A modifier used by UI (e.g. additive
    /// selection in the editor). Unlike [sprint](#structfield.sprint) it is
    /// not frozen while a menu is open, matching [ctrl](#structfield.ctrl).
    pub shift: bool,
    /// Gamepad left-stick movement vector `[x, y]`: `x` is rightward strafe,
    /// `y` is forward. Radial deadzone applied, magnitude at most 1, so partial
    /// deflection walks slower. `[0.0, 0.0]` with no gamepad; frozen (zeroed)
    /// while a menu is open, like the movement keys.
    pub move_axis: [f32; 2],
    /// Gamepad right-stick look vector `[x, y]` in the mouse-delta sign
    /// convention (positive `x` looks right, positive `y` looks down), with
    /// deadzone and response curve applied, magnitude at most 1. Unlike the
    /// pixel-based mouse deltas this is a deflection rate: consumers scale it
    /// by their look speed and the frame time. Frozen while a menu is open.
    pub look_axis: [f32; 2],
    /// The gamepad button pressed this frame, for one frame, or `None`.
    /// Surfaced regardless of menu state (like
    /// [captured_key](#structfield.captured_key)) so the settings menu can
    /// capture a button for rebinding.
    pub captured_button: Option<crate::assets::GamepadButton>,
    /// The UI-navigation pulse this frame, or `None`. A one-frame
    /// [NavDirection](#navdirection) produced from a d-pad press (repeating
    /// while held) or a deliberate left-stick deflection. Surfaced regardless
    /// of menu state (like [captured_key](#structfield.captured_key)); menu
    /// focus movement consumes it only while a screen is active, so during
    /// play the d-pad and stick keep their movement meaning.
    pub nav: Option<crate::assets::NavDirection>,
    /// True for exactly one frame when the confirm button (South) is pressed.
    /// Surfaced regardless of menu state; menus fire their focused control
    /// from it while a screen is active.
    pub confirm: bool,
    /// True for exactly one frame when the back button (East) is pressed.
    /// Surfaced regardless of menu state; menus treat it like Escape while a
    /// screen is active.
    pub back: bool,
    /// Accumulated horizontal mouse movement since the last frame (pixels).
    pub mouse_dx: f32,
    /// Accumulated vertical mouse movement since the last frame (pixels).
    pub mouse_dy: f32,
    /// Accumulated vertical scroll-wheel movement since the last frame. Positive
    /// scrolls the content up (a scrollable UI panel moves its rows up). Cleared
    /// each frame like the mouse deltas.
    pub scroll_delta: f32,
    /// Absolute cursor X position in window pixels (origin top-left).
    /// Only meaningful when the cursor is not captured.
    pub mouse_x: f32,
    /// Absolute cursor Y position in window pixels (origin top-left).
    /// Only meaningful when the cursor is not captured.
    pub mouse_y: f32,
    /// True for exactly one frame when the left mouse button is pressed
    /// while the cursor is not captured.
    pub left_click: bool,
    /// True while the left mouse button is held down (cursor not captured).
    /// Unlike `left_click` this stays true across frames until release, so a
    /// UI drag (e.g. a slider) can track the cursor for its whole duration.
    pub left_button_down: bool,
    /// True for exactly one frame when the right mouse button is pressed
    /// while the cursor is not captured. Used for contextual UI (e.g. a
    /// create-at-cursor menu); it never captures the cursor.
    pub right_click: bool,
    /// Live logical viewport size in pixels `[width, height]`. Used to map
    /// overlay (Screen-owned) UI between its fixed reference resolution and the
    /// window, so menus scale with the window and the cursor still hit-tests
    /// against the scaled controls. `[0.0, 0.0]` before the backend is ready.
    pub viewport: [f32; 2],
    /// True for exactly one frame when the HUD-toggle key (F1) is pressed.
    pub hud_toggle: bool,
    /// True for exactly one frame when Escape is pressed while the cursor is
    /// not captured (menu / UI worlds). Used to fire [KeyBinding](#keybinding)
    /// actions. In worlds that capture the cursor, Escape instead releases the
    /// cursor and this stays false.
    pub escape: bool,
    /// The canonical key pressed this frame, for one frame, or `None`. Surfaced
    /// regardless of menu state (unlike the gameplay keys, which freeze while a
    /// menu is open) so the settings menu can capture a key for rebinding and
    /// scrollable menu lists can react to the arrow keys.
    pub captured_key: Option<crate::assets::Key>,
    /// The printable character typed this frame (Unicode, with shift / layout
    /// applied by the OS), for text-input fields, or `None`. A one-frame pulse
    /// surfaced regardless of menu state, like
    /// [captured_key](#structfield.captured_key). Editing keys (Backspace,
    /// Delete, arrows) arrive via `captured_key`, not here.
    pub typed_char: Option<char>,
}

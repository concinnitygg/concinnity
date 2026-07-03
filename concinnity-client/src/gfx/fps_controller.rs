// src/gfx/fps_controller.rs
//
// Stateful first-person controller scaffolding. Movement + look is currently
// driven by Camera3DSystem against the free functions in client/camera.rs;
// this struct exists for future input-state captures.

#[allow(dead_code)]
pub struct FpsController {
    // Movement speed in world units per second.
    pub move_speed: f32,
    // Mouse sensitivity in radians per pixel.
    pub mouse_sensitivity: f32,
    // Half-width of the player's body for wall collision (world units).
    pub player_radius: f32,

    last_cursor_x: f64,
    last_cursor_y: f64,
    // True after the first poll so the first large cursor jump is discarded.
    cursor_initialised: bool,
}

impl FpsController {
    #[allow(dead_code)]
    pub fn new(move_speed: f32, mouse_sensitivity: f32, player_radius: f32) -> Self {
        Self {
            move_speed,
            mouse_sensitivity,
            player_radius,
            last_cursor_x: 0.0,
            last_cursor_y: 0.0,
            cursor_initialised: false,
        }
    }
}

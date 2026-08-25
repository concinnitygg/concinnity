// src/components/controls_command.rs

/// Runtime-only event sent by GraphicsSystem when a live camera setting changes,
/// read by Camera3DSystem from its `Events<ControlsCommand>` queue.
///
/// These settings (mouse sensitivity, field of view) take effect on the camera,
/// not the renderer, so a change made in the settings menu is handed across as
/// this event rather than read from disk each frame: the camera updates its live
/// value on the same tick. Each field is an Option so one event can carry only
/// the value that changed (None leaves the rest untouched). World authors never
/// declare this type directly.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ControlsCommand {
    /// New mouse-look sensitivity, in radians per pixel. None leaves it
    /// unchanged. Applied live by the camera controller.
    pub mouse_sensitivity: Option<f32>,
    /// New camera vertical field of view, in degrees. None leaves it unchanged.
    /// Applied live to every Camera3D by the camera controller.
    pub fov_y_degrees: Option<f32>,
    /// New gamepad look sensitivity, in radians per second at full stick
    /// deflection. None leaves it unchanged. Applied live by the camera
    /// controller.
    pub gamepad_look_sensitivity: Option<f32>,
    /// New gamepad stick deadzone as a deflection fraction in [0, 1]. None
    /// leaves it unchanged. Applied live by the input sampling.
    pub gamepad_deadzone: Option<f32>,
    /// New gamepad action -> button map after a rebind. None leaves it
    /// unchanged. Applied live by the input sampling.
    pub gamepad_map: Option<crate::components::GamepadMap>,
}

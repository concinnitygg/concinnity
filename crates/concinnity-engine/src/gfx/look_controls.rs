// The look-sensitivity settings every camera controller shares. Both the
// first-person and third-person controllers seed them from the persisted store
// at init and fold live settings-menu changes in each tick; field of view rides
// the same paths but lives on `Camera3D`, so it travels back to the caller,
// which holds the mutable camera borrow.

use crate::components::{Camera3D, ControlsCommand};
use crate::ecs::{EventCursor, PipelineContext};

// The look values a controller carries, borrowed for the duration of an update.
pub(crate) struct Look<'a> {
    pub mouse_sensitivity: &'a mut f32,
    pub(crate) gamepad_look_sensitivity: &'a mut f32,
}

// Override the authored look values with the persisted settings-menu choices;
// `None` keeps the authored value. Applies a persisted FOV to every Camera3D
// directly, since no controller state holds it.
pub(crate) fn apply_persisted(ctx: &mut PipelineContext, look: Look) {
    let settings =
        crate::config::Settings::load(ctx.resource::<concinnity_host::store::paths::StateTree>());
    if let Some(s) = settings.controls.mouse_sensitivity {
        *look.mouse_sensitivity = s;
    }
    if let Some(s) = settings.controls.gamepad_look_sensitivity {
        *look.gamepad_look_sensitivity = s;
    }
    if let Some(fov) = settings.graphics.fov {
        for camera in ctx.query_mut::<Camera3D>() {
            camera.fov_y_degrees = fov;
        }
    }
}

// Fold this tick's ControlsCommand events into the look values. The last
// command this tick wins. Returns a pending field of view for the caller to
// write inside its own Camera3D loop.
pub(crate) fn drain_commands(
    ctx: &PipelineContext,
    cursor: &mut EventCursor,
    look: Look,
) -> Option<f32> {
    let mut pending_fov = None;
    if let Some(events) = ctx.events::<ControlsCommand>() {
        for cmd in events.read(cursor) {
            if let Some(s) = cmd.mouse_sensitivity {
                *look.mouse_sensitivity = s;
            }
            if let Some(s) = cmd.gamepad_look_sensitivity {
                *look.gamepad_look_sensitivity = s;
            }
            if let Some(fov) = cmd.fov_y_degrees {
                pending_fov = Some(fov);
            }
        }
    }
    pending_fov
}

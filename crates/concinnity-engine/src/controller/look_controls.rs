// The look-sensitivity settings every camera controller shares. Both the
// first-person and third-person controllers seed them from the persisted
// choices at init and fold live settings-menu changes in each tick; field of
// view rides the same paths but lives on `Camera3D`, so it travels back to the
// caller, which holds the mutable camera borrow.

use crate::components::{Camera3D, ControlsCommand};
use crate::ecs::{EventCursor, PipelineContext};
use crate::settings::StoredSettings;

// The look values a controller carries, borrowed for the duration of an update.
pub(super) struct Look<'a> {
    pub mouse_sensitivity: &'a mut f32,
    pub gamepad_look_sensitivity: &'a mut f32,
}

impl Look<'_> {
    // Take the look values `cmd` carries, and its field of view into `fov`.
    fn fold(&mut self, cmd: &ControlsCommand, fov: &mut Option<f32>) {
        if let Some(s) = cmd.mouse_sensitivity {
            *self.mouse_sensitivity = s;
        }
        if let Some(s) = cmd.gamepad_look_sensitivity {
            *self.gamepad_look_sensitivity = s;
        }
        if cmd.fov_y_degrees.is_some() {
            *fov = cmd.fov_y_degrees;
        }
    }
}

// Override the authored look values with the persisted settings-menu choices;
// `None` (or no published choices) keeps the authored value. Applies a
// persisted FOV to every Camera3D directly, since no controller state holds it.
pub(super) fn apply_persisted(ctx: &mut PipelineContext<'_>, mut look: Look<'_>) {
    let Some(persisted) = ctx
        .resource::<StoredSettings>()
        .map(StoredSettings::controls_command)
    else {
        return;
    };
    let mut fov = None;
    look.fold(&persisted, &mut fov);
    if let Some(fov) = fov {
        for camera in ctx.query_mut::<Camera3D>() {
            camera.fov_y_degrees = fov;
        }
    }
}

// Fold this tick's ControlsCommand events into the look values. The last
// command this tick wins. Returns a pending field of view for the caller to
// write inside its own Camera3D loop.
pub(super) fn drain_commands(
    ctx: &PipelineContext<'_>,
    cursor: &mut EventCursor,
    mut look: Look<'_>,
) -> Option<f32> {
    let mut pending_fov = None;
    if let Some(events) = ctx.events::<ControlsCommand>() {
        for cmd in events.read(cursor) {
            look.fold(cmd, &mut pending_fov);
        }
    }
    pending_fov
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::camera3d::Camera3DArgs;
    use crate::ecs::World;

    #[test]
    fn persisted_choices_override_the_authored_look() {
        let mut world = World::new();
        world.add_component(Camera3D::bake(Camera3DArgs::default()));
        let mut stored = StoredSettings::default();
        stored.controls.mouse_sensitivity = Some(0.004);
        stored.graphics.fov = Some(80.0);
        world.insert_resource(stored);
        let (mut mouse, mut pad) = (0.001, 2.5);
        apply_persisted(
            &mut world.context(),
            Look {
                mouse_sensitivity: &mut mouse,
                gamepad_look_sensitivity: &mut pad,
            },
        );
        assert_eq!((mouse, pad), (0.004, 2.5));
        let fov = world.query::<Camera3D>().next().map(|c| c.fov_y_degrees);
        assert_eq!(fov, Some(80.0));
    }

    #[test]
    fn no_published_choices_keep_the_authored_look() {
        let mut world = World::new();
        let (mut mouse, mut pad) = (0.001, 2.5);
        apply_persisted(
            &mut world.context(),
            Look {
                mouse_sensitivity: &mut mouse,
                gamepad_look_sensitivity: &mut pad,
            },
        );
        assert_eq!((mouse, pad), (0.001, 2.5));
    }

    // The last command of a tick wins, and a command without a field of view
    // leaves the pending one alone.
    #[test]
    fn the_last_command_of_a_tick_wins() {
        let mut world = World::new();
        let events = world.events_mut::<ControlsCommand>();
        events.send(ControlsCommand {
            mouse_sensitivity: Some(0.002),
            fov_y_degrees: Some(70.0),
            ..ControlsCommand::default()
        });
        events.send(ControlsCommand {
            mouse_sensitivity: Some(0.003),
            ..ControlsCommand::default()
        });
        let (mut mouse, mut pad) = (0.001, 2.5);
        let fov = drain_commands(
            &world.context(),
            &mut EventCursor::default(),
            Look {
                mouse_sensitivity: &mut mouse,
                gamepad_look_sensitivity: &mut pad,
            },
        );
        assert_eq!((mouse, pad, fov), (0.003, 2.5, Some(70.0)));
    }
}

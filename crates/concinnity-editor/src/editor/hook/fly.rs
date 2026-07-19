// src/editor/hook/fly.rs
//
// EditorHook: the edit-mode fly camera. Toggled with F (or the Preview
// panel's row); while on, the editor publishes the `FlyCam` resource so
// InputSystem keeps the navigation keys and mouse deltas live and the backend
// captures the cursor -- the world stays frozen behind the menu override, and
// this drive integrates Camera3D directly (free-fly: W flies where you look).
// Escape exits back to the free-cursor edit mode. The camera keeps its flown
// pose, so entering play mode afterwards continues from there.

use super::*;
use crate::assets::Camera3D;

const MOUSE_SENS: f32 = 0.003;
const FLY_SPEED: f32 = 6.0;
const SPRINT_MULTIPLIER: f32 = 4.0;
// A hitch (breakpoint, window drag) must not teleport the camera.
const MAX_DT: f32 = 0.1;

// One integration step of the fly camera, pure for testability: the new
// (position, yaw, pitch) for a frame's input over `dt` seconds.
pub(super) fn fly_step(
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    input: &FrameInput,
    dt: f32,
) -> ([f32; 3], f32, f32) {
    let yaw = yaw - input.mouse_dx * MOUSE_SENS;
    let pitch = (pitch - input.mouse_dy * MOUSE_SENS).clamp(
        -std::f32::consts::FRAC_PI_2 + 0.01,
        std::f32::consts::FRAC_PI_2 - 0.01,
    );
    // Free-fly basis: forward follows the pitch, right stays horizontal
    // (matching the engine's free-fly controller).
    let cp = pitch.cos();
    let fwd = [-yaw.sin() * cp, pitch.sin(), -yaw.cos() * cp];
    let right = [yaw.cos(), 0.0_f32, -yaw.sin()];
    let mut dir = [0.0_f32; 3];
    let mut add = |v: [f32; 3], sign: f32| {
        dir[0] += v[0] * sign;
        dir[1] += v[1] * sign;
        dir[2] += v[2] * sign;
    };
    if input.forward {
        add(fwd, 1.0);
    }
    if input.backward {
        add(fwd, -1.0);
    }
    if input.right {
        add(right, 1.0);
    }
    if input.left {
        add(right, -1.0);
    }
    let speed = if input.sprint {
        FLY_SPEED * SPRINT_MULTIPLIER
    } else {
        FLY_SPEED
    };
    let step = speed * dt.clamp(0.0, MAX_DT);
    (
        [
            position[0] + dir[0] * step,
            position[1] + dir[1] * step,
            position[2] + dir[2] * step,
        ],
        yaw,
        pitch,
    )
}

impl EditorHook {
    // Flip the fly camera. Entering leaves play mode (the two capture states
    // are exclusive); exiting drops the frame clock so re-entry starts fresh.
    pub(super) fn toggle_fly(&mut self) {
        self.fly = !self.fly;
        if self.fly {
            self.world_capture = false;
        }
        self.fly_clock = None;
    }

    // Per-frame fly drive: integrate Camera3D from the live input. The world
    // is frozen (Camera3DSystem included), so nothing fights the write; the
    // camera's view matrix is rebuilt in place, exactly as the engine
    // controller would.
    pub(super) fn drive_fly(&mut self, input: &FrameInput, world: &mut World) {
        if !self.fly {
            return;
        }
        let now = std::time::Instant::now();
        let dt = self
            .fly_clock
            .map(|last| now.duration_since(last).as_secs_f32())
            .unwrap_or(0.0);
        self.fly_clock = Some(now);
        let Some(cam) = world.query_mut::<Camera3D>().next() else {
            return;
        };
        let (position, yaw, pitch) = fly_step(cam.position, cam.yaw, cam.pitch, input, dt);
        cam.position = position;
        cam.yaw = yaw;
        cam.pitch = pitch;
        cam.view_matrix = concinnity_core::gfx::camera::view_matrix(position, yaw, pitch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fly_step_moves_along_the_look_direction() {
        // Yaw 0 faces -Z: W for a full clamp-window step flies speed*dt out.
        let input = FrameInput {
            forward: true,
            ..Default::default()
        };
        let (pos, yaw, pitch) = fly_step([0.0; 3], 0.0, 0.0, &input, MAX_DT);
        assert!((pos[2] + FLY_SPEED * MAX_DT).abs() < 1e-4, "{pos:?}");
        assert_eq!((yaw, pitch), (0.0, 0.0));

        // Pitched steeply up, W climbs.
        let (pos, _, _) = fly_step([0.0; 3], 0.0, 1.0, &input, MAX_DT);
        assert!(pos[1] > 0.4, "climbs with the pitch: {pos:?}");

        // Sprint scales the step.
        let sprint = FrameInput {
            forward: true,
            sprint: true,
            ..Default::default()
        };
        let (pos, _, _) = fly_step([0.0; 3], 0.0, 0.0, &sprint, MAX_DT);
        assert!((pos[2] + FLY_SPEED * SPRINT_MULTIPLIER * MAX_DT).abs() < 1e-3);
    }

    #[test]
    fn fly_step_looks_with_the_mouse_and_clamps_pitch() {
        let look = FrameInput {
            mouse_dx: 100.0,
            mouse_dy: -50.0,
            ..Default::default()
        };
        let (_, yaw, pitch) = fly_step([0.0; 3], 0.0, 0.0, &look, 0.016);
        assert!((yaw + 100.0 * MOUSE_SENS).abs() < 1e-5);
        assert!((pitch - 50.0 * MOUSE_SENS).abs() < 1e-5);

        // Pitch clamps just short of straight up/down.
        let dive = FrameInput {
            mouse_dy: 1e6,
            ..Default::default()
        };
        let (_, _, pitch) = fly_step([0.0; 3], 0.0, 0.0, &dive, 0.016);
        assert!(pitch > -std::f32::consts::FRAC_PI_2);
    }

    #[test]
    fn fly_step_clamps_a_hitch_dt() {
        let input = FrameInput {
            forward: true,
            ..Default::default()
        };
        let (pos, _, _) = fly_step([0.0; 3], 0.0, 0.0, &input, 10.0);
        assert!(
            (pos[2] + FLY_SPEED * MAX_DT).abs() < 1e-4,
            "a 10s hitch steps at most MAX_DT: {pos:?}"
        );
    }
}

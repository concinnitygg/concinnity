// src/editor/hook/orbit_drive.rs
//
// EditorHook: the Alt+drag tumble around the selection. The pivot is the
// selection's bounds center; angles and the camera's orientation offset are
// captured at press time (math in `editor/orbit.rs`) and stepped from the
// cursor's own frame-to-frame movement, so the drag needs neither cursor
// capture nor the gameplay input gate -- the cursor stays visible and the
// world's camera controller stays frozen.

use super::*;
use crate::assets::Camera3D;

pub(super) struct OrbitDrag {
    pivot: [f32; 3],
    dist: f32,
    // Spherical angles of the camera position around the pivot.
    yaw: f32,
    pitch: f32,
    // The camera orientation's constant offset from those angles.
    yaw_offset: f32,
    pitch_offset: f32,
    last_mouse: [f32; 2],
}

impl EditorHook {
    // Begin a tumble on an Alt+press over the viewport. Returns false (letting
    // the press route normally) when the press is over UI or nothing is
    // selected.
    pub(super) fn try_begin_orbit(
        &mut self,
        input: &FrameInput,
        vp: [f32; 2],
        world: &World,
    ) -> bool {
        let (mx, my) = (input.mouse_x, input.mouse_y);
        if self.sim.playing() || my <= hud::BAR_H || self.over_open_panel(mx, my, vp) {
            return false;
        }
        let Some((mn, mx_b)) = self.selection_bounds(world) else {
            return false;
        };
        let Some(cam) = world.query::<Camera3D>().next() else {
            return false;
        };
        let (pivot, _) = framing::bounding_sphere(mn, mx_b);
        let offset = [
            cam.position[0] - pivot[0],
            cam.position[1] - pivot[1],
            cam.position[2] - pivot[2],
        ];
        let (dist, yaw, pitch) = orbit::spherical_from_offset(offset);
        if dist <= f32::EPSILON {
            return false;
        }
        self.glide = None;
        self.orbit = Some(OrbitDrag {
            pivot,
            dist,
            yaw,
            pitch,
            yaw_offset: cam.yaw - yaw,
            pitch_offset: cam.pitch - pitch,
            last_mouse: [mx, my],
        });
        true
    }

    // Step an in-flight tumble from the cursor's movement; release ends it.
    pub(super) fn drive_orbit(&mut self, input: &FrameInput, world: &mut World) {
        let Some(orbit_drag) = &mut self.orbit else {
            return;
        };
        if !input.left_button_down {
            self.orbit = None;
            return;
        }
        let (mx, my) = (input.mouse_x, input.mouse_y);
        let (dx, dy) = (mx - orbit_drag.last_mouse[0], my - orbit_drag.last_mouse[1]);
        orbit_drag.last_mouse = [mx, my];
        let (yaw, pitch) = orbit::apply_deltas(orbit_drag.yaw, orbit_drag.pitch, dx, dy);
        orbit_drag.yaw = yaw;
        orbit_drag.pitch = pitch;
        let position =
            orbit::position_from_spherical(orbit_drag.pivot, orbit_drag.dist, yaw, pitch);
        let (cam_yaw, cam_pitch) = (yaw + orbit_drag.yaw_offset, pitch + orbit_drag.pitch_offset);
        if let Some(cam) = world.query_mut::<Camera3D>().next() {
            cam.position = position;
            cam.yaw = cam_yaw;
            cam.pitch = cam_pitch;
            cam.view_matrix =
                concinnity_core::gfx::camera::view_matrix(position, cam_yaw, cam_pitch);
        }
    }
}

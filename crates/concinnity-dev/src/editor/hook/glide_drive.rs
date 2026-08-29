// src/editor/hook/glide_drive.rs
//
// EditorHook: frame-selected and the short eased camera glide it (and a
// bookmark recall) rides on. F moves the camera so the whole selection fits
// the view; the pose math lives in `editor/framing.rs`, this drive resolves
// the selection's world bounds and steps the interpolation.

use super::*;
use crate::components::{Camera3D, Transform};
use framing::CameraPose;

const GLIDE_SECS: f32 = 0.25;

// An in-flight camera glide: the fixed endpoints and the wall clock the
// normalized parameter is derived from.
pub(super) struct CameraGlide {
    pub(super) from: CameraPose,
    pub(super) to: CameraPose,
    pub(super) start: std::time::Instant,
}

impl EditorHook {
    // The selection's world-space bounds this frame: mesh members contribute
    // their PickIndex AABB, billboard-backed members (no geometry) their
    // seeded Transform position padded to a small box. `None` when nothing
    // resolves.
    pub(super) fn selection_bounds(&self, world: &World) -> Option<([f32; 3], [f32; 3])> {
        let index = world.resource::<crate::ecs::PickIndex>();
        let boxes = self.selection.iter().filter_map(|name| {
            let id = crate::ecs::asset_id::lookup(name)?;
            if let Some(e) = index.and_then(|i| i.entries.iter().find(|e| e.asset_id == id)) {
                return Some((e.bb_min, e.bb_max));
            }
            let entity = billboard_drive::entity_by_name(world, name)?;
            Some(framing::pad_point(world.get::<Transform>(entity)?.position))
        });
        framing::union_bounds(boxes)
    }

    // F: glide the camera so the selection's bounding sphere fills the view,
    // keeping the current look direction. Takes the viewport rather than the
    // frame input so panel presses (which resolve without input access) can
    // frame too.
    pub(super) fn frame_selection(&mut self, vp: [f32; 2], world: &World) {
        let Some((mn, mx)) = self.selection_bounds(world) else {
            return;
        };
        let Some(cam) = world.query::<Camera3D>().next() else {
            return;
        };
        let aspect = if vp[1] > 0.0 { vp[0] / vp[1] } else { 1.0 };
        let current = CameraPose {
            position: cam.position,
            yaw: cam.yaw,
            pitch: cam.pitch,
        };
        let (center, radius) = framing::bounding_sphere(mn, mx);
        let to = framing::frame_pose(
            &current,
            center,
            radius,
            cam.fov_y_degrees.to_radians(),
            aspect,
        );
        self.start_glide(current, to);
    }

    pub(super) fn start_glide(&mut self, from: CameraPose, to: CameraPose) {
        self.glide = Some(CameraGlide {
            from,
            to,
            start: std::time::Instant::now(),
        });
    }

    // Step an in-flight glide, writing the camera exactly as the fly drive
    // does. Any deliberate navigation input (only live while flying) hands
    // control back immediately.
    pub(super) fn drive_glide(&mut self, input: &FrameInput, world: &mut World) {
        let Some(glide) = &self.glide else {
            return;
        };
        let steered = input.forward
            || input.backward
            || input.left
            || input.right
            || input.mouse_dx != 0.0
            || input.mouse_dy != 0.0;
        if steered {
            self.end_glide();
            return;
        }
        let t = glide.start.elapsed().as_secs_f32() / GLIDE_SECS;
        let pose = framing::lerp_pose(&glide.from, &glide.to, framing::ease(t));
        if let Some(cam) = world.query_mut::<Camera3D>().next() {
            cam.position = pose.position;
            cam.yaw = pose.yaw;
            cam.pitch = pose.pitch;
            cam.view_matrix =
                concinnity_core::gfx::camera::view_matrix(pose.position, pose.yaw, pose.pitch);
        }
        if t >= 1.0 {
            self.end_glide();
        }
    }

    pub(super) fn end_glide(&mut self) {
        self.glide = None;
        // A fly re-entry must not integrate the glide's wall time as one step.
        self.fly_clock = None;
    }
}

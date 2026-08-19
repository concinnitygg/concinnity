// src/editor/hook/axes_drive.rs
//
// EditorHook: the world-origin axes drive. Builds this frame's axis lines
// (`editor/axes.rs`) sized to the live camera's far plane and hands them to the
// renderer's line pass. Unlike the overlay furniture the axes are world
// geometry, so there is nothing to place or hide here: an empty list is the
// off state, and the renderer drops the pass with it.

use super::*;
use crate::assets::Camera3D;
use concinnity_cpu::gfx::lines::Line;

impl EditorHook {
    // Append this frame's axis lines to the shared line buffer. Nothing while
    // the axes are toggled off, while the whole HUD is hidden (F1 clears the
    // viewport for a clean look), or before a camera exists to size them
    // against.
    pub(super) fn push_axis_lines(&self, world: &World, out: &mut Vec<Line>) {
        if !self.axes_visible || !self.hud_visible {
            return;
        }
        let Some(cam) = world.query::<Camera3D>().next() else {
            return;
        };
        axes::push_lines(out, cam.far);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world_with_camera() -> World {
        let mut world = World::new();
        world.add_component(Camera3D {
            position: [0.0; 3],
            view_matrix: concinnity_core::gfx::camera::view_matrix([0.0; 3], 0.0, 0.0),
            fov_y_degrees: 90.0,
            near: 0.05,
            far: 300.0,
            yaw: 0.0,
            pitch: 0.0,
            desired_move: [0.0; 3],
            jump_requested: false,
            interact_requested: false,
            controller: None,
        });
        world
    }

    fn hook() -> EditorHook {
        EditorHook::new("world.jsonl".to_string(), Vec::new())
    }

    fn axis_lines(h: &EditorHook, world: &World) -> Vec<Line> {
        let mut out = Vec::new();
        h.push_axis_lines(world, &mut out);
        out
    }

    #[test]
    fn axes_are_published_by_default() {
        let world = world_with_camera();
        let lines = axis_lines(&hook(), &world);
        assert_eq!(lines.len(), 6, "a solid + fading run per axis");
        // Sized to the camera's far plane, so the runs reach the horizon.
        assert!(lines.iter().any(|l| l.end[0] == 300.0));
    }

    #[test]
    fn toggling_the_axes_off_publishes_nothing() {
        let world = world_with_camera();
        let mut h = hook();
        h.axes_visible = false;
        assert!(axis_lines(&h, &world).is_empty());
    }

    #[test]
    fn hiding_the_hud_takes_the_axes_with_it() {
        let world = world_with_camera();
        let mut h = hook();
        h.hud_visible = false;
        assert!(axis_lines(&h, &world).is_empty());
    }

    #[test]
    fn a_world_without_a_camera_publishes_nothing() {
        assert!(axis_lines(&hook(), &World::new()).is_empty());
    }
}

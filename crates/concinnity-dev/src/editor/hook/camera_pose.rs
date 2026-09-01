// src/editor/hook/camera_pose.rs
//
// EditorHook: reading and writing the live world's camera pose. The first
// `Camera3D` is the one the renderer draws, so every editor drive that moves
// the view (the glide, a bookmark recall, the start screen's attract camera)
// goes through this one seam -- and writes the view matrix with the pose, as
// the engine's own controller does, so the frame draws from what was written.

use super::*;
use crate::components::Camera3D;
use framing::CameraPose;

pub(super) fn read(world: &World) -> Option<CameraPose> {
    let cam = world.query::<Camera3D>().next()?;
    Some(CameraPose {
        position: cam.position,
        yaw: cam.yaw,
        pitch: cam.pitch,
    })
}

pub(super) fn write(world: &mut World, pose: &CameraPose) {
    let Some(cam) = world.query_mut::<Camera3D>().next() else {
        return;
    };
    cam.position = pose.position;
    cam.yaw = pose.yaw;
    cam.pitch = pose.pitch;
    cam.view_matrix =
        concinnity_core::gfx::camera::view_matrix(pose.position, pose.yaw, pose.pitch);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_written_pose_reads_back_with_its_view_matrix() {
        let mut world = World::new();
        assert_eq!(read(&world), None, "no camera, no pose");
        // A write with no camera is a no-op rather than a panic.
        write(
            &mut world,
            &CameraPose {
                position: [1.0, 2.0, 3.0],
                yaw: 0.5,
                pitch: -0.2,
            },
        );

        world.add_component(Camera3D {
            fov_y_degrees: 60.0,
            near: 0.05,
            far: 200.0,
            view_matrix: [[0.0; 4]; 4],
            position: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            desired_move: [0.0; 3],
            jump_requested: false,
            interact_requested: false,
            controller: None,
        });
        let pose = CameraPose {
            position: [4.0, 5.0, 6.0],
            yaw: 0.7,
            pitch: -0.3,
        };
        write(&mut world, &pose);
        assert_eq!(read(&world), Some(pose));
        let cam = world.query::<Camera3D>().next().expect("the camera");
        assert_eq!(
            cam.view_matrix,
            concinnity_core::gfx::camera::view_matrix(pose.position, pose.yaw, pose.pitch)
        );
    }
}

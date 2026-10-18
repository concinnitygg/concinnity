//! The frame's camera at render resolution: its projection, the sub-pixel
//! jitter a temporal pass accumulates over, and what derives from both.

use crate::gfx::frustum::Frustum;
use crate::gfx::jitter;
use crate::render::depth::camera_projection;
use crate::transform::{Mat4, mat4_inverse, mat4_mul};

/// The camera the frame is planned for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraInputs {
    /// World-to-view matrix, column-major.
    pub view: Mat4,
    /// World-space position.
    pub position: [f32; 3],
    /// Vertical field of view in radians.
    pub fov_y_radians: f32,
    /// Near clip distance.
    pub near: f32,
    /// How far the camera sees, or `None` for no limit.
    pub view_distance: Option<f32>,
}

/// Where this frame's samples sit inside each pixel.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Jitter {
    /// No temporal pass accumulates: every frame samples pixel centers.
    #[default]
    None,
    /// The shared Halton (2, 3) cycle at the frame's phase.
    Halton,
    /// An upscaler's own sequence: its offset at the frame's phase, in
    /// render-pixel units.
    Upscaler([f32; 2]),
}

/// The frame's camera, derived once for every pass.
#[derive(Clone, Copy, Debug)]
pub struct FrameCamera {
    /// Render-target width over height.
    pub aspect: f32,
    /// The un-jittered projection.
    pub proj: Mat4,
    /// The un-jittered view-projection, which volumes, clusters and motion
    /// are laid along so they hold still under the jitter.
    pub cur_vp: Mat4,
    /// The view-projection the scene rasterizes with, jitter included.
    pub vp: Mat4,
    /// Inverse of `vp`, for reconstructing world positions from depth.
    pub inv_vp: Mat4,
    /// Inverse of `cur_vp`.
    pub inv_cur_vp: Mat4,
    /// The jitter offset in render-pixel units, zero without a temporal pass.
    pub jitter_px: [f32; 2],
    /// The culling frustum of `vp`.
    pub frustum: Frustum,
}

impl FrameCamera {
    /// The camera `inputs` describes at `size` render pixels, offset by
    /// `jitter` at phase `phase`.
    pub fn new(inputs: &CameraInputs, size: [u32; 2], jitter: Jitter, phase: u32) -> Self {
        let aspect = match size[1] {
            0 => 1.0,
            h => size[0] as f32 / h as f32,
        };
        let proj = camera_projection(inputs.fov_y_radians, aspect, inputs.near);
        let cur_vp = mat4_mul(proj, inputs.view);
        let jitter_px = match jitter {
            Jitter::None => [0.0; 2],
            Jitter::Halton => jitter::offset_in_cycle(phase, jitter::TAA_CYCLE),
            Jitter::Upscaler(offset) => offset,
        };
        let vp = match jitter {
            Jitter::None => cur_vp,
            _ => mat4_mul(
                jitter::jitter_projection(proj, jitter_px, size[0], size[1]),
                inputs.view,
            ),
        };
        Self {
            aspect,
            proj,
            cur_vp,
            vp,
            inv_vp: mat4_inverse(vp),
            inv_cur_vp: mat4_inverse(cur_vp),
            jitter_px,
            frustum: Frustum::from_camera(vp, inputs.view_distance),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::IDENTITY;

    fn inputs() -> CameraInputs {
        CameraInputs {
            view: IDENTITY,
            position: [0.0; 3],
            fov_y_radians: 1.0,
            near: 0.1,
            view_distance: None,
        }
    }

    #[test]
    fn without_jitter_both_view_projections_agree() {
        let cam = FrameCamera::new(&inputs(), [64, 32], Jitter::None, 3);
        assert_eq!(cam.vp, cam.cur_vp);
        assert_eq!(cam.jitter_px, [0.0; 2]);
        assert_eq!(cam.aspect, 2.0);
    }

    #[test]
    fn the_jitter_moves_only_the_rasterized_view_projection() {
        let still = FrameCamera::new(&inputs(), [64, 32], Jitter::None, 3);
        let halton = FrameCamera::new(&inputs(), [64, 32], Jitter::Halton, 3);
        assert_eq!(halton.cur_vp, still.cur_vp);
        assert_ne!(halton.vp, still.vp);
        assert_eq!(
            halton.jitter_px,
            jitter::offset_in_cycle(3, jitter::TAA_CYCLE)
        );
        let up = FrameCamera::new(&inputs(), [64, 32], Jitter::Upscaler([0.25, -0.25]), 3);
        assert_eq!(up.jitter_px, [0.25, -0.25]);
        assert_eq!(
            up.vp,
            mat4_mul(
                jitter::jitter_projection(still.proj, [0.25, -0.25], 64, 32),
                IDENTITY
            )
        );
    }

    #[test]
    fn a_zero_height_target_keeps_a_finite_aspect() {
        assert_eq!(
            FrameCamera::new(&inputs(), [64, 0], Jitter::None, 0).aspect,
            1.0
        );
    }
}

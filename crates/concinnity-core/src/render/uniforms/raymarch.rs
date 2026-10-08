//! The blocks the raymarched SDF volume pass binds.
//!
//! One declaration each, for all three backends: the three hosts bind the same
//! bytes at different slots, and the slots are the only thing they disagree
//! about. Before the pass was single-sourced these were three `#[repr(C)]`
//! copies apiece, kept in step by hand.

use super::{GBufferView, PassCamera};
use crate::components::sdf_volume::SDF_PARAMS_LEN;

/// Per-frame view inputs, 400 bytes. Every field after `inv_vp` is a `float4`
/// lane or smaller, which is what keeps the three targets agreeing: a `float3`
/// occupies 16 bytes in a Metal constant buffer and 12 on SPIR-V and DXIL.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct RaymarchView {
    /// View-projection matrix, column-major.
    pub vp: [[f32; 4]; 4],
    /// Inverse view-projection matrix, column-major.
    pub inv_vp: [[f32; 4]; 4],
    /// World-space camera position in `xyz`; `w` is padding.
    pub cam_pos: [f32; 4],
    /// Render-target size in pixels.
    pub viewport: [f32; 2],
    /// Seconds since the world started, available to the authored field.
    pub time: f32,
    /// Mip count of the bound IBL prefilter cube. 0 means no `EnvironmentMap`
    /// is bound and the ambient helper takes its hemispheric fallback.
    pub prefilter_mip_count: f32,
    /// Rows of the rotation taking a world direction into the environment
    /// cubemap's baked frame, so a volume's ambient turns with the sky.
    pub sky_rot: [[f32; 4]; 3],
    /// This frame's unjittered view-projection, which the G-buffer pre-pass
    /// reprojects a hit through for its motion.
    pub cur_vp: [[f32; 4]; 4],
    /// The previous frame's unjittered view-projection; `cur_vp` when nothing
    /// reads motion.
    pub prev_vp: [[f32; 4]; 4],
    /// The view matrix the pre-pass stores normals and linear depth in.
    pub view_mat: [[f32; 4]; 4],
}

impl RaymarchView {
    /// The view block for `camera`, with no motion: both reprojection matrices
    /// are `camera.vp`.
    pub fn new(camera: &PassCamera) -> Self {
        let c = camera.cam_pos;
        Self {
            vp: camera.vp,
            inv_vp: camera.inv_vp,
            cam_pos: [c[0], c[1], c[2], 0.0],
            viewport: camera.viewport,
            time: camera.time,
            prefilter_mip_count: camera.prefilter_mip_count,
            sky_rot: camera.sky_rot,
            cur_vp: camera.vp,
            prev_vp: camera.vp,
            view_mat: IDENTITY,
        }
    }

    /// The block the G-buffer pre-pass draws volumes with: rasterized through
    /// the pre-pass's jittered VP, reprojected and stored through the rest of
    /// `gbuffer`.
    pub fn for_gbuffer(camera: &PassCamera, gbuffer: &GBufferView) -> Self {
        Self {
            vp: gbuffer.jittered_vp,
            cur_vp: gbuffer.cur_vp,
            prev_vp: gbuffer.prev_vp,
            view_mat: gbuffer.view,
            ..Self::new(camera)
        }
    }
}

const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Per-volume uniforms, 176 bytes. `center` and `extent` each pair with the pad
/// that completes their `float4` lane, for the reason above.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct RaymarchVolumeUniforms {
    /// World-space center of the bounding box.
    pub center: [f32; 3],
    /// Padding completing the lane `center` opens.
    pub _pad0: f32,
    /// Half-widths of the bounding box.
    pub extent: [f32; 3],
    /// Padding completing the lane `extent` opens.
    pub _pad1: f32,
    /// `1 / max_gradient`; the cone-step scale factor.
    pub cone_ratio: f32,
    /// Per-volume march far clip, in meters.
    pub max_distance: f32,
    /// Per-volume step cap, clamped at load.
    pub max_steps: i32,
    /// Non-zero when the volume samples the shadow maps.
    pub receive_shadows: i32,
    /// The authored parameter block the field interprets.
    pub params: [f32; SDF_PARAMS_LEN],
}

/// Which cascade a shadow-caster draw targets, 16 bytes. A root constant on
/// DirectX, a push constant on Vulkan, and a bound buffer on Metal.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct RaymarchShadowCascade {
    /// Index into the cascade light view-projections.
    pub cascade_idx: u32,
    /// Padding to the 16-byte block the hosts allocate.
    pub _pad: [u32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    // The sizes the three hosts allocate for these blocks. The per-field
    // offsets are checked against the compiled shader's layout in
    // `concinnity-device/src/shader_layout/`, which is what catches a shader-side
    // spelling that lays out differently on one target than on the others.
    #[test]
    fn the_blocks_are_the_sizes_the_hosts_bind() {
        assert_eq!(size_of::<RaymarchView>(), 400);
        assert_eq!(size_of::<RaymarchVolumeUniforms>(), 176);
        assert_eq!(size_of::<RaymarchShadowCascade>(), 16);
    }

    #[test]
    fn the_view_block_carries_the_camera_with_a_zero_w() {
        let camera = super::super::view::test_camera();
        let view = RaymarchView::new(&camera);
        assert_eq!(view.vp, camera.vp);
        assert_eq!(view.inv_vp, camera.inv_vp);
        assert_eq!(view.cam_pos, [3.0, 4.0, 5.0, 0.0]);
        assert_eq!(view.viewport, camera.viewport);
        assert_eq!(view.time, camera.time);
        assert_eq!(view.prefilter_mip_count, camera.prefilter_mip_count);
        assert_eq!(view.sky_rot, camera.sky_rot);
        assert_eq!(view.cur_vp, camera.vp);
        assert_eq!(view.prev_vp, camera.vp);
        assert_eq!(view.view_mat, IDENTITY);
    }

    #[test]
    fn the_gbuffer_block_rasterizes_jittered_and_reprojects_unjittered() {
        use crate::render::view_history::ViewFrame;
        let m = |k: f32| [[k; 4]; 4];
        let frame = |k: f32| ViewFrame {
            vp: m(k),
            elapsed: 0.0,
            cam_pos: [0.0; 3],
        };
        let gbuffer = GBufferView::new(m(10.0), m(11.0), frame(12.0), frame(13.0), true);
        let camera = super::super::view::test_camera();
        let view = RaymarchView::for_gbuffer(&camera, &gbuffer);
        assert_eq!(view.vp, m(10.0));
        assert_eq!(view.view_mat, m(11.0));
        assert_eq!(view.cur_vp, m(12.0));
        assert_eq!(view.prev_vp, m(13.0));
        assert_eq!(view.inv_vp, camera.inv_vp);
        assert_eq!(view.time, camera.time);
    }
}

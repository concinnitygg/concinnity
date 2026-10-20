//! One frame's mirror view of a reflector plane: the reflected camera, the
//! cull frustum narrowed to the plane's crop, the cluster grid binned over the
//! reflected view, and the view block the mirror render reads.

use crate::gfx::frustum::Frustum;
use crate::gfx::render_types::ClusterParams;
use crate::render::light::cluster_range::ClusterReach;
use crate::render::planar_reflection::{
    PLANAR_CLIP_BIAS, PixelRect, PlanarMatrices, orient_plane_toward, planar_matrices,
};
use crate::render::uniforms::{FaceCamera, ProbeUniforms};
use crate::transform::{Mat4, mat4_inverse, mat4_mul};

/// The main camera every mirror view of a frame reflects, and the mirror
/// targets' size.
#[derive(Clone, Copy, Debug)]
pub struct MirrorFrame {
    view: Mat4,
    proj: Mat4,
    cam_pos: [f32; 3],
    view_distance: Option<f32>,
    width: u32,
    height: u32,
}

impl MirrorFrame {
    /// The frame seen through `view` and `view_proj`, whose (jittered)
    /// projection the mirror renders share so a reflection lines up with the
    /// reflective fragment's screen-space sample. Mirror targets are
    /// `width` x `height`.
    pub fn new(
        view: Mat4,
        view_proj: Mat4,
        cam_pos: [f32; 3],
        view_distance: Option<f32>,
        (width, height): (u32, u32),
    ) -> Self {
        Self {
            view,
            proj: mat4_mul(view_proj, mat4_inverse(view)),
            cam_pos,
            view_distance,
            width,
            height,
        }
    }
}

/// What a mirror's view block takes beyond the reflected camera.
#[derive(Clone, Copy, Debug)]
pub struct MirrorShading {
    /// Elapsed seconds the mirror's shaders animate at.
    pub elapsed: f32,
    /// Mip count of the bound IBL prefilter cube.
    pub prefilter_mip_count: f32,
    /// Rows of the rotation into the environment cubemaps' frame.
    pub sky_rot: [[f32; 4]; 3],
    /// 1 when the irradiance cube holds diffuse light, else 0.
    pub irradiance: f32,
    /// The scale the mirror target is written at.
    pub pre_exposure: f32,
}

/// The camera reflected across one plane for one frame.
#[derive(Clone, Copy, Debug)]
pub struct MirrorView {
    /// The reflected view, oblique projection and eye.
    pub matrices: PlanarMatrices,
    /// The reflected frustum narrowed to the plane's crop. The oblique clip
    /// rides the view-projection, so it also rejects geometry behind the
    /// reflector.
    pub frustum: Frustum,
}

impl MirrorView {
    /// The view of `frame` reflected across `plane`, oriented toward the
    /// camera so the oblique near-plane clip keeps the camera's side, and
    /// culled to `crop` of the mirror target.
    pub fn new(frame: &MirrorFrame, plane: [f32; 4], crop: PixelRect) -> Self {
        let oriented = orient_plane_toward(plane, frame.cam_pos);
        let matrices = planar_matrices(
            frame.view,
            frame.proj,
            frame.cam_pos,
            oriented,
            PLANAR_CLIP_BIAS,
        );
        let frustum = Frustum::from_camera(
            crop.crop_view_projection(matrices.view_proj, frame.width, frame.height),
            frame.view_distance,
        );
        Self { matrices, frustum }
    }

    /// `base` (the main camera's cluster params) rebinned over the reflected
    /// view, reaching as far as `reach`'s lights and the reflection `probes`
    /// do. The mirror bins the same probes the main camera does.
    pub fn cluster_params(
        &self,
        frame: &MirrorFrame,
        base: &ClusterParams,
        reach: &ClusterReach,
        probes: &[ProbeUniforms],
    ) -> ClusterParams {
        let range = reach.range(self.matrices.eye, base.z_near, probes, frame.view_distance);
        base.with_camera(&self.matrices.cluster_camera(
            base.z_near,
            range,
            frame.width,
            frame.height,
        ))
    }

    /// The face camera the mirror's view block is built from.
    pub fn face_camera(&self, shading: MirrorShading) -> FaceCamera {
        FaceCamera {
            vp: self.matrices.view_proj,
            view: self.matrices.view,
            cam_pos: self.matrices.eye,
            elapsed: shading.elapsed,
            prefilter_mip_count: shading.prefilter_mip_count,
            sky_rot: shading.sky_rot,
            irradiance: shading.irradiance,
            pre_exposure: shading.pre_exposure,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::depth::camera_projection;
    use crate::transform::IDENTITY;

    fn frame(cam_pos: [f32; 3]) -> MirrorFrame {
        let mut view = IDENTITY;
        view[3] = [-cam_pos[0], -cam_pos[1], -cam_pos[2], 1.0];
        let proj = camera_projection(1.0, 1.0, 0.1);
        MirrorFrame::new(view, mat4_mul(proj, view), cam_pos, Some(500.0), (64, 32))
    }

    fn full(frame: &MirrorFrame) -> PixelRect {
        PixelRect {
            x: 0,
            y: 0,
            width: frame.width,
            height: frame.height,
        }
    }

    #[test]
    fn the_frame_recovers_the_projection() {
        let f = frame([1.0, 2.0, 3.0]);
        let expected = camera_projection(1.0, 1.0, 0.1);
        for (got, want) in f.proj.iter().flatten().zip(expected.iter().flatten()) {
            assert!((got - want).abs() < 1e-4, "{:?}", f.proj);
        }
    }

    #[test]
    fn the_view_matches_the_planar_matrices_of_the_oriented_plane() {
        let f = frame([0.0, 5.0, 0.0]);
        // A water plane at y = 1 whose normal points away from the camera.
        let plane = [0.0, -1.0, 0.0, 1.0];
        let v = MirrorView::new(&f, plane, full(&f));
        let expected = planar_matrices(
            f.view,
            f.proj,
            f.cam_pos,
            orient_plane_toward(plane, f.cam_pos),
            PLANAR_CLIP_BIAS,
        );
        assert_eq!(v.matrices.view_proj, expected.view_proj);
        assert!(
            (v.matrices.eye[1] + 3.0).abs() < 1e-4,
            "{:?}",
            v.matrices.eye
        );
        let cam = v.face_camera(MirrorShading {
            elapsed: 2.0,
            prefilter_mip_count: 5.0,
            sky_rot: [[0.0; 4]; 3],
            irradiance: 1.0,
            pre_exposure: 0.5,
        });
        assert_eq!(cam.cam_pos, v.matrices.eye);
        assert_eq!((cam.elapsed, cam.pre_exposure), (2.0, 0.5));
    }

    #[test]
    fn the_mirror_bins_the_main_cameras_probes() {
        let f = frame([0.0, 5.0, 0.0]);
        let v = MirrorView::new(&f, [0.0, 1.0, 0.0, 0.0], full(&f));
        let base = ClusterParams::for_camera(
            &v.matrices.cluster_camera(0.1, 100.0, f.width, f.height),
            4,
            3,
        );
        let params = v.cluster_params(&f, &base, &ClusterReach::default(), &[]);
        assert_eq!(params.num_probes, 3);
        assert_eq!((params.screen_w, params.screen_h), (64.0, 32.0));
    }
}

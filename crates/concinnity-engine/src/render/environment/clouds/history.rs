//! What the clouds carry from one frame to the next: how the previous frame
//! projected the sky, so the target texels a frame does not march are taken
//! from where the previous frame saw them, and whether the panorama already
//! holds the clouds around the camera, so a frame blends its rays into it.

use super::{CloudParams, cloud_target_size};
use crate::render::environment::atmosphere::{AtmosphereFrame, AtmosphereSettings, AtmosphereView};
use crate::transform::{Mat4, mat4_inverse, mat4_mul};
use concinnity_core::math::vec3::{length, sub};

// The share of a panorama texel a frame's ray replaces once the panorama is
// filled: each texel is marched every fourth frame, so it settles over about
// sixteen frames on the clouds filtered across it.
const PANORAMA_SHARE: f32 = 0.25;

// How far the camera may move in one frame, as a share of its height over
// the ground plus a kilometer, before the panorama is marched afresh rather
// than blended toward.
const PANORAMA_JUMP: f32 = 0.1;

/// How one frame projects the sky: directions from its camera, in the planet's
/// fixed axes, onto its clip space, and where that camera was.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CloudProjection {
    /// Takes a vector from the camera in the planet's fixed axes (any
    /// length, `w` = 1) to clip space; only its direction moves the result's
    /// `xy / w`.
    pub view: Mat4,
    /// The camera relative to the planet's center, kilometers, in its fixed
    /// axes.
    pub camera_km: [f32; 3],
}

impl CloudProjection {
    /// The projection of a view whose un-jittered view-projection inverts to
    /// `inv_vp`, its camera at world `cam_pos`, the frame's axes turning into
    /// the planet's fixed ones by the rows `to_fixed`, the camera `camera_km`
    /// from the planet's center in those axes.
    pub fn new(
        inv_vp: Mat4,
        cam_pos: [f32; 3],
        to_fixed: [[f32; 3]; 3],
        camera_km: [f32; 3],
    ) -> Self {
        let vp = mat4_inverse(inv_vp);
        let [x, y, z] = cam_pos;
        let at_camera = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [x, y, z, 1.0],
        ];
        // The rows of `to_fixed` are the fixed axes in the frame's, so as
        // columns they take a fixed-axes vector into the frame.
        let [a, b, c] = to_fixed;
        let from_fixed = [
            [a[0], a[1], a[2], 0.0],
            [b[0], b[1], b[2], 0.0],
            [c[0], c[1], c[2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        Self {
            view: mat4_mul(mat4_mul(vp, at_camera), from_fixed),
            camera_km,
        }
    }
}

// Where the camera was when the panorama last took in the clouds around it.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
struct PanoramaHistory {
    camera_km: Option<[f32; 3]>,
}

impl PanoramaHistory {
    // The share of each marched texel this frame's rays replace for a camera
    // `camera_km` from the planet's center over ground `ground_km` in radius:
    // `PANORAMA_SHARE` while the panorama holds the clouds around a camera
    // near it, else 1, marching it afresh.
    fn share(&mut self, camera_km: [f32; 3], ground_km: f32) -> f32 {
        let height = (length(camera_km) - ground_km).max(0.0);
        let near = self
            .camera_km
            .is_some_and(|last| length(sub(camera_km, last)) <= PANORAMA_JUMP * (height + 1.0));
        self.camera_km = Some(camera_km);
        if near { PANORAMA_SHARE } else { 1.0 }
    }

    // Forget the panorama, so the next frame marches it afresh.
    fn reset(&mut self) {
        self.camera_km = None;
    }
}

/// The previous frame's projection, kept from one frame to the next, and the
/// panorama's own history. A change of the target's size drops the
/// projection, since the history it addresses is gone.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct CloudHistory {
    last: Option<(CloudProjection, [u32; 2])>,
    panorama: PanoramaHistory,
}

impl CloudHistory {
    /// The previous frame's projection for a target `size` texels across and
    /// down, keeping `now` for the next frame.
    pub fn advance(&mut self, now: CloudProjection, size: [u32; 2]) -> Option<CloudProjection> {
        let previous = self
            .last
            .filter(|&(_, s)| s == size)
            .map(|(projection, _)| projection);
        self.last = Some((now, size));
        previous
    }

    /// Forget the previous frame and the panorama, as when a frame draws no
    /// clouds.
    pub fn reset(&mut self) {
        self.last = None;
        self.panorama.reset();
    }

    /// Frame `index`'s block for the clouds of `settings` in `frame`, seen
    /// from `view`, reprojecting through the previous frame's projection;
    /// `None`, and the previous frame forgotten, when the frame draws none.
    pub fn params(
        &mut self,
        settings: &AtmosphereSettings,
        frame: Option<&AtmosphereFrame>,
        view: &AtmosphereView,
        index: u64,
    ) -> Option<CloudParams> {
        let (Some(clouds), Some(cloud_frame)) = (
            settings.clouds.as_ref(),
            frame.and_then(|f| f.clouds.as_ref()),
        ) else {
            self.reset();
            return None;
        };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a pixel count"
        )]
        let scene = view.viewport.map(|v| v.max(1.0) as u32);
        let now = CloudProjection::new(
            view.inv_vp,
            view.cam_pos,
            cloud_frame.to_fixed,
            cloud_frame.camera_km,
        );
        let previous = self.advance(now, cloud_target_size(scene[0], scene[1]));
        let share = self
            .panorama
            .share(cloud_frame.camera_km, settings.bottom_radius);
        Some(
            CloudParams::new(clouds, settings.bottom_radius, cloud_frame, scene, index)
                .with_history(previous)
                .with_panorama_share(share),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::IDENTITY;
    use concinnity_core::approx::close;

    const TOL: f32 = 1e-4;

    // `m` applied to `(v, 1)`, divided through by `w`.
    fn project(m: &Mat4, v: [f32; 3]) -> [f32; 2] {
        let c: [f32; 4] =
            core::array::from_fn(|r| m[0][r] * v[0] + m[1][r] * v[1] + m[2][r] * v[2] + m[3][r]);
        [c[0] / c[3], c[1] / c[3]]
    }

    // A perspective projection looking down -z, camera at the origin.
    fn perspective() -> Mat4 {
        let (near, f) = (0.1f32, 1.0f32);
        [
            [f, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, near, 0.0],
        ]
    }

    // A camera standing anywhere sees a direction where its own projection
    // puts the point that far along it, however far.
    #[test]
    fn a_direction_lands_where_the_view_puts_it() {
        let cam = [10.0, -3.0, 7.0];
        let vp = mat4_mul(
            perspective(),
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-cam[0], -cam[1], -cam[2], 1.0],
            ],
        );
        let rows = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let p = CloudProjection::new(mat4_inverse(vp), cam, rows, [0.0; 3]);
        let d = [0.3, 0.2, -1.0];
        let world = [cam[0] + d[0], cam[1] + d[1], cam[2] + d[2]];
        let want = project(&vp, world);
        for scale in [0.5f32, 1.0, 40.0] {
            let got = project(&p.view, d.map(|c| c * scale));
            assert!(close(got, want, TOL), "{scale}: {got:?} {want:?}");
        }
    }

    // Turning the fixed axes turns which direction lands where.
    #[test]
    fn the_fixed_axes_turn_the_direction() {
        let vp = perspective();
        // Fixed x is the frame's -z.
        let rows = [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let p = CloudProjection::new(mat4_inverse(vp), [0.0; 3], rows, [0.0; 3]);
        assert!(close(project(&p.view, [1.0, 0.0, 0.0]), [0.0, 0.0], TOL));
    }

    #[test]
    fn a_resize_or_reset_drops_the_previous_frame() {
        let a = CloudProjection {
            view: IDENTITY,
            camera_km: [1.0, 2.0, 3.0],
        };
        let b = CloudProjection {
            camera_km: [4.0, 5.0, 6.0],
            ..a
        };
        let mut h = CloudHistory::default();
        assert_eq!(h.advance(a, [8, 8]), None);
        assert_eq!(h.advance(b, [8, 8]), Some(a));
        assert_eq!(h.advance(a, [4, 4]), None);
        h.reset();
        assert_eq!(h.advance(a, [4, 4]), None);
    }

    // The first frame marches the panorama afresh, and later frames blend
    // into it while the camera moves less than a tenth of its height plus a
    // kilometer a frame.
    #[test]
    fn the_panorama_blends_unless_the_camera_jumps() {
        let ground = 50.0;
        let mut p = PanoramaHistory::default();
        assert_eq!(p.share([0.0, 100.0, 0.0], ground), 1.0);
        assert_eq!(p.share([0.0, 99.5, 0.3], ground), PANORAMA_SHARE);
        assert_eq!(p.share([0.0, 92.0, 0.3], ground), 1.0);
        assert_eq!(p.share([0.0, 92.0, 0.3], ground), PANORAMA_SHARE);
        let low = [0.0, ground + 0.002, 0.0];
        assert_eq!(p.share(low, ground), 1.0);
        assert_eq!(p.share([0.05, ground + 0.002, 0.0], ground), PANORAMA_SHARE);
        assert_eq!(p.share([0.3, ground + 0.002, 0.0], ground), 1.0);
        p.reset();
        assert_eq!(p.share([0.3, ground + 0.002, 0.0], ground), 1.0);
    }

    // A frame without clouds forgets the panorama.
    #[test]
    fn a_cloud_reset_marches_the_panorama_afresh() {
        let mut h = CloudHistory::default();
        h.panorama.share([0.0, 60.0, 0.0], 50.0);
        h.reset();
        assert_eq!(h.panorama.share([0.0, 60.0, 0.0], 50.0), 1.0);
    }
}

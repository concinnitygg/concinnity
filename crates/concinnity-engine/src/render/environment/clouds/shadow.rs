//! Where the clouds' shadow map lies: a square on the sphere through the middle
//! of the cloud layers, around the point under the camera, wider as the camera
//! climbs. Its center snaps to whole texels in the planet's fixed axes, so the
//! map's texels stay put on the clouds as the camera moves and the shadows do
//! not crawl.

use crate::gfx::render_types::CloudShadowLanes;
use concinnity_core::math::dvec3::{self, DVec3};
use concinnity_core::math::{narrow_f32, round_f64};

/// Texels along each edge of the clouds' shadow map. `CLOUD_SHADOW_SIZE` in
/// `cloud_types.hlsl`.
pub const SHADOW_SIZE: u32 = 256;

// The square's width, in kilometers, near the ground and as it grows with the
// camera's height, at most.
const WIDTH_KM: f64 = 30.0;
const WIDTH_PER_HEIGHT: f64 = 2.0;
const MAX_WIDTH_KM: f64 = 120.0;

/// The shadow map's square in the planet's fixed axes, kilometers from its
/// center.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CloudShadowSquare {
    /// The square's center, on the sphere of `radius_km`.
    pub center_km: [f32; 3],
    /// The square's east, a unit tangent at its center.
    pub east: [f32; 3],
    /// The square's north, a unit tangent at its center.
    pub north: [f32; 3],
    /// The square's width.
    pub width_km: f32,
    /// The radius of the sphere it lies on, through the layers' middle.
    pub radius_km: f32,
}

impl CloudShadowSquare {
    /// The square for a camera `camera_km` from the planet's center (fixed
    /// axes) over a ground `ground_km` in radius, its sphere `radius_km`.
    pub fn around(camera_km: DVec3, ground_km: f64, radius_km: f64) -> Self {
        let height = (dvec3::length(camera_km) - ground_km).max(0.0);
        let width = (WIDTH_KM + WIDTH_PER_HEIGHT * height).min(MAX_WIDTH_KM);
        let texel = width / f64::from(SHADOW_SIZE);
        let up = dvec3::normalize_or(camera_km, [0.0, 1.0, 0.0]);
        let under = dvec3::scale(up, radius_km);
        let snapped = under.map(|c| round_f64(c / texel) * texel);
        let normal = dvec3::normalize_or(snapped, up);
        // Any axis well away from the normal: the square's east is level with
        // it, so the basis only turns as the square's center moves.
        let axis = if normal[1].abs() < 0.9 {
            [0.0, 1.0, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        let east = dvec3::normalize_or(dvec3::cross(axis, normal), [1.0, 0.0, 0.0]);
        let north = dvec3::cross(normal, east);
        Self {
            center_km: dvec3::to_f32(dvec3::scale(normal, radius_km)),
            east: dvec3::to_f32(east),
            north: dvec3::to_f32(north),
            width_km: narrow_f32(width),
            radius_km: narrow_f32(radius_km),
        }
    }

    /// The square in the frame's axes and meters, for a frame whose rows
    /// `to_fixed` turn it into the fixed axes, with the camera at `camera_pos`
    /// in it and `camera_km` from the planet's center in the fixed axes; the
    /// shadows fall from the light toward `light` (the frame's axes).
    pub fn lanes(
        &self,
        to_fixed: &[DVec3; 3],
        camera_pos: [f32; 3],
        camera_km: DVec3,
        light: [f32; 3],
    ) -> CloudShadowLanes {
        // The frame's axes are the transpose of the fixed ones.
        let to_frame = |v: DVec3| -> DVec3 {
            (0..3).fold([0.0; 3], |acc, i| {
                dvec3::add(acc, dvec3::scale(to_fixed[i], v[i]))
            })
        };
        let offset_m = dvec3::scale(dvec3::sub(dvec3::from_f32(self.center_km), camera_km), 1e3);
        let center = dvec3::add(dvec3::from_f32(camera_pos), to_frame(offset_m));
        let lane = |v: DVec3, w: f32| {
            let [x, y, z] = dvec3::to_f32(v);
            [x, y, z, w]
        };
        let [lx, ly, lz] = light;
        CloudShadowLanes {
            center: lane(center, self.radius_km * 1e3),
            east: lane(
                to_frame(dvec3::from_f32(self.east)),
                1.0 / (self.width_km * 1e3),
            ),
            north: lane(to_frame(dvec3::from_f32(self.north)), 1.0),
            light: [lx, ly, lz, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::approx::close;
    use concinnity_core::math::vec3::{dot, length};

    const IDENTITY: [DVec3; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    #[test]
    fn the_square_lies_under_the_camera_on_its_sphere() {
        let s = CloudShadowSquare::around([0.0, 50.002, 0.0], 50.0, 51.2);
        assert!(close(length(s.center_km), 51.2, 1e-4));
        assert!(close(s.center_km, [0.0, 51.2, 0.0], 0.1));
        assert!(close(dot(s.east, s.north), 0.0, 1e-6));
        assert!(close(dot(s.east, s.center_km), 0.0, 1e-3));
        assert!(close(s.width_km, 30.004, 1e-3));
    }

    #[test]
    fn the_square_widens_as_the_camera_climbs_up_to_a_limit() {
        let at = |h: f64| CloudShadowSquare::around([0.0, 50.0 + h, 0.0], 50.0, 51.0).width_km;
        assert!(close(at(10.0), 50.0, 1e-3));
        assert!(close(at(150.0), 120.0, 1e-3));
    }

    // A small move keeps the square where it was, to the texel.
    #[test]
    fn the_center_snaps_to_whole_texels() {
        let a = CloudShadowSquare::around([0.0, 50.002, 0.0], 50.0, 51.2);
        let b = CloudShadowSquare::around([0.001, 50.002, 0.0], 50.0, 51.2);
        assert!(close(a.center_km, b.center_km, 1e-6));
        let far = CloudShadowSquare::around([1.0, 50.002, 0.0], 50.0, 51.2);
        assert!(!close(a.center_km, far.center_km, 1e-3));
    }

    #[test]
    fn the_lanes_put_the_square_in_the_frame() {
        let s = CloudShadowSquare::around([0.0, 50.002, 0.0], 50.0, 51.2);
        // The camera stands 2 m over the frame's origin, which is on the ground.
        let lanes = s.lanes(
            &IDENTITY,
            [0.0, 2.0, 0.0],
            [0.0, 50.002, 0.0],
            [0.0, 1.0, 0.0],
        );
        assert!(close(lanes.center[1], 1200.0, 1.0));
        assert!(close(lanes.center[3], 51_200.0, 1e-1));
        assert!(close(lanes.east[3], 1.0 / 30_004.0, 1e-10));
        assert!(close(lanes.north[3], 1.0, 0.0));
    }
}

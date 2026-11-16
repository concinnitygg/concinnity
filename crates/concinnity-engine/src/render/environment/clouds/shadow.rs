//! Where the clouds' shadow map lies: a square on the sphere through the middle
//! of the cloud layers, around the point under the camera, wider as the camera
//! climbs. The map lays the sphere out round the square's center by the
//! straight distance to it (a Lambert azimuthal equal-area projection), so its
//! texels all cover the same area of the sphere: at an angle `a` round it from
//! the center they stretch along the radius by `1 / cos(a / 2)` and narrow
//! across it by as much, 1.31 at 80 degrees. Its center snaps to whole texels
//! in the planet's fixed axes, so the map's texels stay put on the clouds as
//! the camera moves and the shadows do not crawl.

use crate::gfx::render_types::CloudShadowLanes;
use concinnity_core::math::dvec3::{self, DVec3};
use concinnity_core::math::{acos, narrow_f32, round_f64, sin};

/// Texels along each edge of the clouds' shadow map. `CLOUD_SHADOW_SIZE` in
/// `cloud_types.hlsl`.
pub const SHADOW_SIZE: u32 = 256;

/// The shadow map's texture: a square of [`SHADOW_SIZE`] texels per cloud
/// layer, the lower layer's over the higher's.
pub const SHADOW_MAP_SIZE: [u32; 2] = [SHADOW_SIZE, 2 * SHADOW_SIZE];

// The square's width, in kilometers, near the ground and as it grows with the
// camera's height, at most.
const WIDTH_KM: f64 = 30.0;
const WIDTH_PER_HEIGHT: f64 = 2.0;
const MAX_WIDTH_KM: f64 = 120.0;
// How far from the point under the camera, in degrees round the sphere, the
// square reaches at most when it covers the cap of the world in view.
const MAX_CAP_DEG: f32 = 80.0;

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
    /// The square's width on the map, whose distances from its center are
    /// straight distances through the sphere.
    pub width_km: f32,
    /// The radius of the sphere it lies on, through the layers' middle.
    pub radius_km: f32,
}

impl CloudShadowSquare {
    /// The square for a camera `camera_km` from the planet's center (fixed
    /// axes) over a ground `ground_km` in radius, its sphere `radius_km`.
    /// `cap` in `[0, 1]` is how far it grows to cover the cap of the world
    /// the camera sees, as from orbit, where every cloud in view casts its
    /// shadow at once.
    pub fn around(camera_km: DVec3, ground_km: f64, radius_km: f64, cap: f64) -> Self {
        let distance = dvec3::length(camera_km);
        let height = (distance - ground_km).max(0.0);
        let near = (WIDTH_KM + WIDTH_PER_HEIGHT * height).min(MAX_WIDTH_KM);
        let seen =
            acos(narrow_f32(ground_km / distance.max(ground_km))).min(MAX_CAP_DEG.to_radians());
        let whole = 4.0 * radius_km * f64::from(sin(0.5 * seen));
        let width = near + (whole.max(near) - near) * cap.clamp(0.0, 1.0);
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

    /// The cosine of the angle round the sphere from the square's center past
    /// which the map covers nothing: its corners'.
    pub fn reach_cosine(&self) -> f32 {
        let corner = core::f32::consts::SQRT_2 * 0.5 * self.width_km / self.radius_km.max(1e-6);
        (1.0 - 0.5 * corner * corner).max(-1.0)
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

// Where on the square's sphere the shadow map's point `uv` (each in [0, 1])
// lies, as `cloud_shadow_kernel` places its texels.
#[cfg(test)]
fn map_point(square: &CloudShadowSquare, uv: [f32; 2]) -> [f32; 3] {
    use concinnity_core::math::vec3::{add, normalize_or, scale};
    let [x, y] = uv.map(|v| (v - 0.5) * square.width_km);
    let chord = (x.hypot(y) / square.radius_km).min(2.0);
    let side = if chord > 0.0 {
        scale(
            add(scale(square.east, x), scale(square.north, y)),
            1.0 / (chord * square.radius_km),
        )
    } else {
        [0.0; 3]
    };
    let n = normalize_or(square.center_km, 1e-9, [0.0, 1.0, 0.0]);
    let c = 1.0 - 0.5 * chord * chord;
    let s = chord * (1.0 - 0.25 * chord * chord).max(0.0).sqrt();
    scale(add(scale(n, c), scale(side, s)), square.radius_km)
}

// The shadow map's point a point `q` on the square's sphere reads, as
// `cloud_shadow_point` and `cloud_shadow_at` find it.
#[cfg(test)]
fn map_uv(square: &CloudShadowSquare, q: [f32; 3]) -> [f32; 2] {
    use concinnity_core::math::vec3::{dot, length, sub};
    let side = [dot(q, square.east), dot(q, square.north)];
    let off = side[0].hypot(side[1]);
    let chord = length(sub(q, square.center_km));
    side.map(|v| if off > 0.0 { v * chord / off } else { 0.0 } / square.width_km + 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::approx::close;
    use concinnity_core::math::vec3::{dot, length};

    const IDENTITY: [DVec3; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    #[test]
    fn the_square_lies_under_the_camera_on_its_sphere() {
        let s = CloudShadowSquare::around([0.0, 50.002, 0.0], 50.0, 51.2, 0.0);
        assert!(close(length(s.center_km), 51.2, 1e-4));
        assert!(close(s.center_km, [0.0, 51.2, 0.0], 0.1));
        assert!(close(dot(s.east, s.north), 0.0, 1e-6));
        assert!(close(dot(s.east, s.center_km), 0.0, 1e-3));
        assert!(close(s.width_km, 30.004, 1e-3));
    }

    #[test]
    fn the_square_widens_as_the_camera_climbs_up_to_a_limit() {
        let at = |h: f64| CloudShadowSquare::around([0.0, 50.0 + h, 0.0], 50.0, 51.0, 0.0).width_km;
        assert!(close(at(10.0), 50.0, 1e-3));
        assert!(close(at(150.0), 120.0, 1e-3));
    }

    // Seen from high over a small world, the square grows to the cap in view
    // as far as `cap` asks, and not past 80 degrees round the sphere.
    #[test]
    fn from_orbit_the_square_covers_the_cap_in_view() {
        let at = |h: f64, cap: f64| {
            CloudShadowSquare::around([0.0, 50.0 + h, 0.0], 50.0, 51.0, cap).width_km
        };
        let whole = 4.0 * 51.0 * sin(0.5 * acos(0.25));
        assert!(close(at(150.0, 1.0), whole, 1e-2));
        assert!(close(at(150.0, 0.0), 120.0, 1e-3));
        assert!(close(at(150.0, 0.5), 60.0 + 0.5 * whole, 1e-2));
        let far = 4.0 * 51.0 * sin(40f32.to_radians());
        assert!(close(at(5000.0, 1.0), far, 1e-1));
        // Near the ground the cap is smaller than the square already is.
        assert!(close(at(0.002, 1.0), at(0.002, 0.0), 0.0));
    }

    // The clouds' light reads the shadow map where its kernel put each texel,
    // across the square and from orbit too.
    #[test]
    fn the_map_reads_back_where_its_kernel_put_it() {
        for (height, cap) in [(0.002, 0.0), (150.0, 1.0)] {
            let s = CloudShadowSquare::around([0.3, 50.0 + height, -0.2], 50.0, 51.2, cap);
            for uv in [[0.5, 0.5], [0.1, 0.8], [0.97, 0.03]] {
                let back = map_uv(&s, map_point(&s, uv));
                assert!(close(back, uv, 1e-4), "{height} {uv:?} {back:?}");
            }
        }
    }

    // From orbit the map's texels at an angle `a` round the sphere from its
    // center stretch along the radius by `1 / cos(a / 2)` and narrow across it
    // by as much, so each covers the same area.
    #[test]
    fn the_texels_cover_the_same_area() {
        let s = CloudShadowSquare::around([0.0, 200.0, 0.0], 50.0, 51.0, 1.0);
        let texel = 1.0 / SHADOW_SIZE as f32;
        let apart = |a: [f32; 2], b: [f32; 2]| {
            let (a, b) = (map_point(&s, a), map_point(&s, b));
            (dot(a, b) / (51.0 * 51.0)).clamp(-1.0, 1.0).acos() * 51.0
        };
        let size = s.width_km * texel;
        let center = apart([0.5, 0.5], [0.5 + texel, 0.5]);
        assert!(close(center, size, 1e-3 * size), "{center}");
        let edge = 0.02;
        let chord = (0.5 - edge) * s.width_km / 51.0;
        let half = (0.5 * chord).asin();
        let along = apart([edge, 0.5], [edge + texel, 0.5]);
        let across = apart([edge, 0.5], [edge, 0.5 + texel]);
        assert!(close(along, size / half.cos(), 1e-2 * size), "{along}");
        assert!(close(across, size * half.cos(), 1e-2 * size), "{across}");
        assert!(close(along * across, size * size, 2e-2 * size * size));
    }

    // Every point of the map lies within its reach, and a point just past a
    // corner lies beyond it.
    #[test]
    fn the_reach_takes_in_the_whole_map() {
        let s = CloudShadowSquare::around([0.0, 200.0, 0.0], 50.0, 51.0, 1.0);
        let n = [0.0, 1.0, 0.0];
        let reach = s.reach_cosine();
        for uv in [
            [0.5, 0.5],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.0],
        ] {
            let c = dot(map_point(&s, uv), n) / 51.0;
            assert!(c >= reach - 1e-5, "{uv:?}: {c} {reach}");
        }
        let past = dot(map_point(&s, [-0.01, -0.01]), n) / 51.0;
        assert!(past < reach, "{past} {reach}");
        // A map wider than the sphere reaches all of it.
        let mut wide = s;
        wide.width_km = 1000.0;
        assert!(close(wide.reach_cosine(), -1.0, 1e-6));
    }

    // A small move keeps the square where it was, to the texel.
    #[test]
    fn the_center_snaps_to_whole_texels() {
        let a = CloudShadowSquare::around([0.0, 50.002, 0.0], 50.0, 51.2, 0.0);
        let b = CloudShadowSquare::around([0.001, 50.002, 0.0], 50.0, 51.2, 0.0);
        assert!(close(a.center_km, b.center_km, 1e-6));
        let far = CloudShadowSquare::around([1.0, 50.002, 0.0], 50.0, 51.2, 0.0);
        assert!(!close(a.center_km, far.center_km, 1e-3));
    }

    #[test]
    fn the_lanes_put_the_square_in_the_frame() {
        let s = CloudShadowSquare::around([0.0, 50.002, 0.0], 50.0, 51.2, 0.0);
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

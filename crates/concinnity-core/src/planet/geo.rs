// Geographic coordinates on a planet: latitude, longitude and height, and the
// conversions between them and the simulated frame.

use super::PlanetFrame;
use super::dvec::{self, DVec3};
use super::shape::PlanetShape;
use crate::math::{Quat, dvec3};

// The north pole's direction from the center, in authored coordinates.
const NORTH: DVec3 = [0.0, 0.0, -1.0];
// Latitude 0, longitude 0: through the middle of the `+Y` cube face.
const PRIME: DVec3 = [0.0, 1.0, 0.0];
// East where the equator meets the prime meridian.
const EAST: DVec3 = [1.0, 0.0, 0.0];

/// Where a point is on a planet.
///
/// Conventions, all fixed to the planet's authored axes:
///
/// - The north pole lies along authored `-Z` from the center, so the equator
///   is the plane through the center square to `Z`.
/// - Latitude 0, longitude 0 lies along authored `+Y` from the center: the
///   middle of the `+Y` cube face, which is where the world starts and the
///   camera stands looking north (`-Z`) at yaw 0.
/// - Latitude is in degrees, `-90` (south) to `90` (north). Longitude is in
///   degrees, `-180` to `180`, growing eastward (toward authored `+X` at the
///   prime meridian). At either pole longitude is reported as 0.
/// - `altitude` is meters above the planet's radius, the height of its lowest
///   ground; `height_above_ground` is meters above the ground directly below,
///   the radius plus the terrain's height there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoPosition {
    /// Degrees north of the equator.
    pub latitude: f64,
    /// Degrees east of the prime meridian.
    pub longitude: f64,
    /// Meters above the planet's radius.
    pub altitude: f64,
    /// Meters above the ground directly below.
    pub height_above_ground: f64,
}

/// A placement on a planet by geographic coordinate: where, how high above the
/// ground, and which way it faces, for content placed on a planet whatever
/// frame the world is simulating in. See [`GeoPosition`] for the conventions.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GeoPose {
    /// Degrees north of the equator.
    pub latitude: f64,
    /// Degrees east of the prime meridian.
    pub longitude: f64,
    /// Meters above the ground directly below.
    pub height_above_ground: f64,
    /// Degrees clockwise from north about the local up, seen from above: 0
    /// faces north, 90 east. At a pole, north is taken as seen from the
    /// prime meridian just short of it.
    pub heading: f64,
}

/// The camera's geographic position and heading, published every step in a
/// planet world for whatever reads where on the planet the camera is (an
/// atmosphere, a map, a site's bounds).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraGeo {
    /// Where the camera is.
    pub position: GeoPosition,
    /// Degrees clockwise from north the camera looks, about the local up.
    pub heading: f64,
}

/// The unit direction from a planet's center at `latitude` / `longitude`
/// degrees, in authored coordinates.
pub fn geo_direction(latitude: f64, longitude: f64) -> DVec3 {
    let (lat, lon) = (latitude.to_radians(), longitude.to_radians());
    let around = dvec3::add(
        dvec3::scale(PRIME, libm::cos(lon)),
        dvec3::scale(EAST, libm::sin(lon)),
    );
    dvec3::add(
        dvec3::scale(around, libm::cos(lat)),
        dvec3::scale(NORTH, libm::sin(lat)),
    )
}

/// The `(latitude, longitude)` in degrees of the direction `dir` from a
/// planet's center (any length), in authored coordinates.
pub fn geo_lat_lon(dir: DVec3) -> (f64, f64) {
    let d = dvec3::normalize_or(dir, PRIME);
    let (e, p) = (dvec3::dot(d, EAST), dvec3::dot(d, PRIME));
    let around = libm::sqrt(e * e + p * p);
    // atan2 keeps its precision at the poles, where asin's slope blows up.
    let lat = libm::atan2(dvec3::dot(d, NORTH), around);
    let lon = if around < 1e-12 {
        0.0
    } else {
        libm::atan2(e, p)
    };
    (lat.to_degrees(), lon.to_degrees())
}

/// The `[east, north, up]` unit axes at the direction `up` from a planet's
/// center, in authored coordinates. At a pole east is authored `+X`.
pub fn geo_axes(up: DVec3) -> [DVec3; 3] {
    let up = dvec3::normalize_or(up, PRIME);
    let east = dvec3::normalize_or(dvec3::cross(NORTH, up), EAST);
    let north = dvec3::cross(up, east);
    [east, north, up]
}

impl PlanetShape {
    /// Where the authored point `p` is on the planet.
    pub fn geo_at(&self, p: DVec3) -> GeoPosition {
        let rel = dvec3::sub(p, self.center);
        let (latitude, longitude) = geo_lat_lon(rel);
        let altitude = dvec3::length(rel) - self.radius;
        let ground = self.height(dvec3::normalize_or(rel, PRIME));
        GeoPosition {
            latitude,
            longitude,
            altitude,
            height_above_ground: altitude - ground,
        }
    }

    /// The authored point `height_above_ground` meters over the ground at
    /// `latitude` / `longitude` degrees.
    pub fn geo_point(&self, latitude: f64, longitude: f64, height_above_ground: f64) -> DVec3 {
        let dir = geo_direction(latitude, longitude);
        let r = self.radius + self.height(dir) + height_above_ground;
        dvec3::add(self.center, dvec3::scale(dir, r))
    }
}

// The heading in degrees of the authored direction `forward` at the direction
// `up`, or 0 for a direction straight up or down.
fn heading_of(up: DVec3, forward: DVec3) -> f64 {
    let [east, north, _] = geo_axes(up);
    let (e, n) = (dvec3::dot(forward, east), dvec3::dot(forward, north));
    if e * e + n * n < 1e-24 {
        return 0.0;
    }
    let h = libm::atan2(e, n).to_degrees();
    if h < 0.0 { h + 360.0 } else { h }
}

// The level authored direction `heading` degrees clockwise from north at `up`.
fn heading_direction(up: DVec3, heading: f64) -> DVec3 {
    let [east, north, _] = geo_axes(up);
    let h = heading.to_radians();
    dvec3::add(
        dvec3::scale(north, libm::cos(h)),
        dvec3::scale(east, libm::sin(h)),
    )
}

impl PlanetFrame {
    /// Where local point `p` is on the planet.
    pub fn geo_at(&self, p: [f32; 3]) -> GeoPosition {
        self.shape.geo_at(self.frame.to_world(p))
    }

    /// The pose of something at local point `p` facing local direction
    /// `forward`, its heading taken from `forward` laid level.
    pub fn geo_pose(&self, p: [f32; 3], forward: [f32; 3]) -> GeoPose {
        let world = self.frame.to_world(p);
        let at = self.shape.geo_at(world);
        GeoPose {
            latitude: at.latitude,
            longitude: at.longitude,
            height_above_ground: at.height_above_ground,
            heading: heading_of(
                dvec3::sub(world, self.shape.center),
                self.frame.dir_to_world(forward),
            ),
        }
    }

    /// Where `pose` is in the simulated frame, and the rotation that turns
    /// something standing at the origin facing `-Z` with `+Y` up into it:
    /// standing on the planet's up there, facing its heading.
    pub fn local_pose(&self, pose: &GeoPose) -> ([f32; 3], Quat) {
        let world = self
            .shape
            .geo_point(pose.latitude, pose.longitude, pose.height_above_ground);
        let up = dvec3::sub(world, self.shape.center);
        let forward = heading_direction(up, pose.heading);
        let up = dvec3::normalize_or(up, PRIME);
        let right = dvec3::cross(forward, up);
        let local = |d: DVec3| dvec3::from_f32(self.frame.dir_to_local(d));
        let basis = [local(right), local(up), local(dvec3::scale(forward, -1.0))];
        (self.frame.to_local(world), dvec::quat_of(&basis))
    }

    /// The yaw that, at local point `p` and about the planet's up there (as
    /// [`crate::gfx::camera::view_matrix_about`] turns it), faces `heading`
    /// degrees clockwise from north.
    pub fn yaw_for_heading(&self, p: [f32; 3], heading: f64) -> f32 {
        let world = self.frame.to_world(p);
        let up_world = dvec3::sub(world, self.shape.center);
        let forward = self
            .frame
            .dir_to_local(heading_direction(up_world, heading));
        let up = self.frame.dir_to_local(up_world);
        let level = untilt(up, forward);
        libm::atan2f(-level[0], -level[2])
    }

    /// The heading in degrees a camera at local point `p` with `yaw` about
    /// the planet's up there looks along.
    pub fn heading_for_yaw(&self, p: [f32; 3], yaw: f32) -> f64 {
        let world = self.frame.to_world(p);
        let up_world = dvec3::sub(world, self.shape.center);
        let up = self.frame.dir_to_local(up_world);
        let level = [-libm::sinf(yaw), 0.0, -libm::cosf(yaw)];
        let forward = crate::gfx::camera::tilt_to_up(up, level);
        heading_of(up_world, self.frame.dir_to_world(forward))
    }
}

// `v`, given in the frame whose up is `up`, turned back into the frame whose
// up is `+Y`: the inverse of `tilt_to_up`.
fn untilt(up: [f32; 3], v: [f32; 3]) -> [f32; 3] {
    crate::gfx::camera::tilt_to_up([-up[0], up[1], -up[2]], v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::quat_rotate;
    use crate::planet::LocalFrame;

    fn shape() -> PlanetShape {
        PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 40.0,
            feature_size: 2_000.0,
            octaves: 6,
            seed: 9,
        }
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    // The angle between two longitudes, wrapped.
    fn lon_gap(a: f64, b: f64) -> f64 {
        ((a - b + 540.0).rem_euclid(360.0) - 180.0).abs()
    }

    #[test]
    fn the_reference_points_sit_where_the_conventions_say() {
        let at = |lat, lon| geo_direction(lat, lon);
        let near = |a: DVec3, b: DVec3| dvec3::length(dvec3::sub(a, b)) < 1e-12;
        assert!(near(at(0.0, 0.0), [0.0, 1.0, 0.0]), "prime meridian on +Y");
        assert!(near(at(90.0, 0.0), [0.0, 0.0, -1.0]), "north pole on -Z");
        assert!(near(at(-90.0, 0.0), [0.0, 0.0, 1.0]), "south pole on +Z");
        assert!(near(at(0.0, 90.0), [1.0, 0.0, 0.0]), "east on +X");
        assert!(near(at(0.0, -90.0), [-1.0, 0.0, 0.0]));
        assert!(near(at(0.0, 180.0), [0.0, -1.0, 0.0]), "the antimeridian");
        // The world's start, the planet's top, is latitude 0, longitude 0.
        let s = shape();
        let start = s.geo_at([0.0, 12.0, 0.0]);
        assert!(close(start.latitude, 0.0, 1e-12) && close(start.longitude, 0.0, 1e-12));
        assert!(close(start.altitude, 12.0, 1e-6));
    }

    #[test]
    fn directions_round_trip_everywhere() {
        for lat in [-89.9, -60.0, -12.5, 0.0, 33.3, 75.0, 89.99] {
            for lon in [-179.5, -90.0, -1.0, 0.0, 45.0, 120.0, 179.9] {
                let (la, lo) = geo_lat_lon(geo_direction(lat, lon));
                assert!(close(la, lat, 1e-9), "{lat} {lon}: {la}");
                assert!(lon_gap(lo, lon) < 1e-9, "{lat} {lon}: {lo}");
            }
        }
    }

    #[test]
    fn the_poles_report_longitude_zero_and_round_trip() {
        for lat in [90.0, -90.0] {
            let (la, lo) = geo_lat_lon(geo_direction(lat, 37.0));
            assert!(close(la, lat, 1e-9));
            assert_eq!(lo, 0.0);
        }
        let s = shape();
        let pole = s.geo_point(90.0, 0.0, 3.0);
        let back = s.geo_at(pole);
        assert!(close(back.latitude, 90.0, 1e-9));
        assert!(close(back.height_above_ground, 3.0, 1e-6));
    }

    // A point placed by latitude, longitude and height comes back with the
    // same three, and its altitude is the ground's height plus that.
    #[test]
    fn points_round_trip_through_the_ground() {
        let s = shape();
        for (lat, lon, h) in [(0.0, 0.0, 1.7), (12.0, -40.0, 0.0), (-55.0, 170.0, 300.0)] {
            let p = s.geo_point(lat, lon, h);
            let g = s.geo_at(p);
            assert!(close(g.latitude, lat, 1e-9) && lon_gap(g.longitude, lon) < 1e-9);
            assert!(close(g.height_above_ground, h, 1e-6), "{g:?}");
            let ground = s.height(geo_direction(lat, lon));
            assert!(close(g.altitude, ground + h, 1e-6));
            assert!((0.0..=40.0).contains(&ground));
        }
    }

    #[test]
    fn the_local_axes_are_east_north_up_and_right_handed() {
        let [e, n, u] = geo_axes([0.0, 1.0, 0.0]);
        assert_eq!(
            (e, n, u),
            ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0])
        );
        for (lat, lon) in [(30.0, 60.0), (-70.0, -120.0), (90.0, 0.0), (-90.0, 0.0)] {
            let [e, n, u] = geo_axes(geo_direction(lat, lon));
            let c = dvec3::cross(e, n);
            assert!(dvec3::length(dvec3::sub(c, u)) < 1e-12, "{lat} {lon}");
            assert!(dvec3::dot(e, n).abs() < 1e-12);
        }
        // Walking north along a meridian, the north axis points to the next
        // point along it.
        let a = geo_direction(10.0, 25.0);
        let b = geo_direction(10.001, 25.0);
        let [_, n, _] = geo_axes(a);
        let step = dvec3::normalize_or(dvec3::sub(b, a), [0.0; 3]);
        assert!(dvec3::dot(step, n) > 0.999_999);
    }

    fn moved_frame() -> PlanetFrame {
        let s = shape();
        // A frame two moves along, the way a long walk leaves it.
        let mut frame = LocalFrame::AUTHORED;
        for at in [[700.0, -5.0, -600.0], [300.0, -1.0, -900.0]] {
            let up = s.up_at(frame.to_world(at));
            frame = frame.recentered(at, up).0;
        }
        PlanetFrame { shape: s, frame }
    }

    // A pose placed in a moved frame comes back as itself, and its rotation
    // stands on the local up facing the heading.
    #[test]
    fn poses_round_trip_through_a_moved_frame() {
        let pf = moved_frame();
        let here = pf.geo_at([0.0; 3]);
        for heading in [0.0, 45.0, 90.0, 200.0, 359.0] {
            let pose = GeoPose {
                latitude: here.latitude + 0.003,
                longitude: here.longitude - 0.002,
                height_above_ground: 2.0,
                heading,
            };
            let (p, q) = pf.local_pose(&pose);
            let forward = quat_rotate(q, [0.0, 0.0, -1.0]);
            let back = pf.geo_pose(p, forward);
            assert!(close(back.latitude, pose.latitude, 1e-6), "{back:?}");
            assert!(lon_gap(back.longitude, pose.longitude) < 1e-6);
            assert!(close(back.height_above_ground, 2.0, 0.02));
            assert!(lon_gap(back.heading, heading) < 1e-3, "{heading}: {back:?}");
            let up = quat_rotate(q, [0.0, 1.0, 0.0]);
            let radial = pf.up_at(p);
            assert!(crate::math::vec3::dot(up, radial) > 0.999_99);
        }
    }

    // A camera's yaw and its heading convert both ways, in a moved frame too;
    // at the start, yaw 0 faces north and turning left (positive yaw) faces
    // west.
    #[test]
    fn yaw_and_heading_convert_both_ways() {
        let start = PlanetFrame {
            shape: shape(),
            frame: LocalFrame::AUTHORED,
        };
        assert!(lon_gap(start.heading_for_yaw([0.0; 3], 0.0), 0.0) < 1e-6);
        let west = start.heading_for_yaw([0.0; 3], core::f32::consts::FRAC_PI_2);
        assert!(lon_gap(west, 270.0) < 1e-4, "{west}");
        for pf in [start, moved_frame()] {
            for p in [[0.0, 2.0, 0.0], [400.0, -3.0, 250.0]] {
                for heading in [0.0, 30.0, 135.0, 300.0] {
                    let yaw = pf.yaw_for_heading(p, heading);
                    let back = pf.heading_for_yaw(p, yaw);
                    assert!(lon_gap(back, heading) < 1e-3, "{heading} {back}");
                }
            }
        }
    }
}

//! A planet sea's motion as the backends draw it. The host computes in f64
//! where the anchor (the camera, snapped to the grid on its face and dropped
//! onto the sphere) falls on each face it can see and hands the shader only
//! small numbers: per face, the rotation from the simulated frame into the
//! face's axes and every pattern's offset at the anchor, as for a flat ocean.

#[cfg(test)]
mod mirror;
#[cfg(test)]
mod shore_mirror;

use super::motion::BodyMotion;
use crate::planet::LocalFrame;
use crate::water::WaterBody;
use crate::water::sea::SeaFace;
use crate::water::sea::shore::{LatticePlace, SHORE_OCTAVES, SHORE_TRAINS, SURF_LACE_PERIOD};
use core::f64::consts::FRAC_PI_2;

use concinnity_core::math::{dvec3, narrow_f32};

/// The faces a sea's shader reads: the three around the corner of the cube
/// nearest the anchor, which hold everything within the horizon.
pub const SEA_FACES: usize = 3;

/// One face's view of the sea at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeaFaceMotion {
    /// The rotation from the simulated frame into the face's axes `[u,
    /// normal, v]`, one row per face axis; each row's `w` is the anchor's
    /// direction from the planet's center along that axis.
    pub rows: [[f32; 4]; 3],
    /// Every pattern's offset at the anchor's place on the face (the wave
    /// phases, the ripple and warp offsets and the cascade tile offsets).
    pub motion: BodyMotion,
}

/// Where a planet sea's shore waves are read from at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeaShoreMotion {
    /// The rotation from the simulated frame into the planet's authored
    /// axes, one row per axis; each row's `w` is the anchor's direction from
    /// the planet's center along that axis.
    pub rows: [[f32; 4]; 3],
    /// Each shore train's phase at the waterline, in `[0, tau)`.
    pub phases: [f32; SHORE_TRAINS],
    /// The anchor's place on the sea in the planet's axes, in meters, each
    /// axis wrapped into one [`SURF_LACE_PERIOD`] centered on zero.
    pub lace_origin: [f32; 3],
    /// Where the anchor's direction lies in each noise octave the shore
    /// reads ([`ShoreWaves::lattice_at`]).
    pub lattice: [LatticePlace; SHORE_OCTAVES],
}

/// What moves on a planet sea at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeaMotion {
    /// The point on the sea the grid is laid out around, in the simulated
    /// frame: under the camera, snapped to whole grid cells on its face.
    pub anchor: [f32; 3],
    /// The sea's radial up at the anchor, in the simulated frame.
    pub up: [f32; 3],
    /// The directions along the sea at the anchor the grid's `x` and `z`
    /// run, in the simulated frame: those its face's `u` and `v` grow along.
    pub tangents: [[f32; 3]; 2],
    /// Meters along the sea per meter of the grid: one over the face's
    /// stretch at the anchor, so the grid's vertices sit on the face's lattice,
    /// spread wider when the camera is too high for the grid to reach its
    /// horizon.
    pub grid_scale: f32,
    /// How far from the anchor along the sea the grid is drawn, in meters:
    /// past the horizon by the sea's crests, no farther.
    pub reach: f32,
    /// The faces around the corner nearest the anchor, one per axis.
    pub faces: [SeaFaceMotion; SEA_FACES],
    /// Where its shore waves are read from; `None` without them.
    pub shore: Option<SeaShoreMotion>,
}

impl WaterBody {
    /// A planet sea's motion `time` seconds into the world, for a camera at
    /// `camera` in the simulated `frame`; `None` for any other body.
    pub fn sea_motion_at(
        &self,
        time: f64,
        frame: &LocalFrame,
        camera: [f32; 3],
    ) -> Option<SeaMotion> {
        let sea = self.sea.as_ref()?;
        let sphere = &sea.sphere;
        let r = sphere.radius;
        let eye = dvec3::sub(frame.to_world(camera), sphere.center);
        let height = dvec3::length(eye) - r;
        let dir = dvec3::normalize_or(eye, [0.0, 1.0, 0.0]);

        let home = SeaFace::holding(dir);
        let a = home.unproject(r, sea.grid.snap(home.project(r, dir)));
        let anchor = frame.to_local(dvec3::add(sphere.center, dvec3::scale(a, r)));
        let [tu, tv] = home.tangents(a);
        let grid_scale = narrow_f32(1.0 / home.stretch(a));

        let faces = core::array::from_fn(|axis| {
            let face = SeaFace::of_axis(axis, a[axis] >= 0.0);
            let at = face.coords(a);
            let rows = core::array::from_fn(|i| {
                let [x, y, z] = frame.dir_to_local(face.axes()[i]);
                [x, y, z, narrow_f32(at[i])]
            });
            SeaFaceMotion {
                rows,
                motion: self.motion_at(time, face.project(r, a)),
            }
        });

        let shore = sea.shore.as_ref().map(|shore| SeaShoreMotion {
            rows: core::array::from_fn(|i| {
                let mut axis = [0.0; 3];
                axis[i] = 1.0;
                let [x, y, z] = frame.dir_to_local(axis);
                [x, y, z, narrow_f32(a[i])]
            }),
            phases: shore.phases_at(time).map(narrow_f32),
            lace_origin: a.map(|c| {
                let p = SURF_LACE_PERIOD;
                let x = c * r + 0.5 * p;
                narrow_f32(x - p * libm::floor(x / p) - 0.5 * p)
            }),
            lattice: shore.lattice_at(a),
        });

        // The arc from the anchor to the farthest sea in view: the camera's
        // horizon and the crests standing up beyond it.
        let crest = f64::from(self.reach()[0]);
        let horizon = |h: f64| {
            if h > 0.0 {
                libm::acos(r / (r + h))
            } else {
                0.0
            }
        };
        let margin = 4.0 * f64::from(sea.grid.inner_cell);
        let drawn = f64::from(sea.grid.reach()) * f64::from(grid_scale);
        let reach = if height > 0.0 {
            (1.02 * r * (horizon(height) + horizon(crest)) + margin).min(FRAC_PI_2 * r)
        } else {
            drawn
        };
        // A camera too high for the grid to reach its horizon spreads it wider.
        let spread = (reach / drawn).max(1.0);
        Some(SeaMotion {
            anchor,
            up: frame.dir_to_local(a),
            tangents: [frame.dir_to_local(tu), frame.dir_to_local(tv)],
            grid_scale: narrow_f32(f64::from(grid_scale) * spread),
            reach: narrow_f32(reach),
            faces,
            shore,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{WaterSurface, Wind};
    use crate::planet::PlanetShape;
    use crate::water::ocean::OceanQuality;
    use crate::wind::WindField;

    fn planet() -> PlanetShape {
        crate::components::Planet::default().shape()
    }

    fn sea_body() -> WaterBody {
        let surface = WaterSurface {
            fetch_meters: 1.0e5,
            ..Default::default()
        };
        let wind = WindField::new(Some(&Wind {
            strength: 9.0,
            ..Wind::default()
        }));
        let sea = crate::water::PlanetWater {
            planet: planet(),
            level: 12.0,
            surface: &surface,
        };
        let (mut bodies, _) = WaterBody::resolve_all([], &wind, OceanQuality::FULL, Some(sea));
        bodies.remove(0)
    }

    // The anchor lies on the sea under the camera, its up is radial, and the
    // grid's axes lie flat on the sea there.
    #[test]
    fn the_anchor_lies_on_the_sea_under_the_camera() {
        let body = sea_body();
        let sphere = body.sea.as_ref().unwrap().sphere;
        let frame = LocalFrame::AUTHORED;
        let camera = [3.2, 40.0, -7.9];
        let m = body.sea_motion_at(5.0, &frame, camera).unwrap();
        let a = frame.to_world(m.anchor);
        let off = dvec3::length(dvec3::sub(a, sphere.center)) - sphere.radius;
        assert!(off.abs() < 1e-2, "{off}");
        assert!((m.anchor[0] - camera[0]).abs() < 2.0 && (m.anchor[2] - camera[2]).abs() < 2.0);
        assert!((m.anchor[1] - 12.0).abs() < 0.01, "{:?}", m.anchor);
        let up = dvec3::from_f32(m.up);
        for t in m.tangents {
            assert!(dvec3::dot(dvec3::from_f32(t), up).abs() < 1e-6);
        }
        assert!((m.grid_scale - 1.0).abs() < 1e-6);
        // 28 m over the sea, the horizon is about 1.7 km out.
        assert!((1500.0..2500.0).contains(&m.reach), "{}", m.reach);
    }

    // The anchor is a point of the planet: a camera that moves within one
    // grid cell keeps it, wherever the frame is.
    #[test]
    fn the_anchor_snaps_to_the_face_lattice() {
        let body = sea_body();
        let frame = LocalFrame::AUTHORED;
        let cell = body.sea.as_ref().unwrap().grid.inner_cell;
        let a = body.sea_motion_at(0.0, &frame, [0.1, 30.0, 0.1]).unwrap();
        let b = body
            .sea_motion_at(0.0, &frame, [0.1 + 0.2 * cell, 31.0, 0.1])
            .unwrap();
        assert_eq!(a.anchor, b.anchor);
        let c = body
            .sea_motion_at(0.0, &frame, [0.1 + 1.0 * cell, 30.0, 0.1])
            .unwrap();
        assert_ne!(a.anchor, c.anchor);
    }

    // From orbit the grid spreads past its own reach to the horizon, which
    // lies most of a quarter turn round the sea.
    #[test]
    fn a_camera_in_orbit_spreads_the_grid_to_its_horizon() {
        let body = sea_body();
        let sea = body.sea.as_ref().unwrap();
        let r = sea.sphere.radius;
        let frame = LocalFrame::AUTHORED;
        let low = body.sea_motion_at(0.0, &frame, [0.0, 40.0, 0.0]).unwrap();
        let high = body
            .sea_motion_at(0.0, &frame, [0.0, narrow_f32(3.0 * r), 0.0])
            .unwrap();
        let horizon = libm::acos(r / (4.0 * r - 12.0)) * r;
        assert!(f64::from(high.reach) > horizon, "{} m", high.reach);
        assert!(f64::from(high.reach) <= FRAC_PI_2 * r);
        assert!(high.reach <= sea.grid.reach() * high.grid_scale * 1.0001);
        assert!(
            high.grid_scale > 1.5 * low.grid_scale,
            "{}",
            high.grid_scale
        );
    }

    // A flat body has no sea motion.
    #[test]
    fn only_a_planet_sea_has_sea_motion() {
        let pond = WaterBody::resolve(&WaterSurface::default(), &WindField::STILL);
        assert!(
            pond.sea_motion_at(0.0, &LocalFrame::AUTHORED, [0.0; 3])
                .is_none()
        );
    }
}

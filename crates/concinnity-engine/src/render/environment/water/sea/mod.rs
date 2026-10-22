//! A planet's sea: water all the way around the world's planet, drawn on the
//! ocean's camera-centered grid bent onto the sea's sphere. Its waves are laid
//! out on the planet rather than in the simulated frame, so they stay where
//! they are on the water when the frame moves to follow the camera: each cube
//! face around the planet carries its own copy of the sea ([`SeaFace`]), and
//! where two faces meet their copies blend across a band, so no seam shows.
//!
//! The host computes in f64 where the anchor (the camera, snapped to the
//! grid on its face and dropped onto the sphere) falls on each face it can
//! see and hands the shader only small numbers: per face, the rotation from
//! the simulated frame into the face's axes and every pattern's offset at the
//! anchor, as for a flat ocean.

mod face;
#[cfg(test)]
mod mirror;
pub mod shore;

use concinnity_core::math::{dvec3, narrow_f32};
pub use face::SeaFace;
use shore::{LatticePlace, SHORE_OCTAVES, SHORE_TRAINS, SURF_LACE_PERIOD, ShoreWaves};

use super::WaterBody;
use super::motion::BodyMotion;
use super::ocean::grid::OceanGrid;
use crate::planet::{DVec3, LocalFrame, PlanetShape};

/// The faces a sea's shader reads: the three around the corner of the cube
/// nearest the anchor, which hold everything within the horizon.
pub const SEA_FACES: usize = 3;

/// How far a sea's grid reaches from its anchor, in planet radii along the
/// sphere's tangent plane, before the stretch of its face: enough for the
/// horizon from about a quarter of a radius up.
const SEA_GRID_REACH_RADII: f32 = 0.6;

// The widest a face's stereographic map stretches, at the cube's corners:
// 2 / (1 + 1 / sqrt(3)).
const MAX_FACE_STRETCH: f32 = 1.268;

// Half the band two faces' seas blend across at their shared edge, in
// meters, and the share of the radius it is held to on a small planet.
const FACE_BLEND_METERS: f64 = 200.0;
const FACE_BLEND_RADII: f64 = 0.02;

/// The sphere a sea lies on: the world's planet radius raised by its sea
/// level, in the planet's authored frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeaSphere {
    /// The planet's center, authored.
    pub center: DVec3,
    /// The sea's radius, in meters.
    pub radius: f64,
}

impl SeaSphere {
    /// The sea `sea_level` meters above `planet`'s radius.
    pub fn new(planet: &PlanetShape, sea_level: f32) -> Self {
        Self {
            center: planet.center,
            radius: planet.radius + f64::from(sea_level),
        }
    }

    /// Half the band, in meters, two faces' seas blend across at their edge.
    pub fn blend_half_width(&self) -> f64 {
        FACE_BLEND_METERS.min(FACE_BLEND_RADII * self.radius)
    }

    /// How fast a face's share grows per unit of its margin (see
    /// [`SeaFace::shares`]): from 0 to 1 across the blend band.
    pub fn blend_rate(&self) -> f64 {
        self.radius / (2.0 * core::f64::consts::SQRT_2 * self.blend_half_width())
    }
}

/// A planet sea's share of a body: the sphere it lies on and the grid it is
/// drawn on, laid out in face meters around the anchor.
#[derive(Debug, Clone, PartialEq)]
pub struct SeaBody {
    /// The sphere.
    pub sphere: SeaSphere,
    /// The camera-centered grid, in face meters.
    pub grid: OceanGrid,
    /// The waves at its shores; `None` on a planet without coasts.
    pub shore: Option<ShoreWaves>,
}

impl SeaBody {
    /// The sea on `sphere` for a sea peaking at `peak_wavelength` meters.
    pub fn new(sphere: SeaSphere, peak_wavelength: f32) -> Self {
        let reach = SEA_GRID_REACH_RADII * MAX_FACE_STRETCH * narrow_f32(sphere.radius);
        Self {
            sphere,
            grid: OceanGrid::reaching(peak_wavelength, reach),
            shore: None,
        }
    }
}

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
    /// stretch at the anchor, so the grid's vertices sit on the face's lattice.
    pub grid_scale: f32,
    /// How far from the anchor along the tangent plane the grid is drawn, in
    /// meters: past the horizon by the sea's crests, no farther.
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

        let crest = f64::from(self.reach()[0]);
        let horizon = |h: f64| {
            if h > 0.0 {
                libm::sqrt(h * (2.0 * r + h))
            } else {
                0.0
            }
        };
        let seen = horizon(height) + horizon(crest);
        let margin = 4.0 * f64::from(sea.grid.inner_cell);
        let drawn = f64::from(sea.grid.reach()) * f64::from(grid_scale);
        let reach = if height > 0.0 {
            (1.02 * seen + margin).min(drawn)
        } else {
            drawn
        };
        Some(SeaMotion {
            anchor,
            up: frame.dir_to_local(a),
            tangents: [frame.dir_to_local(tu), frame.dir_to_local(tv)],
            grid_scale,
            reach: narrow_f32(reach.min(0.95 * r)),
            faces,
            shore,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{WaterShape, WaterSurface, Wind};
    use crate::render::environment::water::ocean::OceanQuality;
    use crate::render::environment::wind::WindField;

    fn planet() -> PlanetShape {
        crate::components::Planet::default().shape()
    }

    fn sea_body() -> WaterBody {
        let surface = WaterSurface {
            shape: WaterShape::PlanetSea,
            sea_level: 12.0,
            fetch_meters: 1.0e5,
            ..Default::default()
        };
        let wind = WindField::new(Some(&Wind {
            strength: 9.0,
            ..Wind::default()
        }));
        let (mut bodies, _) =
            WaterBody::resolve_all([&surface], &wind, OceanQuality::FULL, Some(&planet()));
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

//! A planet's sea: water all the way around the world's planet, drawn on the
//! ocean's camera-centered grid bent onto the sea's sphere. Its waves are laid
//! out on the planet rather than in the simulated frame, so they stay where
//! they are on the water when the frame moves to follow the camera: each cube
//! face around the planet carries its own copy of the sea ([`SeaFace`]), and
//! where two faces meet their copies blend across a band, so no seam shows.

mod face;
pub mod shore;
mod source;

use core::ops::Range;

use concinnity_core::math::{acos, narrow_f32};
pub use face::SeaFace;
use shore::ShoreWaves;
pub use source::{PlanetWater, split_water};

use super::ocean::grid::OceanGrid;
use crate::gfx::render_types::DAYLIGHT_COS_MARGIN;
use crate::planet::{DVec3, PlanetShape};
use crate::render::uniforms::DaylightBlock;

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
    /// The sea `level` meters above `planet`'s radius.
    pub fn new(planet: &PlanetShape, level: f32) -> Self {
        Self {
            center: planet.center,
            radius: planet.radius + f64::from(level),
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
    /// The most its waves carry a vertex sideways, in meters: the sea's own
    /// sway (see [`WaterBody::reach`](crate::water::WaterBody::reach)) and its
    /// shore's.
    pub sway: f32,
}

impl SeaBody {
    /// The sea on `sphere` for a sea peaking at `peak_wavelength` meters.
    pub fn new(sphere: SeaSphere, peak_wavelength: f32) -> Self {
        let reach = SEA_GRID_REACH_RADII * MAX_FACE_STRETCH * narrow_f32(sphere.radius);
        Self {
            sphere,
            grid: OceanGrid::reaching(peak_wavelength, reach),
            shore: None,
            sway: 0.0,
        }
    }

    /// The rings of the grid out from its anchor that a planet's own sky,
    /// under `daylight`, reaches no point of: all of them where no sky lights
    /// the view, none while the camera is high enough for it to reach
    /// everywhere. A ring's points lie no farther along the sea from the
    /// camera than across the grid, since a face's map only ever stretches the
    /// sea, the anchor lies within half a cell of the camera, and a wave moves
    /// a vertex at most [`Self::sway`].
    pub fn rings_without_sky(&self, daylight: &DaylightBlock) -> u32 {
        let rings = self.grid.rings();
        if daylight.center[3] <= 0.5 {
            return rings;
        }
        if daylight.camera_up[3] > 0.0 {
            return 0;
        }
        let reach = acos((daylight.reach[0] + DAYLIGHT_COS_MARGIN).min(1.0))
            * narrow_f32(self.sphere.radius);
        let slack = core::f32::consts::FRAC_1_SQRT_2 * self.grid.inner_cell + self.sway;
        (0..=rings)
            .rev()
            .find(|&k| {
                core::f32::consts::SQRT_2 * self.grid.line(i32::try_from(k).unwrap_or(i32::MAX))
                    + slack
                    <= reach
            })
            .unwrap_or(0)
    }

    /// The index ranges of the grid's ring mesh under `daylight`: the inner
    /// rings, which shade without the planet's own sky, then the rest, which
    /// shade with it.
    pub fn sky_split(&self, daylight: &DaylightBlock) -> [Range<u32>; 2] {
        let inner = self.grid.ring_indices(self.rings_without_sky(daylight));
        let all = self.grid.ring_indices(self.grid.rings());
        [0..inner, inner..all]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::math::cos;

    const RADIUS: f64 = 50_000.0;

    // A sea whose waves carry a vertex up to 6 m sideways.
    fn sea() -> SeaBody {
        let mut sea = SeaBody::new(
            SeaSphere {
                center: [0.0, -RADIUS, 0.0],
                radius: RADIUS,
            },
            56.0,
        );
        sea.sway = 6.0;
        sea
    }

    // A sea standing at a coast carries its shore's reach in its sway, beside
    // its own.
    #[test]
    fn a_coasts_sea_sways_by_its_shore_too() {
        use crate::components::WaterSurface;
        use crate::water::WaterBody;
        use crate::wind::WindField;
        let planet = shore::tests::coast_planet();
        let wind = WindField::new(Some(&crate::components::Wind {
            strength: 7.0,
            ..Default::default()
        }));
        let surface = WaterSurface::default();
        let sea = PlanetWater {
            planet,
            level: 0.0,
            surface: &surface,
        };
        let (bodies, _) = WaterBody::resolve_all(
            [],
            &wind,
            crate::water::ocean::OceanQuality::FULL,
            Some(sea),
        );
        let body = &bodies[0];
        let sea = body.sea.as_ref().expect("a planet sea");
        let shore = sea.shore.as_ref().expect("the coast planet has shores");
        let own = body.reach()[1];
        assert!(own > 0.0);
        assert!((sea.sway - own - narrow_f32(shore.max_shift())).abs() < 1e-4);
    }

    fn sky(camera_reach: f32) -> DaylightBlock {
        DaylightBlock {
            center: [0.0, -50_000.0, 0.0, 1.0],
            camera_up: [0.0, 1.0, 0.0, camera_reach],
            reach: [
                cos(0.5_f32.to_radians()),
                cos(3.0_f32.to_radians()),
                50_000.0,
                5_000.0,
            ],
            ..DaylightBlock::OFF
        }
    }

    // From the ground, the inner rings out to where the sky starts, less the
    // slack, shade without it; the sky reaches every point past them only.
    #[test]
    fn the_rings_the_sky_misses_end_short_of_where_it_starts() {
        let s = sea();
        let k = s.rings_without_sky(&sky(0.0));
        assert!(k > 0 && k < s.grid.rings(), "{k}");
        // 0.5 deg round a 50 km sea is 436 m.
        let start = 0.5_f32.to_radians() * 50_000.0;
        let corner = core::f32::consts::SQRT_2 * s.grid.line(i32::try_from(k).unwrap());
        let next = core::f32::consts::SQRT_2 * s.grid.line(i32::try_from(k + 1).unwrap());
        assert!(corner < start, "{corner} m reaches the sky at {start} m");
        let slack = core::f32::consts::FRAC_1_SQRT_2 * s.grid.inner_cell + s.sway;
        assert!(
            corner + slack <= start,
            "{corner} m plus {slack} m of slack"
        );
        assert!(next + slack > start, "{next} m left short");
        let [inner, outer] = s.sky_split(&sky(0.0));
        assert_eq!(inner, 0..s.grid.ring_indices(k));
        assert_eq!(outer.start, inner.end);
        assert_eq!(outer.end, s.grid.ring_indices(s.grid.rings()));
    }

    #[test]
    fn a_high_camera_draws_the_whole_sea_with_its_sky_and_none_without() {
        let s = sea();
        assert_eq!(s.rings_without_sky(&sky(0.2)), 0);
        assert_eq!(s.sky_split(&sky(0.2))[0], 0..0);
        assert_eq!(s.rings_without_sky(&DaylightBlock::OFF), s.grid.rings());
        assert!(s.sky_split(&DaylightBlock::OFF)[1].is_empty());
    }
}

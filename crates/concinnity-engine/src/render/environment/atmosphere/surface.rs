//! The ground the sky's cubes show below the horizon over a planet. The
//! sky-view table ends a ray that meets the ground on one gray albedo; the
//! reflections and ambient light a surface takes from the cubes should
//! instead see what lies around the camera: the sea, the beach, the meadow.
//!
//! [`SurfacePalette`] is what the planet's ground looks like, resolved once
//! when the world loads. [`SurfaceBelow`] samples it around the camera each
//! time the camera has moved far enough: a table of how much of the ground
//! is water, by direction around the camera and by how far below the horizon
//! the ground is seen, and one color for the land among it.

use crate::components::WaterSurface;
use crate::planet::{DVec3, PlanetFrame, PlanetShape};
use crate::render::environment::grass::PlanetGrass;
use crate::render::environment::ground::GroundCovers;
use crate::render::uniforms::ground::GroundRuleGpu;
use concinnity_core::math::dvec3;
use concinnity_core::math::vec3::{cross, dot, length, normalize_or};
use concinnity_core::math::{cos, narrow_f32};
use libm::sqrt;

/// Directions around the camera the table holds, evenly spaced from the
/// table's east toward its north. Matches `SURFACE_AZIMUTHS` in the shaders.
pub const SURFACE_AZIMUTHS: usize = 8;

/// Angles below the horizon the table holds per direction. Matches
/// `SURFACE_BANDS` in the shaders.
pub const SURFACE_BANDS: usize = 4;

/// The steepest band's angle below the horizon, in degrees; the bands step
/// geometrically from it to [`SURFACE_SHALLOWEST_DEG`]. Matches
/// `SURFACE_STEEPEST_DEG` in the shaders.
pub const SURFACE_STEEPEST_DEG: f32 = 50.0;

/// The shallowest band's angle below the horizon, in degrees. Matches
/// `SURFACE_SHALLOWEST_DEG` in the shaders.
pub const SURFACE_SHALLOWEST_DEG: f32 = 2.5;

/// Entries in the table: band by band, each [`SURFACE_AZIMUTHS`] long.
pub const SURFACE_TABLE_LEN: usize = SURFACE_AZIMUTHS * SURFACE_BANDS;

/// Most grass layers whose color the land takes; later ones are left out.
pub const SURFACE_GRASS_LAYERS: usize = 4;

// How far below sea level, in meters, the ground goes from half water to
// all water, and as far above it to dry land: a gentle beach then fades
// across tens of meters instead of flipping between samples.
const WATERLINE_SOFTNESS_M: f64 = 2.0;

// The least height above the ground or sea the table is sampled from, so
// the nearest band never shrinks onto the camera's own footprint.
const MIN_HEIGHT_M: f64 = 1.5;

// How far the camera may move, as a share of the nearest band's reach,
// before the table is sampled again.
const RESAMPLE_SHARE: f64 = 0.1;

// How far a table may drift from the one the sky cube was drawn with
// before the cube is redrawn: a share of water, and a channel of the land.
const SHARE_TOLERANCE: f32 = 0.1;
const LAND_TOLERANCE: f32 = 0.02;

// How far, in degrees, the table's east may turn in the rendered axes before
// the table is sampled again: the camera's up tilts as it crosses the planet,
// and a rebase turns the axes whole.
const EAST_TURN_DEG: f32 = 0.5;

/// What a planet's sea looks like from above.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeaLook {
    /// Height of the water above the planet's radius, in meters.
    pub level: f32,
    /// Reflectance of deep water's body.
    pub deep_color: [f32; 3],
    /// The water's Fresnel falloff exponent.
    pub fresnel_power: f32,
}

impl SeaLook {
    /// A sea `level` meters above the planet's radius looking like `water`.
    pub fn of(level: f32, water: &WaterSurface) -> Self {
        Self {
            level,
            deep_color: water.deep_color.map(|c| c.clamp(0.0, 1.0)),
            fresnel_power: water.fresnel_power.max(1e-3),
        }
    }
}

/// What a planet's ground looks like from place to place, as far as its
/// color seen from a distance: its material's tint, the covers laid over
/// it and the grass grown on it, each where its rule holds, and its sea.
/// Albedo textures are not sampled; their tints stand for them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfacePalette {
    /// The tint of the planet's own material.
    pub ground: [f32; 3],
    /// The covers laid over it.
    pub covers: GroundCovers,
    /// Each grass layer's rule and its blades' mean color; the first
    /// `grass_count` are grown.
    pub grass: [(GroundRuleGpu, [f32; 3]); SURFACE_GRASS_LAYERS],
    /// Grass layers grown.
    pub grass_count: usize,
    /// The sea, if the planet has one.
    pub sea: Option<SeaLook>,
}

impl SurfacePalette {
    /// The palette of ground tinted `ground` under `covers`, grown with
    /// `grass`, and flooded by `sea`.
    pub fn new(
        ground: [f32; 3],
        covers: GroundCovers,
        grass: Option<&PlanetGrass>,
        sea: Option<SeaLook>,
    ) -> Self {
        let mut out = Self {
            ground,
            covers,
            grass: [(GroundRuleGpu::EVERYWHERE, [0.0; 3]); SURFACE_GRASS_LAYERS],
            grass_count: 0,
            sea,
        };
        let layers = grass.map_or(&[][..], |g| &g.layers[..]);
        for (slot, layer) in out.grass.iter_mut().zip(layers) {
            let look = &layer.look;
            let mean = core::array::from_fn(|i| 0.5 * (look.root_color[i] + look.tip_color[i]));
            *slot = (layer.rule, mean);
            out.grass_count += 1;
        }
        out
    }

    /// The land's color at `altitude` meters above sea level on ground of
    /// steepness `steep`: the material's tint, each cover then each grass
    /// layer laid over it by how fully its rule holds.
    pub fn land_albedo(&self, altitude: f32, steep: f32) -> [f32; 3] {
        let count = (self.covers.count as usize).min(self.covers.covers.len());
        let covers = self.covers.covers[..count].iter().map(|c| (c.rule, c.tint));
        let grass = self.grass[..self.grass_count].iter().copied();
        covers
            .chain(grass)
            .fold(self.ground, |albedo, (rule, color)| {
                let w = rule.weight(altitude, steep, 0.0);
                core::array::from_fn(|i| albedo[i] + (color[i] - albedo[i]) * w)
            })
    }

    // The sea's height above the planet's radius; 0 without a sea, the
    // height ground rules measure altitude from.
    fn sea_level(&self) -> f64 {
        self.sea.map_or(0.0, |s| f64::from(s.level))
    }
}

/// Where the camera stands over a planet, and the direction across the
/// ground the table's directions start from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceView {
    /// The camera relative to the planet's center, in the planet's own axes,
    /// in meters.
    pub camera: DVec3,
    /// The table's east: level at the camera, in the planet's own axes.
    pub east: DVec3,
    /// The same direction in the axes the frame renders in.
    pub east_rendered: [f32; 3],
}

impl SurfaceView {
    /// The view from `cam_pos` in `planet`'s simulated frame. East is fixed
    /// to that frame, so it holds still while the camera moves.
    pub fn of(planet: &PlanetFrame, cam_pos: [f32; 3]) -> Self {
        let up = planet.up_at(cam_pos);
        let east = normalize_or(cross(up, [0.0, 0.0, 1.0]), 1e-3, [0.0; 3]);
        let east_rendered = if length(east) > 0.5 {
            east
        } else {
            normalize_or(cross([1.0, 0.0, 0.0], up), 1e-6, [1.0, 0.0, 0.0])
        };
        Self {
            camera: dvec3::sub(planet.frame.to_world(cam_pos), planet.shape.center),
            east: planet.frame.dir_to_world(east_rendered),
            east_rendered,
        }
    }
}

/// The ground around the camera as the sky's cubes see it below the
/// horizon. Carried to the shaders in the atmosphere's constant block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceBelow {
    /// The mean color of the land among the samples.
    pub land: [f32; 3],
    /// Reflectance of deep water's body.
    pub water: [f32; 3],
    /// The water's Fresnel falloff exponent.
    pub fresnel_power: f32,
    /// The table's east, in the axes the frame renders in.
    pub east: [f32; 3],
    /// How much of the ground is water, in [0, 1], band by band from the
    /// steepest, each band from east toward north.
    pub water_share: [f32; SURFACE_TABLE_LEN],
    // Where the table was sampled from, and how far the camera may move from
    // there before it is sampled again.
    sampled_at: DVec3,
    resample: f64,
}

impl SurfaceBelow {
    /// The table around `view` over `shape` looking like `palette`.
    pub fn sample(palette: &SurfacePalette, shape: &PlanetShape, view: &SurfaceView) -> Self {
        Self::sample_ground(palette, shape.radius, view, |dir| {
            let (point, normal) = shape.surface_and_normal(dir);
            let height = dvec3::length(dvec3::sub(point, shape.center)) - shape.radius;
            (height, normal)
        })
    }

    // The table over ground of `radius` whose height above it and normal in
    // direction `dir` are `ground(dir)`.
    fn sample_ground(
        palette: &SurfacePalette,
        radius: f64,
        view: &SurfaceView,
        ground: impl Fn(DVec3) -> (f64, DVec3),
    ) -> Self {
        let r = dvec3::length(view.camera).max(1e-3);
        let up = dvec3::scale(view.camera, 1.0 / r);
        let north = dvec3::cross(up, view.east);
        let sea_level = palette.sea_level();
        let below = ground(up).0.max(sea_level);
        let height = (r - radius - below).max(MIN_HEIGHT_M);
        let horizon = sqrt(height * (2.0 * radius + height));

        let mut water_share = [0.0; SURFACE_TABLE_LEN];
        let mut land = [0.0f64; 3];
        let mut land_weight = 0.0;
        let mut nearest = horizon;
        for band in 0..SURFACE_BANDS {
            let depression = f64::from(band_depression_deg(band)).to_radians();
            let reach = (height / libm::tan(depression)).min(horizon);
            nearest = nearest.min(reach);
            let (sin_a, cos_a) = libm::sincos(reach / radius);
            for azimuth in 0..SURFACE_AZIMUTHS {
                let phi = core::f64::consts::TAU * azimuth as f64 / SURFACE_AZIMUTHS as f64;
                let (sin_p, cos_p) = libm::sincos(phi);
                let across = dvec3::add(dvec3::scale(view.east, cos_p), dvec3::scale(north, sin_p));
                let dir = dvec3::add(dvec3::scale(up, cos_a), dvec3::scale(across, sin_a));
                let (ground_height, normal) = ground(dir);
                let altitude = ground_height - sea_level;
                let share = if palette.sea.is_some() {
                    (0.5 - altitude / (2.0 * WATERLINE_SOFTNESS_M)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let steep = sqrt((1.0 - dvec3::dot(normal, dir)).max(0.0));
                let albedo = palette.land_albedo(narrow_f32(altitude), narrow_f32(steep));
                for (sum, c) in land.iter_mut().zip(albedo) {
                    *sum += f64::from(c) * (1.0 - share);
                }
                land_weight += 1.0 - share;
                water_share[band * SURFACE_AZIMUTHS + azimuth] = narrow_f32(share);
            }
        }
        let land = if land_weight > 1e-3 {
            land.map(|c| narrow_f32(c / land_weight))
        } else {
            palette.ground
        };
        let sea = palette.sea;
        Self {
            land,
            water: sea.map_or([0.0; 3], |s| s.deep_color),
            fresnel_power: sea.map_or(1.0, |s| s.fresnel_power),
            east: view.east_rendered,
            water_share,
            sampled_at: view.camera,
            resample: nearest * RESAMPLE_SHARE,
        }
    }

    /// Whether the camera of `view` has moved far enough from where this
    /// table was sampled, or its east turned far enough, to sample it again.
    pub fn stale(&self, view: &SurfaceView) -> bool {
        dvec3::length(dvec3::sub(view.camera, self.sampled_at)) > self.resample
            || dot(self.east, view.east_rendered) < cos(EAST_TURN_DEG.to_radians())
    }

    /// Whether the ground has changed enough since `last` for the sky cube
    /// drawn over it to be redrawn.
    pub fn moved_from(&self, last: &Self) -> bool {
        let land = (0..3).any(|i| (self.land[i] - last.land[i]).abs() > LAND_TOLERANCE);
        let water = self
            .water_share
            .iter()
            .zip(&last.water_share)
            .any(|(a, b)| (a - b).abs() > SHARE_TOLERANCE);
        let turned = dot(self.east, last.east) < cos(EAST_TURN_DEG.to_radians());
        land || water || turned || self.water != last.water
    }

    /// The constant block's lanes: the land's color with a 1 marking the
    /// table in use, the water's with its Fresnel exponent, the table's
    /// east, and the shares four to a lane.
    pub fn lanes(
        &self,
    ) -> (
        [f32; 4],
        [f32; 4],
        [f32; 4],
        [[f32; 4]; SURFACE_TABLE_LEN / 4],
    ) {
        let [lr, lg, lb] = self.land;
        let [wr, wg, wb] = self.water;
        let [ex, ey, ez] = self.east;
        let shares =
            core::array::from_fn(|i| core::array::from_fn(|j| self.water_share[i * 4 + j]));
        (
            [lr, lg, lb, 1.0],
            [wr, wg, wb, self.fresnel_power],
            [ex, ey, ez, 0.0],
            shares,
        )
    }
}

/// Band `band`'s angle below the horizon, in degrees: geometric steps from
/// [`SURFACE_STEEPEST_DEG`] to [`SURFACE_SHALLOWEST_DEG`]. The shaders'
/// `surface_band_of` inverts it.
pub fn band_depression_deg(band: usize) -> f32 {
    let t = band as f32 / (SURFACE_BANDS - 1) as f32;
    SURFACE_STEEPEST_DEG * libm::powf(SURFACE_SHALLOWEST_DEG / SURFACE_STEEPEST_DEG, t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::uniforms::ground::GroundCoverGpu;

    const RADIUS: f64 = 50_000.0;

    fn sea() -> SeaLook {
        SeaLook {
            level: 0.0,
            deep_color: [0.02, 0.05, 0.15],
            fresnel_power: 5.0,
        }
    }

    fn palette(sea: Option<SeaLook>) -> SurfacePalette {
        SurfacePalette::new([0.1, 0.08, 0.05], GroundCovers::default(), None, sea)
    }

    // The camera `height` meters above the radius straight up the Y axis,
    // east along X.
    fn view(height: f64) -> SurfaceView {
        SurfaceView {
            camera: [0.0, RADIUS + height, 0.0],
            east: [1.0, 0.0, 0.0],
            east_rendered: [1.0, 0.0, 0.0],
        }
    }

    // Level ground `height` meters above the radius everywhere.
    fn level(height: f64) -> impl Fn(DVec3) -> (f64, DVec3) {
        move |dir| (height, dir)
    }

    #[test]
    fn the_bands_step_from_the_steepest_to_the_shallowest() {
        assert_eq!(band_depression_deg(0), SURFACE_STEEPEST_DEG);
        let last = band_depression_deg(SURFACE_BANDS - 1);
        assert!((last - SURFACE_SHALLOWEST_DEG).abs() < 1e-4, "{last}");
        let ratio = band_depression_deg(1) / band_depression_deg(0);
        assert!((band_depression_deg(2) / band_depression_deg(1) - ratio).abs() < 1e-4);
    }

    #[test]
    fn open_sea_is_all_water_and_dry_land_none() {
        let wet =
            SurfaceBelow::sample_ground(&palette(Some(sea())), RADIUS, &view(10.0), level(-20.0));
        assert!(
            wet.water_share.iter().all(|&s| s == 1.0),
            "{:?}",
            wet.water_share
        );
        assert_eq!(wet.water, sea().deep_color);
        assert_eq!(wet.fresnel_power, 5.0);

        let dry =
            SurfaceBelow::sample_ground(&palette(Some(sea())), RADIUS, &view(30.0), level(20.0));
        assert!(dry.water_share.iter().all(|&s| s == 0.0));
        assert_eq!(dry.land, [0.1, 0.08, 0.05]);
    }

    #[test]
    fn a_planet_without_a_sea_is_all_land() {
        let below = SurfaceBelow::sample_ground(&palette(None), RADIUS, &view(10.0), level(-20.0));
        assert!(below.water_share.iter().all(|&s| s == 0.0));
    }

    // Sea to the west of the camera, land to the east: the directions split
    // by side, each band alike.
    #[test]
    fn a_coast_splits_the_directions() {
        let coast = |dir: DVec3| (if dir[0] < 0.0 { -10.0 } else { 10.0 }, dir);
        let below = SurfaceBelow::sample_ground(&palette(Some(sea())), RADIUS, &view(5.0), coast);
        for band in 0..SURFACE_BANDS {
            let row = &below.water_share[band * SURFACE_AZIMUTHS..][..SURFACE_AZIMUTHS];
            // East (0) is land, west (half way round) is sea.
            assert_eq!(row[0], 0.0, "{row:?}");
            assert_eq!(row[SURFACE_AZIMUTHS / 2], 1.0, "{row:?}");
        }
    }

    // Each band's samples lie as far out as a ray that far below the horizon
    // meets level ground.
    #[test]
    fn the_bands_reach_where_their_rays_meet_the_ground() {
        let height = 10.0;
        let reached = core::cell::RefCell::new(alloc::vec::Vec::new());
        SurfaceBelow::sample_ground(&palette(None), RADIUS, &view(height), |dir| {
            reached.borrow_mut().push(libm::acos(dir[1]) * RADIUS);
            (0.0, dir)
        });
        let reached = reached.into_inner();
        for band in 0..SURFACE_BANDS {
            let depression = f64::from(band_depression_deg(band)).to_radians();
            let expected = height / libm::tan(depression);
            let got = reached[1 + band * SURFACE_AZIMUTHS];
            assert!(
                (got - expected).abs() < 1e-3 * expected,
                "{band}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn covers_and_grass_color_the_land_by_their_rules() {
        let sand = GroundCoverGpu {
            rule: GroundRuleGpu::EVERYWHERE,
            tint: [0.55, 0.47, 0.33],
            ..GroundCoverGpu::default()
        };
        let mut covers = GroundCovers::default();
        covers.covers[0] = sand;
        covers.count = 1;
        let mut p = SurfacePalette::new([0.1; 3], covers, None, None);
        assert_eq!(p.land_albedo(0.0, 0.0), sand.tint);

        let high = crate::components::GroundRule {
            altitude: [Some(2.5), None],
            altitude_edge: 0.6,
            ..Default::default()
        };
        p.grass[0] = (GroundRuleGpu::of(&high), [0.1, 0.4, 0.05]);
        p.grass_count = 1;
        assert_eq!(p.land_albedo(-5.0, 0.0), sand.tint);
        let grown = p.land_albedo(20.0, 0.0);
        assert!(
            grown
                .iter()
                .zip([0.1, 0.4, 0.05])
                .all(|(a, b)| (a - b).abs() < 1e-6),
            "{grown:?}"
        );
    }

    #[test]
    fn the_table_is_sampled_again_once_the_camera_moves() {
        let below =
            SurfaceBelow::sample_ground(&palette(Some(sea())), RADIUS, &view(10.0), level(-20.0));
        assert!(!below.stale(&view(10.0)));
        let mut moved = view(10.0);
        moved.camera[0] += 0.1;
        assert!(!below.stale(&moved));
        moved.camera[0] += 10.0;
        assert!(below.stale(&moved));
        // A slight tilt of the east keeps the table; a turn samples it again.
        let mut tilted = view(10.0);
        tilted.east_rendered = [1.0, 1e-3, 0.0];
        assert!(!below.stale(&tilted));
        tilted.east_rendered = [0.0, 0.0, 1.0];
        assert!(below.stale(&tilted));
    }

    #[test]
    fn the_cube_redraws_only_past_the_tolerances() {
        let a =
            SurfaceBelow::sample_ground(&palette(Some(sea())), RADIUS, &view(10.0), level(-20.0));
        let mut b = a;
        b.water_share[3] -= SHARE_TOLERANCE * 0.5;
        b.land[0] += LAND_TOLERANCE * 0.5;
        assert!(!b.moved_from(&a));
        b.water_share[3] = 0.0;
        assert!(b.moved_from(&a));
    }

    #[test]
    fn the_lanes_carry_the_table_in_order() {
        let mut below =
            SurfaceBelow::sample_ground(&palette(Some(sea())), RADIUS, &view(10.0), level(-20.0));
        below.water_share[5] = 0.25;
        let (land, water, east, shares) = below.lanes();
        assert_eq!(land[3], 1.0);
        assert_eq!(water[3], 5.0);
        assert_eq!(east, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(shares[1][1], 0.25);
    }

    #[test]
    fn the_east_is_level_at_the_camera() {
        let planet = PlanetFrame {
            shape: crate::components::Planet::default().shape(),
            frame: crate::planet::LocalFrame::AUTHORED,
        };
        for cam in [[0.0, 2.0, 0.0], [300.0, 5.0, -40.0]] {
            let v = SurfaceView::of(&planet, cam);
            let up = planet.up_at(cam);
            assert!(dot(up, v.east_rendered).abs() < 1e-5);
            assert!((length(v.east_rendered) - 1.0).abs() < 1e-5);
        }
    }
}

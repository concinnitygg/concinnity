//! The world's weather: where over the planet the cloud layers gather into
//! systems, thin out or clear, as a map the GPU builds once when the world
//! loads (`cloud_weather_kernel` in `atmosphere.hlsl`) and the march reads at
//! every sample. `generator` is its CPU mirror.

mod generator;

use super::noise::hash_unit3;
use concinnity_core::math::{cos, narrow_f32, sin};

// Texels along each edge of one of the weather map's six cube faces, and of
// each face's block in the map: the face and a one-texel gutter round it,
// holding the neighboring faces so a bilinear read across an edge sees no
// seam. `CLOUD_WEATHER_FACE` and `CLOUD_WEATHER_STRIDE` in `cloud_types.hlsl`.
const WEATHER_FACE: u32 = 256;
const WEATHER_STRIDE: u32 = WEATHER_FACE + 2;

/// The weather map's size in texels: its six faces, each in its gutter, in a
/// 3 by 2 atlas.
pub const WEATHER_MAP_SIZE: [u32; 2] = [3 * WEATHER_STRIDE, 2 * WEATHER_STRIDE];

/// The weather over a world's cloud layers: systems of cloud and clear sky
/// hundreds of kilometers across on Earth, spiraling round their storms and
/// drawn out along the latitudes, over a climate that runs from a wet equator
/// through dry subtropics to stormy middle latitudes.
///
/// Every [`Cloud`](#cloud) layer covers its own `coverage` of the sky on
/// average, and the weather moves that cover about: overcast under a system,
/// clear between them. It also sets how tall the clouds grow (the tropics
/// and the storms heap them up), how dense they are (from thin gray veils the
/// ground shows through to white cores) and how far separate clouds merge
/// into sheets, as along a front. Without one, every layer covers the sky
/// evenly.
///
/// `system_size` is the span of a weather system, in meters; systems smaller
/// than about a twelfth of the world's radius are finer than the weather
/// holds and are drawn at that size. `storminess` is how common storms are,
/// in `[0, 1]`, and `swirl` how far the cloud around each winds into a
/// spiral: the way the world spins north of the equator, the other way south
/// of it, and less toward the equator. `belts` is how strongly latitude
/// shapes the cover. The systems travel east at `drift` meters per second.
/// Latitude and east are taken about the axis of the world's
/// [`SkyRotation`](#skyrotation), or straight up without one; on a world with
/// no [`Planet`](#planet) they lie on the sphere the
/// [`Atmosphere`](#atmosphere) wraps. `seed` places the systems: each seed is
/// a different sky.
///
/// ```rust
/// # use concinnity_engine::components::Weather;
/// Weather {
///     system_size: 15_000.0,
///     storminess: 0.6,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Weather {
    /// Span of one weather system, in meters. At least 1000.
    #[asset(default = 1_500_000.0)]
    pub system_size: f32,
    /// How common storms are, in `[0, 1]`.
    #[asset(default = 0.5)]
    pub storminess: f32,
    /// How far the cloud round a storm winds into a spiral, in `[0, 1]`.
    #[asset(default = 0.6)]
    pub swirl: f32,
    /// How strongly latitude shapes the cover, in `[0, 1]`: 0 spreads the
    /// systems evenly over the world.
    #[asset(default = 0.6)]
    pub belts: f32,
    /// How fast the systems travel east, in meters per second. At least 0.
    #[asset(default = 10.0)]
    pub drift: f32,
    /// Places the systems: each seed is a different sky.
    pub seed: u32,
}

// How the map's channels scale a layer: its towering at the least and the
// most convection, its density from the thinnest veil to the densest core,
// and how far a sheet flattens its puffs. `CLOUD_TOWERING_LOW` and the rest in
// `clouds_common.hlsl`.
#[cfg(test)]
const TOWERING_LOW: f32 = 0.35;
#[cfg(test)]
const TOWERING_HIGH: f32 = 1.65;
#[cfg(test)]
const DENSITY_LOW: f32 = 0.05;
#[cfg(test)]
const DENSITY_HIGH: f32 = 3.0;
#[cfg(test)]
const SHEET_FLATTEN: f32 = 0.7;

// What the weather map's texel `texel` makes of a layer: its coverage, how
// tall its clouds grow, and a multiple of its density, 1 at the texel's
// middle and running in log steps to the thinnest veil below and the densest
// core above. `cloud_layer_weather` in `clouds_common.hlsl`.
#[cfg(test)]
fn layer_weather(texel: [f32; 4], coverage: f32, towering: f32) -> [f32; 3] {
    let [lean, convection, density, _] = texel;
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let ln = concinnity_core::math::ln;
    let step = if density < 0.5 {
        ln(DENSITY_LOW) * (1.0 - 2.0 * density)
    } else {
        ln(DENSITY_HIGH) * (2.0 * density - 1.0)
    };
    [
        super::shape::lean_coverage(coverage, lean * 2.0 - 1.0),
        (towering * lerp(TOWERING_LOW, TOWERING_HIGH, convection)).clamp(0.0, 1.0),
        concinnity_core::math::exp(step),
    ]
}

/// The pole a world's weather measures latitude about: the axis its
/// [`SkyRotation`](crate::components::SkyRotation) turns the sky on, flipped
/// when the sky runs backward so the world spins the right-handed way about
/// it, or straight up without one.
pub fn spin_pole(rotation: Option<&crate::components::SkyRotation>) -> [f32; 3] {
    rotation.map_or([0.0, 1.0, 0.0], |r| {
        let spin = if r.degrees_per_second < 0.0 {
            -1.0
        } else {
            1.0
        };
        r.axis.map(|a| a * spin)
    })
}

// Where the map holds the weather now over the unit direction `up`: turned
// back about the pole by the systems' drift. `cloud_layer_weather` in
// `clouds_common.hlsl`.
#[cfg(test)]
fn map_lookup(block: &CloudWeatherBlock, up: [f32; 3]) -> [f32; 3] {
    let pole = [block.pole[0], block.pole[1], block.pole[2]];
    generator::turn(
        up,
        pole,
        -concinnity_core::math::atan2(block.drift[1], block.drift[0]),
    )
}

/// A world's weather, resolved for the sphere its cloud layers lie on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeatherSettings {
    /// The axis latitude is measured about, a unit vector in the planet's
    /// fixed axes.
    pub pole: [f32; 3],
    /// Where in the noise this world's weather starts.
    pub seed_offset: [f32; 3],
    /// Weather systems across one radius of the sphere.
    pub systems_per_radius: f32,
    /// How common storms are.
    pub storminess: f32,
    /// How far each storm winds the cloud round it.
    pub swirl: f32,
    /// How strongly latitude shapes the cover.
    pub belts: f32,
    /// How fast the systems travel east, in radians per second.
    pub drift_rad_s: f32,
    /// The lean the generator gives on average, taken off so each layer
    /// keeps its own coverage over the sphere.
    pub lean_offset: f32,
}

impl WeatherSettings {
    /// The weather `weather` describes over a sphere of `radius_km`, its pole
    /// `pole` (any length; straight up when it has none), about which the
    /// world spins the right-handed way.
    pub fn resolve(weather: &Weather, radius_km: f32, pole: [f32; 3]) -> Self {
        let w = crate::components::validate::weather(weather.clone());
        let radius_m = radius_km.max(1e-3) * 1000.0;
        let uneven = Self {
            pole: concinnity_core::math::vec3::normalize_or(pole, 1e-6, [0.0, 1.0, 0.0]),
            seed_offset: hash_unit3(w.seed).map(|o| o * 97.0),
            systems_per_radius: (radius_m / w.system_size).min(generator::MAX_SYSTEMS_PER_RADIUS),
            storminess: w.storminess,
            swirl: w.swirl,
            belts: w.belts,
            drift_rad_s: w.drift / radius_m,
            lean_offset: 0.0,
        };
        Self {
            lean_offset: generator::mean_lean(&uneven),
            ..uneven
        }
    }

    /// The lanes the shaders read, its systems turned `seconds` of drift
    /// east of where the map holds them.
    pub fn block(&self, seconds: f64) -> CloudWeatherBlock {
        let [px, py, pz] = self.pole;
        let [sx, sy, sz] = self.seed_offset;
        let turn = f64::from(self.drift_rad_s) * seconds;
        let turn = narrow_f32(turn % core::f64::consts::TAU);
        CloudWeatherBlock {
            pole: [px, py, pz, 1.0],
            seed: [sx, sy, sz, self.systems_per_radius],
            look: [self.storminess, self.swirl, self.belts, self.lean_offset],
            drift: [cos(turn), sin(turn), 0.0, 0.0],
        }
    }
}

/// The weather's lanes. Mirrors `CloudWeatherBlock` in `cloud_types.hlsl`.
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CloudWeatherBlock {
    /// xyz: the pole, in the planet's fixed axes; w: 1 when there is weather.
    pub pole: [f32; 4],
    /// xyz: where in the noise the weather starts; w: systems across a
    /// radius.
    pub seed: [f32; 4],
    /// x: storminess; y: swirl; z: belts; w: the mean lean taken off.
    pub look: [f32; 4],
    /// x, y: cosine and sine of how far the systems have drifted east about
    /// the pole; zw unused.
    pub drift: [f32; 4],
}

impl CloudWeatherBlock {
    /// No weather: every layer covers the sky evenly.
    pub const OFF: Self = Self {
        pole: [0.0, 1.0, 0.0, 0.0],
        seed: [0.0; 4],
        look: [0.0; 4],
        drift: [1.0, 0.0, 0.0, 0.0],
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::approx::close;
    use core::mem::{offset_of, size_of};

    #[test]
    fn the_block_layout_matches_the_shader() {
        assert_eq!(size_of::<CloudWeatherBlock>(), 64);
        assert_eq!(offset_of!(CloudWeatherBlock, pole), 0);
        assert_eq!(offset_of!(CloudWeatherBlock, seed), 16);
        assert_eq!(offset_of!(CloudWeatherBlock, look), 32);
        assert_eq!(offset_of!(CloudWeatherBlock, drift), 48);
    }

    #[test]
    fn the_face_size_matches_the_shader() {
        let src = crate::render::shader::embedded::CLOUD_TYPES;
        let uint = |name| crate::render::shader::consts::uint(src, name);
        assert_eq!(uint("CLOUD_WEATHER_FACE"), WEATHER_FACE as usize);
        assert_eq!(uint("CLOUD_WEATHER_STRIDE"), WEATHER_STRIDE as usize);
        assert_eq!(WEATHER_MAP_SIZE, [774, 516]);
    }

    #[test]
    fn the_weather_resolves_for_its_sphere() {
        let w = Weather {
            system_size: 15_000.0,
            drift: 10.0,
            ..Weather::default()
        };
        let s = WeatherSettings::resolve(&w, 50.0, [0.0, 0.0, -2.0]);
        assert!(close(s.systems_per_radius, 50_000.0 / 15_000.0, 1e-5));
        assert!(close(s.pole, [0.0, 0.0, -1.0], 1e-6));
        assert!(close(s.drift_rad_s, 10.0 / 50_000.0, 1e-9));
        let up = WeatherSettings::resolve(&w, 50.0, [0.0; 3]);
        assert!(close(up.pole, [0.0, 1.0, 0.0], 0.0));
    }

    // Systems finer than the map holds are drawn at the finest it does.
    #[test]
    fn systems_are_no_finer_than_the_map() {
        let tiny = Weather {
            system_size: 1000.0,
            ..Weather::default()
        };
        let s = WeatherSettings::resolve(&tiny, 6360.0, [0.0, 1.0, 0.0]);
        assert!(close(
            s.systems_per_radius,
            generator::MAX_SYSTEMS_PER_RADIUS,
            0.0
        ));
    }

    #[test]
    fn the_pole_turns_with_the_sky() {
        use crate::components::SkyRotation;
        let forward = SkyRotation {
            axis: [0.0, 0.0, -1.0],
            degrees_per_second: 1.5,
            angle_deg: 0.0,
        };
        assert_eq!(spin_pole(Some(&forward)), [0.0, 0.0, -1.0]);
        let backward = SkyRotation {
            degrees_per_second: -1.5,
            ..forward
        };
        assert_eq!(spin_pole(Some(&backward)), [0.0, 0.0, 1.0]);
        assert_eq!(spin_pole(None), [0.0, 1.0, 0.0]);
    }

    // The systems drift the way the world spins: against the sky, which a
    // `SkyRotation` turns the other way about the same pole.
    #[test]
    fn the_systems_drift_with_the_worlds_spin() {
        use crate::components::SkyRotation;
        use concinnity_core::math::vec3::{cross, dot};
        let sky = SkyRotation {
            axis: [0.0, 0.0, -1.0],
            degrees_per_second: 1.5,
            angle_deg: 0.0,
        };
        let s = WeatherSettings::resolve(&Weather::default(), 50.0, spin_pole(Some(&sky)));
        let seconds = 0.25 / f64::from(s.drift_rad_s);
        let block = s.block(seconds);
        // A feature the map holds at `g` is drawn where the lookup reads `g`.
        let g = [1.0, 0.0, 0.0];
        let drawn = generator::turn(g, s.pole, 0.25);
        assert!(close(map_lookup(&block, drawn), g, 1e-5));
        assert!(dot(cross(g, drawn), s.pole) > 0.0);
        let carried = sky.turned(g, 10.0);
        assert!(dot(cross(g, carried), s.pole) < 0.0);
    }

    #[test]
    fn the_block_turns_the_systems_with_time() {
        let s = WeatherSettings::resolve(&Weather::default(), 6360.0, [0.0, 1.0, 0.0]);
        let still = s.block(0.0);
        assert_eq!(still.pole[3], 1.0);
        assert_eq!(&still.drift[..2], &[1.0, 0.0]);
        let quarter = f64::from(core::f32::consts::FRAC_PI_2 / s.drift_rad_s);
        let turned = s.block(quarter);
        assert!(close(turned.drift[0], 0.0, 1e-3), "{:?}", turned.drift);
        assert!(close(turned.drift[1], 1.0, 1e-3));
        assert_eq!(CloudWeatherBlock::OFF.pole[3], 0.0);
    }

    #[test]
    fn the_layer_scales_match_the_shader() {
        let src = crate::render::shader::embedded::CLOUDS_COMMON;
        let float = |name| crate::render::shader::consts::float(src, name);
        assert_eq!(float("CLOUD_TOWERING_LOW"), TOWERING_LOW);
        assert_eq!(float("CLOUD_TOWERING_HIGH"), TOWERING_HIGH);
        assert_eq!(float("CLOUD_DENSITY_LOW"), DENSITY_LOW);
        assert_eq!(float("CLOUD_DENSITY_HIGH"), DENSITY_HIGH);
        assert_eq!(float("CLOUD_SHEET_FLATTEN"), SHEET_FLATTEN);
    }

    // The texel's middle keeps the layer as authored; the extremes run from
    // veil to core.
    #[test]
    fn a_texel_scales_the_layer() {
        let [coverage, towering, density] = layer_weather([0.5, 0.5, 0.5, 0.0], 0.45, 0.6);
        assert!(close(coverage, 0.45, 1e-6));
        assert!(close(towering, 0.6, 1e-5));
        assert!(close(density, 1.0, 1e-6));
        let veil = layer_weather([0.5, 0.0, 0.0, 0.0], 0.45, 0.6);
        let core = layer_weather([1.0, 1.0, 1.0, 1.0], 0.45, 0.6);
        assert!(close(veil[2], DENSITY_LOW, 1e-6));
        assert!(close(core[2], DENSITY_HIGH, 1e-5));
        assert!(veil[1] < 0.6 && core[1] > 0.6);
        assert!(close(core[0], 1.0, 1e-6));
    }

    #[test]
    fn each_seed_is_a_different_sky() {
        let at = |seed| {
            WeatherSettings::resolve(
                &Weather {
                    seed,
                    ..Weather::default()
                },
                50.0,
                [0.0, 1.0, 0.0],
            )
            .seed_offset
        };
        assert!(!close(at(0), at(1), 1e-3));
    }
}

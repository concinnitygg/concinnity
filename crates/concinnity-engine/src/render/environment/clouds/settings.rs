//! The `Cloud` layers an atmosphere names, resolved into kilometers and the
//! scales the noise volume is read at.

use super::noise::hash_unit3;
use super::params::CloudMarch;
use crate::components::Cloud;
use crate::wind::WindField;
use concinnity_core::math::SaturatingFrom;

/// The most cloud layers one atmosphere draws. `CLOUD_MAX_LAYERS` in
/// `cloud_types.hlsl`.
pub const MAX_CLOUD_LAYERS: usize = 2;

// Noise cells across one tile of the weather channels, and of the base shape
// and detail channels at their coarsest octave. `noise.rs` builds the volume
// with these counts.
pub(super) const WEATHER_CELLS: f32 = 6.0;
pub(super) const SHAPE_CELLS: f32 = 4.0;
pub(super) const DETAIL_CELLS: f32 = 4.0;

// A base-shape puff spans this many layer thicknesses, bounded by the weather
// cells it sits in.
const PUFF_THICKNESSES: f32 = 0.6;

const METERS_PER_KM: f32 = 1000.0;

/// One cloud layer, in kilometers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudLayer {
    /// Height of the bottom over the ground.
    pub base_km: f32,
    /// Height of the top over the ground.
    pub top_km: f32,
    /// The share of the sky covered, in `[0, 1]`.
    pub coverage: f32,
    /// How tall the clouds grow, in `[0, 1]`.
    pub towering: f32,
    /// Extinction per kilometer at full density.
    pub extinction: f32,
    /// Kilometers one tile of the noise volume spans for the weather.
    pub weather_tile_km: f32,
    /// Kilometers one tile spans for the base shape.
    pub shape_tile_km: f32,
    /// Kilometers one tile spans for the detail.
    pub detail_tile_km: f32,
    /// The drift, in kilometers per second along the planet's fixed `x` and
    /// `z`.
    pub drift_km_s: [f32; 2],
    /// Cloud changes per second.
    pub evolve_rate: f32,
    /// Where in the noise this layer's sky starts, in tiles.
    pub seed_offset: [f32; 3],
    /// The axis the clouds are drawn out along, a unit vector in the planet's
    /// fixed `x` and `z`: the wind's.
    pub stretch_axis: [f32; 2],
    /// How far they are drawn out along it, at least 1.
    pub stretch: f32,
}

impl Default for CloudLayer {
    fn default() -> Self {
        Self {
            base_km: 0.0,
            top_km: 0.0,
            coverage: 0.0,
            towering: 0.0,
            extinction: 0.0,
            weather_tile_km: 1.0,
            shape_tile_km: 1.0,
            detail_tile_km: 1.0,
            drift_km_s: [0.0; 2],
            evolve_rate: 0.0,
            seed_offset: [0.0; 3],
            stretch_axis: [1.0, 0.0],
            stretch: 1.0,
        }
    }
}

impl CloudLayer {
    /// The noise-space map that draws a cloud out along the wind: `p` with its
    /// part along the stretch axis shrunk by the stretch, so the noise there
    /// spans that many times as far. `cloud_stretched` in `clouds_common.hlsl`.
    pub fn stretched(&self, p: [f64; 3]) -> [f64; 3] {
        let [ax, az] = self.stretch_axis.map(f64::from);
        let along = p[0] * ax + p[2] * az;
        let k = 1.0 / f64::from(self.stretch) - 1.0;
        [p[0] + k * along * ax, p[1], p[2] + k * along * az]
    }

    /// The layer an authored `cloud` describes under `wind`.
    pub fn resolve(cloud: &Cloud, wind: &WindField) -> Self {
        let c = crate::components::validate::cloud(cloud.clone());
        let base_km = c.base / METERS_PER_KM;
        let thickness_km = c.thickness / METERS_PER_KM;
        let feature_km = c.feature_size / METERS_PER_KM;
        let puff_km = (thickness_km * PUFF_THICKNESSES).clamp(feature_km * 0.1, feature_km * 0.5);
        let speed = wind.strength * c.wind_drift / METERS_PER_KM;
        Self {
            base_km,
            top_km: base_km + thickness_km,
            coverage: c.coverage,
            towering: c.towering,
            extinction: c.density * METERS_PER_KM,
            weather_tile_km: feature_km * WEATHER_CELLS,
            shape_tile_km: puff_km * SHAPE_CELLS,
            detail_tile_km: c.detail_size / METERS_PER_KM * DETAIL_CELLS,
            drift_km_s: wind.direction.map(|d| d * speed),
            evolve_rate: if c.evolve_seconds > 0.0 {
                1.0 / c.evolve_seconds
            } else {
                0.0
            },
            seed_offset: hash_unit3(c.seed),
            stretch_axis: wind.direction,
            stretch: c.stretch,
        }
    }
}

/// The cloud layers of one atmosphere, lowest first, and how finely they are
/// marched.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudSettings {
    layers: [CloudLayer; MAX_CLOUD_LAYERS],
    count: usize,
    /// How finely the layers are marched.
    pub march: CloudMarch,
}

impl CloudSettings {
    /// The layers `clouds` describe under `wind`, the first
    /// [`MAX_CLOUD_LAYERS`] of them, marched by `march`; `None` for no layers.
    pub fn resolve<'a>(
        clouds: impl IntoIterator<Item = &'a Cloud>,
        wind: &WindField,
        march: CloudMarch,
    ) -> Option<Self> {
        let mut layers = [CloudLayer::default(); MAX_CLOUD_LAYERS];
        let mut count = 0;
        for cloud in clouds {
            if count == MAX_CLOUD_LAYERS {
                tracing::warn!(
                    "an Atmosphere draws at most {MAX_CLOUD_LAYERS} cloud layers; the rest are not drawn"
                );
                break;
            }
            layers[count] = CloudLayer::resolve(cloud, wind);
            count += 1;
        }
        if count == 0 {
            return None;
        }
        layers[..count].sort_by(|a, b| a.base_km.total_cmp(&b.base_km));
        Some(Self {
            layers,
            count,
            march,
        })
    }

    /// The layers, lowest first.
    pub fn layers(&self) -> &[CloudLayer] {
        &self.layers[..self.count]
    }

    /// How many layers there are, as the shader's count lane carries it.
    pub fn count(&self) -> u32 {
        u32::saturating_from(self.count)
    }

    /// The heights over the ground, in kilometers, that every layer lies
    /// between.
    pub fn span_km(&self) -> [f32; 2] {
        self.layers()
            .iter()
            .fold([f32::MAX, f32::MIN], |[lo, hi], l| {
                [lo.min(l.base_km), hi.max(l.top_km)]
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::approx::close;

    fn wind(strength: f32) -> WindField {
        WindField {
            direction: [0.6, 0.8],
            strength,
            ..WindField::STILL
        }
    }

    #[test]
    fn a_layer_resolves_into_kilometers() {
        let layer = CloudLayer::resolve(
            &Cloud {
                base: 600.0,
                thickness: 1200.0,
                density: 0.05,
                feature_size: 5000.0,
                detail_size: 100.0,
                ..Cloud::default()
            },
            &WindField::STILL,
        );
        assert!(close(layer.base_km, 0.6, 1e-6));
        assert!(close(layer.top_km, 1.8, 1e-6));
        assert!(close(layer.extinction, 50.0, 1e-4));
        assert!(close(layer.weather_tile_km, 5.0 * WEATHER_CELLS, 1e-5));
        assert!(close(layer.detail_tile_km, 0.1 * DETAIL_CELLS, 1e-6));
        assert!(close(layer.drift_km_s, [0.0, 0.0], 0.0));
    }

    #[test]
    fn clouds_drift_downwind_faster_than_the_wind() {
        let layer = CloudLayer::resolve(&Cloud::default(), &wind(5.0));
        let speed = Cloud::default().wind_drift * 5.0 / 1000.0;
        assert!(close(layer.drift_km_s[0], 0.6 * speed, 1e-7));
        assert!(close(layer.drift_km_s[1], 0.8 * speed, 1e-7));
    }

    #[test]
    fn a_still_shape_never_evolves() {
        let layer = CloudLayer::resolve(
            &Cloud {
                evolve_seconds: 0.0,
                ..Cloud::default()
            },
            &WindField::STILL,
        );
        assert!(close(layer.evolve_rate, 0.0, 0.0));
    }

    // A puff is a share of the layer's thickness, but never wider than half a
    // weather cell nor finer than a tenth of one.
    #[test]
    fn the_puffs_follow_the_thickness_within_the_weather_cells() {
        let puff = |thickness: f32| {
            CloudLayer::resolve(
                &Cloud {
                    thickness,
                    feature_size: 4000.0,
                    ..Cloud::default()
                },
                &WindField::STILL,
            )
            .shape_tile_km
                / SHAPE_CELLS
        };
        assert!(close(puff(1000.0), 0.6, 1e-5));
        assert!(close(puff(10_000.0), 2.0, 1e-5));
        assert!(close(puff(100.0), 0.4, 1e-5));
    }

    #[test]
    fn layers_sort_lowest_first_and_stop_at_the_limit() {
        let at = |base: f32| Cloud {
            base,
            ..Cloud::default()
        };
        let clouds = [at(3000.0), at(500.0), at(9000.0)];
        let s = CloudSettings::resolve(&clouds, &WindField::STILL, CloudMarch::FULL).unwrap();
        assert_eq!(s.count(), 2);
        assert!(close(s.layers()[0].base_km, 0.5, 1e-6));
        assert!(close(s.layers()[1].base_km, 3.0, 1e-6));
        let [lo, hi] = s.span_km();
        assert!(close(lo, 0.5, 1e-6));
        assert!(close(hi, 4.5, 1e-6));
    }

    // A stretch shrinks only the part along the wind, so the noise spans that
    // many times as far that way; a stretch of 1 changes nothing.
    #[test]
    fn a_stretch_draws_clouds_out_along_the_wind() {
        let layer = CloudLayer::resolve(
            &Cloud {
                stretch: 4.0,
                ..Cloud::default()
            },
            &wind(5.0),
        );
        let along = layer.stretched([0.6, 2.0, 0.8]);
        assert!(close(along, [0.15, 2.0, 0.2], 1e-6), "{along:?}");
        let across = layer.stretched([0.8, 0.0, -0.6]);
        assert!(close(across, [0.8, 0.0, -0.6], 1e-6));
        let round = CloudLayer::resolve(&Cloud::default(), &wind(5.0));
        assert!(close(
            round.stretched([0.6, 2.0, 0.8]),
            [0.6, 2.0, 0.8],
            1e-9
        ));
    }

    #[test]
    fn no_layers_resolve_to_none() {
        assert!(CloudSettings::resolve(&[], &WindField::STILL, CloudMarch::FULL).is_none());
    }

    #[test]
    fn each_seed_is_a_different_sky() {
        let at = |seed| {
            CloudLayer::resolve(
                &Cloud {
                    seed,
                    ..Cloud::default()
                },
                &WindField::STILL,
            )
            .seed_offset
        };
        assert!(!close(at(0), at(1), 1e-3));
        assert!(at(7).iter().all(|o| (0.0..1.0).contains(o)));
    }
}

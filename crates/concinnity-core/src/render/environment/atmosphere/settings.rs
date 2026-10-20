//! The authored `Atmosphere` resolved against the world it sits in.

use super::air::AirProfile;
use super::apparent::ApparentGround;
use crate::components::Atmosphere;
use crate::math::{cos, narrow_f32};
use crate::render::environment::night_sky::NightSkySettings;

/// Meters in a kilometer: world units are meters, the atmosphere's are
/// kilometers.
pub(super) const METERS_PER_KM: f32 = 1000.0;

/// An `Atmosphere` placed around its planet: the medium's coefficients, the
/// radii of the ground and of the top of the air, and where the planet's
/// center is in the world. Built once at init and handed to the backend with
/// the world's other effects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereSettings {
    /// Radius of the lowest ground, in kilometers.
    pub bottom_radius: f32,
    /// Radius of the top of the air, in kilometers.
    pub top_radius: f32,
    /// The air over the planet's own ground.
    pub air: AirProfile,
    /// Mie phase anisotropy.
    pub mie_anisotropy: f32,
    /// Albedo of the ground under the air.
    pub ground_albedo: [f32; 3],
    /// Cosine of the sun disc's angular radius.
    pub sun_cos_radius: f32,
    /// Disc radiance as a multiple of the sunlight.
    pub sun_disc_intensity: f32,
    /// The planet's center in world space, in meters, or `None` when the air
    /// wraps a planet the world does not declare, centered straight below
    /// the world origin.
    pub planet_center: Option<[f64; 3]>,
    /// The radius the sky is drawn over near the ground, when it is not the
    /// true one, and the air as authored over it.
    pub apparent: Option<ApparentGround>,
    /// The night sky drawn behind the air, if the world has one.
    pub night: Option<NightSkySettings>,
}

impl AtmosphereSettings {
    /// Resolve an authored atmosphere. `planet` is the world's planet as its
    /// center and radius in meters; without one the air wraps the asset's own
    /// radius, its top at height 0.
    pub fn resolve(atmosphere: &Atmosphere, planet: Option<([f64; 3], f32)>) -> Self {
        let a = crate::components::validate::atmosphere(atmosphere.clone());
        let bottom_radius = match planet {
            Some((_, radius)) => (radius / METERS_PER_KM).max(0.1),
            None => a.planet_radius_km,
        };
        let half_angle = (a.sun_angular_diameter_deg * 0.5).to_radians();
        let authored = AirProfile {
            height: a.height_km,
            rayleigh_scattering: a.rayleigh_scattering,
            rayleigh_scale_height: a.rayleigh_scale_height_km,
            mie_scattering: a.mie_scattering,
            mie_extinction: core::array::from_fn(|i| a.mie_scattering[i] + a.mie_absorption[i]),
            mie_scale_height: a.mie_scale_height_km,
            ozone_absorption: a.ozone_absorption,
            ozone_center: a.ozone_center_km,
            ozone_half_width: a.ozone_width_km * 0.5,
        };
        let apparent = a.apparent_radius_km.map(|radius| ApparentGround {
            radius,
            air: authored,
            fade_start: a.apparent_radius_fade_km[0],
            fade_end: a.apparent_radius_fade_km[1],
        });
        let air = apparent.map_or(authored, |ap| ap.air_over(bottom_radius));
        Self {
            bottom_radius,
            top_radius: bottom_radius + air.height,
            air,
            mie_anisotropy: a.mie_anisotropy,
            ground_albedo: a.ground_albedo,
            sun_cos_radius: cos(half_angle),
            sun_disc_intensity: a.sun_disc_intensity,
            planet_center: planet.map(|(center, _)| center),
            apparent,
            night: None,
        }
    }

    /// No air at all, around the world's planet or under a flat world as
    /// [`resolve`](Self::resolve) places it: the sky a world with a night sky
    /// and no atmosphere is drawn through. Nothing scatters or dims the light
    /// and the ground below the horizon is black.
    pub fn vacuum(planet: Option<([f64; 3], f32)>) -> Self {
        Self::resolve(
            &Atmosphere {
                rayleigh_scattering: [0.0; 3],
                mie_scattering: [0.0; 3],
                mie_absorption: [0.0; 3],
                ozone_absorption: [0.0; 3],
                ground_albedo: [0.0; 3],
                ..Default::default()
            },
            planet,
        )
    }

    /// These settings with `night` drawn behind the air.
    #[must_use]
    pub fn with_night(self, night: Option<NightSkySettings>) -> Self {
        Self { night, ..self }
    }

    /// The ground radius the sky's tables are built over for a camera
    /// `height` kilometers above the ground: the true radius unless an
    /// apparent one is set.
    pub fn sky_ground_radius(&self, height: f32) -> f32 {
        match self.apparent {
            Some(apparent) => apparent.ground_radius(self.bottom_radius, height),
            None => self.bottom_radius,
        }
    }

    /// These settings over ground of `radius`. With an apparent radius the
    /// air keeps its proportion to the ground, exactly as authored over the
    /// apparent ground; without one it keeps its thickness.
    #[must_use]
    pub fn with_ground_radius(&self, radius: f32) -> Self {
        if radius == self.bottom_radius {
            return *self;
        }
        let air = self.apparent.map_or(self.air, |ap| ap.air_over(radius));
        Self {
            bottom_radius: radius,
            top_radius: radius + air.height,
            air,
            ..*self
        }
    }

    /// The camera relative to the planet's center, in kilometers, from its
    /// world position in meters. `world_from_local` places the simulated
    /// frame's points in the authored world (identity outside a planet
    /// world); the answer is in the simulated frame's axes, which is what the
    /// view rays are in.
    pub fn camera_km(&self, camera: [f64; 3], planet_center_local: Option<[f64; 3]>) -> [f32; 3] {
        let center = match planet_center_local {
            Some(c) => c,
            None => [
                0.0,
                -f64::from(self.bottom_radius) * f64::from(METERS_PER_KM),
                0.0,
            ],
        };
        core::array::from_fn(|i| narrow_f32((camera[i] - center[i]) / f64::from(METERS_PER_KM)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::NightSky;

    #[test]
    fn earth_defaults_resolve_to_the_literature_shell() {
        let s = AtmosphereSettings::resolve(&Atmosphere::default(), None);
        assert_eq!(s.bottom_radius, 6360.0);
        assert_eq!(s.top_radius, 6460.0);
        assert!((s.air.mie_extinction[0] - 0.00444).abs() < 1e-6);
        assert_eq!(s.air.ozone_half_width, 15.0);
        // A 0.53 degree disc has a 0.265 degree radius.
        assert!((s.sun_cos_radius - cos(0.265_f32.to_radians())).abs() < 1e-7);
        assert_eq!(s.planet_center, None);
        assert_eq!(s.apparent, None);
        assert_eq!(s.sky_ground_radius(3.0), 6360.0);
    }

    fn small_planet() -> AtmosphereSettings {
        AtmosphereSettings::resolve(
            &Atmosphere {
                apparent_radius_km: Some(6360.0),
                ..Default::default()
            },
            Some(([0.0, -50_000.0, 0.0], 50_000.0)),
        )
    }

    // An apparent radius builds the sky over a bigger ground near the
    // planet, with the air as authored, and over the true one high above it,
    // the air as thin beside it as Earth's beside Earth.
    #[test]
    fn an_apparent_radius_moves_the_sky_ground() {
        let s = small_planet();
        assert_eq!(s.sky_ground_radius(0.1), 6360.0);
        assert_eq!(s.sky_ground_radius(30.0), 50.0);
        let table = s.with_ground_radius(s.sky_ground_radius(0.1));
        assert_eq!(table.bottom_radius, 6360.0);
        assert_eq!(table.top_radius, 6460.0);
        assert_eq!(s.with_ground_radius(50.0), s);
        let thickness = s.top_radius - s.bottom_radius;
        assert!(
            (thickness - 100.0 * 50.0 / 6360.0).abs() < 1e-4,
            "{thickness}"
        );
        assert!((s.air.rayleigh_scale_height - 8.0 * 50.0 / 6360.0).abs() < 1e-5);
    }

    // Up to the fade's start the tables' air is the authored air, bit for
    // bit.
    #[test]
    fn the_air_near_the_ground_is_as_authored() {
        let s = small_planet();
        let earth = AtmosphereSettings::resolve(&Atmosphere::default(), None);
        for h in [0.0, 0.002, 0.1, 0.5, 1.0] {
            let table = s.with_ground_radius(s.sky_ground_radius(h));
            assert_eq!(table.air, earth.air, "{h}");
            assert_eq!(table.top_radius, earth.top_radius, "{h}");
        }
    }

    // Through the fade the air thins in step with the ground, never jumping,
    // and every column straight up or along the ground keeps its optical
    // depth.
    #[test]
    fn the_air_fades_with_the_ground_keeping_its_depth() {
        use super::super::optics::optical_depth;
        let s = small_planet();
        let table_at = |h: f32| s.with_ground_radius(s.sky_ground_radius(h));
        let ground = table_at(0.0);
        let zenith = optical_depth(&ground, ground.bottom_radius, 1.0);
        let grazing = optical_depth(&ground, ground.bottom_radius, 0.0);
        let mut last = ground.top_radius - ground.bottom_radius;
        for i in 1..=200 {
            let h = 1.0 + 19.0 * i as f32 / 200.0;
            let table = table_at(h);
            let thickness = table.top_radius - table.bottom_radius;
            let ratio = thickness / table.bottom_radius;
            assert!((ratio - 100.0 / 6360.0).abs() < 1e-6, "{h}: {ratio}");
            assert!(thickness < last, "{h}: {thickness} after {last}");
            assert!(last - thickness < 3.0, "{h}: jumped {last} -> {thickness}");
            last = thickness;
            let z = optical_depth(&table, table.bottom_radius, 1.0);
            let g = optical_depth(&table, table.bottom_radius, 0.0);
            for c in 0..3 {
                assert!((z[c] - zenith[c]).abs() < zenith[c] * 1e-3, "{h} {c}");
                assert!((g[c] - grazing[c]).abs() < grazing[c] * 1e-3, "{h} {c}");
            }
        }
        assert_eq!(table_at(20.0), s);
    }

    // A planet's radius replaces the authored one, and the air keeps its
    // authored thickness above it.
    #[test]
    fn a_planet_sets_the_ground_radius() {
        let s = AtmosphereSettings::resolve(
            &Atmosphere::default(),
            Some(([0.0, -50_000.0, 0.0], 50_000.0)),
        );
        assert_eq!(s.bottom_radius, 50.0);
        assert_eq!(s.top_radius, 150.0);
        assert_eq!(s.planet_center, Some([0.0, -50_000.0, 0.0]));
    }

    // A vacuum dims nothing at any height, and keeps the planet it wraps.
    #[test]
    fn a_vacuum_has_no_extinction() {
        let s = AtmosphereSettings::vacuum(Some(([0.0, -50_000.0, 0.0], 50_000.0)));
        assert_eq!(s.bottom_radius, 50.0);
        for h in [0.0, 5.0, 25.0, 90.0] {
            assert_eq!(s.air.extinction_at(h), [0.0; 3]);
        }
        assert_eq!(s.ground_albedo, [0.0; 3]);
        assert!(s.night.is_none());
        let night = NightSkySettings::resolve(&NightSky::default());
        assert_eq!(s.with_night(Some(night)).night, Some(night));
    }

    // Authored nonsense is clamped before it reaches a divisor.
    #[test]
    fn invalid_values_are_clamped() {
        let s = AtmosphereSettings::resolve(
            &Atmosphere {
                height_km: -3.0,
                rayleigh_scale_height_km: 0.0,
                mie_scattering: [-1.0, f32::NAN, 0.5],
                mie_anisotropy: 1.5,
                ..Default::default()
            },
            None,
        );
        assert_eq!(s.top_radius - s.bottom_radius, 1.0);
        assert!(s.air.rayleigh_scale_height > 0.0);
        assert_eq!(s.air.mie_scattering, [0.0, 0.0, 0.5]);
        assert_eq!(s.mie_anisotropy, 0.99);
    }

    #[test]
    fn extinction_thins_with_height_and_peaks_in_the_ozone() {
        let s = AtmosphereSettings::resolve(&Atmosphere::default(), None);
        let ground = s.air.extinction_at(0.0);
        let high = s.air.extinction_at(60.0);
        assert!((0..3).all(|i| high[i] < ground[i] * 0.05));
        // Ozone alone at its center, beside the thin air there.
        let at_layer = s.air.extinction_at(25.0);
        let below = s.air.extinction_at(8.0);
        assert!(at_layer[1] > s.air.ozone_absorption[1]);
        assert!(at_layer[1] - s.air.ozone_absorption[1] < below[1]);
    }

    // Without a planet the ground's top is at the world origin; with one the
    // camera is placed from the planet's center in the simulated frame.
    #[test]
    fn the_camera_is_placed_about_the_center() {
        let flat = AtmosphereSettings::resolve(&Atmosphere::default(), None);
        let cam = flat.camera_km([0.0, 1500.0, 0.0], None);
        assert!((cam[1] - 6361.5).abs() < 1e-3, "{cam:?}");
        let planet = AtmosphereSettings::resolve(
            &Atmosphere::default(),
            Some(([0.0, -50_000.0, 0.0], 50_000.0)),
        );
        let cam = planet.camera_km([0.0, 12.0, 0.0], Some([0.0, -50_000.0, 0.0]));
        assert!((cam[1] - 50.012).abs() < 1e-4, "{cam:?}");
    }
}

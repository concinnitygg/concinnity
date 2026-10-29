//! What the atmosphere needs from each frame: where the camera is about the
//! planet and where the sun is, and the sunlight that reaches the camera.

use super::optics::{
    disc_set_depth, disc_transmittance, disc_transmittance_seen, sun_transmittance,
    sun_transmittance_seen,
};
use super::settings::AtmosphereSettings;
use super::surface::SurfaceBelow;
use crate::components::DirectionalLight;
use crate::render::environment::night_sky::{NightFrame, moon_share};
use concinnity_core::math::vec3::{dot, length, normalize_or, scale};

/// The atmosphere's per-frame inputs, carried from extraction to the
/// backend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereFrame {
    /// The camera relative to the planet's center, in kilometers, in the
    /// axes the frame renders in.
    pub camera_km: [f32; 3],
    /// The ground radius the sky's tables are built over this frame, in
    /// kilometers: the planet's own unless the atmosphere sets an apparent
    /// one.
    pub sky_ground_radius: f32,
    /// Unit direction toward the sun, in the axes the frame renders in.
    pub sun_direction: [f32; 3],
    /// The sunlight above the air: the sun light's `color * intensity`.
    pub sun_illuminance: [f32; 3],
    /// The fraction of that sunlight that reaches the camera, per color.
    pub sun_transmittance: [f32; 3],
    /// How far, in radians, the sun has sunk below the horizon past fully
    /// setting: negative while any of its disc is up.
    pub sun_set_depth: f32,
    /// The night sky behind the air, if the world has one.
    pub night: Option<NightFrame>,
    /// The ground around the camera the sky's cubes show below the horizon;
    /// `None` leaves it the atmosphere's `ground_albedo`.
    pub surface: Option<SurfaceBelow>,
}

impl AtmosphereFrame {
    /// The frame's inputs for a camera `camera_km` from the planet's center,
    /// lit by `sun`, whose direction is already where the sky has carried it.
    pub fn new(settings: &AtmosphereSettings, camera_km: [f32; 3], sun: &DirectionalLight) -> Self {
        let sun_direction = normalize_or(sun.direction, 1e-6, [0.0, 1.0, 0.0]);
        let r = length(camera_km).max(settings.bottom_radius);
        let up = scale(camera_km, 1.0 / length(camera_km).max(1e-6));
        let mu = dot(up, sun_direction);
        let ground = settings.sky_ground_radius(length(camera_km) - settings.bottom_radius);
        let sun_transmittance = if ground == settings.bottom_radius {
            sun_transmittance(settings, r, mu)
        } else {
            sun_transmittance_seen(settings, &settings.with_ground_radius(ground), r, mu)
        };
        Self {
            camera_km,
            sky_ground_radius: ground,
            sun_direction,
            sun_illuminance: scale(sun.color, sun.intensity),
            sun_transmittance,
            sun_set_depth: disc_set_depth(settings, r, mu, settings.sun_cos_radius),
            night: None,
            surface: None,
        }
    }

    /// This frame with `night` behind the air: its moon's light dimmed by
    /// the air between it and the camera as the sun's is, and switched on
    /// by the handover once the sun has set.
    #[must_use]
    pub fn with_night(self, settings: &AtmosphereSettings, mut night: NightFrame) -> Self {
        if let (Some(moon), Some(resolved)) =
            (night.moon.as_mut(), settings.night.and_then(|n| n.moon))
        {
            let r = length(self.camera_km).max(settings.bottom_radius);
            let up = scale(self.camera_km, 1.0 / length(self.camera_km).max(1e-6));
            let mu = dot(up, moon.direction);
            let cos_radius = resolved.cos_radius;
            moon.transmittance = if self.sky_ground_radius == settings.bottom_radius {
                disc_transmittance(settings, r, mu, cos_radius)
            } else {
                let table = settings.with_ground_radius(self.sky_ground_radius);
                disc_transmittance_seen(settings, &table, r, mu, cos_radius)
            };
            moon.share = moon_share(self.sun_set_depth);
        }
        Self {
            night: Some(night),
            ..self
        }
    }

    /// This frame with the ground below the horizon seen as `surface`.
    #[must_use]
    pub fn with_surface(self, surface: Option<SurfaceBelow>) -> Self {
        Self { surface, ..self }
    }

    /// The camera in the sky tables' space: as high above their ground as it
    /// is above the true ground, along the same up.
    pub fn table_camera_km(&self, settings: &AtmosphereSettings) -> [f32; 3] {
        if self.sky_ground_radius == settings.bottom_radius {
            return self.camera_km;
        }
        let r = length(self.camera_km).max(1e-6);
        let height = r - settings.bottom_radius;
        scale(self.camera_km, (self.sky_ground_radius + height) / r)
    }

    /// The camera's height above the ground, in kilometers.
    pub fn camera_height(&self, settings: &AtmosphereSettings) -> f32 {
        length(self.camera_km) - settings.bottom_radius
    }
}

/// The sun light as it reaches the camera through the air: `sun`'s color
/// dimmed and tinted by the transmittance of its path, so the scene is lit
/// white at noon, orange at sunset and not at all once the sun has set.
pub fn sun_light(frame: &AtmosphereFrame, sun: &DirectionalLight) -> DirectionalLight {
    DirectionalLight {
        color: core::array::from_fn(|i| sun.color[i] * frame.sun_transmittance[i]),
        ..*sun
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Atmosphere;

    fn earth() -> AtmosphereSettings {
        AtmosphereSettings::resolve(&Atmosphere::default(), None)
    }

    fn sun_at(direction: [f32; 3]) -> DirectionalLight {
        DirectionalLight {
            direction,
            color: [1.0, 1.0, 1.0],
            intensity: 3.0,
            angular_radius_deg: 0.0,
        }
    }

    // Overhead the sun lights the scene nearly white; at sunset it is weaker
    // and orange; below the horizon it is out.
    #[test]
    fn the_sun_light_follows_the_sun_through_the_day() {
        let s = earth();
        let camera = [0.0, s.bottom_radius + 0.002, 0.0];
        let light = |dir| {
            let sun = sun_at(dir);
            sun_light(&AtmosphereFrame::new(&s, camera, &sun), &sun)
        };
        let noon = light([0.0, 1.0, 0.0]);
        assert!(noon.color.iter().all(|&c| c > 0.75), "{:?}", noon.color);
        assert_eq!(noon.intensity, 3.0);
        let sunset = light([1.0, 0.04, 0.0]);
        assert!(
            sunset.color[0] > 2.0 * sunset.color[2],
            "{:?}",
            sunset.color
        );
        assert!(sunset.color[0] < noon.color[0]);
        let night = light([0.3, -0.5, 0.0]);
        assert_eq!(night.color, [0.0; 3]);
    }

    // The frame carries the sun's illuminance and a unit direction whatever
    // length the light was authored with.
    #[test]
    fn the_frame_normalizes_and_scales_the_sun() {
        let s = earth();
        let frame = AtmosphereFrame::new(
            &s,
            [0.0, s.bottom_radius + 1.0, 0.0],
            &DirectionalLight {
                direction: [0.0, 4.0, 0.0],
                color: [1.0, 0.5, 0.25],
                intensity: 2.0,
                angular_radius_deg: 0.0,
            },
        );
        assert_eq!(frame.sun_direction, [0.0, 1.0, 0.0]);
        assert_eq!(frame.sun_illuminance, [2.0, 1.0, 0.5]);
        assert!((frame.camera_height(&s) - 1.0).abs() < 1e-3);
        assert_eq!(frame.sky_ground_radius, s.bottom_radius);
        assert_eq!(frame.table_camera_km(&s), frame.camera_km);
    }

    // A night sky's moon is dimmed by the air like the sun, and lights the
    // scene only once the sun has set.
    #[test]
    fn the_moon_is_lit_through_the_air_after_sunset() {
        use crate::components::NightSky;
        use crate::render::environment::night_sky::NightSkySettings;
        let s = earth().with_night(Some(NightSkySettings::resolve(&NightSky::default())));
        let camera = [0.0, s.bottom_radius + 0.002, 0.0];
        let night_with_moon = |moon_dir: [f32; 3]| {
            let mut n = NightFrame::new(
                s.night.as_ref().expect("night"),
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                &crate::sky::SkyOrientation::default(),
            );
            if let Some(m) = n.moon.as_mut() {
                m.direction = moon_dir;
            }
            n
        };
        let day = AtmosphereFrame::new(&s, camera, &sun_at([0.0, 1.0, 0.0]));
        let night = AtmosphereFrame::new(&s, camera, &sun_at([0.0, -0.5, 0.8]));
        assert!(day.sun_set_depth < 0.0 && night.sun_set_depth > 0.2);
        let high = night.with_night(&s, night_with_moon([0.0, 1.0, 0.0]));
        let low = night.with_night(&s, night_with_moon([1.0, 0.03, 0.0]));
        let set = night.with_night(&s, night_with_moon([1.0, -0.2, 0.0]));
        let moon = |f: AtmosphereFrame| f.night.and_then(|n| n.moon).expect("moon");
        assert!(moon(high).transmittance.iter().all(|&t| t > 0.75));
        assert!(moon(low).transmittance[0] > 2.0 * moon(low).transmittance[2]);
        assert_eq!(moon(set).transmittance, [0.0; 3]);
        assert_eq!(moon(high).share, 1.0);
        assert_eq!(
            moon(day.with_night(&s, night_with_moon([0.0, 1.0, 0.0]))).share,
            0.0
        );
    }

    // Near a small planet with an apparent radius the tables sit over the
    // bigger ground at the camera's true height, and the sun is redder; high
    // above it the frame is the planet's own.
    #[test]
    fn an_apparent_radius_moves_the_tables_near_the_ground() {
        let small = |apparent| {
            AtmosphereSettings::resolve(
                &Atmosphere {
                    apparent_radius_km: apparent,
                    ..Default::default()
                },
                Some(([0.0, -50_000.0, 0.0], 50_000.0)),
            )
        };
        let (own, seen) = (small(None), small(Some(6360.0)));
        let sun = sun_at([1.0, 0.05, 0.0]);
        let ground = [0.0, 50.09, 0.0];
        let a = AtmosphereFrame::new(&own, ground, &sun);
        let b = AtmosphereFrame::new(&seen, ground, &sun);
        assert_eq!(b.sky_ground_radius, 6360.0);
        let table = b.table_camera_km(&seen);
        assert!((table[1] - 6360.09).abs() < 1e-2, "{table:?}");
        let red = |f: &AtmosphereFrame| f.sun_transmittance[0] / f.sun_transmittance[2];
        assert!(
            red(&b) > 2.0 * red(&a),
            "{:?} {:?}",
            a.sun_transmittance,
            b.sun_transmittance
        );
        // High above, the tables are the planet's own and its air, as thin
        // beside it as Earth's, lies below the camera: the sun is undimmed.
        let high = AtmosphereFrame::new(&seen, [0.0, 80.0, 0.0], &sun);
        assert_eq!(high.sky_ground_radius, 50.0);
        assert_eq!(high.table_camera_km(&seen), high.camera_km);
        assert_eq!(high.sun_transmittance, [1.0; 3]);
    }
}

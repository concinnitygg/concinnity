//! The constant block every atmosphere shader binds.

use super::frame::AtmosphereFrame;
use super::lut::AERIAL_MAX_DISTANCE_KM;
use super::settings::{AtmosphereSettings, METERS_PER_KM};
use crate::math::vec3::length;
use crate::render::night_sky::{MAX_GALAXIES, NightLanes};

/// The atmosphere's constants, shared by the table kernels, the sky draw and
/// the haze pass. Mirrors `AtmosphereParams` in atmosphere_common.hlsl. Every
/// member is a `float4` lane so the three shader targets lay it out alike.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AtmosphereParams {
    /// Inverse of the un-jittered view-projection: turns a screen position
    /// and depth back into a world point, and a froxel into its view ray.
    pub inv_vp: [[f32; 4]; 4],
    /// xyz = Rayleigh scattering per km, w = -1 / its scale height.
    pub rayleigh: [f32; 4],
    /// xyz = Mie scattering per km, w = -1 / its scale height.
    pub mie_scattering: [f32; 4],
    /// xyz = Mie extinction per km, w = the Mie phase anisotropy.
    pub mie_extinction: [f32; 4],
    /// xyz = ozone absorption per km, w = the layer's center height in km.
    pub ozone: [f32; 4],
    /// xyz = the ground's albedo, w = half the ozone layer's width in km.
    pub ground_albedo: [f32; 4],
    /// x = the tables' ground radius, y = their top radius, z = the aerial
    /// volume's reach, all km; w = meters per world unit over meters per km
    /// (world units to km).
    pub radii: [f32; 4],
    /// xyz = the camera relative to the tables' planet center in km, w = its
    /// distance from the center.
    pub camera: [f32; 4],
    /// The planet the sky is seen over, in km: x = its ground radius, y = its
    /// top radius, z = the camera's distance from its center, w = unused.
    /// Equal to the tables' unless the atmosphere has an apparent radius.
    pub view_ground: [f32; 4],
    /// xyz = unit direction toward the sun, w = cosine of the disc's radius.
    pub sun_direction: [f32; 4],
    /// xyz = the sunlight above the air, w = unused.
    pub sun_illuminance: [f32; 4],
    /// xyz = the sun disc's radiance as seen from the camera, w = unused.
    pub sun_disc: [f32; 4],
    /// xyz = the camera's world position, w = unused.
    pub cam_pos: [f32; 4],
    /// xy = the target's size in pixels, zw = its reciprocal.
    pub viewport: [f32; 4],
    /// Rows of the rotation into the environment cubes' frame
    /// (`SkyOrientation::sample_rows`): the irradiance cube is written in the
    /// frame every cube sample is taken in.
    pub sky_rot: [[f32; 4]; 3],
    /// The night sky (`NightLanes::night`): star and Milky Way brightness,
    /// the faintest magnitude and the galaxy count; all zero without one.
    pub night: [f32; 4],
    /// xyz = the Milky Way's pole on the celestial sphere, w = the seed.
    pub galactic_pole: [f32; 4],
    /// xyz = the direction of its core, w = stars per steradian.
    pub galactic_center: [f32; 4],
    /// xyz = toward the moon, w = cosine of its radius (2 without one).
    pub moon_direction: [f32; 4],
    /// xyz = the moon's face lit head-on, w = its dark side's share.
    pub moon_face: [f32; 4],
    /// xyz = up across the moon's face, w = 1 when the night sky draws.
    pub moon_up: [f32; 4],
    /// Two lanes per galaxy: direction and brightness, then long axis and
    /// shape.
    pub galaxies: [[f32; 4]; 2 * MAX_GALAXIES],
}

/// The view the atmosphere is drawn for.
#[derive(Debug, Clone, Copy)]
pub struct AtmosphereView {
    /// Inverse of the un-jittered view-projection.
    pub inv_vp: [[f32; 4]; 4],
    /// The camera's world position.
    pub cam_pos: [f32; 3],
    /// The HDR target's size in pixels.
    pub viewport: [f32; 2],
    /// Rows of the sky's inverse rotation.
    pub sky_rot: [[f32; 4]; 3],
}

impl AtmosphereParams {
    /// The block for `frame` drawn from `view`.
    pub fn new(s: &AtmosphereSettings, frame: &AtmosphereFrame, view: &AtmosphereView) -> Self {
        let v3 = |v: [f32; 3], w: f32| [v[0], v[1], v[2], w];
        let table = s.with_ground_radius(frame.sky_ground_radius);
        let camera = frame.table_camera_km(s);
        let disc: [f32; 3] = core::array::from_fn(|i| {
            frame.sun_illuminance[i] * frame.sun_transmittance[i] * s.sun_disc_intensity
        });
        let viewport = [view.viewport[0].max(1.0), view.viewport[1].max(1.0)];
        let night = match (&s.night, &frame.night) {
            (Some(settings), Some(night)) => {
                NightLanes::new(settings, night, frame.sun_illuminance)
            }
            _ => NightLanes::OFF,
        };
        Self {
            inv_vp: view.inv_vp,
            rayleigh: v3(s.rayleigh_scattering, -1.0 / s.rayleigh_scale_height),
            mie_scattering: v3(s.mie_scattering, -1.0 / s.mie_scale_height),
            mie_extinction: v3(s.mie_extinction, s.mie_anisotropy),
            ozone: v3(s.ozone_absorption, s.ozone_center),
            ground_albedo: v3(s.ground_albedo, s.ozone_half_width),
            radii: [
                table.bottom_radius,
                table.top_radius,
                AERIAL_MAX_DISTANCE_KM,
                1.0 / METERS_PER_KM,
            ],
            camera: v3(camera, length(camera)),
            view_ground: [s.bottom_radius, s.top_radius, length(frame.camera_km), 0.0],
            sun_direction: v3(frame.sun_direction, s.sun_cos_radius),
            sun_illuminance: v3(frame.sun_illuminance, 0.0),
            sun_disc: v3(disc, 0.0),
            cam_pos: v3(view.cam_pos, 0.0),
            viewport: [
                viewport[0],
                viewport[1],
                1.0 / viewport[0],
                1.0 / viewport[1],
            ],
            sky_rot: view.sky_rot,
            night: night.night,
            galactic_pole: night.galactic_pole,
            galactic_center: night.galactic_center,
            moon_direction: night.moon_direction,
            moon_face: night.moon_face,
            moon_up: night.moon_up,
            galaxies: night.galaxies,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Atmosphere, DirectionalLight};
    use core::mem::{offset_of, size_of};

    // Layout must match `AtmosphereParams` in atmosphere_types.hlsl: every
    // member a float4 lane, so MSL, SPIR-V and DXIL agree.
    #[test]
    fn atmosphere_params_layout_matches_the_shader() {
        assert_eq!(size_of::<AtmosphereParams>(), 928);
        assert_eq!(offset_of!(AtmosphereParams, inv_vp), 0);
        assert_eq!(offset_of!(AtmosphereParams, rayleigh), 64);
        assert_eq!(offset_of!(AtmosphereParams, mie_scattering), 80);
        assert_eq!(offset_of!(AtmosphereParams, mie_extinction), 96);
        assert_eq!(offset_of!(AtmosphereParams, ozone), 112);
        assert_eq!(offset_of!(AtmosphereParams, ground_albedo), 128);
        assert_eq!(offset_of!(AtmosphereParams, radii), 144);
        assert_eq!(offset_of!(AtmosphereParams, camera), 160);
        assert_eq!(offset_of!(AtmosphereParams, view_ground), 176);
        assert_eq!(offset_of!(AtmosphereParams, sun_direction), 192);
        assert_eq!(offset_of!(AtmosphereParams, sun_illuminance), 208);
        assert_eq!(offset_of!(AtmosphereParams, sun_disc), 224);
        assert_eq!(offset_of!(AtmosphereParams, cam_pos), 240);
        assert_eq!(offset_of!(AtmosphereParams, viewport), 256);
        assert_eq!(offset_of!(AtmosphereParams, sky_rot), 272);
        assert_eq!(offset_of!(AtmosphereParams, night), 320);
        assert_eq!(offset_of!(AtmosphereParams, galactic_pole), 336);
        assert_eq!(offset_of!(AtmosphereParams, galactic_center), 352);
        assert_eq!(offset_of!(AtmosphereParams, moon_direction), 368);
        assert_eq!(offset_of!(AtmosphereParams, moon_face), 384);
        assert_eq!(offset_of!(AtmosphereParams, moon_up), 400);
        assert_eq!(offset_of!(AtmosphereParams, galaxies), 416);
    }

    #[test]
    fn the_block_carries_the_settings_and_the_frame() {
        let s = AtmosphereSettings::resolve(&Atmosphere::default(), None);
        let sun = DirectionalLight {
            direction: [0.0, 1.0, 0.0],
            color: [1.0; 3],
            intensity: 2.0,
        };
        let frame = AtmosphereFrame::new(&s, [0.0, 6360.5, 0.0], &sun);
        let p = AtmosphereParams::new(
            &s,
            &frame,
            &AtmosphereView {
                inv_vp: crate::transform::IDENTITY,
                cam_pos: [1.0, 2.0, 3.0],
                viewport: [1600.0, 900.0],
                sky_rot: crate::sky::SkyOrientation::IDENTITY_ROWS,
            },
        );
        assert_eq!(p.radii[0], 6360.0);
        assert!((p.rayleigh[3] + 1.0 / 8.0).abs() < 1e-7);
        assert_eq!(p.camera[3], 6360.5);
        // Without an apparent radius the sky is seen over the tables' planet.
        assert_eq!(p.view_ground, [p.radii[0], p.radii[1], p.camera[3], 0.0]);

        assert_eq!(p.viewport[2], 1.0 / 1600.0);
        // No night sky: its lanes are off and no direction reaches the moon.
        assert_eq!(p.moon_up[3], 0.0);
        assert_eq!(p.moon_direction[3], 2.0);
        assert_eq!(p.night, [0.0; 4]);
        // The disc is the sunlight through the air, times its brightness.
        let expected = 2.0 * frame.sun_transmittance[0] * s.sun_disc_intensity;
        assert!((p.sun_disc[0] - expected).abs() < 1e-3);
    }

    // With an apparent radius the tables sit over the bigger ground, the
    // camera as high above it as above the true one, and the view lane keeps
    // the true planet.
    #[test]
    fn an_apparent_radius_splits_the_tables_from_the_view() {
        let s = AtmosphereSettings::resolve(
            &Atmosphere {
                apparent_radius_km: Some(6360.0),
                ..Default::default()
            },
            Some(([0.0, -50_000.0, 0.0], 50_000.0)),
        );
        let sun = DirectionalLight::default();
        let frame = AtmosphereFrame::new(&s, [0.0, 50.1, 0.0], &sun);
        let p = AtmosphereParams::new(
            &s,
            &frame,
            &AtmosphereView {
                inv_vp: crate::transform::IDENTITY,
                cam_pos: [0.0; 3],
                viewport: [1.0, 1.0],
                sky_rot: crate::sky::SkyOrientation::IDENTITY_ROWS,
            },
        );
        assert_eq!([p.radii[0], p.radii[1]], [6360.0, 6460.0]);
        assert!((p.camera[3] - 6360.1).abs() < 1e-2, "{:?}", p.camera);
        assert_eq!([p.view_ground[0], p.view_ground[1]], [50.0, 150.0]);
        assert!((p.view_ground[2] - 50.1).abs() < 1e-4);
    }
}

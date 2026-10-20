//! A physically based sky: the `Atmosphere` a world declares, resolved into the numbers
//! the renderer draws it from, and the CPU halves of what the shaders compute.
//!
//! The technique is Hillaire's (EGSR 2020): a few small lookup tables built on
//! the GPU every frame from the air's scattering coefficients, then read by
//! the passes that need the sky. The transmittance table holds how much light
//! survives a straight path out of the air from any height and angle; the
//! multiple-scattering table holds the light scattered more than once, so one
//! scattering event per ray accounts for all orders; the sky-view table holds
//! the sky's radiance around the camera, read by the sky draw; the aerial
//! volume holds the light scattered into, and the light lost from, every view
//! ray out to some distance, read when surfaces are hazed. A final step
//! projects the sky onto spherical harmonics and writes the irradiance cube
//! the surfaces take their ambient light from.
//!
//! Every length here is in kilometers, as in the literature: scattering
//! coefficients are per kilometer, and the planet and camera are placed in
//! kilometers about the planet's center.
//!
//! The irradiance cube carries the sky's diffuse light; its reflections come
//! from a small cube the sky is drawn into and prefiltered like a reflection
//! probe, redrawn only as the sky moves (`sky_cube`). A small planet can have
//! its sky drawn as if over a bigger one near the ground (`apparent`).
//!
//! The shader side is `shaders/atmosphere_common.hlsl` (shared math) and

//! `shaders/atmosphere.hlsl` (the table kernels and the haze pass). The
//! table parameterizations, the transmittance integral and the spherical
//! harmonics are mirrored here line for line, so they are tested on the CPU,
//! and the sun's color is computed from the same transmittance the GPU table
//! holds.

mod air;
mod air_tables;
mod apparent;
mod frame;
mod lut;
mod optics;
mod params;
mod readiness;
mod settings;
mod sh;
mod sky_cube;

pub use air::AirProfile;
pub use air_tables::AirTables;
pub use apparent::{ApparentGround, table_view_angle, zenith_horizon};
pub use frame::{AtmosphereFrame, sun_light};
pub use lut::{
    AERIAL_MAX_DISTANCE_KM, AERIAL_SIZE, IRRADIANCE_FACE, MULTISCATTER_SIZE, SKY_CUBE_FACE,
    SKY_VIEW_HEIGHT, SKY_VIEW_WIDTH, TRANSMITTANCE_HEIGHT, TRANSMITTANCE_WIDTH,
    aerial_slice_distance, aerial_slice_of, aerial_span, bspline_taps, sky_angles,
    sky_view_direction, sky_view_row_taps, sky_view_uv, transmittance_r_mu, transmittance_uv,
};
pub use optics::{
    disc_set_depth, disc_visible_fraction, optical_depth, sun_visible_fraction, transmittance,
};
pub use params::{AtmosphereParams, AtmosphereView};
pub use readiness::TableReadiness;
pub use settings::AtmosphereSettings;
pub use sh::{Sh9, fibonacci_direction, sh9_basis};
pub use sky_cube::{
    SKY_CUBE_HEIGHT_FRACTION, SKY_CUBE_HEIGHT_KM, SKY_CUBE_PLAN, SKY_CUBE_TURN_DEG, SkyCubeKey,
    SkyCubeSchedule, SkyCubeWork,
};

/// The air above the ground: a physically based sky that scatters sunlight
/// into blue overhead, white haze at the horizon and red at sunset, and that
/// looks right from the ground to high above it.
///
/// The air is a shell `height_km` thick around the planet. With a
/// [Planet](#planet) in the world it wraps that planet, at its center and
/// radius. Without one, it wraps a sphere of `planet_radius_km` whose top is at
/// height 0, so a flat world sits on a gently curved ground far below it.
///
/// Two kinds of particle scatter the light. Air molecules (Rayleigh) scatter
/// short wavelengths most, which makes the sky blue and the low sun red; they
/// thin by a factor of e every `rayleigh_scale_height_km` up. Haze and dust
/// (Mie) scatter all colors alike, mostly forward, which whitens the horizon
/// and draws a glow around the sun; they hug the ground more closely. An ozone
/// layer `ozone_center_km` up absorbs a little orange, which keeps the
/// twilight sky blue rather than gray. Coefficients are per kilometer of air
/// at the ground (for ozone, at the layer's center).
///
/// The sun is the first [`DirectionalLight`](#directionallight): its direction
/// is where the sun is, and its `color * intensity` is the sunlight above the
/// air. The light that reaches the camera is that sunlight dimmed and
/// reddened by the air it crossed, and that is what lights the scene: white at
/// noon, orange and weak at sunset, gone once the sun is below the horizon. A
/// [`SkyRotation`](#skyrotation) is the day cycle: with its axis along the
/// planet's pole the sun rises and sets as the sky turns, and on a planet its
/// height also follows where on the planet the camera stands.
///
/// The sky is drawn behind everything, with the sun's disc in it. Distant
/// surfaces fade into the air's color (aerial perspective), behind any
/// [`VolumetricFog`](#volumetricfog), which stays the nearer layer. The sky
/// lights the scene in place of an [`EnvironmentMap`](#environmentmap): its
/// light fills the shadows, and shiny surfaces reflect it, following the sun
/// through the day.
///
/// One per world. The defaults are Earth's.
///
/// ```rust
/// # use concinnity_core::components::Atmosphere;
/// Atmosphere {
///     mie_scattering: [0.008, 0.008, 0.008],
///     ground_albedo: [0.2, 0.25, 0.15],
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
pub struct Atmosphere {
    /// Thickness of the air above the lowest ground, in kilometers. At
    /// least 1.
    #[asset(default = 100.0)]
    pub height_km: f32,
    /// Radius of the ground the air wraps, in kilometers, when the world has
    /// no `Planet`; a planet's own radius is used when it has one. At least 1.
    #[asset(default = 6360.0)]
    pub planet_radius_km: f32,
    /// Rayleigh (air molecule) scattering per kilometer at the ground, for
    /// red, green and blue.
    #[asset(default = [0.005_802, 0.013_558, 0.0331])]
    pub rayleigh_scattering: [f32; 3],
    /// Height over which the air thins by a factor of e, in kilometers.
    #[asset(default = 8.0)]
    pub rayleigh_scale_height_km: f32,
    /// Mie (haze) scattering per kilometer at the ground.
    #[asset(default = [0.003_996, 0.003_996, 0.003_996])]
    pub mie_scattering: [f32; 3],
    /// Mie (haze) absorption per kilometer at the ground: haze that darkens
    /// rather than brightens.
    #[asset(default = [0.000_444, 0.000_444, 0.000_444])]
    pub mie_absorption: [f32; 3],
    /// Height over which the haze thins by a factor of e, in kilometers.
    #[asset(default = 1.2)]
    pub mie_scale_height_km: f32,
    /// How strongly haze scatters forward, in `[-0.99, 0.99]`: higher draws a
    /// tighter, brighter glow around the sun.
    #[asset(default = 0.8)]
    pub mie_anisotropy: f32,
    /// Ozone absorption per kilometer at the center of the ozone layer.
    #[asset(default = [0.00065, 0.001_881, 0.000_085])]
    pub ozone_absorption: [f32; 3],
    /// Height of the ozone layer's densest point, in kilometers.
    #[asset(default = 25.0)]
    pub ozone_center_km: f32,
    /// Full width of the ozone layer, in kilometers: its density falls off
    /// linearly to nothing half this far above and below its center.
    #[asset(default = 30.0)]
    pub ozone_width_km: f32,
    /// Linear-space color of the ground as seen from far above: the light it
    /// bounces back into the air.
    #[asset(default = [0.3, 0.3, 0.3])]
    pub ground_albedo: [f32; 3],
    /// Angular diameter of the sun's disc, in degrees. Clamped to
    /// `[0.05, 10]`.
    #[asset(default = 0.53)]
    pub sun_angular_diameter_deg: f32,
    /// Brightness of the sun's disc, as a multiple of the sunlight it casts.
    /// 0 hides the disc.
    #[asset(default = 1000.0)]
    pub sun_disc_intensity: f32,
    /// Radius, in kilometers, of the planet the sky is drawn over while the
    /// camera is near the ground, in place of the true one. A small planet's
    /// own sky is a flat deep blue with a pale sun at sunset, since a ray
    /// along its horizon crosses little air; drawn over an Earth-sized radius
    /// it gets a pale horizon and red sunsets. The haze over the ground keeps
    /// its true distances, and the horizon still meets the ground where it
    /// really is. At least 1. `null` (the default) draws the sky over the
    /// true radius.
    pub apparent_radius_km: Option<f32>,
    /// Heights above the ground, in kilometers, between which the sky fades
    /// from `apparent_radius_km` to the true radius as the camera climbs, so
    /// the planet's limb seen from high up is its own. The air's heights
    /// describe it over the apparent ground and shrink in proportion with the
    /// radius, its coefficients growing to keep its optical depth.
    #[asset(default = [1.0, 20.0])]
    pub apparent_radius_fade_km: [f32; 2],
}

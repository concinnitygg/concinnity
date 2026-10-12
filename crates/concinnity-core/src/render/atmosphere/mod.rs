//! A physically based sky: the `Atmosphere` asset resolved into the numbers
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

pub use air_tables::AirTables;
pub use apparent::{ApparentGround, table_view_angle, zenith_horizon};
pub use frame::{AtmosphereFrame, sun_light};
pub use lut::{
    AERIAL_MAX_DISTANCE_KM, AERIAL_SIZE, IRRADIANCE_FACE, MULTISCATTER_SIZE, SKY_CUBE_FACE,
    SKY_VIEW_HEIGHT, SKY_VIEW_WIDTH, TRANSMITTANCE_HEIGHT, TRANSMITTANCE_WIDTH,
    aerial_slice_distance, aerial_slice_of, sky_view_direction, sky_view_uv, transmittance_r_mu,
    transmittance_uv,
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

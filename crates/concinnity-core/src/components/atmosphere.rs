// Physically based sky schema.

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
/// The sun is the first [DirectionalLight](#directionallight): its direction
/// is where the sun is, and its `color * intensity` is the sunlight above the
/// air. The light that reaches the camera is that sunlight dimmed and
/// reddened by the air it crossed, and that is what lights the scene: white at
/// noon, orange and weak at sunset, gone once the sun is below the horizon. A
/// [SkyRotation](#skyrotation) is the day cycle: with its axis along the
/// planet's pole the sun rises and sets as the sky turns, and on a planet its
/// height also follows where on the planet the camera stands.
///
/// The sky is drawn behind everything, with the sun's disc in it. Distant
/// surfaces fade into the air's color (aerial perspective), behind any
/// [VolumetricFog](#volumetricfog), which stays the nearer layer. The sky
/// lights the scene in place of an [EnvironmentMap](#environmentmap): its
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
    #[asset(default = [0.005802, 0.013558, 0.0331])]
    pub rayleigh_scattering: [f32; 3],
    /// Height over which the air thins by a factor of e, in kilometers.
    #[asset(default = 8.0)]
    pub rayleigh_scale_height_km: f32,
    /// Mie (haze) scattering per kilometer at the ground.
    #[asset(default = [0.003996, 0.003996, 0.003996])]
    pub mie_scattering: [f32; 3],
    /// Mie (haze) absorption per kilometer at the ground: haze that darkens
    /// rather than brightens.
    #[asset(default = [0.000444, 0.000444, 0.000444])]
    pub mie_absorption: [f32; 3],
    /// Height over which the haze thins by a factor of e, in kilometers.
    #[asset(default = 1.2)]
    pub mie_scale_height_km: f32,
    /// How strongly haze scatters forward, in `[-0.99, 0.99]`: higher draws a
    /// tighter, brighter glow around the sun.
    #[asset(default = 0.8)]
    pub mie_anisotropy: f32,
    /// Ozone absorption per kilometer at the center of the ozone layer.
    #[asset(default = [0.00065, 0.001881, 0.000085])]
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

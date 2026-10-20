// Point-light schema.

/// A spherical point light with quadratic distance attenuation.
///
/// The forward renderer lights every surface from all declared point lights (up
/// to a large per-scene budget). Secondary effects (volumetric fog, SDF
/// raymarching, and reflection-probe capture) still consider only the first 8.
///
/// ```rust
/// # use concinnity_core::components::PointLight;
/// PointLight {
///     position: [2.0, 2.5, -3.0],
///     color: [1.0, 0.8, 0.5],
///     intensity: 8.0,
///     range: 6.0,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct PointLight {
    /// World-space position of the light source.
    #[asset(default = [0.0, 2.5, 0.0])]
    pub position: [f32; 3],
    /// Linear-space RGB color of the light.
    #[asset(default = [1.0, 1.0, 1.0])]
    pub color: [f32; 3],
    /// Intensity multiplier applied to the color.
    #[asset(default = 8.0)]
    pub intensity: f32,
    /// Maximum reach in world units; attenuation is zero at this distance.
    #[asset(default = 6.0)]
    pub range: f32,
}

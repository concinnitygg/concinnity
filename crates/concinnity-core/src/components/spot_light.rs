// Spot-light schema.

/// A cone-shaped local light: a point light restricted to the cone around
/// `direction`, with a soft edge between `inner_angle` and `outer_angle`.
///
/// Distance attenuation matches [PointLight](#pointlight); the cone adds an
/// angular falloff that is full brightness inside the inner cone and fades to
/// black at the outer cone. Spot lights share the same per-scene local-light
/// budget as point lights and are culled by the same clustered pass. Secondary
/// effects (volumetric fog, SDF raymarching, and reflection-probe capture) do
/// not consider them.
///
/// ```rust
/// # use concinnity_core::components::SpotLight;
/// SpotLight {
///     position: [0.0, 4.0, -2.0],
///     direction: [0.0, -1.0, 0.0],
///     color: [1.0, 0.9, 0.7],
///     intensity: 20.0,
///     range: 10.0,
///     inner_angle: 18.0,
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
pub struct SpotLight {
    /// World-space position of the light source.
    #[asset(default = [0.0, 4.0, 0.0])]
    pub position: [f32; 3],
    /// Direction the cone points, away from the light. Does not need to be
    /// normalized; defaults to straight down when degenerate.
    #[asset(default = [0.0, -1.0, 0.0])]
    pub direction: [f32; 3],
    /// Linear-space RGB color of the light.
    #[asset(default = [1.0, 1.0, 1.0])]
    pub color: [f32; 3],
    /// Intensity multiplier applied to the color.
    #[asset(default = 20.0)]
    pub intensity: f32,
    /// Maximum reach in world units; attenuation is zero at this distance.
    #[asset(default = 10.0)]
    pub range: f32,
    /// Half-angle in degrees of the fully lit inner cone. Clamped to
    /// `outer_angle`.
    #[asset(default = 18.0)]
    pub inner_angle: f32,
    /// Half-angle in degrees at which the cone fades to black. Clamped to
    /// (0, 89.9].
    #[asset(default = 30.0)]
    pub outer_angle: f32,
    /// Whether this light casts shadows. Shadowed spots claim one slice of the
    /// spot shadow map in declaration order; once the slices are used up the
    /// remaining spots still light the scene but cast nothing.
    #[asset(default = true)]
    pub cast_shadows: bool,
}

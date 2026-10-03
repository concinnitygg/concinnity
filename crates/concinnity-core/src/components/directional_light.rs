// Directional-light schema.

/// An infinitely distant directional light (sun, moon, or sky fill).
///
/// Up to 4 directional lights may be declared; extras beyond 4 are silently ignored.
/// When no directional light is present, a built-in warm sun is used as a fallback.
///
/// ```rust
/// # use concinnity_core::components::DirectionalLight;
/// DirectionalLight {
///     direction: [-0.3, 0.85, 0.4],
///     color: [1.0, 0.95, 0.8],
///     intensity: 1.0,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct DirectionalLight {
    /// Direction pointing toward the light source. Does not need to be
    /// normalized.
    #[asset(default = [-0.3, 0.85, 0.4])]
    pub direction: [f32; 3],
    /// Linear-space RGB color of the light.
    #[asset(default = [1.0, 1.0, 1.0])]
    pub color: [f32; 3],
    /// Intensity multiplier applied to the color.
    #[asset(default = 1.0)]
    pub intensity: f32,
}

impl DirectionalLight {
    /// A light contributing nothing, used to pad a fixed-size set.
    pub const ZERO: Self = Self {
        direction: [0.0; 3],
        color: [0.0; 3],
        intensity: 0.0,
    };
}

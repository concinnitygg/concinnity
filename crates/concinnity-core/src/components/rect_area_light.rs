// Rectangular area-light schema.

/// A rectangular area light: a glowing panel that lights the scene from its
/// whole surface rather than from a single point.
///
/// Unlike a [PointLight](#pointlight) or [SpotLight](#spotlight), the softness of
/// the shadow terminator and the shape of the specular highlight follow the
/// panel's real dimensions, so a wide softbox wraps light around a surface and
/// leaves a stretched rectangular reflection on glossy materials. Use it for
/// windows, ceiling panels, screens, and practical lights.
///
/// The panel is positioned by `center`, oriented by `normal` (the direction it
/// emits), and sized by `half_size`, matching [GlassPanel](#glasspanel).
///
/// ```rust
/// # use concinnity_core::components::RectAreaLight;
/// RectAreaLight {
///     center: [0.0, 3.0, -4.0],
///     normal: [0.0, 0.0, 1.0],
///     half_size: [1.5, 1.0],
///     color: [1.0, 0.95, 0.85],
///     intensity: 12.0,
///     range: 18.0,
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
pub struct RectAreaLight {
    /// World-space position of the panel's center.
    #[asset(default = [0.0, 3.0, 0.0])]
    pub center: [f32; 3],
    /// Direction the panel emits. Normalized on load; defaults to `+Z` when
    /// degenerate.
    #[asset(default = [0.0, -1.0, 0.0])]
    pub normal: [f32; 3],
    /// Half-width and half-height of the panel, in world units.
    #[asset(default = [1.0, 1.0])]
    pub half_size: [f32; 2],
    /// Linear-space RGB color of the light.
    #[asset(default = [1.0, 1.0, 1.0])]
    pub color: [f32; 3],
    /// Intensity multiplier applied to the color.
    #[asset(default = 12.0)]
    pub intensity: f32,
    /// Maximum reach in world units; attenuation is zero at this distance.
    #[asset(default = 18.0)]
    pub range: f32,
    /// When true the panel emits from both faces. A one-sided panel lights only
    /// the half-space its `normal` points into.
    pub two_sided: bool,
}

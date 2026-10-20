// Screen-space ambient-occlusion schema.

/// Screen-space ambient occlusion: darkens creases and contact areas where
/// ambient light is occluded. The `ambient_occlusion` section of
/// [`GraphicsConfig`](#graphicsconfig); a field left unset takes the default
/// below.
///
/// On by default, which is the look on capable hardware: the `Auto` graphics
/// quality preset forces it off on the lowest tier. A ceiling only ever
/// reduces, so turning it off here holds on every tier.
///
/// ```rust
/// # use concinnity_core::components::{AmbientOcclusionOverrides, GraphicsConfig};
/// GraphicsConfig {
///     ambient_occlusion: AmbientOcclusionOverrides {
///         radius: Some(1.0),
///         ..Default::default()
///     },
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
#[asset(section)]
#[serde(default)]
pub struct AmbientOcclusion {
    /// Whether the occlusion pass runs.
    #[asset(default = true, apply = rebuild)]
    pub enabled: bool,
    /// How far the search reaches for occluders, in world units. Larger values
    /// pick up broader, softer occlusion.
    #[asset(default = 0.5, min = 0.001, apply = live)]
    pub radius: f32,
    /// Occlusion strength. 1.0 is the natural amount; higher values exaggerate
    /// the contact darkening.
    #[asset(default = 1.0, min = 0.0, max = 4.0, apply = live)]
    pub intensity: f32,
}

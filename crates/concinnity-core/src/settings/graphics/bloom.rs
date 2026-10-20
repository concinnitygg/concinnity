// Bloom schema.

/// Bloom: light from the brightest parts of the image spilling softly into
/// their surroundings. The `bloom` section of [`GraphicsConfig`](#graphicsconfig);
/// a field left unset takes the default below.
///
/// ```rust
/// # use concinnity_core::components::{BloomOverrides, GraphicsConfig};
/// GraphicsConfig {
///     bloom: BloomOverrides {
///         intensity: Some(0.3),
///         threshold: Some(1.2),
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
pub struct Bloom {
    /// Additive bloom contribution. 0 skips bloom entirely.
    #[asset(default = 0.6, min = 0.0, apply = live)]
    pub intensity: f32,
    /// Brightness threshold. Pixels brighter than this contribute fully; pixels
    /// within `knee` below it ramp in softly.
    #[asset(default = 1.0, min = 0.0, apply = live)]
    pub threshold: f32,
    /// Width of the soft knee just below `threshold`.
    #[asset(default = 0.5, min = 0.0, apply = live)]
    pub knee: f32,
}

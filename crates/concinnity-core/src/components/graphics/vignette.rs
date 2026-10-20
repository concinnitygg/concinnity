// Vignette schema.

/// Darkening toward the corners of the frame. The `vignette` section of
/// [`GraphicsConfig`](#graphicsconfig); left unset, there is no vignette.
///
/// ```rust
/// # use concinnity_core::components::{GraphicsConfig, VignetteOverrides};
/// GraphicsConfig {
///     vignette: VignetteOverrides {
///         strength: Some(0.3),
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
pub struct Vignette {
    /// Vignette strength in `[0, 1]`. 0 disables the corner darkening.
    #[asset(min = 0.0, max = 1.0, apply = live)]
    pub strength: f32,
}

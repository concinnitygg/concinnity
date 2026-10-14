// Background color schema.

/// What the frame shows where no surface is drawn and no
/// [EnvironmentMap](#environmentmap) serves as the backdrop. The `background`
/// section of [GraphicsConfig](#graphicsconfig); left unset, the default below
/// applies.
///
/// ```rust
/// # use concinnity_core::components::{BackgroundOverrides, GraphicsConfig};
/// GraphicsConfig {
///     background: BackgroundOverrides {
///         color: Some([0.1, 0.1, 0.15, 1.0]),
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
pub struct Background {
    /// Background color [r, g, b, a] in linear 0..1 space.
    #[asset(default = [0.01, 0.01, 0.02, 1.0], apply = restart)]
    pub color: [f32; 4],
}

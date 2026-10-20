// Scene texture-filtering schema.

/// How the scene's surface textures (albedo and normal maps) are filtered. The
/// `texture_filtering` section of [`GraphicsConfig`](#graphicsconfig); a field
/// left unset takes the default below, capped by the quality tier.
///
/// The default describes the quality capable hardware runs: the `Auto`
/// graphics quality preset resolves the detected GPU into a ceiling that caps
/// it tier by tier.
///
/// ```rust
/// # use concinnity_core::components::{GraphicsConfig, TextureFilteringOverrides};
/// GraphicsConfig {
///     texture_filtering: TextureFilteringOverrides {
///         anisotropy: Some(8),
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
pub struct TextureFiltering {
    /// Maximum anisotropic-filtering degree, e.g. 8. Higher keeps textures
    /// viewed at a grazing angle (floors, walls receding into the distance)
    /// sharp instead of blurring along the minor axis, at a small sampling
    /// cost. `1` disables anisotropy (plain trilinear). `16` by default, capped
    /// by the quality preset down to 4 on the lowest tier.
    #[asset(default = 16, min = 1, max = 16, apply = restart)]
    pub anisotropy: u32,
}

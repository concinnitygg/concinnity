// Scene texture-filtering schema.

/// How the scene's surface textures (albedo and normal maps) are filtered.
/// One per world; with none declared, the defaults below apply.
///
/// The default describes the quality capable hardware runs: the `Auto`
/// graphics quality preset resolves the detected GPU into a ceiling that caps
/// it tier by tier.
///
/// ```rust
/// # use concinnity_core::components::TextureFiltering;
/// TextureFiltering { anisotropy: 8 };
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
pub struct TextureFiltering {
    /// Maximum anisotropic-filtering degree, e.g. 8. Higher keeps textures
    /// viewed at a grazing angle (floors, walls receding into the distance)
    /// sharp instead of blurring along the minor axis, at a small sampling
    /// cost. `1` disables anisotropy (plain trilinear). `16` by default, capped
    /// by the quality preset down to 4 on the lowest tier.
    #[asset(default = 16, min = 1, max = 16, apply = restart)]
    pub anisotropy: u32,
}

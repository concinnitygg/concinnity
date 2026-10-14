// Background color schema.

/// What the frame shows where no surface is drawn and no
/// [EnvironmentMap](#environmentmap) serves as the backdrop. One per world;
/// with none declared, the default below applies.
///
/// ```rust
/// # use concinnity_core::components::Background;
/// Background {
///     color: [0.1, 0.1, 0.15, 1.0],
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
pub struct Background {
    /// Background color [r, g, b, a] in linear 0..1 space.
    #[asset(default = [0.01, 0.01, 0.02, 1.0], apply = restart)]
    pub color: [f32; 4],
}

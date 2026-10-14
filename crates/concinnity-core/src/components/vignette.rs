// Vignette schema.

/// Darkening toward the corners of the frame. One per world; with none
/// declared, there is no vignette.
///
/// ```rust
/// # use concinnity_core::components::Vignette;
/// Vignette { strength: 0.3 };
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
pub struct Vignette {
    /// Vignette strength in `[0, 1]`. 0 disables the corner darkening.
    #[asset(min = 0.0, max = 1.0, apply = live)]
    pub strength: f32,
}

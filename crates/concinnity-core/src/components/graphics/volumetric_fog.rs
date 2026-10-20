// Environmental volumetric fog schema.

/// Environmental volumetric fog: a single lit medium that wraps the scene,
/// thicker near the ground and thinning with height, with extra glow around the
/// sun.
///
/// Only one `VolumetricFog` is honored: the first declared instance wins;
/// later instances are silently dropped. With none declared, there is no fog.
///
/// ```rust
/// # use concinnity_core::components::VolumetricFog;
/// VolumetricFog {
///     density: 0.08,
///     color: [0.75, 0.82, 0.95],
///     height_falloff: 0.18,
///     max_distance: 160.0,
///     phase_g: 0.5,
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
pub struct VolumetricFog {
    /// Master toggle. `false` disables the fog even when this asset is present.
    #[asset(default = true)]
    pub enabled: bool,
    /// Linear-space RGB tint of the fog: the color the camera sees in the far
    /// distance.
    #[asset(default = [0.7, 0.78, 0.85])]
    pub color: [f32; 3],
    /// Base thickness of the fog at `height_reference` (per world unit). Higher
    /// is thicker. Floored at 0.
    #[asset(default = 0.05)]
    pub density: f32,
    /// How quickly the fog thins with height above `height_reference`. 0 keeps
    /// it uniform; larger values pin it to the ground.
    #[asset(default = 0.2)]
    pub height_falloff: f32,
    /// World-space Y at which the fog reaches full `density`. It thickens below
    /// this height and thins above it.
    pub height_reference: f32,
    /// Maximum distance the fog covers from the camera, in world units. Past
    /// this, distant geometry stays clear.
    #[asset(default = 200.0)]
    pub max_distance: f32,
    /// Sun-glow anisotropy in `(-1, 1)`. Positive values concentrate brightness
    /// around the sun (haloes), negative values scatter away from it, 0 is
    /// uniform.
    #[asset(default = 0.4)]
    pub phase_g: f32,
    /// Constant ambient brightness so the fog keeps some color in shaded areas.
    #[asset(default = 0.15)]
    pub ambient: f32,
}

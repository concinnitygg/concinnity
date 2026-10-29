// Visibility-culling schema.

/// How the renderer culls objects the camera cannot see. The `culling` section
/// of [`GraphicsConfig`](#graphicsconfig); a field left unset takes the default
/// below.
///
/// ```rust
/// # use concinnity_engine::components::{CullingOverrides, GraphicsConfig};
/// GraphicsConfig {
///     culling: CullingOverrides {
///         two_pass_occlusion: Some(false),
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
pub struct Culling {
    /// Two-pass occlusion culling. Reduces objects popping in a frame late when
    /// they're revealed by camera or occluder motion, at the cost of extra
    /// culling work each frame. Needs the bindless GPU-cull path.
    #[asset(default = true, apply = restart)]
    pub two_pass_occlusion: bool,
}

// Visibility-culling schema.

/// How the renderer culls objects the camera cannot see. One per world; with
/// none declared, two-pass occlusion culling is off.
///
/// ```rust
/// # use concinnity_core::components::Culling;
/// Culling {
///     two_pass_occlusion: false,
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
pub struct Culling {
    /// Two-pass occlusion culling. Reduces objects popping in a frame late when
    /// they're revealed by camera or occluder motion, at the cost of extra
    /// culling work each frame. Needs the bindless GPU-cull path.
    #[asset(default = true, apply = restart)]
    pub two_pass_occlusion: bool,
}

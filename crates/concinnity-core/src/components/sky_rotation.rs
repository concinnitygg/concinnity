// Celestial-sphere rotation schema.

/// Turns the whole celestial sphere: the sky, the image-based lighting it
/// casts, every [DirectionalLight](#directionallight), and any
/// [Prop](#prop) hung on it.
///
/// One per world. The rotation at elapsed time `t` is `angle_deg +
/// degrees_per_second * t` about `axis`, taken in the sense a planet's own
/// spin gives the sky: with the default axis a body rises from `+Z`, passes
/// overhead through `+Y`, and sets toward `-Z`.
///
/// The component's own entity carries that rotation as its transform, so a
/// `Prop` naming this asset as its `parent` orbits with the sky. Reflection
/// probes are baked once and do not turn.
///
/// ```rust
/// # use concinnity_core::components::SkyRotation;
/// SkyRotation {
///     axis: [1.0, 0.0, 0.0],
///     degrees_per_second: 3.0,
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
pub struct SkyRotation {
    /// The celestial pole in world space: the axis the sphere turns about.
    /// Does not need to be normalized.
    #[asset(default = [1.0, 0.0, 0.0])]
    pub axis: [f32; 3],
    /// Turn rate in degrees per second. Negative runs the sky backwards.
    #[asset(default = 1.0)]
    pub degrees_per_second: f32,
    /// The angle the sky starts at, in degrees.
    pub angle_deg: f32,
}

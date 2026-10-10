// World wind schema.

/// The wind blowing across the world: a steady breeze along one horizontal
/// direction, broken up by gusts that roll through with it.
///
/// One per world. Anything that sways reads the same wind, so grass, foliage
/// and other moving surfaces lean and ripple together. With none declared the
/// air is still.
///
/// Gusts are patches of stronger and weaker air, about `gust_scale` meters
/// across, carried downwind at the wind's own speed. `gustiness` sets how far
/// they swing the strength: 0 is a steady wind, 1 lets it drop to calm and
/// peak at double between gusts.
///
/// ```rust
/// # use concinnity_core::components::Wind;
/// Wind {
///     direction: [1.0, 0.3],
///     strength: 5.0,
///     gustiness: 0.6,
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
pub struct Wind {
    /// The horizontal direction the wind blows toward, `[x, z]`. Does not need
    /// to be normalized.
    #[asset(default = [1.0, 0.0])]
    pub direction: [f32; 2],
    /// Mean wind speed, in meters per second. 0 is still air.
    #[asset(default = 3.0)]
    pub strength: f32,
    /// How strongly gusts vary the speed, in [0, 1].
    #[asset(default = 0.5)]
    pub gustiness: f32,
    /// Typical width of one gust, in meters.
    #[asset(default = 20.0)]
    pub gust_scale: f32,
}

impl Wind {
    /// The blowing direction as a unit `[x, z]` vector, or `+X` when the
    /// authored direction has no length.
    pub fn unit_direction(&self) -> [f32; 2] {
        let [x, z] = self.direction;
        let len = crate::math::hypot(x, z);
        if len > 1e-6 && len.is_finite() {
            [x / len, z / len]
        } else {
            [1.0, 0.0]
        }
    }
}

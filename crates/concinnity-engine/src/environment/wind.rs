//! The world's wind: the `Wind` a world declares, and that wind as the shaders
//! read it, the two `float4` rows `wind.hlsl` evaluates gusts from.

/// The wind every swaying surface samples, resolved from the world's `Wind`.
/// Still air when the world declares none.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindField {
    /// Unit horizontal direction the wind blows toward, `[x, z]`.
    pub direction: [f32; 2],
    /// Mean speed, in meters per second.
    pub strength: f32,
    /// How strongly gusts vary the speed, in [0, 1].
    pub gustiness: f32,
    /// Typical gust width, in meters.
    pub gust_scale: f32,
}

impl WindField {
    /// Still air: no speed and no gusts.
    pub const STILL: Self = Self {
        direction: [1.0, 0.0],
        strength: 0.0,
        gustiness: 0.0,
        gust_scale: 20.0,
    };

    /// The field `wind` describes, or still air for `None`. Clamped the way
    /// the asset validator clamps it, so an unvalidated runtime value reads the
    /// same as a baked one.
    pub fn new(wind: Option<&Wind>) -> Self {
        match wind {
            Some(w) => {
                let w = crate::components::validate::wind(w.clone());
                Self {
                    direction: w.direction,
                    strength: w.strength,
                    gustiness: w.gustiness,
                    gust_scale: w.gust_scale,
                }
            }
            None => Self::STILL,
        }
    }

    /// The two rows `wind.hlsl` reads: `(direction.x, direction.z, strength,
    /// gustiness)` and `(1 / gust_scale, 0, 0, 0)`.
    pub fn gpu_rows(&self) -> [[f32; 4]; 2] {
        [
            [
                self.direction[0],
                self.direction[1],
                self.strength,
                self.gustiness,
            ],
            [1.0 / self.gust_scale, 0.0, 0.0, 0.0],
        ]
    }
}

/// The wind blowing across the world: a steady breeze along one horizontal
/// direction, broken up by gusts that roll through with it.
///
/// One per world. Anything that moves with it reads the same wind, so grass
/// and foliage lean and water roughens together, the same gusts passing over
/// each. With none declared the air is still.
///
/// Gusts are patches of stronger and weaker air, about `gust_scale` meters
/// across, carried downwind at the wind's own speed. `gustiness` sets how far
/// they swing the strength: 0 is a steady wind, 1 lets it drop to calm and
/// peak at double between gusts.
///
/// ```rust
/// # use concinnity_engine::components::Wind;
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
        let len = concinnity_core::math::hypot(x, z);
        if len > 1e-6 && len.is_finite() {
            [x / len, z / len]
        } else {
            [1.0, 0.0]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_wind_is_still_air() {
        assert_eq!(WindField::new(None), WindField::STILL);
        assert_eq!(WindField::STILL.gpu_rows()[0][2], 0.0);
    }

    #[test]
    fn a_declared_wind_is_normalized_and_packed() {
        let field = WindField::new(Some(&Wind {
            direction: [0.0, -2.0],
            strength: 4.0,
            gustiness: 0.25,
            gust_scale: 8.0,
        }));
        assert_eq!(field.direction, [0.0, -1.0]);
        assert_eq!(
            field.gpu_rows(),
            [[0.0, -1.0, 4.0, 0.25], [0.125, 0.0, 0.0, 0.0]]
        );
    }

    #[test]
    fn a_degenerate_gust_scale_never_divides_by_zero() {
        let field = WindField::new(Some(&Wind {
            gust_scale: 0.0,
            ..Wind::default()
        }));
        assert!(field.gpu_rows()[1][0].is_finite());
    }
}

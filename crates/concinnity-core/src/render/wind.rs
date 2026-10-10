//! The world's wind as the shaders read it: a `Wind` asset resolved into the
//! two `float4` rows `wind.hlsl` evaluates gusts from.

use crate::components::Wind;

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

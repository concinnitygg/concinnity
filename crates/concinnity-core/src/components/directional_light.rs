// Directional-light schema.

/// An infinitely distant directional light (sun, moon, or sky fill).
///
/// Up to 4 directional lights may be declared; extras beyond 4 are silently ignored.
/// When no directional light is present, a built-in warm sun is used as a fallback.
///
/// ```rust
/// # use concinnity_core::components::DirectionalLight;
/// DirectionalLight {
///     direction: [-0.3, 0.85, 0.4],
///     color: [1.0, 0.95, 0.8],
///     intensity: 1.0,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct DirectionalLight {
    /// Direction pointing toward the light source. Does not need to be
    /// normalised.
    pub direction: [f32; 3],
    /// Linear-space RGB colour of the light.
    pub color: [f32; 3],
    /// Intensity multiplier applied to the colour.
    pub intensity: f32,
}

impl DirectionalLight {
    /// A light contributing nothing, used to pad a fixed-size set.
    pub const ZERO: Self = Self {
        direction: [0.0; 3],
        color: [0.0; 3],
        intensity: 0.0,
    };
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: [-0.3, 0.85, 0.4],
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_sun_points_down_from_above() {
        // The direction is toward the light, so a positive Y component is what
        // makes the default read as an overhead sun rather than an uplight.
        let l = DirectionalLight::default();
        assert!(l.direction[1] > 0.0);
        assert_eq!(l.color, [1.0, 1.0, 1.0]);
        assert_eq!(l.intensity, 1.0);
    }

    #[test]
    fn an_authored_sun_parses_and_round_trips_through_postcard() {
        let l: DirectionalLight =
            serde_json::from_str(r#"{"direction":[0,1,0],"color":[1,0.9,0.7],"intensity":3}"#)
                .unwrap();
        assert_eq!(l.direction, [0.0, 1.0, 0.0]);
        assert_eq!(l.color, [1.0, 0.9, 0.7]);
        assert_eq!(l.intensity, 3.0);

        let bytes = postcard::to_allocvec(&l).unwrap();
        let back: DirectionalLight = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.color, [1.0, 0.9, 0.7]);
        assert_eq!(back.intensity, 3.0);
    }
}

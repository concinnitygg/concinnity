// Spot-light schema.

/// A cone-shaped local light: a point light restricted to the cone around
/// `direction`, with a soft edge between `inner_angle` and `outer_angle`.
///
/// Distance attenuation matches [PointLight](#pointlight); the cone adds an
/// angular falloff that is full brightness inside the inner cone and fades to
/// black at the outer cone. Spot lights share the same per-scene local-light
/// budget as point lights and are culled by the same clustered pass. Secondary
/// effects (volumetric fog, SDF raymarching, and reflection-probe capture) do
/// not consider them.
///
/// ```rust
/// # use concinnity_asset::SpotLight;
/// SpotLight {
///     position: [0.0, 4.0, -2.0],
///     direction: [0.0, -1.0, 0.0],
///     color: [1.0, 0.9, 0.7],
///     intensity: 20.0,
///     range: 10.0,
///     inner_angle: 18.0,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SpotLight {
    /// World-space position of the light source.
    pub position: [f32; 3],
    /// Direction the cone points, away from the light. Does not need to be
    /// normalised; defaults to straight down when degenerate.
    pub direction: [f32; 3],
    /// Linear-space RGB colour of the light.
    pub color: [f32; 3],
    /// Intensity multiplier applied to the colour.
    pub intensity: f32,
    /// Maximum reach in world units; attenuation is zero at this distance.
    pub range: f32,
    /// Half-angle in degrees of the fully lit inner cone. Clamped to
    /// `outer_angle`.
    pub inner_angle: f32,
    /// Half-angle in degrees at which the cone fades to black. Clamped to
    /// (0, 89.9].
    pub outer_angle: f32,
    /// Whether this light casts shadows. Shadowed spots claim one slice of the
    /// spot shadow map in declaration order; once the slices are used up the
    /// remaining spots still light the scene but cast nothing.
    pub cast_shadows: bool,
}

impl Default for SpotLight {
    fn default() -> Self {
        Self {
            position: [0.0, 4.0, 0.0],
            direction: [0.0, -1.0, 0.0],
            color: [1.0, 1.0, 1.0],
            intensity: 20.0,
            range: 10.0,
            inner_angle: 18.0,
            outer_angle: 30.0,
            cast_shadows: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_spot_points_down_with_a_soft_edged_cone() {
        let l = SpotLight::default();
        assert_eq!(l.position, [0.0, 4.0, 0.0]);
        assert_eq!(l.direction, [0.0, -1.0, 0.0]);
        // The inner cone is narrower than the outer one, so the falloff band
        // exists and the edge is not a hard circle.
        assert!(l.inner_angle < l.outer_angle);
        assert_eq!(l.intensity, 20.0);
        assert_eq!(l.range, 10.0);
        assert!(l.cast_shadows);
    }

    #[test]
    fn an_authored_spot_parses_and_round_trips_through_postcard() {
        let l: SpotLight = serde_json::from_str(
            r#"{"position":[2,3,-1],"direction":[0,-1,0.5],"color":[1,0.9,0.7],
                "intensity":45,"range":18,"inner_angle":10,"outer_angle":25,
                "cast_shadows":false}"#,
        )
        .unwrap();
        assert!(!l.cast_shadows);

        let bytes = postcard::to_allocvec(&l).unwrap();
        let back: SpotLight = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.position, [2.0, 3.0, -1.0]);
        assert_eq!(back.direction, [0.0, -1.0, 0.5]);
        assert_eq!(back.color, [1.0, 0.9, 0.7]);
        assert_eq!(back.intensity, 45.0);
        assert_eq!(back.range, 18.0);
        assert_eq!((back.inner_angle, back.outer_angle), (10.0, 25.0));
    }
}

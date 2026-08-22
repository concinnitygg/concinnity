// Rectangular area-light schema.

/// A rectangular area light: a glowing panel that lights the scene from its
/// whole surface rather than from a single point.
///
/// Unlike a [PointLight](#pointlight) or [SpotLight](#spotlight), the softness of
/// the shadow terminator and the shape of the specular highlight follow the
/// panel's real dimensions, so a wide softbox wraps light around a surface and
/// leaves a stretched rectangular reflection on glossy materials. Use it for
/// windows, ceiling panels, screens, and practical lights.
///
/// The panel is positioned by `centre`, oriented by `normal` (the direction it
/// emits), and sized by `half_size`, matching [GlassPanel](#glasspanel).
///
/// ```rust
/// # use concinnity_asset::RectAreaLight;
/// RectAreaLight {
///     centre: [0.0, 3.0, -4.0],
///     normal: [0.0, 0.0, 1.0],
///     half_size: [1.5, 1.0],
///     color: [1.0, 0.95, 0.85],
///     intensity: 12.0,
///     range: 18.0,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RectAreaLight {
    /// World-space position of the panel's centre.
    pub centre: [f32; 3],
    /// Direction the panel emits. Normalised on load; defaults to `+Z` when
    /// degenerate.
    pub normal: [f32; 3],
    /// Half-width and half-height of the panel, in world units.
    pub half_size: [f32; 2],
    /// Linear-space RGB colour of the light.
    pub color: [f32; 3],
    /// Intensity multiplier applied to the colour.
    pub intensity: f32,
    /// Maximum reach in world units; attenuation is zero at this distance.
    pub range: f32,
    /// When true the panel emits from both faces. A one-sided panel lights only
    /// the half-space its `normal` points into.
    pub two_sided: bool,
}

impl Default for RectAreaLight {
    fn default() -> Self {
        Self {
            centre: [0.0, 3.0, 0.0],
            normal: [0.0, -1.0, 0.0],
            half_size: [1.0, 1.0],
            color: [1.0, 1.0, 1.0],
            intensity: 12.0,
            range: 18.0,
            two_sided: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_panel_is_a_ceiling_light_facing_down() {
        let l = RectAreaLight::default();
        assert_eq!(l.centre, [0.0, 3.0, 0.0]);
        assert_eq!(l.normal, [0.0, -1.0, 0.0]);
        assert_eq!(l.half_size, [1.0, 1.0]);
        assert_eq!(l.intensity, 12.0);
        assert_eq!(l.range, 18.0);
        // One-sided: the back of the panel emits nothing.
        assert!(!l.two_sided);
    }

    #[test]
    fn an_authored_panel_parses_and_round_trips_through_postcard() {
        let l: RectAreaLight = serde_json::from_str(
            r#"{"centre":[0,1.5,-4],"normal":[0,0,1],"half_size":[2,0.5],
                "color":[1,0.95,0.9],"intensity":30,"range":25,"two_sided":true}"#,
        )
        .unwrap();
        assert!(l.two_sided);
        assert_eq!(l.normal, [0.0, 0.0, 1.0]);

        let bytes = postcard::to_allocvec(&l).unwrap();
        let back: RectAreaLight = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.centre, [0.0, 1.5, -4.0]);
        assert_eq!(back.half_size, [2.0, 0.5]);
        assert_eq!(back.color, [1.0, 0.95, 0.9]);
        assert_eq!(back.intensity, 30.0);
        assert_eq!(back.range, 25.0);
    }
}

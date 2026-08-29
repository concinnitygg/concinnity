// Environmental volumetric fog schema.

/// Environmental volumetric fog: a single lit medium that wraps the scene,
/// thicker near the ground and thinning with height, with extra glow around the
/// sun.
///
/// Only one `VolumetricFog` is honoured: the first declared instance wins;
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct VolumetricFog {
    /// Master toggle. `false` disables the fog even when this asset is present.
    pub enabled: bool,
    /// Linear-space RGB tint of the fog: the colour the camera sees in the far
    /// distance.
    pub color: [f32; 3],
    /// Base thickness of the fog at `height_reference` (per world unit). Higher
    /// is thicker. Floored at 0.
    pub density: f32,
    /// How quickly the fog thins with height above `height_reference`. 0 keeps
    /// it uniform; larger values pin it to the ground.
    pub height_falloff: f32,
    /// World-space Y at which the fog reaches full `density`. It thickens below
    /// this height and thins above it.
    pub height_reference: f32,
    /// Maximum distance the fog covers from the camera, in world units. Past
    /// this, distant geometry stays clear.
    pub max_distance: f32,
    /// Sun-glow anisotropy in `(-1, 1)`. Positive values concentrate brightness
    /// around the sun (haloes), negative values scatter away from it, 0 is
    /// uniform.
    pub phase_g: f32,
    /// Constant ambient brightness so the fog keeps some colour in shaded areas.
    pub ambient: f32,
}

impl Default for VolumetricFog {
    fn default() -> Self {
        Self {
            enabled: true,
            color: [0.7, 0.78, 0.85],
            density: 0.05,
            height_falloff: 0.2,
            height_reference: 0.0,
            max_distance: 200.0,
            phase_g: 0.4,
            ambient: 0.15,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declaring_fog_turns_it_on_at_a_thin_forward_scattering_density() {
        // The asset exists to add fog, so `enabled` starts true: `false` is the
        // way to keep a declared fog around while switching it off.
        let f = VolumetricFog::default();
        assert!(f.enabled);
        assert_eq!(f.density, 0.05);
        assert_eq!(f.height_falloff, 0.2);
        assert_eq!(f.height_reference, 0.0);
        assert_eq!(f.max_distance, 200.0);
        // Positive g scatters forward, so the sun haloes rather than backlights.
        assert!(f.phase_g > 0.0);
        assert_eq!(f.ambient, 0.15);
    }

    #[test]
    fn an_authored_fog_parses_and_round_trips_through_postcard() {
        let f: VolumetricFog = serde_json::from_str(
            r#"{"enabled":false,"color":[0.5,0.5,0.6],"density":0.2,"height_falloff":0.05,
                "height_reference":12,"max_distance":80,"phase_g":-0.3,"ambient":0.4}"#,
        )
        .unwrap();
        assert!(!f.enabled);
        assert!(f.phase_g < 0.0);

        let bytes = postcard::to_allocvec(&f).unwrap();
        let back: VolumetricFog = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.color, [0.5, 0.5, 0.6]);
        assert_eq!(back.density, 0.2);
        assert_eq!(back.height_falloff, 0.05);
        assert_eq!(back.height_reference, 12.0);
        assert_eq!(back.max_distance, 80.0);
        assert_eq!(back.ambient, 0.4);
    }
}

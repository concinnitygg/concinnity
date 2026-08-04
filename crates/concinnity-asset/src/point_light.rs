// Point-light schema.

/// A spherical point light with quadratic distance attenuation.
///
/// The forward renderer lights every surface from all declared point lights (up
/// to a large per-scene budget). Secondary effects (volumetric fog, SDF
/// raymarching, and reflection-probe capture) still consider only the first 8.
///
/// ```jsonl
/// {"name":"lamp","type":"PointLight","args":{"position":[2.0,2.5,-3.0],"color":[1.0,0.8,0.5],"intensity":8.0,"range":6.0}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PointLight {
    /// World-space position of the light source.
    pub position: [f32; 3],
    /// Linear-space RGB colour of the light.
    pub color: [f32; 3],
    /// Intensity multiplier applied to the colour.
    pub intensity: f32,
    /// Maximum reach in world units; attenuation is zero at this distance.
    pub range: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: [0.0, 2.5, 0.0],
            color: [1.0, 1.0, 1.0],
            intensity: 8.0,
            range: 6.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_lamp_hangs_above_head_height_with_a_room_sized_reach() {
        let l = PointLight::default();
        assert_eq!(l.position, [0.0, 2.5, 0.0]);
        assert_eq!(l.color, [1.0, 1.0, 1.0]);
        assert_eq!(l.intensity, 8.0);
        assert_eq!(l.range, 6.0);
    }

    #[test]
    fn an_authored_lamp_parses_and_round_trips_through_postcard() {
        let l: PointLight = serde_json::from_str(
            r#"{"position":[2,2.5,-3],"color":[1,0.8,0.5],"intensity":12,"range":9}"#,
        )
        .unwrap();
        assert_eq!(l.position, [2.0, 2.5, -3.0]);
        assert_eq!(l.color, [1.0, 0.8, 0.5]);

        let bytes = postcard::to_allocvec(&l).unwrap();
        let back: PointLight = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.intensity, 12.0);
        assert_eq!(back.range, 9.0);
    }
}

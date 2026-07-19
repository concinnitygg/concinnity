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

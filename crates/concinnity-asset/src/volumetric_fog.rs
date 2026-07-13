// Environmental volumetric fog schema.

/// Environmental volumetric fog: a single lit medium that wraps the scene,
/// thicker near the ground and thinning with height, with extra glow around the
/// sun.
///
/// Only one `VolumetricFog` is honoured: the first declared instance wins;
/// later instances are silently dropped. With none declared, there is no fog.
///
/// ```jsonl
/// {"name":"fog","type":"VolumetricFog","args":{"density":0.08,"color":[0.75,0.82,0.95],"height_falloff":0.18,"max_distance":160.0,"phase_g":0.5}}
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

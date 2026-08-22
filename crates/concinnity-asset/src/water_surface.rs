// Animated water-surface schema.

use crate::AssetId;
use alloc::vec;
use alloc::vec::Vec;

/// Maximum number of waves per water surface. Shared by the render backends'
/// wave uniforms and the build-side water validator.
pub const MAX_WATER_WAVES: usize = 4;

/// One wave in a water surface's motion. A surface sums up to four of these
/// to displace its flat grid. Each wave travels
/// horizontally along `direction`, rising and falling with `amplitude` peak
/// height, `wavelength` distance between crests, and `speed` metres per second.
/// `steepness` in [0, 1] pinches the crests and broadens the troughs (choppier
/// water).
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct WaterWave {
    /// Peak height of the wave, in world units.
    pub amplitude: f32,
    /// Distance between successive crests, in world units.
    pub wavelength: f32,
    /// Horizontal travel speed, in metres per second.
    pub speed: f32,
    /// Horizontal travel direction `[x, z]`.
    pub direction: [f32; 2],
    /// Crest sharpness in [0, 1]. 0 is a smooth sine; higher pinches crests and
    /// broadens troughs.
    pub steepness: f32,
}

impl Default for WaterWave {
    fn default() -> Self {
        Self {
            amplitude: 0.15,
            wavelength: 4.0,
            speed: 1.0,
            direction: [1.0, 0.0],
            steepness: 0.4,
        }
    }
}

/// A translucent animated water surface.
///
/// A flat, subdivided horizontal surface whose vertices ripple with summed
/// waves. It refracts and reflects the scene, blends from a shallow to a deep
/// colour with depth, and adds shoreline foam.
///
/// The surface is positioned by `centre` and sized by `extent` (XZ
/// half-widths). The mesh itself is flat; all height variation comes from the
/// animated waves.
///
/// ```rust
/// # use concinnity_asset::WaterSurface;
/// WaterSurface {
///     centre: [0.0, 0.4, 0.0],
///     extent: [12.0, 8.0],
///     subdivisions: 96,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct WaterSurface {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// World-space position of the surface's centre.
    pub centre: [f32; 3],
    /// Half-width and half-depth of the surface `[x, z]`, in world units.
    pub extent: [f32; 2],
    /// Grid subdivisions across the surface. Higher gives smoother waves.
    /// Clamped to [8, 255].
    pub subdivisions: u32,
    /// The waves summed to animate the surface (up to 4). Defaults to a single
    /// gentle wave.
    pub waves: Vec<WaterWave>,
    /// Linear-space RGB colour of deep water.
    pub deep_colour: [f32; 3],
    /// Linear-space RGB colour of shallow water near the shore.
    pub shallow_colour: [f32; 3],
    /// Depth over which the colour blends from shallow to deep, in metres.
    pub depth_falloff_metres: f32,
    /// Width of the shoreline foam band, in metres.
    pub foam_width_metres: f32,
    /// Strength of the shoreline foam, in [0, 1].
    pub foam_intensity: f32,
    /// Sharpness of the grazing-angle reflection. Higher confines reflections to
    /// steeper viewing angles.
    pub fresnel_power: f32,
    /// Surface roughness in [0, 1]. Higher gives blurrier reflections.
    pub roughness: f32,
    /// How strongly the surface bends the view of what's beneath it.
    pub refraction_strength: f32,
    /// When false the surface is skipped each frame.
    pub visible: bool,
}

impl Default for WaterSurface {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            centre: [0.0, 0.0, 0.0],
            extent: [10.0, 10.0],
            subdivisions: 64,
            waves: vec![WaterWave::default()],
            deep_colour: [0.02, 0.05, 0.15],
            shallow_colour: [0.20, 0.50, 0.55],
            depth_falloff_metres: 4.0,
            foam_width_metres: 0.30,
            foam_intensity: 0.8,
            fresnel_power: 5.0,
            roughness: 0.05,
            refraction_strength: 0.15,
            visible: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_wave_travels_along_positive_x() {
        let w = WaterWave::default();
        assert_eq!(w.amplitude, 0.15);
        assert_eq!(w.wavelength, 4.0);
        assert_eq!(w.speed, 1.0);
        assert_eq!(w.direction, [1.0, 0.0]);
        assert_eq!(w.steepness, 0.4);
    }

    #[test]
    fn a_blank_surface_already_has_one_wave_so_it_is_not_a_flat_plane() {
        let s = WaterSurface::default();
        assert_eq!(s.waves.len(), 1);
        assert_eq!(s.waves[0].amplitude, WaterWave::default().amplitude);
        assert_eq!(s.extent, [10.0, 10.0]);
        assert_eq!(s.subdivisions, 64);
        // Deep water is darker and bluer than shallow: the depth gradient is
        // what reads as water rather than a tinted mirror.
        assert!(s.deep_colour[2] > s.deep_colour[0]);
        assert!(s.shallow_colour[1] > s.deep_colour[1]);
        assert_eq!(s.depth_falloff_metres, 4.0);
        assert_eq!(s.foam_width_metres, 0.3);
        assert_eq!(s.foam_intensity, 0.8);
        assert_eq!(s.fresnel_power, 5.0);
        assert_eq!(s.roughness, 0.05);
        assert_eq!(s.refraction_strength, 0.15);
        assert!(s.visible);
    }

    #[test]
    fn a_multi_wave_surface_parses_and_round_trips_through_postcard() {
        let s: WaterSurface = serde_json::from_str(
            r#"{"centre":[0,0.2,-5],"extent":[40,25],"subdivisions":128,
                "waves":[{"amplitude":0.4,"wavelength":12,"direction":[0.7,0.7]},
                         {"amplitude":0.05,"wavelength":1.5,"speed":2.5,"steepness":0.1}],
                "deep_colour":[0,0.02,0.1],"shallow_colour":[0.1,0.4,0.45],
                "depth_falloff_metres":8,"foam_width_metres":0.6,"foam_intensity":1.2,
                "fresnel_power":4,"roughness":0.02,"refraction_strength":0.3,
                "visible":false}"#,
        )
        .unwrap();
        assert_eq!(s.waves.len(), 2);
        // A wave that mentions only some fields keeps the wave defaults.
        assert_eq!(s.waves[0].speed, 1.0);
        assert_eq!(s.waves[0].steepness, 0.4);
        assert_eq!(s.waves[1].direction, [1.0, 0.0]);
        assert!(!s.visible);

        let bytes = postcard::to_allocvec(&s).unwrap();
        let back: WaterSurface = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.centre, [0.0, 0.2, -5.0]);
        assert_eq!(back.extent, [40.0, 25.0]);
        assert_eq!(back.subdivisions, 128);
        assert_eq!(back.waves[1].speed, 2.5);
        assert_eq!(back.depth_falloff_metres, 8.0);
        assert_eq!(back.foam_intensity, 1.2);
        assert_eq!(back.refraction_strength, 0.3);
        assert_eq!(back.asset_id, AssetId::default());
    }
}

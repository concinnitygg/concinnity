// Animated water-surface schema.

use alloc::vec;
use alloc::vec::Vec;

/// Maximum number of waves per water surface. Shared by the render backends'
/// wave uniforms and the build-side water validator.
pub const MAX_WATER_WAVES: usize = 4;

/// One wave in a water surface's motion. A surface sums up to four of these
/// to displace its flat grid. Each wave travels
/// horizontally along `direction`, rising and falling with `amplitude` peak
/// height, `wavelength` distance between crests, and `speed` meters per second.
/// `steepness` in [0, 1] pinches the crests and broadens the troughs (choppier
/// water).
#[derive(
    Debug,
    Clone,
    Copy,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct WaterWave {
    /// Peak height of the wave, in world units.
    #[asset(default = 0.15)]
    pub amplitude: f32,
    /// Distance between successive crests, in world units.
    #[asset(default = 4.0)]
    pub wavelength: f32,
    /// Horizontal travel speed, in meters per second.
    #[asset(default = 1.0)]
    pub speed: f32,
    /// Horizontal travel direction `[x, z]`.
    #[asset(default = [1.0, 0.0])]
    pub direction: [f32; 2],
    /// Crest sharpness in [0, 1]. 0 is a smooth sine; higher pinches crests and
    /// broadens troughs.
    #[asset(default = 0.4)]
    pub steepness: f32,
}

/// A translucent animated water surface.
///
/// A flat, subdivided horizontal surface whose vertices ripple with summed
/// waves. It refracts and reflects the scene, blends from a shallow to a deep
/// color with depth, and adds shoreline foam.
///
/// The surface is positioned by `center` and sized by `extent` (XZ
/// half-widths). The mesh itself is flat; all height variation comes from the
/// animated waves.
///
/// Its sun glint rides the waves, and its reflection and refraction move with
/// neither the surface nor what they show, so temporal anti-aliasing and
/// upscaling favor the current frame where the glint shows and, within about
/// 16 meters of the camera, where the reflection is strong. Farther off they
/// keep their history, which holds distant waves still.
///
/// ```rust
/// # use concinnity_core::components::WaterSurface;
/// WaterSurface {
///     center: [0.0, 0.4, 0.0],
///     extent: [12.0, 8.0],
///     subdivisions: 96,
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
pub struct WaterSurface {
    /// World-space position of the surface's center.
    pub center: [f32; 3],
    /// Half-width and half-depth of the surface `[x, z]`, in world units.
    #[asset(default = [10.0, 10.0])]
    pub extent: [f32; 2],
    /// Grid subdivisions across the surface. Higher gives smoother waves.
    /// Clamped to [8, 255].
    #[asset(default = 64)]
    pub subdivisions: u32,
    /// The waves summed to animate the surface (up to 4). Defaults to a single
    /// gentle wave.
    #[asset(default = vec![WaterWave::default()])]
    pub waves: Vec<WaterWave>,
    /// Linear-space RGB color of deep water.
    #[asset(default = [0.02, 0.05, 0.15])]
    pub deep_color: [f32; 3],
    /// Linear-space RGB color of shallow water near the shore.
    #[asset(default = [0.20, 0.50, 0.55])]
    pub shallow_color: [f32; 3],
    /// Depth over which the color blends from shallow to deep, in meters.
    #[asset(default = 4.0)]
    pub depth_falloff_meters: f32,
    /// Width of the shoreline foam band, in meters.
    #[asset(default = 0.30)]
    pub foam_width_meters: f32,
    /// Strength of the shoreline foam, in [0, 1].
    #[asset(default = 0.8)]
    pub foam_intensity: f32,
    /// Sharpness of the grazing-angle reflection. Higher confines reflections to
    /// steeper viewing angles.
    #[asset(default = 5.0)]
    pub fresnel_power: f32,
    /// Surface roughness in [0, 1]. Higher gives blurrier reflections, and
    /// pushes a mirrored reflection further off its line with each wave: a
    /// near-mirror surface keeps its reflection almost still.
    #[asset(default = 0.05)]
    pub roughness: f32,
    /// How strongly the surface bends the view of what's beneath it.
    #[asset(default = 0.15)]
    pub refraction_strength: f32,
    /// When false the surface is skipped each frame.
    #[asset(default = true)]
    pub visible: bool,
}

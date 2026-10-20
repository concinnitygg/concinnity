// Procedural grass schema.

/// The look of a grass cover: individual blades, grown on the GPU each frame
/// and swayed by the world's [Wind](#wind).
///
/// A `Grass` grows only where a [Terrain](#terrain) layer names it: the
/// terrain sets where the blades root and which way the ground slopes, and the
/// layer's mask sets where they grow. Blades grow at `density` per square
/// meter in clumps about `clump_size` meters across whose blades share their
/// height and lean, crowding toward each clump's middle by `clumping` and
/// thinning between clumps, and are placed the same way on every run. Blades
/// on a slope lean downhill.
///
/// Each blade is real geometry: a tapered strip that curves under its own
/// weight and the wind, shaded from `root_color` at the ground to `tip_color`
/// at the tip, with light passing through it when the sun is behind. Blades are
/// drawn out to a fixed distance from the camera and thin out toward it.
///
/// The ground under the blades is strewn with dead grass of `litter_color`,
/// the more so the denser the grass grows, and lies darker the more leaf
/// shades it. Across it, patches of bare earth (the terrain's own material)
/// and lighter, drier litter break it up at a few scales.
///
/// Past where the blades end, the ground is shaded as the field they make seen
/// from afar: the same colors, clumps and bare patches, brighter and greener
/// along the ground toward the horizon, darker looking straight down into it.
/// Across hundreds of meters the field also drifts between greener and drier
/// stretches, drier still on slopes, by `macro_variation`.
///
/// ```rust
/// # use concinnity_core::components::Grass;
/// Grass {
///     height: 0.6,
///     density: 150.0,
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
pub struct Grass {
    /// Mean blade height, in meters.
    #[asset(default = 0.5)]
    pub height: f32,
    /// How much blade heights vary around `height`, as a fraction of it in
    /// [0, 1].
    #[asset(default = 0.4)]
    pub height_variance: f32,
    /// Blade width at the root, in meters. Blades taper to a point.
    #[asset(default = 0.03)]
    pub width: f32,
    /// Blades per square meter.
    #[asset(default = 120.0)]
    pub density: f32,
    /// Typical clump diameter, in meters.
    #[asset(default = 0.6)]
    pub clump_size: f32,
    /// How strongly blades crowd toward the middle of their clump, in [0, 1],
    /// leaving the ground between clumps sparser but never bare. 0 spreads
    /// them evenly. The blade count is the same either way.
    #[asset(default = 0.5)]
    pub clumping: f32,
    /// How firmly blades resist bending, in [0, 1]. 0 lies flat in a strong
    /// wind; 1 barely moves.
    #[asset(default = 0.5)]
    pub stiffness: f32,
    /// Linear-space RGB color at the blade root.
    #[asset(default = [0.025, 0.06, 0.012])]
    pub root_color: [f32; 3],
    /// Linear-space RGB color at the blade tip.
    #[asset(default = [0.2, 0.32, 0.065])]
    pub tip_color: [f32; 3],
    /// Linear-space RGB color of the dead grass on the ground under the
    /// blades. `None` derives it from `root_color`, darker and browner.
    pub litter_color: Option<[f32; 3]>,
    /// How much hue and brightness vary between clumps and blades, in [0, 1].
    #[asset(default = 0.35)]
    pub color_variation: f32,
    /// How much the field's color drifts over hundreds of meters to
    /// kilometers, in [0, 1]: patches of deeper and lighter green, drier
    /// stretches, and drier grass on slopes. 0 grows one color everywhere.
    #[asset(default = 0.5)]
    pub macro_variation: f32,
    /// When false no terrain grows this grass.
    #[asset(default = true)]
    pub visible: bool,
}

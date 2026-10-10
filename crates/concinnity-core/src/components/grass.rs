// Procedural grass schema.

/// A field of individual grass blades, grown on the GPU each frame and swayed
/// by the world's [Wind](#wind).
///
/// The field covers a flat rectangle: `center` sets its middle (and the
/// ground height the blades root at), `extent` its half-width and half-depth.
/// Blades are scattered evenly at `density` per square meter, gathered into
/// clumps about `clump_size` meters across whose blades share their height and
/// lean, and placed the same way on every run.
///
/// Each blade is real geometry: a tapered strip that curves under its own
/// weight and the wind, shaded from `root_color` at the ground to `tip_color`
/// at the tip, with light passing through it when the sun is behind. Blades are
/// drawn out to a fixed distance from the camera and thin out toward it.
///
/// One field is drawn per world: the first visible `Grass` declared.
///
/// ```rust
/// # use concinnity_core::components::Grass;
/// Grass {
///     center: [0.0, 0.0, 0.0],
///     extent: [30.0, 30.0],
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
    /// World-space center of the field; its height is the ground the blades
    /// grow from.
    pub center: [f32; 3],
    /// Half-width and half-depth of the field `[x, z]`, in meters.
    #[asset(default = [10.0, 10.0])]
    pub extent: [f32; 2],
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
    /// How much hue and brightness vary between clumps and blades, in [0, 1].
    #[asset(default = 0.35)]
    pub color_variation: f32,
    /// When false the field is not drawn.
    #[asset(default = true)]
    pub visible: bool,
}

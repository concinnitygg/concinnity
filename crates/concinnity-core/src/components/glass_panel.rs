// Colored glass-panel schema.

/// A flat translucent panel of colored glass. A fixed-orientation rectangular
/// quad that refracts and tints the scene behind it and brightens the
/// grazing-angle rim with a Fresnel highlight.
///
/// Unlike [WaterSurface](#watersurface) it has no animation, no surface
/// displacement, and no depth-based color. It's a simple building block for
/// translucent surfaces such as windows, ice, holograms, or force fields.
///
/// The panel is positioned by `center`, oriented by `normal` (the facing
/// direction), and sized by `half_size` (half-width along the panel's tangent,
/// half-height along its bitangent).
///
/// ```rust
/// # use concinnity_core::components::GlassPanel;
/// GlassPanel {
///     center: [0.0, 2.0, -3.0],
///     normal: [0.0, 0.0, 1.0],
///     half_size: [2.0, 1.5],
///     tint: [0.6, 0.85, 0.9],
///     opacity: 0.45,
///     refraction_strength: 0.04,
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
pub struct GlassPanel {
    /// World-space position of the panel's center.
    #[asset(default = [0.0, 1.0, 0.0])]
    pub center: [f32; 3],
    /// Facing direction of the panel. Normalized on load; defaults to +Z when
    /// degenerate.
    #[asset(default = [0.0, 0.0, 1.0])]
    pub normal: [f32; 3],
    /// Half-width and half-height of the panel, in world units.
    #[asset(default = [1.0, 1.0])]
    pub half_size: [f32; 2],
    /// Linear-space RGB color the glass tints the scene behind it.
    #[asset(default = [0.7, 0.85, 0.95])]
    pub tint: [f32; 3],
    /// How opaque the glass is, in [0, 1]. 0 = clear, 1 = fully opaque tint.
    #[asset(default = 0.5)]
    pub opacity: f32,
    /// How strongly the glass bends the view of what's behind it. 0 = no
    /// refraction.
    #[asset(default = 0.04)]
    pub refraction_strength: f32,
    /// Sharpness of the grazing-angle rim highlight. Higher values confine the
    /// brightening to steeper viewing angles.
    #[asset(default = 4.0)]
    pub fresnel_power: f32,
    /// When false the panel is skipped each frame.
    #[asset(default = true)]
    pub visible: bool,
}

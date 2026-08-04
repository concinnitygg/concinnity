// Coloured glass-panel schema.

use crate::AssetId;

/// A flat translucent panel of coloured glass. A fixed-orientation rectangular
/// quad that refracts and tints the scene behind it and brightens the
/// grazing-angle rim with a Fresnel highlight.
///
/// Unlike [WaterSurface](#watersurface) it has no animation, no surface
/// displacement, and no depth-based colour. It's a simple building block for
/// translucent surfaces such as windows, ice, holograms, or force fields.
///
/// The panel is positioned by `centre`, oriented by `normal` (the facing
/// direction), and sized by `half_size` (half-width along the panel's tangent,
/// half-height along its bitangent).
///
/// ```jsonl
/// {"name":"window","type":"GlassPanel","args":{
///   "centre":[0.0,2.0,-3.0],
///   "normal":[0.0,0.0,1.0],
///   "half_size":[2.0,1.5],
///   "tint":[0.6,0.85,0.9],
///   "opacity":0.45,
///   "refraction_strength":0.04,
///   "fresnel_power":4.0
/// }}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GlassPanel {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// World-space position of the panel's centre.
    pub centre: [f32; 3],
    /// Facing direction of the panel. Normalised on load; defaults to +Z when
    /// degenerate.
    pub normal: [f32; 3],
    /// Half-width and half-height of the panel, in world units.
    pub half_size: [f32; 2],
    /// Linear-space RGB colour the glass tints the scene behind it.
    pub tint: [f32; 3],
    /// How opaque the glass is, in [0, 1]. 0 = clear, 1 = fully opaque tint.
    pub opacity: f32,
    /// How strongly the glass bends the view of what's behind it. 0 = no
    /// refraction.
    pub refraction_strength: f32,
    /// Sharpness of the grazing-angle rim highlight. Higher values confine the
    /// brightening to steeper viewing angles.
    pub fresnel_power: f32,
    /// When false the panel is skipped each frame.
    pub visible: bool,
}

impl Default for GlassPanel {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            centre: [0.0, 1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            half_size: [1.0, 1.0],
            tint: [0.7, 0.85, 0.95],
            opacity: 0.5,
            refraction_strength: 0.04,
            fresnel_power: 4.0,
            visible: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_panel_is_a_visible_half_transparent_unit_square() {
        let g = GlassPanel::default();
        assert_eq!(g.centre, [0.0, 1.0, 0.0]);
        assert_eq!(g.normal, [0.0, 0.0, 1.0]);
        assert_eq!(g.half_size, [1.0, 1.0]);
        assert_eq!(g.opacity, 0.5);
        assert_eq!(g.refraction_strength, 0.04);
        assert_eq!(g.fresnel_power, 4.0);
        assert!(g.visible);
        // The default tint is a faint cool cast, not neutral white.
        assert_eq!(g.tint, [0.7, 0.85, 0.95]);
    }

    #[test]
    fn an_authored_panel_parses_and_round_trips_through_postcard() {
        let g: GlassPanel = serde_json::from_str(
            r#"{"centre":[2,1,-3],"normal":[1,0,0],"half_size":[0.6,1.2],
                "tint":[1,1,1],"opacity":0.2,"refraction_strength":0.1,"visible":false}"#,
        )
        .unwrap();
        assert_eq!(g.normal, [1.0, 0.0, 0.0]);
        assert!(!g.visible);

        let bytes = postcard::to_allocvec(&g).unwrap();
        let back: GlassPanel = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.centre, [2.0, 1.0, -3.0]);
        assert_eq!(back.half_size, [0.6, 1.2]);
        assert_eq!(back.tint, [1.0, 1.0, 1.0]);
        assert_eq!(back.opacity, 0.2);
        assert_eq!(back.refraction_strength, 0.1);
        // Fresnel was not authored, so it keeps the schema default.
        assert_eq!(back.fresnel_power, 4.0);
        assert_eq!(back.asset_id, AssetId::default());
    }
}

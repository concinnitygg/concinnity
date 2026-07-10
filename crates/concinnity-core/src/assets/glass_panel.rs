// src/assets/glass_panel.rs

use crate::assets::{GlassPanel, GlassPanelGeometry};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

impl Component for GlassPanel {
    const NAME: &'static str = "GlassPanel";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        args.normal = args.unit_normal();
        args.half_size[0] = args.half_size[0].max(1e-3);
        args.half_size[1] = args.half_size[1].max(1e-3);
        args.opacity = args.opacity.clamp(0.0, 1.0);
        args.refraction_strength = args.refraction_strength.max(0.0);
        args.fresnel_power = args.fresnel_power.max(0.0);
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::Component;

    #[test]
    fn from_args_normalizes_normal() {
        let g = GlassPanel::from_args(GlassPanel {
            normal: [0.0, 0.0, 4.0],
            ..Default::default()
        });
        let len = (g.normal[0].powi(2) + g.normal[1].powi(2) + g.normal[2].powi(2)).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
        assert!((g.normal[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn from_args_falls_back_on_degenerate_normal() {
        let g = GlassPanel::from_args(GlassPanel {
            normal: [0.0, 0.0, 0.0],
            ..Default::default()
        });
        assert_eq!(g.normal, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn from_args_clamps_ranges() {
        let g = GlassPanel::from_args(GlassPanel {
            half_size: [-2.0, 0.0],
            opacity: 1.5,
            refraction_strength: -0.1,
            fresnel_power: -3.0,
            ..Default::default()
        });
        assert!(g.half_size[0] > 0.0 && g.half_size[1] > 0.0);
        assert_eq!(g.opacity, 1.0);
        assert_eq!(g.refraction_strength, 0.0);
        assert_eq!(g.fresnel_power, 0.0);
    }
}

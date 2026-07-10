// src/assets/material.rs

use crate::assets::Material;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

impl Component for Material {
    const NAME: &'static str = "Material";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn ref_fields() -> &'static [(&'static str, &'static str)] {
        &[
            ("albedo", "Texture"),
            ("normal_map", "Texture"),
            ("emissive_map", "Texture"),
            ("orm_map", "Texture"),
            ("albedo_secondary", "Texture"),
            ("normal_secondary", "Texture"),
        ]
    }

    fn from_args(mut args: Self) -> Self {
        args.roughness = args.roughness.clamp(0.0, 1.0);
        args.metallic = args.metallic.clamp(0.0, 1.0);
        args.macro_variation = args.macro_variation.clamp(0.0, 1.0);
        args.terrain_blend = args.terrain_blend.clamp(0.0, 1.0);
        args.secondary_blend_sharpness = args.secondary_blend_sharpness.clamp(0.0, 1.0);
        args.opacity = args.opacity.clamp(0.0, 1.0);
        // See-through glass is by definition transparent; opting into it implies
        // the transparent pass even if the author only set `see_through`.
        if args.see_through {
            args.transparent = true;
        }
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
    fn default_is_opaque_and_not_see_through() {
        let m = Material::default();
        assert!(!m.transparent);
        assert!(!m.see_through);
        assert_eq!(m.opacity, 1.0);
    }

    #[test]
    fn see_through_implies_transparent() {
        // A material that opts into see-through but leaves `transparent` at its
        // default must still route through the transparent pass.
        let m = Material::from_args(Material {
            see_through: true,
            ..Material::default()
        });
        assert!(m.see_through);
        assert!(m.transparent);
    }

    #[test]
    fn transparent_without_see_through_stays_opaque_layer() {
        // The importer's glass detection sets `transparent` only; that material
        // stays Layer 1 (opaque reflective) and does not flip see-through on.
        let m = Material::from_args(Material {
            transparent: true,
            ..Material::default()
        });
        assert!(m.transparent);
        assert!(!m.see_through);
    }
}

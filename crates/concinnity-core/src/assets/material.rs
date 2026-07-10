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

impl crate::check::cross_reference::CrossReferenced for Material {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let arg = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();

        let albedo = arg("albedo");
        if !albedo.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: albedo.to_string(),
                error: format!(
                    "Material '{}': albedo texture '{}' not found, add a Texture asset with that name",
                    name, albedo
                ),
            });
        }

        let normal_map = arg("normal_map");
        if !normal_map.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: normal_map.to_string(),
                error: format!(
                    "Material '{}': normal_map texture '{}' not found, add a Texture asset with that name",
                    name, normal_map
                ),
            });
        }

        let emissive_map = arg("emissive_map");
        if !emissive_map.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: emissive_map.to_string(),
                error: format!(
                    "Material '{}': emissive_map texture '{}' not found, add a Texture asset with that name",
                    name, emissive_map
                ),
            });
        }

        let orm_map = arg("orm_map");
        if !orm_map.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: orm_map.to_string(),
                error: format!(
                    "Material '{}': orm_map texture '{}' not found, add a Texture asset with that name",
                    name, orm_map
                ),
            });
        }

        let albedo_secondary = arg("albedo_secondary");
        if !albedo_secondary.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: albedo_secondary.to_string(),
                error: format!(
                    "Material '{}': albedo_secondary texture '{}' not found, add a Texture asset with that name",
                    name, albedo_secondary
                ),
            });
        }

        let normal_secondary = arg("normal_secondary");
        if !normal_secondary.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: normal_secondary.to_string(),
                error: format!(
                    "Material '{}': normal_secondary texture '{}' not found, add a Texture asset with that name",
                    name, normal_secondary
                ),
            });
        }

        refs
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

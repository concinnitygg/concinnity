// src/assets/voxel_world.rs

use crate::assets::VoxelWorld;
use crate::ecs::{AssetOrigin, CompanionSpec, Component};

impl Component for VoxelWorld {
    const NAME: &'static str = "VoxelWorld";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }

    fn companions(_args: &serde_json::Value, _world: &[serde_json::Value]) -> Vec<CompanionSpec> {
        vec![CompanionSpec {
            name: "GraphicsConfig",
            asset_type: "GraphicsConfig",
            args: serde_json::json!({}),
        }]
    }
}

impl crate::check::cross_reference::CrossReferenced for VoxelWorld {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let mut refs = Vec::new();

        let palette = args
            .get("palette")
            .and_then(|v| v.as_array())
            .map(|a| a.as_slice())
            .unwrap_or(&[]);
        for (i, entry) in palette.iter().enumerate() {
            let bt_name = entry.as_str().unwrap_or("");
            if bt_name.is_empty() {
                refs.push(CrossRef::Issue(format!(
                    "VoxelWorld '{}': palette[{}] is not a valid BlockType name",
                    name, i
                )));
            } else {
                refs.push(CrossRef::Resolve {
                    kind: RefKind::BlockType,
                    target: bt_name.to_string(),
                    error: format!(
                        "VoxelWorld '{}': palette[{}] BlockType '{}' not found, add a BlockType asset with that name",
                        name, i, bt_name
                    ),
                });
            }
        }

        if let Some(mat) = args.get("material").and_then(|v| v.as_str())
            && !mat.is_empty()
        {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Material,
                target: mat.to_string(),
                error: format!(
                    "VoxelWorld '{}': material '{}' not found, add a Material asset with that name",
                    name, mat
                ),
            });
        }

        refs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_a_modest_window() {
        let w = VoxelWorld::default();
        assert_eq!(w.chunk_blocks(), [16, 24, 16]);
        assert_eq!(w.view_radius(), 5);
        assert_eq!(w.load_budget(), 3);
        assert_eq!(w.chunk_world_size(), (16.0, 16.0));
    }

    #[test]
    fn degenerate_args_are_floored_and_clamped() {
        let w = VoxelWorld {
            chunk_blocks: [0, 0, 0],
            block_size: -1.0,
            view_radius: 9999,
            load_budget: 0,
            ..VoxelWorld::default()
        };
        assert_eq!(w.chunk_blocks(), [1, 1, 1]);
        assert!(w.block_size() > 0.0);
        assert_eq!(w.view_radius(), 32);
        assert_eq!(w.load_budget(), 1);
    }

    #[test]
    fn deserialises_from_jsonl_args_with_defaults_for_omitted_fields() {
        let w: VoxelWorld = serde_json::from_str(r#"{"seed":7,"view_radius":8}"#).expect("parse");
        assert_eq!(w.seed, 7);
        assert_eq!(w.view_radius(), 8);
        // omitted fields fall back to the defaults
        assert_eq!(w.chunk_blocks(), [16, 24, 16]);
        assert_eq!(w.load_budget(), 3);
    }

    #[test]
    fn round_trips_through_args() {
        let w = VoxelWorld {
            seed: 99,
            chunk_blocks: [8, 32, 8],
            block_size: 2.0,
            view_radius: 4,
            impostor_radius: 12,
            impostor_step: 2,
            load_budget: 5,
            palette: Vec::new(),
            material: None,
        };
        let back = VoxelWorld::from_args(w.to_args());
        assert_eq!(back.seed, 99);
        assert_eq!(back.chunk_blocks, [8, 32, 8]);
        assert_eq!(back.block_size, 2.0);
        assert_eq!(back.impostor_radius, 12);
        assert_eq!(back.impostor_step, 2);
        assert_eq!(back.load_budget, 5);
    }

    #[test]
    fn impostors_disabled_by_default() {
        let w = VoxelWorld::default();
        // Default impostor_radius 0 -> clamped up to view_radius -> no far band.
        assert_eq!(w.impostor_radius(), w.view_radius());
        assert!(!w.impostors_enabled());
        assert_eq!(w.impostor_step(), 4);
    }

    #[test]
    fn impostor_radius_enables_the_far_band_and_clamps() {
        let w = VoxelWorld {
            view_radius: 5,
            impostor_radius: 16,
            impostor_step: 0,
            ..VoxelWorld::default()
        };
        assert_eq!(w.impostor_radius(), 16);
        assert!(w.impostors_enabled());
        // step floored at 1.
        assert_eq!(w.impostor_step(), 1);

        // An impostor radius below the view radius disables impostors.
        let w2 = VoxelWorld {
            view_radius: 8,
            impostor_radius: 4,
            ..VoxelWorld::default()
        };
        assert_eq!(w2.impostor_radius(), w2.view_radius());
        assert!(!w2.impostors_enabled());
    }
}

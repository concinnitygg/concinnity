// src/assets/voxel_chunk.rs

use crate::assets::VoxelChunk;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

impl Component for VoxelChunk {
    const NAME: &'static str = "VoxelChunk";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    const PAYLOAD: AssetPayload = AssetPayload::Compiled;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        args.block_size = args.block_size.max(0.0);
        if args.lod_levels == 0 {
            args.lod_levels = 1;
        }
        args.lod_levels = args.lod_levels.min(8);
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }
    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

impl crate::check::cross_reference::CrossReferenced for VoxelChunk {
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
                    "VoxelChunk '{}': palette[{}] is not a valid BlockType name",
                    name, i
                )));
            } else {
                refs.push(CrossRef::Resolve {
                    kind: RefKind::BlockType,
                    target: bt_name.to_string(),
                    error: format!(
                        "VoxelChunk '{}': palette[{}] BlockType '{}' not found, add a BlockType asset with that name",
                        name, i, bt_name
                    ),
                });
            }
        }

        refs
    }
}

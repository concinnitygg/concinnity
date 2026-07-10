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

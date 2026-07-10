// src/assets/block_type.rs

use crate::assets::BlockType;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

impl Component for BlockType {
    const NAME: &'static str = "BlockType";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

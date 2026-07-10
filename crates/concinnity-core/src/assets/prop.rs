// src/assets/prop.rs

use crate::assets::Prop;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

impl Component for Prop {
    const NAME: &'static str = "Prop";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(mut args: Self) -> Self {
        args.cull_distance = args.cull_distance.max(0.0);
        args.is_held = false;
        args
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

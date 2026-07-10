// src/assets/light_rig.rs

use crate::assets::LightRig;
use crate::ecs::{AssetOrigin, Component};

impl Component for LightRig {
    const NAME: &'static str = "LightRig";
    const ORIGIN: AssetOrigin = AssetOrigin::BuildOnly;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}

// src/assets/point_light.rs

use crate::assets::PointLight;
use crate::ecs::{AssetOrigin, Component};

impl Component for PointLight {
    const NAME: &'static str = "PointLight";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    // A pass-through leaf (`Args = Self`): its baked component is its authored
    // args, so cook emits it on the baked path and the runtime loads it via the
    // default `from_baked` (identical to the authored reconstruction).
    const BAKED: bool = true;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        args.intensity = args.intensity.max(0.0);
        args.range = args.range.max(0.0);
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}

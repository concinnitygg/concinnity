// src/assets/point_light.rs

use crate::assets::PointLight;
use crate::ecs::{AssetOrigin, Component};

impl Component for PointLight {
    const NAME: &'static str = "PointLight";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
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

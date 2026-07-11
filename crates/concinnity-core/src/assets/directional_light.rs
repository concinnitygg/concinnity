// src/assets/directional_light.rs

use crate::assets::DirectionalLight;
use crate::ecs::{AssetOrigin, Component};

impl Component for DirectionalLight {
    const NAME: &'static str = "DirectionalLight";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    // Pass-through leaf: baked component is its authored args (see PointLight).
    const BAKED: bool = true;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        args.intensity = args.intensity.max(0.0);
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}

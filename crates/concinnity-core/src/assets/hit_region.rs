// src/assets/hit_region.rs

use crate::assets::HitRegion;
use crate::ecs::{AssetOrigin, Component};

impl Component for HitRegion {
    const NAME: &'static str = "HitRegion";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn ref_fields() -> &'static [(&'static str, &'static str)] {
        &[("label", "TextLabel"), ("view", "View")]
    }

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}

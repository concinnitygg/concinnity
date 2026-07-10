// src/assets/prop_body.rs

use crate::assets::PropBody;
use crate::ecs::{AssetOrigin, Component};

impl Component for PropBody {
    const NAME: &'static str = "PropBody";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}

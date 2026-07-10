// src/assets/window.rs

use crate::assets::Window;
use crate::ecs::{AssetOrigin, Component};

impl Component for Window {
    const NAME: &'static str = "Window";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}

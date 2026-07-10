// src/assets/camera_shot.rs

use crate::assets::CameraShot;
use crate::ecs::{AssetOrigin, Component};

impl Component for CameraShot {
    const NAME: &'static str = "CameraShot";
    const ORIGIN: AssetOrigin = AssetOrigin::BuildOnly;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}

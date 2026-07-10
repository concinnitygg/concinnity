// src/assets/scene_import.rs

use crate::assets::SceneImport;
use crate::ecs::{AssetOrigin, Component};

impl Component for SceneImport {
    const NAME: &'static str = "SceneImport";
    const ORIGIN: AssetOrigin = AssetOrigin::BuildOnly;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}

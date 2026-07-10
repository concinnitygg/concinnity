// src/assets/story_import.rs

use crate::assets::StoryImport;
use crate::ecs::{AssetOrigin, Component};

impl Component for StoryImport {
    const NAME: &'static str = "StoryImport";
    const ORIGIN: AssetOrigin = AssetOrigin::BuildOnly;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}

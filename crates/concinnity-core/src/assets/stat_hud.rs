// src/assets/stat_hud.rs
//
// StatHud component (pure data). The runtime behavior that reads it lives in
// the client crate's `hud::stat_hud`.

use crate::assets::StatHud;
use crate::ecs::{AssetOrigin, Component};

impl Component for StatHud {
    const NAME: &'static str = "StatHud";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}

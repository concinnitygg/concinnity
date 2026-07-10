// src/assets/debug_hud.rs
//
// DebugHud component (pure data). The runtime behavior that reads it lives in
// the client crate's `hud::debug_hud`.

use crate::assets::DebugHud;
use crate::ecs::{AssetOrigin, Component};

impl Component for DebugHud {
    const NAME: &'static str = "DebugHud";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}

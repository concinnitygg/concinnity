// src/assets/physics_config.rs
//
// World-level physics configuration. The Rapier simulation itself is the
// internal physics system in the client crate's `physics::system`.

use crate::assets::PhysicsConfig;
use crate::ecs::{AssetOrigin, Component};

impl Component for PhysicsConfig {
    const NAME: &'static str = "PhysicsConfig";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}

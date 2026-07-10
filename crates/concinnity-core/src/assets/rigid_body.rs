// src/assets/rigid_body.rs

use crate::assets::RigidBody;
use crate::ecs::{AssetOrigin, Component};

impl Component for RigidBody {
    const NAME: &'static str = "RigidBody";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(mut args: Self) -> Self {
        // Runtime state is always reset on construction.
        args.is_grounded = true;
        args
    }
}

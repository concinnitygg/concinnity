// src/assets/reflection_probe.rs

use crate::assets::ReflectionProbe;
use crate::ecs::{AssetOrigin, Component};

impl Component for ReflectionProbe {
    const NAME: &'static str = "ReflectionProbe";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        // Half-extents are sizes: keep them non-negative so the influence box is
        // never inverted.
        for e in &mut args.half_extents {
            *e = e.max(0.0);
        }
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}

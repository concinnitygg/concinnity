// src/assets/water_surface.rs

use crate::assets::{WaterSurface, WaterWave};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, CompanionSpec, Component};

/// Maximum number of waves per water surface.
pub const MAX_WATER_WAVES: usize = 4;

impl Component for WaterSurface {
    const NAME: &'static str = "WaterSurface";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        args.subdivisions = args.subdivisions.clamp(8, 255);
        if args.waves.len() > MAX_WATER_WAVES {
            args.waves.truncate(MAX_WATER_WAVES);
        }
        if args.waves.is_empty() {
            args.waves.push(WaterWave::default());
        }
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }

    fn companions(_args: &serde_json::Value, _world: &[serde_json::Value]) -> Vec<CompanionSpec> {
        vec![CompanionSpec {
            name: "GraphicsConfig",
            asset_type: "GraphicsConfig",
            args: serde_json::json!({}),
        }]
    }
}

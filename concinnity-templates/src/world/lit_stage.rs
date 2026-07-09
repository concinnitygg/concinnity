// A neutral three-point-ish lighting rig: a key directional plus two fill point
// lights, so an unlit scene reads with shape and depth without any post stack.

use crate::asset;
use crate::spec::AssetSpec;
use alloc::vec;
use alloc::vec::Vec;

pub fn assets() -> Vec<AssetSpec> {
    vec![
        asset::directional_light("stage_key", [1.0, 0.97, 0.92], [-0.4, 0.8, 0.3], 2.2),
        asset::point_light("stage_fill_l", [0.6, 0.7, 1.0], [-4.0, 3.0, 2.0], 8.0, 20.0),
        asset::point_light("stage_fill_r", [1.0, 0.8, 0.6], [4.0, 2.5, -2.0], 6.0, 20.0),
    ]
}

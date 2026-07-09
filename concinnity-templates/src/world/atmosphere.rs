// An atmosphere pass: volumetric fog plus a procedural-sky environment, for a
// hazy outdoor mood without touching the lighting.

use crate::asset;
use crate::spec::AssetSpec;
use alloc::vec;
use alloc::vec::Vec;

pub fn assets() -> Vec<AssetSpec> {
    vec![
        asset::environment_map_sky("atmo_env"),
        asset::volumetric_fog("atmo_fog", 0.035, [0.7, 0.78, 0.9], 0.12, 140.0, 0.6),
    ]
}

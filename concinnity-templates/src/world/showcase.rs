// Aesthetic-defaults template: layers visual polish so a plain scene lands in a
// richer setting -- a warm key light, bloom + tonemap, an IBL environment from
// the procedural sky (no .hdr needed), and volumetric fog.

use crate::asset;
use crate::spec::AssetSpec;
use alloc::vec;
use alloc::vec::Vec;

pub fn assets() -> Vec<AssetSpec> {
    vec![
        asset::directional_light("ambient_light", [1.0, 0.95, 0.8], [-0.3, 0.85, 0.4], 1.1),
        asset::post_process("showcase_post", 0.7, 1.1, 0.0),
        asset::environment_map_sky("showcase_env"),
        asset::volumetric_fog("showcase_fog", 0.02, [0.75, 0.82, 0.95], 0.18, 180.0, 0.5),
    ]
}

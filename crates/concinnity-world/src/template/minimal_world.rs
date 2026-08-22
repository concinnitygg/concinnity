// A minimal 3D world: a navigable camera, a warm sun, a self-contained room, and
// a procedural-sky environment. Every asset is standalone (no source files, no
// cross-references), so applying the template drops a complete little scene into
// any fresh world -- something to look at the moment it renders.

use crate::spec::AssetSpec;
use crate::spec::asset;

pub(super) fn assets() -> Vec<AssetSpec> {
    vec![
        asset::camera("world_camera", [0.0, 2.4, 9.0], 0.0, -0.12),
        asset::directional_light("world_sun", [1.0, 0.96, 0.86], [-0.35, 0.85, 0.35], 2.2),
        asset::room("world_room", [16.0, 20.0, 5.0]),
        asset::environment_map_sky("world_sky"),
    ]
}

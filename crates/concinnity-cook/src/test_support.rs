// Fixtures shared by the crate's unit tests.

use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;

// An entry of `asset_type` with handle `id` and the given args.
pub(crate) fn asset(
    id: &str,
    asset_type: RegisteredType,
    args: serde_json::Value,
) -> WorldJsonlAsset {
    WorldJsonlAsset {
        id: id.to_string(),
        asset_type,
        args,
    }
}

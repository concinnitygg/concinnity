//! Shared scaffolding for the pipeline tests: the world-asset literal and the
//! on-disk source fixture.

pub(super) fn wja(
    name: &str,
    asset_type: crate::authoring::registry::RegisteredType,
    args: serde_json::Value,
) -> crate::authoring::world::WorldJsonlAsset {
    crate::authoring::world::WorldJsonlAsset {
        name: name.to_string(),
        asset_type,
        args,
    }
}

// Write a fixture container into `dir` and return its path as a string.
pub(super) fn write_fixture(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> String {
    concinnity_testing::utf8(&concinnity_testing::write_into(dir.path(), name, bytes))
}

//! Shared scaffolding for the pipeline tests: the world-asset literal and the
//! on-disk source fixture.

pub(super) fn wja(
    name: &str,
    ty: &str,
    args: serde_json::Value,
) -> crate::authoring::world::WorldJsonlAsset {
    crate::authoring::world::WorldJsonlAsset {
        name: name.to_string(),
        asset_type: ty.to_string(),
        args,
    }
}

// Write a fixture container into `dir` and return its path as a string.
pub(super) fn write_fixture(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> String {
    concinnity_testing::utf8(&concinnity_testing::write_into(dir.path(), name, bytes))
}

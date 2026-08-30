//! Shared scaffolding for the pipeline tests: the world-asset literal, the
//! on-disk source fixture, and the lock that serialises default-shader builds.

// Default-shader compilation writes intermediates to a shared
// data path keyed by asset name, so tests whose worlds pull in
// the default Shader (any rendering world) must not build concurrently.
pub(super) static SHADER_BUILD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).expect("write fixture");
    path.to_string_lossy().into_owned()
}

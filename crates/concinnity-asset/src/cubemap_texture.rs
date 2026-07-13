// HDR cubemap texture schema.

use crate::AssetId;
use crate::PayloadLocator;
use alloc::string::String;

/// A six-face HDR cubemap baked from an equirectangular Radiance HDR source.
///
/// The build resamples the source into six square HDR faces of `face_size`
/// pixels each, used as an environment / image-based-lighting source.
///
/// ```jsonl
/// {"name":"env_studio","type":"CubemapTexture","args":{"source":"assets/hdri/studio.hdr","face_size":512}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CubemapTexture {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Path to the source equirectangular HDR (`.hdr`) file, relative to the
    /// project root.
    pub source: String,
    /// Edge length of each cube face in pixels. Must be a power of two.
    pub face_size: u32,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Default for CubemapTexture {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            source: String::new(),
            face_size: 256,
            locator: None,
        }
    }
}

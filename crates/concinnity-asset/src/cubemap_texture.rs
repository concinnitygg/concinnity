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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_cubemap_bakes_at_the_default_face_size() {
        let c = CubemapTexture::default();
        assert!(c.source.is_empty());
        assert_eq!(c.face_size, 256);
        assert_eq!(c.asset_id, AssetId::default());
        assert!(c.locator.is_none());
    }

    #[test]
    fn an_authored_face_size_parses_and_round_trips_through_postcard() {
        let c: CubemapTexture =
            serde_json::from_str(r#"{"source":"sky.hdr","face_size":1024}"#).unwrap();
        assert_eq!(c.source, "sky.hdr");
        assert_eq!(c.face_size, 1024);

        let bytes = postcard::to_allocvec(&c).unwrap();
        let back: CubemapTexture = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.face_size, 1024);
        // Identity and payload location are injected, never carried on the wire.
        assert_eq!(back.asset_id, AssetId::default());
        assert!(back.locator.is_none());
    }
}

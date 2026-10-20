// HDR cubemap texture schema.

use crate::ecs::PayloadLocator;
use alloc::string::String;

/// A six-face HDR cubemap baked from an equirectangular Radiance HDR source.
///
/// The build resamples the source into six square HDR faces of `face_size`
/// pixels each, used as an environment / image-based-lighting source.
///
/// ```rust
/// # use concinnity_core::components::CubemapTexture;
/// CubemapTexture {
///     source: "assets/hdri/studio.hdr".into(),
///     face_size: 512,
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct CubemapTexture {
    /// Path to the source equirectangular HDR (`.hdr`) file, relative to the
    /// project root.
    pub source: String,
    /// Edge length of each cube face in pixels. Must be a power of two.
    #[asset(default = 256)]
    pub face_size: u32,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

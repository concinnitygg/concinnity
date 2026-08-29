// 2D texture image schema.

use crate::ecs::PayloadLocator;
use crate::ecs::asset_id::AssetId;
use alloc::string::String;

/// A 2D texture image.
///
/// Use the `generator` field for built-in patterns or supply a `source` file path.
///
/// **Built-in generators:**
///
/// **Choosing a room texture**: for neutral indoor spaces prefer `plaster` (cream-white) or `concrete` (grey). `brick` is reddish-orange, only use it when you explicitly want that look. `stone` (dark grey-blue) suits dungeons or medieval rooms.
///
/// ```rust
/// # use concinnity_core::components::Texture;
/// Texture {
///     generator: "brick".into(),
///     resolution: 512,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Texture {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Procedural generator name. Empty or omitted means use `source` instead.
    pub generator: String,
    /// Path to the source image, relative to the project root.
    /// Used only when `generator` is empty. A `.glb` path is allowed, use
    /// `image_index` to pick which embedded image to use.
    pub source: String,
    /// When `source` points to a `.glb` file, which embedded image to import.
    /// Ignored for regular image files.
    pub image_index: u32,
    /// Resolution hint for procedural generators (width = height). Defaults to
    /// 512. Ignored for file-backed textures.
    pub resolution: u32,
    /// Optional ceiling on the longest edge of a file-backed image, in pixels.
    /// `0` (the default) keeps the source resolution. When set and the source is
    /// larger, the image is box-filtered down so its longest edge is at most this
    /// value. Useful to keep very large source maps (4K+) from bloating the
    /// compiled scene, which stores uncompressed pixels.
    pub max_size: u32,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Default for Texture {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            generator: String::new(),
            source: String::new(),
            image_index: 0,
            resolution: 512,
            max_size: 0,
            locator: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_texture_generates_at_the_default_resolution_and_is_uncapped() {
        let t = Texture::default();
        assert!(t.source.is_empty());
        assert!(t.generator.is_empty());
        assert_eq!(t.image_index, 0);
        assert_eq!(t.resolution, 512);
        // Zero means "no downscale cap", not a zero-sized image.
        assert_eq!(t.max_size, 0);
        assert!(t.locator.is_none());
    }

    #[test]
    fn a_generated_texture_names_its_generator_instead_of_a_source() {
        let t: Texture =
            serde_json::from_str(r#"{"generator":"checker","resolution":128}"#).unwrap();
        assert_eq!(t.generator, "checker");
        assert!(t.source.is_empty());
        assert_eq!(t.resolution, 128);
    }

    #[test]
    fn an_imported_image_parses_and_round_trips_through_postcard() {
        let t: Texture =
            serde_json::from_str(r#"{"source":"bistro.fbx","image_index":7,"max_size":1024}"#)
                .unwrap();
        // The index picks one image out of a multi-image source archive.
        assert_eq!(t.image_index, 7);

        let bytes = postcard::to_allocvec(&t).unwrap();
        let back: Texture = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.source, "bistro.fbx");
        assert_eq!(back.image_index, 7);
        assert_eq!(back.max_size, 1024);
        assert_eq!(back.asset_id, AssetId::default());
        assert!(back.locator.is_none());
    }
}

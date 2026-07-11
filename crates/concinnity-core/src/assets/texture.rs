// src/assets/texture.rs
//
// `Texture`'s `Component` impl is generated centrally (see
// `cn_impl_components!`); this module keeps only its build-time source binding.

use crate::assets::Texture;

impl crate::build::SourceBacked for Texture {
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        // Procedural textures (with a non-empty `generator`) have no source file.
        if args
            .get("generator")
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false)
        {
            return None;
        }
        args.get("source")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}

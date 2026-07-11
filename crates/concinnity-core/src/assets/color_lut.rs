// src/assets/color_lut.rs
//
// `ColorLut`'s `Component` impl is generated centrally (see
// `cn_impl_components!`); this module keeps only its build-time source binding.

use crate::assets::ColorLut;

impl crate::build::SourceBacked for ColorLut {
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        args.get("source")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}

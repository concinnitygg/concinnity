// src/assets/font.rs
//
// `Font`'s `Component` impl is generated centrally (see `cn_impl_components!`);
// this module keeps only its build-time source binding.

use crate::assets::Font;

impl crate::build::SourceBacked for Font {
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        args.get("path")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}

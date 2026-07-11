// src/assets/mesh.rs
//
// `Mesh`'s `Component` impl is generated centrally (see `cn_impl_components!`);
// this module keeps only its build-time source binding.

use crate::assets::Mesh;

impl crate::build::SourceBacked for Mesh {
    // A glTF-sourced Mesh needs its `.glb` fetched before the build's desugar
    // pass can expand it; an inline-authored mesh has no source.
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        args.get("source")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}

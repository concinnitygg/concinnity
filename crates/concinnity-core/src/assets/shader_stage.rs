// src/assets/shader_stage.rs
//
// Runtime behavior for the ShaderStage asset. The authored schema
// (ShaderKind, the ShaderStage struct, and its Default) lives in
// concinnity-asset; this file keeps the `Component` impl and the
// `ShaderStageExt::current_platform_source` extension the engine init and
// hot-reload paths use. The JSON-args source selection and validation live in
// concinnity-world (`source_args`, `check::shader`). The schema types are
// re-exported so `crate::assets::shader_stage::ShaderKind` paths keep
// resolving.

pub use concinnity_asset::{ShaderKind, ShaderStage};

use crate::ecs::{Component, PayloadLocator};

// Resolve the source filename for the current build platform from a stage's
// declared `source` / `sources`. Mirrors the build-time selection
// (concinnity-world `source_args`) so the hot-reload subsystem picks the
// same per-platform source the build read at compile time. Returns `None` when
// no current-platform source is declared (e.g. a stage that only declares `glsl`
// running on the Metal backend, which loads the embedded GLSL fallback at init
// and has no on-disk file to hot-reload). Exposed as an extension trait because
// the schema type now lives in concinnity-asset.
pub trait ShaderStageExt {
    fn current_platform_source(&self) -> Option<String>;
}

impl ShaderStageExt for ShaderStage {
    fn current_platform_source(&self) -> Option<String> {
        let platform = crate::platform::Platform::current();
        if let Some(sources) = &self.sources
            && let Some(src) = sources.get(platform.key())
        {
            return Some(src.clone());
        }
        if self.source.is_empty() {
            return None;
        }
        let ext = std::path::Path::new(&self.source)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        if platform.accepts_ext(ext) {
            Some(self.source.clone())
        } else {
            None
        }
    }
}

impl Component for ShaderStage {
    const NAME: &'static str = "ShaderStage";

    fn from_baked(bytes: &[u8]) -> Result<Self, crate::result::CnResult> {
        Ok(postcard::from_bytes(bytes)?)
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }
}

/// Returns the platform key used to look up entries in the `sources` map.
pub fn platform_key() -> &'static str {
    crate::platform::Platform::current().key()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_kind_maps_each_stage() {
        assert_eq!(ShaderKind::Vertex.compile_kind(), "vertex");
        assert_eq!(ShaderKind::VertexInstanced.compile_kind(), "vertex");
        assert_eq!(ShaderKind::Fragment.compile_kind(), "fragment");
        assert_eq!(ShaderKind::default(), ShaderKind::Vertex);
    }

    #[test]
    fn default_declares_metal_and_hlsl_sources() {
        let s = ShaderStage::default();
        let sources = s.sources.expect("default has a sources map");
        assert_eq!(
            sources.get("metal").map(String::as_str),
            Some("default.metal")
        );
        assert_eq!(
            sources.get("hlsl").map(String::as_str),
            Some("default_vert.hlsl")
        );
        assert!(s.source.is_empty());
    }

    #[test]
    fn current_platform_source_resolves_for_any_backend() {
        // Declaring every platform source resolves on whichever backend the
        // test build targets.
        let stage = ShaderStage {
            kind: ShaderKind::Vertex,
            source: String::new(),
            sources: Some(
                [("metal", "v.metal"), ("hlsl", "v.hlsl"), ("glsl", "v.glsl")]
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            ),
            locator: None,
        };
        assert!(stage.current_platform_source().is_some());
    }
}

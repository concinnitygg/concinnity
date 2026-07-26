// src/assets/shader.rs
//
// Runtime behavior for the Shader asset. The authored schema (Shader,
// StageSource, ShaderKind, and the ShaderPayload container) lives in
// concinnity-asset; this file keeps the `Component` impl and the
// `StageSourceExt::current_platform_source` extension the engine init and
// hot-reload paths use. The JSON-args source selection and validation live in
// concinnity-world (`source_args`, `check::shader`).

pub use concinnity_asset::{Shader, ShaderKind, ShaderPayload, StageSource};

use crate::ecs::{Component, PayloadLocator};

// Resolve the source filename for the current build platform from a stage's
// declared `source` / `sources`. Mirrors the build-time selection
// (concinnity-world `source_args`) so the hot-reload subsystem picks the
// same per-platform source the build read at compile time. Returns `None` when
// no current-platform source is declared (e.g. a stage that only declares `glsl`
// running on the Metal backend, which loads the embedded GLSL fallback at init
// and has no on-disk file to hot-reload). Exposed as an extension trait because
// the schema type lives in concinnity-asset.
pub trait StageSourceExt {
    fn current_platform_source(&self) -> Option<String>;
}

impl StageSourceExt for StageSource {
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
        let ext = super::path_extension(&self.source).unwrap_or("");
        if platform.accepts_ext(ext) {
            Some(self.source.clone())
        } else {
            None
        }
    }
}

impl Component for Shader {
    const NAME: &'static str = "Shader";

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
    fn current_platform_source_resolves_for_any_backend() {
        // Declaring every platform source resolves on whichever backend the
        // test build targets.
        let stage = StageSource::per_platform([
            ("metal", "v.metal"),
            ("hlsl", "v.hlsl"),
            ("glsl", "v.glsl"),
        ]);
        assert!(stage.current_platform_source().is_some());
    }

    #[test]
    fn single_source_resolves_only_for_matching_extensions() {
        let stage = StageSource {
            source: "v.metal".to_string(),
            sources: None,
        };
        let platform = crate::platform::Platform::current();
        assert_eq!(
            stage.current_platform_source().is_some(),
            platform.accepts_ext("metal")
        );
    }
}

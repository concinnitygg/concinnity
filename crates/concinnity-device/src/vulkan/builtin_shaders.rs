// The Vulkan half of the single-source shader compile: everything the
// declarations in `concinnity_core::render::shader_programs::vk::TABLE` need
// a compiler, a content-addressed cache, or a filesystem for, over the
// declarations it brings into the backend's scope.

use concinnity_core::platform::Platform;
use concinnity_core::render::error::RenderResult;
pub(super) use concinnity_core::render::shader_programs::shared::*;
pub(super) use concinnity_core::render::shader_programs::spd::{
    HIZ_SPD_MSAA, HIZ_SPD_SINGLE, HIZ_SPD_TAIL,
};
use concinnity_shader::HlslTarget;

use crate::shader::builtin::{Backend, Program};

const VULKAN: Backend = Backend {
    platform: Platform::Vulkan,
    target: |_| HlslTarget::Spirv,
    embedded: embedded_spirv,
};

// What a declaration can do once a compiler and a cache are in reach. A trait
// because `ShaderProgram` is defined in `core::render`, which is `no_std` and
// knows nothing about either.
pub(crate) trait CompileProgram: Program {
    // Assemble the exact source text this variant compiles.
    #[cfg(test)]
    fn source(&self, hot_reload: bool) -> String {
        VULKAN.source(self.variant(), hot_reload)
    }

    // The SPIR-V for this variant, embedded, cached or compiled (see
    // `shader::builtin`).
    fn compile(&self, hot_reload: bool) -> RenderResult<Vec<u8>> {
        VULKAN.compile(self.variant(), hot_reload)
    }
}

impl<P: Program> CompileProgram for P {}

// The build script's precompiled SPIR-V. Every name misses on a host that had
// no shader compiler at build time, and the renderer compiles instead -- the
// only path that then needs one at runtime.
include!(concat!(env!("OUT_DIR"), "/engine_spirv.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gates_lead_and_nothing_else_is_injected() {
        const BACKEND: &str = "#define CN_BACKEND_VULKAN 1\n";
        for program in [&MAIN_BINDLESS_FRAG, &SSR_RESOLVE, &LIGHT_CULL] {
            let src = program.source(false);
            assert!(!src.strip_prefix(BACKEND).unwrap().starts_with("#define"));
        }
        let src = HIZ_SPD_MSAA.source(false);
        assert!(src.starts_with(&format!("{BACKEND}#define HIZ_SPD_MSAA 1\n")));
        let src = FOG_FRAG.at(true).source(false);
        assert!(src.starts_with(&format!("{BACKEND}#define USE_MSAA 1\n")));
    }

    #[test]
    fn every_key_field_tracks_the_program() {
        let a = MAIN_BINDLESS_VERT.source(false);
        let key = VULKAN.cache_key(MAIN_BINDLESS_VERT.variant(), &a);
        assert_eq!(key.compiler, "hlsl");
        assert_eq!(key.target, "spirv");
        assert_eq!(key.entry, "vertex_main_bindless");
    }

    // Every variant the table declares is embedded under the name the renderer
    // looks it up by, with the digest of the source it assembles in either
    // hot-reload mode, so an unedited binary compiles nothing at init. A build
    // host without a compiler embeds nothing, and every name misses together.
    #[test]
    fn every_variant_is_embedded_under_its_lookup_name() {
        use concinnity_core::render::shader_programs::vk::TABLE;
        use concinnity_core::render::shader_source::source_digest;

        let embedded: Vec<_> = TABLE
            .variants()
            .map(|v| (v, embedded_spirv(&v.artifact_name())))
            .collect();
        if embedded.iter().all(|(_, e)| e.is_none()) {
            return;
        }
        for (v, e) in embedded {
            let name = v.artifact_name();
            let (digest, _) = e.unwrap_or_else(|| panic!("{name}: not embedded"));
            for hot_reload in [false, true] {
                assert_eq!(digest, source_digest(&v.source(hot_reload)), "{name}");
            }
        }
    }
}

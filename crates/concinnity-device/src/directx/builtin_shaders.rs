// The DirectX half of the single-source shader compile: everything the
// declarations in `concinnity_core::render::shader_programs::dx::TABLE` need
// a compiler, a content-addressed cache, or a filesystem for, over the
// declarations it brings into the backend's scope.

use concinnity_core::platform::Platform;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::shader_programs::dx;
pub(super) use concinnity_core::render::shader_programs::shared::*;
pub(super) use concinnity_core::render::shader_programs::spd::{
    HIZ_SPD_MSAA, HIZ_SPD_SINGLE, HIZ_SPD_TAIL,
};
use concinnity_shader::HlslTarget;

use crate::shader::builtin::{Backend, Program};

const DIRECTX: Backend = Backend {
    platform: Platform::DirectX,
    target: |program| HlslTarget::Dxil {
        shader_model_6_5: dx::shader_model_6_5(program),
    },
    embedded: embedded_dxil,
};

// What a declaration can do once a compiler and a cache are in reach. A trait
// because `ShaderProgram` is defined in `core::render`, which is `no_std` and
// knows nothing about either.
pub(crate) trait CompileProgram: Program {
    // Assemble the exact source text this variant compiles.
    #[cfg(test)]
    fn source(&self, hot_reload: bool) -> String {
        DIRECTX.source(self.variant(), hot_reload)
    }

    // The DXIL for this variant, embedded, cached or compiled (see
    // `shader::builtin`).
    fn compile(&self, hot_reload: bool) -> RenderResult<Vec<u8>> {
        DIRECTX.compile(self.variant(), hot_reload)
    }
}

impl<P: Program> CompileProgram for P {}

// The build script's precompiled DXIL, keyed by each variant's artifact name.
// Every name misses on a host that had no shader compiler at build time, and
// the renderer compiles instead -- which is the only path that then needs one
// at runtime.
//
// The key is the artifact name rather than the entry point because one entry
// compiles into several programs: `fog_fragment` alone yields an MSAA and a
// non-MSAA DXIL from the same file, and the ray-traced glass entry yields four.
// Keying on anything the variant defines do not reach would hand one variant's
// bytes to another -- a wrong render, not a failed one. The program tables'
// soundness check in `core::render` locks it.
include!(concat!(env!("OUT_DIR"), "/engine_dxil.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_backend_define_and_variant_gates_lead_the_source() {
        const BACKEND: &str = "#define CN_BACKEND_DIRECTX 1\n";
        let after_backend = |p: &dyn CompileProgram| {
            let src = p.source(false);
            src.strip_prefix(BACKEND)
                .unwrap_or_else(|| {
                    panic!("{}: the backend define leads", p.variant().program.label)
                })
                .to_string()
        };
        assert!(after_backend(&HIZ_SPD_MSAA).starts_with("#define HIZ_SPD_MSAA 1\n"));
        assert!(after_backend(&FOG_FRAG.at(true)).starts_with("#define USE_MSAA 1\n"));
        // Neither the texture pool nor the probe set takes a count.
        assert!(!after_backend(&MAIN_BINDLESS_FRAG).starts_with("#define"));
        assert!(!after_backend(&SSR_RESOLVE).starts_with("#define"));
    }

    // The key must carry the fields that separate one program's artifact from
    // another's: the toolchain, the entry, and the shader model the target
    // implies.
    #[test]
    fn the_cache_key_tracks_the_program() {
        let src = MAIN_BINDLESS_VERT.source(false);
        let key = DIRECTX.cache_key(MAIN_BINDLESS_VERT.variant(), &src);
        assert_eq!(key.compiler, "hlsl");
        assert_eq!(key.entry, "vertex_main_bindless");
        assert_eq!(key.target, "dxil");
        let target = |v| DIRECTX.cache_key(v, &src).target;
        assert_eq!(target(RT_REFLECTIONS_FRAG.variant()), "dxil-6.5");
        assert_eq!(target(GLASS_FRAG_RT.at(true)), "dxil-6.5");
    }

    // Every variant the table declares is embedded under the name the renderer
    // looks it up by, with the digest of the source it assembles in either
    // hot-reload mode, so an unedited binary compiles nothing at init. A build
    // host without a compiler embeds nothing, and every name misses together.
    #[test]
    fn every_variant_is_embedded_under_its_lookup_name() {
        use concinnity_core::render::shader_programs::dx::TABLE;
        use concinnity_core::render::shader_source::source_digest;

        let embedded: Vec<_> = TABLE
            .variants()
            .map(|v| (v, embedded_dxil(&v.artifact_name())))
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

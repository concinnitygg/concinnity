// The DirectX half of the single-source shader compile: everything the
// declarations in `concinnity_engine::render::shader::programs::dx::TABLE` need
// a compiler, a content-addressed cache, or a filesystem for, over the
// declarations it brings into the backend's scope.

use concinnity_core::platform::Platform;
use concinnity_engine::render::backend::error::RenderResult;
pub(super) use concinnity_engine::render::shader::programs::commands::CULL_SCATTER;
use concinnity_engine::render::shader::programs::dx;
pub(super) use concinnity_engine::render::shader::programs::shared::*;
pub(super) use concinnity_engine::render::shader::programs::spd::{
    HIZ_SPD_MSAA, HIZ_SPD_SINGLE, HIZ_SPD_TAIL,
};
use concinnity_shader::HlslTarget;

use crate::shader::builtin::{Backend, Program};

const DIRECTX: Backend = Backend {
    platform: Platform::DirectX,
    target: |program| HlslTarget::Dxil {
        shader_model_6_5: dx::shader_model_6_5(program),
    },
};

// Every DirectX program as a player fetches it, for an export to ship.
pub(crate) fn shipped_plan() -> Vec<crate::shader::shipped::Planned> {
    DIRECTX.plan(&dx::TABLE)
}

// What a declaration can do once a compiler and a cache are in reach. A trait
// because `ShaderProgram` is defined in `core::render`, which is `no_std` and
// knows nothing about either.
pub(crate) trait CompileProgram: Program {
    // Assemble the exact source text this variant compiles.
    #[cfg(test)]
    fn source(&self, hot_reload: bool) -> String {
        DIRECTX.source(self.variant(), hot_reload)
    }

    // The DXIL for this variant, shipped, cached or compiled (see
    // `shader::builtin`).
    fn compile(&self, hot_reload: bool) -> RenderResult<Vec<u8>> {
        DIRECTX.compile(self.variant(), hot_reload)
    }
}

impl<P: Program> CompileProgram for P {}

#[cfg(test)]
mod tests {
    use super::*;

    // The plan an export compiles is every variant the table declares, under
    // the names and sources the runtime looks them up by.
    #[test]
    fn the_shipped_plan_is_every_variant_as_a_player_fetches_it() {
        use concinnity_engine::render::shader::source::source_digest;
        let plan: Vec<(String, u64)> = shipped_plan()
            .iter()
            .map(|p| (p.name.clone(), source_digest(&p.source)))
            .collect();
        let expected: Vec<(String, u64)> = dx::TABLE
            .variants()
            .map(|v| (v.artifact_name(), source_digest(&DIRECTX.source(v, false))))
            .collect();
        assert_eq!(plan, expected);
        let mut names: Vec<&str> = plan.iter().map(|(name, _)| name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), plan.len(), "two artifacts share a name");
    }

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
        assert_eq!(target(RT_REFLECTIONS_TRACE.variant()), "dxil-6.5");
        assert_eq!(target(GLASS_FRAG_RT.at(true)), "dxil-6.5");
    }
}

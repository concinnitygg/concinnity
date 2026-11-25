// The Vulkan half of the single-source shader compile: everything the
// declarations in `concinnity_engine::render::shader::programs::vk::TABLE` need
// a compiler, a content-addressed cache, or a filesystem for, over the
// declarations it brings into the backend's scope.

use concinnity_core::platform::Platform;
use concinnity_engine::render::backend::error::RenderResult;
pub(super) use concinnity_engine::render::shader::programs::commands::CULL_SCATTER;
pub(super) use concinnity_engine::render::shader::programs::shared::*;
pub(super) use concinnity_engine::render::shader::programs::spd::{
    HIZ_SPD_MSAA, HIZ_SPD_SINGLE, HIZ_SPD_TAIL,
};
use concinnity_shader::HlslTarget;

use crate::shader::builtin::{Backend, Program};

const VULKAN: Backend = Backend {
    platform: Platform::Vulkan,
    target: |_| HlslTarget::Spirv,
};

// Every Vulkan program as a player fetches it, for an export to ship.
pub(crate) fn shipped_plan() -> Vec<crate::shader::shipped::Planned> {
    VULKAN.plan(&concinnity_engine::render::shader::programs::vk::TABLE)
}

// What a declaration can do once a compiler and a cache are in reach. A trait
// because `ShaderProgram` is defined in `core::render`, which is `no_std` and
// knows nothing about either.
pub(crate) trait CompileProgram: Program {
    // Assemble the exact source text this variant compiles.
    #[cfg(test)]
    fn source(&self, hot_reload: bool) -> String {
        VULKAN.source(self.variant(), hot_reload)
    }

    // The SPIR-V for this variant, shipped, cached or compiled (see
    // `shader::builtin`).
    fn compile(&self, hot_reload: bool) -> RenderResult<Vec<u8>> {
        VULKAN.compile(self.variant(), hot_reload)
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
        let expected: Vec<(String, u64)> = concinnity_engine::render::shader::programs::vk::TABLE
            .variants()
            .map(|v| (v.artifact_name(), source_digest(&VULKAN.source(v, false))))
            .collect();
        assert_eq!(plan, expected);
        let mut names: Vec<&str> = plan.iter().map(|(name, _)| name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), plan.len(), "two artifacts share a name");
    }

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
}

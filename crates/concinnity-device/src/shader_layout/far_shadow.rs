//! The far shadow's sample, compiled into every pass that reads the sun's
//! shadow on every target, through the same program table the layout check
//! reflects. No pass binds a far map yet, so a stand-in tap reads the cascade
//! array.

use concinnity_core::platform::Platform;
use concinnity_engine::render::shader::programs::shared;
use concinnity_shader::HlslTarget;

use super::programs::{self, Program};

const TAP: (&str, &str) = (
    "FAR_SHADOW_TAP(uv, ref)",
    "shadow_map_cmp(float3((uv), 0.0), (ref))",
);

static GRASS_FRAG: Program = Program {
    row: &shared::GRASS_FRAG,
    splices: &[],
};

static WATER_FRAG: Program = Program {
    row: &shared::WATER_FRAG,
    splices: &[],
};

// The forward and canopy passes and the grass take the wide filter; the fog
// takes one tap; water and the raymarched surfaces read it too.
#[test]
fn the_far_shadow_sample_compiles_in_every_pass_that_reads_it() {
    concinnity_shader::require_dxc!();
    let targets = [
        (Platform::Vulkan, HlslTarget::Spirv),
        (
            Platform::DirectX,
            HlslTarget::Dxil {
                shader_model_6_5: false,
            },
        ),
        (Platform::Metal, HlslTarget::Msl),
    ];
    let passes = [
        &programs::MAIN_BINDLESS_FRAG,
        &programs::MAIN_BINDLESS_CANOPY_FRAG,
        &GRASS_FRAG,
        &WATER_FRAG,
        &programs::FOG_FROXEL,
        &programs::RAYMARCH_FRAG,
    ];
    let mut failures = Vec::new();
    for program in passes {
        for (platform, target) in targets {
            let source = program.source_with(platform, &[TAP]);
            assert!(source.contains("#define FAR_SHADOW_TAP(uv, ref) shadow_map_cmp"));
            let job = concinnity_shader::HlslJob {
                source: &source,
                file_name: program.row.file,
                entry: program.row.entry,
                target,
            };
            let work = crate::shader::compiler_work::dir().expect("compiler work dir");
            if let Err(e) = concinnity_shader::compile(&job, work.path()) {
                failures.push(format!("{} on {}: {e}", program.row.entry, platform.key()));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

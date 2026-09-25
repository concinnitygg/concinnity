//! The single-source programs the layout check reflects, and the per-target
//! invocation that reads their layouts back.
//!
//! One program per family is enough: a struct's declaration is shared by every
//! entry in its file, and the reflection keeps every block the source binds
//! whether the entry reads it or not, so any entry reports the same bytes. Where
//! two files declare the same struct name (`ShadowUniforms` is in both
//! `main_bindless.hlsl` and `fog.hlsl`) both are listed, because they are
//! separate declarations that can drift apart.
//!
//! Each program is a row of the backends' own tables (`shader_programs` in
//! core), so it reflects the gates a backend compiles it with: a variant
//! compiles only with its gate, and reflecting a family without it would read a
//! declaration no backend compiles. The backend define comes from the
//! assembler, as it does for them.

use std::collections::BTreeMap;

use crate::shader::source;
use concinnity_core::platform::Platform;
use concinnity_core::render::shader_programs::{ShaderProgram, metal, shared};
use concinnity_core::render::shader_source::Splice;
use concinnity_shader::layout::StructLayout;

// The module whose decorations state `platform`'s layout. The split is the
// point: MSL sizes a `float3` at 16 bytes where SPIR-V and DXIL pack a scalar
// after it, so a mirror has to be checked against each. The Vulkan and Metal
// legs read one compiled under the Vulkan artifact's layout rules --
// spirv-cross pads the MSL it emits to the offsets the module declares, so the
// two cannot differ -- and the DirectX leg one compiled under DirectX packing
// rules, which is what `-fvk-use-dx-layout` is for. Nothing on a non-Windows
// host can read a DXIL container's own reflection.
fn layout_target(platform: Platform) -> concinnity_shader::HlslTarget {
    match platform {
        Platform::Metal | Platform::Vulkan => concinnity_shader::HlslTarget::SpirvWithVulkanLayout,
        Platform::DirectX => concinnity_shader::HlslTarget::SpirvWithDxLayout,
    }
}

// One entry point to reflect.
pub(super) struct Program {
    pub row: &'static ShaderProgram,
    // Text spliced in at a marker no file in the shader tree can fill. Only the
    // raymarched volumes need one: their source is completed by a world's own
    // distance field, so reflecting them means supplying a stand-in for it.
    pub splices: &'static [Splice<'static>],
}

impl Program {
    // The exact text the renderer compiles for this variant on `platform`.
    fn source(&self, platform: Platform) -> String {
        let defines = self.row.at(false).defines();
        source::assemble(false, platform, self.row.file, &defines, self.splices)
    }
}

// The MSL the Metal backend compiles for `program`.
pub(super) fn msl(program: &Program) -> Result<String, String> {
    let source = program.source(Platform::Metal);
    let job = concinnity_shader::HlslJob {
        source: &source,
        file_name: program.row.file,
        entry: program.row.entry,
        target: concinnity_shader::HlslTarget::Msl,
    };
    let work = crate::shader::compiler_work::dir()?;
    concinnity_shader::compile(&job, work.path())
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .map_err(|e| format!("{} (metal): {e}", program.row.entry))
}

// Every struct `program` declares, laid out the way `platform` lays it out,
// read off the decorations of the SPIR-V module the artifact is built from,
// which states every offset outright.
pub(super) fn layouts(
    program: &Program,
    platform: Platform,
) -> Result<BTreeMap<String, StructLayout>, String> {
    let source = program.source(platform);
    let job = concinnity_shader::HlslJob {
        source: &source,
        file_name: program.row.file,
        entry: program.row.entry,
        target: layout_target(platform),
    };
    let work = crate::shader::compiler_work::dir()?;
    let spirv = concinnity_shader::compile(&job, work.path())
        .map_err(|e| format!("{} ({}): {e}", program.row.entry, platform.key()))?;
    concinnity_shader::layout::struct_layouts(&spirv)
        .map_err(|e| format!("{} ({}): {e}", program.row.entry, platform.key()))
}

// The fragment, whose MSL the argument-buffer checks read as well.
pub(super) static MAIN_BINDLESS_FRAG: Program = Program {
    row: &shared::MAIN_BINDLESS_FRAG,
    splices: &[],
};

// The same fragment compiled around a world `shade` that samples only the last
// texture member, through the last sampler.
pub(super) static MAIN_BINDLESS_FRAG_LATE_MEMBER_SHADE: Program = Program {
    splices: &[Splice::inline(
        "{SURFACE_FRAGMENT}",
        "float4 shade(VertexOut v, GpuObjectData od) { return float4(ltc_magnitude_sample(v.uv), 0.0, 1.0); }",
    )],
    ..MAIN_BINDLESS_FRAG
};

// The phase-1 variant: the only one that declares every struct the family has.
// Metal runs it as the decision half.
pub(super) static CULL_KERNEL: Program = Program {
    row: &shared::CULL_PHASE1,
    splices: &[],
};

pub(super) static LIGHT_CULL_KERNEL: Program = Program {
    row: &shared::LIGHT_CULL,
    splices: &[],
};

// The RT skinning kernel. `CN_BACKEND_DIRECTX` picks the DirectX root
// signature's slot numbering; the mesh payloads it walks are byte-addressed and so reflect no
// layout of their own (see `mesh_payload_offsets_match_the_kernel`).
pub(super) static RT_SKIN_KERNEL: Program = Program {
    row: &shared::RT_SKIN,
    splices: &[],
};

pub(super) static GBUFFER_PREPASS_VERT: Program = Program {
    row: &shared::GBUFFER_PREPASS_VERT_BINDLESS,
    splices: &[],
};

pub(super) static SHADOW_VERT: Program = Program {
    row: &shared::SHADOW_VERT,
    splices: &[],
};

pub(super) static GLASS_VERT: Program = Program {
    row: &shared::GLASS_VERT,
    splices: &[],
};

// The file is ray-traced only, but the ray query is unreachable from the vertex
// entry, so it compiles on every target.
pub(super) static GLASS_MESH_VERT: Program = Program {
    row: &shared::GLASS_MESH_VERT,
    splices: &[],
};

pub(super) static WATER_VERT: Program = Program {
    row: &shared::WATER_VERT,
    splices: &[],
};

pub(super) static RT_REFLECTIONS_FRAG: Program = Program {
    row: &shared::RT_REFLECTIONS_FRAG,
    splices: &[],
};

pub(super) static DECAL_VERT: Program = Program {
    row: &shared::DECAL_VERT,
    splices: &[],
};

pub(super) static LINE_VERT: Program = Program {
    row: &shared::LINE_VERT,
    splices: &[],
};

pub(super) static PARTICLE_VERT: Program = Program {
    row: &shared::PARTICLE_VERT,
    splices: &[],
};

pub(super) static TEXT_VERT: Program = Program {
    row: &shared::TEXT_VERT,
    splices: &[],
};

pub(super) static TAA_FRAG: Program = Program {
    row: &shared::TAA_FRAG,
    splices: &[],
};

pub(super) static BLOOM_PREFILTER: Program = Program {
    row: &shared::BLOOM_PREFILTER,
    splices: &[],
};

pub(super) static COMPOSITE_FRAG: Program = Program {
    row: &shared::COMPOSITE_FRAG,
    splices: &[],
};

pub(super) static SSAO_KERNEL: Program = Program {
    row: &shared::SSAO_KERNEL,
    splices: &[],
};

pub(super) static SSR_RESOLVE: Program = Program {
    row: &shared::SSR_RESOLVE,
    splices: &[],
};

pub(super) static SSGI_GATHER: Program = Program {
    row: &shared::SSGI_GATHER,
    splices: &[],
};

pub(super) static FOG_FROXEL: Program = Program {
    row: &shared::FOG_FROXEL,
    splices: &[],
};

pub(super) static AUTO_EXPOSURE_BUILD: Program = Program {
    row: &shared::AUTO_EXPOSURE_BUILD,
    splices: &[],
};

pub(super) static HIZ_INIT_SINGLE: Program = Program {
    row: &metal::HIZ_INIT_SINGLE,
    splices: &[],
};

// A stand-in distance field for the raymarch programs. Their source is only
// complete once a world supplies one, so reflecting them means splicing a field
// here -- synthetic, in source, because a test may not read a world's.
const SDF_STANDIN: Splice<'static> = Splice::inline(
    "{SDF_BODY}",
    "float map(float3 p, SdfParams q, float t) { return sdSphere(p, 0.5); }\n\
     SdfSurface shade(float3 p, float3 n, SdfParams q, float t, float2 uv) {\n\
         SdfSurface s; s.albedo = float3(1.0, 1.0, 1.0); s.roughness = 0.5;\n\
         s.metallic = 0.0; s.emissive = float3(0.0, 0.0, 0.0);\n\
         s.transmitted = float3(0.0, 0.0, 0.0); return s; }\n",
);

pub(super) static RAYMARCH_FRAG: Program = Program {
    row: &ShaderProgram {
        file: "raymarch.hlsl",
        entry: "raymarch_fragment",
        label: "raymarch_frag.hlsl",
        gates: &["RAYMARCH_SURFACE"],
        msaa: false,
    },
    splices: &[SDF_STANDIN],
};

// The cascade block is a push constant in every SPIR-V leg, and a push constant
// carries no binding for the reflection to keep, so it is read through the one
// entry that uses it.
pub(super) static RAYMARCH_SHADOW_VERT: Program = Program {
    row: &ShaderProgram {
        file: "raymarch.hlsl",
        entry: "raymarch_shadow_vertex",
        label: "raymarch_shadow_vert.hlsl",
        gates: &["RAYMARCH_SHADOW"],
        msaa: false,
    },
    splices: &[SDF_STANDIN],
};

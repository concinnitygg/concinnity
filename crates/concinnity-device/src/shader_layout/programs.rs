// src/shader_layout/programs.rs
//
// The single-source programs the layout check reflects, and the per-target
// invocation that reads their layouts back.
//
// One program per family is enough: a struct's declaration is shared by every
// entry in its file, so the smallest entry that declares it reports the same
// bytes the heaviest one does. Where two files declare the same struct name
// (`ShadowUniforms` is in both `main_bindless.slang` and `fog.slang`) both are
// listed, because they are separate declarations that can drift apart.
//
// The defines mirror the backends' own program tables
// (`{vulkan,directx}/slang_builtins.rs`, `metal/slang_shaders.rs`): a variant
// compiles only with its gate, and each backend adds its own host-shape gate on
// top -- `METAL_ABI` or `METAL_BINDINGS` where the Metal slots are pinned,
// `DXIL_ABI` where the root signature is. Reflecting a family without its gate
// would read a declaration no backend compiles.

use std::collections::BTreeMap;

use concinnity_slang as slang;

use crate::shader_layout::reflect::{self, ShaderStruct};
use crate::slang_source;

type Defines = &'static [(&'static str, &'static str)];

// Which backend's layout rules slangc applies. The split is the point: MSL
// sizes a `float3` at 16 bytes where SPIR-V and DXIL pack a scalar after it,
// so a mirror has to be checked against each.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Target {
    Metal,
    Vulkan,
    DirectX,
}

impl Target {
    pub(super) const ALL: [Target; 3] = [Target::Metal, Target::Vulkan, Target::DirectX];

    pub(super) fn label(self) -> &'static str {
        match self {
            Target::Metal => "metal",
            Target::Vulkan => "vulkan",
            Target::DirectX => "directx",
        }
    }

    // Reflection reads the same layout from the text targets as from the binary
    // ones (`hlsl` and `dxil` emit byte-identical reflection), and neither needs
    // a platform toolchain: `metallib` wants Xcode, `dxil` wants dxcompiler.
    fn slang_target(self, profile: &'static str) -> slang::SlangTarget {
        match self {
            Target::Metal => slang::SlangTarget::Metal,
            Target::Vulkan => slang::SlangTarget::Spirv,
            Target::DirectX => slang::SlangTarget::Hlsl(profile),
        }
    }
}

// One entry point to reflect.
pub(super) struct Program {
    pub file: &'static str,
    pub entry: &'static str,
    // Shader-model profile for the DirectX leg, from `directx/slang_builtins.rs`.
    pub profile: &'static str,
    // Variant gates and capacities every backend injects for this entry.
    pub common: Defines,
    // What each backend's own program table adds on top.
    pub metal: Defines,
    pub vulkan: Defines,
    pub directx: Defines,
}

impl Program {
    // The exact text the renderer compiles for this variant on `target`.
    fn source(&self, target: Target) -> String {
        let backend = match target {
            Target::Metal => self.metal,
            Target::Vulkan => self.vulkan,
            Target::DirectX => self.directx,
        };
        let defines: Vec<(&str, &str)> = self.common.iter().chain(backend).copied().collect();
        slang_source::assemble(false, self.file, &defines)
    }
}

// Every struct `program` declares, laid out the way `target` lays it out.
pub(super) fn layouts(
    program: &Program,
    target: Target,
) -> Result<BTreeMap<String, ShaderStruct>, String> {
    let source = program.source(target);
    let job = slang::SlangJob {
        source: &source,
        file_name: program.file,
        entries: &[program.entry],
        target: target.slang_target(program.profile),
    };
    let work_dir = std::env::temp_dir().join("cn_shader_layout");
    let json = slang::reflect(&job, &work_dir)
        .map_err(|e| format!("{} ({}): {e}", program.entry, target.label()))?;
    reflect::structs(&json)
}

// The reflection-probe array length and the bindless texture-pool capacity, as
// the backends bake them in. A const assert in `super` pins the first to the
// Rust constant the mirrored `ProbeSet` array uses; the pool sizes only the
// texture argument buffer, which no mirrored struct reads.
const PROBES: (&str, &str) = ("MAX_PROBES", "8");
const POOL: (&str, &str) = ("POOL_SIZE", "1024");

pub(super) static MAIN_BINDLESS_VERT: Program = Program {
    file: "main_bindless.slang",
    entry: "vertex_main_bindless",
    profile: "vs_6_0",
    common: &[PROBES],
    metal: &[("METAL_ABI", "1"), POOL],
    vulkan: &[POOL],
    directx: &[("DXIL_ABI", "1")],
};

pub(super) static LIGHT_CULL_KERNEL: Program = Program {
    file: "light_cull.slang",
    entry: "light_cull_kernel",
    profile: "cs_6_0",
    common: &[],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

// The RT skinning kernel. `METAL_BINDINGS` picks the Metal host's slot
// numbering; the mesh payloads it walks are byte-addressed and so reflect no
// layout of their own (see `mesh_payload_offsets_match_the_kernel`).
pub(super) static RT_SKIN_KERNEL: Program = Program {
    file: "rt_skin.slang",
    entry: "rt_skin",
    profile: "cs_6_5",
    common: &[],
    metal: &[("METAL_BINDINGS", "1")],
    vulkan: &[],
    directx: &[],
};

pub(super) static GBUFFER_PREPASS_VERT: Program = Program {
    file: "gbuffer_prepass.slang",
    entry: "gbuffer_prepass_vertex",
    profile: "vs_6_0",
    common: &[("GB_STATIC", "1")],
    metal: &[("METAL_BINDINGS", "1")],
    vulkan: &[],
    directx: &[("DXIL_ABI", "1")],
};

pub(super) static GBUFFER_PREPASS_FRAG: Program = Program {
    file: "gbuffer_prepass.slang",
    entry: "gbuffer_prepass_fragment",
    profile: "ps_6_0",
    common: &[("GB_FRAGMENT", "1")],
    metal: &[("METAL_BINDINGS", "1")],
    vulkan: &[],
    directx: &[("DXIL_ABI", "1")],
};

pub(super) static SHADOW_VERT: Program = Program {
    file: "shadow.slang",
    entry: "shadow_vertex_main",
    profile: "vs_6_0",
    common: &[("SHADOW_STATIC", "1")],
    metal: &[("METAL_BINDINGS", "1")],
    vulkan: &[],
    directx: &[("DXIL_ABI", "1")],
};

pub(super) static GLASS_VERT: Program = Program {
    file: "glass.slang",
    entry: "glass_vertex",
    profile: "vs_6_0",
    common: &[PROBES],
    metal: &[("METAL_ABI", "1")],
    vulkan: &[("USE_MSAA", "1")],
    directx: &[("DXIL_ABI", "1")],
};

// The glass mesh vertex stage declares both of its blocks: it reads the model
// matrix out of the per-mesh params. The file is ray-traced only, but the ray
// query is unreachable from the vertex entry, so slangc compiles it on the Metal
// target too and all three reflect.
pub(super) static GLASS_MESH_VERT: Program = Program {
    file: "glass_mesh.slang",
    entry: "glass_mesh_vertex",
    profile: "vs_6_0",
    common: &[PROBES],
    metal: &[("METAL_ABI", "1")],
    vulkan: &[("USE_MSAA", "1")],
    directx: &[("DXIL_ABI", "1")],
};

// The water vertex stage is the smallest entry that declares the whole water
// block set: the Gerstner sum reads the wave table, so `WaterParams` (and the
// `WaterWave` element it arrays) survive into the vertex reflection.
pub(super) static WATER_VERT: Program = Program {
    file: "water.slang",
    entry: "water_vertex",
    profile: "vs_6_0",
    common: &[PROBES],
    metal: &[("METAL_ABI", "1")],
    vulkan: &[("USE_MSAA", "1")],
    directx: &[("DXIL_ABI", "1")],
};

pub(super) static RT_REFLECTIONS_FRAG: Program = Program {
    file: "rt_reflections.slang",
    entry: "rt_reflections_fragment",
    profile: "ps_6_5",
    common: &[PROBES],
    metal: &[("METAL_ABI", "1")],
    vulkan: &[],
    directx: &[("DXIL_ABI", "1")],
};

pub(super) static DECAL_VERT: Program = Program {
    file: "decal.slang",
    entry: "decal_vertex",
    profile: "vs_6_0",
    common: &[],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

pub(super) static LINE_VERT: Program = Program {
    file: "line.slang",
    entry: "line_vertex",
    profile: "vs_6_0",
    common: &[],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

pub(super) static PARTICLE_VERT: Program = Program {
    file: "particle.slang",
    entry: "particle_vertex",
    profile: "vs_6_0",
    common: &[],
    metal: &[("METAL_BINDINGS", "1")],
    vulkan: &[],
    directx: &[("DXIL_ABI", "1")],
};

pub(super) static TEXT_VERT: Program = Program {
    file: "text.slang",
    entry: "text_vertex_main",
    profile: "vs_6_0",
    common: &[],
    metal: &[("METAL_BINDINGS", "1")],
    vulkan: &[],
    directx: &[],
};

pub(super) static TAA_FRAG: Program = Program {
    file: "taa.slang",
    entry: "taa_fragment_main",
    profile: "ps_6_0",
    common: &[],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

pub(super) static BLOOM_PREFILTER: Program = Program {
    file: "bloom.slang",
    entry: "bloom_prefilter_fragment",
    profile: "ps_6_0",
    common: &[("BLOOM_PREFILTER", "1")],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

pub(super) static COMPOSITE_FRAG: Program = Program {
    file: "composite.slang",
    entry: "composite_fragment",
    profile: "ps_6_0",
    common: &[],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

pub(super) static SSAO_KERNEL: Program = Program {
    file: "ssao.slang",
    entry: "ssao_kernel_fragment",
    profile: "ps_6_0",
    common: &[("SSAO_KERNEL", "1")],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

pub(super) static SSR_RESOLVE: Program = Program {
    file: "ssr.slang",
    entry: "ssr_resolve_fragment",
    profile: "ps_6_0",
    common: &[PROBES],
    metal: &[],
    vulkan: &[],
    directx: &[("SPLIT_PROBE_SAMPLER", "1")],
};

pub(super) static SSGI_GATHER: Program = Program {
    file: "ssgi.slang",
    entry: "ssgi_gather_fragment",
    profile: "ps_6_0",
    common: &[("SSGI_GATHER", "1")],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

pub(super) static FOG_FROXEL: Program = Program {
    file: "fog.slang",
    entry: "fog_froxel_kernel",
    profile: "cs_6_0",
    common: &[("FOG_FROXEL", "1")],
    metal: &[],
    vulkan: &[],
    directx: &[("DXIL_SPLIT", "1")],
};

pub(super) static AUTO_EXPOSURE_BUILD: Program = Program {
    file: "auto_exposure.slang",
    entry: "histogram_build",
    profile: "cs_6_0",
    common: &[("AE_BUILD", "1")],
    metal: &[("METAL_BINDINGS", "1")],
    vulkan: &[],
    directx: &[],
};

pub(super) static HIZ_INIT_SINGLE: Program = Program {
    file: "hiz_build.slang",
    entry: "hiz_init_single",
    profile: "cs_6_0",
    common: &[("HIZ_INIT_SINGLE", "1")],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

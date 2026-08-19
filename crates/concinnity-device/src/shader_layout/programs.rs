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
// sizes a constant-buffer `float3` at 16 bytes where SPIR-V and DXIL pack a
// scalar after it, so a mirror has to be checked against each.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Target {
    Metal,
    Vulkan,
    DirectX,
}

impl Target {
    pub const ALL: [Target; 3] = [Target::Metal, Target::Vulkan, Target::DirectX];

    pub fn label(self) -> &'static str {
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
    pub embedded: &'static str,
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
        slang_source::assemble(false, self.file, self.embedded, &defines)
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

const MAIN_BINDLESS: &str = include_str!("../shaders/main_bindless.slang");
const LIGHT_CULL: &str = include_str!("../shaders/light_cull.slang");
const GBUFFER_PREPASS: &str = include_str!("../shaders/gbuffer_prepass.slang");
const SHADOW: &str = include_str!("../shaders/shadow.slang");
const GLASS: &str = include_str!("../shaders/glass.slang");
const RT_REFLECTIONS: &str = include_str!("../shaders/rt_reflections.slang");
const DECAL: &str = include_str!("../shaders/decal.slang");
const LINE: &str = include_str!("../shaders/line.slang");
const PARTICLE: &str = include_str!("../shaders/particle.slang");
const TEXT: &str = include_str!("../shaders/text.slang");
const TAA: &str = include_str!("../shaders/taa.slang");
const BLOOM: &str = include_str!("../shaders/bloom.slang");
const COMPOSITE: &str = include_str!("../shaders/composite.slang");
const SSAO: &str = include_str!("../shaders/ssao.slang");
const SSR: &str = include_str!("../shaders/ssr.slang");
const SSGI: &str = include_str!("../shaders/ssgi.slang");
const FOG: &str = include_str!("../shaders/fog.slang");
const AUTO_EXPOSURE: &str = include_str!("../shaders/auto_exposure.slang");
const HIZ_BUILD: &str = include_str!("../shaders/hiz_build.slang");

// The reflection-probe array length and the bindless texture-pool capacity, as
// the backends bake them in. A const assert in `super` pins the first to the
// Rust constant the mirrored `ProbeSet` array uses; the pool sizes only the
// texture argument buffer, which no mirrored struct reads.
const PROBES: (&str, &str) = ("MAX_PROBES", "8");
const POOL: (&str, &str) = ("POOL_SIZE", "1024");

pub(super) static MAIN_BINDLESS_VERT: Program = Program {
    file: "main_bindless.slang",
    embedded: MAIN_BINDLESS,
    entry: "vertex_main_bindless",
    profile: "vs_6_0",
    common: &[PROBES],
    metal: &[("METAL_ABI", "1"), POOL],
    vulkan: &[POOL],
    directx: &[("DXIL_ABI", "1")],
};

pub(super) static LIGHT_CULL_KERNEL: Program = Program {
    file: "light_cull.slang",
    embedded: LIGHT_CULL,
    entry: "light_cull_kernel",
    profile: "cs_6_0",
    common: &[],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

pub(super) static GBUFFER_PREPASS_VERT: Program = Program {
    file: "gbuffer_prepass.slang",
    embedded: GBUFFER_PREPASS,
    entry: "gbuffer_prepass_vertex",
    profile: "vs_6_0",
    common: &[("GB_STATIC", "1")],
    metal: &[("METAL_BINDINGS", "1")],
    vulkan: &[],
    directx: &[("DXIL_ABI", "1")],
};

pub(super) static GBUFFER_PREPASS_FRAG: Program = Program {
    file: "gbuffer_prepass.slang",
    embedded: GBUFFER_PREPASS,
    entry: "gbuffer_prepass_fragment",
    profile: "ps_6_0",
    common: &[("GB_FRAGMENT", "1")],
    metal: &[("METAL_BINDINGS", "1")],
    vulkan: &[],
    directx: &[("DXIL_ABI", "1")],
};

pub(super) static SHADOW_VERT: Program = Program {
    file: "shadow.slang",
    embedded: SHADOW,
    entry: "shadow_vertex_main",
    profile: "vs_6_0",
    common: &[("SHADOW_STATIC", "1")],
    metal: &[("METAL_BINDINGS", "1")],
    vulkan: &[],
    directx: &[("DXIL_ABI", "1")],
};

pub(super) static GLASS_VERT: Program = Program {
    file: "glass.slang",
    embedded: GLASS,
    entry: "glass_vertex",
    profile: "vs_6_0",
    common: &[PROBES],
    metal: &[("METAL_ABI", "1")],
    vulkan: &[("USE_MSAA", "1")],
    directx: &[("DXIL_ABI", "1")],
};

pub(super) static RT_REFLECTIONS_FRAG: Program = Program {
    file: "rt_reflections.slang",
    embedded: RT_REFLECTIONS,
    entry: "rt_reflections_fragment",
    profile: "ps_6_5",
    common: &[PROBES],
    metal: &[("METAL_ABI", "1")],
    vulkan: &[],
    directx: &[("DXIL_ABI", "1")],
};

pub(super) static DECAL_VERT: Program = Program {
    file: "decal.slang",
    embedded: DECAL,
    entry: "decal_vertex",
    profile: "vs_6_0",
    common: &[],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

pub(super) static LINE_VERT: Program = Program {
    file: "line.slang",
    embedded: LINE,
    entry: "line_vertex",
    profile: "vs_6_0",
    common: &[],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

pub(super) static PARTICLE_VERT: Program = Program {
    file: "particle.slang",
    embedded: PARTICLE,
    entry: "particle_vertex",
    profile: "vs_6_0",
    common: &[],
    metal: &[("METAL_BINDINGS", "1")],
    vulkan: &[],
    directx: &[("DXIL_ABI", "1")],
};

pub(super) static TEXT_VERT: Program = Program {
    file: "text.slang",
    embedded: TEXT,
    entry: "text_vertex_main",
    profile: "vs_6_0",
    common: &[],
    metal: &[("METAL_BINDINGS", "1")],
    vulkan: &[],
    directx: &[],
};

pub(super) static TAA_FRAG: Program = Program {
    file: "taa.slang",
    embedded: TAA,
    entry: "taa_fragment_main",
    profile: "ps_6_0",
    common: &[],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

pub(super) static BLOOM_PREFILTER: Program = Program {
    file: "bloom.slang",
    embedded: BLOOM,
    entry: "bloom_prefilter_fragment",
    profile: "ps_6_0",
    common: &[("BLOOM_PREFILTER", "1")],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

pub(super) static COMPOSITE_FRAG: Program = Program {
    file: "composite.slang",
    embedded: COMPOSITE,
    entry: "composite_fragment",
    profile: "ps_6_0",
    common: &[],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

pub(super) static SSAO_KERNEL: Program = Program {
    file: "ssao.slang",
    embedded: SSAO,
    entry: "ssao_kernel_fragment",
    profile: "ps_6_0",
    common: &[("SSAO_KERNEL", "1")],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

pub(super) static SSR_RESOLVE: Program = Program {
    file: "ssr.slang",
    embedded: SSR,
    entry: "ssr_resolve_fragment",
    profile: "ps_6_0",
    common: &[PROBES],
    metal: &[],
    vulkan: &[],
    directx: &[("SPLIT_PROBE_SAMPLER", "1")],
};

pub(super) static SSGI_GATHER: Program = Program {
    file: "ssgi.slang",
    embedded: SSGI,
    entry: "ssgi_gather_fragment",
    profile: "ps_6_0",
    common: &[("SSGI_GATHER", "1")],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

pub(super) static FOG_FROXEL: Program = Program {
    file: "fog.slang",
    embedded: FOG,
    entry: "fog_froxel_kernel",
    profile: "cs_6_0",
    common: &[("FOG_FROXEL", "1")],
    metal: &[],
    vulkan: &[],
    directx: &[("DXIL_SPLIT", "1")],
};

pub(super) static AUTO_EXPOSURE_BUILD: Program = Program {
    file: "auto_exposure.slang",
    embedded: AUTO_EXPOSURE,
    entry: "histogram_build",
    profile: "cs_6_0",
    common: &[("AE_BUILD", "1")],
    metal: &[("METAL_BINDINGS", "1")],
    vulkan: &[],
    directx: &[],
};

pub(super) static HIZ_INIT_SINGLE: Program = Program {
    file: "hiz_build.slang",
    embedded: HIZ_BUILD,
    entry: "hiz_init_single",
    profile: "cs_6_0",
    common: &[("HIZ_INIT_SINGLE", "1")],
    metal: &[],
    vulkan: &[],
    directx: &[],
};

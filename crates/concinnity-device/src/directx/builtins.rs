// src/directx/builtins.rs
//
// The declarative table of every built-in HLSL program the DirectX backend
// compiles at runtime. Each program is declared exactly once: its source file,
// entry point, target profile, compiler, and how the compile source is
// assembled from the file (defines, prelude, probe_common / object_common
// injection). Renderer init and hot-reload compile through
// `HlslProgram::compile`, and the
// export-time precompile iterates `ALL` to populate a bundle's shader cache
// from the very same declarations, so the two can never drift.
//
// Not declared here: the single-source `.slang` programs (`slang_builtins`),
// which compile to DXIL through slangc rather than FXC; and the SdfVolume
// raymarch pipelines (raymarch.rs), whose fragment source embeds
// world-authored shader text and therefore cannot be enumerated ahead of a
// world. Both compile at init through the same cache.

use std::borrow::Cow;

use super::pipeline::{reflection_cut_prelude, shader_source};

// Shared reflection-probe sampling (box-parallax partition-of-unity blend),
// prepended to every shader that samples the probe cube array.
const PROBE_COMMON_HLSL: &str = include_str!("shaders/probe_common.hlsl");

// The shared bindless per-object record, substituted into every pass that
// strides the per-frame object StructuredBuffer at its `{OBJECT_DATA}` marker.
// Sole HLSL declaration of `GpuObjectData`: the main pass, G-buffer prepass,
// shadow pass and cull kernel all read the same buffer, so a per-shader copy is
// a silent layout-drift hazard.
const OBJECT_COMMON_HLSL: &str = include_str!("shaders/object_common.hlsl");

// Remap probe_common's register bindings for passes whose root signature
// places the probe cube array / sampler elsewhere than the forward pass.
const RT_PROBE_DEFINES: &str =
    "#define PROBE_CUBES_REGISTER t10\n#define PROBE_SAMPLER_REGISTER s3\n";
const GLASS_RT_PROBE_DEFINES: &str = "#define PROBE_CUBES_REGISTER t20\n";

pub(crate) enum Compiler {
    // FXC (`D3DCompile`), shader model 5.1.
    Fxc,
    // DXC (`dxcompiler.dll`), shader model 6.5 for DXR 1.1 inline ray tracing.
    Dxc,
}

// How a program's compile source is assembled from its shader file. Prefixes
// are prepended in the order below; `probe_common` slots in front of the body
// with a separating newline. The body's markers are substituted afterwards.
pub(crate) struct Assembly {
    // Prepend `#define USE_MSAA {0|1}` from `Ctx::msaa`.
    pub msaa: bool,
    // Prepend the reflection roughness-cut prelude (see `reflection_cut_prelude`).
    pub cut: bool,
    // Prepend probe_common.hlsl.
    pub probe: bool,
    // Register remap defines emitted ahead of probe_common, for passes whose
    // root signature places the cube array / sampler elsewhere than the forward
    // pass. Empty keeps probe_common's defaults.
    pub probe_registers: &'static str,
    // Substitute `{OBJECT_DATA}` with the shared `GpuObjectData` declaration.
    pub object_data: bool,
}

const PLAIN: Assembly = Assembly {
    msaa: false,
    cut: false,
    probe: false,
    probe_registers: "",
    object_data: false,
};

impl Assembly {
    // MSAA values this assembly can produce, for the export-time enumeration.
    fn msaa_variants(&self) -> &'static [bool] {
        if self.msaa { &[false, true] } else { &[false] }
    }
}

// Inputs a call site supplies to assemble a program's source.
pub(crate) struct Ctx {
    pub hot_reload: bool,
    pub msaa: bool,
}

impl Ctx {
    // For programs whose assembly ignores the MSAA state.
    pub fn plain(hot_reload: bool) -> Self {
        Self {
            hot_reload,
            msaa: false,
        }
    }
}

pub(crate) struct HlslProgram {
    // File name under `src/directx/shaders/` for the `cn debug` disk-first resolve.
    pub file: &'static str,
    pub embedded: &'static str,
    pub entry: &'static str,
    pub target: &'static str,
    pub compiler: Compiler,
    pub assembly: Assembly,
}

impl HlslProgram {
    fn body(&self, hot_reload: bool) -> Cow<'static, str> {
        shader_source(hot_reload, self.file, self.embedded)
    }

    // Assemble the exact source text this program compiles under `ctx`.
    pub fn source(&self, ctx: &Ctx) -> String {
        let assembly = &self.assembly;
        let mut src = String::new();
        if assembly.cut {
            src.push_str(&reflection_cut_prelude());
        }
        if assembly.msaa {
            src.push_str(msaa_define(ctx.msaa));
        }
        if assembly.probe {
            src.push_str(assembly.probe_registers);
            src.push_str(&shader_source(
                ctx.hot_reload,
                "probe_common.hlsl",
                PROBE_COMMON_HLSL,
            ));
            src.push('\n');
        }
        src.push_str(&self.body(ctx.hot_reload));
        if assembly.object_data {
            let object_common =
                shader_source(ctx.hot_reload, "object_common.hlsl", OBJECT_COMMON_HLSL);
            src = src.replace("{OBJECT_DATA}", &object_common);
        }
        src
    }

    pub fn compile(&self, ctx: &Ctx) -> Result<Vec<u8>, String> {
        let source = self.source(ctx);
        match self.compiler {
            Compiler::Fxc => super::pipeline::compile_hlsl(&source, self.entry, self.target),
            Compiler::Dxc => super::dxc::compile_hlsl_dxc(&source, self.entry, self.target),
        }
    }
}

fn msaa_define(msaa: bool) -> &'static str {
    if msaa {
        "#define USE_MSAA 1\n"
    } else {
        "#define USE_MSAA 0\n"
    }
}

// Compile every declared program (all enumerable variants) into `out_dir`,
// reusing local cache artifacts where present. DXC programs are skipped as a
// group when `dxcompiler.dll` is unavailable, mirroring the runtime fallback
// (RT stays off, SSR takes over), and reported rather than failing the export.
pub(crate) fn precompile(out_dir: &std::path::Path, report: &mut crate::precompile::Report) {
    for program in ALL {
        for &msaa in program.assembly.msaa_variants() {
            let ctx = Ctx {
                hot_reload: false,
                msaa,
            };
            let source = program.source(&ctx);
            let key = match program.compiler {
                Compiler::Fxc => {
                    super::pipeline::fxc_cache_key(&source, program.entry, program.target)
                }
                Compiler::Dxc => super::dxc::dxc_cache_key(&source, program.entry, program.target),
            };
            let compile = || match program.compiler {
                Compiler::Fxc => {
                    super::pipeline::compile_hlsl(&source, program.entry, program.target)
                }
                Compiler::Dxc => {
                    super::dxc::compile_hlsl_dxc(&source, program.entry, program.target)
                }
            };
            report.record(
                &format!("{} {}", program.entry, program.target),
                crate::shader_cache::ensure_in(out_dir, &key, compile),
            );
        }
    }
}

// Embedded sources shared by several programs.
const MAIN_VERT_HLSL: &str = include_str!("shaders/main_vert.hlsl");
const MAIN_FRAG_HLSL: &str = include_str!("shaders/main_frag.hlsl");
const SHADOW_VERT_HLSL: &str = include_str!("shaders/shadow_vert.hlsl");
const CULL_HLSL: &str = include_str!("shaders/cull.hlsl");
const AUTO_EXPOSURE_HLSL: &str = include_str!("shaders/auto_exposure.hlsl");
const GLASS_HLSL: &str = include_str!("shaders/glass.hlsl");
const GLASS_RT_HLSL: &str = include_str!("shaders/glass_rt.hlsl");
const RT_REFLECTIONS_HLSL: &str = include_str!("shaders/rt_reflections.hlsl");
const GBUFFER_PREPASS_FRAG_HLSL: &str = include_str!("shaders/gbuffer_prepass_frag.hlsl");
const GBUFFER_PREPASS_SKINNED_VERT_HLSL: &str =
    include_str!("shaders/gbuffer_prepass_skinned_vert.hlsl");

// Declaration shorthand: FXC, single `main` entry, no assembly.
const fn fxc_main(file: &'static str, embedded: &'static str, target: &'static str) -> HlslProgram {
    HlslProgram {
        file,
        embedded,
        entry: "main",
        target,
        compiler: Compiler::Fxc,
        assembly: PLAIN,
    }
}

pub(super) static TEXT_VERT: HlslProgram = fxc_main(
    "text_vert.hlsl",
    include_str!("shaders/text_vert.hlsl"),
    "vs_5_1",
);
pub(super) static TEXT_FRAG: HlslProgram = fxc_main(
    "text_frag.hlsl",
    include_str!("shaders/text_frag.hlsl"),
    "ps_5_1",
);

// The main geometry pass. Both vertex entry points share main_vert.hlsl.
pub(super) static MAIN_VERT: HlslProgram = HlslProgram {
    file: "main_vert.hlsl",
    embedded: MAIN_VERT_HLSL,
    entry: "vertex_main",
    target: "vs_5_1",
    compiler: Compiler::Fxc,
    assembly: PLAIN,
};
pub(super) static MAIN_VERT_INSTANCED: HlslProgram = HlslProgram {
    file: "main_vert.hlsl",
    embedded: MAIN_VERT_HLSL,
    entry: "vertex_main_instanced",
    target: "vs_5_1",
    compiler: Compiler::Fxc,
    assembly: PLAIN,
};
pub(super) static MAIN_FRAG: HlslProgram = fxc_main("main_frag.hlsl", MAIN_FRAG_HLSL, "ps_5_1");
pub(super) static SHADOW_VERT: HlslProgram =
    fxc_main("shadow_vert.hlsl", SHADOW_VERT_HLSL, "vs_5_1");

pub(super) static SHADOW_BINDLESS_VERT: HlslProgram = HlslProgram {
    assembly: Assembly {
        object_data: true,
        ..PLAIN
    },
    ..fxc_main(
        "shadow_bindless_vert.hlsl",
        include_str!("shaders/shadow_bindless_vert.hlsl"),
        "vs_5_1",
    )
};
pub(super) static SKINNED_VERT: HlslProgram = fxc_main(
    "skinned_vert.hlsl",
    include_str!("shaders/skinned_vert.hlsl"),
    "vs_5_1",
);
pub(super) static SKINNED_SHADOW_VERT: HlslProgram = fxc_main(
    "skinned_shadow_vert.hlsl",
    include_str!("shaders/skinned_shadow_vert.hlsl"),
    "vs_5_1",
);

const fn fxc_cull(entry: &'static str) -> HlslProgram {
    HlslProgram {
        entry,
        assembly: Assembly {
            object_data: true,
            ..PLAIN
        },
        ..fxc_main("cull.hlsl", CULL_HLSL, "cs_5_1")
    }
}

pub(super) static CULL: HlslProgram = fxc_cull("main");
pub(super) static CULL_PHASE2: HlslProgram = fxc_cull("main_phase2");
pub(super) static CULL_SHADOW: HlslProgram = fxc_cull("main_shadow");

pub(super) static AUTO_EXPOSURE_BUILD: HlslProgram = HlslProgram {
    entry: "build",
    ..fxc_main("auto_exposure.hlsl", AUTO_EXPOSURE_HLSL, "cs_5_1")
};
pub(super) static AUTO_EXPOSURE_AVERAGE: HlslProgram = HlslProgram {
    entry: "average",
    ..fxc_main("auto_exposure.hlsl", AUTO_EXPOSURE_HLSL, "cs_5_1")
};

pub(super) static FOG_VERT: HlslProgram = HlslProgram {
    assembly: Assembly {
        msaa: true,
        ..PLAIN
    },
    ..fxc_main(
        "fog_vert.hlsl",
        include_str!("shaders/fog_vert.hlsl"),
        "vs_5_1",
    )
};
pub(super) static FOG_FRAG: HlslProgram = HlslProgram {
    assembly: Assembly {
        msaa: true,
        ..PLAIN
    },
    ..fxc_main(
        "fog_frag.hlsl",
        include_str!("shaders/fog_frag.hlsl"),
        "ps_5_1",
    )
};
pub(super) static FOG_FROXEL: HlslProgram = fxc_main(
    "fog_froxel.hlsl",
    include_str!("shaders/fog_froxel.hlsl"),
    "cs_5_1",
);

pub(super) static DECAL_VERT: HlslProgram = HlslProgram {
    assembly: Assembly {
        msaa: true,
        ..PLAIN
    },
    ..fxc_main(
        "decal_vert.hlsl",
        include_str!("shaders/decal_vert.hlsl"),
        "vs_5_1",
    )
};
pub(super) static DECAL_FRAG: HlslProgram = HlslProgram {
    assembly: Assembly {
        msaa: true,
        ..PLAIN
    },
    ..fxc_main(
        "decal_frag.hlsl",
        include_str!("shaders/decal_frag.hlsl"),
        "ps_5_1",
    )
};

pub(super) static LINE_VERT: HlslProgram = fxc_main(
    "line_vert.hlsl",
    include_str!("shaders/line_vert.hlsl"),
    "vs_5_1",
);
pub(super) static LINE_FRAG: HlslProgram = HlslProgram {
    assembly: Assembly {
        msaa: true,
        ..PLAIN
    },
    ..fxc_main(
        "line_frag.hlsl",
        include_str!("shaders/line_frag.hlsl"),
        "ps_5_1",
    )
};

pub(super) static PARTICLE_SIMULATE: HlslProgram = fxc_main(
    "particle_simulate.hlsl",
    include_str!("shaders/particle_simulate.hlsl"),
    "cs_5_1",
);
pub(super) static PARTICLE_VERT: HlslProgram = fxc_main(
    "particle_vert.hlsl",
    include_str!("shaders/particle_vert.hlsl"),
    "vs_5_1",
);
pub(super) static PARTICLE_FRAG: HlslProgram = fxc_main(
    "particle_frag.hlsl",
    include_str!("shaders/particle_frag.hlsl"),
    "ps_5_1",
);

const GLASS_DECL: HlslProgram = HlslProgram {
    file: "glass.hlsl",
    embedded: GLASS_HLSL,
    entry: "vs_main",
    target: "vs_5_1",
    compiler: Compiler::Fxc,
    assembly: Assembly {
        msaa: true,
        probe: true,
        ..PLAIN
    },
};
pub(super) static GLASS_VERT: HlslProgram = GLASS_DECL;
pub(super) static GLASS_FRAG: HlslProgram = HlslProgram {
    entry: "ps_main",
    target: "ps_5_1",
    ..GLASS_DECL
};

pub(super) static GBUFFER_PREPASS_VERT: HlslProgram = fxc_main(
    "gbuffer_prepass_vert.hlsl",
    include_str!("shaders/gbuffer_prepass_vert.hlsl"),
    "vs_5_1",
);
pub(super) static GBUFFER_PREPASS_VERT_INSTANCED: HlslProgram = fxc_main(
    "gbuffer_prepass_instanced_vert.hlsl",
    include_str!("shaders/gbuffer_prepass_instanced_vert.hlsl"),
    "vs_5_1",
);
pub(super) static GBUFFER_PREPASS_VERT_SKINNED: HlslProgram = fxc_main(
    "gbuffer_prepass_skinned_vert.hlsl",
    GBUFFER_PREPASS_SKINNED_VERT_HLSL,
    "vs_5_1",
);
pub(super) static GBUFFER_PREPASS_FRAG: HlslProgram = fxc_main(
    "gbuffer_prepass_frag.hlsl",
    GBUFFER_PREPASS_FRAG_HLSL,
    "ps_5_1",
);
pub(super) static GBUFFER_BINDLESS_VERT: HlslProgram = HlslProgram {
    assembly: Assembly {
        object_data: true,
        ..PLAIN
    },
    ..fxc_main(
        "gbuffer_bindless_vert.hlsl",
        include_str!("shaders/gbuffer_bindless_vert.hlsl"),
        "vs_5_1",
    )
};
pub(super) static GBUFFER_BINDLESS_FRAG: HlslProgram = fxc_main(
    "gbuffer_bindless_frag.hlsl",
    include_str!("shaders/gbuffer_bindless_frag.hlsl"),
    "ps_5_1",
);

// SM 6.5 programs (DXC): hardware ray-traced reflections, RT glass, and the
// RT skinned-vertex refit kernel.
const RT_REFLECTIONS_DECL: HlslProgram = HlslProgram {
    file: "rt_reflections.hlsl",
    embedded: RT_REFLECTIONS_HLSL,
    entry: "rt_fullscreen_vert",
    target: "vs_6_5",
    compiler: Compiler::Dxc,
    assembly: Assembly {
        cut: true,
        probe: true,
        probe_registers: RT_PROBE_DEFINES,
        ..PLAIN
    },
};
pub(super) static RT_FULLSCREEN_VERT: HlslProgram = RT_REFLECTIONS_DECL;
pub(super) static RT_REFLECTIONS_FRAG: HlslProgram = HlslProgram {
    entry: "rt_reflections_frag",
    target: "ps_6_5",
    ..RT_REFLECTIONS_DECL
};
pub(super) static RT_REFLECTIONS_FRAG_TEXTURED: HlslProgram = HlslProgram {
    entry: "rt_reflections_frag_textured",
    target: "ps_6_5",
    ..RT_REFLECTIONS_DECL
};

const GLASS_RT_DECL: HlslProgram = HlslProgram {
    file: "glass_rt.hlsl",
    embedded: GLASS_RT_HLSL,
    entry: "vs_main",
    target: "vs_6_5",
    compiler: Compiler::Dxc,
    assembly: Assembly {
        msaa: true,
        probe: true,
        probe_registers: GLASS_RT_PROBE_DEFINES,
        ..PLAIN
    },
};
pub(super) static GLASS_RT_VERT: HlslProgram = GLASS_RT_DECL;
pub(super) static GLASS_RT_FRAG: HlslProgram = HlslProgram {
    entry: "ps_main_rt",
    target: "ps_6_5",
    ..GLASS_RT_DECL
};
pub(super) static GLASS_RT_FRAG_TEXTURED: HlslProgram = HlslProgram {
    entry: "ps_main_rt_textured",
    target: "ps_6_5",
    ..GLASS_RT_DECL
};

pub(super) static RT_SKIN: HlslProgram = HlslProgram {
    file: "rt_skin.hlsl",
    embedded: include_str!("shaders/rt_skin.hlsl"),
    entry: "rt_skin",
    target: "cs_6_5",
    compiler: Compiler::Dxc,
    assembly: PLAIN,
};

// Every declared program, iterated by the export-time precompile.
pub(crate) static ALL: &[&HlslProgram] = &[
    &TEXT_VERT,
    &TEXT_FRAG,
    &MAIN_VERT,
    &MAIN_FRAG,
    &MAIN_VERT_INSTANCED,
    &SHADOW_VERT,
    &SHADOW_BINDLESS_VERT,
    &SKINNED_VERT,
    &SKINNED_SHADOW_VERT,
    &CULL,
    &CULL_PHASE2,
    &CULL_SHADOW,
    &AUTO_EXPOSURE_BUILD,
    &AUTO_EXPOSURE_AVERAGE,
    &FOG_VERT,
    &FOG_FRAG,
    &FOG_FROXEL,
    &DECAL_VERT,
    &DECAL_FRAG,
    &LINE_VERT,
    &LINE_FRAG,
    &PARTICLE_SIMULATE,
    &PARTICLE_VERT,
    &PARTICLE_FRAG,
    &GLASS_VERT,
    &GLASS_FRAG,
    &GBUFFER_PREPASS_VERT,
    &GBUFFER_PREPASS_VERT_INSTANCED,
    &GBUFFER_PREPASS_VERT_SKINNED,
    &GBUFFER_PREPASS_FRAG,
    &GBUFFER_BINDLESS_VERT,
    &GBUFFER_BINDLESS_FRAG,
    &RT_FULLSCREEN_VERT,
    &RT_REFLECTIONS_FRAG,
    &RT_REFLECTIONS_FRAG_TEXTURED,
    &GLASS_RT_VERT,
    &GLASS_RT_FRAG,
    &GLASS_RT_FRAG_TEXTURED,
    &RT_SKIN,
];

#[cfg(test)]
mod tests {
    use super::*;

    // Two programs collide when they would compile identical source text to
    // the same entry + target with the same compiler; the table must not
    // declare the same slot twice.
    #[test]
    fn table_has_no_duplicate_programs() {
        let mut seen = std::collections::HashSet::new();
        for p in ALL {
            let compiler = match p.compiler {
                Compiler::Fxc => "fxc",
                Compiler::Dxc => "dxc",
            };
            let ctx = Ctx::plain(false);
            assert!(
                seen.insert((compiler, p.source(&ctx), p.entry, p.target)),
                "duplicate program: {} {}",
                p.entry,
                p.target
            );
        }
    }

    #[test]
    fn msaa_assemblies_enumerate_both_variants() {
        assert_eq!(FOG_VERT.assembly.msaa_variants(), &[false, true]);
        assert_eq!(GLASS_FRAG.assembly.msaa_variants(), &[false, true]);
        assert_eq!(GLASS_RT_FRAG.assembly.msaa_variants(), &[false, true]);
        assert_eq!(TEXT_VERT.assembly.msaa_variants(), &[false]);
        assert_eq!(RT_REFLECTIONS_FRAG.assembly.msaa_variants(), &[false]);
    }

    // The assembled text must match the shapes the pass code historically
    // built by hand: define first, then the optional preludes, then the body.
    #[test]
    fn assembly_orders_prefixes_before_the_body() {
        let ctx = Ctx {
            hot_reload: false,
            msaa: true,
        };
        let fog = FOG_VERT.source(&ctx);
        assert!(fog.starts_with("#define USE_MSAA 1\n"));

        let glass = GLASS_FRAG.source(&ctx);
        assert!(glass.starts_with("#define USE_MSAA 1\n"));
        assert!(glass.ends_with(GLASS_HLSL));
        assert!(glass.contains(PROBE_COMMON_HLSL));

        let rt = RT_REFLECTIONS_FRAG.source(&Ctx::plain(false));
        assert!(rt.starts_with("static const float REFLECTION_ROUGHNESS_CUT"));
        assert!(rt.contains("#define PROBE_CUBES_REGISTER t10"));
        assert!(rt.ends_with(RT_REFLECTIONS_HLSL));
    }

    // Every program's assembled source must embed its body, so a shader edit is
    // always visible to the cache key. A body carrying the `{OBJECT_DATA}`
    // marker is split by the substitution, so both halves are checked.
    #[test]
    fn every_program_source_contains_its_embedded_body() {
        for p in ALL {
            let src = p.source(&Ctx::plain(false));
            for part in p.embedded.split("{OBJECT_DATA}") {
                assert!(src.contains(part), "{} {} lost its body", p.entry, p.target);
            }
        }
    }

    // Every program that strides the per-frame object StructuredBuffer gets the
    // shared record spliced in, and no other program carries a stray
    // declaration: the whole point of the fragment is that `GpuObjectData`
    // exists exactly once.
    #[test]
    fn object_data_programs_splice_the_shared_record() {
        let mut spliced = 0usize;
        for p in ALL {
            let src = p.source(&Ctx::plain(false));
            let declares = src.contains("struct GpuObjectData");
            assert_eq!(
                declares, p.assembly.object_data,
                "{} {}: declares GpuObjectData = {declares}, object_data = {}",
                p.entry, p.target, p.assembly.object_data
            );
            assert!(
                !src.contains("{OBJECT_DATA}"),
                "{} {} left {{OBJECT_DATA}}",
                p.entry,
                p.target
            );
            spliced += usize::from(declares);
        }
        assert_eq!(spliced, 5, "object-data program count changed");
    }

    // The sky shell's half-extent tracks the camera far plane, so its corners
    // always fall outside it. Every vertex path that rasterises scene geometry
    // must pin sky verts to the far plane or those corners clip and the clear
    // colour shows through. Skinned and instanced paths are excluded: neither
    // ever carries skybox geometry. The bindless main pass ships from
    // `src/shaders/main_bindless.slang` and is locked in `slang_builtins`.
    #[test]
    fn scene_vertex_shaders_pin_sky_to_the_far_plane() {
        const MUST_PIN: &[(&str, &str)] = &[
            (
                "gbuffer_prepass_vert.hlsl",
                include_str!("shaders/gbuffer_prepass_vert.hlsl"),
            ),
            (
                "gbuffer_bindless_vert.hlsl",
                include_str!("shaders/gbuffer_bindless_vert.hlsl"),
            ),
        ];
        for (name, src) in MUST_PIN {
            assert!(
                src.contains("v.color.b > 1.5"),
                "{name} lost the skybox sentinel test"
            );
            assert!(
                src.contains("o.sv_pos.z = o.sv_pos.w"),
                "{name} lost the far-plane pin"
            );
        }
    }
}

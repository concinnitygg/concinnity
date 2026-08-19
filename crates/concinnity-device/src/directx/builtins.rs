// src/directx/builtins.rs
//
// The declarative table of every built-in HLSL program the DirectX backend
// compiles at runtime. Each program is declared exactly once: its source file,
// entry point, target profile, compiler, and whether its body takes the shared
// object_common injection. Renderer init and hot-reload compile through
// `HlslProgram::compile`, and the export-time precompile iterates `ALL` to
// populate a bundle's shader cache from the very same declarations, so the two
// can never drift.
//
// What is left here is the per-draw main pass, its skinned and instanced
// vertex siblings, the cull kernel and the RT skinned refit: everything else
// ships from `src/shaders/*.slang` (`slang_builtins`), which compiles to DXIL
// through slangc rather than FXC. Also not declared here: the SdfVolume
// raymarch pipelines (raymarch.rs), whose fragment source embeds
// world-authored shader text and therefore cannot be enumerated ahead of a
// world. Both compile at init through the same cache.

use std::borrow::Cow;

use super::pipeline::shader_source;

// The shared bindless per-object record, substituted into every pass that
// strides the per-frame object StructuredBuffer at its `{OBJECT_DATA}` marker.
// Sole HLSL declaration of `GpuObjectData`: the main pass, G-buffer prepass,
// shadow pass and cull kernel all read the same buffer, so a per-shader copy is
// a silent layout-drift hazard.
const OBJECT_COMMON_HLSL: &str = include_str!("shaders/object_common.hlsl");

pub(crate) enum Compiler {
    // FXC (`D3DCompile`), shader model 5.1.
    Fxc,
    // DXC (`dxcompiler.dll`), shader model 6.5 for DXR 1.1 inline ray tracing.
    Dxc,
}

pub(crate) struct HlslProgram {
    // File name under `src/directx/shaders/` for the `cn debug` disk-first resolve.
    pub file: &'static str,
    pub embedded: &'static str,
    pub entry: &'static str,
    pub target: &'static str,
    pub compiler: Compiler,
    // Substitute `{OBJECT_DATA}` with the shared `GpuObjectData` declaration.
    pub object_data: bool,
}

impl HlslProgram {
    fn body(&self, hot_reload: bool) -> Cow<'static, str> {
        shader_source(hot_reload, self.file, self.embedded)
    }

    // Assemble the exact source text this program compiles.
    pub fn source(&self, hot_reload: bool) -> String {
        let mut src = self.body(hot_reload).into_owned();
        if self.object_data {
            let object_common = shader_source(hot_reload, "object_common.hlsl", OBJECT_COMMON_HLSL);
            src = src.replace("{OBJECT_DATA}", &object_common);
        }
        src
    }

    pub fn compile(&self, hot_reload: bool) -> Result<Vec<u8>, String> {
        let source = self.source(hot_reload);
        match self.compiler {
            Compiler::Fxc => super::pipeline::compile_hlsl(&source, self.entry, self.target),
            Compiler::Dxc => super::dxc::compile_hlsl_dxc(&source, self.entry, self.target),
        }
    }
}

// Compile every declared program into `out_dir`, reusing local cache artifacts
// where present. DXC programs are skipped as a group when `dxcompiler.dll` is
// unavailable, mirroring the runtime fallback (RT stays off, SSR takes over),
// and reported rather than failing the export.
pub(crate) fn precompile(out_dir: &std::path::Path, report: &mut crate::precompile::Report) {
    for program in ALL {
        let source = program.source(false);
        let key = match program.compiler {
            Compiler::Fxc => super::pipeline::fxc_cache_key(&source, program.entry, program.target),
            Compiler::Dxc => super::dxc::dxc_cache_key(&source, program.entry, program.target),
        };
        let compile = || match program.compiler {
            Compiler::Fxc => super::pipeline::compile_hlsl(&source, program.entry, program.target),
            Compiler::Dxc => super::dxc::compile_hlsl_dxc(&source, program.entry, program.target),
        };
        report.record(
            &format!("{} {}", program.entry, program.target),
            crate::shader_cache::ensure_in(out_dir, &key, compile),
        );
    }
}

// Embedded sources shared by several programs.
const MAIN_VERT_HLSL: &str = include_str!("shaders/main_vert.hlsl");
const MAIN_FRAG_HLSL: &str = include_str!("shaders/main_frag.hlsl");
const CULL_HLSL: &str = include_str!("shaders/cull.hlsl");

// Declaration shorthand: FXC, single `main` entry, no object_data splice.
const fn fxc_main(file: &'static str, embedded: &'static str, target: &'static str) -> HlslProgram {
    HlslProgram {
        file,
        embedded,
        entry: "main",
        target,
        compiler: Compiler::Fxc,
        object_data: false,
    }
}

// The main geometry pass. Both vertex entry points share main_vert.hlsl.
pub(super) static MAIN_VERT: HlslProgram = HlslProgram {
    file: "main_vert.hlsl",
    embedded: MAIN_VERT_HLSL,
    entry: "vertex_main",
    target: "vs_5_1",
    compiler: Compiler::Fxc,
    object_data: false,
};
pub(super) static MAIN_VERT_INSTANCED: HlslProgram = HlslProgram {
    file: "main_vert.hlsl",
    embedded: MAIN_VERT_HLSL,
    entry: "vertex_main_instanced",
    target: "vs_5_1",
    compiler: Compiler::Fxc,
    object_data: false,
};
pub(super) static MAIN_FRAG: HlslProgram = fxc_main("main_frag.hlsl", MAIN_FRAG_HLSL, "ps_5_1");
pub(super) static SKINNED_VERT: HlslProgram = fxc_main(
    "skinned_vert.hlsl",
    include_str!("shaders/skinned_vert.hlsl"),
    "vs_5_1",
);

const fn fxc_cull(entry: &'static str) -> HlslProgram {
    HlslProgram {
        entry,
        object_data: true,
        ..fxc_main("cull.hlsl", CULL_HLSL, "cs_5_1")
    }
}

pub(super) static CULL: HlslProgram = fxc_cull("main");
pub(super) static CULL_PHASE2: HlslProgram = fxc_cull("main_phase2");
pub(super) static CULL_SHADOW: HlslProgram = fxc_cull("main_shadow");

// The one SM 6.5 program left on this table (DXC): the RT skinned-vertex refit
// kernel. Its ray-traced siblings are single-source now.
pub(super) static RT_SKIN: HlslProgram = HlslProgram {
    file: "rt_skin.hlsl",
    embedded: include_str!("shaders/rt_skin.hlsl"),
    entry: "rt_skin",
    target: "cs_6_5",
    compiler: Compiler::Dxc,
    object_data: false,
};

// Every declared program, iterated by the export-time precompile.
pub(crate) static ALL: &[&HlslProgram] = &[
    &MAIN_VERT,
    &MAIN_FRAG,
    &MAIN_VERT_INSTANCED,
    &SKINNED_VERT,
    &CULL,
    &CULL_PHASE2,
    &CULL_SHADOW,
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
            assert!(
                seen.insert((compiler, p.source(false), p.entry, p.target)),
                "duplicate program: {} {}",
                p.entry,
                p.target
            );
        }
    }

    // Every program's assembled source must embed its body, so a shader edit is
    // always visible to the cache key. A body carrying the `{OBJECT_DATA}`
    // marker is split by the substitution, so both halves are checked.
    #[test]
    fn every_program_source_contains_its_embedded_body() {
        for p in ALL {
            let src = p.source(false);
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
            let src = p.source(false);
            let declares = src.contains("struct GpuObjectData");
            assert_eq!(
                declares, p.object_data,
                "{} {}: declares GpuObjectData = {declares}, object_data = {}",
                p.entry, p.target, p.object_data
            );
            assert!(
                !src.contains("{OBJECT_DATA}"),
                "{} {} left {{OBJECT_DATA}}",
                p.entry,
                p.target
            );
            spliced += usize::from(declares);
        }
        assert_eq!(spliced, 3, "object-data program count changed");
    }
}

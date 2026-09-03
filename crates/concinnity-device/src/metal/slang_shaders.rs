// Single-source engine shader libraries for the Metal backend.
//
// Each entry names one metallib variant compiled from a `.slang` file under
// `src/shaders/` (the backend-neutral single-source directory). The fast path
// loads the metallib the build script precompiled; source compilation remains
// for hot-reload (disk edits must win) and for binaries built on a host
// without slangc, whose embedded lookup misses these names. The runtime
// compile assembles its source through `crate::slang_source`, invokes slangc
// through `concinnity-slang`, and caches the metallib in the content-addressed
// shader cache, so a given source text compiles at most once per machine.
//
// The `.slang` declarations reproduce the engine's Metal binding layout
// exactly (see main_bindless.slang's METAL_ABI block); the build script's
// `assert_slang_metal_abi` locks the emitted slot assignment.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLFunction, MTLLibrary};

use concinnity_slang as slang;

use super::pipeline::{load_library, ns_str};

// The Metal texture-pool capacity and probe-array length the build script
// baked into the precompiled metallibs. Locked to the crate's own constants
// by `defines_match_the_crate_constants` below.
#[cfg(test)]
mod build_defines {
    include!(concat!(env!("OUT_DIR"), "/slang_metal_defines.rs"));
}

// One single-source metallib variant: the lookup `name` the build script
// registered, its `.slang` file, entry points, and variant defines.
pub(super) struct SlangLib {
    pub name: &'static str,
    pub file: &'static str,
    pub entries: &'static [&'static str],
    pub defines: &'static [(&'static str, &'static str)],
}

const MAIN_DEFINES: &[(&str, &str)] = &[
    ("METAL_ABI", "1"),
    ("POOL_SIZE", "1024"),
    ("MAX_PROBES", "8"),
];

// The SSR resolve reads the reflection-probe array, so it bakes in the same
// probe count the main pass does.
const PROBE_DEFINES: &[(&str, &str)] = &[("MAX_PROBES", "8")];

pub(super) static MAIN_BINDLESS_VERT: SlangLib = SlangLib {
    name: "main_bindless_vert.slang",
    file: "main_bindless.slang",
    entries: &["vertex_main_bindless"],
    defines: MAIN_DEFINES,
};
pub(super) static MAIN_BINDLESS_FRAG: SlangLib = SlangLib {
    name: "main_bindless_frag.slang",
    file: "main_bindless.slang",
    entries: &["fragment_main_bindless"],
    defines: MAIN_DEFINES,
};
pub(super) static LIGHT_CULL: SlangLib = SlangLib {
    name: "light_cull.slang",
    file: "light_cull.slang",
    entries: &["light_cull_kernel"],
    defines: &[],
};
pub(super) static RT_SKIN: SlangLib = SlangLib {
    name: "rt_skin.slang",
    file: "rt_skin.slang",
    entries: &["rt_skin"],
    defines: &[("METAL_BINDINGS", "1")],
};
pub(super) static HIZ_INIT_MSAA: SlangLib = SlangLib {
    name: "hiz_init_msaa.slang",
    file: "hiz_build.slang",
    entries: &["hiz_init_msaa"],
    defines: &[("HIZ_INIT_MSAA", "1")],
};
pub(super) static HIZ_DOWNSAMPLE: SlangLib = SlangLib {
    name: "hiz_downsample.slang",
    file: "hiz_build.slang",
    entries: &["hiz_downsample"],
    defines: &[("HIZ_DOWNSAMPLE", "1")],
};

// The runtime reflection-probe prefilter. One variant per kernel, so each
// declares exactly the textures it binds (Metal assigns indices in declaration
// order). `METAL_BINDINGS` puts the params at buffer(0), which is where the
// compute encoder writes them.
pub(super) static PROBE_MIP0: SlangLib = SlangLib {
    name: "probe_mip0.slang",
    file: "probe_prefilter.slang",
    entries: &["probe_mip0"],
    defines: &[("PROBE_MIP0", "1"), ("METAL_BINDINGS", "1")],
};
pub(super) static PROBE_DOWNSAMPLE: SlangLib = SlangLib {
    name: "probe_downsample.slang",
    file: "probe_prefilter.slang",
    entries: &["probe_downsample"],
    defines: &[("PROBE_DOWNSAMPLE", "1"), ("METAL_BINDINGS", "1")],
};
pub(super) static PROBE_GGX: SlangLib = SlangLib {
    name: "probe_ggx.slang",
    file: "probe_prefilter.slang",
    entries: &["probe_ggx"],
    defines: &[("PROBE_GGX", "1"), ("METAL_BINDINGS", "1")],
};

// The G-buffer pre-pass and shadow families. Every entry compiles on its own so
// it declares only the resources it binds, and `METAL_BINDINGS` selects the
// constant shape this host writes (see the file headers).
const GB_STATIC: &[(&str, &str)] = &[("GB_STATIC", "1"), ("METAL_BINDINGS", "1")];
const GB_INSTANCED: &[(&str, &str)] = &[("GB_INSTANCED", "1"), ("METAL_BINDINGS", "1")];
const GB_SKINNED: &[(&str, &str)] = &[("GB_SKINNED", "1"), ("METAL_BINDINGS", "1")];
const GB_BINDLESS: &[(&str, &str)] = &[("GB_BINDLESS", "1"), ("METAL_BINDINGS", "1")];
const GB_FRAGMENT: &[(&str, &str)] = &[("GB_FRAGMENT", "1"), ("METAL_BINDINGS", "1")];
const GB_FRAGMENT_BINDLESS: &[(&str, &str)] =
    &[("GB_FRAGMENT_BINDLESS", "1"), ("METAL_BINDINGS", "1")];
const SHADOW_STATIC: &[(&str, &str)] = &[("SHADOW_STATIC", "1"), ("METAL_BINDINGS", "1")];
const SHADOW_SKINNED: &[(&str, &str)] = &[("SHADOW_SKINNED", "1"), ("METAL_BINDINGS", "1")];
const SHADOW_BINDLESS: &[(&str, &str)] = &[("SHADOW_BINDLESS", "1"), ("METAL_BINDINGS", "1")];

pub(super) static GBUFFER_PREPASS_VERT: SlangLib = SlangLib {
    name: "gbuffer_prepass_vert.slang",
    file: "gbuffer_prepass.slang",
    entries: &["gbuffer_prepass_vertex"],
    defines: GB_STATIC,
};
pub(super) static GBUFFER_PREPASS_VERT_INSTANCED: SlangLib = SlangLib {
    name: "gbuffer_prepass_vert_instanced.slang",
    file: "gbuffer_prepass.slang",
    entries: &["gbuffer_prepass_vertex_instanced"],
    defines: GB_INSTANCED,
};
pub(super) static GBUFFER_PREPASS_VERT_SKINNED: SlangLib = SlangLib {
    name: "gbuffer_prepass_vert_skinned.slang",
    file: "gbuffer_prepass.slang",
    entries: &["gbuffer_prepass_vertex_skinned"],
    defines: GB_SKINNED,
};
pub(super) static GBUFFER_PREPASS_VERT_BINDLESS: SlangLib = SlangLib {
    name: "gbuffer_prepass_vert_bindless.slang",
    file: "gbuffer_prepass.slang",
    entries: &["gbuffer_prepass_vertex_bindless"],
    defines: GB_BINDLESS,
};
pub(super) static GBUFFER_PREPASS_FRAG: SlangLib = SlangLib {
    name: "gbuffer_prepass_frag.slang",
    file: "gbuffer_prepass.slang",
    entries: &["gbuffer_prepass_fragment"],
    defines: GB_FRAGMENT,
};
pub(super) static GBUFFER_PREPASS_FRAG_BINDLESS: SlangLib = SlangLib {
    name: "gbuffer_prepass_frag_bindless.slang",
    file: "gbuffer_prepass.slang",
    entries: &["gbuffer_prepass_fragment_bindless"],
    defines: GB_FRAGMENT_BINDLESS,
};
pub(super) static SHADOW_VERT: SlangLib = SlangLib {
    name: "shadow_vert.slang",
    file: "shadow.slang",
    entries: &["shadow_vertex_main"],
    defines: SHADOW_STATIC,
};
pub(super) static SHADOW_VERT_SKINNED: SlangLib = SlangLib {
    name: "shadow_vert_skinned.slang",
    file: "shadow.slang",
    entries: &["shadow_vertex_main_skinned"],
    defines: SHADOW_SKINNED,
};
pub(super) static SHADOW_VERT_BINDLESS: SlangLib = SlangLib {
    name: "shadow_vert_bindless.slang",
    file: "shadow.slang",
    entries: &["shadow_vertex_bindless"],
    defines: SHADOW_BINDLESS,
};

// The fullscreen-triangle vertex stage every ported post pass pairs with; one
// library serves them all, so their fragment metallibs carry no vertex entry.
pub(super) static FULLSCREEN_VERT: SlangLib = SlangLib {
    name: "fullscreen_vert.slang",
    file: "fullscreen.slang",
    entries: &["fullscreen_vertex"],
    defines: &[],
};
pub(super) static TAA_FRAG: SlangLib = SlangLib {
    name: "taa_frag.slang",
    file: "taa.slang",
    entries: &["taa_fragment_main"],
    defines: &[],
};
pub(super) static BLOOM_PREFILTER: SlangLib = SlangLib {
    name: "bloom_prefilter.slang",
    file: "bloom.slang",
    entries: &["bloom_prefilter_fragment"],
    defines: &[("BLOOM_PREFILTER", "1")],
};
pub(super) static BLOOM_DOWNSAMPLE: SlangLib = SlangLib {
    name: "bloom_downsample.slang",
    file: "bloom.slang",
    entries: &["bloom_downsample_fragment"],
    defines: &[("BLOOM_DOWNSAMPLE", "1")],
};
pub(super) static BLOOM_UPSAMPLE: SlangLib = SlangLib {
    name: "bloom_upsample.slang",
    file: "bloom.slang",
    entries: &["bloom_upsample_fragment"],
    defines: &[("BLOOM_UPSAMPLE", "1")],
};
pub(super) static COMPOSITE_FRAG: SlangLib = SlangLib {
    name: "composite_frag.slang",
    file: "composite.slang",
    entries: &["composite_fragment"],
    defines: &[],
};
pub(super) static SSAO_KERNEL: SlangLib = SlangLib {
    name: "ssao_kernel.slang",
    file: "ssao.slang",
    entries: &["ssao_kernel_fragment"],
    defines: &[("SSAO_KERNEL", "1")],
};
pub(super) static SSAO_BLUR: SlangLib = SlangLib {
    name: "ssao_blur.slang",
    file: "ssao.slang",
    entries: &["ssao_blur_fragment"],
    defines: &[("SSAO_BLUR", "1")],
};
pub(super) static SSR_RESOLVE: SlangLib = SlangLib {
    name: "ssr_resolve.slang",
    file: "ssr.slang",
    entries: &["ssr_resolve_fragment"],
    defines: PROBE_DEFINES,
};
pub(super) static SSGI_GATHER: SlangLib = SlangLib {
    name: "ssgi_gather.slang",
    file: "ssgi.slang",
    entries: &["ssgi_gather_fragment"],
    defines: &[("SSGI_GATHER", "1")],
};
pub(super) static SSGI_COMPOSITE: SlangLib = SlangLib {
    name: "ssgi_composite.slang",
    file: "ssgi.slang",
    entries: &["ssgi_composite_fragment"],
    defines: &[("SSGI_COMPOSITE", "1")],
};
pub(super) static REFLECTION_BLUR: SlangLib = SlangLib {
    name: "reflection_blur.slang",
    file: "reflection.slang",
    entries: &["reflection_blur_fragment"],
    defines: &[("REFLECTION_BLUR", "1")],
};
pub(super) static REFLECTION_COMPOSITE: SlangLib = SlangLib {
    name: "reflection_composite.slang",
    file: "reflection.slang",
    entries: &["reflection_composite_fragment"],
    defines: &[("REFLECTION_COMPOSITE", "1")],
};

// The compute kernels and the fog family. `METAL_BINDINGS` carries the buffer
// index this host writes the params to, where Vulkan and DirectX both take them
// as a push / root constant. The fog fragment always reads the resolved
// single-sample depth here, so it compiles with `USE_MSAA 0`.
pub(super) static FOG_FROXEL: SlangLib = SlangLib {
    name: "fog_froxel.slang",
    file: "fog.slang",
    entries: &["fog_froxel_kernel"],
    defines: &[("FOG_FROXEL", "1")],
};
pub(super) static FOG_FRAG: SlangLib = SlangLib {
    name: "fog_frag.slang",
    file: "fog.slang",
    entries: &["fog_fragment"],
    defines: &[("USE_MSAA", "0")],
};
pub(super) static AUTO_EXPOSURE_BUILD: SlangLib = SlangLib {
    name: "auto_exposure_build.slang",
    file: "auto_exposure.slang",
    entries: &["histogram_build"],
    defines: &[("AE_BUILD", "1"), ("METAL_BINDINGS", "1")],
};
pub(super) static AUTO_EXPOSURE_AVERAGE: SlangLib = SlangLib {
    name: "auto_exposure_average.slang",
    file: "auto_exposure.slang",
    entries: &["histogram_average"],
    defines: &[("AE_AVERAGE", "1"), ("METAL_BINDINGS", "1")],
};
pub(super) static PARTICLE_SIMULATE: SlangLib = SlangLib {
    name: "particle_simulate.slang",
    file: "particle_simulate.slang",
    entries: &["particle_simulate"],
    defines: &[("METAL_BINDINGS", "1")],
};

// The remaining raster families: the particle billboard pair, the projected
// decal, world-space lines and the text / sprite overlay. Each has real vertex
// geometry, so unlike the post passes they keep their own vertex entry rather
// than pairing with `fullscreen.slang`. The two depth-reading fragments always
// compile against the resolved single-sample depth here, the way the fog
// fragment does; only Vulkan reads the multisampled original.
pub(super) static PARTICLE_VERT: SlangLib = SlangLib {
    name: "particle_vert.slang",
    file: "particle.slang",
    entries: &["particle_vertex"],
    defines: &[("METAL_BINDINGS", "1")],
};
pub(super) static PARTICLE_FRAG: SlangLib = SlangLib {
    name: "particle_frag.slang",
    file: "particle.slang",
    entries: &["particle_fragment"],
    defines: &[("METAL_BINDINGS", "1")],
};
pub(super) static DECAL_VERT: SlangLib = SlangLib {
    name: "decal_vert.slang",
    file: "decal.slang",
    entries: &["decal_vertex"],
    defines: &[],
};
pub(super) static DECAL_FRAG: SlangLib = SlangLib {
    name: "decal_frag.slang",
    file: "decal.slang",
    entries: &["decal_fragment"],
    defines: &[("USE_MSAA", "0")],
};
pub(super) static LINE_VERT: SlangLib = SlangLib {
    name: "line_vert.slang",
    file: "line.slang",
    entries: &["line_vertex"],
    defines: &[],
};
pub(super) static LINE_FRAG: SlangLib = SlangLib {
    name: "line_frag.slang",
    file: "line.slang",
    entries: &["line_fragment"],
    defines: &[("USE_MSAA", "0")],
};
pub(super) static TEXT_VERT: SlangLib = SlangLib {
    name: "text_vert.slang",
    file: "text.slang",
    entries: &["text_vertex_main"],
    defines: &[("METAL_BINDINGS", "1")],
};
pub(super) static TEXT_FRAG: SlangLib = SlangLib {
    name: "text_frag.slang",
    file: "text.slang",
    entries: &["text_fragment_main"],
    defines: &[("METAL_BINDINGS", "1")],
};

// The ray-traced families. Only ever loaded on a device that supports ray
// tracing: the trace is compiled in, and the hosts build these pipelines only
// once an acceleration structure exists. The textured variants read the
// bindless pool, so they bake its capacity in; the flat ones do not declare it
// at all, which keeps the pool's Metal buffer slot free in a non-bindless world.
const RT_DEFINES: &[(&str, &str)] = &[("METAL_ABI", "1"), ("MAX_PROBES", "8")];
const RT_TEXTURED_DEFINES: &[(&str, &str)] = &[
    ("METAL_ABI", "1"),
    ("RT_TEXTURED", "1"),
    ("POOL_SIZE", "1024"),
    ("MAX_PROBES", "8"),
];
const GLASS_DEFINES: &[(&str, &str)] = &[("METAL_ABI", "1"), ("MAX_PROBES", "8")];
const GLASS_RT_DEFINES: &[(&str, &str)] =
    &[("METAL_ABI", "1"), ("GLASS_RT", "1"), ("MAX_PROBES", "8")];
const GLASS_RT_TEXTURED_DEFINES: &[(&str, &str)] = &[
    ("METAL_ABI", "1"),
    ("GLASS_RT", "1"),
    ("RT_TEXTURED", "1"),
    ("POOL_SIZE", "1024"),
    ("MAX_PROBES", "8"),
];
// The see-through glass mesh family is ray-traced only, so there is no non-RT
// gate to set: the trace is what makes the mesh see-through rather than the
// opaque reflective glass the main pass draws when RT is off.
const GLASS_MESH_DEFINES: &[(&str, &str)] = &[("METAL_ABI", "1"), ("MAX_PROBES", "8")];
const GLASS_MESH_TEXTURED_DEFINES: &[(&str, &str)] = &[
    ("METAL_ABI", "1"),
    ("RT_TEXTURED", "1"),
    ("POOL_SIZE", "1024"),
    ("MAX_PROBES", "8"),
];
const WATER_DEFINES: &[(&str, &str)] = &[("METAL_ABI", "1"), ("MAX_PROBES", "8")];
const WATER_RT_DEFINES: &[(&str, &str)] =
    &[("METAL_ABI", "1"), ("WATER_RT", "1"), ("MAX_PROBES", "8")];
const WATER_RT_TEXTURED_DEFINES: &[(&str, &str)] = &[
    ("METAL_ABI", "1"),
    ("WATER_RT", "1"),
    ("RT_TEXTURED", "1"),
    ("POOL_SIZE", "1024"),
    ("MAX_PROBES", "8"),
];

pub(super) static RT_REFLECTIONS_FRAG: SlangLib = SlangLib {
    name: "rt_reflections_frag.slang",
    file: "rt_reflections.slang",
    entries: &["rt_reflections_fragment"],
    defines: RT_DEFINES,
};
pub(super) static RT_REFLECTIONS_FRAG_TEXTURED: SlangLib = SlangLib {
    name: "rt_reflections_frag_textured.slang",
    file: "rt_reflections.slang",
    entries: &["rt_reflections_fragment"],
    defines: RT_TEXTURED_DEFINES,
};
pub(super) static GLASS_VERT: SlangLib = SlangLib {
    name: "glass_vert.slang",
    file: "glass.slang",
    entries: &["glass_vertex"],
    defines: GLASS_DEFINES,
};
pub(super) static GLASS_FRAG: SlangLib = SlangLib {
    name: "glass_frag.slang",
    file: "glass.slang",
    entries: &["glass_fragment"],
    defines: GLASS_DEFINES,
};
pub(super) static GLASS_FRAG_RT: SlangLib = SlangLib {
    name: "glass_frag_rt.slang",
    file: "glass.slang",
    entries: &["glass_rt_fragment"],
    defines: GLASS_RT_DEFINES,
};
pub(super) static GLASS_FRAG_RT_TEXTURED: SlangLib = SlangLib {
    name: "glass_frag_rt_textured.slang",
    file: "glass.slang",
    entries: &["glass_rt_fragment"],
    defines: GLASS_RT_TEXTURED_DEFINES,
};

pub(super) static GLASS_MESH_VERT: SlangLib = SlangLib {
    name: "glass_mesh_vert.slang",
    file: "glass_mesh.slang",
    entries: &["glass_mesh_vertex"],
    defines: GLASS_MESH_DEFINES,
};
pub(super) static GLASS_MESH_FRAG_RT: SlangLib = SlangLib {
    name: "glass_mesh_frag_rt.slang",
    file: "glass_mesh.slang",
    entries: &["glass_mesh_rt_fragment"],
    defines: GLASS_MESH_DEFINES,
};
pub(super) static GLASS_MESH_FRAG_RT_TEXTURED: SlangLib = SlangLib {
    name: "glass_mesh_frag_rt_textured.slang",
    file: "glass_mesh.slang",
    entries: &["glass_mesh_rt_fragment"],
    defines: GLASS_MESH_TEXTURED_DEFINES,
};

pub(super) static WATER_VERT: SlangLib = SlangLib {
    name: "water_vert.slang",
    file: "water.slang",
    entries: &["water_vertex"],
    defines: WATER_DEFINES,
};
pub(super) static WATER_FRAG: SlangLib = SlangLib {
    name: "water_frag.slang",
    file: "water.slang",
    entries: &["water_fragment"],
    defines: WATER_DEFINES,
};
pub(super) static WATER_FRAG_RT: SlangLib = SlangLib {
    name: "water_frag_rt.slang",
    file: "water.slang",
    entries: &["water_rt_fragment"],
    defines: WATER_RT_DEFINES,
};
pub(super) static WATER_FRAG_RT_TEXTURED: SlangLib = SlangLib {
    name: "water_frag_rt_textured.slang",
    file: "water.slang",
    entries: &["water_rt_fragment"],
    defines: WATER_RT_TEXTURED_DEFINES,
};

// Every registered variant, for the coverage test in `metallib.rs`.
#[cfg(test)]
pub(super) static ALL: &[&SlangLib] = &[
    &MAIN_BINDLESS_VERT,
    &MAIN_BINDLESS_FRAG,
    &LIGHT_CULL,
    &RT_SKIN,
    &HIZ_INIT_MSAA,
    &HIZ_DOWNSAMPLE,
    &PROBE_MIP0,
    &PROBE_DOWNSAMPLE,
    &PROBE_GGX,
    &GBUFFER_PREPASS_VERT,
    &GBUFFER_PREPASS_VERT_INSTANCED,
    &GBUFFER_PREPASS_VERT_SKINNED,
    &GBUFFER_PREPASS_VERT_BINDLESS,
    &GBUFFER_PREPASS_FRAG,
    &GBUFFER_PREPASS_FRAG_BINDLESS,
    &SHADOW_VERT,
    &SHADOW_VERT_SKINNED,
    &SHADOW_VERT_BINDLESS,
    &FULLSCREEN_VERT,
    &TAA_FRAG,
    &BLOOM_PREFILTER,
    &BLOOM_DOWNSAMPLE,
    &BLOOM_UPSAMPLE,
    &COMPOSITE_FRAG,
    &SSAO_KERNEL,
    &SSAO_BLUR,
    &SSR_RESOLVE,
    &SSGI_GATHER,
    &SSGI_COMPOSITE,
    &REFLECTION_BLUR,
    &REFLECTION_COMPOSITE,
    &FOG_FROXEL,
    &FOG_FRAG,
    &AUTO_EXPOSURE_BUILD,
    &AUTO_EXPOSURE_AVERAGE,
    &PARTICLE_SIMULATE,
    &PARTICLE_VERT,
    &PARTICLE_FRAG,
    &DECAL_VERT,
    &DECAL_FRAG,
    &LINE_VERT,
    &LINE_FRAG,
    &TEXT_VERT,
    &TEXT_FRAG,
    &RT_REFLECTIONS_FRAG,
    &RT_REFLECTIONS_FRAG_TEXTURED,
    &GLASS_VERT,
    &GLASS_FRAG,
    &GLASS_FRAG_RT,
    &GLASS_FRAG_RT_TEXTURED,
    &GLASS_MESH_VERT,
    &GLASS_MESH_FRAG_RT,
    &GLASS_MESH_FRAG_RT_TEXTURED,
    &WATER_VERT,
    &WATER_FRAG,
    &WATER_FRAG_RT,
    &WATER_FRAG_RT_TEXTURED,
];

impl SlangLib {
    // The exact source text this variant compiles, assembled the way every
    // backend assembles it.
    fn source(&self, hot_reload: bool) -> String {
        crate::slang_source::assemble(hot_reload, self.file, self.defines)
    }

    // Produce this variant's MTLLibrary. Fast path: the metallib the build
    // script precompiled and embedded. Fallback (hot-reload, or a build host
    // without slangc): compile at runtime through the shader cache.
    pub(super) fn library(
        &self,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        hot_reload: bool,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
        if !hot_reload && let Some(bytes) = super::metallib::embedded_metallib(self.name) {
            return load_library(device, bytes)
                .map_err(|e| format!("{}: failed to load precompiled metallib: {e}", self.name));
        }
        let source = self.source(hot_reload);
        let entry = self.entries.join("+");
        let key = crate::shader_cache::Key {
            compiler: "slang",
            source: &source,
            entry: &entry,
            target: "metallib",
            options: 0,
        };
        let bytes = crate::shader_cache::cached(&key, self.name, || {
            let job = slang::SlangJob {
                source: &source,
                file_name: self.name,
                entries: self.entries,
                target: slang::SlangTarget::Metallib,
            };
            let work = crate::compiler_work::dir()?;
            slang::compile(&job, work.path())
        })?;
        load_library(device, &bytes)
            .map_err(|e| format!("{}: metallib load failed: {e}", self.name))
    }
}

// A single-entry variant's function, ready for a pipeline descriptor. Every
// variant compiles on its own, so a two-stage pipeline takes its vertex and its
// fragment from separate libraries and the two pair by semantic.
pub(super) fn entry_function(
    device: &ProtocolObject<dyn MTLDevice>,
    lib: &SlangLib,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLFunction>>, String> {
    let library = lib.library(device, hot_reload)?;
    let entry = lib.entries[0];
    library
        .newFunctionWithName(&ns_str(entry))
        .ok_or_else(|| format!("{entry} not found in {}", lib.name))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The values the build script baked into the precompiled metallibs must
    // match the capacities the host binds, or the shader indexes past what
    // the argument encoder wrote.
    #[test]
    fn defines_match_the_crate_constants() {
        use super::build_defines::{SLANG_METAL_MAX_PROBES, SLANG_METAL_POOL_SIZE};
        assert_eq!(
            SLANG_METAL_POOL_SIZE,
            super::super::context::BINDLESS_TEXTURE_COUNT
        );
        assert_eq!(
            SLANG_METAL_MAX_PROBES,
            concinnity_core::render::uniforms::MAX_PROBES
        );
        let want_pool = SLANG_METAL_POOL_SIZE.to_string();
        let want_probes = SLANG_METAL_MAX_PROBES.to_string();
        for lib in ALL {
            let value = |key: &str| lib.defines.iter().find(|(k, _)| *k == key).map(|(_, v)| *v);
            if let Some(pool) = value("POOL_SIZE") {
                assert_eq!(pool, want_pool.as_str(), "{}", lib.name);
            }
            if let Some(probes) = value("MAX_PROBES") {
                assert_eq!(probes, want_probes.as_str(), "{}", lib.name);
            }
        }
        // The pair that pins the generated constants themselves.
        for lib in [&MAIN_BINDLESS_VERT, &MAIN_BINDLESS_FRAG] {
            let value = |key: &str| lib.defines.iter().find(|(k, _)| *k == key).map(|(_, v)| *v);
            assert_eq!(value("POOL_SIZE"), Some(want_pool.as_str()));
            assert_eq!(value("MAX_PROBES"), Some(want_probes.as_str()));
        }
    }

    // Every variant assembles non-empty source with its defines up front, in
    // both hot-reload modes (embedded and disk must agree on existence), and
    // with every `{...}` fragment marker spliced -- an unreplaced marker would
    // reach slangc as a syntax error at renderer init rather than here.
    #[test]
    fn variants_assemble_source_with_their_defines() {
        for lib in ALL {
            for hot_reload in [false, true] {
                let src = lib.source(hot_reload);
                assert!(!src.trim().is_empty(), "{}: empty source", lib.name);
                for marker in ["{POST_COMMON}", "{OBJECT_COMMON}", "{PARTICLE_TYPES}"] {
                    assert!(
                        !src.contains(marker),
                        "{}: unspliced fragment marker {marker}",
                        lib.name
                    );
                }
                for (k, v) in lib.defines {
                    assert!(
                        src.starts_with('#') && src.contains(&format!("#define {k} {v}\n")),
                        "{}: missing injected define {k}",
                        lib.name
                    );
                }
            }
        }
    }

    // The reflection-cut constant in every single-source fragment that gates on
    // it must track the canonical Rust value, like the equivalent locks on the
    // MSL sources: the resolve, the composite blur, and the forward pass all
    // decide "does this surface reflect" from it and cannot disagree.
    #[test]
    fn reflection_roughness_cut_matches_canonical() {
        let expected = format!(
            "static const float REFLECTION_ROUGHNESS_CUT = {:?};",
            crate::gfx::ssr::REFLECTION_ROUGHNESS_CUT
        );
        for (name, src) in [
            (
                "main_bindless.slang",
                concinnity_core::render::shaders::MAIN_BINDLESS,
            ),
            ("ssr.slang", concinnity_core::render::shaders::SSR),
            (
                "reflection.slang",
                concinnity_core::render::shaders::REFLECTION,
            ),
            (
                "rt_reflections.slang",
                concinnity_core::render::shaders::RT_REFLECTIONS,
            ),
        ] {
            assert!(
                src.contains(&expected),
                "{name} REFLECTION_ROUGHNESS_CUT drifted from \
                 concinnity_core::gfx::ssr::REFLECTION_ROUGHNESS_CUT"
            );
        }
    }

    // Every entry-point name the renderer looks up must exist in the source it
    // is declared against, so a rename on one side fails a test rather than a
    // pipeline build.
    #[test]
    fn entries_exist_in_their_sources() {
        for lib in ALL {
            for entry in lib.entries {
                let src = concinnity_core::render::shaders::embedded(lib.file)
                    .unwrap_or_else(|| panic!("{}: no embedded {}", lib.name, lib.file));
                assert!(
                    src.contains(&format!(" {entry}(")),
                    "{}: entry {entry} not found in {}",
                    lib.name,
                    lib.file
                );
            }
        }
    }
}

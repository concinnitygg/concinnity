//! Per-frame graph builder. On Metal, every pass that ran inline
//! through `draw_frame` now dispatches through a single
//! `build_frame_graph()` → `execute_graph()` pair.
//!
//! The frame builder declares conditional passes based on the
//! `FrameGraphInputs` struct (one bool per gated pass). Read / write
//! declarations on each pass let the compile pass derive:
//!
//!   * Execution order (toposort over RAW / WAW / WAR edges, ties broken
//!     by declaration order).
//!   * Per-pass barriers (`pass.barriers_before` per resource state
//!     transition). Metal mostly ignores these (Apple GPUs handle most
//!     hazards implicitly); the Vulkan / DirectX executors emit
//!     `vkCmdPipelineBarrier` / `D3D12_RESOURCE_BARRIER` from them.
//!   * Transient resource lifetimes (`PassRange` per resource), the
//!     aliasing input.
//!
//! Resources split into two origins. `import_texture` = engine-owned: the
//! resource outlives the frame (the cross-frame shadow map, the TAA history
//! `scene_color`, the froxel volume, the cross-frame Hi-Z pyramid) and the
//! backend always owns its GPU object.
//!
//! A resource is declared only where a pass actually writes it. Several engine
//! bindings point two names at one texture depending on configuration --
//! `scene_pre_taa` is `hdr_resolve` without a reflection resolve, `scene_color`
//! is the pre-TAA scene without TAA, `hdr_color` is the resolve target without
//! MSAA -- and declaring the second name anyway would give one GPU object two
//! independent barrier timelines. The builder threads the upstream handle
//! through instead, so one texture is always one resource. `create_texture` =
//! transient: single-frame intermediates (hdr intermediates excepted)
//! the aliasing planner ([`super::alias`]) may pack into shared physical memory,
//! since their `[first, last]` lifetimes are disjoint. In practice only
//! `ao_output` and `bloom_top` are independently poolable today; the other
//! `create_texture` intermediates fold into the long-lived gbuffer MRT
//! (`velocity`, `ssr_gbuffer`) or are themselves long-lived (`gbuffer`), so a
//! backend pool leaves them backend-owned. The planner sizes each transient
//! from its desc; the origin marks aliasing candidacy and has no effect on pass
//! order or barriers.

use crate::gfx::render_types::NUM_SHADOW_CASCADES;
use crate::render::post::ssgi;
use concinnity_core::math::SaturatingFrom;

use super::types::BufferHandle;
use super::{
    BufferDesc, BufferUsage, CompiledGraph, GraphBuilder, GraphError, PassId, PassKind,
    PixelFormat, ReadStages, TextureDesc, TextureHandle, TextureSize, TextureUsage,
    full_mip_levels,
};

/// Per-frame inputs that gate conditional passes. Built by `draw_frame`
/// from the live `MtlContext` state and consumed by `build_frame_graph`
/// so the conditional-inclusion decisions made here match what the
/// executor will dispatch.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one enable per render graph pass"
)]
pub struct FrameGraphInputs {
    /// `true` when a `ShadowStage` is in the world (i.e. the backend's
    /// shadow pipeline + cascade uniforms are live). Skips the Shadow
    /// pass when false rather than relying on the encoder's early
    /// return, so the compiled graph reflects what actually runs.
    pub shadow_enabled: bool,
    /// Per-cascade slice dimensions of the shadow-map array texture.
    /// Carried so the imported `shadow_map` resource carries its real
    /// shape for aliasing; ignored by the executor.
    pub shadow_map_size: u32,
    /// Pixel dimensions of the HDR off-screen targets the Main pass
    /// writes (and the post stack consumes). Carried for aliasing;
    /// ignored by the executor.
    pub hdr_width: u32,
    /// HDR target height in pixels.
    pub hdr_height: u32,
    /// MSAA sample count of the HDR color + depth attachments, typically
    /// 4. The resolve target is single-sample regardless.
    pub hdr_sample_count: u32,
    /// `true` when GPU-driven cull is going to run this frame, i.e. the
    /// bindless static path is configured AND there is geometry to cull
    /// AND the per-frame `object_buffer` / `draw_args` buffers built. The
    /// graph adds the Cull compute pass and the Main read-edge from
    /// `draw_args` only when this is on; otherwise Main draws no static
    /// geometry and the graph has no dependency on the cull output.
    pub bindless_cull_enabled: bool,
    /// `true` when the auto-exposure compute pipelines are built (i.e.
    /// the world declared `Exposure.auto`). The graph
    /// appends an `AutoExposure` compute pass that reads the Main pass's
    /// `hdr_resolve_v1` (pre-decoration) and writes the histogram +
    /// readback buffer. The compile pass's WAR step pins `AutoExposure`
    /// before the first `hdr_resolve` post-Main writer (Decals or Fog or
    /// `ParticlesDraw`) so `AutoExposure` samples the un-decorated scene.
    pub auto_exposure_enabled: bool,
    /// `true` when `Bloom.intensity > 0.0`. The graph
    /// adds a `Bloom` pass that thresholds / downsamples / upsamples the
    /// post-TAA scene into the bloom mip chain; Composite reads the
    /// bloom output so the toposort orders Bloom before Composite.
    pub bloom_enabled: bool,
    /// `true` when TAA is on (the velocity pre-pass only runs as part of
    /// the TAA stack). The graph adds a `Velocity` render pass that
    /// writes the per-pixel motion-vector buffer `TaaResolve` consumes;
    /// `TaaResolve` declares the read so Velocity → `TaaResolve` is explicit.
    pub velocity_enabled: bool,
    /// `true` when TAA is on. The graph adds a `TaaResolve` render pass
    /// that reads the pre-TAA scene (SSR resolve output or `hdr_resolve`)
    /// and writes the imported `scene_color` Bloom + Composite consume.
    pub taa_enabled: bool,
    /// `true` when SSR is on. The graph adds an `SsrResolve` render pass
    /// that reads the post-decoration `hdr_resolve` and writes the
    /// imported `scene_pre_taa` texture, which only exists when this or
    /// `rt_reflections_enabled` is set. When TAA is also on, `TaaResolve`
    /// reads the post-SsrResolve version; with TAA off, Bloom +
    /// Composite read that version directly.
    pub ssr_enabled: bool,
    /// `true` when the particle system is going to run this frame:
    /// `particle_pipelines` built AND at least one live emitter. The graph
    /// adds two nodes: a `ParticlesSim` compute pass that integrates every
    /// live emitter's persistent particle pool in place, and a
    /// `ParticlesDraw` render pass that reads those pools in its vertex
    /// stage and blend-writes `hdr_resolve`. Sim depends on nothing else in
    /// the frame, so the schedule is free to run it on the async queue
    /// alongside the raster front.
    pub particles_enabled: bool,
    /// `true` when a live emitter draws after the see-through surfaces. The
    /// graph adds a `ParticlesLateDraw` render pass after `Transparent` that
    /// draws those emitters from the same pools over the pre-TAA scene.
    pub late_particles_enabled: bool,
    /// `true` when a `VolumetricFog` is in the world. The graph adds a
    /// `Fog` render pass between Decals and `ParticlesDraw` on the
    /// `hdr_resolve` RMW chain.
    pub fog_enabled: bool,
    /// `true` when at least one `Decal` is in the world AND the decal
    /// pipeline is built. The graph adds a `Decals` render pass at the
    /// head of the `hdr_resolve` post-Main RMW chain.
    pub decals_enabled: bool,
    /// `true` when SSR needs the geometry pre-pass; matches
    /// `self.ssr_settings.is_some()`. One of the three consumer gates that make
    /// the graph emit `GBufferPrepass`; SSR then marches against its
    /// normal+depth output, which `SsaoBlur` reads as well.
    pub ssr_prepass_enabled: bool,
    /// `true` when SSAO should run; matches
    /// `self.ssao_settings.is_some()`. The graph adds an `SsaoBlur`
    /// render pass that dispatches the bundled `encode_ssao` (which
    /// internally encodes `SsaoDepth` + `SsaoKernel` + `SsaoBlur`). `SsaoBlur`
    /// writes `ao_output`; Main reads it. `SsaoDepth` + `SsaoKernel`
    /// stay as timing-only `PassIds`.
    pub ssao_enabled: bool,
    /// `true` when temporal upscaling is on (e.g. `MetalFX` on Metal). The
    /// graph adds an `Upscale` pass between the post-SSR scene and the
    /// Bloom + Composite stack that reads `scene_pre_taa` + `velocity`
    /// and writes the imported `scene_color` at output resolution. When
    /// this is on, `TaaResolve` is *not* added: the upscaler does
    /// temporal accumulation itself, so adding TAA on top would
    /// double-temporal. `velocity_enabled` should still be on (the
    /// scaler consumes motion vectors); the engine layer is responsible
    /// for keeping the two flags in sync.
    pub upscale_enabled: bool,
    /// `true` when at least one transparent / translucent draw is in the
    /// world (water, glass, ...). The graph adds a `Transparent` render
    /// pass after `SsrResolve` and before `TaaResolve` / `Upscale` that
    /// reads the latest scene-pre-taa color + main depth and
    /// alpha-blends translucent geometry back-to-front into the same
    /// target. The pass aggregates N draws, each owns its own
    /// pipeline + descriptor set, the executor receives the sorted list
    /// at encode time.
    pub transparent_enabled: bool,
    /// `true` when the world has flat reflectors holding mirror slots and the
    /// transparent pass samples their mirrors this frame (water always does; glass
    /// only while the per-pixel trace is not live). The graph adds a
    /// `PlanarReflection` render pass ahead of `Transparent` that writes the
    /// mirror targets `Transparent` reads. It is a structural gate: whether any
    /// reflector is on screen is decided when the pass encodes, so looking toward
    /// or away from a reflector never changes the graph. Only meaningful alongside
    /// `transparent_enabled`, the mirrors' one consumer.
    pub planar_reflection_enabled: bool,
    /// `true` when the world's prop reflection probe captures this frame: the
    /// graph adds a `PropProbe` render pass after the shadow maps and the cull
    /// and ahead of `Main`, whose probe set holds the cube it writes.
    pub prop_probe_enabled: bool,
    /// `true` when a system submitted world-space lines this frame AND the
    /// backend's line pipeline is live. The graph adds a `Lines` render pass at
    /// the tail of the `hdr_resolve` RMW chain: it blend-writes the scene color
    /// and samples the resolved scene depth so a line behind geometry is
    /// occluded by it. A frame with no lines omits the node entirely.
    pub lines_enabled: bool,
    /// `true` when at least one visible `SdfVolume` is in the world AND
    /// the backend's raymarch pipeline is live. The graph adds a
    /// `Raymarch` render pass between `AutoExposure` and `Decals` on the
    /// `hdr_resolve` RMW chain: it reads the head of the chain (so
    /// `AutoExposure` samples the pre-raymarch scene) and writes the next
    /// version that Decals then bumps further. The pass also RMWs the
    /// main depth attachment so subsequent passes see raymarched
    /// surfaces' depth, and that read-modify-write is declared, which is
    /// what makes the post-Raymarch depth version the one every later
    /// decoration pass samples.
    pub raymarch_enabled: bool,
    /// `true` when two-pass Hi-Z occlusion culling is requested
    /// (`Culling.two_pass_occlusion`) AND the bindless GPU-cull
    /// path is active this frame. Only meaningful alongside
    /// `bindless_cull_enabled`; the builder ANDs the two so a world that
    /// asks for two-pass without a bindless shader simply gets the
    /// single-pass path. When on, the graph inserts `HizBuild` → `Cull2`
    /// → `Main2` between `Main` and the post-decoration chain: `HizBuild`
    /// rebuilds the Hi-Z pyramid from phase-1 depth, `Cull2` re-tests the
    /// objects phase-1 cull marked occluded, and `Main2` redraws the
    /// disoccluded survivors. `Main2`'s `hdr_resolve` write becomes the head
    /// of the post chain so `AutoExposure` / Decals / Fog / SSR see the
    /// combined two-pass result.
    pub two_pass_occlusion_enabled: bool,
    /// `true` when screen-space global illumination is on
    /// (`IndirectLighting.technique == "ssgi"`); matches
    /// `self.ssgi_settings.is_some()`. The graph inserts an `Ssgi` render
    /// pass reading the lit scene right after `Raymarch` (its bounce-radiance
    /// source) and before `Decals`. SSGI reuses the SSR pre-pass G-buffer for
    /// normals + depth, so `ssr_prepass_enabled` is forced on whenever this is
    /// set.
    pub ssgi_enabled: bool,
    /// `true` while the main pass samples SSGI's resolved visibility and
    /// bounce, which it does whenever SSGI is built: the graph inserts an
    /// `SsgiResolve` render pass ahead of `Main` that writes them. Stays on
    /// when a show flag or a live change turns `ssgi_enabled` off, so the
    /// bound target keeps carrying the occlusion; without the trace its
    /// history lapses and the bounce goes to zero.
    pub ssgi_resolve_enabled: bool,
    /// `true` when hardware ray-traced reflections are live (RT requested + GPU
    /// supports it + the scene acceleration structure built); matches
    /// `self.rt_accel.is_some()`. The graph adds an `RtReflections` render pass
    /// in the *same slot* as `SsrResolve` (reads the post-decoration
    /// `hdr_resolve`, writes `scene_pre_taa`). RT *takes precedence* over SSR: a
    /// world may enable both, and where this is set the builder inserts
    /// `RtReflections` and omits `SsrResolve`, so at most one of them is in the
    /// graph. Like SSGI it reuses the SSR depth + normal + roughness pre-pass,
    /// so `ssr_prepass_enabled` is forced on whenever this is set.
    pub rt_reflections_enabled: bool,
    /// `true` when the backend built the G-buffer targets. One
    /// `GBufferPrepass` node writes view-space normal+depth, roughness, and
    /// motion in one traversal and every screen-space consumer reads that one
    /// output, so the builder emits it when this is set and any of
    /// `ssr_prepass_enabled || ssao_enabled || velocity_enabled` is on. A
    /// backend sets it from the same condition it builds those targets under,
    /// which is why the two gates are separate: this one says the targets
    /// exist, the consumers say whether anything reads them.
    pub gbuffer_prepass_enabled: bool,
    /// `true` when an opaque full-screen menu backdrop covers the scene, so
    /// nothing the world passes produce is visible. The builder masks every
    /// gated world pass off and collapses the graph to `Main -> Composite`
    /// (Composite still presents the menu overlay). The backend pairs this with
    /// an empty visible set so the surviving Main pass is a bare clear; the
    /// opaque overlay then covers it.
    pub world_hidden: bool,
    /// `true` when the scene has local lights or reflection probes to cluster.
    /// The graph adds a `LightCull` compute pass before Main that bins both into
    /// per-cluster lists, which Main, `SsrResolve`, `RtReflections` and Transparent
    /// read (RAW edges). While false, every reader iterates the whole light and
    /// probe set.
    pub clustering_enabled: bool,
    /// `true` when the composite samples the SSAO output directly (the
    /// occlusion view mode). Declares a Composite read of `ao_output`, so the
    /// pool-aliased transient stays live to the end of the frame instead of
    /// dying after Main. No effect while `ssao_enabled` is false.
    pub composite_reads_ao: bool,
    /// `true` when the composite samples the prepass's motion directly (the
    /// motion view mode). Declares a Composite read of `gbuffer_velocity`, so
    /// the pooled target stays live to the end of the frame. No effect while
    /// the prepass does not run.
    pub composite_reads_motion: bool,
    /// `true` when the composite samples the reactive mask directly (the
    /// reactive view mode). Makes the composite one of the mask's readers, so
    /// a frame with a writer declares it even with TAA and the upscaler off.
    pub composite_reads_reactive: bool,
    /// Number of spot shadow map slices to render, i.e. how many spot lights cast
    /// shadows. Zero skips the `SpotShadow` pass and its imported array entirely.
    pub shadowed_spot_count: u32,
    /// Per-slice edge of the spot shadow map array, so the imported resource
    /// carries its real dimensions.
    pub spot_shadow_slice_size: u32,
    /// `true` when the GPU-cull path built a Hi-Z pyramid, so the frame ends by
    /// reducing its final depth into that pyramid for the next frame's phase-1
    /// cull. The graph adds a terminal `HizFinal` compute pass reading the last
    /// depth version and writing the pyramid, plus a `Cull` read of the pyramid
    /// the previous frame left there, which is what orders this frame's cull
    /// ahead of the rebuild that overwrites it.
    pub hiz_build_enabled: bool,
    /// `true` when the backend built the reactive mask: the single-channel
    /// render-resolution target the particle and transparent passes write
    /// alongside the scene, where 1 tells a temporal pass to trust the current
    /// frame over its history. A target-exists gate like
    /// `gbuffer_prepass_enabled`; the builder declares the mask only for a
    /// frame where a writer runs and TAA or the upscaler reads it (see
    /// [`FrameGraphInputs::reactive_mask_live`]).
    pub reactive_mask_enabled: bool,
    /// `true` when the upscaler takes the reactive mask (see
    /// [`crate::render::temporal::reactive_mask::ReactiveReader::reads`]). No effect while
    /// `upscale_enabled` is false.
    pub upscale_reads_reactive: bool,
    /// `true` when the backend built the disocclusion mark. While an upscaler
    /// that takes the reactive mask runs, the graph adds a `Disocclusion` pass
    /// after `GBufferPrepass` that writes the mask first, marking what this
    /// frame uncovers (see [`FrameGraphInputs::disocclusion_live`]).
    pub disocclusion_enabled: bool,
    /// `true` when the world grows a grass field and the backend built its
    /// pipelines. The graph adds a `Grass` compute pass that writes the
    /// visible-blade buffer and its indirect draw arguments, which
    /// `GBufferPrepass` and `Main` read to draw the blades at the tail of their
    /// opaque geometry. It also adds the `GrassBend` compute pass ahead of it,
    /// which relaxes and stamps the bend field the kernel reads.
    pub grass_enabled: bool,
    /// `true` when the grass also casts into the nearest shadow cascade this
    /// frame: the backend built the grass shadow draw and that cascade
    /// re-renders. With `grass_enabled` and `shadow_enabled` the graph adds a
    /// `GrassShadow` compute pass whose blades and draw arguments `Shadow`
    /// reads.
    pub grass_shadow_enabled: bool,
    /// `true` when the world declares an `Atmosphere` and the backend built
    /// its tables. The graph adds an `Atmosphere` compute pass ahead of `Main`
    /// (which draws the sky from its tables and lights with its irradiance)
    /// and an `AerialPerspective` render pass on the `hdr_resolve` chain ahead
    /// of `Fog`.
    pub atmosphere_enabled: bool,
    /// The atmosphere has cloud layers: the march and the composite run after
    /// its haze.
    pub clouds_enabled: bool,
    /// `true` when a water surface takes its sea from a simulated ocean and
    /// the backend built the simulation. With `transparent_enabled` the graph
    /// adds a `WaterSim` compute pass whose cascade arrays `Transparent` reads
    /// in its vertex and fragment stages.
    pub water_sim_enabled: bool,
}

impl FrameGraphInputs {
    /// Whether this frame writes and reads the reactive mask: the backend
    /// built it, the particle or transparent pass runs to write it, and TAA,
    /// an upscaler that takes it, or the reactive view reads it. Without a reader the writers leave it
    /// unstored; without a writer the readers are told there is none. A hidden
    /// world runs neither.
    pub fn reactive_mask_live(&self) -> bool {
        self.reactive_mask_enabled
            && !self.world_hidden
            && (self.particles_enabled || self.transparent_enabled || self.disocclusion_live())
            && (self.temporal_reads_reactive() || self.composite_reads_reactive)
    }

    /// Whether this frame's `Disocclusion` pass runs: the backend built it, the
    /// G-buffer pre-pass runs, and the upscaler reads the reactive mask it
    /// writes.
    pub fn disocclusion_live(&self) -> bool {
        self.disocclusion_enabled
            && self.reactive_mask_enabled
            && !self.world_hidden
            && self.gbuffer_live()
            && self.upscale_enabled
            && self.upscale_reads_reactive
    }

    // Whether the G-buffer pre-pass runs: its targets exist and a consumer
    // reads them.
    fn gbuffer_live(&self) -> bool {
        self.gbuffer_prepass_enabled
            && (self.ssr_prepass_enabled || self.ssao_enabled || self.velocity_enabled)
    }

    // The upscaler replaces TAA when it runs, so it alone decides whether the
    // temporal pass reads the mask.
    fn temporal_reads_reactive(&self) -> bool {
        if self.upscale_enabled {
            self.upscale_reads_reactive
        } else {
            self.taa_enabled
        }
    }

    /// Every gated pass off, at a representative resolution.
    ///
    /// A neutral base a caller can flip individual flags on, e.g. to plan a
    /// worst-case graph for transient-memory allocation (where the allocation
    /// must cover every per-frame graph, not just the current frame's active
    /// passes), or to build a representative graph for a backend's own tests.
    pub fn all_off() -> Self {
        Self {
            shadow_enabled: false,
            shadow_map_size: 2048,
            hdr_width: 1280,
            hdr_height: 720,
            hdr_sample_count: 1,
            bindless_cull_enabled: false,
            auto_exposure_enabled: false,
            bloom_enabled: false,
            velocity_enabled: false,
            taa_enabled: false,
            ssr_enabled: false,
            particles_enabled: false,
            late_particles_enabled: false,
            fog_enabled: false,
            decals_enabled: false,
            ssr_prepass_enabled: false,
            ssao_enabled: false,
            upscale_enabled: false,
            transparent_enabled: false,
            planar_reflection_enabled: false,
            prop_probe_enabled: false,
            lines_enabled: false,
            raymarch_enabled: false,
            two_pass_occlusion_enabled: false,
            ssgi_enabled: false,
            ssgi_resolve_enabled: false,
            rt_reflections_enabled: false,
            gbuffer_prepass_enabled: false,
            world_hidden: false,
            clustering_enabled: false,
            composite_reads_ao: false,
            composite_reads_motion: false,
            composite_reads_reactive: false,
            shadowed_spot_count: 0,
            spot_shadow_slice_size: 512,
            hiz_build_enabled: false,
            reactive_mask_enabled: false,
            upscale_reads_reactive: false,
            disocclusion_enabled: false,
            grass_enabled: false,
            grass_shadow_enabled: false,
            atmosphere_enabled: false,
            clouds_enabled: false,
            water_sim_enabled: false,
        }
    }
}

// Every gated pass flag, paired with the setter that turns it on. The two
// exhaustive sweeps ([`super::validate`] and [`super::transient`]) are only
// as wide as this table, so a new gated pass belongs here as well as in
// `all_off`.
#[cfg(test)]
pub(crate) type FlagSetter = fn(&mut FrameGraphInputs);
#[cfg(test)]
pub(crate) const GATED_FLAGS: &[(&str, FlagSetter)] = &[
    ("shadow", |i| i.shadow_enabled = true),
    ("bindless_cull", |i| i.bindless_cull_enabled = true),
    ("auto_exposure", |i| i.auto_exposure_enabled = true),
    ("bloom", |i| i.bloom_enabled = true),
    ("velocity", |i| i.velocity_enabled = true),
    ("taa", |i| i.taa_enabled = true),
    ("ssr", |i| i.ssr_enabled = true),
    ("particles", |i| i.particles_enabled = true),
    ("late_particles", |i| {
        i.particles_enabled = true;
        i.late_particles_enabled = true;
    }),
    ("fog", |i| i.fog_enabled = true),
    ("decals", |i| i.decals_enabled = true),
    ("ssr_prepass", |i| i.ssr_prepass_enabled = true),
    ("ssao", |i| i.ssao_enabled = true),
    ("upscale", |i| i.upscale_enabled = true),
    ("transparent", |i| i.transparent_enabled = true),
    ("planar_reflection", |i| {
        i.transparent_enabled = true;
        i.planar_reflection_enabled = true;
    }),
    ("prop_probe", |i| {
        i.bindless_cull_enabled = true;
        i.prop_probe_enabled = true;
    }),
    ("lines", |i| i.lines_enabled = true),
    ("raymarch", |i| i.raymarch_enabled = true),
    ("two_pass_occlusion", |i| {
        i.two_pass_occlusion_enabled = true;
    }),
    ("ssgi", |i| i.ssgi_enabled = true),
    ("ssgi_resolve", |i| i.ssgi_resolve_enabled = true),
    ("rt_reflections", |i| i.rt_reflections_enabled = true),
    ("gbuffer_prepass", |i| i.gbuffer_prepass_enabled = true),
    ("world_hidden", |i| i.world_hidden = true),
    ("clustered_lighting", |i| i.clustering_enabled = true),
    ("composite_reads_ao", |i| i.composite_reads_ao = true),
    ("composite_reads_motion", |i| {
        i.composite_reads_motion = true;
    }),
    ("composite_reads_reactive", |i| {
        i.composite_reads_reactive = true;
    }),
    ("shadowed_spots", |i| i.shadowed_spot_count = 2),
    ("hiz_build", |i| i.hiz_build_enabled = true),
    ("reactive_mask", |i| i.reactive_mask_enabled = true),
    ("upscale_reads_reactive", |i| {
        i.upscale_reads_reactive = true;
    }),
    ("disocclusion", |i| {
        i.gbuffer_prepass_enabled = true;
        i.velocity_enabled = true;
        i.reactive_mask_enabled = true;
        i.upscale_enabled = true;
        i.upscale_reads_reactive = true;
        i.disocclusion_enabled = true;
    }),
    ("grass", |i| i.grass_enabled = true),
    ("atmosphere", |i| i.atmosphere_enabled = true),
    ("clouds", |i| {
        i.atmosphere_enabled = true;
        i.clouds_enabled = true;
    }),
    ("grass_shadow", |i| {
        i.grass_enabled = true;
        i.shadow_enabled = true;
        i.grass_shadow_enabled = true;
    }),
    ("water_sim", |i| {
        i.transparent_enabled = true;
        i.water_sim_enabled = true;
    }),
];

// Build the full per-frame render graph. Conditional passes are
// included based on the `inputs` flags. The compile pass derives
// execution order, per-pass barriers, and resource lifetimes via
// RAW + WAW + WAR edges over the version-chained read / write
// declarations.
//
// Order (with all flags on):
//
// ```text
// Cull → GBufferPrepass → SsaoBlur → Shadow → ParticlesSim → Main
//   → AutoExposure → Raymarch → Decals → Fog → ParticlesDraw → SsrResolve
//   → Transparent → TaaResolve → Bloom → HizFinal → Composite
// ```
//
// Main depth has a shorter chain over the same spine: Main writes it,
// Main2 and Raymarch bump it, and Decals / Fog / Lines / Transparent /
// HizFinal all sample the last version. HizFinal is the frame's terminal
// depth consumer, which is what keeps the depth live to the end of the
// graph rather than only to the last decoration.
//
// The hdr_resolve version chain (Main writes v1, AutoExposure reads
// v1 (WAR-pinned before subsequent writers), Decals → v2, Fog → v3,
// ParticlesDraw → v4, SsrResolve reads v4) is the spine that
// orders the bulk of the post stack. scene_pre_taa / scene_color /
// bloom_top each have their own short version chains that branch off
// the spine where a pass writes them. Transparent extends whichever
// chain carries the pre-TAA scene -- scene_pre_taa when a reflection
// resolve produced it, hdr_resolve itself otherwise -- so TaaResolve /
// Upscale pick up translucent geometry as part of temporal accumulation.
//
// When `two_pass_occlusion_enabled` is on the spine gains a phase-2
// prefix: `Cull → Main → HizBuild → Cull2 → Main2 → AutoExposure →
// …`. `Main` writes hdr_resolve v1 / hdr_depth v1; `HizBuild` reads
// the depth and writes the Hi-Z pyramid; `Cull2` reads the pyramid +
// the phase-1 status buffer and writes `draw_args2`; `Main2` RMWs
// hdr_color / hdr_depth / hdr_resolve → v2, and that v2 (not v1)
// becomes the head AutoExposure reads and the RMW chain extends.

// The four attachments the unified G-buffer pre-pass writes in one draw. They
// are separate resources rather than one handle because their shapes differ
// (three color formats and a depth target) and so do their consumers, so one
// handle would give each of them the union of four lifetimes.
#[derive(Copy, Clone)]
struct GBufferHandles {
    normal_depth: TextureHandle,
    roughness: TextureHandle,
    velocity: TextureHandle,
    depth: TextureHandle,
}

// The grass pass's two outputs. The blades are read by the draws' vertex stage
// and the arguments by the indirect draw itself, so each needs its own read
// state and is its own resource.
#[derive(Copy, Clone)]
struct GrassHandles {
    blades: BufferHandle,
    args: BufferHandle,
}

impl GrassHandles {
    // Declare a draw pass's reads of both outputs.
    fn read_into(self, pass: &mut super::builder::PassBuilder<'_>) {
        pass.read_buffer_in_stage(self.blades, ReadStages::VERTEX);
        pass.read_buffer(self.args);
    }
}

/// Compile the frame graph for `inputs`: the pass list above, gated down to the
/// passes this frame actually runs.
///
/// # Errors
///
/// Returns an error if the gated pass list does not compile into a graph.
pub fn build_frame_graph(inputs: &FrameGraphInputs) -> Result<CompiledGraph, GraphError> {
    // When an opaque menu backdrop hides the scene, every world pass is wasted:
    // nothing it produces is visible. Force every gated world pass off so the
    // graph collapses to the minimal `Main -> Composite` (Composite still
    // presents the overlay). Main survives as a bare clear because the backend
    // feeds it an empty visible set this frame; the opaque overlay covers it.
    // Spread from `all_off` rather than from `inputs`, so a gate added to
    // `FrameGraphInputs` later is off here by construction. Only the fields
    // named below survive the mask: the frame's real dimensions, and the three
    // flags that describe graph shape rather than a gated pass.
    let masked = if inputs.world_hidden {
        Some(FrameGraphInputs {
            shadow_map_size: inputs.shadow_map_size,
            hdr_width: inputs.hdr_width,
            hdr_height: inputs.hdr_height,
            hdr_sample_count: inputs.hdr_sample_count,
            gbuffer_prepass_enabled: inputs.gbuffer_prepass_enabled,
            world_hidden: true,
            composite_reads_ao: inputs.composite_reads_ao,
            composite_reads_motion: inputs.composite_reads_motion,
            composite_reads_reactive: inputs.composite_reads_reactive,
            hiz_build_enabled: inputs.hiz_build_enabled,
            reactive_mask_enabled: inputs.reactive_mask_enabled,
            ..FrameGraphInputs::all_off()
        })
    } else {
        None
    };
    let inputs = masked.as_ref().unwrap_or(inputs);

    let mut b = GraphBuilder::new();

    // Engine-owned imports the Main pass writes into. hdr_resolve is the scene
    // spine: also written by Decals / Fog / ParticlesDraw and read by
    // AutoExposure / SsrResolve, so its version chain is the longest.
    //
    // `hdr_color` is the multisample color attachment, and it exists only when
    // the world is multisampled. Without MSAA there is no separate resolve step
    // and the single color target *is* the spine, which every backend already
    // reflects (Vulkan leaves `color_images` empty; DirectX and Metal leave
    // their `resolve` field `None` and bind `color`). Declaring it
    // unconditionally would put two graph resources on one GPU object, and the
    // moment either became graph-driven they would transition it twice from
    // states it was no longer in.
    let hdr = HdrTargets {
        color: (inputs.hdr_sample_count > 1)
            .then(|| b.import_texture("hdr_color", hdr_color_desc(inputs))),
        depth: b.import_texture("hdr_depth", hdr_depth_desc(inputs)),
        resolve: b.import_texture("hdr_resolve", hdr_resolve_desc(inputs)),
    };
    // The reactive mask: written beside the scene by the particle and
    // transparent passes, read by TAA or the upscaler. Backend-owned, since the
    // writers' pipelines draw into it whether or not this frame stores it.
    let mut reactive_mask = inputs
        .reactive_mask_live()
        .then(|| b.import_texture("reactive_mask", reactive_mask_desc(inputs)));

    // Two-pass occlusion only applies when the bindless GPU-cull path is
    // active: Hi-Z occlusion rides that path. ANDing here means a world
    // that requests two-pass without a bindless shader falls back to the
    // single-pass path with no orphaned phase-2 nodes.
    let two_pass = inputs.bindless_cull_enabled && inputs.two_pass_occlusion_enabled;

    // The Hi-Z depth pyramid, imported up front because `Cull` reads the version
    // the *previous* frame left there before `HizFinal` (and, under two-pass,
    // the mid-frame `HizBuild`) overwrites it. Never a transient: its contents
    // cross the frame boundary, so the aliasing planner must not place it.
    let hiz_pyramid = (inputs.hiz_build_enabled || two_pass)
        .then(|| b.import_texture("hiz_pyramid", hiz_pyramid_desc(inputs)));

    let cull = add_cull_and_grass(&mut b, inputs, two_pass, hiz_pyramid);
    let prepass = add_prepasses(&mut b, inputs, &cull, &mut reactive_mask);
    let scene_inputs = add_scene_inputs(&mut b, inputs, &cull);
    let front = FrontEnd {
        cull,
        prepass,
        scene_inputs,
    };

    let main_out = add_main(&mut b, hdr, &front);

    // Two-pass occlusion phase 2: rebuild the Hi-Z pyramid from phase-1
    // depth (HizBuild), re-test the objects phase-1 cull marked occluded
    // (Cull2), and redraw the disoccluded survivors (Main2). Main2 RMWs
    // hdr_color / hdr_depth / hdr_resolve, so its hdr_resolve write becomes
    // the head of the post-decoration chain: AutoExposure and every later
    // RMW pass see the combined phase-1 + phase-2 scene. Without two-pass
    // the head stays at Main's hdr_resolve_v1.
    let (hiz_cur, scene_head) = if let (true, Some(hiz)) = (two_pass, hiz_pyramid) {
        let (hiz_v1, scene) =
            add_occlusion_phase2(&mut b, hiz, front.cull.cull_status, hdr.color, main_out);
        (Some(hiz_v1), scene)
    } else {
        (hiz_pyramid, main_out)
    };

    // AutoExposure (compute) reads the post-main scene (hdr_resolve_head:
    // Main2's output under two-pass, Main's otherwise). The compile pass's
    // WAR step pins it before the first hdr_resolve writer that bumps the
    // next version (Raymarch / Decals / Fog / ParticlesDraw), so
    // AutoExposure samples the un-decorated scene even though the GPU
    // texture object is the same one those passes later blend-write.
    if inputs.auto_exposure_enabled {
        b.add_pass(PassId::AutoExposure, PassKind::Compute)
            .read_texture(scene_head.resolve);
    }

    // The motion half of the pre-pass output: TaaResolve / Upscale read it, and
    // that read edge pins the pre-pass ahead of them in the toposort.
    let velocity_v1 = front.prepass.gbuffer.map(|g| g.velocity);

    let decorated = add_decorations(&mut b, inputs, scene_head, &front, &mut reactive_mask);
    let scene_pre_taa_cur =
        add_reflections_and_transparent(&mut b, inputs, decorated, &front, &mut reactive_mask);
    let scene_pre_taa_cur = add_clouds(
        &mut b,
        inputs,
        SceneVersions {
            resolve: scene_pre_taa_cur,
            depth: decorated.depth,
        },
        &front,
    );
    let scene_pre_taa_cur = add_late_particles(
        &mut b,
        inputs,
        SceneVersions {
            resolve: scene_pre_taa_cur,
            depth: decorated.depth,
        },
        &front,
        &mut reactive_mask,
    );
    let scene_color_cur = add_temporal(
        &mut b,
        inputs,
        scene_pre_taa_cur,
        front.prepass.gbuffer,
        reactive_mask,
    );

    let bloom_top_v1 = if inputs.bloom_enabled {
        let bloom_top = b.create_texture("bloom_top", bloom_top_desc());
        Some(
            b.add_pass(PassId::Bloom, PassKind::Render)
                .read_texture(scene_color_cur)
                .write_texture(bloom_top),
        )
    } else {
        None
    };

    // HizFinal (compute) reduces the frame's final depth into the Hi-Z pyramid
    // the next frame's phase-1 Cull tests against. Declared last of the
    // depth consumers so it reads the version every decoration pass has
    // finished with, which is also what keeps the depth's graph lifetime
    // honest: without this node the depth would look dead after the last
    // decoration while a post-graph pass still read it.
    if let (true, Some(hiz)) = (inputs.hiz_build_enabled, hiz_cur) {
        let mut hizf = b.add_pass(PassId::HizFinal, PassKind::Compute);
        hizf.read_texture(decorated.depth);
        let _ = hizf.write_texture(hiz);
    }

    // Composite (the presenter) reads scene_color + optional bloom_top,
    // and writes the swapchain via `presents()`. The occlusion and motion view
    // modes add an ao_output / velocity read so the pooled transient survives
    // to the present.
    {
        let mut composite = b.add_pass(PassId::Composite, PassKind::Render);
        composite.read_texture(scene_color_cur);
        if let Some(h) = bloom_top_v1 {
            composite.read_texture(h);
        }
        if inputs.composite_reads_ao
            && let Some(h) = front.prepass.ao_output
        {
            composite.read_texture(h);
        }
        if inputs.composite_reads_motion
            && let Some(v) = velocity_v1
        {
            composite.read_texture(v);
        }
        if inputs.composite_reads_reactive
            && let Some(m) = reactive_mask
        {
            composite.read_texture(m);
        }
        composite.presents();
    }

    b.compile()
}

// The engine-owned HDR imports Main writes into.
#[derive(Copy, Clone)]
struct HdrTargets {
    color: Option<TextureHandle>,
    depth: TextureHandle,
    resolve: TextureHandle,
}

// The current versions of the scene color spine (hdr_resolve) and main depth,
// walked forward as each pass read-modify-writes them.
#[derive(Copy, Clone)]
struct SceneVersions {
    resolve: TextureHandle,
    depth: TextureHandle,
}

// What the GPU cull and the grass kernels produce for the raster passes.
#[derive(Copy, Clone)]
struct CullOutputs {
    draw_args: Option<BufferHandle>,
    cull_status: Option<BufferHandle>,
    grass: Option<GrassHandles>,
    grass_shadow: Option<GrassHandles>,
}

// The pre-pass outputs: the G-buffer, ambient occlusion, SSGI's resolve, and
// the reactive mask as the disocclusion mark left it.
#[derive(Copy, Clone)]
struct PrepassOutputs {
    gbuffer: Option<GBufferHandles>,
    ao_output: Option<TextureHandle>,
    ssgi_indirect: Option<TextureHandle>,
    disocclusion: Option<TextureHandle>,
}

impl PrepassOutputs {
    // The normal+depth half of the pre-pass output: SSR marches against it and
    // SSAO reads the same handle.
    fn ssr_gbuffer(&self) -> Option<TextureHandle> {
        self.gbuffer.map(|g| g.normal_depth)
    }
}

// The shadow maps and compute outputs Main and the later scene passes read.
#[derive(Copy, Clone)]
struct SceneInputs {
    shadow: Option<TextureHandle>,
    spot_shadow: Option<TextureHandle>,
    prop_probe: Option<TextureHandle>,
    atmosphere_luts: Option<TextureHandle>,
    cluster_lists: Option<BufferHandle>,
    particle_pool: Option<BufferHandle>,
    water_cascades: Option<TextureHandle>,
}

// Everything declared ahead of Main that the passes from Main on read.
struct FrontEnd {
    cull: CullOutputs,
    prepass: PrepassOutputs,
    scene_inputs: SceneInputs,
}

// Cull (compute) writes the indirect-draw args buffer Main consumes
// through executeCommandsInBuffer. Under two-pass occlusion it also
// writes a per-object status buffer (drawn / hi-z-candidate / culled)
// that `Cull2` reads to decide which phase-1-occluded objects to
// re-test against the rebuilt pyramid.
fn add_cull_and_grass(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    two_pass: bool,
    hiz_pyramid: Option<TextureHandle>,
) -> CullOutputs {
    let (draw_args_v1, cull_status_v1) = if inputs.bindless_cull_enabled {
        // Import both buffers up front: a live `PassBuilder` holds `&mut b`,
        // so the resource declarations have to happen before `add_pass`.
        let draw_args = b.import_buffer("draw_args", draw_args_desc());
        let cull_status = if two_pass {
            Some(b.import_buffer("cull_status", cull_status_desc()))
        } else {
            None
        };
        let mut cull = b.add_pass(PassId::Cull, PassKind::Compute);
        // The previous frame's pyramid is this cull's occlusion test. Declaring
        // the read is what gives the terminal rebuild a WAR edge to wait on.
        if let Some(h) = hiz_pyramid {
            cull.read_texture(h);
        }
        let da = cull.write_buffer(draw_args);
        let cs = cull_status.map(|h| cull.write_buffer(h));
        (Some(da), cs)
    } else {
        (None, None)
    };

    // Grass (compute): places, culls and appends this frame's visible blades,
    // and fills the indirect arguments the pre-pass and Main draw them with.
    // Its tiles are occlusion-tested against the pyramid the previous frame
    // left, a read that, like Cull's, gives the pyramid's rebuilds a WAR edge to
    // wait on. It reads nothing else, so the schedule may overlap it with the
    // raster front. Both buffers are persistent and backend-owned, hence
    // imported.
    //
    // GrassBend (compute) runs first: it relaxes the camera-centered bend field
    // and stamps this frame's characters and bodies into it, reading the half
    // the last frame wrote and writing the other, so the one persistent buffer
    // is both its read and its write. Both kernels that place blades read it.
    let grass_bend_v1 = inputs.grass_enabled.then(|| {
        let field = b.import_buffer("grass_bend", grass_bend_desc());
        let mut bend = b.add_pass(PassId::GrassBend, PassKind::Compute);
        bend.read_buffer(field);
        bend.write_buffer(field)
    });
    let grass_v1 = if let Some(bend) = grass_bend_v1 {
        let blades = b.import_buffer("grass_blades", grass_blades_desc());
        let args = b.import_buffer("grass_args", draw_args_desc());
        let mut grass = b.add_pass(PassId::Grass, PassKind::Compute);
        if let Some(h) = hiz_pyramid {
            grass.read_texture(h);
        }
        grass.read_buffer(bend);
        Some(GrassHandles {
            blades: grass.write_buffer(blades),
            args: grass.write_buffer(args),
        })
    } else {
        None
    };

    // GrassShadow (compute): the same kernel placing the blades the nearest
    // cascade sees, into their own buffers, which the Shadow pass draws.
    let grass_shadow_v1 = match grass_bend_v1 {
        Some(bend) if inputs.shadow_enabled && inputs.grass_shadow_enabled => {
            let blades = b.import_buffer("grass_shadow_blades", grass_blades_desc());
            let args = b.import_buffer("grass_shadow_args", draw_args_desc());
            let mut shadow = b.add_pass(PassId::GrassShadow, PassKind::Compute);
            shadow.read_buffer(bend);
            Some(GrassHandles {
                blades: shadow.write_buffer(blades),
                args: shadow.write_buffer(args),
            })
        }
        _ => None,
    };

    CullOutputs {
        draw_args: draw_args_v1,
        cull_status: cull_status_v1,
        grass: grass_v1,
        grass_shadow: grass_shadow_v1,
    }
}

// The G-buffer pre-pass and the screen-space passes that read only it: SSAO and
// SSGI's resolve.
fn add_prepasses(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    cull: &CullOutputs,
    reactive_mask: &mut Option<TextureHandle>,
) -> PrepassOutputs {
    // Geometry pre-pass: one node writes the view-space normal+depth /
    // roughness / velocity / depth that SSR, SSAO, SSGI, RT, TAA, and the
    // upscaler read. Runs when the backend built the targets and any of those
    // consumers is on.
    let gbuffer_v1 = if inputs.gbuffer_live() {
        let normal_depth =
            b.create_texture("gbuffer_normal_depth", gbuffer_normal_depth_desc(inputs));
        let roughness = b.create_texture("gbuffer_roughness", gbuffer_roughness_desc(inputs));
        let velocity = b.create_texture("gbuffer_velocity", velocity_desc(inputs));
        let depth = b.create_texture("gbuffer_depth", gbuffer_depth_desc(inputs));
        let mut gb = b.add_pass(PassId::GBufferPrepass, PassKind::Render);
        // When the GPU-driven cull path is active the pre-pass reuses the main
        // pass's per-frame indirect command buffer (camera frustum, same cull
        // output), so it must run after Cull. Reading the cull-produced draw_args
        // buffer pins that ordering in the toposort (a no-op when bindless cull is
        // off, where draw_args_v1 is None). Mirrors the Main pass's edge.
        if let Some(h) = cull.draw_args {
            gb.read_buffer(h);
        }
        if let Some(g) = cull.grass {
            g.read_into(&mut gb);
        }
        // One draw writes all four attachments; they are separate resources
        // because their shapes and their consumers differ.
        Some(GBufferHandles {
            normal_depth: gb.write_texture(normal_depth),
            roughness: gb.write_texture(roughness),
            velocity: gb.write_texture(velocity),
            depth: gb.write_texture(depth),
        })
    } else {
        None
    };

    let ssr_gbuffer_v1 = gbuffer_v1.map(|g| g.normal_depth);

    // Disocclusion: the reactive mask's first writer, marking where this
    // frame's G-buffer uncovers a surface last frame's did not show.
    let disocclusion_v1 = match (gbuffer_v1, *reactive_mask) {
        (Some(g), Some(mask)) if inputs.disocclusion_live() => {
            let mut disocclusion = b.add_pass(PassId::Disocclusion, PassKind::Render);
            disocclusion.read_texture(g.normal_depth);
            disocclusion.read_texture(g.velocity);
            Some(disocclusion.write_texture(mask))
        }
        _ => None,
    };
    if disocclusion_v1.is_some() {
        *reactive_mask = disocclusion_v1;
    }

    // SSAO bundle writes ao_output. PassId::SsaoBlur is the single
    // graph node for the entire encode_ssao bundle; SsaoDepth +
    // SsaoKernel keep their per-pass timing slots via inline
    // `pass_timing.attach_render` calls inside encode_ssao but they're
    // not graph nodes (the executor rejects them if mis-added).
    let ao_output_v1 = if inputs.ssao_enabled {
        let ao_output = b.create_texture("ao_output", ao_output_desc(inputs));
        let mut ssao = b.add_pass(PassId::SsaoBlur, PassKind::Render);
        if let Some(h) = ssr_gbuffer_v1 {
            ssao.read_texture(h);
        }
        Some(ssao.write_texture(ao_output))
    } else {
        None
    };

    // SSGI's resolve: last frame's accumulation reprojected onto this frame's
    // G-buffer, folded with the occlusion into the target Main samples for its
    // indirect light. Backend-owned and carried across frames, like the
    // accumulation it reads.
    let ssgi_indirect_v1 = inputs.ssgi_resolve_enabled.then(|| {
        let indirect = b.import_texture(ssgi::INDIRECT_LABEL, ssgi_indirect_desc(inputs));
        let mut resolve = b.add_pass(PassId::SsgiResolve, PassKind::Render);
        if let Some(g) = gbuffer_v1 {
            resolve.read_texture(g.normal_depth);
            resolve.read_texture(g.velocity);
        }
        if let Some(h) = ao_output_v1 {
            resolve.read_texture(h);
        }
        resolve.write_texture(indirect)
    });

    PrepassOutputs {
        gbuffer: gbuffer_v1,
        ao_output: ao_output_v1,
        ssgi_indirect: ssgi_indirect_v1,
        disocclusion: disocclusion_v1,
    }
}

// The shadow maps and the compute passes ahead of Main: the atmosphere tables,
// clustered binning, and the particle and ocean simulations.
fn add_scene_inputs(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    cull: &CullOutputs,
) -> SceneInputs {
    // Shadow optionally precedes Main and produces the shadow_map
    // handle Main samples. When off, Main does not declare a shadow_map
    // read, mirroring the encoder's `enable_shadows` shader path.
    let shadow_v1 = if inputs.shadow_enabled {
        let shadow_map = b.import_texture("shadow_map", shadow_map_desc(inputs.shadow_map_size));
        let mut shadow = b.add_pass(PassId::Shadow, PassKind::Render);
        // The GPU-driven cascade path draws from a per-cascade indirect command
        // buffer the cull pass fills from the same cull output, so it must run
        // after Cull. Reading the cull-produced draw_args buffer pins that
        // ordering in the toposort, the way GBufferPrepass and Main do, and on a
        // backend that submits the two queues separately it is what puts a
        // cross-queue wait between an async cull and this pass. A no-op when
        // bindless cull is off, where the cascades draw per-object.
        if let Some(h) = cull.draw_args {
            shadow.read_buffer(h);
        }
        if let Some(g) = cull.grass_shadow {
            g.read_into(&mut shadow);
        }
        Some(shadow.write_texture(shadow_map))
    } else {
        None
    };

    // Clustered binning (compute): bins the scene's local lights and reflection
    // probes into per-cluster light lists and probe masks. Writes the imported
    // cluster buffer; Main's read below pins LightCull before Main in the
    // toposort. Backend-owned buffer, so the import is a dependency-tracking
    // stub.
    // The atmosphere's tables (compute). Backend-owned and carried across
    // frames, so the import only orders the pass ahead of its readers: Main and
    // the mirrors (the sky and the ambient light) and the haze pass.
    let atmosphere_luts = inputs.atmosphere_enabled.then(|| {
        let luts = b.import_texture("atmosphere_luts", atmosphere_luts_desc());
        b.add_pass(PassId::Atmosphere, PassKind::Compute)
            .write_texture(luts)
    });

    let cluster_lists = if inputs.clustering_enabled {
        let lists = b.import_buffer("cluster_lists", cluster_lists_desc());
        Some(
            b.add_pass(PassId::LightCull, PassKind::Compute)
                .write_buffer(lists),
        )
    } else {
        None
    };

    // Particle simulation (compute): ages, integrates and respawns every live
    // emitter's pool in place. One label covers the whole per-emitter set; each
    // backend's registry resolves it to the live emitters' pool buffers, the way
    // `draw_args` resolves to this frame's slot of a ring.
    //
    // The kernel reads no depth and no scene texture -- its inputs are the pools
    // and the frame's params -- so its only
    // successor is ParticlesDraw and it is concurrent with the whole raster
    // front. Declared here rather than beside the draw so the compiled order puts
    // it ahead of the passes it can overlap: the async queue runs its own passes
    // in order, and a position after a compute pass that waits on Shadow would
    // inherit that wait.
    //
    // The pools are persistent (one per emitter, not one per frame in flight) and
    // each frame integrates them in place, reading what the last frame wrote,
    // which is why they are imported rather than created.
    let particle_pool_v1 = if inputs.particles_enabled {
        let pools = b.import_buffer("particle_pool", particle_pool_desc());
        let mut sim = b.add_pass(PassId::ParticlesSim, PassKind::Compute);
        sim.read_buffer(pools);
        Some(sim.write_buffer(pools))
    } else {
        None
    };

    // The simulated ocean (compute): evolves and transforms every simulated
    // sea into its cascade arrays, which only the water draw reads. Like the
    // particle simulation it depends on nothing else in the frame, so it is
    // declared ahead of the raster front it can overlap. The arrays are
    // backend-owned and rewritten whole each frame.
    let water_cascades = (inputs.water_sim_enabled && inputs.transparent_enabled).then(|| {
        let cascades = b.import_texture("water_cascades", water_cascades_desc());
        b.add_pass(PassId::WaterSim, PassKind::Compute)
            .write_texture(cascades)
    });

    // Spot shadows: one depth-only render per shadowed spot into its slice of
    // the spot shadow array. Like the cascade pass it precedes Main, which
    // samples the array; backend-owned, so the import tracks dependencies only.
    // Its slices draw from the same cull output the cascades do, so it reads
    // draw_args for the same edge after Cull.
    let spot_shadow_v1 = if inputs.shadowed_spot_count > 0 {
        let spot_map = b.import_texture(
            "spot_shadow_map",
            spot_shadow_map_desc(inputs.spot_shadow_slice_size, inputs.shadowed_spot_count),
        );
        let mut spot = b.add_pass(PassId::SpotShadow, PassKind::Render);
        if let Some(h) = cull.draw_args {
            spot.read_buffer(h);
        }
        Some(spot.write_texture(spot_map))
    } else {
        None
    };

    // The prop reflection probe: one cube face of the scene seen from the
    // prop, lit like Main (so after the shadow maps) from the frame's records
    // (so after Cull), and on the cycle's last face the convolution into the
    // probe's cube. The cube is encoder-owned, so the import only orders the
    // pass ahead of Main, the first pass reading the probe set.
    let prop_probe = inputs.prop_probe_enabled.then(|| {
        let cube = b.import_texture("prop_probe", prop_probe_desc());
        let mut capture = b.add_pass(PassId::PropProbe, PassKind::Render);
        if let Some(h) = shadow_v1 {
            capture.read_texture(h);
        }
        if let Some(h) = spot_shadow_v1 {
            capture.read_texture(h);
        }
        if let Some(h) = cull.draw_args {
            capture.read_buffer(h);
        }
        if let Some(h) = atmosphere_luts {
            capture.read_texture(h);
        }
        capture.write_texture(cube)
    });

    SceneInputs {
        shadow: shadow_v1,
        spot_shadow: spot_shadow_v1,
        prop_probe,
        atmosphere_luts,
        cluster_lists,
        particle_pool: particle_pool_v1,
        water_cascades,
    }
}

// Main pass: reads optional shadow_map / spot_shadow_map / draw_args /
// ao_output / cluster lists; writes the three HDR targets. Captures
// hdr_resolve_v1 (head of the hdr_resolve RMW chain, the version
// AutoExposure reads when two-pass is off) and hdr_depth_v1 (the depth
// HizBuild reduces under two-pass).
fn add_main(b: &mut GraphBuilder, hdr: HdrTargets, front: &FrontEnd) -> SceneVersions {
    let mut main = b.add_pass(PassId::Main, PassKind::Render);
    if let Some(h) = front.scene_inputs.shadow {
        main.read_texture(h);
    }
    if let Some(h) = front.scene_inputs.prop_probe {
        main.read_texture(h);
    }
    if let Some(h) = front.scene_inputs.spot_shadow {
        main.read_texture(h);
    }
    if let Some(h) = front.cull.draw_args {
        main.read_buffer(h);
    }
    if let Some(h) = front.scene_inputs.cluster_lists {
        main.read_buffer(h);
    }
    if let Some(h) = front.scene_inputs.atmosphere_luts {
        main.read_texture(h);
    }
    if let Some(h) = front.prepass.ao_output {
        main.read_texture(h);
    }
    if let Some(h) = front.prepass.ssgi_indirect {
        main.read_texture(h);
    }
    // Main does not sample the mask; the read lands the mark ahead of Main
    // with the other pre-pass consumers, so the compute passes after Main stay
    // ordered against every render pass.
    if let Some(h) = front.prepass.disocclusion {
        main.read_texture(h);
    }
    if let Some(g) = front.cull.grass {
        g.read_into(&mut main);
    }
    if let Some(h) = hdr.color {
        let _ = main.write_texture(h);
    }
    let depth_v1 = main.write_texture(hdr.depth);
    let resolve_v1 = main.write_texture(hdr.resolve);
    SceneVersions {
        resolve: resolve_v1,
        depth: depth_v1,
    }
}

// The two-pass occlusion phase 2 over Main's output, returning the rebuilt
// pyramid and Main2's scene versions.
fn add_occlusion_phase2(
    b: &mut GraphBuilder,
    hiz: TextureHandle,
    cull_status: Option<BufferHandle>,
    hdr_color: Option<TextureHandle>,
    main_out: SceneVersions,
) -> (TextureHandle, SceneVersions) {
    // HizBuild (compute): read phase-1 depth, write the Hi-Z pyramid.
    // The depth RAW edge pins it after Main; the pyramid write is a WAR
    // against Cull's read of the previous frame's contents.
    let mut hizb = b.add_pass(PassId::HizBuild, PassKind::Compute);
    hizb.read_texture(main_out.depth);
    let hiz_v1 = hizb.write_texture(hiz);

    // Cull2 (compute): read the rebuilt pyramid + the phase-1 status
    // buffer, write a second indirect-draw-args buffer Main2 consumes.
    let draw_args2 = b.import_buffer("draw_args2", draw_args_desc());
    let mut cull2 = b.add_pass(PassId::Cull2, PassKind::Compute);
    cull2.read_texture(hiz_v1);
    if let Some(cs) = cull_status {
        cull2.read_buffer(cs);
    }
    let draw_args2_v1 = cull2.write_buffer(draw_args2);

    // Main2 (render): read the phase-2 draw args; RMW hdr_color /
    // hdr_depth / hdr_resolve. The draw_args2 RAW edge pins it after
    // Cull2; the hdr_depth write (WAR vs HizBuild's read) pins it after
    // HizBuild; the hdr_color / hdr_resolve WAW edges pin it after Main.
    let mut main2 = b.add_pass(PassId::Main2, PassKind::Render);
    main2.read_buffer(draw_args2_v1);
    let depth = main2.write_texture(main_out.depth);
    if let Some(h) = hdr_color {
        let _ = main2.write_texture(h);
    }
    let resolve = main2.write_texture(main_out.resolve);
    (hiz_v1, SceneVersions { resolve, depth })
}

// hdr_resolve post-Main RMW chain: Raymarch → Decals → Fog →
// ParticlesDraw, each blend- or opaque-writing on top of the
// previous version. The handle walks forward through `h` so each
// write picks up the latest version, giving the compile pass clean
// WAW edges to derive the chain order. Raymarch slots first so its
// depth+color write is visible to every later post-decoration
// pass; AutoExposure's WAR-read on hdr_resolve_head pins it before
// Raymarch (so SDF brightness doesn't skew exposure for the same
// frame), matching the doc's chosen one-frame-lag trade-off.
//
// Main depth rides the same pattern: Raymarch sphere-traces against it and
// writes the hit depth back, so it bumps the depth version, and every later
// decoration samples that version rather than the one Main left.
fn add_decorations(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    head: SceneVersions,
    front: &FrontEnd,
    reactive_mask: &mut Option<TextureHandle>,
) -> SceneVersions {
    let SceneVersions {
        resolve: mut h,
        depth: mut depth_cur,
    } = head;
    let ssr_gbuffer_v1 = front.prepass.ssr_gbuffer();
    let gbuffer_v1 = front.prepass.gbuffer;
    if inputs.raymarch_enabled {
        let mut rm = b.add_pass(PassId::Raymarch, PassKind::Render);
        rm.read_texture(h);
        h = rm.write_texture(h);
        depth_cur = rm.write_texture(depth_cur);
    }
    // SSGI reads the lit scene (its bounce-radiance source) into its own
    // accumulation, which the next frame's resolve hands Main. Slots right
    // after Raymarch so it can bounce raymarched surfaces too, and before
    // Decals / Fog / Particles, which are not light a surface reflects.
    if inputs.ssgi_enabled {
        let mut ssgi = b.add_pass(PassId::Ssgi, PassKind::Render);
        ssgi.read_texture(h);
        // The trace is against the pre-pass view normal + linear depth, weighs
        // each hit by its reflectance, and the accumulation reprojects through
        // the motion vectors; with no G-buffer there is nothing to trace
        // against and the encoder skips.
        if let Some(g) = ssr_gbuffer_v1 {
            ssgi.read_texture(g);
        }
        if let Some(g) = gbuffer_v1 {
            ssgi.read_texture(g.roughness);
            ssgi.read_texture(g.velocity);
        }
    }
    if inputs.decals_enabled {
        // Projected decals reconstruct each pixel's world position from the
        // scene depth, so the pass samples depth while blend-writing color.
        let mut decals = b.add_pass(PassId::Decals, PassKind::Render);
        decals.read_texture(depth_cur);
        h = decals.write_texture(h);
    }
    if let Some(luts) = front.scene_inputs.atmosphere_luts {
        // The air in front of each surface, under the fog: a volumetric fog is
        // the nearer layer, so it blends over the haze rather than under it.
        let mut aerial = b.add_pass(PassId::AerialPerspective, PassKind::Render);
        aerial.read_texture(luts);
        aerial.read_texture(depth_cur);
        h = aerial.write_texture(h);
    }
    if inputs.fog_enabled {
        // FogFroxel (compute) populates the 3D scatter/transmittance
        // volume the Fog fragment shader samples. The post-write handle
        // (`froxel_v1`) is what Fog reads: that gives the compile pass
        // a clean RAW edge so FogFroxel runs before Fog in the toposort.
        // All three backends implement the froxel path; the Fog render
        // pass trilinear-samples the volume by (screen_uv, view_z).
        let froxel_v0 = b.import_texture("fog_froxel_volume", froxel_volume_desc());
        let mut froxel = b.add_pass(PassId::FogFroxel, PassKind::Compute);
        // Each slab does a cascade tap, so the kernel is a second reader of the
        // shadow map alongside Main. Declaring it puts the compute stage into the
        // read run's union, which is what makes one transition serve both.
        if let Some(h) = front.scene_inputs.shadow {
            froxel.read_texture(h);
        }
        let froxel_v1 = froxel.write_texture(froxel_v0);
        let mut fog_pass = b.add_pass(PassId::Fog, PassKind::Render);
        fog_pass.read_texture(froxel_v1);
        // Scene depth bounds the ray march / froxel lookup per pixel.
        fog_pass.read_texture(depth_cur);
        h = fog_pass.write_texture(h);
    }
    if let Some(pools) = front.scene_inputs.particle_pool {
        // The billboard vertex shader pulls each particle out of the pool the
        // sim just wrote, so the read is in the VERTEX stage rather than the
        // FRAGMENT one a render pass otherwise implies. Naming it is what makes
        // the derived transition a vertex-visible one on both explicit backends.
        let mut particles = b.add_pass(PassId::ParticlesDraw, PassKind::Render);
        particles.read_buffer_in_stage(pools, ReadStages::VERTEX);
        // The fragment tests scene depth itself, so opaque geometry hides a
        // sprite behind it.
        particles.read_texture(depth_cur);
        h = particles.write_texture(h);
        *reactive_mask = reactive_mask.map(|m| particles.write_texture(m));
    }
    if inputs.lines_enabled {
        // Last of the hdr_resolve decorations: line geometry draws over the
        // lit + decorated scene, and SSR / TAA then treat it like any other
        // scene content. Samples depth rather than testing against it, so a
        // line behind geometry fades instead of disappearing.
        let mut lines = b.add_pass(PassId::Lines, PassKind::Render);
        lines.read_texture(depth_cur);
        h = lines.write_texture(h);
    }
    SceneVersions {
        resolve: h,
        depth: depth_cur,
    }
}

// The clouds over the pre-TAA scene the see-through surfaces left, so they
// stand in front of the sea as well as the ground: the march reads the
// atmosphere's tables and the depth it stops at, and its target is
// backend-owned, so the import only orders it ahead of the composite. Returns
// the pre-TAA scene's next version.
fn add_clouds(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    scene: SceneVersions,
    front: &FrontEnd,
) -> TextureHandle {
    let Some(luts) = front
        .scene_inputs
        .atmosphere_luts
        .filter(|_| inputs.clouds_enabled)
    else {
        return scene.resolve;
    };
    let target = b.import_texture("cloud_target", cloud_target_desc(inputs));
    let mut march = b.add_pass(PassId::Clouds, PassKind::Compute);
    march.read_texture(luts);
    march.read_texture(scene.depth);
    let target = march.write_texture(target);
    let mut composite = b.add_pass(PassId::CloudComposite, PassKind::Render);
    composite.read_texture(target);
    composite.read_texture(scene.depth);
    composite.write_texture(scene.resolve)
}

// The emitters that draw over the see-through surfaces: from the pools
// `ParticlesDraw` read, onto the pre-TAA scene Transparent left, testing the
// same depth. Returns the pre-TAA scene's next version.
fn add_late_particles(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    scene: SceneVersions,
    front: &FrontEnd,
    reactive_mask: &mut Option<TextureHandle>,
) -> TextureHandle {
    let Some(pools) = front
        .scene_inputs
        .particle_pool
        .filter(|_| inputs.late_particles_enabled)
    else {
        return scene.resolve;
    };
    let mut late = b.add_pass(PassId::ParticlesLateDraw, PassKind::Render);
    late.read_buffer_in_stage(pools, ReadStages::VERTEX);
    late.read_texture(scene.depth);
    let resolve = late.write_texture(scene.resolve);
    *reactive_mask = reactive_mask.map(|m| late.write_texture(m));
    resolve
}

// Planar mirrors, the reflection resolve, and Transparent, returning the
// pre-TAA scene.
fn add_reflections_and_transparent(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    decorated: SceneVersions,
    front: &FrontEnd,
    reactive_mask: &mut Option<TextureHandle>,
) -> TextureHandle {
    let SceneVersions {
        resolve: hdr_resolve_cur,
        depth: depth_cur,
    } = decorated;
    let SceneInputs {
        shadow: shadow_v1,
        spot_shadow: spot_shadow_v1,
        atmosphere_luts,
        cluster_lists,
        water_cascades,
        ..
    } = front.scene_inputs;
    let gbuffer_v1 = front.prepass.gbuffer;
    let ssr_gbuffer_v1 = front.prepass.ssr_gbuffer();

    // Planar mirrors: the scene re-rendered across each reflector plane, lit like
    // Main (so after the shadow maps) from the frame's culled geometry (so after
    // Cull). The mirror targets are encoder-owned, so the import only orders the
    // mirrors ahead of their one reader, Transparent.
    let planar_mirrors = if inputs.planar_reflection_enabled && inputs.transparent_enabled {
        let mirrors = b.import_texture("planar_mirrors", planar_mirrors_desc(inputs));
        let mut planar = b.add_pass(PassId::PlanarReflection, PassKind::Render);
        if let Some(h) = shadow_v1 {
            planar.read_texture(h);
        }
        if let Some(h) = spot_shadow_v1 {
            planar.read_texture(h);
        }
        if let Some(h) = front.cull.draw_args {
            planar.read_buffer(h);
        }
        if let Some(h) = atmosphere_luts {
            planar.read_texture(h);
        }
        Some(planar.write_texture(mirrors))
    } else {
        None
    };

    // `scene_pre_taa` is a distinct texture only when a pass writes it:
    // SsrResolve / RtReflections produce it, and Transparent read-modify-writes
    // it. With neither resolve the engine binds the pre-TAA scene name straight
    // to `hdr_resolve`, so declaring it here would put two graph resources on
    // one GPU object -- each with its own barrier timeline over the same memory.
    // Threading the upstream handle expresses that binding with one resource.
    if inputs.rt_reflections_enabled || inputs.ssr_enabled {
        let scene_pre_taa = b.import_texture("scene_pre_taa", scene_color_desc(inputs));
        // SsrResolve and RtReflections occupy the same slot: both read the
        // post-decoration hdr_resolve and write scene_pre_taa. Hardware RT
        // *takes precedence* over SSR: a world can enable both (RT on the
        // backend / GPU that supports it, SSR as the cross-backend fallback),
        // and where RT is live the builder picks it and omits SsrResolve. Only
        // one of the two is ever inserted.
        // Both resolves trace against the pre-pass view normal + linear depth
        // and pick their blur radius from its roughness. Declaring those reads
        // is what keeps the G-buffer's modeled lifetime as long as its real
        // one: roughness has no other consumer, so without this it would look
        // dead the moment the pre-pass finished.
        let mut current = if inputs.rt_reflections_enabled {
            let mut rt = b.add_pass(PassId::RtReflections, PassKind::Render);
            rt.read_texture(hdr_resolve_cur);
            if let Some(g) = ssr_gbuffer_v1 {
                rt.read_texture(g);
            }
            if let Some(g) = gbuffer_v1 {
                rt.read_texture(g.roughness);
            }
            // A missed ray's probe fallback reads its cluster's probe mask.
            if let Some(h) = cluster_lists {
                rt.read_buffer(h);
            }
            rt.write_texture(scene_pre_taa)
        } else {
            let mut ssr = b.add_pass(PassId::SsrResolve, PassKind::Render);
            ssr.read_texture(hdr_resolve_cur);
            if let Some(g) = ssr_gbuffer_v1 {
                ssr.read_texture(g);
            }
            if let Some(g) = gbuffer_v1 {
                ssr.read_texture(g.roughness);
            }
            // A missed ray's probe fallback reads its cluster's probe mask.
            if let Some(h) = cluster_lists {
                ssr.read_buffer(h);
            }
            ssr.write_texture(scene_pre_taa)
        };
        if inputs.transparent_enabled {
            let mut trans = b.add_pass(PassId::Transparent, PassKind::Render);
            // Pin Transparent after the whole post-decoration hdr_resolve chain
            // (Main → Decals → Fog → ParticlesDraw), which the scene_pre_taa
            // edge below does not imply.
            trans.read_texture(hdr_resolve_cur);
            // Depth is read at its latest version, not the imported v0:
            // reading v0 would be a WAR against Main's write and pin
            // Transparent *before* Main, closing a cycle.
            trans.read_texture(depth_cur);
            if let Some(h) = cluster_lists {
                trans.read_buffer(h);
            }
            if let Some(h) = planar_mirrors {
                trans.read_texture(h);
            }
            // The water body is lit through the sun's cascaded shadow.
            if let Some(h) = shadow_v1 {
                trans.read_texture(h);
            }
            if let Some(h) = water_cascades {
                trans.read_texture_in_stage(h, ReadStages::VERTEX | ReadStages::FRAGMENT);
            }
            // RMW the resolve output. The read declares the sample dependency
            // (translucents sample the resolved scene for refraction); the
            // write produces the blended version downstream passes consume.
            trans.read_texture(current);
            current = trans.write_texture(current);
            *reactive_mask = reactive_mask.map(|m| trans.write_texture(m));
        }
        current
    } else if inputs.transparent_enabled {
        // With no reflection resolve the pre-TAA scene *is* `hdr_resolve` and
        // glass blends straight into it, so Transparent extends the hdr_resolve
        // chain by one version instead of branching a second resource onto the
        // same object.
        let mut trans = b.add_pass(PassId::Transparent, PassKind::Render);
        trans.read_texture(depth_cur);
        if let Some(h) = cluster_lists {
            trans.read_buffer(h);
        }
        if let Some(h) = planar_mirrors {
            trans.read_texture(h);
        }
        if let Some(h) = shadow_v1 {
            trans.read_texture(h);
        }
        if let Some(h) = water_cascades {
            trans.read_texture_in_stage(h, ReadStages::VERTEX | ReadStages::FRAGMENT);
        }
        trans.read_texture(hdr_resolve_cur);
        *reactive_mask = reactive_mask.map(|m| trans.write_texture(m));
        trans.write_texture(hdr_resolve_cur)
    } else {
        hdr_resolve_cur
    }
}

// `scene_color` is the engine-owned output the post-TAA composite stack
// consumes, and -- with TAA on -- the history slot next frame samples, which
// is why it stays imported rather than becoming a transient. Declared only
// when TaaResolve or Upscale writes it, for the same one-object-one-resource
// reason as above: with neither, the engine binds the name to the latest
// pre-TAA scene texture. The two writers are mutually exclusive -- the
// upscaler does its own temporal accumulation, so layering TaaResolve on top
// would double-temporal the scene.
fn add_temporal(
    b: &mut GraphBuilder,
    inputs: &FrameGraphInputs,
    scene_pre_taa_cur: TextureHandle,
    gbuffer_v1: Option<GBufferHandles>,
    reactive_mask: Option<TextureHandle>,
) -> TextureHandle {
    let velocity_v1 = gbuffer_v1.map(|g| g.velocity);
    if inputs.upscale_enabled {
        let scene_color = b.import_texture("scene_color", scene_color_desc(inputs));
        // Compute, not render: the temporal upscaler is a dispatch, so its reads
        // want the non-pixel shader-resource state. Declaring it as a render
        // pass put the fragment stage in the read-stage union and left the
        // backend flipping the scene and the motion buffer by hand.
        let mut up = b.add_pass(PassId::Upscale, PassKind::Compute);
        up.read_texture(scene_pre_taa_cur);
        // Explicit velocity read so the toposort pins Velocity →
        // Upscale. The scaler consumes motion vectors directly.
        if let Some(v) = velocity_v1 {
            up.read_texture(v);
        }
        // The scaler also samples the pre-pass depth (single-sample at render
        // resolution), which is the only consumer that target has.
        if let Some(g) = gbuffer_v1 {
            up.read_texture(g.depth);
        }
        if let Some(m) = reactive_mask {
            up.read_texture(m);
        }
        up.write_texture(scene_color)
    } else if inputs.taa_enabled {
        let scene_color = b.import_texture("scene_color", scene_color_desc(inputs));
        let mut taa = b.add_pass(PassId::TaaResolve, PassKind::Render);
        taa.read_texture(scene_pre_taa_cur);
        // Explicit velocity read so the toposort pins Velocity →
        // TaaResolve. Without this the order rests on declaration order
        // alone.
        if let Some(v) = velocity_v1 {
            taa.read_texture(v);
        }
        if let Some(m) = reactive_mask {
            taa.read_texture(m);
        }
        taa.write_texture(scene_color)
    } else {
        scene_pre_taa_cur
    }
}

fn atmosphere_luts_desc() -> TextureDesc {
    // Stands for the atmosphere's whole table set, whose largest member is the
    // aerial-perspective volume; the backend owns and transitions the tables.
    use crate::render::environment::atmosphere::AERIAL_SIZE;
    TextureDesc::volume_3d(
        TextureSize::Absolute(AERIAL_SIZE),
        TextureSize::Absolute(AERIAL_SIZE),
        AERIAL_SIZE,
        PixelFormat::Rgba16Float,
        TextureUsage::STORAGE.union(TextureUsage::SHADER_READ),
    )
}

fn cloud_target_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // RGBA16F hazed cloud light + transmittance at half the HDR dims;
    // backend-owned, so the graph only tracks the dependency.
    use crate::render::environment::clouds::cloud_target_size;
    let [w, h] = cloud_target_size(inputs.hdr_width, inputs.hdr_height);
    TextureDesc::texture_2d(
        TextureSize::Absolute(w),
        TextureSize::Absolute(h),
        PixelFormat::Rgba16Float,
        TextureUsage::STORAGE.union(TextureUsage::SHADER_READ),
    )
}

fn water_cascades_desc() -> TextureDesc {
    // Stands for both cascade arrays of every simulation, at the finest
    // quality; the backend owns them and transitions them itself.
    use crate::render::uniforms::ocean::ocean_mip_levels;
    use crate::water::ocean::OceanQuality;
    let size = OceanQuality::FULL.size;
    TextureDesc::texture_2d(
        TextureSize::Absolute(size),
        TextureSize::Absolute(size),
        PixelFormat::Rgba16Float,
        TextureUsage::STORAGE.union(TextureUsage::SHADER_READ),
    )
    .with_array_layers(OceanQuality::FULL.cascades)
    .with_mip_levels(ocean_mip_levels(size))
}

fn froxel_volume_desc() -> TextureDesc {
    // The volumetric-fog froxel volume: a 3D texture the fog kernel writes and
    // the fog pass samples. Its Z extent also rides in `FogFroxelParams.
    // froxel_dims` so shaders can map indices to volume UVs.
    TextureDesc::volume_3d(
        TextureSize::Absolute(FOG_FROXEL_X),
        TextureSize::Absolute(FOG_FROXEL_Y),
        FOG_FROXEL_Z,
        PixelFormat::Rgba16Float,
        TextureUsage::STORAGE.union(TextureUsage::SHADER_READ),
    )
}

/// X/Y/Z dimensions of the volumetric-fog froxel volume. Sized to keep the
/// per-frame compute cost modest (~230 k threads per dispatch) while
/// preserving enough screen-space detail for shaft-of-light shadowing.
/// Backends that implement the froxel path read these constants directly;
/// the values also ride in `FogFroxelParams.froxel_dims` so shaders can map
/// between absolute indices and normalized volume UVs without recompiling.
pub const FOG_FROXEL_X: u32 = 80;
/// Fog froxels down the screen. See [`FOG_FROXEL_X`].
pub const FOG_FROXEL_Y: u32 = 45;
/// Fog froxel depth slices. See [`FOG_FROXEL_X`].
pub const FOG_FROXEL_Z: u32 = 64;

fn shadow_map_desc(size: u32) -> TextureDesc {
    TextureDesc::texture_2d(
        TextureSize::Absolute(size.max(1)),
        TextureSize::Absolute(size.max(1)),
        PixelFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL.union(TextureUsage::SHADER_READ),
    )
    .with_array_layers(u32::saturating_from(NUM_SHADOW_CASCADES))
}

fn spot_shadow_map_desc(slice_size: u32, slices: u32) -> TextureDesc {
    TextureDesc::texture_2d(
        TextureSize::Absolute(slice_size.max(1)),
        TextureSize::Absolute(slice_size.max(1)),
        PixelFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL.union(TextureUsage::SHADER_READ),
    )
    .with_array_layers(slices.max(1))
}

fn hdr_color_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    render_res_2d(
        inputs,
        PixelFormat::Rgba16Float,
        TextureUsage::RENDER_TARGET,
    )
    .with_sample_count(inputs.hdr_sample_count.max(1))
}

fn hdr_depth_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    render_res_2d(
        inputs,
        PixelFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL.union(TextureUsage::SHADER_READ),
    )
    .with_sample_count(inputs.hdr_sample_count.max(1))
}

// A single-sample 2D target at the HDR render resolution, which is what most of
// the frame's off-screen targets are. Render resolution is not the drawable
// extent under temporal upscaling, so these are `Absolute` off the inputs
// rather than `Drawable`.
fn render_res_2d(
    inputs: &FrameGraphInputs,
    format: PixelFormat,
    usage: TextureUsage,
) -> TextureDesc {
    TextureDesc::texture_2d(
        TextureSize::Absolute(inputs.hdr_width.max(1)),
        TextureSize::Absolute(inputs.hdr_height.max(1)),
        format,
        usage,
    )
}

fn draw_args_desc() -> BufferDesc {
    BufferDesc {
        size_bytes: None,
        usage: BufferUsage::STORAGE.union(BufferUsage::INDIRECT),
    }
}

fn grass_blades_desc() -> BufferDesc {
    // One `GpuGrassBlade` per visible blade. `STORAGE` alone, like the particle
    // pools: the kernel appends through an unordered-access binding and the
    // draws read it as a shader resource in their vertex stage. The executor
    // owns the allocation (sized to the field's blade capacity).
    BufferDesc {
        size_bytes: None,
        usage: BufferUsage::STORAGE,
    }
}

fn grass_bend_desc() -> BufferDesc {
    // Two halves of packed bends, one per frame parity: the bend pass reads one
    // and writes the other through the same read-write binding, and the
    // placement kernels read both. The executor owns the allocation.
    BufferDesc {
        size_bytes: None,
        usage: BufferUsage::STORAGE,
    }
}

fn cull_status_desc() -> BufferDesc {
    // One u32 per draw object: phase-1 cull writes drawn / hi-z-candidate /
    // culled, Cull2 reads it. Both phases bind it the same read-write way, so it
    // never transitions to a read state and its ordering comes from an execution
    // barrier; `UNORDERED` is what says so. The executor owns the allocation
    // (sized to the live draw-object count).
    BufferDesc {
        size_bytes: None,
        usage: BufferUsage::STORAGE.union(BufferUsage::UNORDERED),
    }
}

fn particle_pool_desc() -> BufferDesc {
    // One `GpuParticle` slot per particle, per emitter. `STORAGE` alone, so the
    // class is `StorageBuffer`: the sim reads and writes the pool through a
    // storage / unordered-access binding, but the draw reads it as a plain
    // shader resource in its vertex stage, so the resource really does leave the
    // unordered-access state between the two nodes. `UnorderedBuffer` is for the
    // buffer both sides bind the same read-write way (`cull_status`), which
    // never transitions and orders through a UAV barrier instead; that would
    // leave DirectX's vertex-stage SRV read unsynchronized. The executor owns
    // the allocation (one buffer per live emitter, sized to its pool).
    BufferDesc {
        size_bytes: None,
        usage: BufferUsage::STORAGE,
    }
}

fn cluster_lists_desc() -> BufferDesc {
    // Per-cluster light lists and probe masks LightCull writes and Main, the
    // reflection resolves and Transparent read. An identity stub: the backend
    // owns the real (persistent) buffer, so the graph only tracks the read/write
    // dependency, not the allocation.
    BufferDesc {
        size_bytes: None,
        usage: BufferUsage::STORAGE,
    }
}

fn hiz_pyramid_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // R32Float depth-mip pyramid rebuilt mid-frame from phase-1 depth, MAX
    // reduction. The cull kernel samples the coarse levels, so the chain is
    // most of the footprint and the desc carries its real length -- the same
    // `floor(log2(max)) + 1` each backend's Hi-Z build derives.
    render_res_2d(
        inputs,
        PixelFormat::R32Float,
        TextureUsage::STORAGE.union(TextureUsage::SHADER_READ),
    )
    .with_mip_levels(full_mip_levels(
        inputs.hdr_width.max(1),
        inputs.hdr_height.max(1),
    ))
}

// The unified G-buffer pre-pass writes four separate targets in one draw. They
// are four graph resources rather than one handle because their shapes differ
// (three color formats and a depth target) and so do their consumers, and a
// resource the aliaser may place has to name the memory it actually needs.
fn gbuffer_normal_depth_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // RGBA16F view-space normal + linear depth, read by SSR / SSAO / SSGI / RT.
    render_res_2d(
        inputs,
        PixelFormat::Rgba16Float,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

fn gbuffer_roughness_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // RGBA8 perceptual roughness (r) + head-on specular reflectance (gba), read
    // by the reflection resolves for their gloss, blur radius and Fresnel.
    // Clears to roughness 1.0 (fully rough), so a pixel the pre-pass never
    // rasterizes reflects nothing -- the one graph target whose cleared
    // background carries meaning, and the reason `TextureDesc` models a clear
    // value at all.
    render_res_2d(
        inputs,
        PixelFormat::Rgba8Unorm,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
    .with_clear_color([1.0, 0.0, 0.0, 0.0])
}

fn gbuffer_depth_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // The pre-pass's own depth attachment. Single-sample regardless of the
    // main pass's MSAA: the pre-pass rasterizes once.
    render_res_2d(
        inputs,
        PixelFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL.union(TextureUsage::SHADER_READ),
    )
}

fn reactive_mask_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // R8 at HDR dims: 0 keeps the temporal history, 1 discards it.
    render_res_2d(
        inputs,
        PixelFormat::R8Unorm,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

fn ao_output_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // R8 occlusion at HDR dims; sampled by Main's ambient term.
    render_res_2d(
        inputs,
        PixelFormat::R8Unorm,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

fn ssgi_indirect_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // RGBA16F visibility + bounce at HDR dims; sampled by Main's ambient term.
    render_res_2d(
        inputs,
        PixelFormat::Rgba16Float,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

fn velocity_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // RG16F motion-vector buffer at HDR dims, sampled by TaaResolve.
    render_res_2d(
        inputs,
        PixelFormat::Rg16Float,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

pub(crate) fn bloom_top_desc() -> TextureDesc {
    // bloom_top is the bloom chain's half-resolution top octave; the prefilter
    // pass writes into it and the upsample chain accumulates back into it for
    // Composite to sample.
    //
    // Half the *drawable* extent, not half the render resolution: every backend
    // builds its bloom chain from the output extent, so under temporal
    // upscaling (where render resolution is smaller) an `hdr_width >> 1` desc
    // names a texture no backend creates.
    TextureDesc::texture_2d(
        TextureSize::DrawableScaled(0.5),
        TextureSize::DrawableScaled(0.5),
        PixelFormat::Rgba16Float,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

fn hdr_resolve_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    render_res_2d(
        inputs,
        PixelFormat::Rgba16Float,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

fn prop_probe_desc() -> TextureDesc {
    // Stands for the probe's cube face the pass renders, at the largest face
    // it may capture; the backend owns the capture and the cube.
    use crate::render::probe::prop_reflection_probe::MAX_PROP_PROBE_RESOLUTION;
    TextureDesc::texture_2d(
        TextureSize::Absolute(MAX_PROP_PROBE_RESOLUTION),
        TextureSize::Absolute(MAX_PROP_PROBE_RESOLUTION),
        PixelFormat::Rgba16Float,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

fn planar_mirrors_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // Every plane's mirror under one label, described at render resolution.
    render_res_2d(
        inputs,
        PixelFormat::Rgba16Float,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

fn scene_color_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // The engine-owned scene_color texture the post stack consumes is
    // single-sample at HDR dims regardless of whether the per-frame
    // resolution lands on the TAA history / the reflection composite's output /
    // hdr_resolve.
    render_res_2d(inputs, PixelFormat::Rgba16Float, TextureUsage::SHADER_READ)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    // The production defaults, with MSAA on so the resolve-chain tests have a
    // multisampled hdr target to exercise.
    fn all_off() -> FrameGraphInputs {
        FrameGraphInputs {
            hdr_sample_count: 4,
            ..FrameGraphInputs::all_off()
        }
    }

    #[test]
    fn minimum_graph_is_main_then_composite() {
        let g = build_frame_graph(&all_off()).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(order, vec![PassId::Main, PassId::Composite]);
        assert!(g.passes[1].presents);
    }

    #[test]
    fn world_hidden_collapses_to_minimum_graph() {
        // Every heavy world pass requested, but the opaque menu backdrop is up:
        // the builder must mask them all off and yield the bare Main -> Composite
        // graph, with Composite still the presenter for the overlay.
        let mut i = all_off();
        i.shadow_enabled = true;
        i.bindless_cull_enabled = true;
        i.ssr_prepass_enabled = true;
        i.ssao_enabled = true;
        i.ssgi_enabled = true;
        i.rt_reflections_enabled = true;
        i.bloom_enabled = true;
        i.taa_enabled = true;
        i.velocity_enabled = true;
        i.two_pass_occlusion_enabled = true;
        i.world_hidden = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(order, vec![PassId::Main, PassId::Composite]);
        assert!(g.passes[1].presents);
    }

    #[test]
    fn shadow_orders_before_main() {
        let mut i = all_off();
        i.shadow_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(order, vec![PassId::Shadow, PassId::Main, PassId::Composite]);
    }

    #[test]
    fn cull_orders_before_main_via_draw_args() {
        let mut i = all_off();
        i.bindless_cull_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(order, vec![PassId::Cull, PassId::Main, PassId::Composite]);
        assert_eq!(g.passes[0].kind, PassKind::Compute);
    }

    #[test]
    fn two_pass_inserts_phase2_chain_after_main() {
        // With bindless cull + two-pass on, the graph gains the phase-2
        // prefix Cull → Main → HizBuild → Cull2 → Main2, strictly ordered.
        let mut i = all_off();
        i.bindless_cull_enabled = true;
        i.two_pass_occlusion_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::Cull,
                PassId::Main,
                PassId::HizBuild,
                PassId::Cull2,
                PassId::Main2,
                PassId::Composite,
            ]
        );
        assert_eq!(g.passes[2].kind, PassKind::Compute); // HizBuild
        assert_eq!(g.passes[3].kind, PassKind::Compute); // Cull2
        assert_eq!(g.passes[4].kind, PassKind::Render); // Main2
    }

    // Each cull packs its draws through one group-count scratch per frame
    // slot, which phase 2 reuses: phase 1 packs within `Cull`, and Main and
    // its pre-pass draw phase 1's lists, all before `Cull2` writes it again.
    // Every feature that adds passes between them is on, so none may move a
    // reader of phase 1's lists past `Cull2`.
    #[test]
    fn phase_ones_lists_are_drawn_before_phase_two_culls() {
        let mut i = all_off();
        for (_, enable) in GATED_FLAGS {
            enable(&mut i);
        }
        i.world_hidden = false;
        let g = build_frame_graph(&i).expect("compiles");
        let at = |id| {
            g.passes
                .iter()
                .position(|p| p.id == id)
                .unwrap_or_else(|| panic!("{id:?} in the graph"))
        };
        let cull2 = at(PassId::Cull2);
        for reader in [PassId::Cull, PassId::Main, PassId::GBufferPrepass] {
            assert!(at(reader) < cull2, "{reader:?} runs before Cull2");
        }
        assert!(at(PassId::Cull) < at(PassId::GBufferPrepass));
    }

    // Index of `label` in the compiled graph's resource arena.
    fn resource_of(g: &CompiledGraph, label: &str) -> usize {
        g.resources
            .iter()
            .position(|r| r.label == label)
            .unwrap_or_else(|| panic!("{label} missing from the graph"))
    }

    // The late particles draw over what Transparent left, before the temporal
    // pass takes it, with or without a reflection resolve between; without a
    // late emitter there is no late pass.
    #[test]
    fn late_particles_draw_after_transparent_and_before_taa() {
        for ssr in [false, true] {
            let mut i = all_off();
            i.particles_enabled = true;
            i.transparent_enabled = true;
            i.taa_enabled = true;
            i.ssr_enabled = ssr;
            let g = build_frame_graph(&i).expect("compiles");
            assert!(g.passes.iter().all(|p| p.id != PassId::ParticlesLateDraw));
            i.late_particles_enabled = true;
            let g = build_frame_graph(&i).expect("compiles");
            let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
            let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
            assert!(pos(PassId::ParticlesDraw) < pos(PassId::Transparent));
            assert!(
                pos(PassId::Transparent) < pos(PassId::ParticlesLateDraw),
                "ssr {ssr}"
            );
            assert!(
                pos(PassId::ParticlesLateDraw) < pos(PassId::TaaResolve),
                "ssr {ssr}"
            );
        }
    }

    #[test]
    fn hiz_final_closes_the_frame_over_the_last_depth_version() {
        // The terminal pyramid rebuild must read the depth version every
        // decoration pass has already written and read, so main depth stays live
        // to the end of the graph. Raymarch bumps depth to v2, so HizFinal reads
        // v2 and lands after Lines / Transparent, which read the same version.
        let mut i = all_off();
        i.hiz_build_enabled = true;
        i.raymarch_enabled = true;
        i.lines_enabled = true;
        i.transparent_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::Lines) < pos(PassId::HizFinal));
        assert!(pos(PassId::Transparent) < pos(PassId::HizFinal));
        assert!(pos(PassId::HizFinal) < pos(PassId::Composite));

        let depth = resource_of(&g, "hdr_depth");
        let hizf = &g.passes[pos(PassId::HizFinal)];
        let read = hizf
            .reads
            .iter()
            .find(|r| r.resource_index() == depth)
            .expect("HizFinal reads depth");
        assert_eq!(read.version(), 2, "Main writes v1, Raymarch bumps to v2");
        // And it is the graph's last touch of depth: the resource's lifetime has
        // to reach this pass or an aliaser could reuse the memory under it.
        assert_eq!(g.resources[depth].lifetime.last, pos(PassId::HizFinal));
    }

    #[test]
    fn cull_reads_the_pyramid_the_terminal_build_overwrites() {
        // Phase-1 cull tests against the pyramid the previous frame left. That
        // read is what gives the terminal rebuild a write-after-read edge; without
        // it the rebuild is unordered against the cull that is still sampling.
        let mut i = all_off();
        i.hiz_build_enabled = true;
        i.bindless_cull_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        let hiz = resource_of(&g, "hiz_pyramid");
        assert!(
            g.passes[pos(PassId::Cull)]
                .reads
                .iter()
                .any(|r| r.resource_index() == hiz)
        );
        assert!(pos(PassId::Cull) < pos(PassId::HizFinal));
    }

    #[test]
    fn grass_reads_the_pyramid_ahead_of_its_rebuilds() {
        // The grass kernel occlusion-tests its tiles against the pyramid the
        // previous frame left, so both rebuilds must wait for it, the mid-frame
        // one under two-pass occlusion included.
        for two_pass in [false, true] {
            let mut i = all_off();
            i.hiz_build_enabled = true;
            i.bindless_cull_enabled = true;
            i.two_pass_occlusion_enabled = two_pass;
            i.grass_enabled = true;
            let g = build_frame_graph(&i).expect("compiles");
            let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
            let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
            let hiz = resource_of(&g, "hiz_pyramid");
            assert!(
                g.passes[pos(PassId::Grass)]
                    .reads
                    .iter()
                    .any(|r| r.resource_index() == hiz)
            );
            assert!(pos(PassId::Grass) < pos(PassId::HizFinal));
            if two_pass {
                assert!(pos(PassId::Grass) < pos(PassId::HizBuild));
            }
        }
    }

    // The bend pass stamps the field both placement kernels read, and the
    // cascade's blades are drawn by the Shadow pass, so the order is bend,
    // then the kernels, then their draws.
    #[test]
    fn grass_bends_before_it_places_and_casts_before_main() {
        let mut i = all_off();
        i.bindless_cull_enabled = true;
        i.gbuffer_prepass_enabled = true;
        i.velocity_enabled = true;
        i.grass_enabled = true;
        i.shadow_enabled = true;
        i.grass_shadow_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::GrassBend) < pos(PassId::Grass));
        assert!(pos(PassId::GrassBend) < pos(PassId::GrassShadow));
        assert!(pos(PassId::GrassShadow) < pos(PassId::Shadow));
        assert!(pos(PassId::Grass) < pos(PassId::GBufferPrepass));
        assert!(pos(PassId::Shadow) < pos(PassId::Main));
        let blades = resource_of(&g, "grass_shadow_blades");
        assert!(
            g.passes[pos(PassId::Shadow)]
                .reads
                .iter()
                .any(|r| r.resource_index() == blades)
        );

        // Without shadows, or with the cascade's draw off, nothing is placed
        // for it; the bend pass still runs for the drawn blades.
        for (shadow, cast) in [(false, true), (true, false)] {
            i.shadow_enabled = shadow;
            i.grass_shadow_enabled = cast;
            let g = build_frame_graph(&i).expect("compiles");
            assert!(g.passes.iter().all(|p| p.id != PassId::GrassShadow));
            assert!(g.passes.iter().any(|p| p.id == PassId::GrassBend));
        }
    }

    #[test]
    fn hiz_final_off_means_no_pyramid_in_the_graph() {
        // A world without the GPU-cull path builds no pyramid, so neither the node
        // nor the resource may appear (an imported resource with no pass would
        // still take a registry entry in every backend).
        let g = build_frame_graph(&all_off()).expect("compiles");

        assert!(!g.passes.iter().map(|p| p.id).any(|x| x == PassId::HizFinal));
        assert!(!g.resources.iter().any(|r| r.label == "hiz_pyramid"));
    }

    #[test]
    fn a_hidden_world_still_rebuilds_the_pyramid() {
        // Masking drops every world pass, but the pyramid feeds the *next* frame's
        // cull, so the terminal build survives: dropping it would leave a stale
        // pyramid the frame after the menu closes.
        let mut i = all_off();
        i.hiz_build_enabled = true;
        i.bindless_cull_enabled = true;
        i.world_hidden = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![PassId::Main, PassId::HizFinal, PassId::Composite]
        );
    }

    #[test]
    fn depth_readers_all_sample_the_post_raymarch_version() {
        // Raymarch writes hit depth back, so every later decoration must sample
        // the version it produced. Reading Main's version instead would be a
        // write-after-read against Raymarch and pin the readers ahead of it.
        let mut i = all_off();
        i.raymarch_enabled = true;
        i.decals_enabled = true;
        i.fog_enabled = true;
        i.lines_enabled = true;
        i.transparent_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        let depth = resource_of(&g, "hdr_depth");
        for pass in [
            PassId::Decals,
            PassId::Fog,
            PassId::Lines,
            PassId::Transparent,
        ] {
            let read = g.passes[pos(pass)]
                .reads
                .iter()
                .find(|r| r.resource_index() == depth)
                .unwrap_or_else(|| panic!("{pass:?} reads depth"));
            assert_eq!(read.version(), 2, "{pass:?}");
        }
        assert!(pos(PassId::Raymarch) < pos(PassId::Decals));
    }

    #[test]
    fn the_msaa_color_attachment_is_declared_only_when_multisampled() {
        // Without MSAA there is no resolve step and the single color target is
        // the spine, so declaring `hdr_color` too would put two graph resources
        // on one GPU object. Every backend already reflects this: Vulkan leaves
        // `color_images` empty, DirectX and Metal leave `resolve` None.
        let mut i = all_off();
        i.hdr_sample_count = 1;
        let g = build_frame_graph(&i).expect("compiles");
        assert!(!g.resources.iter().any(|r| r.label == "hdr_color"));
        assert!(g.resources.iter().any(|r| r.label == "hdr_resolve"));

        i.hdr_sample_count = 4;
        let g = build_frame_graph(&i).expect("compiles");
        assert!(g.resources.iter().any(|r| r.label == "hdr_color"));
        assert!(g.resources.iter().any(|r| r.label == "hdr_resolve"));
    }

    #[test]
    fn dropping_the_msaa_attachment_keeps_the_phase_order() {
        // The hdr_color write-after-write is one of three edges pinning Main2
        // after Main; the depth and hdr_resolve writes carry the order on their
        // own, so a single-sampled two-pass frame still runs in phase order.
        let mut i = all_off();
        i.hdr_sample_count = 1;
        i.bindless_cull_enabled = true;
        i.two_pass_occlusion_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::Cull,
                PassId::Main,
                PassId::HizBuild,
                PassId::Cull2,
                PassId::Main2,
                PassId::Composite,
            ]
        );
    }

    #[test]
    fn two_pass_without_bindless_cull_is_noop() {
        // Two-pass rides the bindless GPU-cull path; requesting it without
        // a bindless shader must not insert any phase-2 nodes.
        let mut i = all_off();
        i.two_pass_occlusion_enabled = true; // bindless_cull_enabled left false
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(order, vec![PassId::Main, PassId::Composite]);
        assert!(!order.contains(&PassId::HizBuild));
        assert!(!order.contains(&PassId::Cull2));
        assert!(!order.contains(&PassId::Main2));
    }

    #[test]
    fn two_pass_shifts_post_chain_head_to_main2() {
        // AutoExposure + the RMW chain must read Main2's hdr_resolve (v2),
        // not Main's (v1), so the post stack sees the combined two-pass
        // scene. Main writes v1, Main2 writes v2, AutoExposure reads v2,
        // Decals bumps to v3.
        let mut i = all_off();
        i.bindless_cull_enabled = true;
        i.two_pass_occlusion_enabled = true;
        i.auto_exposure_enabled = true;
        i.decals_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::Main2) < pos(PassId::AutoExposure));
        assert!(pos(PassId::AutoExposure) < pos(PassId::Decals));
        // Version walk: Main2 RMWs hdr_resolve to v2, Decals to v3.
        let main2 = &g.passes[pos(PassId::Main2)];
        // hdr_resolve is the last write Main2 declares (depth, color, resolve).
        assert_eq!(main2.writes.last().unwrap().version(), 2);
        let decals = &g.passes[pos(PassId::Decals)];
        assert_eq!(decals.writes[0].version(), 3);
    }

    #[test]
    fn ssao_orders_before_main_via_ao_output() {
        let mut i = all_off();
        i.ssao_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![PassId::SsaoBlur, PassId::Main, PassId::Composite]
        );
    }

    #[test]
    fn ao_output_barriers_are_graph_driven() {
        // The DirectX + Vulkan executors emit `ao_output`'s transitions from
        // these barriers (resolving them to RENDER_TARGET / COLOR_ATTACHMENT on
        // SsaoBlur and back to the sampled state on Main). Pin the exact pair
        // so the executor's stripped inline / render-pass-baked transitions
        // stay matched to what the graph derives.
        use super::super::ResourceState;
        let mut i = all_off();
        i.ssao_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let pass = |id: PassId| g.passes.iter().find(|p| p.id == id).expect("present");

        let ssao = g.pass_barriers_for(pass(PassId::SsaoBlur), &["ao_output"]);
        assert_eq!(ssao.len(), 1, "SsaoBlur has exactly one ao_output barrier");
        assert_eq!(ssao[0].1.source_state(), ResourceState::Undefined);
        assert_eq!(ssao[0].1.to_state(), ResourceState::Write);

        let main = g.pass_barriers_for(pass(PassId::Main), &["ao_output"]);
        assert_eq!(main.len(), 1, "Main has exactly one ao_output barrier");
        assert_eq!(main[0].1.source_state(), ResourceState::Write);
        assert_eq!(main[0].1.to_state(), ResourceState::Read);
    }

    #[test]
    fn shadow_map_barriers_are_graph_driven() {
        // The executors emit `shadow_map`'s transitions from these barriers. The
        // graph derives the producer (Undefined -> Write) + the Main consumer
        // (Write -> Read); each backend resolves the producer against the
        // resource's resting state. DirectX rests it sampled, so the producer is
        // the real PIXEL_SHADER_RESOURCE -> DEPTH_WRITE cross-frame reset (folded
        // off the old inline restore); Main's consumer replaces the encoder's
        // stripped sampled transition. With SSAO also on, Main carries both
        // shadow_map and ao_output barriers, exercising multi-resource emission
        // in one pass.
        use super::super::ResourceState;
        let mut i = all_off();
        i.shadow_enabled = true;
        i.ssao_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let pass = |id: PassId| g.passes.iter().find(|p| p.id == id).expect("present");

        let shadow = g.pass_barriers_for(pass(PassId::Shadow), &["shadow_map"]);
        assert_eq!(shadow.len(), 1, "Shadow has exactly one shadow_map barrier");
        assert_eq!(shadow[0].1.source_state(), ResourceState::Undefined);
        assert_eq!(shadow[0].1.to_state(), ResourceState::Write);

        let main = g.pass_barriers_for(pass(PassId::Main), &["shadow_map"]);
        assert_eq!(main.len(), 1, "Main has exactly one shadow_map barrier");
        assert_eq!(main[0].1.source_state(), ResourceState::Write);
        assert_eq!(main[0].1.to_state(), ResourceState::Read);

        // Main carries both migrated resources' barriers in one pass.
        let both = g.pass_barriers_for(pass(PassId::Main), &["shadow_map", "ao_output"]);
        assert_eq!(
            both.len(),
            2,
            "Main carries shadow_map + ao_output barriers"
        );
    }

    #[test]
    fn fog_froxel_volume_barriers_are_graph_driven() {
        // The executors emit `fog_froxel_volume`'s transitions from these
        // barriers. FogFroxel's producer (Undefined -> Write) is the compute
        // write, a real sampled -> storage open on both backends now: DirectX
        // resolves it to PIXEL_SHADER_RESOURCE -> UNORDERED_ACCESS, Vulkan to
        // SHADER_READ_ONLY -> GENERAL (both rest the volume sampled, with no
        // inline reset). Fog's consumer (Write -> Read) is the storage-write ->
        // sampled close the fragment reads through.
        use super::super::ResourceState;
        let mut i = all_off();
        i.fog_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let pass = |id: PassId| g.passes.iter().find(|p| p.id == id).expect("present");

        let froxel = g.pass_barriers_for(pass(PassId::FogFroxel), &["fog_froxel_volume"]);
        assert_eq!(
            froxel.len(),
            1,
            "FogFroxel has exactly one fog_froxel_volume barrier"
        );
        assert_eq!(froxel[0].1.source_state(), ResourceState::Undefined);
        assert_eq!(froxel[0].1.to_state(), ResourceState::Write);

        let fog = g.pass_barriers_for(pass(PassId::Fog), &["fog_froxel_volume"]);
        assert_eq!(
            fog.len(),
            1,
            "Fog has exactly one fog_froxel_volume barrier"
        );
        assert_eq!(fog[0].1.source_state(), ResourceState::Write);
        assert_eq!(fog[0].1.to_state(), ResourceState::Read);
    }

    #[test]
    fn ssr_prepass_and_ssao_share_gbuffer_pinning_order() {
        let mut i = all_off();
        i.gbuffer_prepass_enabled = true;
        i.ssr_prepass_enabled = true;
        i.ssao_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::GBufferPrepass,
                PassId::SsaoBlur,
                PassId::Main,
                PassId::Composite,
            ]
        );
    }

    #[test]
    fn one_prepass_node_serves_ssr_ssao_and_velocity() {
        // SSR, SSAO and TAA all read the one GBufferPrepass output: SSAO its
        // normal+depth, TAA its motion.
        let mut i = all_off();
        i.gbuffer_prepass_enabled = true;
        i.ssr_prepass_enabled = true;
        i.ssao_enabled = true;
        i.velocity_enabled = true;
        i.taa_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::GBufferPrepass));
        let gb = order
            .iter()
            .position(|p| *p == PassId::GBufferPrepass)
            .unwrap();
        let ssao = order.iter().position(|p| *p == PassId::SsaoBlur).unwrap();
        let main = order.iter().position(|p| *p == PassId::Main).unwrap();
        let taa = order.iter().position(|p| *p == PassId::TaaResolve).unwrap();
        assert!(
            gb < ssao && ssao < main,
            "GBufferPrepass before SsaoBlur+Main"
        );
        assert!(gb < taa, "GBufferPrepass before TaaResolve");
    }

    #[test]
    fn gbuffer_prepass_runs_for_ssao_only() {
        // SSAO alone (no SSR / velocity) still triggers the merged node.
        let mut i = all_off();
        i.gbuffer_prepass_enabled = true;
        i.ssao_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::GBufferPrepass,
                PassId::SsaoBlur,
                PassId::Main,
                PassId::Composite,
            ]
        );
    }

    #[test]
    fn gbuffer_prepass_runs_for_velocity_only() {
        // Velocity alone (TAA, no SSR/SSAO) still triggers the pre-pass.
        let mut i = all_off();
        i.gbuffer_prepass_enabled = true;
        i.velocity_enabled = true;
        i.taa_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");

        assert!(
            g.passes
                .iter()
                .map(|p| p.id)
                .any(|x| x == PassId::GBufferPrepass)
        );
    }

    #[test]
    fn gbuffer_prepass_orders_after_cull_via_draw_args() {
        // The GPU-driven G-buffer pre-pass reuses the main pass's per-frame
        // indirect command buffer, so it must run after Cull. With bindless cull
        // on and a G-buffer consumer active, the draw_args read edge pins
        // Cull -> GBufferPrepass (-> Main).
        let mut i = all_off();
        i.bindless_cull_enabled = true;
        i.gbuffer_prepass_enabled = true;
        i.ssao_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let cull = order.iter().position(|p| *p == PassId::Cull).unwrap();
        let gb = order
            .iter()
            .position(|p| *p == PassId::GBufferPrepass)
            .unwrap();
        let main = order.iter().position(|p| *p == PassId::Main).unwrap();
        assert!(cull < gb, "Cull before GBufferPrepass");
        assert!(gb < main, "GBufferPrepass before Main");
    }

    #[test]
    fn gbuffer_prepass_omitted_when_no_consumers() {
        // The flag on but no consumer active: no pre-pass node at all.
        let mut i = all_off();
        i.gbuffer_prepass_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(order, vec![PassId::Main, PassId::Composite]);
    }

    #[test]
    fn auto_exposure_war_pinned_before_first_hdr_writer() {
        // AutoExposure reads hdr_resolve_v1. Decals writes v2 (when
        // enabled). The WAR edge from AutoExposure to Decals pins
        // AutoExposure before Decals; without it, the toposort could
        // place them in either order.
        let mut i = all_off();
        i.auto_exposure_enabled = true;
        i.decals_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::Main,
                PassId::AutoExposure,
                PassId::Decals,
                PassId::Composite,
            ]
        );
    }

    #[test]
    fn full_hdr_chain_orders_decals_fog_particles_then_ssr() {
        let mut i = all_off();
        i.decals_enabled = true;
        i.fog_enabled = true;
        i.particles_enabled = true;
        i.ssr_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                // ParticlesSim depends on nothing, so it keeps the position its
                // declaration gave it: ahead of the raster front it can overlap.
                PassId::ParticlesSim,
                PassId::Main,
                PassId::Decals,
                PassId::FogFroxel,
                PassId::Fog,
                PassId::ParticlesDraw,
                PassId::SsrResolve,
                PassId::Composite,
            ]
        );
        // Version chain on hdr_resolve walks 1 → 2 → 3 → 4 with
        // SsrResolve reading v4. FogFroxel slots between Decals and Fog
        // (writing the froxel volume to v1) but doesn't touch hdr_resolve,
        // so the version walk skips it.
        let at = |id| &g.passes[g.pass_index(id).expect("present")];
        assert_eq!(at(PassId::Decals).writes[0].version(), 2);
        assert_eq!(at(PassId::Fog).writes[0].version(), 3);
        assert_eq!(at(PassId::ParticlesDraw).writes[0].version(), 4);
        assert_eq!(at(PassId::SsrResolve).reads[0].version(), 4);
        // The draw's pool read is the vertex-stage one the sim's transition has
        // to cover; nothing else in the frame reads it.
        let pool_read = at(PassId::ParticlesDraw)
            .reads
            .iter()
            .find(|r| g.resources[r.resource_index()].label == "particle_pool")
            .expect("the draw declares the pool read");
        assert_eq!(pool_read.stage(), ReadStages::VERTEX);
        assert!(
            at(PassId::ParticlesDraw)
                .reads
                .iter()
                .any(|r| g.resources[r.resource_index()].label == "hdr_depth"),
            "the draw declares the scene depth read its fragment tests against"
        );
    }

    #[test]
    fn upscale_replaces_taa_and_pins_after_the_prepass() {
        // Upscale takes TaaResolve's slot when temporal upscaling is on.
        // TaaResolve must not appear in the compiled graph (the scaler
        // does temporal accumulation itself), and the pre-pass that produces
        // the motion vectors must precede Upscale via the explicit read.
        let mut i = all_off();
        i.gbuffer_prepass_enabled = true;
        i.velocity_enabled = true;
        i.upscale_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::Upscale));
        assert!(!order.contains(&PassId::TaaResolve));
        assert!(
            order
                .iter()
                .position(|p| *p == PassId::GBufferPrepass)
                .unwrap()
                < order.iter().position(|p| *p == PassId::Upscale).unwrap()
        );
    }

    #[test]
    fn upscale_takes_precedence_when_both_taa_and_upscale_requested() {
        // If both flags somehow arrive set (the engine layer should
        // forbid this, but the graph is the safety net), Upscale wins
        // and TaaResolve is omitted.
        let mut i = all_off();
        i.velocity_enabled = true;
        i.taa_enabled = true;
        i.upscale_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::Upscale));
        assert!(!order.contains(&PassId::TaaResolve));
    }

    #[test]
    fn velocity_taa_pinned_via_explicit_read() {
        // TaaResolve reads the velocity buffer explicitly so the toposort
        // orders the pre-pass that wrote it before TaaResolve via RAW (not
        // declaration order).
        let mut i = all_off();
        i.gbuffer_prepass_enabled = true;
        i.velocity_enabled = true;
        i.taa_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(
            order
                .iter()
                .position(|p| *p == PassId::GBufferPrepass)
                .unwrap()
                < order.iter().position(|p| *p == PassId::TaaResolve).unwrap()
        );
    }

    #[test]
    fn both_reflection_resolves_read_the_cluster_lists() {
        // A missed ray blends the probes its cluster bins, so each resolve must
        // order after LightCull and keep the lists live until it has read them.
        for rt in [false, true] {
            let mut i = all_off();
            i.ssr_enabled = true;
            i.ssr_prepass_enabled = true;
            i.rt_reflections_enabled = rt;
            i.clustering_enabled = true;
            let g = build_frame_graph(&i).expect("compiles");
            let resolve = if rt {
                PassId::RtReflections
            } else {
                PassId::SsrResolve
            };
            let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
            let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
            let lists = resource_of(&g, "cluster_lists");
            assert!(
                g.passes[pos(resolve)]
                    .reads
                    .iter()
                    .any(|r| r.resource_index() == lists),
                "{resolve:?} reads the cluster lists"
            );
            assert!(pos(PassId::LightCull) < pos(resolve));
            assert!(g.resources[lists].lifetime.last >= pos(resolve));
        }
    }

    #[test]
    fn transparent_reads_the_cluster_lists_with_and_without_a_resolve() {
        // Glass and water blend the probes their cluster bins, on either side of
        // the reflection-resolve branch.
        for ssr in [false, true] {
            let mut i = all_off();
            i.ssr_enabled = ssr;
            i.ssr_prepass_enabled = ssr;
            i.transparent_enabled = true;
            i.clustering_enabled = true;
            let g = build_frame_graph(&i).expect("compiles");
            let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
            let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
            let lists = resource_of(&g, "cluster_lists");
            let trans = pos(PassId::Transparent);
            assert!(
                g.passes[trans]
                    .reads
                    .iter()
                    .any(|r| r.resource_index() == lists),
                "Transparent reads the cluster lists (ssr: {ssr})"
            );
            assert!(pos(PassId::LightCull) < trans);
            assert!(g.resources[lists].lifetime.last >= trans);
        }
    }

    #[test]
    fn transparent_reads_the_shadow_map_with_and_without_a_resolve() {
        // The water body is lit through the sun's cascades, so Transparent reads
        // the shadow map Main sampled, on either side of the resolve branch.
        for ssr in [false, true] {
            let mut i = all_off();
            i.shadow_enabled = true;
            i.ssr_enabled = ssr;
            i.ssr_prepass_enabled = ssr;
            i.transparent_enabled = true;
            let g = build_frame_graph(&i).expect("compiles");
            let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
            let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
            let shadow = resource_of(&g, "shadow_map");
            let trans = pos(PassId::Transparent);
            assert!(
                g.passes[trans]
                    .reads
                    .iter()
                    .any(|r| r.resource_index() == shadow),
                "Transparent reads the shadow map (ssr: {ssr})"
            );
            assert!(pos(PassId::Shadow) < trans);
            assert!(g.resources[shadow].lifetime.last >= trans);
        }
    }

    #[test]
    fn transparent_pinned_between_ssr_resolve_and_taa() {
        // Transparent extends the scene_pre_taa chain by one version
        // after SsrResolve, so the toposort orders SsrResolve →
        // Transparent → TaaResolve via RAW + WAW edges on the same
        // texture.
        let mut i = all_off();
        i.ssr_enabled = true;
        i.taa_enabled = true;
        i.velocity_enabled = true;
        i.transparent_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::SsrResolve) < pos(PassId::Transparent));
        assert!(pos(PassId::Transparent) < pos(PassId::TaaResolve));
    }

    #[test]
    fn planar_mirrors_render_after_their_inputs_and_before_transparent() {
        for ssr in [false, true] {
            let mut i = all_off();
            i.bindless_cull_enabled = true;
            i.shadow_enabled = true;
            i.shadowed_spot_count = 1;
            i.ssr_enabled = ssr;
            i.transparent_enabled = true;
            i.planar_reflection_enabled = true;
            let g = build_frame_graph(&i).expect("compiles");
            let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
            let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
            let planar = pos(PassId::PlanarReflection);
            assert!(pos(PassId::Cull) < planar, "ssr: {ssr}");
            assert!(pos(PassId::Shadow) < planar, "ssr: {ssr}");
            assert!(pos(PassId::SpotShadow) < planar, "ssr: {ssr}");
            assert!(planar < pos(PassId::Transparent), "ssr: {ssr}");
            let mirrors = resource_of(&g, "planar_mirrors");
            assert!(
                g.passes[pos(PassId::Transparent)]
                    .reads
                    .iter()
                    .any(|r| r.resource_index() == mirrors),
                "Transparent reads the mirrors (ssr: {ssr})"
            );
        }
    }

    #[test]
    fn planar_mirrors_need_a_transparent_pass_to_read_them() {
        let mut i = all_off();
        i.planar_reflection_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        assert!(g.passes.iter().all(|p| p.id != PassId::PlanarReflection));
        assert!(g.resources.iter().all(|r| r.label != "planar_mirrors"));

        i.planar_reflection_enabled = false;
        i.transparent_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        assert!(g.passes.iter().all(|p| p.id != PassId::PlanarReflection));
    }

    #[test]
    fn transparent_works_without_ssr() {
        // Without a reflection resolve there is no scene_pre_taa texture, so
        // Transparent RMWs hdr_resolve directly and TaaResolve reads that.
        let mut i = all_off();
        i.taa_enabled = true;
        i.velocity_enabled = true;
        i.transparent_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::Transparent));
        assert!(!order.contains(&PassId::SsrResolve));
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::Main) < pos(PassId::Transparent));
        assert!(pos(PassId::Transparent) < pos(PassId::TaaResolve));
    }

    // A resource is declared only where a pass writes it. The engine points
    // several names at one texture depending on configuration, and declaring the
    // second name anyway would give one GPU object two barrier timelines, each
    // transitioning it from a state the other just left. These pin the three
    // configurations where that could recur.

    #[test]
    fn no_reflection_resolve_means_no_scene_pre_taa_resource() {
        // With neither SSR nor RT the engine binds the pre-TAA scene name to
        // hdr_resolve itself, so a scene_pre_taa resource would be a second
        // handle on that object.
        let mut i = all_off();
        i.taa_enabled = true;
        i.velocity_enabled = true;
        i.transparent_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        assert!(
            !g.resources.iter().any(|r| r.label == "scene_pre_taa"),
            "scene_pre_taa is hdr_resolve here; it must not be declared twice"
        );
        // And it reappears the moment a pass genuinely writes it.
        i.ssr_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        assert!(g.resources.iter().any(|r| r.label == "scene_pre_taa"));
    }

    #[test]
    fn no_temporal_pass_means_no_scene_color_resource() {
        // Neither TaaResolve nor Upscale runs, so scene_color is bound to the
        // latest pre-TAA scene texture and Composite reads that version.
        let mut i = all_off();
        i.ssr_enabled = true;
        i.bloom_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        assert!(
            !g.resources.iter().any(|r| r.label == "scene_color"),
            "scene_color is the pre-TAA scene here; it must not be declared twice"
        );
        // Bloom and Composite consume the SsrResolve output instead.
        let pre_taa = resource_of(&g, "scene_pre_taa");
        let bloom = g
            .passes
            .iter()
            .find(|p| p.id == PassId::Bloom)
            .expect("bloom present");
        assert!(
            bloom.reads.iter().any(|r| r.resource_index() == pre_taa),
            "Bloom reads the resolve output directly"
        );
        i.taa_enabled = true;
        i.velocity_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        assert!(g.resources.iter().any(|r| r.label == "scene_color"));
    }

    #[test]
    fn glass_without_a_reflection_resolve_extends_the_hdr_chain() {
        // Transparent blends into whichever texture carries the pre-TAA scene.
        // With no resolve that is hdr_resolve, so its write must bump the
        // hdr_resolve version rather than branch a second resource -- otherwise
        // the glass blend and the decoration chain order independently over one
        // object.
        let mut i = all_off();
        i.transparent_enabled = true;
        i.decals_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let hdr = resource_of(&g, "hdr_resolve");
        let trans = g
            .passes
            .iter()
            .find(|p| p.id == PassId::Transparent)
            .expect("transparent present");
        let write = trans
            .writes
            .iter()
            .find(|w| w.resource_index() == hdr)
            .expect("Transparent writes hdr_resolve");
        // It RMWs: the version it produces is one past the decoration chain's.
        let read = trans
            .reads
            .iter()
            .find(|r| r.resource_index() == hdr)
            .expect("Transparent reads hdr_resolve");
        assert_eq!(
            write.version(),
            read.version() + 1,
            "the glass blend extends the chain it read"
        );
        // Composite sees the blended version, not the pre-glass one.
        let composite = g
            .passes
            .iter()
            .find(|p| p.id == PassId::Composite)
            .expect("composite present");
        assert!(
            composite
                .reads
                .iter()
                .any(|r| r.resource_index() == hdr && r.version() == write.version()),
            "Composite reads the post-glass version"
        );
    }

    #[test]
    fn transparent_off_means_no_slot() {
        // The pass is omitted when nothing in the world is transparent:
        // no orphan slot, no executor stub triggered.
        let mut i = all_off();
        i.ssr_enabled = true;
        i.taa_enabled = true;
        i.velocity_enabled = true;
        // transparent_enabled left at false.
        let g = build_frame_graph(&i).expect("compiles");

        assert!(
            !g.passes
                .iter()
                .map(|p| p.id)
                .any(|x| x == PassId::Transparent)
        );
    }

    #[test]
    fn lines_close_the_hdr_decoration_chain() {
        // The node RMWs hdr_resolve last, so the lines draw over the lit +
        // decorated scene and SSR / TAA then consume them like any other
        // scene content.
        let mut i = all_off();
        i.decals_enabled = true;
        i.particles_enabled = true;
        i.ssr_enabled = true;
        i.lines_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::Decals) < pos(PassId::Lines));
        assert!(pos(PassId::ParticlesDraw) < pos(PassId::Lines));
        assert!(pos(PassId::Lines) < pos(PassId::SsrResolve));
    }

    #[test]
    fn lines_off_means_no_slot() {
        // A frame that published no lines omits the pass entirely.
        let g = build_frame_graph(&all_off()).expect("compiles");
        assert!(!g.passes.iter().any(|p| p.id == PassId::Lines));
    }

    #[test]
    fn a_hidden_world_drops_the_lines() {
        // Behind an opaque menu backdrop nothing of the world is visible, so
        // the masked graph drops the lines with every other world pass.
        let mut i = all_off();
        i.lines_enabled = true;
        i.world_hidden = true;
        let g = build_frame_graph(&i).expect("compiles");
        assert!(!g.passes.iter().any(|p| p.id == PassId::Lines));
    }

    #[test]
    fn ssgi_off_means_no_slot() {
        // IBL-only indirect lighting: the pass is omitted entirely.
        let g = build_frame_graph(&all_off()).expect("compiles");

        assert!(!g.passes.iter().map(|p| p.id).any(|x| x == PassId::Ssgi));
    }

    #[test]
    fn rt_reflections_off_means_no_slot() {
        // No ray tracing requested: the pass is omitted entirely.
        let g = build_frame_graph(&all_off()).expect("compiles");

        assert!(
            !g.passes
                .iter()
                .map(|p| p.id)
                .any(|x| x == PassId::RtReflections)
        );
    }

    #[test]
    fn rt_reflections_occupy_the_ssr_resolve_slot() {
        // RtReflections reads the post-decoration hdr_resolve and writes
        // scene_pre_taa, exactly where SsrResolve would, so it orders after
        // ParticlesDraw and before TaaResolve.
        let mut i = all_off();
        i.rt_reflections_enabled = true;
        i.particles_enabled = true;
        i.taa_enabled = true;
        i.velocity_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(order.contains(&PassId::RtReflections));
        assert!(pos(PassId::ParticlesDraw) < pos(PassId::RtReflections));
        assert!(pos(PassId::RtReflections) < pos(PassId::TaaResolve));
    }

    #[test]
    fn rt_reflections_take_precedence_over_ssr_resolve() {
        // RT alone inserts RtReflections, not SsrResolve.
        let mut i = all_off();
        i.rt_reflections_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::RtReflections));
        assert!(!order.contains(&PassId::SsrResolve));

        // With both flags set (RT available + SSR fallback authored), hardware
        // RT wins and SsrResolve is omitted; never two in the same slot.
        let mut both = all_off();
        both.ssr_enabled = true;
        both.rt_reflections_enabled = true;
        let g = build_frame_graph(&both).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::RtReflections));
        assert!(!order.contains(&PassId::SsrResolve));
    }

    #[test]
    fn ssgi_reads_the_lit_scene_before_decals_and_writes_none_of_it() {
        // AutoExposure and SSGI both read hdr_resolve_v1; SSGI keeps what it
        // gathers in its own targets, so Decals RMWs to v2 after both.
        let mut i = all_off();
        i.auto_exposure_enabled = true;
        i.ssgi_enabled = true;
        i.decals_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::Main) < pos(PassId::Ssgi));
        assert!(pos(PassId::Ssgi) < pos(PassId::Decals));
        assert!(pos(PassId::AutoExposure) < pos(PassId::Decals));
        let ssgi = &g.passes[pos(PassId::Ssgi)];
        assert!(ssgi.writes.is_empty());
        let main = &g.passes[pos(PassId::Main)];
        let main_resolve = main.writes.last().expect("hdr_resolve");
        assert!(
            ssgi.reads
                .iter()
                .any(|r| r.resource_index() == main_resolve.resource_index()
                    && r.version() == main_resolve.version())
        );
        let decals = &g.passes[pos(PassId::Decals)];
        assert_eq!(decals.writes[0].version(), 2);
    }

    #[test]
    fn the_ssgi_resolve_precedes_main_which_reads_it() {
        let mut i = all_off();
        i.gbuffer_prepass_enabled = true;
        i.ssr_prepass_enabled = true;
        i.velocity_enabled = true;
        i.ssao_enabled = true;
        i.ssgi_enabled = true;
        i.ssgi_resolve_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::GBufferPrepass) < pos(PassId::SsgiResolve));
        assert!(pos(PassId::SsaoBlur) < pos(PassId::SsgiResolve));
        assert!(pos(PassId::SsgiResolve) < pos(PassId::Main));
        let resolve = &g.passes[pos(PassId::SsgiResolve)];
        let [indirect] = resolve.writes[..] else {
            panic!("the resolve writes its one output");
        };
        let main = &g.passes[pos(PassId::Main)];
        assert!(main.reads.iter().any(|r| r.resource_version() == indirect));
        // It reads the occlusion it folds in.
        let ssao = &g.passes[pos(PassId::SsaoBlur)];
        assert!(
            resolve
                .reads
                .iter()
                .any(|r| r.resource_version() == ssao.writes[0])
        );
    }

    // The mark is the mask's first writer, ahead of Main like the other
    // pre-pass consumers, and only an upscaler that reads the mask brings it.
    #[test]
    fn the_disocclusion_mark_writes_the_mask_ahead_of_main_and_the_particles() {
        let mut i = all_off();
        i.gbuffer_prepass_enabled = true;
        i.velocity_enabled = true;
        i.reactive_mask_enabled = true;
        i.upscale_enabled = true;
        i.upscale_reads_reactive = true;
        i.disocclusion_enabled = true;
        i.particles_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::GBufferPrepass) < pos(PassId::Disocclusion));
        assert!(pos(PassId::Disocclusion) < pos(PassId::Main));
        assert!(pos(PassId::Disocclusion) < pos(PassId::ParticlesDraw));
        let mark = &g.passes[pos(PassId::Disocclusion)];
        let [mask] = mark.writes[..] else {
            panic!("the mark writes the mask alone");
        };
        assert_eq!(mask.version(), 1);
        let particles = &g.passes[pos(PassId::ParticlesDraw)];
        assert!(
            particles
                .writes
                .iter()
                .any(|w| w.resource_index() == mask.resource_index() && w.version() == 2)
        );

        i.upscale_reads_reactive = false;
        let g = build_frame_graph(&i).expect("compiles");
        assert!(g.pass_index(PassId::Disocclusion).is_none());
    }

    #[test]
    fn ssgi_after_raymarch() {
        // With both on, SSGI reads the post-raymarch scene: Raymarch v1→v2,
        // SSGI reads v2, SsrResolve reads v2 too.
        let mut i = all_off();
        i.raymarch_enabled = true;
        i.ssgi_enabled = true;
        i.ssr_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::Raymarch) < pos(PassId::Ssgi));
        let raymarch = &g.passes[pos(PassId::Raymarch)];
        let ssgi = &g.passes[pos(PassId::Ssgi)];
        assert!(
            ssgi.reads
                .iter()
                .any(|r| r.resource_version() == raymarch.writes[0])
        );
    }

    #[test]
    fn ssgi_reads_the_prepass_normals_and_motion() {
        // The accumulation reprojects through the pre-pass's motion vectors, so
        // SSGI reads both of the pre-pass channels it uses, which also pins it
        // after the pre-pass.
        let mut i = all_off();
        i.gbuffer_prepass_enabled = true;
        i.ssr_prepass_enabled = true;
        i.velocity_enabled = true;
        i.ssgi_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        let prepass = &g.passes[pos(PassId::GBufferPrepass)];
        let ssgi = &g.passes[pos(PassId::Ssgi)];
        let [normal_depth, roughness, velocity, _depth] = prepass.writes[..] else {
            panic!("the pre-pass writes four channels");
        };
        for channel in [normal_depth, roughness, velocity] {
            assert!(
                ssgi.reads
                    .iter()
                    .any(|r| r.resource_index() == channel.resource_index())
            );
        }
        assert!(pos(PassId::GBufferPrepass) < pos(PassId::Ssgi));
    }

    #[test]
    fn raymarch_off_means_no_slot() {
        // No `SdfVolume` in the world: pass is omitted, no executor stub
        // ever fires.
        let g = build_frame_graph(&all_off()).expect("compiles");

        assert!(!g.passes.iter().map(|p| p.id).any(|x| x == PassId::Raymarch));
    }

    #[test]
    fn raymarch_pinned_between_auto_exposure_and_decals() {
        // AutoExposure reads hdr_resolve_v1 (WAR); Raymarch RMWs to v2;
        // Decals RMWs to v3. The toposort orders the three through the
        // version chain without needing declaration-order tie-breaks.
        let mut i = all_off();
        i.auto_exposure_enabled = true;
        i.raymarch_enabled = true;
        i.decals_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::Main,
                PassId::AutoExposure,
                PassId::Raymarch,
                PassId::Decals,
                PassId::Composite,
            ]
        );
        // Version chain on hdr_resolve walks 1 → 2 → 3.
        let raymarch = &g.passes[2];
        assert_eq!(raymarch.writes[0].version(), 2);
        let decals = &g.passes[3];
        assert_eq!(decals.writes[0].version(), 3);
    }

    // The atmosphere's tables are built before Main draws the sky and lights
    // with them, and the haze lands on the decorated scene under the fog.
    #[test]
    fn the_atmosphere_feeds_main_and_hazes_under_the_fog() {
        let mut i = all_off();
        i.atmosphere_enabled = true;
        i.decals_enabled = true;
        i.fog_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let at = |p| order.iter().position(|x| *x == p).expect("pass present");
        assert!(at(PassId::Atmosphere) < at(PassId::Main));
        assert!(at(PassId::Decals) < at(PassId::AerialPerspective));
        assert!(at(PassId::AerialPerspective) < at(PassId::Fog));
        let off = build_frame_graph(&all_off()).expect("compiles");
        assert!(
            off.passes
                .iter()
                .all(|p| p.id != PassId::Atmosphere && p.id != PassId::AerialPerspective)
        );
    }

    // The ocean is simulated ahead of the transparent pass, which reads its
    // arrays in the water's vertex and fragment stages, with or without a
    // reflection resolve; with no transparent pass there is nothing to feed.
    #[test]
    fn the_ocean_simulation_feeds_the_water_draw() {
        for ssr in [false, true] {
            let mut i = all_off();
            i.transparent_enabled = true;
            i.water_sim_enabled = true;
            i.ssr_enabled = ssr;
            let g = build_frame_graph(&i).expect("compiles");
            let at = |id| g.pass_index(id).expect("present");
            assert!(at(PassId::WaterSim) < at(PassId::Transparent));
            let read = g.passes[at(PassId::Transparent)]
                .reads
                .iter()
                .find(|r| g.resources[r.resource_index()].label == "water_cascades")
                .expect("the water draw declares the cascade read");
            assert_eq!(read.stage(), ReadStages::VERTEX | ReadStages::FRAGMENT);
        }
        let mut i = all_off();
        i.water_sim_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        assert!(g.pass_index(PassId::WaterSim).is_none());
    }

    #[test]
    fn raymarch_works_without_auto_exposure_or_decals() {
        // Standalone Raymarch RMWs hdr_resolve_v1 → v2; SsrResolve reads
        // v2 instead of v1. Nothing else in the post chain.
        let mut i = all_off();
        i.raymarch_enabled = true;
        i.ssr_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::Main,
                PassId::Raymarch,
                PassId::SsrResolve,
                PassId::Composite,
            ]
        );
        let raymarch = &g.passes[1];
        assert_eq!(raymarch.writes[0].version(), 2);
        let ssr = &g.passes[2];
        assert_eq!(ssr.reads[0].version(), 2);
    }

    #[test]
    fn full_graph_orders_all_passes_correctly() {
        // Everything on: every pass shows up in the expected order.
        // This is the showcase configuration.
        let mut i = all_off();
        i.shadow_enabled = true;
        i.bindless_cull_enabled = true;
        i.auto_exposure_enabled = true;
        i.bloom_enabled = true;
        i.velocity_enabled = true;
        i.taa_enabled = true;
        i.ssr_enabled = true;
        i.particles_enabled = true;
        i.fog_enabled = true;
        i.decals_enabled = true;
        i.ssr_prepass_enabled = true;
        i.ssao_enabled = true;
        i.gbuffer_prepass_enabled = true;
        i.transparent_enabled = true;
        i.raymarch_enabled = true;

        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();

        // Spot-check relative ordering rather than the exact list: with
        // many independent passes the toposort has flexibility on
        // tie-breaks.
        fn idx(order: &[PassId], p: PassId) -> usize {
            order.iter().position(|x| *x == p).expect("pass present")
        }
        // Cull / GBufferPrepass / SsaoBlur / Shadow / SSAO all precede Main.
        assert!(idx(&order, PassId::Cull) < idx(&order, PassId::Main));
        assert!(idx(&order, PassId::GBufferPrepass) < idx(&order, PassId::Main));
        assert!(idx(&order, PassId::SsaoBlur) < idx(&order, PassId::Main));
        assert!(idx(&order, PassId::Shadow) < idx(&order, PassId::Main));
        // The pre-pass precedes SsaoBlur (SSAO reads its normal+depth).
        assert!(idx(&order, PassId::GBufferPrepass) < idx(&order, PassId::SsaoBlur));
        // AutoExposure post-Main, pre-Raymarch (WAR-pinned on hdr_resolve_v1).
        assert!(idx(&order, PassId::Main) < idx(&order, PassId::AutoExposure));
        assert!(idx(&order, PassId::AutoExposure) < idx(&order, PassId::Raymarch));
        // Raymarch leads the hdr_resolve RMW chain so Decals / Fog /
        // ParticlesDraw blend on top of the raymarched color.
        assert!(idx(&order, PassId::Raymarch) < idx(&order, PassId::Decals));
        // hdr_resolve chain.
        assert!(idx(&order, PassId::Decals) < idx(&order, PassId::Fog));
        assert!(idx(&order, PassId::Fog) < idx(&order, PassId::ParticlesDraw));
        assert!(idx(&order, PassId::ParticlesDraw) < idx(&order, PassId::SsrResolve));
        // FogFroxel populates the volume Fog samples, so it must precede Fog.
        assert!(idx(&order, PassId::FogFroxel) < idx(&order, PassId::Fog));
        // The pre-pass that writes the motion vectors precedes TaaResolve.
        assert!(idx(&order, PassId::GBufferPrepass) < idx(&order, PassId::TaaResolve));
        // Post-TAA chain. Transparent slots between SsrResolve and TaaResolve.
        assert!(idx(&order, PassId::SsrResolve) < idx(&order, PassId::Transparent));
        assert!(idx(&order, PassId::Transparent) < idx(&order, PassId::TaaResolve));
        assert!(idx(&order, PassId::TaaResolve) < idx(&order, PassId::Bloom));
        assert!(idx(&order, PassId::Bloom) < idx(&order, PassId::Composite));
        // Composite is the presenter and runs last.
        assert_eq!(order.last(), Some(&PassId::Composite));
        assert!(g.passes.last().unwrap().presents);
    }

    #[test]
    fn shadow_orders_after_cull_through_the_draw_args_read() {
        // The GPU-driven cascade path draws from an indirect command buffer the
        // cull pass fills, so Shadow has to depend on Cull. Without the edge a
        // backend that runs Cull on the async queue has nothing ordering the two.
        let mut i = all_off();
        i.shadow_enabled = true;
        i.bindless_cull_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let cull = g.pass_index(PassId::Cull).expect("Cull is in the graph");
        let shadow = g
            .pass_index(PassId::Shadow)
            .expect("Shadow is in the graph");
        assert!(
            g.depends_on(cull, shadow),
            "Shadow does not depend on Cull: {:?}",
            g.passes.iter().map(|p| p.id).collect::<Vec<_>>()
        );
        assert!(
            g.pass_precedes(cull, shadow),
            "the schedule does not order them"
        );
    }

    #[test]
    fn spot_shadow_orders_after_cull_through_the_draw_args_read() {
        let mut i = all_off();
        i.shadowed_spot_count = 3;
        i.spot_shadow_slice_size = 512;
        i.bindless_cull_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let cull = g.pass_index(PassId::Cull).expect("Cull is in the graph");
        let spot = g
            .pass_index(PassId::SpotShadow)
            .expect("SpotShadow is in the graph");
        assert!(
            g.depends_on(cull, spot),
            "SpotShadow does not depend on Cull"
        );
        assert!(
            g.pass_precedes(cull, spot),
            "the schedule does not order them"
        );
    }

    #[test]
    fn shadow_without_bindless_cull_declares_no_cull_read() {
        // A world drawing its cascades per-object has no cull output to read, so
        // the edge must not appear and Shadow stays the graph's first pass.
        let mut i = all_off();
        i.shadow_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let shadow = g
            .pass_index(PassId::Shadow)
            .expect("Shadow is in the graph");
        assert!(g.passes[shadow].reads.is_empty());
    }

    // The passes that read or write `label`, in compiled order, with whether
    // each writes it.
    fn touches(g: &CompiledGraph, label: &str) -> Vec<(PassId, bool)> {
        let Some(index) = g.resources.iter().position(|r| r.label == label) else {
            return Vec::new();
        };
        g.passes
            .iter()
            .filter_map(|p| {
                let writes = p.writes.iter().any(|w| w.resource_index() == index);
                let reads = p.reads.iter().any(|r| r.resource_index() == index);
                (writes || reads).then_some((p.id, writes))
            })
            .collect()
    }

    fn reactive_writers_and_taa() -> FrameGraphInputs {
        let mut i = all_off();
        i.reactive_mask_enabled = true;
        i.particles_enabled = true;
        i.transparent_enabled = true;
        i.taa_enabled = true;
        i.velocity_enabled = true;
        i
    }

    #[test]
    fn the_reactive_mask_runs_from_its_writers_to_taa() {
        let g = build_frame_graph(&reactive_writers_and_taa()).expect("compiles");
        assert_eq!(
            touches(&g, "reactive_mask"),
            [
                (PassId::ParticlesDraw, true),
                (PassId::Transparent, true),
                (PassId::TaaResolve, false),
            ]
        );
    }

    #[test]
    fn the_upscaler_reads_the_reactive_mask_in_place_of_taa() {
        let mut i = reactive_writers_and_taa();
        i.particles_enabled = false;
        i.upscale_enabled = true;
        i.upscale_reads_reactive = true;
        let g = build_frame_graph(&i).expect("compiles");
        assert_eq!(
            touches(&g, "reactive_mask"),
            [(PassId::Transparent, true), (PassId::Upscale, false)]
        );
    }

    // An upscaler with no reactive input leaves the mask out of the frame.
    #[test]
    fn an_upscaler_that_takes_no_mask_declares_none() {
        let mut i = reactive_writers_and_taa();
        i.upscale_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        assert!(touches(&g, "reactive_mask").is_empty());
    }

    // A frame with no reader, or no writer, declares no mask at all, so it
    // costs nothing there.
    #[test]
    fn the_reactive_mask_needs_a_writer_and_a_reader() {
        let mut no_reader = reactive_writers_and_taa();
        no_reader.taa_enabled = false;
        let mut no_writer = reactive_writers_and_taa();
        no_writer.particles_enabled = false;
        no_writer.transparent_enabled = false;
        let mut not_built = reactive_writers_and_taa();
        not_built.reactive_mask_enabled = false;
        for i in [no_reader, no_writer, not_built] {
            assert!(!i.reactive_mask_live());
            let g = build_frame_graph(&i).expect("compiles");
            assert!(touches(&g, "reactive_mask").is_empty());
        }
        assert!(reactive_writers_and_taa().reactive_mask_live());
    }

    // The reactive view reads the mask itself, so it needs no temporal pass.
    #[test]
    fn the_reactive_view_reads_the_mask_without_a_temporal_pass() {
        let mut i = reactive_writers_and_taa();
        i.taa_enabled = false;
        i.velocity_enabled = false;
        i.composite_reads_reactive = true;
        let g = build_frame_graph(&i).expect("compiles");
        assert_eq!(
            touches(&g, "reactive_mask"),
            [
                (PassId::ParticlesDraw, true),
                (PassId::Transparent, true),
                (PassId::Composite, false),
            ]
        );
    }
}

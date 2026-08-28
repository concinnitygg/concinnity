// src/render_graph/frame.rs
//
// Per-frame graph builder. On Metal, every pass that ran inline
// through `draw_frame` now dispatches through a single
// `build_frame_graph()` → `execute_graph()` pair.
//
// The frame builder declares conditional passes based on the
// `FrameGraphInputs` struct (one bool per gated pass). Read / write
// declarations on each pass let the compile pass derive:
//
//   * Execution order (toposort over RAW / WAW / WAR edges, ties broken
//     by declaration order).
//   * Per-pass barriers (`pass.barriers_before` per resource state
//     transition). Metal mostly ignores these (Apple GPUs handle most
//     hazards implicitly); the Vulkan / DirectX executors emit
//     `vkCmdPipelineBarrier` / `D3D12_RESOURCE_BARRIER` from them.
//   * Transient resource lifetimes (`PassRange` per resource), the
//     aliasing input.
//
// Resources split into two origins. `import_texture` = engine-owned: the
// resource outlives the frame (the cross-frame shadow map, the TAA history
// `scene_color`, the froxel volume, the cross-frame Hi-Z pyramid) and the
// backend always owns its GPU object.
//
// A resource is declared only where a pass actually writes it. Several engine
// bindings point two names at one texture depending on configuration --
// `scene_pre_taa` is `hdr_resolve` without a reflection resolve, `scene_color`
// is the pre-TAA scene without TAA, `hdr_color` is the resolve target without
// MSAA -- and declaring the second name anyway would give one GPU object two
// independent barrier timelines. The builder threads the upstream handle
// through instead, so one texture is always one resource. `create_texture` =
// transient: single-frame intermediates (hdr intermediates excepted)
// the aliasing planner ([`super::alias`]) may pack into shared physical memory,
// since their `[first, last]` lifetimes are disjoint. In practice only
// `ao_output` and `bloom_top` are independently poolable today; the other
// `create_texture` intermediates fold into the long-lived gbuffer MRT
// (`velocity`, `ssr_gbuffer`) or are themselves long-lived (`gbuffer`), so a
// backend pool leaves them backend-owned. The planner sizes each transient
// from its desc; the origin marks aliasing candidacy and has no effect on pass
// order or barriers.

use crate::render_types::NUM_SHADOW_CASCADES;

use super::{
    BufferDesc, BufferUsage, CompiledGraph, GraphBuilder, GraphError, PassId, PassKind,
    PixelFormat, TextureDesc, TextureHandle, TextureSize, TextureUsage, full_mip_levels,
};

/// Per-frame inputs that gate conditional passes. Built by `draw_frame`
/// from the live `MtlContext` state and consumed by `build_frame_graph`
/// so the conditional-inclusion decisions made here match what the
/// executor will dispatch.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
    /// MSAA sample count of the HDR colour + depth attachments, typically
    /// 4. The resolve target is single-sample regardless.
    pub hdr_sample_count: u32,
    /// `true` when GPU-driven cull is going to run this frame, i.e. the
    /// bindless static path is configured AND there is geometry to cull
    /// AND the per-frame `object_buffer` / `draw_args` buffers built. The
    /// graph adds the Cull compute pass and the Main read-edge from
    /// `draw_args` only when this is on; otherwise Main draws via the
    /// legacy per-draw path with no graph dependency on the cull output.
    pub bindless_cull_enabled: bool,
    /// `true` when the auto-exposure compute pipelines are built (i.e.
    /// the world declared `PostProcessConfig.auto_exposure`). The graph
    /// appends an `AutoExposure` compute pass that reads the Main pass's
    /// `hdr_resolve_v1` (pre-decoration) and writes the histogram +
    /// readback buffer. The compile pass's WAR step pins AutoExposure
    /// before the first hdr_resolve post-Main writer (Decals or Fog or
    /// ParticlesDraw) so AutoExposure samples the un-decorated scene.
    pub auto_exposure_enabled: bool,
    /// `true` when `PostProcessConfig.bloom_intensity > 0.0`. The graph
    /// adds a `Bloom` pass that thresholds / downsamples / upsamples the
    /// post-TAA scene into the bloom mip chain; Composite reads the
    /// bloom output so the toposort orders Bloom before Composite.
    pub bloom_enabled: bool,
    /// `true` when TAA is on (the velocity pre-pass only runs as part of
    /// the TAA stack). The graph adds a `Velocity` render pass that
    /// writes the per-pixel motion-vector buffer TaaResolve consumes;
    /// TaaResolve declares the read so Velocity → TaaResolve is explicit.
    pub velocity_enabled: bool,
    /// `true` when TAA is on. The graph adds a `TaaResolve` render pass
    /// that reads the pre-TAA scene (SSR resolve output or hdr_resolve)
    /// and writes the imported `scene_color` Bloom + Composite consume.
    pub taa_enabled: bool,
    /// `true` when SSR is on. The graph adds an `SsrResolve` render pass
    /// that reads the post-decoration `hdr_resolve` and writes the
    /// imported `scene_pre_taa` texture, which only exists when this or
    /// `rt_reflections_enabled` is set. When TAA is also on, TaaResolve
    /// reads the post-SsrResolve version; with TAA off, Bloom +
    /// Composite read that version directly.
    pub ssr_enabled: bool,
    /// `true` when the particle system is going to run this frame:
    /// `particle_pipelines` built AND at least one live emitter. The
    /// graph adds a `ParticlesDraw` render pass that blend-writes
    /// `hdr_resolve`. The bundled ParticlesSim compute sub-pass runs
    /// inside the same `encode_particles` call so it keeps its per-pass
    /// timing slot without needing its own graph node.
    pub particles_enabled: bool,
    /// `true` when a `VolumetricFog` is in the world. The graph adds a
    /// `Fog` render pass between Decals and ParticlesDraw on the
    /// hdr_resolve RMW chain.
    pub fog_enabled: bool,
    /// `true` when at least one `Decal` is in the world AND the decal
    /// pipeline is built. The graph adds a `Decals` render pass at the
    /// head of the hdr_resolve post-Main RMW chain.
    pub decals_enabled: bool,
    /// `true` when the SSR pre-pass should run; matches
    /// `self.ssr_settings.is_some()`. The graph adds an `SsrPrepass`
    /// render pass that writes the imported `ssr_gbuffer` texture;
    /// SsaoBlur reads it when SSAO is also on (G-buffer sharing).
    pub ssr_prepass_enabled: bool,
    /// `true` when SSAO should run; matches
    /// `self.ssao_settings.is_some()`. The graph adds an `SsaoBlur`
    /// render pass that dispatches the bundled `encode_ssao` (which
    /// internally encodes SsaoPrepass + SsaoKernel + SsaoBlur). SsaoBlur
    /// writes `ao_output`; Main reads it. SsaoPrepass + SsaoKernel
    /// stay as timing-only PassIds (same pattern as ParticlesSim).
    pub ssao_enabled: bool,
    /// `true` when temporal upscaling is on (e.g. MetalFX on Metal). The
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
    /// reads the latest scene-pre-taa colour + main depth and
    /// alpha-blends translucent geometry back-to-front into the same
    /// target. The pass aggregates N draws, each owns its own
    /// pipeline + descriptor set, the executor receives the sorted list
    /// at encode time.
    pub transparent_enabled: bool,
    /// `true` when a system submitted world-space lines this frame AND the
    /// backend's line pipeline is live. The graph adds a `Lines` render pass at
    /// the tail of the hdr_resolve RMW chain: it blend-writes the scene colour
    /// and samples the resolved scene depth so a line behind geometry is
    /// occluded by it. A frame with no lines omits the node entirely.
    pub lines_enabled: bool,
    /// `true` when at least one visible `SdfVolume` is in the world AND
    /// the backend's raymarch pipeline is live. The graph adds a
    /// `Raymarch` render pass between `AutoExposure` and `Decals` on the
    /// hdr_resolve RMW chain: it reads the head of the chain (so
    /// AutoExposure samples the pre-raymarch scene) and writes the next
    /// version that Decals then bumps further. The pass also RMWs the
    /// main depth attachment so subsequent passes see raymarched
    /// surfaces' depth, and that read-modify-write is declared, which is
    /// what makes the post-Raymarch depth version the one every later
    /// decoration pass samples.
    pub raymarch_enabled: bool,
    /// `true` when two-pass Hi-Z occlusion culling is requested
    /// (`PostProcessConfig.occlusion_two_pass`) AND the bindless GPU-cull
    /// path is active this frame. Only meaningful alongside
    /// `bindless_cull_enabled`; the builder ANDs the two so a world that
    /// asks for two-pass without a bindless shader simply gets the
    /// single-pass path. When on, the graph inserts `HizBuild` → `Cull2`
    /// → `Main2` between `Main` and the post-decoration chain: `HizBuild`
    /// rebuilds the Hi-Z pyramid from phase-1 depth, `Cull2` re-tests the
    /// objects phase-1 cull marked occluded, and `Main2` redraws the
    /// disoccluded survivors. `Main2`'s hdr_resolve write becomes the head
    /// of the post chain so AutoExposure / Decals / Fog / SSR see the
    /// combined two-pass result.
    pub two_pass_occlusion_enabled: bool,
    /// `true` when screen-space global illumination is on
    /// (`PostProcessConfig.indirect_lighting == "ssgi"`); matches
    /// `self.ssgi_settings.is_some()`. The graph inserts an `Ssgi` render pass
    /// on the hdr_resolve RMW chain right after `Raymarch` and before `Decals`:
    /// it reads the head of the chain (the lit scene, its bounce-radiance
    /// source) and writes the next version with the gathered indirect term
    /// additively composited in. SSGI reuses the SSR pre-pass G-buffer for
    /// normals + depth, so `ssr_prepass_enabled` is forced on whenever this is
    /// set.
    pub ssgi_enabled: bool,
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
    /// `true` to collapse the SSR / SSAO / velocity geometry pre-passes into a
    /// single `GBufferPrepass` node that writes view-space normal+depth,
    /// roughness, and motion in one traversal: every consumer reads that one
    /// output. When set, the builder emits `GBufferPrepass` (gated on any of
    /// `ssr_prepass_enabled || ssao_enabled || velocity_enabled`) instead of the
    /// separate `SsrPrepass` + `Velocity` nodes.
    pub unified_gbuffer_prepass: bool,
    /// `true` when an opaque full-screen menu backdrop covers the scene, so
    /// nothing the world passes produce is visible. The builder masks every
    /// gated world pass off and collapses the graph to `Main -> Composite`
    /// (Composite still presents the menu overlay). The backend pairs this with
    /// an empty visible set so the surviving Main pass is a bare clear; the
    /// opaque overlay then covers it.
    pub world_hidden: bool,
    /// `true` when the scene has local lights to cluster. The graph adds a
    /// `LightCull` compute pass before Main that bins the lights into per-cluster
    /// lists Main reads (RAW edge). A backend with no light-cull pipeline keeps
    /// this false and iterates the local lights directly.
    pub clustered_lighting_enabled: bool,
    /// `true` when the composite samples the SSAO output directly (the
    /// occlusion view mode). Declares a Composite read of `ao_output`, so the
    /// pool-aliased transient stays live to the end of the frame instead of
    /// dying after Main. No effect while `ssao_enabled` is false.
    pub composite_reads_ao: bool,
    /// Number of spot shadow map slices to render, i.e. how many spot lights cast
    /// shadows. Zero skips the SpotShadow pass and its imported array entirely.
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
}

impl FrameGraphInputs {
    // Every gated pass off, at a representative resolution. A neutral base a
    // caller can flip individual flags on, e.g. to plan a worst-case graph for
    // transient-memory allocation (where the allocation must cover every
    // per-frame graph, not just the current frame's active passes).
    pub(crate) fn all_off() -> Self {
        FrameGraphInputs {
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
            fog_enabled: false,
            decals_enabled: false,
            ssr_prepass_enabled: false,
            ssao_enabled: false,
            upscale_enabled: false,
            transparent_enabled: false,
            lines_enabled: false,
            raymarch_enabled: false,
            two_pass_occlusion_enabled: false,
            ssgi_enabled: false,
            rt_reflections_enabled: false,
            unified_gbuffer_prepass: false,
            world_hidden: false,
            clustered_lighting_enabled: false,
            composite_reads_ao: false,
            shadowed_spot_count: 0,
            spot_shadow_slice_size: 512,
            hiz_build_enabled: false,
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
    ("fog", |i| i.fog_enabled = true),
    ("decals", |i| i.decals_enabled = true),
    ("ssr_prepass", |i| i.ssr_prepass_enabled = true),
    ("ssao", |i| i.ssao_enabled = true),
    ("upscale", |i| i.upscale_enabled = true),
    ("transparent", |i| i.transparent_enabled = true),
    ("lines", |i| i.lines_enabled = true),
    ("raymarch", |i| i.raymarch_enabled = true),
    ("two_pass_occlusion", |i| {
        i.two_pass_occlusion_enabled = true
    }),
    ("ssgi", |i| i.ssgi_enabled = true),
    ("rt_reflections", |i| i.rt_reflections_enabled = true),
    ("unified_gbuffer", |i| i.unified_gbuffer_prepass = true),
    ("world_hidden", |i| i.world_hidden = true),
    ("clustered_lighting", |i| {
        i.clustered_lighting_enabled = true
    }),
    ("composite_reads_ao", |i| i.composite_reads_ao = true),
    ("shadowed_spots", |i| i.shadowed_spot_count = 2),
    ("hiz_build", |i| i.hiz_build_enabled = true),
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
// Cull → SsrPrepass → SsaoBlur → Shadow → Main → AutoExposure
//   → Raymarch → Velocity → Decals → Fog → ParticlesDraw → SsrResolve
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
// (three colour formats and a depth target) and so do their consumers, so one
// handle would give each of them the union of four lifetimes.
#[derive(Copy, Clone)]
struct GBufferHandles {
    normal_depth: TextureHandle,
    roughness: TextureHandle,
    velocity: TextureHandle,
    depth: TextureHandle,
}

/// Compile the frame graph for `inputs`: the pass list above, gated down to the
/// passes this frame actually runs.
pub fn build_frame_graph(inputs: &FrameGraphInputs) -> Result<CompiledGraph, GraphError> {
    // When an opaque menu backdrop hides the scene, every world pass is wasted:
    // nothing it produces is visible. Force every gated world pass off so the
    // graph collapses to the minimal `Main -> Composite` (Composite still
    // presents the overlay). Main survives as a bare clear because the backend
    // feeds it an empty visible set this frame; the opaque overlay covers it.
    let masked = if inputs.world_hidden {
        Some(FrameGraphInputs {
            shadow_enabled: false,
            bindless_cull_enabled: false,
            auto_exposure_enabled: false,
            bloom_enabled: false,
            velocity_enabled: false,
            taa_enabled: false,
            ssr_enabled: false,
            particles_enabled: false,
            fog_enabled: false,
            decals_enabled: false,
            ssr_prepass_enabled: false,
            ssao_enabled: false,
            upscale_enabled: false,
            transparent_enabled: false,
            lines_enabled: false,
            raymarch_enabled: false,
            two_pass_occlusion_enabled: false,
            ssgi_enabled: false,
            rt_reflections_enabled: false,
            clustered_lighting_enabled: false,
            shadowed_spot_count: 0,
            spot_shadow_slice_size: 512,
            ..*inputs
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
    // `hdr_color` is the multisample colour attachment, and it exists only when
    // the world is multisampled. Without MSAA there is no separate resolve step
    // and the single colour target *is* the spine, which every backend already
    // reflects (Vulkan leaves `color_images` empty; DirectX and Metal leave
    // their `resolve` field `None` and bind `color`). Declaring it
    // unconditionally would put two graph resources on one GPU object, and the
    // moment either became graph-driven they would transition it twice from
    // states it was no longer in.
    let hdr_color = (inputs.hdr_sample_count > 1)
        .then(|| b.import_texture("hdr_color", hdr_color_desc(inputs)));
    let hdr_depth = b.import_texture("hdr_depth", hdr_depth_desc(inputs));
    let hdr_resolve = b.import_texture("hdr_resolve", hdr_resolve_desc(inputs));

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

    // Cull (compute) writes the indirect-draw args buffer Main consumes
    // through executeCommandsInBuffer. Under two-pass occlusion it also
    // writes a per-object status buffer (drawn / hi-z-candidate / culled)
    // that `Cull2` reads to decide which phase-1-occluded objects to
    // re-test against the rebuilt pyramid.
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

    // Unified G-buffer pre-pass: one node writes the view-space normal+depth /
    // roughness / velocity / depth that SSR, SSAO, SSGI, RT, TAA, and the
    // upscaler read, replacing the separate SsrPrepass + Velocity nodes. Runs
    // when any of those consumers is on. Every backend takes this path when its
    // G-buffer targets are built; the separate nodes below are the fallback for
    // a build without them.
    let gbuffer_v1 = if inputs.unified_gbuffer_prepass
        && (inputs.ssr_prepass_enabled || inputs.ssao_enabled || inputs.velocity_enabled)
    {
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
        if let Some(h) = draw_args_v1 {
            gb.read_buffer(h);
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

    // SSR pre-pass writes the SSR G-buffer; SSAO reads it when both are on (the
    // shared-G-buffer fast path). Under the unified path the merged node above
    // supplies the same normal+depth handle, so this separate node is skipped.
    let ssr_gbuffer_v1 = if let Some(g) = gbuffer_v1 {
        Some(g.normal_depth)
    } else if inputs.ssr_prepass_enabled {
        let ssr_gbuffer = b.create_texture("ssr_gbuffer", ssr_gbuffer_desc(inputs));
        Some(
            b.add_pass(PassId::SsrPrepass, PassKind::Render)
                .write_texture(ssr_gbuffer),
        )
    } else {
        None
    };

    // SSAO bundle writes ao_output. PassId::SsaoBlur is the single
    // graph node for the entire encode_ssao bundle; SsaoPrepass +
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

    // Shadow optionally precedes Main and produces the shadow_map
    // handle Main samples. When off, Main does not declare a shadow_map
    // read, mirroring the encoder's `enable_shadows` shader path.
    let shadow_v1 = if inputs.shadow_enabled {
        let shadow_map = b.import_texture("shadow_map", shadow_map_desc(inputs.shadow_map_size));
        Some(
            b.add_pass(PassId::Shadow, PassKind::Render)
                .write_texture(shadow_map),
        )
    } else {
        None
    };

    // Clustered light binning (compute): bins the scene's local lights into
    // per-cluster index lists. Writes the imported cluster buffer; Main's read
    // below pins LightCull before Main in the toposort. Backend-owned buffer, so
    // the import is a dependency-tracking stub.
    let cluster_lights_v1 = if inputs.clustered_lighting_enabled {
        let cluster_lights = b.import_buffer("cluster_light_list", cluster_light_list_desc());
        Some(
            b.add_pass(PassId::LightCull, PassKind::Compute)
                .write_buffer(cluster_lights),
        )
    } else {
        None
    };

    // Spot shadows: one depth-only render per shadowed spot into its slice of
    // the spot shadow array. Like the cascade pass it precedes Main, which
    // samples the array; backend-owned, so the import tracks dependencies only.
    let spot_shadow_v1 = if inputs.shadowed_spot_count > 0 {
        let spot_map = b.import_texture(
            "spot_shadow_map",
            spot_shadow_map_desc(inputs.spot_shadow_slice_size, inputs.shadowed_spot_count),
        );
        Some(
            b.add_pass(PassId::SpotShadow, PassKind::Render)
                .write_texture(spot_map),
        )
    } else {
        None
    };

    // Main pass: reads optional shadow_map / spot_shadow_map / draw_args /
    // ao_output / cluster lights; writes the three HDR targets. Captures hdr_resolve_v1 (head of the
    // hdr_resolve RMW chain, the version AutoExposure reads when two-pass
    // is off) and hdr_depth_v1 (the depth HizBuild reduces under two-pass).
    let (hdr_resolve_v1, hdr_depth_v1) = {
        let mut main = b.add_pass(PassId::Main, PassKind::Render);
        if let Some(h) = shadow_v1 {
            main.read_texture(h);
        }
        if let Some(h) = spot_shadow_v1 {
            main.read_texture(h);
        }
        if let Some(h) = draw_args_v1 {
            main.read_buffer(h);
        }
        if let Some(h) = cluster_lights_v1 {
            main.read_buffer(h);
        }
        if let Some(h) = ao_output_v1 {
            main.read_texture(h);
        }
        if let Some(h) = hdr_color {
            let _ = main.write_texture(h);
        }
        let depth_v1 = main.write_texture(hdr_depth);
        let resolve_v1 = main.write_texture(hdr_resolve);
        (resolve_v1, depth_v1)
    };

    // Two-pass occlusion phase 2: rebuild the Hi-Z pyramid from phase-1
    // depth (HizBuild), re-test the objects phase-1 cull marked occluded
    // (Cull2), and redraw the disoccluded survivors (Main2). Main2 RMWs
    // hdr_color / hdr_depth / hdr_resolve, so its hdr_resolve write becomes
    // the head of the post-decoration chain: AutoExposure and every later
    // RMW pass see the combined phase-1 + phase-2 scene. Without two-pass
    // the head stays at Main's hdr_resolve_v1.
    let mut hiz_cur = hiz_pyramid;
    let mut depth_cur = hdr_depth_v1;
    let hdr_resolve_head = if let (true, Some(hiz)) = (two_pass, hiz_pyramid) {
        // HizBuild (compute): read phase-1 depth, write the Hi-Z pyramid.
        // The depth RAW edge pins it after Main; the pyramid write is a WAR
        // against Cull's read of the previous frame's contents.
        let mut hizb = b.add_pass(PassId::HizBuild, PassKind::Compute);
        hizb.read_texture(depth_cur);
        let hiz_v1 = hizb.write_texture(hiz);
        hiz_cur = Some(hiz_v1);

        // Cull2 (compute): read the rebuilt pyramid + the phase-1 status
        // buffer, write a second indirect-draw-args buffer Main2 consumes.
        let draw_args2 = b.import_buffer("draw_args2", draw_args_desc());
        let mut cull2 = b.add_pass(PassId::Cull2, PassKind::Compute);
        cull2.read_texture(hiz_v1);
        if let Some(cs) = cull_status_v1 {
            cull2.read_buffer(cs);
        }
        let draw_args2_v1 = cull2.write_buffer(draw_args2);

        // Main2 (render): read the phase-2 draw args; RMW hdr_color /
        // hdr_depth / hdr_resolve. The draw_args2 RAW edge pins it after
        // Cull2; the hdr_depth write (WAR vs HizBuild's read) pins it after
        // HizBuild; the hdr_color / hdr_resolve WAW edges pin it after Main.
        let mut main2 = b.add_pass(PassId::Main2, PassKind::Render);
        main2.read_buffer(draw_args2_v1);
        depth_cur = main2.write_texture(depth_cur);
        if let Some(h) = hdr_color {
            let _ = main2.write_texture(h);
        }
        main2.write_texture(hdr_resolve_v1)
    } else {
        hdr_resolve_v1
    };

    // AutoExposure (compute) reads the post-main scene (hdr_resolve_head:
    // Main2's output under two-pass, Main's otherwise). The compile pass's
    // WAR step pins it before the first hdr_resolve writer that bumps the
    // next version (Raymarch / Decals / Fog / ParticlesDraw), so
    // AutoExposure samples the un-decorated scene even though the GPU
    // texture object is the same one those passes later blend-write.
    if inputs.auto_exposure_enabled {
        b.add_pass(PassId::AutoExposure, PassKind::Compute)
            .read_texture(hdr_resolve_head);
    }

    // Velocity (render) writes the per-pixel motion-vector buffer TaaResolve /
    // Upscale consume. The read edge from those passes pins it ahead of them in
    // the toposort. Under the unified path the merged G-buffer node already
    // carries velocity, so TAA / Upscale read that handle and this separate node
    // is skipped.
    let velocity_v1 = if let Some(g) = gbuffer_v1 {
        Some(g.velocity)
    } else if inputs.velocity_enabled {
        let velocity = b.create_texture("velocity", velocity_desc(inputs));
        Some(
            b.add_pass(PassId::Velocity, PassKind::Render)
                .write_texture(velocity),
        )
    } else {
        None
    };

    // hdr_resolve post-Main RMW chain: Raymarch → Decals → Fog →
    // ParticlesDraw, each blend- or opaque-writing on top of the
    // previous version. The handle walks forward through `h` so each
    // write picks up the latest version, giving the compile pass clean
    // WAW edges to derive the chain order. Raymarch slots first so its
    // depth+colour write is visible to every later post-decoration
    // pass; AutoExposure's WAR-read on hdr_resolve_head pins it before
    // Raymarch (so SDF brightness doesn't skew exposure for the same
    // frame), matching the doc's chosen one-frame-lag trade-off.
    //
    // Main depth rides the same pattern: Raymarch sphere-traces against it and
    // writes the hit depth back, so it bumps the depth version, and every later
    // decoration samples that version rather than the one Main left.
    let mut h = hdr_resolve_head;
    if inputs.raymarch_enabled {
        let mut rm = b.add_pass(PassId::Raymarch, PassKind::Render);
        rm.read_texture(h);
        h = rm.write_texture(h);
        depth_cur = rm.write_texture(depth_cur);
    }
    // SSGI reads the lit scene (its bounce-radiance source) and RMWs the
    // gathered + denoised indirect term back in. Slots right after Raymarch so
    // it can bounce raymarched surfaces too, and before Decals / Fog /
    // Particles so those decorations layer on top of the indirect light.
    // AutoExposure's WAR-read on hdr_resolve_head pins it ahead of SSGI, so the
    // added bounce doesn't skew the same frame's exposure (the same one-frame
    // trade-off Raymarch documents).
    if inputs.ssgi_enabled {
        let mut ssgi = b.add_pass(PassId::Ssgi, PassKind::Render);
        ssgi.read_texture(h);
        // The gather is against the pre-pass view normal + linear depth; with
        // no G-buffer there is nothing to gather against and the encoder skips.
        if let Some(g) = ssr_gbuffer_v1 {
            ssgi.read_texture(g);
        }
        h = ssgi.write_texture(h);
    }
    if inputs.decals_enabled {
        // Projected decals reconstruct each pixel's world position from the
        // scene depth, so the pass samples depth while blend-writing colour.
        let mut decals = b.add_pass(PassId::Decals, PassKind::Render);
        decals.read_texture(depth_cur);
        h = decals.write_texture(h);
    }
    if inputs.fog_enabled {
        // FogFroxel (compute) populates the 3D scatter/transmittance
        // volume the Fog fragment shader samples. The post-write handle
        // (`froxel_v1`) is what Fog reads: that gives the compile pass
        // a clean RAW edge so FogFroxel runs before Fog in the toposort.
        // All three backends implement the froxel path; the Fog render
        // pass trilinear-samples the volume by (screen_uv, view_z).
        let froxel_v0 = b.import_texture("fog_froxel_volume", froxel_volume_desc(inputs));
        let mut froxel = b.add_pass(PassId::FogFroxel, PassKind::Compute);
        // Each slab does a cascade tap, so the kernel is a second reader of the
        // shadow map alongside Main. Declaring it puts the compute stage into the
        // read run's union, which is what makes one transition serve both.
        if let Some(h) = shadow_v1 {
            froxel.read_texture(h);
        }
        let froxel_v1 = froxel.write_texture(froxel_v0);
        let mut fog_pass = b.add_pass(PassId::Fog, PassKind::Render);
        fog_pass.read_texture(froxel_v1);
        // Scene depth bounds the ray march / froxel lookup per pixel.
        fog_pass.read_texture(depth_cur);
        h = fog_pass.write_texture(h);
    }
    if inputs.particles_enabled {
        h = b
            .add_pass(PassId::ParticlesDraw, PassKind::Render)
            .write_texture(h);
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
    let hdr_resolve_cur = h;

    // `scene_pre_taa` is a distinct texture only when a pass writes it:
    // SsrResolve / RtReflections produce it, and Transparent read-modify-writes
    // it. With neither resolve the engine binds the pre-TAA scene name straight
    // to `hdr_resolve`, so declaring it here would put two graph resources on
    // one GPU object -- each with its own barrier timeline over the same memory.
    // Threading the upstream handle expresses that binding with one resource.
    let scene_pre_taa_cur = if inputs.rt_reflections_enabled || inputs.ssr_enabled {
        let scene_pre_taa = b.import_texture("scene_pre_taa", scene_color_desc(inputs));
        // SsrResolve and RtReflections occupy the same slot: both read the
        // post-decoration hdr_resolve and write scene_pre_taa. Hardware RT
        // *takes precedence* over SSR: a world can enable both (RT on the
        // backend / GPU that supports it, SSR as the cross-backend fallback),
        // and where RT is live the builder picks it and omits SsrResolve. Only
        // one of the two is ever inserted.
        // Both resolves trace against the pre-pass view normal + linear depth
        // and pick their blur radius from its roughness. Declaring those reads
        // is what keeps the G-buffer's modelled lifetime as long as its real
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
            // RMW the resolve output. The read declares the sample dependency
            // (translucents sample the resolved scene for refraction); the
            // write produces the blended version downstream passes consume.
            trans.read_texture(current);
            current = trans.write_texture(current);
        }
        current
    } else if inputs.transparent_enabled {
        // With no reflection resolve the pre-TAA scene *is* `hdr_resolve` and
        // glass blends straight into it, so Transparent extends the hdr_resolve
        // chain by one version instead of branching a second resource onto the
        // same object.
        let mut trans = b.add_pass(PassId::Transparent, PassKind::Render);
        trans.read_texture(depth_cur);
        trans.read_texture(hdr_resolve_cur);
        trans.write_texture(hdr_resolve_cur)
    } else {
        hdr_resolve_cur
    };

    // `scene_color` is the engine-owned output the post-TAA composite stack
    // consumes, and -- with TAA on -- the history slot next frame samples, which
    // is why it stays imported rather than becoming a transient. Declared only
    // when TaaResolve or Upscale writes it, for the same one-object-one-resource
    // reason as above: with neither, the engine binds the name to the latest
    // pre-TAA scene texture. The two writers are mutually exclusive -- the
    // upscaler does its own temporal accumulation, so layering TaaResolve on top
    // would double-temporal the scene.
    let scene_color_cur = if inputs.upscale_enabled {
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
        taa.write_texture(scene_color)
    } else {
        scene_pre_taa_cur
    };

    let bloom_top_v1 = if inputs.bloom_enabled {
        let bloom_top = b.create_texture("bloom_top", bloom_top_desc(inputs));
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
        hizf.read_texture(depth_cur);
        let _ = hizf.write_texture(hiz);
    }

    // Composite (the presenter) reads scene_color + optional bloom_top,
    // and writes the swapchain via `presents()`. The occlusion view mode adds
    // an ao_output read so the pooled transient survives to the present.
    {
        let mut composite = b.add_pass(PassId::Composite, PassKind::Render);
        composite.read_texture(scene_color_cur);
        if let Some(h) = bloom_top_v1 {
            composite.read_texture(h);
        }
        if inputs.composite_reads_ao
            && let Some(h) = ao_output_v1
        {
            composite.read_texture(h);
        }
        composite.presents();
    }

    b.compile()
}

fn froxel_volume_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // The volumetric-fog froxel volume: a 3D texture the fog kernel writes and
    // the fog pass samples. Its Z extent also rides in `FogFroxelParams.
    // froxel_dims` so shaders can map indices to volume UVs.
    let _ = inputs;
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
/// between absolute indices and normalised volume UVs without recompiling.
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
    .with_array_layers(NUM_SHADOW_CASCADES as u32)
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

fn cluster_light_list_desc() -> BufferDesc {
    // Per-cluster light-index lists LightCull writes and Main reads. An identity
    // stub: the backend owns the real (persistent) buffer, so the graph only
    // tracks the read/write dependency, not the allocation.
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
// (three colour formats and a depth target) and so do their consumers, and a
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
    // R8 perceptual roughness, read by the reflection resolve to pick its
    // blur radius. Clears to 1.0 (fully rough), so a pixel the pre-pass never
    // rasterises reflects nothing -- the one graph target whose cleared
    // background carries meaning, and the reason `TextureDesc` models a clear
    // value at all.
    render_res_2d(
        inputs,
        PixelFormat::R8Unorm,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
    .with_clear_color([1.0, 0.0, 0.0, 0.0])
}

fn gbuffer_depth_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // The pre-pass's own depth attachment. Single-sample regardless of the
    // main pass's MSAA: the pre-pass rasterises once.
    render_res_2d(
        inputs,
        PixelFormat::Depth32Float,
        TextureUsage::DEPTH_STENCIL.union(TextureUsage::SHADER_READ),
    )
}

fn ssr_gbuffer_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // RGBA16F view-space normal + linear depth at HDR dims; shared with
    // SSAO when both passes are on.
    render_res_2d(
        inputs,
        PixelFormat::Rgba16Float,
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

fn velocity_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // RG16F motion-vector buffer at HDR dims, sampled by TaaResolve.
    render_res_2d(
        inputs,
        PixelFormat::Rg16Float,
        TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
    )
}

fn bloom_top_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // bloom_top is `bloom_targets.mips[0]`, the bloom chain's half-resolution
    // top octave; the prefilter pass writes into it and the upsample chain
    // accumulates back into it for Composite to sample.
    //
    // Half the *drawable* extent, not half the render resolution: every backend
    // builds its bloom chain from the output extent, so under temporal
    // upscaling (where render resolution is smaller) an `hdr_width >> 1` desc
    // names a texture no backend creates.
    let _ = inputs;
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

fn scene_color_desc(inputs: &FrameGraphInputs) -> TextureDesc {
    // The engine-owned scene_color texture the post stack consumes is
    // single-sample at HDR dims regardless of whether the per-frame
    // resolution lands on taa_targets / ssr_targets.output / hdr_resolve.
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

    // Index of `label` in the compiled graph's resource arena.
    fn resource_of(g: &CompiledGraph, label: &str) -> usize {
        g.resources
            .iter()
            .position(|r| r.label == label)
            .unwrap_or_else(|| panic!("{label} missing from the graph"))
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
    fn hiz_final_off_means_no_pyramid_in_the_graph() {
        // A world without the GPU-cull path builds no pyramid, so neither the node
        // nor the resource may appear (an imported resource with no pass would
        // still take a registry entry in every backend).
        let g = build_frame_graph(&all_off()).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(!order.contains(&PassId::HizFinal));
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
    fn the_msaa_colour_attachment_is_declared_only_when_multisampled() {
        // Without MSAA there is no resolve step and the single colour target is
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
        i.ssr_prepass_enabled = true;
        i.ssao_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::SsrPrepass,
                PassId::SsaoBlur,
                PassId::Main,
                PassId::Composite,
            ]
        );
    }

    #[test]
    fn unified_gbuffer_prepass_replaces_ssr_and_velocity() {
        // With the unified flag on, one GBufferPrepass node stands in for the
        // separate SsrPrepass + Velocity nodes; SSAO reads its output and TAA
        // reads its motion. Neither old node appears.
        let mut i = all_off();
        i.unified_gbuffer_prepass = true;
        i.ssr_prepass_enabled = true;
        i.ssao_enabled = true;
        i.velocity_enabled = true;
        i.taa_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::GBufferPrepass));
        assert!(!order.contains(&PassId::SsrPrepass));
        assert!(!order.contains(&PassId::Velocity));
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
    fn unified_gbuffer_prepass_runs_for_ssao_only() {
        // SSAO alone (no SSR / velocity) still triggers the merged node.
        let mut i = all_off();
        i.unified_gbuffer_prepass = true;
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
    fn unified_gbuffer_prepass_runs_for_velocity_only() {
        // Velocity alone (TAA, no SSR/SSAO) still triggers the merged node, and
        // the standalone Velocity node is not emitted.
        let mut i = all_off();
        i.unified_gbuffer_prepass = true;
        i.velocity_enabled = true;
        i.taa_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::GBufferPrepass));
        assert!(!order.contains(&PassId::Velocity));
    }

    #[test]
    fn gbuffer_prepass_orders_after_cull_via_draw_args() {
        // The GPU-driven G-buffer pre-pass reuses the main pass's per-frame
        // indirect command buffer, so it must run after Cull. With bindless cull
        // on and a G-buffer consumer active, the draw_args read edge pins
        // Cull -> GBufferPrepass (-> Main).
        let mut i = all_off();
        i.bindless_cull_enabled = true;
        i.unified_gbuffer_prepass = true;
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
    fn unified_gbuffer_prepass_omitted_when_no_consumers() {
        // The flag on but no consumer active: no pre-pass node at all.
        let mut i = all_off();
        i.unified_gbuffer_prepass = true;
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
        let decals = &g.passes[1];
        assert_eq!(decals.writes[0].version(), 2);
        let fog = &g.passes[3];
        assert_eq!(fog.writes[0].version(), 3);
        let particles = &g.passes[4];
        assert_eq!(particles.writes[0].version(), 4);
        let ssr = &g.passes[5];
        assert_eq!(ssr.reads[0].version(), 4);
    }

    #[test]
    fn upscale_replaces_taa_and_pins_after_velocity() {
        // Upscale takes TaaResolve's slot when temporal upscaling is on.
        // TaaResolve must not appear in the compiled graph (the scaler
        // does temporal accumulation itself), and Velocity must precede
        // Upscale via the explicit motion-vector read.
        let mut i = all_off();
        i.velocity_enabled = true;
        i.upscale_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(order.contains(&PassId::Upscale));
        assert!(!order.contains(&PassId::TaaResolve));
        assert!(
            order.iter().position(|p| *p == PassId::Velocity).unwrap()
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
        // TaaResolve reads the velocity buffer explicitly so the
        // toposort orders Velocity before TaaResolve via RAW (not
        // declaration order).
        let mut i = all_off();
        i.velocity_enabled = true;
        i.taa_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        // Main runs first, then Velocity + TaaResolve in compile-pass
        // order. TaaResolve reads scene_color v0 (imported v0 rule) +
        // velocity v1.
        assert!(
            order.iter().position(|p| *p == PassId::Velocity).unwrap()
                < order.iter().position(|p| *p == PassId::TaaResolve).unwrap()
        );
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
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(!order.contains(&PassId::Transparent));
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
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(!order.contains(&PassId::Ssgi));
    }

    #[test]
    fn rt_reflections_off_means_no_slot() {
        // No ray tracing requested: the pass is omitted entirely.
        let g = build_frame_graph(&all_off()).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(!order.contains(&PassId::RtReflections));
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
    fn ssgi_pinned_between_auto_exposure_and_decals() {
        // AutoExposure reads hdr_resolve_v1 (WAR); SSGI RMWs to v2; Decals
        // RMWs to v3. The toposort orders the three through the version chain.
        let mut i = all_off();
        i.auto_exposure_enabled = true;
        i.ssgi_enabled = true;
        i.decals_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::Main,
                PassId::AutoExposure,
                PassId::Ssgi,
                PassId::Decals,
                PassId::Composite,
            ]
        );
        // Version chain on hdr_resolve walks 1 → 2 → 3.
        let ssgi = &g.passes[2];
        assert_eq!(ssgi.writes[0].version(), 2);
        let decals = &g.passes[3];
        assert_eq!(decals.writes[0].version(), 3);
    }

    #[test]
    fn ssgi_after_raymarch_on_the_chain() {
        // With both on, SSGI reads the post-raymarch scene: Raymarch v1→v2,
        // SSGI v2→v3, SsrResolve reads v3.
        let mut i = all_off();
        i.raymarch_enabled = true;
        i.ssgi_enabled = true;
        i.ssr_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::Raymarch) < pos(PassId::Ssgi));
        assert!(pos(PassId::Ssgi) < pos(PassId::SsrResolve));
        let ssgi = &g.passes[pos(PassId::Ssgi)];
        assert_eq!(ssgi.writes[0].version(), 3);
    }

    #[test]
    fn raymarch_off_means_no_slot() {
        // No `SdfVolume` in the world: pass is omitted, no executor stub
        // ever fires.
        let g = build_frame_graph(&all_off()).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert!(!order.contains(&PassId::Raymarch));
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
        // Cull / SsrPrepass / SsaoBlur / Shadow / SSAO all precede Main.
        assert!(idx(&order, PassId::Cull) < idx(&order, PassId::Main));
        assert!(idx(&order, PassId::SsrPrepass) < idx(&order, PassId::Main));
        assert!(idx(&order, PassId::SsaoBlur) < idx(&order, PassId::Main));
        assert!(idx(&order, PassId::Shadow) < idx(&order, PassId::Main));
        // SsrPrepass precedes SsaoBlur (G-buffer share).
        assert!(idx(&order, PassId::SsrPrepass) < idx(&order, PassId::SsaoBlur));
        // AutoExposure post-Main, pre-Raymarch (WAR-pinned on hdr_resolve_v1).
        assert!(idx(&order, PassId::Main) < idx(&order, PassId::AutoExposure));
        assert!(idx(&order, PassId::AutoExposure) < idx(&order, PassId::Raymarch));
        // Raymarch leads the hdr_resolve RMW chain so Decals / Fog /
        // ParticlesDraw blend on top of the raymarched colour.
        assert!(idx(&order, PassId::Raymarch) < idx(&order, PassId::Decals));
        // hdr_resolve chain.
        assert!(idx(&order, PassId::Decals) < idx(&order, PassId::Fog));
        assert!(idx(&order, PassId::Fog) < idx(&order, PassId::ParticlesDraw));
        assert!(idx(&order, PassId::ParticlesDraw) < idx(&order, PassId::SsrResolve));
        // FogFroxel populates the volume Fog samples, so it must precede Fog.
        assert!(idx(&order, PassId::FogFroxel) < idx(&order, PassId::Fog));
        // Velocity precedes TaaResolve.
        assert!(idx(&order, PassId::Velocity) < idx(&order, PassId::TaaResolve));
        // Post-TAA chain. Transparent slots between SsrResolve and TaaResolve.
        assert!(idx(&order, PassId::SsrResolve) < idx(&order, PassId::Transparent));
        assert!(idx(&order, PassId::Transparent) < idx(&order, PassId::TaaResolve));
        assert!(idx(&order, PassId::TaaResolve) < idx(&order, PassId::Bloom));
        assert!(idx(&order, PassId::Bloom) < idx(&order, PassId::Composite));
        // Composite is the presenter and runs last.
        assert_eq!(order.last(), Some(&PassId::Composite));
        assert!(g.passes.last().unwrap().presents);
    }
}

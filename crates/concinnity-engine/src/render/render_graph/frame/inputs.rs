// What a frame asks of the graph: one gate per conditional pass, and the
// dimensions the graph's targets are sized from.

/// Per-frame inputs that gate conditional passes. The frame planner builds
/// them from what a backend has built and what the world holds this frame,
/// and `build_frame_graph` consumes them, so the conditional-inclusion
/// decisions made here match what every backend's executor dispatches.
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
    /// single-pass path. When on, the graph inserts `HizEarly` → `CullLate`
    /// → `MainLate` between `Main` and the post-decoration chain: `HizEarly`
    /// rebuilds the Hi-Z pyramid from early depth, `CullLate` re-tests the
    /// objects early cull marked occluded, and `MainLate` redraws the
    /// disoccluded survivors. `MainLate`'s `hdr_resolve` write becomes the head
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
    /// reducing its final depth into that pyramid for the next frame's early
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
    /// `true` when the grass also casts into a shadow cascade this frame: the
    /// backend built the grass shadow draw and a cascade near enough the
    /// camera for blades to shadow re-renders. With `grass_enabled` and
    /// `shadow_enabled` the graph adds a `GrassShadow` compute pass whose
    /// blades and draw arguments `Shadow` reads.
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
    /// an upscaler that takes it, or the reactive view reads it. Without a
    /// reader the writers leave it unstored; without a writer the readers are
    /// told there is none. A hidden world runs neither.
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
    pub(super) fn gbuffer_live(&self) -> bool {
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

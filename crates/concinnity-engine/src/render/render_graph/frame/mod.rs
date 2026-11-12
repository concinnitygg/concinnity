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

mod back;
mod descs;
mod front;
mod inputs;
#[cfg(test)]
mod tests;

use super::types::BufferHandle;
use super::{CompiledGraph, GraphBuilder, GraphError, PassId, PassKind, ReadStages, TextureHandle};

use back::{
    add_clouds, add_decorations, add_late_occlusion, add_late_particles, add_main,
    add_reflections_and_transparent, add_temporal,
};
pub(crate) use descs::bloom_top_desc;
pub use descs::{FAR_SHADOW_MAP_RESTS, FOG_FROXEL_X, FOG_FROXEL_Y, FOG_FROXEL_Z};
use descs::{
    hdr_color_desc, hdr_depth_desc, hdr_resolve_desc, hiz_pyramid_desc, reactive_mask_desc,
};
use front::{add_cull_and_grass, add_prepasses, add_scene_inputs};
pub use inputs::FrameGraphInputs;
#[cfg(test)]
pub(crate) use inputs::GATED_FLAGS;

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
// MainLate and Raymarch bump it, and Decals / Fog / Lines / Transparent /
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
// When `two_pass_occlusion_enabled` is on the spine gains a late
// prefix: `Cull → Main → HizEarly → CullLate → MainLate → AutoExposure →
// …`. `Main` writes hdr_resolve v1 / hdr_depth v1; `HizEarly` reads
// the depth and writes the Hi-Z pyramid; `CullLate` reads the pyramid +
// the early status buffer and writes `draw_args_late`; `MainLate` RMWs
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
    // single-pass path with no orphaned late nodes.
    let two_pass = inputs.bindless_cull_enabled && inputs.two_pass_occlusion_enabled;

    // The Hi-Z depth pyramid, imported up front because `Cull` reads the version
    // the *previous* frame left there before `HizFinal` (and, under two-pass,
    // the mid-frame `HizEarly`) overwrites it. Never a transient: its contents
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

    // Two-pass occlusion, late pass: rebuild the Hi-Z pyramid from early
    // depth (HizEarly), re-test the objects early cull marked occluded
    // (CullLate), and redraw the disoccluded survivors (MainLate). MainLate RMWs
    // hdr_color / hdr_depth / hdr_resolve, so its hdr_resolve write becomes
    // the head of the post-decoration chain: AutoExposure and every later
    // RMW pass see the combined early + late scene. Without two-pass
    // the head stays at Main's hdr_resolve_v1.
    let (hiz_cur, scene_head) = if let (true, Some(hiz)) = (two_pass, hiz_pyramid) {
        let (hiz_v1, scene) =
            add_late_occlusion(&mut b, hiz, front.cull.cull_status, hdr.color, main_out);
        (Some(hiz_v1), scene)
    } else {
        (hiz_pyramid, main_out)
    };

    // AutoExposure (compute) reads the post-main scene (hdr_resolve_head:
    // MainLate's output under two-pass, Main's otherwise). The compile pass's
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
    // the next frame's early Cull tests against. Declared last of the
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
    far_shadow: Option<TextureHandle>,
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

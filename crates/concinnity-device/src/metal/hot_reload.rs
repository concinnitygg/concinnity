// src/metal/hot_reload.rs
//
// Filesystem watcher driving Metal shader hot-reload. A background notify
// watcher tails `<CARGO_MANIFEST_DIR>/src/metal/shaders/` and, on any modify
// event for a `.metal` file, flips a shared `Arc<AtomicBool>`. The main thread
// polls that flag at the top of `draw_frame` and calls
// `MtlContext::reload_shaders` when it's set. Same flag is also set by the
// `reload-shaders` debug command, so the two trigger paths converge.
//
// All entirely a dev-loop concern: only constructed when
// `MtlContext::new` is called with `hot_reload = true`. Production `cn run`
// never instantiates it.
#![deny(unsafe_op_in_unsafe_fn)]

use notify::{Event, EventKind, RecursiveMode, Watcher};
use objc2::rc::Retained;
use objc2_metal::{MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::auto_exposure::build_auto_exposure_pipelines;
use super::context::MtlContext;
use super::cull::build_shadow_cull_pipeline;
use super::decal::build_decal_pipeline;
use super::descriptors::{VertexAttr, VertexLayout, vertex_descriptor};
use super::fog::build_fog_pipeline;
use super::hiz::build_hiz_pipelines;
use super::init::pipelines::{
    MainPipelineBundle, build_main_pipeline, build_shadow_bindless_pipeline, build_shadow_pipeline,
    make_vertex_descriptor,
};
use super::pipeline::{build_post_pipeline, build_text_pipeline};
use super::post::{
    build_bloom_pipelines, build_gbuffer_bindless_pipeline, build_reflection_blur_pipeline,
    build_reflection_composite_pipeline, build_rt_reflection_pipeline, build_ssao_pipeline,
    build_ssgi_composite_pipeline, build_ssgi_gather_pipeline, build_ssr_pipeline,
    build_taa_pipeline,
};
use super::resources::skinning::{build_skinned_shadow_pipeline, make_skinned_vertex_descriptor};
use crate::metal::slang_shaders::{SSAO_BLUR, SSAO_KERNEL};

// Rebuild a built-in pipeline only when it is currently live. Expands to
// `if $cond { Some($build?) } else { None }`: the rebuild-then-swap pattern
// `reload_shaders` repeats for every optional pipeline: a `None` field stays
// `None`, and any compile error (the `?`) aborts the whole reload before the
// swap, leaving the live pipelines untouched.
macro_rules! rebuild_if_live {
    ($cond:expr_2021, $build:expr_2021 $(,)?) => {
        if $cond { Some($build?) } else { None }
    };
}

// Live watcher handle. Held by `MtlContext` purely to keep the watcher
// thread alive; dropping it stops the watcher. The flag itself is shared
// via `MtlContext`'s `hot_reload.reload_pending`.
pub(crate) struct WatcherHandle {
    // We don't read `_watcher` after construction; notify keeps its own
    // listener thread for as long as the handle is alive.
    #[expect(
        dead_code,
        reason = "notify keeps its listener thread alive while the handle lives; never read after construction"
    )]
    watcher: notify::RecommendedWatcher,
}

// Spawn a `notify` watcher over the Metal shader source directory and wire
// it to flip `flag` on any `.metal` file modify event. The path is derived
// from `CARGO_MANIFEST_DIR` at compile time so the watcher works no matter
// where the binary is launched from, but only as long as the source tree
// still exists at that path. A shipped binary should never be hot-reload-
// enabled, so the missing-path case logs and returns `None` instead of
// failing the whole context init.
pub(crate) fn spawn(flag: Arc<AtomicBool>) -> Option<WatcherHandle> {
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("metal")
        .join("shaders");
    if !dir.is_dir() {
        tracing::warn!(
            "hot-reload: shader source dir {} not found; watcher disabled (debug \
             command still works)",
            dir.display()
        );
        return None;
    }

    // Suppress event bursts: editors (vim, VSCode) frequently emit several
    // close-write / rename events per save. Coalesce by a small debounce so
    // one save triggers exactly one reload.
    let debounce = Duration::from_millis(150);
    let last_fire = std::sync::Mutex::new(Instant::now() - debounce);
    let flag_for_cb = Arc::clone(&flag);
    let mut watcher = match notify::recommended_watcher(move |res: notify::Result<Event>| {
        let event = match res {
            Ok(e) => e,
            Err(e) => {
                tracing::debug!("hot-reload watcher error: {e}");
                return;
            }
        };
        if !is_relevant(&event) {
            return;
        }
        let mut last = match last_fire.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let now = Instant::now();
        if now.duration_since(*last) < debounce {
            return;
        }
        *last = now;
        tracing::info!(
            "hot-reload: detected change to {:?}, scheduling shader rebuild",
            event.paths
        );
        flag_for_cb.store(true, Ordering::SeqCst);
    }) {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!("hot-reload: failed to create notify watcher: {e}");
            return None;
        }
    };

    if let Err(e) = watcher.watch(&dir, RecursiveMode::NonRecursive) {
        tracing::warn!(
            "hot-reload: failed to watch {} ({}); watcher disabled",
            dir.display(),
            e
        );
        return None;
    }

    // The single-source shader directory rides the same watcher: a `.slang`
    // save rebuilds through the same flag. Best-effort, like the main dir.
    let slang_dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("shaders");
    if slang_dir.is_dir()
        && let Err(e) = watcher.watch(&slang_dir, RecursiveMode::NonRecursive)
    {
        tracing::warn!(
            "hot-reload: failed to watch {} ({e}); .slang edits will not trigger reloads",
            slang_dir.display()
        );
    }

    tracing::info!(
        "hot-reload: watching {} for .metal and .slang changes",
        dir.display()
    );
    Some(WatcherHandle { watcher })
}

// True when this notify event is a modify of a shader source file we care
// about (`.metal`, or a single-source `.slang`). Filters out unrelated paths
// (e.g. swap files, sub-directory churn) and the non-mutating events notify
// emits (e.g. access/metadata).
fn is_relevant(event: &Event) -> bool {
    if !matches!(
        event.kind,
        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
    ) {
        return false;
    }
    event
        .paths
        .iter()
        .any(|p| p.extension().is_some_and(|e| e == "metal" || e == "slang"))
}

// Static-vertex-layout descriptor used by the velocity / SSAO / SSR pre-pass
// rebuilds during hot-reload. Matches the layout `MtlContext::new` builds at
// init; kept in sync by construction since both touch the 56-byte static
// `Vertex` struct.
fn static_vertex_descriptor() -> Retained<MTLVertexDescriptor> {
    vertex_descriptor(
        &[
            VertexAttr {
                index: 0,
                format: MTLVertexFormat::Float3,
                offset: 0,
                buffer_index: 1,
            },
            VertexAttr {
                index: 1,
                format: MTLVertexFormat::Float3,
                offset: 12,
                buffer_index: 1,
            },
            VertexAttr {
                index: 2,
                format: MTLVertexFormat::Float3,
                offset: 24,
                buffer_index: 1,
            },
            VertexAttr {
                index: 3,
                format: MTLVertexFormat::Float3,
                offset: 36,
                buffer_index: 1,
            },
            VertexAttr {
                index: 4,
                format: MTLVertexFormat::Float2,
                offset: 48,
                buffer_index: 1,
            },
        ],
        &[VertexLayout {
            buffer_index: 1,
            stride: std::mem::size_of::<crate::gfx::mesh_payload::Vertex>(),
            step: MTLVertexStepFunction::PerVertex,
        }],
    )
}

impl MtlContext {
    // True when the shared shader-reload flag is set. Cheap atomic load; called
    // at the top of `draw_frame`. Returns false when hot-reload is off so
    // the production path never enters the reload branch.
    pub(super) fn shader_reload_requested(&self) -> bool {
        self.hot_reload
            .reload_pending
            .as_ref()
            .map(|f| f.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    // Clear the pending-reload flag. Called after `reload_shaders` regardless
    // of outcome so a failed rebuild does not loop forever.
    pub(super) fn clear_shader_reload_flag(&self) {
        if let Some(flag) = &self.hot_reload.reload_pending {
            flag.store(false, Ordering::SeqCst);
        }
    }

    // Rebuild every built-in Metal renderer pipeline from disk-resident source.
    // Each pipeline is constructed into a temporary first; only when every
    // rebuild succeeds does the context atomically swap them in. Any compile
    // or link error logs the underlying message and leaves the live pipelines
    // untouched: a typo in a shader edit won't crash the running session.
    //
    // Covers the main pass and its GPU cull (one builder, since the cull's
    // argument encoder comes from the pipeline it feeds) and the skinned
    // G-buffer pre-pass and skinned shadow variants, which compile from the
    // same single-source files as their static siblings under a different
    // entry. A world Shader's own pair still wins in the main build, so a save
    // to an engine template never swaps a world's program for the engine's.
    pub(super) fn reload_shaders(&mut self) -> Result<(), String> {
        if !self.hot_reload.enabled {
            return Ok(());
        }
        let device = &self.device;
        let hr = true;

        // Build every replacement into a temporary first. A `?` early-return
        // here means we never overwrite a live pipeline with a failed build:
        // any compile error leaves the running session rendering with the
        // previous shader source.
        let post = build_post_pipeline(device, self.swap_pixel_format, hr)?;
        let bloom = rebuild_if_live!(
            self.bloom_pipelines.is_some(),
            build_bloom_pipelines(device, hr)
        );

        let text = rebuild_if_live!(
            self.text.pipeline_state.is_some(),
            build_text_pipeline(device, self.swap_pixel_format, hr)
        );
        let taa = rebuild_if_live!(
            self.taa.pipeline_state.is_some(),
            build_taa_pipeline(device, hr)
        );
        // The main pass and the GPU cull come from one builder, because the
        // cull's argument encoder is derived from the pipeline it feeds. A
        // world Shader's pair wins here exactly as it does at init, so a save
        // to an engine template does not swap a world's program for the
        // engine's.
        let main = rebuild_if_live!(
            self.pipeline_state.is_some(),
            build_main_pipeline(
                device,
                &make_vertex_descriptor(),
                self.world_shader.as_ref(),
                hr,
            )
        );
        // The engine sampler block rides the fresh fragment's encoder.
        let main_sampler_args = match main
            .as_ref()
            .and_then(|m| m.bindless_sampler_arg_encoder.as_ref())
        {
            Some(enc) => Some(super::init::pipelines::build_bindless_sampler_args(
                device,
                enc,
                &self.sampler,
                &self.shadow.sampler,
                &self.cube_sampler,
            )?),
            None => None,
        };
        // Hi-Z build kernels are engine built-ins (independent of the world
        // shader); rebuild them whenever a Hi-Z resource exists so a saved
        // edit to `hiz_build.slang` is picked up. The texture + mip views are
        // kept: only the pipelines swap.
        let hiz = rebuild_if_live!(self.cull.hiz.is_some(), build_hiz_pipelines(device, hr));
        let auto_ev = rebuild_if_live!(
            self.auto_exposure.pipelines.is_some(),
            build_auto_exposure_pipelines(device, hr)
        );
        let decal = rebuild_if_live!(
            self.decal.pipeline.is_some(),
            build_decal_pipeline(device, hr)
        );
        let fog = rebuild_if_live!(self.fog.pipeline.is_some(), build_fog_pipeline(device, hr));

        // The shadow pipeline needs the static vertex layout.
        let static_vdesc = static_vertex_descriptor();
        let ssao_kernel = rebuild_if_live!(
            self.ssao.kernel_pipeline.is_some(),
            build_ssao_pipeline(device, &SSAO_KERNEL, hr)
        );
        let ssao_blur = rebuild_if_live!(
            self.ssao.blur_pipeline.is_some(),
            build_ssao_pipeline(device, &SSAO_BLUR, hr)
        );
        // The G-buffer pipeline builds its own two-stream vertex descriptor
        // internally.
        let gbuffer_bindless = rebuild_if_live!(
            self.gbuffer.bindless_pipeline.is_some(),
            build_gbuffer_bindless_pipeline(device, hr)
        );
        let ssr_resolve = rebuild_if_live!(
            self.ssr.resolve_pipeline.is_some(),
            build_ssr_pipeline(device, hr)
        );
        // The probe cube argument encoder is read off the SSR resolve fragment,
        // so an edit to that shader's block has to reach it whether or not the
        // world runs SSR.
        let probe_cube_arg_encoder = super::probe_cubes::probe_cube_arg_encoder(device, hr)?;
        let reflection_composite = rebuild_if_live!(
            self.ssr.composite_pipeline.is_some(),
            build_reflection_composite_pipeline(device, hr)
        );
        let reflection_blur = rebuild_if_live!(
            self.ssr.blur_pipeline.is_some(),
            build_reflection_blur_pipeline(device, hr)
        );
        let ssgi_gather = rebuild_if_live!(
            self.ssgi.gather_pipeline.is_some(),
            build_ssgi_gather_pipeline(device, hr)
        );
        let ssgi_composite = rebuild_if_live!(
            self.ssgi.composite_pipeline.is_some(),
            build_ssgi_composite_pipeline(device, hr)
        );
        let rt_reflections = rebuild_if_live!(
            self.rt.pipeline.is_some(),
            build_rt_reflection_pipeline(
                device,
                &crate::metal::slang_shaders::RT_REFLECTIONS_FRAG,
                hr
            )
        );
        let rt_reflections_textured = rebuild_if_live!(
            self.rt.pipeline_textured.is_some(),
            build_rt_reflection_pipeline(
                device,
                &crate::metal::slang_shaders::RT_REFLECTIONS_FRAG_TEXTURED,
                hr
            )
        );

        // The skinned shadow caster rides the 80-byte skinned vertex layout.
        let skinned_vdesc = if self.skinned.shadow_pipeline_state.is_some() {
            Some(make_skinned_vertex_descriptor())
        } else {
            None
        };

        // Shadow pass shaders are engine-internal (compiled from
        // `shadow.metal`), so they rebuild here alongside the other
        // built-ins rather than in `update_world_shader_pipelines`. The static
        // shadow pipeline shares the 56-byte static layout; the skinned one
        // rides the 80-byte skinned layout.
        let shadow = rebuild_if_live!(
            self.shadow.pipeline_state.is_some(),
            build_shadow_pipeline(device, &static_vdesc, hr)
        );
        let skinned_shadow = rebuild_if_live!(
            self.skinned.shadow_pipeline_state.is_some(),
            build_skinned_shadow_pipeline(
                device,
                skinned_vdesc.as_ref().expect("skinned vdesc just built"),
                hr,
            )
        );

        // GPU-driven cascaded-shadow pipelines: the frustum-only shadow
        // decision kernel (from cull.slang) + the depth-only bindless shadow
        // render pipeline. Both engine-internal, so they rebuild here. Gated on
        // the live shadow-bindless path.
        let shadow_cull = rebuild_if_live!(
            self.cull.shadow_pipeline.is_some(),
            build_shadow_cull_pipeline(device, hr)
        );
        let shadow_bindless = rebuild_if_live!(
            self.cull.shadow_bindless_pipeline.is_some(),
            build_shadow_bindless_pipeline(device, &static_vdesc, hr)
        );

        // All builds succeeded: swap into the live context. After this
        // point the next frame's draw calls bind the freshly compiled
        // pipelines.
        self.post_pipeline_state = post;
        if let Some(b) = bloom {
            self.bloom_pipelines = Some(b);
        }
        if let Some(p) = text {
            self.text.pipeline_state = Some(p);
        }
        if let Some(p) = taa {
            self.taa.pipeline_state = Some(p);
        }
        if let Some(p) = main {
            self.pipeline_state = Some(p.pipeline_state);
            self.bindless_tex_arg_encoder = p.bindless_tex_arg_encoder;
            self.bindless_sampler_args = main_sampler_args;
            self.cull.pipeline = Some(p.cull.decide);
            self.cull.pipeline_phase2 = Some(p.cull.decide_phase2);
            self.cull.encode_pipeline = Some(p.cull.encode);
            self.cull.icb_arg_encoder = Some(p.cull.icb_arg_encoder);
            // Force the ICB rebuilds on the next frame so every argument buffer
            // is re-encoded with the arg encoder the new encode kernel produced.
            // The status buffers and the phase-2 ICB are rebuilt by the same
            // `ensure_*_capacity` passes that rebuild the ICBs.
            self.cull.icbs = Vec::new();
            self.cull.icb_arg_buffer = None;
            self.cull.icb_capacity = 0;
            self.cull.icbs_2 = Vec::new();
            self.cull.icb_2_arg_buffer = None;
            self.cull.status_buffer = None;
            self.cull.shadow_icb = None;
            self.cull.shadow_icb_arg_buffer = None;
            self.cull.shadow_status = None;
            self.cull.shadow_icb_capacity = 0;
        }
        if let Some((init_pipeline, downsample_pipeline)) = hiz
            && let Some(h) = self.cull.hiz.as_mut()
        {
            h.swap_pipelines(init_pipeline, downsample_pipeline);
        }
        if let Some(p) = auto_ev {
            self.auto_exposure.pipelines = Some(p);
        }
        if let Some(p) = decal {
            self.decal.pipeline = Some(p);
        }
        if let Some(p) = fog {
            self.fog.pipeline = Some(p);
        }
        if let Some(p) = ssao_kernel {
            self.ssao.kernel_pipeline = Some(p);
        }
        if let Some(p) = ssao_blur {
            self.ssao.blur_pipeline = Some(p);
        }
        if let Some(p) = gbuffer_bindless {
            self.gbuffer.bindless_pipeline = Some(p);
        }
        if let Some(p) = ssr_resolve {
            self.ssr.resolve_pipeline = Some(p);
        }
        self.probe_cube_arg_encoder = probe_cube_arg_encoder;
        if let Some(p) = reflection_composite {
            self.ssr.composite_pipeline = Some(p);
        }
        if let Some(p) = reflection_blur {
            self.ssr.blur_pipeline = Some(p);
        }
        if let Some(p) = ssgi_gather {
            self.ssgi.gather_pipeline = Some(p);
        }
        if let Some(p) = ssgi_composite {
            self.ssgi.composite_pipeline = Some(p);
        }
        if let Some(p) = rt_reflections {
            self.rt.pipeline = Some(p);
        }
        if let Some(p) = rt_reflections_textured {
            self.rt.pipeline_textured = Some(p);
        }
        if let Some(p) = shadow {
            self.shadow.pipeline_state = Some(p);
        }
        if let Some(p) = skinned_shadow {
            self.skinned.shadow_pipeline_state = Some(p);
        }
        if let Some(p) = shadow_cull {
            self.cull.shadow_pipeline = Some(p);
        }
        if let Some(p) = shadow_bindless {
            self.cull.shadow_bindless_pipeline = Some(p);
        }
        Ok(())
    }

    // Rebuild the world-loaded main pipeline from a freshly compiled Shader
    // payload. Driven by asset hot-reload (`cn debug` only) when one of the
    // Shader's files is saved or `reload-assets` is fired. Mirrors the
    // rebuild-then-swap safety pattern of [`Self::reload_shaders`]: every
    // replacement is constructed into a temporary first, and the swap only
    // runs when every build succeeds, so a typo in a shader edit leaves the
    // live pipelines untouched and the session keeps rendering.
    //
    // Every draw a world Shader reaches goes through the GPU-driven pass, so
    // this is the one pipeline it owns. The shadow and G-buffer pipelines
    // compile from engine-internal source and are covered by
    // [`Self::reload_shaders`].
    pub(super) fn update_world_shader_pipelines(
        &mut self,
        programs: &concinnity_core::components::ShaderPrograms,
    ) -> Result<(), String> {
        let world = Some(programs);

        // Build everything into temporaries first. Any `?` early-return
        // leaves the live pipelines untouched, mirroring `reload_shaders`.
        // A scene-less world never built a main pipeline; there is nothing
        // for the fresh world-shader programs to replace.
        let vert_desc = make_vertex_descriptor();
        let new_main = if self.pipeline_state.is_some() {
            Some(build_main_pipeline(
                &self.device,
                &vert_desc,
                world,
                self.hot_reload.enabled,
            )?)
        } else {
            None
        };

        // The engine sampler block rides the fresh fragment's encoder; built
        // here (still before the swap) so a failure leaves the live state
        // untouched.
        let new_sampler_args = match new_main
            .as_ref()
            .and_then(|m| m.bindless_sampler_arg_encoder.as_ref())
        {
            Some(enc) => Some(super::init::pipelines::build_bindless_sampler_args(
                &self.device,
                enc,
                &self.sampler,
                &self.shadow.sampler,
                &self.cube_sampler,
            )?),
            None => None,
        };

        // All builds succeeded: swap into the live context. After this
        // point the next frame's draw calls bind the freshly compiled
        // pipelines.
        if let Some(new_main) = new_main {
            let MainPipelineBundle {
                pipeline_state,
                cull,
                bindless_tex_arg_encoder,
                bindless_sampler_arg_encoder: _,
            } = new_main;
            self.pipeline_state = Some(pipeline_state);
            // Swap the cull state with the pipeline; `two_pass_occlusion` keeps
            // its init-time resolution.
            self.cull.pipeline = Some(cull.decide);
            self.cull.pipeline_phase2 = Some(cull.decide_phase2);
            self.cull.encode_pipeline = Some(cull.encode);
            self.cull.icb_arg_encoder = Some(cull.icb_arg_encoder);
            self.bindless_tex_arg_encoder = bindless_tex_arg_encoder;
            self.bindless_sampler_args = new_sampler_args;
            // Force fresh ICBs on the next frame so every argument buffer is
            // re-encoded with the new encoder. Matches the `cull` swap in
            // `reload_shaders`; the status buffers and phase-2 ICB rebuild
            // alongside.
            self.cull.icbs = Vec::new();
            self.cull.icb_arg_buffer = None;
            self.cull.icb_capacity = 0;
            self.cull.icbs_2 = Vec::new();
            self.cull.icb_2_arg_buffer = None;
            self.cull.status_buffer = None;
            self.cull.shadow_icb = None;
            self.cull.shadow_icb_arg_buffer = None;
            self.cull.shadow_status = None;
            self.cull.shadow_icb_capacity = 0;
        }

        self.world_shader = Some(programs.clone());
        Ok(())
    }
}

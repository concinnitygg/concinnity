// src/vulkan/hot_reload.rs
//
// Filesystem watcher driving Vulkan shader hot-reload. A background notify
// watcher tails `<CARGO_MANIFEST_DIR>/src/vulkan/shaders/` and, on any modify
// event for a known shader-source extension, flips a shared
// `Arc<AtomicBool>`. The main thread polls that flag at the top of
// `draw_frame` and calls `VkContext::reload_shaders` when it is set. The
// same flag is also set by the `reload-shaders` debug command, so
// the two trigger paths converge.
//
// Entirely a dev-loop concern, only constructed when `VkContext::new` is
// called with `hot_reload = true`. Production `cn run` never instantiates
// it. Mirrors `directx/hot_reload.rs` and `metal/hot_reload.rs`.

use notify::{Event, EventKind, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ash::vk;

use super::auto_exposure::{AutoExposureResources, compile_auto_exposure_shaders};
use super::context::VkContext;
use super::pipeline::{
    BucketPipelineTargets, build_bucket_pipeline, compile_bindless_shaders,
    compile_composite_shaders, compile_cull_shader, compile_cull_shader_phase2,
    compile_text_shaders, create_composite_pipeline, create_cull_pipeline, create_text_pipeline,
};
use super::post::bloom::{compile_bloom_shaders, create_bloom_pipeline};
use super::post::ssao::rebuild_ssao_pipelines;
use super::post::ssr::rebuild_ssr_pipelines;
use super::post::taa::rebuild_taa_pipelines;

// Rebuild a feature's pipeline(s) into a temporary only when the feature is
// live, propagating any compile/create error out of the enclosing
// `reload_shaders`. `$cond` is the liveness check (`self.x.is_some()`); `$build`
// is the build expression (which re-accesses `self.x` and may use `?` internally
// for the shader compile). Expands to `Some(build?)` when live, `None`
// otherwise, so the swap phase below can pair each rebuilt `Some(_)` with its
// live target uniformly. Mirrors `directx/hot_reload.rs::rebuild_if_live!`.
macro_rules! rebuild_if_live {
    ($cond:expr_2021, $build:expr_2021 $(,)?) => {
        if $cond { Some($build?) } else { None }
    };
}

// Shader-source extensions the watcher reacts to: every program is a `.slang`
// now. The helper rejects every other event so editor swap files, README
// updates, and tmp files don't trigger a rebuild.
const SHADER_EXTENSIONS: &[&str] = &["slang"];

// Live watcher handle. Held by `VkContext` purely to keep the watcher
// thread alive; dropping it stops the watcher. The flag itself is shared
// via [`VkContext::shader_reload_pending`].
pub(crate) struct WatcherHandle {
    // notify keeps its own listener thread alive for as long as the handle
    // exists; we never read this field after construction.
    #[expect(
        dead_code,
        reason = "notify keeps its listener thread alive while the handle lives; never read after construction"
    )]
    watcher: notify::RecommendedWatcher,
}

// Spawn a `notify` watcher over the Vulkan shader source directory and
// wire it to flip `flag` on any modify event for a known shader extension.
// The path is derived from `CARGO_MANIFEST_DIR` at compile time so the
// watcher works no matter where the binary is launched from, but only as
// long as the source tree still exists at that path. A shipped binary
// should never be hot-reload-enabled, so the missing-path case logs and
// returns `None` instead of failing the whole context init.
pub(crate) fn spawn(flag: Arc<AtomicBool>) -> Option<WatcherHandle> {
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("vulkan")
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
        "hot-reload: watching {} for {} changes",
        dir.display(),
        SHADER_EXTENSIONS.join("/"),
    );
    Some(WatcherHandle { watcher })
}

// True when this notify event is a modify of a known shader file. Filters
// out unrelated paths (e.g. swap files, sub-directory churn) and the
// non-mutating events notify emits (e.g. access/metadata).
fn is_relevant(event: &Event) -> bool {
    if !matches!(
        event.kind,
        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
    ) {
        return false;
    }
    event.paths.iter().any(|p| {
        p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
            SHADER_EXTENSIONS
                .iter()
                .any(|&se| se.eq_ignore_ascii_case(e))
        })
    })
}

impl VkContext {
    // True when the shared shader-reload flag is set. Cheap atomic load;
    // called at the top of `draw_frame`. Returns false when hot-reload is
    // off so the production path never enters the reload branch.
    pub(in crate::vulkan) fn shader_reload_requested(&self) -> bool {
        self.hot_reload
            .reload_pending
            .as_ref()
            .map(|f| f.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    // Clear the pending-reload flag. Called after `reload_shaders`
    // regardless of outcome so a failed rebuild does not loop forever.
    pub(in crate::vulkan) fn clear_shader_reload_flag(&self) {
        if let Some(flag) = &self.hot_reload.reload_pending {
            flag.store(false, Ordering::SeqCst);
        }
    }

    // Rebuild every built-in Vulkan pipeline from disk-resident source.
    // Each pipeline is constructed into a temporary first; only when every
    // rebuild succeeds does the context swap them in (after destroying
    // the displaced ones). Any GLSL compile or pipeline-create failure
    // logs the underlying message and leaves the live pipelines untouched:
    // a typo in a shader edit won't crash the running session.
    //
    // Covers every runtime-bundled pipeline whose source lives in
    // `vulkan/shaders/`: composite, text, bloom (prefilter / downsample /
    // upsample), bindless main (when live), GPU-cull compute, auto-exposure
    // (build + average), projected-decal, volumetric-fog, SSAO (prepass
    // static / instanced / skinned, kernel, blur), SSR (prepass static /
    // instanced / skinned, resolve), and TAA (velocity static / instanced,
    // resolve). The world-loaded main / shadow / instanced / skinned
    // pipelines remain out of scope; same split as DirectX. The caller
    // has already `device_wait_idle`'d so swapping pipelines out from
    // under in-flight command buffers is safe.
    pub(in crate::vulkan) fn reload_shaders(&mut self) -> Result<(), String> {
        if !self.hot_reload.enabled {
            return Ok(());
        }
        let device = self.device.clone();
        let device = &device;
        let hr = true;

        // Build every replacement into a temporary first. A `?` early-return
        // here means we never overwrite a live pipeline with a failed build:
        // any compile error leaves the running session rendering with the
        // previous shader source.

        // Composite (always live).
        let (composite_vs, composite_ps) = compile_composite_shaders(hr)?;
        let composite_pipeline = create_composite_pipeline(
            device,
            self.composite.render_pass.handle(),
            self.composite.pipeline_layout.handle(),
            &composite_vs,
            &composite_ps,
        )?;

        // Text (only when the world declared text atlases).
        let text_pipeline = rebuild_if_live!(self.text.pipeline.is_some(), {
            let (tv, tf) = compile_text_shaders(hr)?;
            create_text_pipeline(
                device,
                self.composite.render_pass.handle(),
                self.text.pipeline_layout.handle(),
                &tv,
                &tf,
                vk::SampleCountFlags::TYPE_1,
            )
        });

        // Bloom (always live, 3 pipelines).
        let bloom_shaders = compile_bloom_shaders(hr)?;
        let bloom_prefilter = create_bloom_pipeline(
            device,
            self.bloom.write_pass.handle(),
            self.bloom.pipeline_layout.handle(),
            &bloom_shaders.vert,
            &bloom_shaders.prefilter,
            false,
        )?;
        let bloom_downsample = create_bloom_pipeline(
            device,
            self.bloom.write_pass.handle(),
            self.bloom.pipeline_layout.handle(),
            &bloom_shaders.vert,
            &bloom_shaders.downsample,
            false,
        )?;
        let bloom_upsample = create_bloom_pipeline(
            device,
            self.bloom.blend_pass.handle(),
            self.bloom.pipeline_layout.handle(),
            &bloom_shaders.vert,
            &bloom_shaders.upsample,
            true,
        )?;

        // Bucket 0 of the GPU-driven main pass, from the engine's freshly
        // compiled pair; a world default Shader's own pair is spliced into the
        // same templates, so it is rebuilt against them too.
        let bindless_main_pipeline = rebuild_if_live!(
            self.cull.bindless_pipeline_layout.is_some() && self.cull.bindless_pipeline.is_some(),
            {
                let engine_pair = compile_bindless_shaders(
                    hr,
                    self.cull.bindless_pool_size,
                    self.descriptors.probe_cube_count,
                )?;
                let pipeline =
                    self.build_world_main_pipeline(self.world_shader.as_ref(), &engine_pair)?;
                Ok::<_, String>((pipeline, engine_pair))
            }
        );
        let cull_pipeline = rebuild_if_live!(
            self.cull.cull_pipeline_layout.is_some() && self.cull.cull_pipeline.is_some(),
            {
                let cs = compile_cull_shader(hr)?;
                create_cull_pipeline(
                    device,
                    self.cull
                        .cull_pipeline_layout
                        .as_ref()
                        .expect("cull pipeline layout is live alongside its pipeline")
                        .handle(),
                    &cs,
                )
            }
        );
        // Phase-2 cull (two-pass occlusion), rebuilt alongside phase 1 from the
        // same source with the `CULL_PHASE2` define + the shared layout.
        let cull_pipeline_phase2 = rebuild_if_live!(
            self.cull.cull_pipeline_layout.is_some() && self.cull.cull_pipeline_phase2.is_some(),
            {
                let cs = compile_cull_shader_phase2(hr)?;
                create_cull_pipeline(
                    device,
                    self.cull
                        .cull_pipeline_layout
                        .as_ref()
                        .expect("cull pipeline layout is live alongside its phase-2 pipeline")
                        .handle(),
                    &cs,
                )
            }
        );
        // Hi-Z build kernels (live alongside the cull pipeline).
        let hiz_pipelines = rebuild_if_live!(
            self.cull.hiz.is_some(),
            self.cull
                .hiz
                .as_ref()
                .expect("hi-Z resources are live alongside the cull pipeline")
                .recompile_pipelines(device, hr)
        );

        // Auto-exposure (gated on the post-process config). Builds the histogram
        // + average compute pipelines; the trailing `.map` tuples them so the
        // whole build is one Result expression for the macro.
        let auto_exposure_pipelines = rebuild_if_live!(self.auto_exposure.resources.is_some(), {
            let ae = self
                .auto_exposure
                .resources
                .as_ref()
                .expect("auto-exposure resources are live");
            let (build_cs, average_cs) = compile_auto_exposure_shaders(hr)?;
            let build = AutoExposureResources::create_compute_pipeline(
                device,
                ae.build_pipeline_layout(),
                &build_cs,
            )?;
            AutoExposureResources::create_compute_pipeline(
                device,
                ae.average_pipeline_layout(),
                &average_cs,
            )
            .map(|average| (build, average))
        });

        // Decal (always built when DecalResources exists, which is
        // unconditional in `VkContext::new`).
        let decal_pipeline = rebuild_if_live!(
            self.decal.resources.is_some(),
            super::decal::rebuild_decal_pipeline(
                device,
                self.decal.resources.as_ref().expect("decal state is live"),
                self.msaa_samples != vk::SampleCountFlags::TYPE_1,
                hr,
            )
        );

        // Lines (only once a frame published some and the lazy build ran).
        let line_pipeline = rebuild_if_live!(
            self.lines.resources.is_some(),
            super::line::rebuild_line_pipeline(
                device,
                self.lines
                    .resources
                    .as_ref()
                    .expect("line resources are live"),
                self.msaa_samples != vk::SampleCountFlags::TYPE_1,
                hr,
            )
        );

        // Fog (only when the world declared a VolumetricFog). Rebuilds both the
        // fullscreen render pipeline and the froxel-volume compute kernel; the
        // trailing `.map` tuples them into one Result for the macro.
        let fog_pipelines = rebuild_if_live!(self.fog.resources.is_some(), {
            let fog = self.fog.resources.as_ref().expect("fog resources are live");
            let render = super::fog::rebuild_fog_pipeline(
                device,
                fog,
                self.msaa_samples != vk::SampleCountFlags::TYPE_1,
                hr,
            )?;
            super::fog::rebuild_fog_froxel_pipeline(device, fog, hr).map(|froxel| (render, froxel))
        });

        // SSAO (only when PostProcessConfig opted in). Rebuilds prepass
        // static / instanced / skinned + kernel + blur in one shot.
        let ssao_rebuilt = rebuild_if_live!(
            self.ssao.is_some(),
            rebuild_ssao_pipelines(
                device,
                self.ssao.as_ref().expect("SSAO resources are live"),
                hr
            )
        );

        // SSR (only when PostProcessConfig opted in). Rebuilds prepass
        // static / instanced / skinned + resolve in one shot.
        let ssr_rebuilt = rebuild_if_live!(
            self.ssr.is_some(),
            rebuild_ssr_pipelines(
                device,
                self.ssr.as_ref().expect("SSR resources are live"),
                hr
            )
        );

        // SSGI (only when indirect_lighting: ssgi). Rebuilds gather + composite.
        let ssgi_rebuilt = rebuild_if_live!(
            self.ssgi.is_some(),
            crate::vulkan::post::ssgi::rebuild_ssgi_pipelines(
                device,
                self.ssgi.as_ref().expect("SSGI resources are live"),
                hr,
            )
        );

        // RT reflections (only when the world opted in + the GPU supports it).
        // Rebuilds the flat + textured ray-query pipelines.
        let rt_rebuilt = rebuild_if_live!(
            self.rt_reflections.is_some(),
            crate::vulkan::post::rt_reflections::rebuild_rt_pipelines(
                device,
                self.rt_reflections
                    .as_ref()
                    .expect("RT reflection resources are live"),
                hr,
            )
        );

        // Reflection composite (only when a reflection path owns the scene image).
        // Rebuilds the roughness blur + composite pipelines.
        let reflection_composite_rebuilt = rebuild_if_live!(
            self.reflection_composite.is_some(),
            crate::vulkan::post::reflection_composite::rebuild_reflection_composite_pipelines(
                device,
                self.reflection_composite
                    .as_ref()
                    .expect("reflection composite resources are live"),
                hr,
            )
        );

        // TAA (only when PostProcessConfig opted in). Rebuilds the resolve
        // pipeline; the velocity channel lives on the unified G-buffer pre-pass.
        let taa_rebuilt = rebuild_if_live!(
            self.taa.is_some(),
            rebuild_taa_pipelines(
                device,
                self.taa.as_ref().expect("TAA resources are live"),
                hr
            )
        );

        // Particles (only when ≥1 emitter is live or has ever been
        // added at runtime). Rebuilds the compute + render pipelines in
        // one shot.
        let particle_rebuilt = rebuild_if_live!(
            self.particle.resources.is_some(),
            self.particle
                .resources
                .as_ref()
                .expect("particle resources are live")
                .rebuild_pipelines(device, hr)
        );

        // All builds succeeded: swap the freshly compiled pipelines in. Each
        // assignment drops the pipeline it displaces, which retires it through
        // the device's queue rather than destroying it under a submission that
        // may still name it.
        self.composite.pipeline = composite_pipeline;

        if let Some(new_pipeline) = text_pipeline {
            self.text.pipeline = Some(new_pipeline);
        }

        self.bloom.pipeline_prefilter = bloom_prefilter;
        self.bloom.pipeline_downsample = bloom_downsample;
        self.bloom.pipeline_upsample = bloom_upsample;

        if let Some((new_pipeline, engine_pair)) = bindless_main_pipeline {
            self.cull.bindless_pipeline = Some(new_pipeline);
            self.cull.bindless_main_spv = engine_pair;
        }
        // The wireframe twins were built from the pre-reload shaders; drop them
        // so the next wireframe frame rebuilds against these.
        self.invalidate_wireframe_pipelines();
        if let Some(new_pipeline) = cull_pipeline {
            self.cull.cull_pipeline = Some(new_pipeline);
        }
        if let Some(new_pipeline) = cull_pipeline_phase2 {
            self.cull.cull_pipeline_phase2 = Some(new_pipeline);
        }
        if let (Some((init, downsample)), Some(hiz)) = (hiz_pipelines, self.cull.hiz.as_mut()) {
            hiz.swap_pipelines(init, downsample);
        }

        if let (Some((build, average)), Some(ae)) = (
            auto_exposure_pipelines,
            self.auto_exposure.resources.as_mut(),
        ) {
            ae.swap_pipelines(build, average);
        }

        if let (Some(new_pipeline), Some(decals)) = (decal_pipeline, self.decal.resources.as_mut())
        {
            decals.pipeline = new_pipeline;
        }
        if let (Some(new_pipeline), Some(lines)) = (line_pipeline, self.lines.resources.as_mut()) {
            lines.pipeline = new_pipeline;
        }
        if let (Some((render, froxel)), Some(fog)) = (fog_pipelines, self.fog.resources.as_mut()) {
            fog.pipeline = render;
            fog.froxel_pipeline = froxel;
        }
        if let (Some(rebuilt), Some(ssao)) = (ssao_rebuilt, self.ssao.as_mut()) {
            ssao.swap_pipelines(rebuilt);
        }
        if let (Some(rebuilt), Some(ssr)) = (ssr_rebuilt, self.ssr.as_mut()) {
            ssr.swap_pipelines(rebuilt);
        }
        if let (Some(rebuilt), Some(ssgi)) = (ssgi_rebuilt, self.ssgi.as_mut()) {
            ssgi.swap_pipelines(rebuilt);
        }
        if let (Some(rebuilt), Some(rt)) = (rt_rebuilt, self.rt_reflections.as_mut()) {
            rt.swap_pipelines(rebuilt);
        }
        if let (Some(rebuilt), Some(rc)) = (
            reflection_composite_rebuilt,
            self.reflection_composite.as_mut(),
        ) {
            rc.swap_pipelines(rebuilt);
        }
        if let (Some(rebuilt), Some(taa)) = (taa_rebuilt, self.taa.as_mut()) {
            taa.swap_pipelines(rebuilt);
        }
        if let (Some((cp, rp)), Some(p)) = (particle_rebuilt, self.particle.resources.as_mut()) {
            p.swap_pipelines(cp, rp);
        }
        Ok(())
    }

    // Rebuild bucket 0 of the GPU-driven main pass from a freshly compiled world
    // Shader and hot-swap it. Driven by asset hot-reload (`cn debug` only) when
    // one of the Shader's files is saved or the debug-WS `reload-assets`
    // command fires. Mirrors the rebuild-then-swap safety pattern of
    // `reload_shaders`: the replacement is constructed first and the swap only
    // runs when the build succeeds, so a typo in a shader edit leaves the live
    // pipeline untouched and the session keeps rendering. Sibling of
    // `DxContext::update_world_shader_pipelines` /
    // `MtlContext::update_world_shader_pipelines`.
    //
    // Buckets past 0 and the shadow / G-buffer / cull pipelines are
    // engine-internal or scene-owned and are not rebuilt here.
    //
    // Reached only through the bin's `cn debug` runtime-mutation path (dead
    // from the FFI lib crate's roots, live in the concinnity binary), like the
    // other runtime-mutation methods on `VkContext`.
    pub(crate) fn update_world_shader_pipelines(
        &mut self,
        programs: &concinnity_core::components::ShaderPrograms,
    ) -> Result<(), String> {
        let new_main =
            self.build_world_main_pipeline(Some(programs), &self.cull.bindless_main_spv)?;
        // Drain the GPU before destroying the displaced pipeline so no in-flight
        // command buffer still references it: the debug hot-reload drive does
        // not `wait_idle` for us, unlike the built-in `reload_shaders` path the
        // draw loop guards.
        self.wait_idle();
        self.cull.bindless_pipeline = Some(new_main);
        self.world_shader = Some(programs.clone());
        self.invalidate_wireframe_pipelines();
        Ok(())
    }

    // Bucket 0's pipeline against the live bindless layout: the world default
    // Shader's pair where `world` declares one, `engine_pair` otherwise. Errors
    // when the GPU-driven pass is not live, which means the world has nothing
    // to draw.
    fn build_world_main_pipeline(
        &self,
        world: Option<&concinnity_core::components::ShaderPrograms>,
        engine_pair: &(Vec<u8>, Vec<u8>),
    ) -> Result<crate::vulkan::owned::OwnedPipeline, String> {
        let layout = self
            .cull
            .bindless_pipeline_layout
            .as_ref()
            .ok_or_else(|| "the GPU-driven main pass is not live".to_string())?;
        build_bucket_pipeline(
            &self.device,
            BucketPipelineTargets {
                render_pass: self.main_render_pass.handle(),
                layout: layout.handle(),
                msaa_samples: self.msaa_samples,
                swapchain_format: self.swapchain.format,
                hot_reload: self.hot_reload.enabled,
                pool_size: self.cull.bindless_pool_size,
                probe_count: self.descriptors.probe_cube_count as usize,
            },
            0,
            crate::gfx::backend_init::WorldShader {
                programs: world,
                deferred: false,
            },
            engine_pair,
        )
    }
}

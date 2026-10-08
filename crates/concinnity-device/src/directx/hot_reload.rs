//! D3D12 shader hot-reload: `DxContext::reload_shaders` rebuilds every live
//! built-in PSO from the checkout's shader sources. The `reload-shaders` debug
//! command sets the shared flag, and the main thread polls it at the top of
//! `draw_frame`.
//!
//! Entirely a dev-loop concern: the flag exists only when `DxContext::new` is
//! called with `hot_reload = true`. Production `cn run` never sets it. Mirrors
//! src/metal/hot_reload.rs.

use concinnity_core::render::backend_init;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::world_pipelines::{check_rebuild, replace_bucket};
use std::sync::atomic::Ordering;

use super::context::DxContext;
use super::init::pipelines::{BucketPipelineTargets, BucketPsos, build_bucket_pipeline};

// Shader hot-reload state. `enabled` is true only under `cn debug`: it routes
// every built-in shader source resolve through the disk-first path (false
// under `cn run`, where the embedded source is the only one). `reload_pending`
// is the atomic flag the debug `reload-shaders` command sets, polled at the top
// of `draw_frame` to trigger a PSO rebuild; `Some` only when `enabled`, and the
// debug server reads its `Arc` clone via `GraphicsSystem`.
pub(in crate::directx) struct HotReloadState {
    pub enabled: bool,
    pub reload_pending: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    // Bumped by every engine-template reload, so a world Shader pipeline a
    // worker built from the templates before it is never installed after it.
    pub generation: u64,
}

impl HotReloadState {
    pub(super) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            reload_pending: enabled
                .then(|| std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false))),
            generation: 0,
        }
    }
}

// Rebuild a feature's PSO(s) into a temporary only when `$cond` says the
// feature is live, propagating any compile/create error out of the enclosing
// `reload_shaders`. Expands to `Some(build?)` when live, `None` otherwise. A
// feature whose resources sit behind an `Option` maps over that `Option`
// instead, so its build borrows them directly. Mirrors
// `metal/hot_reload.rs::rebuild_if_live!`.
macro_rules! rebuild_if_live {
    ($cond:expr_2021, $build:expr_2021 $(,)?) => {
        if $cond { Some($build?) } else { None }
    };
}

impl DxContext {
    // True when the shared shader-reload flag is set. Cheap atomic load;
    // called at the top of `draw_frame`. Returns false when hot-reload is
    // off so the production path never enters the reload branch.
    pub(super) fn shader_reload_requested(&self) -> bool {
        self.hot_reload
            .reload_pending
            .as_ref()
            .map(|f| f.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    // Clear the pending-reload flag. Called after `reload_shaders`
    // regardless of outcome so a failed rebuild does not loop forever.
    pub(super) fn clear_shader_reload_flag(&self) {
        if let Some(flag) = &self.hot_reload.reload_pending {
            flag.store(false, Ordering::SeqCst);
        }
    }

    // Rebuild every built-in D3D12 PSO from disk-resident source. Each PSO
    // is constructed into a temporary first; only when every rebuild
    // succeeds does the context atomically swap them in. Any HLSL compile
    // or PSO-create failure logs the underlying message and leaves the
    // live pipelines untouched; a typo in a shader edit won't crash the
    // running session.
    //
    // Covers every runtime-bundled PSO: composite, text, bloom (prefilter
    // / downsample / upsample), GPU-cull compute, auto-exposure (build +
    // average), projected-decal, transparent (glass + water), volumetric-fog, the
    // sky, the G-buffer pre-pass with the sky's motion behind it, SSAO (depth
    // copy, kernel, blur), SSR (resolve), the reflection composite (blur,
    // composite), TAA (resolve), and every shader bucket of the GPU-driven main
    // pass when it is live, each from its world Shader's pair where the world
    // declares one. The shadow PSO is out of scope here.
    pub(super) fn reload_shaders(&mut self) -> RenderResult<()> {
        if !self.hot_reload.enabled {
            return Ok(());
        }
        self.hot_reload.generation += 1;
        let device = &self.hw.device;
        let info_queue = self.hw.info_queue.as_ref();
        let hr = true;

        // Build every replacement into a temporary first. A `?` early-return
        // here means we never overwrite a live pipeline with a failed build:
        // any compile error leaves the running session rendering with the
        // previous shader source.

        // Composite (always live).
        let (composite_vs, composite_ps) = super::pipeline::compile_composite_shaders(hr)?;
        let composite_pso = super::context::dump_on_err(
            info_queue,
            super::pipeline::create_composite_pso(
                device,
                &self.composite.root_sig,
                &composite_vs,
                &composite_ps,
                self.swapchain.format,
            ),
        )?;

        // Text (only when the world declared text atlases).
        let text_pso = rebuild_if_live!(self.text.pso.is_some(), {
            let (text_vs, text_ps) = super::pipeline::compile_text_shaders(hr)?;
            super::context::dump_on_err(
                info_queue,
                super::pipeline::create_text_pso(
                    device,
                    &self.text.root_sig,
                    &text_vs,
                    &text_ps,
                    self.swapchain.format,
                    1,
                ),
            )
        });

        // Bloom (live outside a resize).
        let bloom_rebuilt = rebuild_if_live!(
            self.bloom.is_some(),
            concinnity_core::render::post::bloom::build_pipelines(&self.post_device(0))
        );

        // Every shader bucket's main and G-buffer pre-pass PSOs, from the
        // engine's freshly compiled programs; a world Shader's own files are
        // spliced into the same templates, so they are rebuilt against them
        // too. Each bucket rebuilds as a pair, so shading and the G-buffer
        // never compile from different templates.
        let bindless_main_pso = rebuild_if_live!(
            self.cull.main_bindless_root_sig.is_some() && self.cull.main_bindless_pso.is_some(),
            {
                let engine = super::init::pipelines::compile_main_bindless_shaders(hr)?;
                let psos = self.build_world_main_pso(self.world_shader.as_ref(), &engine)?;
                let world_buckets = self.rebuild_world_buckets(&engine)?;
                Ok::<_, RenderError>((psos, world_buckets, engine))
            }
        );
        // The cull PSO, and its phase-2 twin (two-pass occlusion) when built,
        // both against the shared root signature.
        let cull_psos = self
            .cull
            .cull_kernels
            .as_ref()
            .map(|kernels| {
                let create = |cs: &[u8]| {
                    super::context::dump_on_err(
                        info_queue,
                        super::cull::create_cull_pso(device, &kernels.root_sig, cs),
                    )
                };
                let pso = create(&super::cull::compile_cull_shader(hr)?)?;
                let phase2 = kernels
                    .pso_phase2
                    .as_ref()
                    .map(|_| create(&super::cull::compile_cull_shader_phase2(hr)?))
                    .transpose()?;
                Ok::<_, RenderError>((pso, phase2))
            })
            .transpose()?;

        // Hi-Z (only when cull pipeline is live; same gating condition).
        // Rebuilds all three SPD kernels against the existing root signatures
        // so the live cull root binding stays valid. The tail takes the
        // signature without the depth SRV table.
        let hiz_rebuilt = if let Some(hiz) = self.cull.hiz.as_ref() {
            let (spd_single_cs, spd_msaa_cs, spd_tail_cs) = super::hiz::compile_hiz_shaders(hr)?;
            let spd_single_pso = super::context::dump_on_err(
                info_queue,
                super::pso::compute_pso(device, &hiz.root_sig, &spd_single_cs, "hiz spd_single"),
            )?;
            let spd_msaa_pso = super::context::dump_on_err(
                info_queue,
                super::pso::compute_pso(device, &hiz.root_sig, &spd_msaa_cs, "hiz spd_msaa"),
            )?;
            let spd_tail_pso = super::context::dump_on_err(
                info_queue,
                super::pso::compute_pso(device, &hiz.tail_root_sig, &spd_tail_cs, "hiz spd_tail"),
            )?;
            Some((spd_single_pso, spd_msaa_pso, spd_tail_pso))
        } else {
            None
        };

        // Auto-exposure (gated on the post-process config).
        let (auto_exp_build, auto_exp_average) = if let Some(ae) =
            self.auto_exposure.resources.as_ref()
        {
            let (build_cs, average_cs) = super::auto_exposure::compile_auto_exposure_shaders(hr)?;
            let build_pso = super::context::dump_on_err(
                info_queue,
                super::pso::compute_pso(
                    device,
                    ae.build_root_sig(),
                    &build_cs,
                    "auto-exposure build",
                ),
            )?;
            let average_pso = super::context::dump_on_err(
                info_queue,
                super::pso::compute_pso(
                    device,
                    ae.average_root_sig(),
                    &average_cs,
                    "auto-exposure average",
                ),
            )?;
            (Some(build_pso), Some(average_pso))
        } else {
            (None, None)
        };

        // Decal (always built when DecalResources exists, which is unconditional).
        let msaa_samples = self.targets.hdr.msaa_samples;
        let sky_pso = super::context::dump_on_err(
            info_queue,
            super::sky::build_sky_pso(device, self.sky.root_sig(), msaa_samples, hr),
        )?;
        // The sky's motion behind the G-buffer pre-pass; the pre-pass's own
        // PSOs rebuild with the main pass's above.
        let gbuffer_sky_pso = self
            .gbuffer
            .as_ref()
            .map(|gb| gb.sky.rebuild_pso(device, info_queue))
            .transpose()?;
        let decal_pso = self
            .decal
            .state
            .as_ref()
            .map(|decals| {
                super::decal::rebuild_decal_pso(
                    device,
                    &decals.root_sig,
                    msaa_samples,
                    hr,
                    info_queue,
                )
            })
            .transpose()?;

        // Lines (only once a frame published some and the lazy build ran).
        let line_pso = self
            .lines
            .resources
            .as_ref()
            .map(|lines| super::line::rebuild_line_pso(device, lines, msaa_samples, hr, info_queue))
            .transpose()?;

        // The transparent pass's two producers (each only when the world declared
        // its asset). Both rebuild against the pass's shared root signature.
        let glass_pso = self
            .transparent
            .as_ref()
            .filter(|t| t.has_glass())
            .map(|t| {
                super::glass::rebuild_glass_pso(device, t.root_sig(), msaa_samples, hr, info_queue)
            })
            .transpose()?;
        let water_pso = self
            .transparent
            .as_ref()
            .filter(|t| t.has_water())
            .map(|t| {
                super::water::rebuild_water_pso(device, t.root_sig(), msaa_samples, hr, info_queue)
            })
            .transpose()?;

        // Fog (only when the world declared a VolumetricFog). Both the
        // render PSO (fragment volume sampler) and the compute PSO
        // (froxel-volume kernel) rebuild from the same `fog.metal`-style
        // source pair.
        let fog_psos = self
            .fog
            .resources
            .as_ref()
            .map(|fog| {
                let render = super::fog::rebuild_fog_pso(
                    device,
                    &fog.root_sig,
                    msaa_samples,
                    hr,
                    info_queue,
                )?;
                super::fog::rebuild_fog_froxel_pso(device, &fog.froxel_root_sig, hr, info_queue)
                    .map(|froxel| (render, froxel))
            })
            .transpose()?;

        // SSAO (only when PostProcessConfig opted in).
        let ssao_rebuilt = rebuild_if_live!(
            self.ssao.resources.is_some(),
            concinnity_core::render::post::ssao::build_pipelines(&self.post_device(0))
        );

        // SSR (only when the resolve itself is authored).
        let ssr_rebuilt = rebuild_if_live!(
            self.ssr.as_ref().is_some_and(|s| s.resolve.is_some()),
            concinnity_core::render::post::ssr::build_pipeline(&self.post_device(0))
        );

        // SSGI (only when PostProcessConfig.indirect_lighting == ssgi).
        let ssgi_rebuilt = rebuild_if_live!(
            self.ssgi.is_some(),
            concinnity_core::render::post::ssgi::build_pipelines(&self.post_device(0))
        );

        // TAA (only when PostProcessConfig.aa_mode).
        let taa_rebuilt = rebuild_if_live!(
            self.taa.is_some(),
            concinnity_core::render::post::taa::build_pipeline(&self.post_device(0))
        );

        // RT reflections (only when DXR + DXC compile + accel build all succeeded
        // at init). The shader compiles through DXC (SM 6.5).
        let rt_rebuilt = self
            .rt_reflections
            .as_ref()
            .map(|rt| {
                super::post::rt_reflections::rebuild_rt_reflections_pipelines(
                    device, rt, hr, info_queue,
                )
            })
            .transpose()?;

        // Reflection composite (blur + composite PSOs); only when built.
        let refl_composite_rebuilt = rebuild_if_live!(
            self.reflection_composite.is_some(),
            concinnity_core::render::post::reflection_composite::build_pipelines(
                &self.post_device(0)
            )
        );

        // All builds succeeded; swap into the live context. After this
        // point the next frame's draw calls bind the freshly compiled
        // pipelines.
        self.composite.pso = composite_pso;
        if let Some(p) = text_pso {
            self.text.pso = Some(p);
        }
        if let (Some(rebuilt), Some(bloom)) = (bloom_rebuilt, self.bloom.as_mut()) {
            bloom.swap_pipelines(rebuilt);
        }
        if let Some((psos, world_buckets, engine)) = bindless_main_pso {
            self.cull.main_bindless_pso = Some(psos.main);
            self.cull.main_prepass_pso = psos.prepass;
            self.swap_world_buckets(world_buckets);
            self.cull.bindless_main_shaders = engine;
        }
        self.sky.swap_pso(sky_pso);
        if let (Some(sky), Some(gb)) = (gbuffer_sky_pso, self.gbuffer.as_mut()) {
            gb.sky.swap_pso(sky);
        }
        // The wireframe twins were built from the pre-reload shaders; drop them
        // so the next wireframe frame rebuilds against these.
        self.invalidate_wireframe_pipelines();
        if let (Some((pso, phase2)), Some(kernels)) = (cull_psos, self.cull.cull_kernels.as_mut()) {
            kernels.pso = pso;
            kernels.pso_phase2 = phase2;
        }
        if let (Some((single, msaa, tail)), Some(hiz)) = (hiz_rebuilt, self.cull.hiz.as_mut()) {
            hiz.swap_pipelines(single, msaa, tail);
        }
        if let (Some(build), Some(average), Some(ae)) = (
            auto_exp_build,
            auto_exp_average,
            self.auto_exposure.resources.as_mut(),
        ) {
            ae.swap_pipelines(build, average);
        }
        if let (Some(pso), Some(decals)) = (decal_pso, self.decal.state.as_mut()) {
            decals.pso = pso;
        }
        if let (Some(pso), Some(lines)) = (line_pso, self.lines.resources.as_mut()) {
            lines.pso = pso;
        }
        if let Some(transparent) = self.transparent.as_mut() {
            transparent.swap_pipelines(glass_pso, water_pso);
        }
        if let (Some((render, froxel)), Some(fog)) = (fog_psos, self.fog.resources.as_mut()) {
            fog.pso = render;
            fog.froxel_pso = froxel;
        }
        if let (Some(rebuilt), Some(ssao)) = (ssao_rebuilt, self.ssao.resources.as_mut()) {
            ssao.swap_pipelines(rebuilt);
        }
        if let (Some(rebuilt), Some(ssr)) = (ssr_rebuilt, self.ssr.as_mut()) {
            ssr.swap_pipeline(rebuilt);
        }
        if let (Some(rebuilt), Some(ssgi)) = (ssgi_rebuilt, self.ssgi.as_mut()) {
            ssgi.swap_pipelines(rebuilt);
        }
        if let (Some(rebuilt), Some(rt)) = (rt_rebuilt, self.rt_reflections.as_mut()) {
            super::post::rt_reflections::swap_rt_reflections_pipelines(rt, rebuilt);
        }
        if let (Some(rebuilt), Some(rc)) =
            (refl_composite_rebuilt, self.reflection_composite.as_mut())
        {
            rc.swap_pipelines(rebuilt);
        }
        if let (Some(rebuilt), Some(taa)) = (taa_rebuilt, self.taa.as_mut()) {
            taa.pass.swap_pipeline(rebuilt);
        }
        Ok(())
    }
}

// World-Shader runtime hot-swap (RenderBackend::update_world_shader)

// cn-debug-only runtime-mutation surface; dead from the FFI lib crate's roots,
// live in the concinnity binary. See the note on the analogous block in
// [directx/particle.rs].
impl DxContext {
    // Rebuild bucket 0 of the GPU-driven main pass and its G-buffer pre-pass
    // from the world default Shader's freshly compiled programs and hot-swap
    // them, for `update_world_shader`, or swap in `prepared` when a worker
    // already built them. The replacements are built first; a compile /
    // PSO-create failure early-returns with the live pipelines untouched,
    // mirroring `reload_shaders`.
    pub(in crate::directx) fn update_default_world_shader(
        &mut self,
        programs: &concinnity_core::components::ShaderPrograms,
        prepared: Option<BucketPsos>,
    ) -> RenderResult<()> {
        let new = replace_bucket(
            0,
            self.cull.main_prepass_pso.is_some(),
            prepared,
            |p: &BucketPsos| p.prepass.is_some(),
            || self.build_world_main_pso(Some(programs), &self.cull.bindless_main_shaders),
        )?;
        // Drain the GPU before the swap releases the displaced PSOs: a command
        // list does not keep one alive, and the debug reload drive does not
        // wait for us.
        self.wait_idle();
        self.cull.main_bindless_pso = Some(new.main);
        self.cull.main_prepass_pso = new.prepass;
        self.world_shader = Some(programs.clone());
        self.invalidate_wireframe_pipelines();
        Ok(())
    }

    // Bucket 0's PSOs against the live root signatures: the world default
    // Shader's programs where `world` declares one, `engine_default` otherwise.
    // Errors when the GPU-driven pass is not live, which means the world has
    // nothing to draw, when the main PSO fails, or when the rebuild would drop
    // the live pre-pass, so a live pair is kept whole.
    pub(super) fn build_world_main_pso(
        &self,
        world: Option<&concinnity_core::components::ShaderPrograms>,
        engine_default: &super::init::pipelines::BindlessMainShaders,
    ) -> RenderResult<BucketPsos> {
        let targets = self
            .world_pso_targets()
            .ok_or_else(|| RenderError::Other("the GPU-driven main pass is not live".into()))?;
        let psos = build_bucket_pipeline(
            &self.hw.device,
            self.hw.info_queue.as_ref(),
            BucketPipelineTargets {
                root_sigs: &targets.root_sigs,
                msaa_samples: self.targets.hdr.msaa_samples,
                engine_default,
                hot_reload: self.hot_reload.enabled,
            },
            0,
            backend_init::WorldShader {
                programs: world,
                deferred: false,
            },
        )?;
        check_rebuild(
            0,
            self.cull.main_prepass_pso.is_some(),
            psos.prepass.is_some(),
        )?;
        Ok(psos)
    }
}

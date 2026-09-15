//! Runtime application of the Quality-group settings (TAA / SSAO / SSR / RT
//! reflections / SSGI / auto-exposure). Each gates a render pass whose GPU
//! resources (pipelines, render targets, the ray-tracing acceleration structure)
//! are built once at init from the world's PostProcessConfig, so applying a
//! change at runtime means rebuilding those resources, not flipping a uniform.
//!
//! The rebuild reuses the init stages' effect builders -- the exact path
//! `MtlContext::new` runs -- so a live toggle produces resources byte-identical
//! to a launch with the same config. Only the toggle-controlled subset is rebuilt;
//! bloom, decals, fog, particles, and the uploaded geometry are untouched (so no
//! particle-sim reset and no multi-second geometry re-upload).

use concinnity_core::components;
use concinnity_core::gfx::render_types;
use concinnity_core::render::backend;
use concinnity_core::render::backend::QualitySettings;
use concinnity_core::render::error::{RenderError, RenderResult};

use super::auto_exposure::AutoExposureGpu;
use super::context::MtlContext;
use super::init::effects::{
    EffectSettings, build_auto_exposure, build_gbuffer, build_ssao, build_ssgi, build_ssr,
    build_taa,
};
use super::init::ray_tracing::build_rt_pipelines;
use super::init::targets::build_transient_pool;
use super::post::post_device::MtlPostDevice;
use super::post::{GBufferState, SsaoState, SsgiState, SsrState, TaaState};
use super::raytrace::{
    RtGpu, RtPipelines, RtSceneGeometry, RtStaticGeometry, RtTextureCounts, build_rt_accel,
    raytracing_supported,
};
use super::transient_pool::TransientTexturePool;

// The toggle-controlled subset of the effects stack: the features the Quality
// settings group switches on and off at runtime (TAA, SSAO, SSR, SSGI, RT
// reflection pipelines, auto-exposure) plus the resources they share (the
// G-buffer pre-pass + the transient pool). Bloom, decals, fog, and particles are
// NOT here: bloom is always on (only its uniforms change, live), and
// decals/fog/particles are world-content effects a quality toggle never affects
// (and rebuilding particles would reset their live GPU pools).
struct QualityEffects {
    taa: TaaState,
    ssao: SsaoState,
    transient_pool: TransientTexturePool,
    ssr: SsrState,
    gbuffer: GBufferState,
    ssgi: SsgiState,
    rt: RtPipelines,
    auto_exposure: AutoExposureGpu,
}

impl MtlContext {
    // Turn display sync (vsync) on or off at runtime via the view's backing
    // CAMetalLayer. Setting displaySyncEnabled is an idempotent property write
    // (no swapchain rebuild on Metal), so a redundant call is cheap. Backend
    // specific: Vulkan reaches the same end by rebuilding the swapchain with a
    // different present mode, so this does not live on the shared window layer.
    pub(crate) fn set_vsync(&mut self, on: bool) {
        super::init::set_display_sync(&self.window().view, on);
    }

    // Replace the live post-process tunables. They are pushed to the bloom
    // prefilter + composite shaders every frame (see draw/composite.rs), so a
    // change takes effect on the next draw with no allocation or pipeline
    // rebuild. The composite's display-output flags are not part of the payload,
    // so the EDR path negotiated at init survives every push. Auto-exposure,
    // when on, overwrites `exposure` each frame from the adapted EV, so a static
    // exposure change is only visible with auto-exposure off.
    pub(crate) fn update_post_process(&mut self, tunables: render_types::PostProcessTunables) {
        self.post_process.set_tunables(tunables);
    }

    // Set the live ambient (IBL) light scale. `ambient_intensity` lives in
    // `LightUniforms`, which the main lighting pass uploads every frame, so the
    // change takes effect on the next draw with no allocation. It is not
    // re-derived per frame (unlike auto-exposure's `exposure`), so the value
    // stands until changed again.
    pub(crate) fn set_ambient_intensity(&mut self, value: f32) {
        self.light_uniforms.ambient_intensity = value;
    }

    // Set the live shadow cascade re-render cadence. The scheduler reads
    // `shadow.update` at the start of each shadow pass, so a change takes effect
    // on the next draw. Every cascade is already primed, so switching policy never
    // leaves a slice unsampled (priming is one-shot per cascade, not per policy).
    pub(crate) fn set_shadow_update(&mut self, update: components::ShadowUpdate) {
        self.shadow.update = update;
    }

    // Set the live shadow distance (world units). The per-frame cascade-split
    // computation reads `shadow.distance` each draw, so a change takes effect on
    // the next frame with no allocation (it sizes no GPU resource).
    pub(crate) fn set_shadow_distance(&mut self, distance: u32) {
        self.shadow.distance = distance;
    }

    // Set the live shadow cascade count (1..=4). The per-frame split + schedule
    // read `shadow.cascades` each draw; only the first `count` of the four slots
    // are rendered + sampled, so a change takes effect on the next frame with no
    // resize (the shadow-map array stays sized for the 4-cascade capacity).
    pub(crate) fn set_shadow_cascades(&mut self, count: u32) {
        self.shadow.cascades = count;
    }

    // Update the live scalar sub-tunables of the SSAO / SSR / SSGI / auto-exposure
    // passes without rebuilding anything. The draw path rebuilds each pass's
    // per-frame uniform from these stored `*Settings` structs every frame
    // (`settings.params(...)`), so mutating the stored struct here is picked up on
    // the next draw. Only a feature that is currently on has a settings struct to
    // mutate; the rest are skipped (the value still persists for the next launch).
    // SSAO / SSR / auto-exposure settings are fully scalar, so they are replaced
    // wholesale; SSGI keeps its gather resolution / ray / step counts (those size
    // the gather target or ride `apply_quality_settings`), so only its scalar
    // intensity / distance are updated.
    pub(crate) fn update_quality_params(&mut self, q: backend::QualitySettings) {
        if let (Some(live), Some(cur)) = (q.ssao, self.ssao.settings.as_mut()) {
            *cur = live;
        }
        if let (Some(live), Some(cur)) = (q.ssr, self.ssr.settings.as_mut()) {
            *cur = live;
        }
        if let (Some(live), Some(cur)) = (q.ssgi, self.ssgi.settings.as_mut()) {
            cur.intensity = live.intensity;
            cur.max_distance = live.max_distance;
        }
        if let (Some(live), Some(cur)) = (q.auto_exposure, self.auto_exposure.settings.as_mut()) {
            *cur = live;
        }
    }

    // Rebuild the toggle-controlled effects in place to match `q`, applied
    // between frames (the GraphicsSystem drain runs before the next
    // `draw_frame`). An effect build failure returns before anything is
    // swapped; a later failure keeps the rebuilt state and returns the first
    // error once the rest has been applied.
    pub(crate) fn apply_quality_settings(&mut self, q: QualitySettings) -> RenderResult<()> {
        // RT reflections only when the GPU supports hardware ray tracing;
        // otherwise the toggle persists + value-syncs but renders nothing,
        // matching the init-time fallback. The refusal is reported the way the
        // other two backends report theirs, so a request driven from the debug
        // port or a persisted settings file does not read as a success.
        let rt_capable = raytracing_supported(&self.hw.device);
        if q.rt_reflections.is_some() && !rt_capable {
            tracing::warn!(
                "ray-traced reflections requested but the device does not support \
                 hardware ray tracing; keeping SSR"
            );
        }
        let rt_settings = q.rt_reflections.filter(|_| rt_capable);

        // TAA is bypassed while the MetalFX upscaler is active (the scaler does
        // its own temporal accumulation); the velocity pre-pass + G-buffer are
        // needed when TAA is effectively on OR the upscaler is active. Mirrors
        // the `effective_taa_enabled` / `velocity_needed` derivation in
        // `MtlContext::new`. Render dimensions come from the live HDR targets
        // (render-resolution, already post-upscale).
        let upscaling_active = self.upscale.scaler.is_some();
        let taa_effective = q.taa && !upscaling_active;
        let needs_velocity = taa_effective || upscaling_active;
        let settings = EffectSettings {
            ssao: &q.ssao,
            ssr: &q.ssr,
            ssgi: &q.ssgi,
            rt_reflection: &rt_settings,
            auto_exposure: &q.auto_exposure,
            reflection_blur_scale: q.reflection_blur_scale,
            auto_exposure_bias_ev: q.auto_exposure_bias_ev,
        };

        let effects = self.build_quality_effects(&settings, taa_effective, needs_velocity)?;
        let mut first_err: Option<RenderError> = None;

        // Swap the screen-space feature state in. The old `Retained` targets drop
        // here; any in-flight command buffer still referencing them holds its own
        // Metal retain until the GPU retires the frame, so the swap is safe
        // between frames. The render graph is rebuilt from these gates every
        // frame (no cached graph to invalidate).
        // A rebuilt pass starts with its ring at slot 0 and its history
        // invalid, so the first frame after a toggle passes through.
        self.taa.enabled = effects.taa.enabled;
        self.taa.pass = effects.taa.pass;
        self.ssao = effects.ssao;
        self.targets.transient_pool = effects.transient_pool;
        // The rebuilt pool holds a fresh `bloom_top`, so the bloom chain's top
        // mip (a handle into the old pool) is stale. Re-point it rather than
        // rebuilding the chain: the extent is unchanged, so the mips below it
        // are still correct.
        match self.targets.transient_pool.bloom_top() {
            Ok(top) => self.targets.bloom.mips[0] = top,
            Err(e) => first_err = Some(format!("bloom top mip: {e}").into()),
        }
        self.ssr = effects.ssr;
        self.gbuffer = effects.gbuffer;
        self.ssgi = effects.ssgi;

        // RT resolve pipelines come from the rebuild; the acceleration structure
        // is built here (it needs the resident geometry buffers) when RT turns
        // on, and dropped when it turns off. Skinned geometry is seeded into the
        // BVH by the next frame's per-frame update, matching the init path.
        self.rt.settings = rt_settings;
        self.rt.pipelines = effects.rt;
        if self.rt.settings.is_some() {
            if self.rt.accel.is_none() {
                match build_rt_accel(
                    RtGpu {
                        device: &self.hw.device,
                        command_queue: &self.hw.command_queue,
                        frames_in_flight: self.frames_in_flight,
                    },
                    RtStaticGeometry {
                        vertex_buffer: &self.scene.vertex_buffer,
                        index_buffer: &self.scene.index_buffer,
                    },
                    RtSceneGeometry {
                        draw_objects: &self.draw.objects,
                        clusters: &self.instanced.clusters,
                    },
                    RtTextureCounts {
                        albedo_count: self.scene.textures.len(),
                    },
                    None,
                    self.seethrough_meshes_enabled(),
                ) {
                    Ok(Some(a)) => {
                        tracing::info!(
                            "ray-traced reflections: built BVH over {} static objects",
                            a.blas.len()
                        );
                        self.rt.accel = Some(a);
                    }
                    Ok(None) => tracing::warn!(
                        "ray-traced reflections toggled on but the scene has no static geometry; no BVH built"
                    ),
                    Err(e) => {
                        let e = e.context("RT accel build");
                        if first_err.is_some() {
                            tracing::error!("apply_quality_settings: {e}");
                        } else {
                            first_err = Some(e);
                        }
                    }
                }
            }
        } else {
            self.rt.accel = None;
        }
        // Reset the failure streak so a later toggle-on starts clean.
        self.rt.update_failed = false;

        // Auto-exposure. When it turns off the static path uses
        // `self.post_process.exposure` (the authored / slider EV), already set,
        // so only the GPU state is swapped here; the frame clock carries over.
        self.auto_exposure = AutoExposureGpu {
            last_elapsed: self.auto_exposure.last_elapsed,
            ..effects.auto_exposure
        };
        first_err.map_or(Ok(()), Err)
    }

    // Build the toggle-controlled effects (see [`QualityEffects`]) in the order
    // init builds them. Render dimensions come from the live HDR targets
    // (render-resolution, already post-upscale). Output dimensions come from the
    // live bloom chain, which was built at them; the rebuilt pool sizes
    // `bloom_top` off the same pair, so the new top mip drops back into the chain
    // unchanged. The RT acceleration structure is the caller's responsibility.
    fn build_quality_effects(
        &self,
        settings: &EffectSettings,
        taa_enabled: bool,
        needs_velocity: bool,
    ) -> RenderResult<QualityEffects> {
        let device = &*self.hw.device;
        let hot_reload = self.hot_reload.enabled;
        let post_device = MtlPostDevice {
            device,
            sampler: &self.composite.sampler,
            cube_sampler: &self.scene.cube_sampler,
            probes: None,
            timing: None,
            hot_reload,
        };
        let render = (self.targets.hdr.width, self.targets.hdr.height);
        let output = (self.targets.bloom.width, self.targets.bloom.height);
        let gbuffer_enabled = settings.gbuffer_needed(needs_velocity);
        Ok(QualityEffects {
            taa: build_taa(&post_device, taa_enabled, render)?,
            ssao: build_ssao(&self.hw.allocator, settings, render, hot_reload)?,
            transient_pool: build_transient_pool(
                device,
                settings.ssao.is_some(),
                gbuffer_enabled,
                render,
                output,
            )?,
            ssr: build_ssr(&post_device, settings, render)?,
            gbuffer: build_gbuffer(device, gbuffer_enabled, render, hot_reload)?,
            ssgi: build_ssgi(&post_device, settings, render)?,
            rt: build_rt_pipelines(device, settings.rt_reflection, hot_reload)?,
            auto_exposure: build_auto_exposure(
                device,
                settings,
                self.frames_in_flight,
                hot_reload,
            )?,
        })
    }
}

//! Runtime application of the Quality-group settings (TAA / SSAO / SSR / SSGI /
//! auto-exposure). Each gates a render pass whose GPU resources (pipelines,
//! render targets, descriptor sets) are built once at init from the world's
//! PostProcessConfig, so applying a change at runtime means building or tearing
//! down those resources, not flipping a uniform.
//!
//! The reconcile below brings each feature's `Option` field to the desired state
//! (constructing a turning-on feature with the same `*Resources::new` the init
//! path runs, tearing down a turning-off one), then defers the whole target
//! rebuild + descriptor rewire to `rebuild_swapchain` -- the exact path a window
//! resize takes. Reusing it means a live toggle produces resources rewired
//! identically to a launch with the same config, with no second copy of the
//! intricate per-reader rewiring to drift. Bloom, decals, fog, particles, and
//! the uploaded geometry are untouched.
//!
//! Ray-traced reflections toggle the same way, with two extra costs: turning
//! them on builds the scene acceleration structure (`build_rt_accel`, a one-shot
//! fence-waited BLAS + TLAS over current geometry) plus the inline-`rayQueryEXT`
//! reflection pass, so a live enable hitches once proportional to triangle count.
//! And RT is only live-toggleable when the device is RT-capable -- the ray-query
//! device extensions are enabled at creation whenever capable (see
//! `create_logical_device`), since an extension cannot be added later; on an
//! RT-incapable GPU or under XeSS the toggle no-ops with a warning and RT stays
//! whatever it launched as (persisted for the next launch).

use ash::vk;
use concinnity_core::gfx::auto_exposure;
use concinnity_core::render::backend::QualitySettings;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::post::rt_reflections;
use concinnity_core::render::render_graph::{PoolGates, plan_pool_slots};

use super::context::VkContext;
use super::post::SsaoResources;

impl VkContext {
    // Bring the toggle-controlled features to match `q`, applied between frames
    // (the GraphicsSystem reads the SettingCommand before the next draw_frame).
    // A build failure returns early and leaves the prior state intact.
    pub(crate) fn apply_quality_settings(&mut self, q: QualitySettings) -> RenderResult<()> {
        // Every teardown / rebuild below frees or replaces GPU resources a prior
        // frame may still reference; drain the device first so the swap is safe.
        // Idle, a replaced resource's range is reused by its successor.
        self.wait_idle();
        let _idle = self.hw.alloc.idle_scope();

        // Desired enabled state per feature, from the resolved QualitySettings.
        // RT is additionally gated on the device being RT-capable: a non-capable
        // device (or XeSS) did not enable the ray-query extensions at creation,
        // so it cannot build the acceleration structure at runtime -- the toggle
        // no-ops with a warning and RT stays whatever it launched as.
        let rt_settings = q.rt_reflections.filter(|_| self.hw.rt_capable);
        let desired_rt = rt_settings.is_some();
        if q.rt_reflections.is_some() && !self.hw.rt_capable {
            tracing::warn!(
                "ray-traced reflections requested but the device is not RT-capable \
                 (no ray-query extensions / XeSS active); keeping SSR"
            );
        }
        let desired_ssr = q.ssr.is_some();
        let desired_ssgi = q.ssgi.is_some();
        let desired_ssao = q.ssao.is_some();
        // TAA resources are forced present while temporal upscaling is active
        // (the upscaler consumes the velocity pre-pass); the TAA resolve is then
        // dropped from the graph. Mirrors the init `taa_enabled` derivation.
        let upscale_on = self.upscale.is_some();
        let desired_taa = q.taa || upscale_on;

        // The SSR pre-pass resources (`SsrResources`) exist whenever SSR, SSGI,
        // or RT is on (SSGI and RT reuse the SSR resolve's plumbing); mirrors the
        // init `ssr_opt` gate. The unified G-buffer pre-pass is needed by any
        // screen-space consumer of the merged buffer (RT reads its per-frame
        // normal+depth and roughness).
        let ssr_needed = desired_ssr || desired_ssgi || desired_rt;
        let gbuffer_needed = ssr_needed || desired_ssao || desired_taa;

        let hdr_views: Vec<vk::ImageView> = self
            .targets
            .hdr_resolve_images
            .iter()
            .map(|i| i.view)
            .collect();

        // Unified G-buffer pre-pass (shared dependency): build it before any
        // consumer that samples it. Kept alive once built (a later toggle-off of
        // the last consumer leaves it resident until the next launch / resize),
        // which is harmless: with no consumer the graph omits its readers.
        if gbuffer_needed && self.gbuffer.is_none() {
            // Its three color channels are pool-owned, so the pool has to place
            // them before the pre-pass framebuffers can reference them. Rebuild
            // with the G-buffer gate on first; the `rebuild_swapchain` later in
            // this call rebuilds the pool once more and re-points every reader.
            // The cached framebuffers name pooled views, so they go first.
            self.post.cache.forget_views();
            self.targets.transient_pool.rebuild(
                &super::transient_pool::TransientPoolGpu {
                    instance: &self.hw.instance,
                    device: &self.hw.device,
                    physical_device: self.hw.physical_device,
                    command_pool: self.commands.command_pool,
                    queue: self.hw.graphics_queue,
                },
                self.frames_in_flight,
                &plan_pool_slots(
                    PoolGates {
                        ssao: self.ssao.is_some(),
                        gbuffer: true,
                    },
                    (
                        self.targets.render_extent.width,
                        self.targets.render_extent.height,
                    ),
                    (self.swapchain.extent.width, self.swapchain.extent.height),
                )?,
            )?;
            let pooled = self
                .targets
                .transient_pool
                .gbuffer_pooled(self.frames_in_flight);
            let gb = super::post::gbuffer::GbufferResources::new(
                super::post::gbuffer::GbufferDeviceCtx {
                    alloc: &self.hw.alloc,
                    device: &self.hw.device,
                },
                super::post::gbuffer::GbufferQueueCtx {
                    command_pool: self.commands.command_pool,
                    queue: self.hw.graphics_queue,
                },
                super::post::gbuffer::GbufferExtent {
                    width: self.targets.render_extent.width,
                    height: self.targets.render_extent.height,
                    frames: self.frames_in_flight,
                },
                &pooled,
                self.hot_reload.enabled,
            )?;
            self.gbuffer = Some(gb);
        }

        // TAA.
        if desired_taa && self.taa.is_none() {
            let taa = super::post::taa::TaaResources::new(
                &self.post_device(0),
                self.frames_in_flight,
                self.targets.render_extent,
            )?;
            self.taa = Some(taa);
        } else if !desired_taa && self.taa.is_some() {
            // The cached framebuffers name the accumulation images' views, so
            // they go before the images do.
            self.post.cache.forget_views();
            self.taa = None;
        }

        // SSR resolve + reflection target. Built whenever SSR / SSGI / RT is on;
        // its settings carry whether SSR itself is authored, so a build kept alive
        // for SSGI or RT takes an SSR toggle through them alone.
        if ssr_needed && self.ssr.is_none() {
            let ssr = super::post::ssr::SsrResources::new(
                &self.post_device(0),
                q.ssr,
                self.targets.render_extent,
            )?;
            self.ssr = Some(ssr);
        } else if !ssr_needed && self.ssr.is_some() {
            // The cached framebuffers name the reflection target's view, so they
            // go before it does.
            self.post.cache.forget_views();
            self.ssr = None;
        } else if let Some(ssr) = self.ssr.as_mut() {
            ssr.settings = q.ssr;
        }

        // SSGI (samples the unified G-buffer's per-frame normal+depth views). A
        // new trace resolution resizes every target, so it rebuilds the pass; the
        // ray count rides the settings.
        let ssgi_rescaled = match (q.ssgi, self.ssgi.as_ref()) {
            (Some(settings), Some(live)) => settings.gi_scale != live.settings.gi_scale,
            _ => false,
        };
        if let Some(settings) = q.ssgi
            && (self.ssgi.is_none() || ssgi_rescaled)
        {
            if ssgi_rescaled {
                self.post.cache.forget_views();
                self.ssgi = None;
            }
            let ssgi = super::post::ssgi::SsgiResources::new(
                &self.post_device(0),
                settings,
                self.targets.render_extent,
            )?;
            self.ssgi = Some(ssgi);
        } else if !desired_ssgi && self.ssgi.is_some() {
            self.post.cache.forget_views();
            self.ssgi = None;
        } else if let (Some(settings), Some(ssgi)) = (q.ssgi, self.ssgi.as_mut()) {
            ssgi.settings = settings;
        }

        // Auto-exposure. When it turns off the static authored EV drives exposure
        // again (the GraphicsSystem re-pushes `update_post_process` after this
        // call), so only the GPU state is swapped here.
        if let Some(settings) = q.auto_exposure.as_ref()
            && self.auto_exposure.resources.is_none()
        {
            let resources = crate::vulkan::auto_exposure::AutoExposureResources::new(
                &self.hw.alloc,
                &self.hw.device,
                self.frames_in_flight,
                &hdr_views,
                self.hot_reload.enabled,
            )?;
            self.auto_exposure.resources = Some(resources);
            self.auto_exposure.adaptation = Some(auto_exposure::ExposureAdaptation::new(
                *settings,
                q.auto_exposure_bias_ev,
            ));
        } else if q.auto_exposure.is_none()
            && let Some(mut ae) = self.auto_exposure.resources.take()
        {
            ae.destroy(&self.hw.device);
            self.auto_exposure.adaptation = None;
        }

        // SSAO. Its occlusion target is the transient pool's per-frame
        // `ao_output`, which only exists while SSAO is on; `rebuild_swapchain`
        // below rebuilds the pool from the now-toggled `self.ssao` and re-points
        // binding 6 at the rebuilt views. The pass reads its output per frame.
        match (q.ssao, self.ssao.is_some()) {
            (Some(settings), false) => {
                let ssao =
                    SsaoResources::new(&self.post_device(0), settings, self.targets.render_extent)?;
                self.ssao = Some(ssao);
            }
            (None, true) => {
                // The cached framebuffers name the raw occlusion's view, so they
                // go before it does.
                self.post.cache.forget_views();
                self.ssao = None;
            }
            _ => {}
        }

        // Ray-traced reflections. Turning on builds the scene acceleration
        // structure (one-shot, fence-waited) + the inline-`rayQueryEXT` pass;
        // turning off tears both down. The G-buffer pre-pass RT samples is
        // already built above (`gbuffer_needed` folds in `desired_rt`).
        // `rebuild_swapchain` below then rebuilds the RT output target; the
        // per-frame TLAS / geometry descriptors are wired by the next
        // `rt_dynamic_update`.
        match (rt_settings, self.rt_reflections.as_mut()) {
            // Already live: take the new trace resolution / shadow choice, which
            // `rebuild_swapchain` below sizes the output target from.
            (Some(settings), Some(rt)) => rt.settings = settings,
            (Some(settings), None) => self.build_rt_runtime(settings)?,
            (None, Some(_)) => {
                if let Some(mut rt) = self.rt_reflections.take() {
                    rt.destroy(&self.hw.device);
                }
                self.rt.destroy_accels();
                self.rt.skin = None;
            }
            (None, None) => {}
        }

        // The composite follows the ACTUAL post-build RT state, so a failed RT
        // enable falls back to the SSR resolve. `rebuild_swapchain` below then
        // routes the scene image through whichever path is left.
        self.reconcile_reflection_composite(q.reflection_blur_scale)?;

        // Rebuild every target + rewire every reader / the composite chain via
        // the resize path. It rebuilds the transient pool + bloom from the
        // reconciled `self.ssao`, rebuilds each `Some` feature's targets, and
        // re-points the composite scene input down the upscale > TAA >
        // reflection-composite > HDR priority chain.
        self.rebuild_swapchain()
    }

    // Build the RT reflection pass + acceleration structure at runtime (a live
    // toggle-on). Mirrors the init RT block: a shader-compile failure leaves
    // `rt_reflections` `None` and the renderer falls back to the SSR resolve when
    // authored (a soft failure, returns `Ok`), while an empty scene or an
    // AS-build error leaves only `rt.accel` `None` until a topology change seeds
    // it. The pass samples the unified G-buffer pre-pass, so without one the
    // enable is skipped the same way. The caller has drained the device
    // (`wait_idle`); `rebuild_swapchain` refreshes the output target after.
    fn build_rt_runtime(
        &mut self,
        settings: rt_reflections::RtReflectionSettings,
    ) -> RenderResult<()> {
        let hdr_views: Vec<vk::ImageView> = self
            .targets
            .hdr_resolve_images
            .iter()
            .map(|i| i.view)
            .collect();
        let Some(gb) = self.gbuffer.as_ref() else {
            tracing::warn!(
                "RT reflections need the unified G-buffer pre-pass, which is missing \
                 (keeping SSR)"
            );
            return Ok(());
        };
        let nd_views = gb.normal_depth_views();
        let rough_views = gb.roughness_views();
        // The textured hit variant indexes the bindless pool, so it compiles
        // against the length the pool set layout was built with; 0 when there
        // is no bindless layout, in which case the variant is not built.
        let bindless_pool_size = self.cull.bindless_pool_size;
        let rt = match super::post::rt_reflections::RtReflectionsResources::new(
            super::post::rt_reflections::RtBuild {
                alloc: &self.hw.alloc,
                device: &self.hw.device,
                width: self.targets.render_extent.width,
                height: self.targets.render_extent.height,
                frames: self.frames_in_flight,
            },
            settings,
            super::post::rt_reflections::RtStaticInputs {
                vertex_buffer: self.geometry.vertex_buffer.buffer(),
                index_buffer: self.geometry.index_buffer.buffer(),
                hdr_resolve_views: &hdr_views,
                gbuffer_views: &nd_views,
                roughness_views: &rough_views,
            },
            super::post::rt_reflections::RtLayoutConfig {
                bindless_set_layout: self.cull.bindless_set_layout.as_ref().map(|l| l.handle()),
                global_set_layout: self.descriptors.global_set_layout.handle(),
                pool_size: bindless_pool_size,
                hot_reload: self.hot_reload.enabled,
            },
        ) {
            Ok(rt) => rt,
            Err(e) => {
                tracing::warn!("RT reflections pass build failed (keeping SSR): {e}");
                return Ok(());
            }
        };
        self.rt_reflections = Some(rt);
        self.rt.skin = crate::vulkan::raytrace::build_rt_skin(
            &self.hw.alloc,
            &self.hw.device,
            self.hot_reload.enabled,
        );
        self.rt.accel = self.build_scene_accel_or_warn();
        self.forget_wired_accel();
        Ok(())
    }
}

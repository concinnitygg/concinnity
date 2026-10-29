//! Runtime application of the Quality-group settings (TAA / SSAO / SSR / SSGI /
//! RT reflections / auto-exposure). Each gates a render pass whose GPU
//! resources (pipelines, render targets, descriptor sets) are built once at
//! init from the world's settings assets, so applying a change at runtime means
//! building or tearing down those resources, not flipping a uniform.
//!
//! A change rebuilds only the features its plan names, with the same
//! `*Resources::new` the init path runs, and never waits on the device or
//! rebuilds the swapchain. Everything a replaced feature owns is an owned
//! handle or a pooled lease, which the device holds for `frames_in_flight + 1`
//! frames after it drops, so a frame recorded before the change keeps what it
//! names. The sets and framebuffers written ahead of a frame that name a
//! feature's output (the composite's scene input, the screen-space indirect
//! light, the transparent and particle passes' scene, the glass reflection
//! layers) are re-pointed per frame slot, past that slot's fence, by
//! `feature_wires.rs`. The transient pool is planned for every reachable
//! feature, so no toggle repacks it.
//!
//! Turning RT on still builds the scene acceleration structure with a CPU wait
//! (`build_rt_accel`, a one-shot fence-waited BLAS + TLAS over current
//! geometry), so a live enable hitches once proportional to triangle count. RT
//! is only live-toggleable when the device is RT-capable -- the ray-query device
//! extensions are enabled at creation whenever capable (see
//! `create_logical_device`), since an extension cannot be added later. The
//! resolved settings never turn RT on for an RT-incapable GPU or beside XeSS.

use ash::vk;
use concinnity_engine::gfx::{auto_exposure, render_types};
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::backend::feature_plan::{BackendFacts, FeatureChange, FeaturePlan};
use concinnity_engine::render::backend::init;
use concinnity_engine::render::post::rt_reflections;
use concinnity_engine::render::render_graph::{BackendTraits, BuildStatus, FeatureGates};
use concinnity_engine::settings::ResolvedGraphics;
use concinnity_engine::settings::apply::{
    ApplyPlan, RebuildFeature, RebuildSet, SettingsTarget, apply_plan, rebuild_each,
};

use super::context::VkContext;
use super::post::SsaoResources;

impl VkContext {
    // Rebuild `features` to match `g`, between frames (the GraphicsSystem
    // drain runs before the next `draw_frame`). A feature whose build fails
    // keeps its previous state; the first error comes back once every feature
    // has run.
    fn rebuild_planned(&mut self, g: &ResolvedGraphics, features: RebuildSet) -> RenderResult<()> {
        // TAA resources stay present while the built upscaler consumes the
        // velocity pre-pass; the TAA resolve is then dropped from the graph.
        let plan = FeaturePlan::resolve(
            g,
            BackendFacts {
                rt_capable: self.hw.rt_capable,
                upscaling: self.upscale.is_some(),
            },
        );
        let rebuilt = rebuild_each(features, |feature| self.rebuild_feature(feature, g, &plan));
        // SSR, SSGI and RT share the SSR resources.
        let shares_ssr = [
            RebuildFeature::Ssr,
            RebuildFeature::Ssgi,
            RebuildFeature::RtReflections,
        ]
        .into_iter()
        .any(|f| features.contains(f));
        let ssr = if shares_ssr {
            self.rebuild_ssr(g, &plan)
        } else {
            Ok(())
        };
        self.refresh_gates(g);
        rebuilt.and(ssr)
    }

    fn rebuild_feature(
        &mut self,
        feature: RebuildFeature,
        g: &ResolvedGraphics,
        plan: &FeaturePlan,
    ) -> RenderResult<()> {
        match feature {
            RebuildFeature::GBuffer => self.rebuild_gbuffer(plan.gbuffer),
            RebuildFeature::Taa => self.rebuild_taa(plan.temporal),
            RebuildFeature::Ssao => self.rebuild_ssao(g),
            // After the loop, with SSGI and RT, which share its resources.
            RebuildFeature::Ssr => Ok(()),
            RebuildFeature::Ssgi => self.rebuild_ssgi(g),
            RebuildFeature::RtReflections => self.rebuild_rt(plan.rt),
            // Follows the post-build RT state, so a failed RT enable falls back
            // to the SSR resolve.
            RebuildFeature::ReflectionComposite => {
                self.reconcile_reflection_composite(g.reflection_blur_scale())
            }
            RebuildFeature::AutoExposure => self.rebuild_auto_exposure(g),
        }
    }

    // The unified G-buffer pre-pass. Kept alive once built (a later toggle-off
    // of the last consumer leaves it resident until the next launch), which is
    // harmless: with no consumer the graph omits its readers. Its three color
    // channels are pooled, and the pool places them whatever the settings.
    fn rebuild_gbuffer(&mut self, wanted: bool) -> RenderResult<()> {
        if wanted && self.gbuffer.is_none() {
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
        // Also retries what an earlier change failed to bring up for its pre-pass.
        if self.gbuffer.is_some() {
            self.enable_gbuffer_prepass()?;
        }
        Ok(())
    }

    // The TAA resources, present while TAA or the upscaler consumes velocity.
    fn rebuild_taa(&mut self, wanted: bool) -> RenderResult<()> {
        match FeatureChange::of(wanted, self.taa.is_some()) {
            FeatureChange::Build => {
                let taa = super::post::taa::TaaResources::new(
                    &self.post_device(0),
                    self.frames_in_flight,
                    self.targets.render_extent,
                )?;
                self.taa = Some(taa);
            }
            FeatureChange::Drop => {
                // The cached framebuffers name the accumulation images' views.
                self.post.cache.forget_views();
                self.taa = None;
            }
            FeatureChange::Update | FeatureChange::Absent => {}
        }
        Ok(())
    }

    // The SSR resolve + reflection target. Built whenever SSR / SSGI / RT is
    // on; its settings carry whether SSR itself is authored, so a build kept
    // alive for SSGI or RT takes an SSR toggle through them alone.
    fn rebuild_ssr(&mut self, g: &ResolvedGraphics, plan: &FeaturePlan) -> RenderResult<()> {
        match FeatureChange::of(plan.ssr_resources, self.ssr.is_some()) {
            FeatureChange::Build => {
                let ssr = super::post::ssr::SsrResources::new(
                    &self.post_device(0),
                    g.ssr(),
                    self.targets.render_extent,
                )?;
                self.ssr = Some(ssr);
            }
            FeatureChange::Drop => {
                // The cached framebuffers name the reflection target's view.
                self.post.cache.forget_views();
                self.ssr = None;
            }
            FeatureChange::Update | FeatureChange::Absent => {
                if let Some(ssr) = self.ssr.as_mut() {
                    ssr.settings = g.ssr();
                }
            }
        }
        Ok(())
    }

    // SSGI (samples the unified G-buffer's per-frame normal+depth views). A
    // new trace resolution resizes every target, so it builds a fresh pass; the
    // ray count rides the settings.
    fn rebuild_ssgi(&mut self, g: &ResolvedGraphics) -> RenderResult<()> {
        let rescaled = match (g.ssgi(), self.ssgi.as_ref()) {
            (Some(settings), Some(live)) => settings.gi_scale != live.settings.gi_scale,
            _ => false,
        };
        match (g.ssgi(), self.ssgi.is_some()) {
            (Some(settings), built) if !built || rescaled => {
                let ssgi = super::post::ssgi::SsgiResources::new(
                    &self.post_device(0),
                    settings,
                    self.targets.render_extent,
                )?;
                self.post.cache.forget_views();
                self.ssgi = Some(ssgi);
            }
            (Some(settings), _) => {
                if let Some(ssgi) = self.ssgi.as_mut() {
                    ssgi.settings = settings;
                }
            }
            (None, true) => {
                self.post.cache.forget_views();
                self.ssgi = None;
            }
            (None, false) => {}
        }
        Ok(())
    }

    // SSAO. Its occlusion target is the pool's per-frame `ao_output`, which the
    // pool places whatever the settings; the main pass and the composite read
    // it only while SSAO is built.
    fn rebuild_ssao(&mut self, g: &ResolvedGraphics) -> RenderResult<()> {
        match (g.ssao(), self.ssao.is_some()) {
            (Some(settings), false) => {
                let ssao =
                    SsaoResources::new(&self.post_device(0), settings, self.targets.render_extent)?;
                self.ssao = Some(ssao);
            }
            (None, true) => {
                // The cached framebuffers name the raw occlusion's view.
                self.post.cache.forget_views();
                self.ssao = None;
            }
            _ => {}
        }
        Ok(())
    }

    // Ray-traced reflections. Turning on builds the inline-`rayQueryEXT` pass
    // and the scene acceleration structure; turning off drops the pass and
    // retires the BVH. A new trace resolution rebuilds the output target. The
    // glass reflection layers follow the trace divisor.
    fn rebuild_rt(
        &mut self,
        settings: Option<rt_reflections::RtReflectionSettings>,
    ) -> RenderResult<()> {
        let size = (
            self.targets.render_extent.width,
            self.targets.render_extent.height,
        );
        match (settings, self.rt_reflections.as_mut()) {
            (Some(settings), Some(rt)) => {
                rt.retrace(&self.hw.alloc, &self.hw.device, settings, size)?;
            }
            (Some(settings), None) => self.build_rt_runtime(settings),
            (None, Some(_)) => {
                if let Some(mut rt) = self.rt_reflections.take() {
                    rt.destroy(&self.hw.device);
                }
                self.rt.retire_accel();
                self.rt.skin = None;
            }
            (None, None) => {}
        }
        let divisor = self
            .rt_reflections
            .as_ref()
            .map_or(1, |rt| rt.settings.divisor);
        if let Some(t) = self.transparent.as_mut() {
            t.set_reflection_divisor(&self.hw.alloc, &self.hw.device, size, divisor)?;
        }
        Ok(())
    }

    // Auto-exposure. When it turns off the static authored EV drives exposure
    // again (the plan pushes the composite after this), so only the GPU state
    // is swapped here.
    fn rebuild_auto_exposure(&mut self, g: &ResolvedGraphics) -> RenderResult<()> {
        if let Some(settings) = g.auto_exposure().as_ref()
            && self.auto_exposure.resources.is_none()
        {
            let hdr_views: Vec<vk::ImageView> = self
                .targets
                .hdr_resolve_images
                .iter()
                .map(|i| i.view)
                .collect();
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
                g.auto_exposure_bias_ev(),
            ));
        } else if g.auto_exposure().is_none()
            && let Some(mut ae) = self.auto_exposure.resources.take()
        {
            ae.destroy(&self.hw.device);
            self.auto_exposure.adaptation = None;
        }
        Ok(())
    }

    // Gate the runtime-switched passes on `graphics` and what is built now.
    // The post chain reads the reflection composite's output as its scene
    // through sets written ahead of the frame.
    pub(in crate::vulkan) fn refresh_gates(&mut self, graphics: &ResolvedGraphics) {
        let traits = BackendTraits {
            reactive_mask: self.hw.device.writer_targets().carries_mask(),
            composite_scene_fixed: true,
        };
        self.state.planner.gates = FeatureGates::new(graphics, &self.build_status(), traits);
    }

    fn build_status(&self) -> BuildStatus {
        BuildStatus {
            taa: self.taa.is_some(),
            ssao: self.ssao.is_some(),
            ssr_resolve: self.ssr_authored(),
            rt_pass: self.rt_reflections.is_some(),
            ssgi: self.ssgi.is_some(),
            auto_exposure: self.auto_exposure.resources.is_some(),
            gbuffer: self.gbuffer_targets().is_some(),
        }
    }

    // Build the RT reflection pass + acceleration structure at runtime (a live
    // toggle-on). Mirrors the init RT block: a shader-compile failure leaves
    // `rt_reflections` `None` and the renderer falls back to the SSR resolve when
    // authored (a soft failure, returns `Ok`), while an empty scene or an
    // AS-build error leaves only `rt.accel` `None` until a topology change seeds
    // it. The pass samples the unified G-buffer pre-pass, so without one the
    // enable is skipped the same way. The G-buffer rebuild, which runs first,
    // has built the pre-pass it samples.
    fn build_rt_runtime(&mut self, settings: rt_reflections::RtReflectionSettings) {
        let hdr_views: Vec<vk::ImageView> = self
            .targets
            .hdr_resolve_images
            .iter()
            .map(|i| i.view)
            .collect();
        let Some(gb) = self.gbuffer_targets() else {
            tracing::warn!(
                "RT reflections need the unified G-buffer pre-pass, which is missing \
                 (keeping SSR)"
            );
            return;
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
                bindless_set_layout: self
                    .cull
                    .bindless_set_layout
                    .as_ref()
                    .map(super::owned::OwnedSetLayout::handle),
                global_set_layout: self.descriptors.global_set_layout.handle(),
                pool_size: bindless_pool_size,
                hot_reload: self.hot_reload.enabled,
            },
        ) {
            Ok(rt) => rt,
            Err(e) => {
                tracing::warn!("RT reflections pass build failed (keeping SSR): {e}");
                return;
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
    }
}

impl VkContext {
    // Apply a settings change in the plan order every backend shares.
    pub(crate) fn apply_settings(
        &mut self,
        graphics: &ResolvedGraphics,
        plan: &ApplyPlan,
    ) -> RenderResult<()> {
        apply_plan(self, graphics, plan)
    }
}

impl SettingsTarget for VkContext {
    fn rebuild_features(
        &mut self,
        graphics: &ResolvedGraphics,
        features: RebuildSet,
    ) -> RenderResult<()> {
        self.rebuild_planned(graphics, features)
    }

    fn push_feature_params(&mut self, graphics: &ResolvedGraphics) {
        self.update_quality_params(graphics);
    }

    fn push_post_process(&mut self, tunables: render_types::PostProcessTunables) {
        self.update_post_process(tunables);
    }

    fn push_ambient(&mut self, value: f32) {
        self.set_ambient_intensity(value);
    }

    fn push_shadow_cadence(&mut self, cadence: init::ShadowCadence) {
        self.set_shadow_cadence(cadence);
    }

    fn push_vsync(&mut self, on: bool) {
        self.set_vsync(on);
    }
}

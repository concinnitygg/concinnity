//! Runtime application of the Quality-group settings (TAA / SSAO / SSR / SSGI /
//! RT reflections / auto-exposure). Each gates a render pass whose GPU
//! resources (pipelines, render targets) are built once at init from the
//! world's settings assets, so applying a change at runtime means building or
//! tearing down those resources.
//!
//! A change rebuilds only the features its plan names, and never waits on the
//! device. DirectX picks its scene-input and occlusion sources every frame
//! (`scene_srv_for_post`, `ssao_ao_srv_gpu`, and each feature's render-graph
//! gate read the live `Option` state at encode time), so a frame recorded after
//! the apply names only the new resources. A frame recorded before it keeps
//! everything it names alive: a replaced feature's COM objects wait in
//! `retired_effects` for `frames_in_flight + 1` frames, and every
//! shader-visible descriptor a rebuilt feature writes is a fresh slot of the
//! post block, whose freed slots wait out the same window. The transient pool
//! is planned for every reachable feature, so no toggle repacks it.
//!
//! Turning RT on still builds the scene acceleration structure with a CPU wait
//! (`build_rt_accel`, a one-shot fence-waited BLAS + TLAS over current
//! geometry), so a live enable hitches once proportional to triangle count.

use concinnity_engine::gfx::{auto_exposure, render_types};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::feature_plan::{BackendFacts, FeatureChange, FeaturePlan};
use concinnity_engine::render::backend::init;
use concinnity_engine::render::post::device::PostExtent;
use concinnity_engine::render::post::reflection_path::ReflectionPath;
use concinnity_engine::render::post::rt_reflections;
use concinnity_engine::render::render_graph::{BackendTraits, BuildStatus, FeatureGates};
use concinnity_engine::settings::ResolvedGraphics;
use concinnity_engine::settings::apply::{
    ApplyPlan, RebuildFeature, RebuildSet, SettingsTarget, apply_plan, rebuild_each,
};

use super::auto_exposure::AutoExposureResources;
use super::context::DxContext;
use super::post::gbuffer::GbufferSlots;
use super::post::post_device::PostTarget;
use super::post::reflection_composite::{DxReflectionCompositePass, build_reflection_composite};
use super::post::rt_reflections::{RetiredRtOutput, RtReflectionsResources};
use super::post::ssao::SsaoResources;
use super::post::ssgi::SsgiResources;
use super::post::ssr::SsrResources;
use super::post::taa::TaaResources;
use super::raytrace::SkinPipeline;
use super::transparent::GlassReflectionLayers;

// The fixed descriptor-heap slots of the live-toggleable features outside the
// post block. Minted once at init (the slots are reserved unconditionally,
// independent of the init-time quality gates) and stashed on `DxContext` so a
// live toggle can construct a launched-off feature into its slot without
// re-deriving the heap layout. The G-buffer is kept once built and its colors
// live in a pool no toggle repacks, so its slots are written once per resize.
#[derive(Clone, Copy)]
pub(in crate::directx) struct QualitySlotHandles {
    pub gbuffer: GbufferSlots,
    pub glass_reflection: super::transparent::GlassReflectionSlots,
}

// A feature's state a settings change replaced, held until no frame in flight
// can name it, then dropped.
#[expect(
    dead_code,
    reason = "held so the COM objects stay valid until the retire pool drops them"
)]
pub(in crate::directx) enum RetiredEffect {
    Taa(TaaResources),
    Ssr(SsrResources),
    Ssgi(SsgiResources),
    Ssao(SsaoResources),
    Composite(DxReflectionCompositePass),
    Blur(PostTarget),
    AutoExposure(AutoExposureResources),
    RtPass(RtReflectionsResources),
    RtOutput(RetiredRtOutput),
    RtSkin(SkinPipeline),
    GlassLayers(GlassReflectionLayers),
}

impl DxContext {
    // Rebuild `features` to match `g`, between frames (the GraphicsSystem
    // drain runs before the next `draw_frame`). A feature whose build fails
    // keeps its previous state; the first error comes back once every feature
    // has run.
    fn rebuild_planned(&mut self, g: &ResolvedGraphics, features: RebuildSet) -> RenderResult<()> {
        // RT is gated on the GPU reporting the DXR 1.1 tier. TAA resources stay
        // present while the built upscaler consumes the velocity pre-pass; the
        // TAA resolve is then dropped from the graph.
        let plan = FeaturePlan::resolve(
            g,
            BackendFacts {
                rt_capable: self.hw.rt_capable,
                upscaling: self.upscale.backend.is_some(),
            },
        );
        let rebuilt = rebuild_each(features, |feature| self.rebuild_feature(feature, g, &plan));
        // SSR, SSGI and RT share the SSR resources (the resolve half is SSR's).
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
            RebuildFeature::ReflectionComposite => self.rebuild_reflection_composite(g, plan),
            RebuildFeature::AutoExposure => self.rebuild_auto_exposure(g, plan.auto_exposure),
        }
    }

    // Park `effect` until the frames recorded so far have retired.
    fn retire_effect(&mut self, effect: RetiredEffect) {
        self.retired_effects.push(effect);
    }

    fn render_size(&self) -> (u32, u32) {
        (
            self.targets.extent.render_width,
            self.targets.extent.render_height,
        )
    }

    // The unified G-buffer pre-pass. Kept alive once built (a later toggle-off
    // of the last consumer leaves it resident until the next launch), which is
    // harmless: with no consumer the graph omits its readers. Its three color
    // targets are pooled, and the pool places them whatever the settings.
    fn rebuild_gbuffer(&mut self, wanted: bool) -> RenderResult<()> {
        if wanted && self.gbuffer.is_none() {
            let pooled = self
                .targets
                .transient_pool
                .gbuffer_pooled()
                .ok_or_else(|| {
                    RenderError::Other("transient pool missing the gbuffer color targets".into())
                })?;
            let (width, height) = self.render_size();
            let gbuffer = super::post::gbuffer::GbufferResources::new(
                super::post::gbuffer::GbufferDeviceCtx {
                    alloc: &self.hw.alloc,
                    info_queue: self.hw.info_queue.as_ref(),
                    hot_reload: self.hot_reload.enabled,
                },
                super::post::gbuffer::GbufferExtent { width, height },
                self.quality_slots.gbuffer,
                &pooled,
            )?;
            self.gbuffer = Some(gbuffer);
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
                let (width, height) = self.render_size();
                self.taa = Some(TaaResources::new(&self.post_device(0), width, height)?);
            }
            FeatureChange::Drop => {
                if let Some(old) = self.taa.take() {
                    self.retire_effect(RetiredEffect::Taa(old));
                }
            }
            FeatureChange::Update | FeatureChange::Absent => {}
        }
        Ok(())
    }

    // The SSR resources: built whenever SSR, SSGI or RT runs, with the resolve
    // half only while SSR itself does. A change in either builds a fresh set.
    fn rebuild_ssr(&mut self, g: &ResolvedGraphics, plan: &FeaturePlan) -> RenderResult<()> {
        let has_resolve = self.ssr.as_ref().map(|s| s.resolve.is_some());
        let wanted = plan.ssr_resources.then_some(plan.ssr);
        if wanted == has_resolve {
            return Ok(());
        }
        let fresh = match wanted {
            Some(resolve) => {
                let (width, height) = self.render_size();
                Some(SsrResources::new(
                    &self.post_device(0),
                    width,
                    height,
                    g.ssr().filter(|_| resolve),
                )?)
            }
            None => None,
        };
        if let Some(old) = std::mem::replace(&mut self.ssr, fresh) {
            self.retire_effect(RetiredEffect::Ssr(old));
        }
        Ok(())
    }

    // SSGI. A new trace resolution resizes every target, so it builds a fresh
    // pass; the ray count rides the settings.
    fn rebuild_ssgi(&mut self, g: &ResolvedGraphics) -> RenderResult<()> {
        let wanted = g.ssgi();
        let rescaled = match (wanted, self.ssgi.as_ref()) {
            (Some(settings), Some(live)) => settings.gi_scale != live.settings.gi_scale,
            _ => false,
        };
        match (wanted, self.ssgi.is_some()) {
            (Some(settings), built) if !built || rescaled => {
                let (width, height) = self.render_size();
                let fresh = SsgiResources::new(&self.post_device(0), width, height, settings)?;
                if let Some(old) = self.ssgi.replace(fresh) {
                    self.retire_effect(RetiredEffect::Ssgi(old));
                }
            }
            (Some(settings), _) => {
                if let Some(ssgi) = self.ssgi.as_mut() {
                    ssgi.settings = settings;
                }
            }
            (None, _) => {
                if let Some(old) = self.ssgi.take() {
                    self.retire_effect(RetiredEffect::Ssgi(old));
                }
            }
        }
        Ok(())
    }

    // SSAO over the pool's `ao_output`, which the pool places whatever the
    // settings. The main pass's occlusion binding (`ssao_ao_srv_gpu`) falls
    // back to the 1x1 white slot when it is off.
    fn rebuild_ssao(&mut self, g: &ResolvedGraphics) -> RenderResult<()> {
        match (g.ssao(), self.ssao.resources.is_some()) {
            (Some(settings), false) => {
                let ao_output = self
                    .targets
                    .transient_pool
                    .resource_for("ao_output")
                    .ok_or_else(|| RenderError::Other("transient pool missing ao_output".into()))?
                    .clone();
                let (width, height) = self.render_size();
                let ssao = SsaoResources::new(
                    &self.post_device(0),
                    settings,
                    PostExtent { width, height },
                    &ao_output,
                )?;
                self.ssao.resources = Some(ssao);
            }
            (None, true) => {
                if let Some(old) = self.ssao.resources.take() {
                    self.retire_effect(RetiredEffect::Ssao(old));
                }
            }
            _ => {}
        }
        Ok(())
    }

    // Ray-traced reflections. Turning on builds the DXR pass and the scene
    // acceleration structure; turning off retires both, and
    // `scene_srv_for_post` falls back to the SSR resolve / HDR next frame. A
    // new trace resolution moves the output to fresh descriptors. The glass
    // reflection pre-pass follows the trace divisor.
    fn rebuild_rt(
        &mut self,
        settings: Option<rt_reflections::RtReflectionSettings>,
    ) -> RenderResult<()> {
        let size = self.render_size();
        match (settings, self.rt_reflections.as_mut()) {
            (Some(settings), Some(rt)) => {
                if let Some(old) = rt.retrace(&self.hw.device, &self.post, settings, size)? {
                    self.retire_effect(RetiredEffect::RtOutput(old));
                }
            }
            (Some(settings), None) => self.build_rt_runtime(settings),
            (None, Some(_)) => {
                if let Some(old) = self.rt_reflections.take() {
                    self.retire_effect(RetiredEffect::RtPass(old));
                }
                if let Some(old) = self.rt.skin.take() {
                    self.retire_effect(RetiredEffect::RtSkin(old));
                }
                if let Some(accel) = self.rt.accel.take() {
                    self.rt.retired.push(self.rt.retire_tick, accel);
                }
            }
            (None, None) => {}
        }
        let divisor = self
            .rt_reflections
            .as_ref()
            .map_or(1, |rt| rt.settings.divisor);
        if let Some(t) = self.transparent.as_mut()
            && let Some(old) =
                t.set_reflection_divisor(&self.hw.device, &self.post, size, divisor)?
        {
            self.retire_effect(RetiredEffect::GlassLayers(old));
        }
        Ok(())
    }

    // The roughness-aware reflection composite, present whenever a resolve
    // can composite (`ReflectionPath::composite`, the gate init builds it
    // under). It follows the post-build RT state, so a failed RT enable falls
    // back to the SSR resolve. A live composite takes a new blur resolution in
    // place; every consumer reads its targets' descriptors per frame.
    fn rebuild_reflection_composite(
        &mut self,
        g: &ResolvedGraphics,
        plan: &FeaturePlan,
    ) -> RenderResult<()> {
        let wanted = ReflectionPath::new(
            plan.ssr,
            self.rt_reflections.is_some(),
            self.rt.accel.is_some(),
        )
        .composite();
        let (width, height) = self.render_size();
        match FeatureChange::of(wanted, self.reflection_composite.is_some()) {
            FeatureChange::Build => {
                let rc = build_reflection_composite(
                    &self.post_device(0),
                    g.reflection_blur_scale(),
                    width,
                    height,
                )?;
                self.reflection_composite = Some(rc);
            }
            FeatureChange::Drop => {
                if let Some(old) = self.reflection_composite.take() {
                    self.retire_effect(RetiredEffect::Composite(old));
                }
            }
            FeatureChange::Update => {
                if let Some(mut rc) = self.reflection_composite.take() {
                    let replaced = rc.set_blur_scale(
                        &self.post_device(0),
                        g.reflection_blur_scale(),
                        PostExtent { width, height },
                    );
                    self.reflection_composite = Some(rc);
                    if let Some(old) = replaced? {
                        self.retire_effect(RetiredEffect::Blur(old));
                    }
                }
            }
            FeatureChange::Absent => {}
        }
        Ok(())
    }

    // Auto-exposure: its own root UAVs and readback ring, no descriptor-heap
    // slots. When it turns off the static authored EV drives exposure again
    // (the plan pushes the composite after this), so only the GPU and
    // adaptation state is swapped here.
    fn rebuild_auto_exposure(&mut self, g: &ResolvedGraphics, wanted: bool) -> RenderResult<()> {
        match FeatureChange::of(wanted, self.auto_exposure.resources.is_some()) {
            FeatureChange::Build => {
                let resources =
                    AutoExposureResources::new(&self.hw.alloc, self.hot_reload.enabled)?;
                self.auto_exposure.resources = Some(resources);
                self.auto_exposure.adaptation = g.auto_exposure().map(|settings| {
                    auto_exposure::ExposureAdaptation::new(settings, g.auto_exposure_bias_ev())
                });
            }
            FeatureChange::Drop => {
                if let Some(old) = self.auto_exposure.resources.take() {
                    self.retire_effect(RetiredEffect::AutoExposure(old));
                }
                self.auto_exposure.adaptation = None;
            }
            FeatureChange::Update | FeatureChange::Absent => {}
        }
        Ok(())
    }

    // Gate the runtime-switched passes on `graphics` and what is built now.
    // Every scene target set carries the reactive mask, and the post chain
    // picks its scene input each frame.
    pub(super) fn refresh_gates(&mut self, graphics: &ResolvedGraphics) {
        let traits = BackendTraits {
            reactive_mask: true,
            composite_scene_fixed: false,
        };
        self.state.planner.gates = FeatureGates::new(graphics, &self.build_status(), traits);
    }

    fn build_status(&self) -> BuildStatus {
        BuildStatus {
            taa: self.taa.is_some(),
            ssao: self.ssao.resources.is_some(),
            ssr_resolve: self.ssr.as_ref().is_some_and(|s| s.resolve.is_some()),
            rt_pass: self.rt_reflections.is_some(),
            ssgi: self.ssgi.is_some(),
            auto_exposure: self.auto_exposure.resources.is_some(),
            gbuffer: self.gbuffer.is_some(),
        }
    }

    // Build the RT reflection pass + acceleration structure at runtime (a live
    // toggle-on). Mirrors the init RT block: a shader-compile failure leaves
    // `rt_reflections` `None` and the renderer stays on SSR (a soft failure,
    // returns `Ok`), while an empty scene or an AS-build error leaves only
    // `rt.accel` `None` until `rt_dynamic_update` seeds it. The G-buffer
    // rebuild, which runs first, has built the pre-pass it samples. The skin
    // pipeline is built before the BVH, since whether skinned geometry can join
    // decides whether a scene with no static geometry gets one (a build failure
    // is non-fatal: static geometry still reflects, just without skinned hits).
    fn build_rt_runtime(&mut self, settings: rt_reflections::RtReflectionSettings) {
        let hot_reload = self.hot_reload.enabled;
        let rt = match RtReflectionsResources::new(
            super::post::rt_reflections::RtBuildContext {
                alloc: &self.hw.alloc,
                width: self.targets.extent.render_width,
                height: self.targets.extent.render_height,
            },
            settings,
            &self.post,
            super::post::rt_reflections::RtBuildInit {
                info_queue: self.hw.info_queue.as_ref(),
                hot_reload,
            },
        ) {
            Ok(rt) => rt,
            Err(e) => {
                tracing::warn!("RT reflections pass build failed (keeping SSR): {e}");
                return;
            }
        };
        self.rt_reflections = Some(rt);
        self.rt.skin = super::raytrace::build_rt_skin(&self.hw.device, hot_reload);
        self.rebuild_rt_accel();
    }
}

impl DxContext {
    // Apply a settings change in the plan order every backend shares.
    pub(crate) fn apply_settings(
        &mut self,
        graphics: &ResolvedGraphics,
        plan: &ApplyPlan,
    ) -> RenderResult<()> {
        apply_plan(self, graphics, plan)
    }
}

impl SettingsTarget for DxContext {
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

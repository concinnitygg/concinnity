//! Runtime application of the Quality-group settings (TAA / SSAO / SSR / RT
//! reflections / SSGI / auto-exposure). Each gates a render pass whose GPU
//! resources (pipelines, render targets, the ray-tracing acceleration structure)
//! are built once at init from the world's settings assets, so applying a
//! change at runtime means rebuilding those resources, not flipping a uniform.
//!
//! A change rebuilds only the features its plan names, with the init stages'
//! builders, so a live toggle produces the resources a launch with the same
//! settings would. A replaced feature's state is parked in `retired_effects`
//! until no frame in flight can name it: a buffer or texture a pass reads only
//! through an argument buffer is not retained by the command buffer. The
//! transient pool is planned for every reachable feature, so no toggle repacks
//! it. Nothing here waits on the GPU, apart from the acceleration structure an
//! RT enable seeds.

use concinnity_engine::gfx::render_types;
use concinnity_engine::render::backend::WindowControl;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::backend::feature_plan::{BackendFacts, FeatureChange, FeaturePlan};
use concinnity_engine::render::backend::init;
use concinnity_engine::render::config::ResolvedGraphics;
use concinnity_engine::render::config::apply::{
    FeatureRebuild, RebuildFeature, RebuildSet, SettingsTarget, rebuild_planned,
};
use concinnity_engine::render::post::device::PostExtent;
use concinnity_engine::render::post::disocclusion::DisocclusionPass;
use concinnity_engine::render::post::ssao::SsaoPass;
use concinnity_engine::render::post::ssgi::SsgiPass;
use concinnity_engine::render::post::ssr::SsrPass;
use concinnity_engine::render::render_graph::{BackendTraits, BuildStatus};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLTexture;

use super::auto_exposure::AutoExposureGpu;
use super::context::MtlContext;
use super::init::effects::{EffectSettings, build_auto_exposure, build_gbuffer};
use super::init::ray_tracing::build_rt_pipelines;
use super::post::post_device::{MtlPostDevice, PostTarget};
use super::post::reflection_composite::MtlReflectionCompositePass;
use super::post::ssao::MtlSsaoPass;
use super::post::ssgi::MtlSsgiPass;
use super::post::ssr::MtlSsrPass;
use super::post::taa::MtlTaaPass;
use super::post::{
    GBufferState, MtlDisocclusionPass, build_reflection_composite, build_taa_pass,
    create_reflection_target, disocclusion_wanted,
};
use super::raytrace::{
    RtGpu, RtSceneGeometry, RtStaticGeometry, RtTextureCounts, build_rt_accel, raytracing_supported,
};
use concinnity_engine::render::post::reflection_denoise::ReflectionDenoiser;

// Every HDR target set carries the reactive mask, and the post chain picks its
// scene input each frame.
const METAL_TRAITS: BackendTraits = BackendTraits {
    reactive_mask: true,
    composite_scene_fixed: false,
};

// A feature's state a settings change replaced, held until every frame that
// could name it has retired on the GPU, then dropped.
#[expect(
    dead_code,
    reason = "held so the Metal objects stay valid until the retire pool drops them"
)]
pub(super) enum RetiredEffect {
    Taa(MtlTaaPass),
    Ssao(MtlSsaoPass),
    SsrResolve(MtlSsrPass),
    Reflection(Retained<ProtocolObject<dyn MTLTexture>>),
    Composite(MtlReflectionCompositePass),
    Denoiser(ReflectionDenoiser<PostTarget>),
    GBuffer(GBufferState),
    Ssgi(MtlSsgiPass),
    AutoExposure(AutoExposureGpu),
    Disocclusion(MtlDisocclusionPass),
}

// The per-feature settings a rebuild builds to, with ray tracing only as the
// plan allows it on this device.
fn effect_settings(graphics: &ResolvedGraphics, plan: &FeaturePlan) -> EffectSettings {
    let mut settings = EffectSettings::new(graphics);
    settings.rt_reflection = plan.rt;
    settings
}

impl MtlContext {
    fn build_status(&self) -> BuildStatus {
        BuildStatus {
            taa: self.taa.pass.is_some(),
            ssao: self.ssao.pass.is_some(),
            ssr_resolve: self.ssr.resolve.is_some(),
            rt_pass: self.rt.settings.is_some(),
            ssgi: self.ssgi.pass.is_some(),
            auto_exposure: self.auto_exposure.pipelines.is_some(),
            gbuffer: self.gbuffer.targets.is_some(),
        }
    }

    // Park `effect` until the frames recorded so far have retired. Between
    // frames, so it is tagged with the next frame's id.
    fn retire_effect(&mut self, effect: RetiredEffect) {
        self.retired_effects.push(self.frame_ring_index, effect);
    }

    // A device for building post passes: pipelines and targets only, nothing
    // encodes through it.
    fn build_device(&self) -> MtlPostDevice<'_> {
        MtlPostDevice {
            device: &self.hw.device,
            sampler: &self.composite.sampler,
            cube_sampler: &self.scene.cube_sampler,
            probes: None,
            timing: None,
            hot_reload: self.hot_reload.enabled,
        }
    }

    fn render_extent(&self) -> PostExtent {
        PostExtent {
            width: self.targets.hdr.width,
            height: self.targets.hdr.height,
        }
    }

    // The unified G-buffer pre-pass: its feature-owned depth and the
    // model-history kernel. Its color channels live in the transient pool,
    // which already places them. Every bucket's pre-pass pipeline follows.
    fn rebuild_gbuffer(&mut self, wanted: bool) -> RenderResult<()> {
        match FeatureChange::of(wanted, self.gbuffer.targets.is_some()) {
            FeatureChange::Build => {
                let render = (self.targets.hdr.width, self.targets.hdr.height);
                self.gbuffer =
                    build_gbuffer(&self.hw.device, true, render, self.hot_reload.enabled)?;
            }
            FeatureChange::Drop => {
                let old = std::mem::replace(
                    &mut self.gbuffer,
                    GBufferState {
                        targets: None,
                        history_pipeline: None,
                    },
                );
                self.retire_effect(RetiredEffect::GBuffer(old));
            }
            FeatureChange::Update | FeatureChange::Absent => return Ok(()),
        }
        self.sync_prepass_pipelines();
        Ok(())
    }

    // The TAA resolve. A rebuilt pass starts with its history invalid, so the
    // first frame after a toggle passes through.
    fn rebuild_taa(&mut self, wanted: bool) -> RenderResult<()> {
        match FeatureChange::of(wanted, self.taa.pass.is_some()) {
            FeatureChange::Build => {
                let extent = self.render_extent();
                self.taa.pass = Some(build_taa_pass(
                    &self.build_device(),
                    extent.width,
                    extent.height,
                )?);
            }
            FeatureChange::Drop => {
                if let Some(old) = self.taa.pass.take() {
                    self.retire_effect(RetiredEffect::Taa(old));
                }
            }
            FeatureChange::Update | FeatureChange::Absent => {}
        }
        self.taa.enabled = wanted;
        Ok(())
    }

    // The SSAO kernel and blur. Its blurred output is the pool's `ao_output`
    // and its white fallback stays, so only the pass moves.
    fn rebuild_ssao(&mut self, settings: &EffectSettings) -> RenderResult<()> {
        match FeatureChange::of(settings.ssao.is_some(), self.ssao.pass.is_some()) {
            FeatureChange::Build => {
                let pass = SsaoPass::new(&self.build_device(), self.render_extent())?;
                self.ssao.pass = Some(pass);
            }
            FeatureChange::Drop => {
                if let Some(old) = self.ssao.pass.take() {
                    self.retire_effect(RetiredEffect::Ssao(old));
                }
            }
            FeatureChange::Update | FeatureChange::Absent => {}
        }
        self.ssao.settings = settings.ssao;
        Ok(())
    }

    // The SSR resolve, built only while SSR itself is on.
    fn rebuild_ssr_resolve(&mut self, settings: &EffectSettings) -> RenderResult<()> {
        match FeatureChange::of(settings.ssr.is_some(), self.ssr.resolve.is_some()) {
            FeatureChange::Build => self.ssr.resolve = Some(SsrPass::new(&self.build_device())?),
            FeatureChange::Drop => {
                if let Some(old) = self.ssr.resolve.take() {
                    self.retire_effect(RetiredEffect::SsrResolve(old));
                }
            }
            FeatureChange::Update | FeatureChange::Absent => {}
        }
        self.ssr.settings = settings.ssr;
        Ok(())
    }

    // The composite blending the reflection target over the scene, present
    // while SSR or RT runs. A new blur resolution builds a fresh one.
    fn rebuild_reflection_composite(&mut self, settings: &EffectSettings) -> RenderResult<()> {
        let wanted = settings.ssr.is_some() || settings.rt_reflection.is_some();
        let rescaled = self
            .ssr
            .composite
            .as_ref()
            .is_some_and(|c| c.blur_scale_differs(settings.reflection_blur_scale));
        let change = FeatureChange::of(wanted, self.ssr.composite.is_some());
        // The live ray-traced pass, which the rebuild ahead of this one set.
        let denoise = self
            .rt
            .settings
            .filter(|_| self.rt.pipelines.reflections.is_some())
            .map(|rt| rt.divisor);
        let render = (self.targets.hdr.width, self.targets.hdr.height);
        if change == FeatureChange::Build || (change == FeatureChange::Update && rescaled) {
            let fresh = build_reflection_composite(
                &self.build_device(),
                settings.reflection_blur_scale,
                render,
                denoise,
            )?;
            if let Some(old) = self.ssr.composite.replace(fresh) {
                self.retire_effect(RetiredEffect::Composite(old));
            }
        } else if change == FeatureChange::Update
            && let Some(mut composite) = self.ssr.composite.take()
        {
            // The denoiser follows the ray-traced resolve and its trace scale.
            let extent = PostExtent {
                width: render.0,
                height: render.1,
            };
            let replaced = composite.set_denoise(&self.build_device(), denoise, extent);
            self.ssr.composite = Some(composite);
            if let Some(old) = replaced? {
                self.retire_effect(RetiredEffect::Denoiser(old));
            }
        } else if change == FeatureChange::Drop
            && let Some(old) = self.ssr.composite.take()
        {
            self.retire_effect(RetiredEffect::Composite(old));
        }
        Ok(())
    }

    // The SSGI pyramid, trace and composite. A new trace resolution resizes
    // every target, so it builds a fresh pass; the ray count rides the
    // settings.
    fn rebuild_ssgi(&mut self, settings: &EffectSettings) -> RenderResult<()> {
        let rescaled = match (settings.ssgi, self.ssgi.settings) {
            (Some(new), Some(live)) => new.gi_scale != live.gi_scale,
            _ => false,
        };
        let change = FeatureChange::of(settings.ssgi.is_some(), self.ssgi.pass.is_some());
        if let Some(s) = settings.ssgi
            && (change == FeatureChange::Build || rescaled)
        {
            let fresh = SsgiPass::new(&self.build_device(), s.gi_scale, self.render_extent())?;
            if let Some(old) = self.ssgi.pass.replace(fresh) {
                self.retire_effect(RetiredEffect::Ssgi(old));
            }
        } else if change == FeatureChange::Drop
            && let Some(old) = self.ssgi.pass.take()
        {
            self.retire_effect(RetiredEffect::Ssgi(old));
        }
        self.ssgi.settings = settings.ssgi;
        Ok(())
    }

    // Ray-traced reflections: the resolve pipelines stay once built, turning on
    // seeds the acceleration structure, and turning off retires it.
    fn rebuild_rt(&mut self, settings: &EffectSettings) -> RenderResult<()> {
        self.rt.settings = settings.rt_reflection;
        if self.rt.settings.is_none() {
            // Between frames, so it is tagged with the next frame's id.
            self.rt.retire_accel(self.frame_ring_index);
            return Ok(());
        }
        if self.rt.pipelines.reflections.is_none() {
            self.rt.pipelines = build_rt_pipelines(
                &self.hw.device,
                settings.rt_reflection.as_ref(),
                self.hot_reload.enabled,
            )?;
        }
        // Reset the failure streak so a later toggle-on starts clean.
        self.rt.update_streak = concinnity_engine::render::rt_accel::FailureStreak::default();
        if self.rt.accel.is_some() {
            return Ok(());
        }
        let built = build_rt_accel(
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
                draw_objects: &self.state.draw.objects,
                clusters: &self.instanced.clusters,
            },
            RtTextureCounts {
                albedo_count: self.scene.textures.len(),
            },
            None,
            self.seethrough_meshes_enabled(),
        )
        .map_err(|e| e.context("RT accel build"))?;
        if let Some(a) = built {
            tracing::info!(
                "ray-traced reflections: built BVH over {} static objects",
                a.blas().len()
            );
            self.rt.accel = Some(a);
        } else {
            tracing::warn!(
                "ray-traced reflections toggled on but the scene has no static geometry; no BVH built"
            );
        }
        Ok(())
    }

    // Auto-exposure. When it turns off the static path uses
    // `self.post_process.exposure` (the authored / slider EV), so only the GPU
    // state is swapped; the frame clock carries over.
    fn rebuild_auto_exposure(&mut self, settings: &EffectSettings) -> RenderResult<()> {
        let change = FeatureChange::of(
            settings.auto_exposure.is_some(),
            self.auto_exposure.pipelines.is_some(),
        );
        if !matches!(change, FeatureChange::Build | FeatureChange::Drop) {
            return Ok(());
        }
        let fresh = build_auto_exposure(
            &self.hw.device,
            settings,
            self.frames_in_flight,
            self.hot_reload.enabled,
        )?;
        let last_elapsed = self.auto_exposure.last_elapsed;
        let old = std::mem::replace(
            &mut self.auto_exposure,
            AutoExposureGpu {
                last_elapsed,
                ..fresh
            },
        );
        self.retire_effect(RetiredEffect::AutoExposure(old));
        Ok(())
    }
}

impl FeatureRebuild for MtlContext {
    // The resolved settings never ask for ray tracing the device cannot run; a
    // request that does is a probe / backend disagreement, reported and refused
    // rather than built.
    fn backend_facts(&self) -> BackendFacts {
        BackendFacts {
            rt_capable: raytracing_supported(&self.hw.device),
            upscaling: self.upscale.scaler.is_some(),
        }
    }

    fn rebuild_feature(
        &mut self,
        feature: RebuildFeature,
        graphics: &ResolvedGraphics,
        plan: &FeaturePlan,
    ) -> RenderResult<()> {
        let settings = effect_settings(graphics, plan);
        match feature {
            RebuildFeature::GBuffer => self.rebuild_gbuffer(plan.gbuffer),
            RebuildFeature::Taa => self.rebuild_taa(plan.taa_resolve),
            RebuildFeature::Ssao => self.rebuild_ssao(&settings),
            RebuildFeature::Ssr => self.rebuild_ssr_resolve(&settings),
            RebuildFeature::Ssgi => self.rebuild_ssgi(&settings),
            RebuildFeature::RtReflections => self.rebuild_rt(&settings),
            RebuildFeature::ReflectionComposite => self.rebuild_reflection_composite(&settings),
            RebuildFeature::AutoExposure => self.rebuild_auto_exposure(&settings),
        }
    }

    // The reflection target SSR, SSGI and RT write, present while any of them
    // runs and reduced to the RT trace resolution when RT fills it.
    fn reconcile_reflections(
        &mut self,
        graphics: &ResolvedGraphics,
        plan: &FeaturePlan,
    ) -> RenderResult<()> {
        let settings = effect_settings(graphics, plan);
        let wanted = settings.reflections();
        let trace_scale = settings.rt_reflection.map_or(1, |rt| rt.divisor.max(1));
        let rescaled = trace_scale != self.ssr.trace_scale;
        let change = FeatureChange::of(wanted, self.ssr.reflection.is_some());
        if change == FeatureChange::Build || (change == FeatureChange::Update && rescaled) {
            let fresh = create_reflection_target(
                &self.hw.device,
                self.targets.hdr.width,
                self.targets.hdr.height,
                trace_scale,
            )?;
            if let Some(old) = self.ssr.reflection.replace(fresh) {
                self.retire_effect(RetiredEffect::Reflection(old));
            }
        } else if change == FeatureChange::Drop
            && let Some(old) = self.ssr.reflection.take()
        {
            self.retire_effect(RetiredEffect::Reflection(old));
        }
        self.ssr.trace_scale = trace_scale;
        Ok(())
    }

    // The disocclusion mark follows the G-buffer the features above left
    // built.
    fn reconcile_disocclusion(&mut self) -> RenderResult<()> {
        let wanted = disocclusion_wanted(&self.upscale, self.gbuffer.targets.is_some());
        match FeatureChange::of(wanted, self.disocclusion.is_some()) {
            FeatureChange::Build => {
                self.disocclusion = Some(DisocclusionPass::new(
                    &self.build_device(),
                    self.render_extent(),
                )?);
            }
            FeatureChange::Drop => {
                if let Some(old) = self.disocclusion.take() {
                    self.retire_effect(RetiredEffect::Disocclusion(old));
                }
            }
            FeatureChange::Update | FeatureChange::Absent => {}
        }
        Ok(())
    }

    // Gate the runtime-switched passes on `graphics` and what is built now.
    fn refresh_gates(&mut self, graphics: &ResolvedGraphics) {
        self.state
            .planner
            .configure(graphics, &self.build_status(), METAL_TRAITS);
    }
}

impl SettingsTarget for MtlContext {
    fn built_shadow_maps(&self) -> init::ShadowMapSizes {
        self.shadow.map_sizes()
    }

    // Rebuild `features` to match `graphics`, between frames (the GraphicsSystem
    // drain runs before the next `draw_frame`). A feature whose build fails
    // keeps its previous state; the first error comes back once every feature
    // has run.
    fn rebuild_features(
        &mut self,
        graphics: &ResolvedGraphics,
        features: RebuildSet,
    ) -> RenderResult<()> {
        rebuild_planned(self, graphics, features)
    }

    // Update the live scalar tunables of the SSAO, SSR, SSGI and auto-exposure
    // passes, read from the stored settings each frame. Only features that are
    // on have settings to update; SSGI's resolution and ray count ride its
    // rebuild.
    fn push_feature_params(&mut self, graphics: &ResolvedGraphics) {
        if let (Some(live), Some(cur)) = (graphics.ssao(), self.ssao.settings.as_mut()) {
            *cur = live;
        }
        if let (Some(live), Some(cur)) = (graphics.ssr(), self.ssr.settings.as_mut()) {
            *cur = live;
        }
        if let (Some(live), Some(cur)) = (graphics.ssgi(), self.ssgi.settings.as_mut()) {
            cur.intensity = live.intensity;
            cur.max_distance = live.max_distance;
        }
        if let (Some(live), Some(cur)) = (
            graphics.auto_exposure(),
            self.auto_exposure.adaptation.as_mut(),
        ) {
            cur.settings = live;
        }
        self.refresh_gates(graphics);
    }

    // Replace the live post-process tunables, pushed every frame. The display
    // output flags are not part of the payload, so the EDR path survives.
    // Auto-exposure overwrites `exposure` each frame.
    fn push_post_process(&mut self, tunables: render_types::PostProcessTunables) {
        self.post_process.set_tunables(tunables);
    }

    // Set the live ambient light scale, uploaded every frame.
    fn push_ambient(&mut self, value: f32) {
        self.state.lights.set_ambient(value);
    }

    fn push_shadow_cadence(&mut self, cadence: init::ShadowCadence) {
        self.state.planner.shadow.cadence = cadence;
    }

    fn push_vsync(&mut self, on: bool) {
        WindowControl::set_vsync(self, on);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effect_settings_take_the_graphics_with_ray_tracing_from_the_plan() {
        let graphics = ResolvedGraphics::default();
        assert!(graphics.rt_reflections().is_some());
        let facts = BackendFacts {
            rt_capable: false,
            upscaling: false,
        };
        let plan = FeaturePlan::resolve(&graphics, facts);
        let settings = effect_settings(&graphics, &plan);
        assert_eq!(settings.rt_reflection, None);
        assert_eq!(settings.ssao, graphics.ssao());
    }
}

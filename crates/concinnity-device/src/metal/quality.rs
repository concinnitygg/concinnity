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
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::backend::feature_plan::{BackendFacts, FeatureChange, FeaturePlan};
use concinnity_engine::render::backend::init;
use concinnity_engine::render::post::device::PostExtent;
use concinnity_engine::render::post::ssao::SsaoPass;
use concinnity_engine::render::post::ssgi::SsgiPass;
use concinnity_engine::render::post::ssr::SsrPass;
use concinnity_engine::render::render_graph::{BackendTraits, BuildStatus, FeatureGates};
use concinnity_engine::settings::ResolvedGraphics;
use concinnity_engine::settings::apply::{
    ApplyPlan, RebuildFeature, RebuildSet, SettingsTarget, apply_plan, rebuild_each,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLTexture;

use super::auto_exposure::AutoExposureGpu;
use super::context::MtlContext;
use super::init::effects::{EffectSettings, build_auto_exposure, build_gbuffer};
use super::init::ray_tracing::build_rt_pipelines;
use super::post::post_device::MtlPostDevice;
use super::post::reflection_composite::MtlReflectionCompositePass;
use super::post::ssao::MtlSsaoPass;
use super::post::ssgi::MtlSsgiPass;
use super::post::ssr::MtlSsrPass;
use super::post::taa::MtlTaaPass;
use super::post::{
    GBufferState, build_reflection_composite, build_taa_pass, create_reflection_target,
};
use super::raytrace::{
    RtGpu, RtSceneGeometry, RtStaticGeometry, RtTextureCounts, build_rt_accel, raytracing_supported,
};

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
    GBuffer(GBufferState),
    Ssgi(MtlSsgiPass),
    AutoExposure(AutoExposureGpu),
}

// What a rebuild reads besides the context: the per-feature settings and the
// features the settings want built on this device.
struct Wanted {
    settings: EffectSettings,
    plan: FeaturePlan,
}

impl MtlContext {
    // Turn display sync (vsync) on or off at runtime via the view's backing
    // CAMetalLayer. Setting displaySyncEnabled is an idempotent property write
    // (no swapchain rebuild on Metal), so a redundant call is cheap. Backend
    // specific: Vulkan reaches the same end by rebuilding the swapchain with a
    // different present mode, so this does not live on the shared window layer.
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "`forward!` reaches it from the trait's `&mut self` method of the same name"
    )]
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
        self.lights.set_ambient(value);
    }

    pub(crate) fn set_shadow_cadence(&mut self, cadence: init::ShadowCadence) {
        self.state.planner.shadow.cadence = cadence;
    }

    // Update the live scalar sub-tunables of the SSAO / SSR / SSGI / auto-exposure
    // passes without rebuilding anything. The draw path rebuilds each pass's
    // per-frame uniform from these stored `*Settings` structs every frame
    // (`settings.params(...)`), so mutating the stored struct here is picked up on
    // the next draw. Only a feature that is currently on has a settings struct to
    // mutate; the rest are skipped (the value still persists for the next launch).
    // SSAO / SSR / auto-exposure settings are fully scalar, so they are replaced
    // wholesale; SSGI's trace resolution and ray count ride its rebuild (the
    // resolution sizes its targets), so only its scalar intensity / distance
    // are updated.
    pub(crate) fn update_quality_params(&mut self, g: &ResolvedGraphics) {
        if let (Some(live), Some(cur)) = (g.ssao(), self.ssao.settings.as_mut()) {
            *cur = live;
        }
        if let (Some(live), Some(cur)) = (g.ssr(), self.ssr.settings.as_mut()) {
            *cur = live;
        }
        if let (Some(live), Some(cur)) = (g.ssgi(), self.ssgi.settings.as_mut()) {
            cur.intensity = live.intensity;
            cur.max_distance = live.max_distance;
        }
        if let (Some(live), Some(cur)) = (g.auto_exposure(), self.auto_exposure.adaptation.as_mut())
        {
            cur.settings = live;
        }
        self.refresh_gates(g);
    }

    // Gate the runtime-switched passes on `graphics` and what is built now.
    pub(super) fn refresh_gates(&mut self, graphics: &ResolvedGraphics) {
        self.state.planner.gates = FeatureGates::new(graphics, &self.build_status(), METAL_TRAITS);
    }

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

    // Rebuild `features` to match `g`, between frames (the GraphicsSystem
    // drain runs before the next `draw_frame`). A feature whose build fails
    // keeps its previous state; the first error comes back once every feature
    // has run.
    fn rebuild_planned(&mut self, g: &ResolvedGraphics, features: RebuildSet) -> RenderResult<()> {
        // The resolved settings never ask for ray tracing the device cannot
        // run; a request that does is a probe / backend disagreement, reported
        // and refused rather than built.
        let plan = FeaturePlan::resolve(
            g,
            BackendFacts {
                rt_capable: raytracing_supported(&self.hw.device),
                upscaling: self.upscale.scaler.is_some(),
            },
        );
        let mut settings = EffectSettings::new(g);
        settings.rt_reflection = plan.rt;
        let wanted = Wanted { settings, plan };
        let rebuilt = rebuild_each(features, |feature| self.rebuild_feature(feature, &wanted));
        // SSR, SSGI and RT share the reflection target.
        let shares_target = [
            RebuildFeature::Ssr,
            RebuildFeature::Ssgi,
            RebuildFeature::RtReflections,
        ]
        .into_iter()
        .any(|f| features.contains(f));
        let target = if shares_target {
            self.reconcile_reflection_target(&wanted.settings)
        } else {
            Ok(())
        };
        self.refresh_gates(g);
        rebuilt.and(target)
    }

    fn rebuild_feature(&mut self, feature: RebuildFeature, wanted: &Wanted) -> RenderResult<()> {
        match feature {
            RebuildFeature::GBuffer => self.rebuild_gbuffer(wanted.plan.gbuffer),
            RebuildFeature::Taa => self.rebuild_taa(wanted.plan.taa_resolve),
            RebuildFeature::Ssao => self.rebuild_ssao(&wanted.settings),
            RebuildFeature::Ssr => self.rebuild_ssr_resolve(&wanted.settings),
            RebuildFeature::Ssgi => self.rebuild_ssgi(&wanted.settings),
            RebuildFeature::RtReflections => self.rebuild_rt(&wanted.settings),
            RebuildFeature::ReflectionComposite => {
                self.rebuild_reflection_composite(&wanted.settings)
            }
            RebuildFeature::AutoExposure => self.rebuild_auto_exposure(&wanted.settings),
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
        if change == FeatureChange::Build || (change == FeatureChange::Update && rescaled) {
            let render = (self.targets.hdr.width, self.targets.hdr.height);
            let fresh = build_reflection_composite(
                &self.build_device(),
                settings.reflection_blur_scale,
                render,
            )?;
            if let Some(old) = self.ssr.composite.replace(fresh) {
                self.retire_effect(RetiredEffect::Composite(old));
            }
        } else if change == FeatureChange::Drop
            && let Some(old) = self.ssr.composite.take()
        {
            self.retire_effect(RetiredEffect::Composite(old));
        }
        Ok(())
    }

    // The reflection target SSR, SSGI and RT write, present while any of them
    // runs and reduced to the RT trace resolution when RT fills it.
    fn reconcile_reflection_target(&mut self, settings: &EffectSettings) -> RenderResult<()> {
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

    // Ray-traced reflections. The resolve pipelines are kept once built, so a
    // toggle back on compiles nothing; the gate is `rt.settings`. Turning on
    // seeds the acceleration structure (it needs the resident geometry
    // buffers), turning off retires it. Skinned geometry joins the BVH on the
    // next frame's per-frame update, matching the init path.
    fn rebuild_rt(&mut self, settings: &EffectSettings) -> RenderResult<()> {
        self.rt.settings = settings.rt_reflection;
        if self.rt.settings.is_none() {
            // Between frames, so it is tagged with the next frame's id.
            self.rt.retire_accel(self.frame_ring_index);
            return Ok(());
        }
        if self.rt.pipelines.resolve.is_none() {
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

impl MtlContext {
    // Apply a settings change in the plan order every backend shares.
    pub(crate) fn apply_settings(
        &mut self,
        graphics: &ResolvedGraphics,
        plan: &ApplyPlan,
    ) -> RenderResult<()> {
        apply_plan(self, graphics, plan)
    }
}

impl SettingsTarget for MtlContext {
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

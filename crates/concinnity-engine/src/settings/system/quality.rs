// The quality-governed settings rows: the master "Graphics Quality" preset, the
// feature toggles, and the cycle quality knobs. A per-row change opts the preset
// out to Custom.

use concinnity_core::components::SettingCommand;
use concinnity_core::ecs::PipelineContext;
use concinnity_core::render::backend::QualitySettings;
use concinnity_core::render::error::RenderError;
use concinnity_core::render::ops::RenderOps;

use super::SettingsState;
use super::apply::RowOptions;
use super::rows::{set_cached_row_label, set_label_content};
use crate::config::Settings;
use crate::gfx::quality_preset;
use crate::gfx::system as gsys;
use crate::settings;
use crate::settings::SettingKey;

impl SettingsState {
    // A preset is a performance ceiling over the world's authored look (it never
    // enables a feature the world did not author), so picking a tier or Auto
    // clears the per-row quality overrides and re-derives the toggles, knobs, and
    // render scale from the world's authored config under the new ceiling. Custom
    // resolves to the no-op ceiling (the world's look). See gfx/quality_preset.rs.
    pub(super) fn apply_quality_preset(
        &mut self,
        ctx: &mut PipelineContext,
        ops: &mut RenderOps,
        cfg: &mut Settings,
        cmd: &SettingCommand,
    ) -> bool {
        let opts = settings::options(SettingKey::GraphicsQuality).unwrap_or(&[]);
        let cur = quality_preset::preset_index(self.quality_preset);
        let next = settings::cycle(cur, opts.len(), cmd.op);
        let preset = quality_preset::preset_at(next);
        self.quality_preset = preset;
        let ceiling = quality_preset::resolve_ceiling(preset, &self.gpu_profile);

        // Re-derive the live quality toggles from the world baseline under the
        // new ceiling (force off where disallowed; never turn on), then clamp
        // every cycle knob (the overrides are cleared).
        self.post_config = self.authored_post_config.clone();
        for (key, allowed) in [
            (SettingKey::Ssao, ceiling.ssao),
            (SettingKey::Ssr, ceiling.ssr),
            (
                SettingKey::RayTracedReflections,
                ceiling.ray_traced_reflections,
            ),
            (SettingKey::Ssgi, ceiling.ssgi),
            (SettingKey::AutoExposure, ceiling.auto_exposure),
        ] {
            if !allowed {
                gsys::set_quality_toggle(&mut self.post_config, key, false);
            }
        }
        for key in settings::QUALITY_CYCLE_KEYS {
            gsys::clamp_quality_cycle(&mut self.post_config, key, &ceiling, false);
        }
        // The composite FXAA flag rides PostProcessParams, so refresh it from the
        // re-derived AA mode before the push below.
        self.post_process.fxaa = self.post_config.aa_mode.fxaa_flag();
        let quality = gsys::derive_quality_settings(&self.post_config);
        record_quality_apply(ops, quality);
        // Auto-exposure may have flipped off; re-push the static post-process
        // params so exposure reverts.
        let params = self.post_process;
        ops.record(move |backend| backend.update_post_process(params));
        // Restart-required: the render scale only updates the row label (the
        // upscaler and targets are sized at init).
        self.render_scale = quality_preset::more_aggressive_upscale(
            self.authored_post_config.upscale_quality,
            ceiling.min_upscale,
        );
        // Re-derive the shadow knobs from the authored baselines. The cadence,
        // distance, and cascade count are live; the resolution and anisotropy are
        // restart-required, so they only relabel.
        self.shadow_map_size = self.authored_shadow_map_size.min(ceiling.shadow_map_size);
        self.shadow_update =
            quality_preset::clamp_shadow_update(self.authored_shadow_update, &ceiling);
        let update = self.shadow_update;
        ops.record(move |backend| backend.set_shadow_update(update));
        self.shadow_distance = self.authored_shadow_distance.min(ceiling.shadow_distance);
        let distance = self.shadow_distance;
        ops.record(move |backend| backend.set_shadow_distance(distance));
        self.shadow_cascades = self.authored_shadow_cascades.min(ceiling.shadow_cascades);
        let count = self.shadow_cascades;
        ops.record(move |backend| backend.set_shadow_cascades(count));
        self.anisotropy = self.authored_anisotropy.min(ceiling.anisotropy);

        // Persist the preset and drop the per-row quality overrides, so the next
        // launch re-resolves them from the world and ceiling exactly as this live
        // re-derive did.
        cfg.graphics.quality_preset = Some(preset);
        cfg.graphics.aa_mode = None;
        cfg.graphics.ssao = None;
        cfg.graphics.ssr = None;
        cfg.graphics.ray_traced_reflections = None;
        cfg.graphics.ssgi = None;
        cfg.graphics.auto_exposure = None;
        cfg.graphics.ssgi_resolution = None;
        cfg.graphics.ssgi_rays = None;
        cfg.graphics.ssgi_steps = None;
        cfg.graphics.reflection_blur_resolution = None;
        cfg.graphics.shadow_map_size = None;
        cfg.graphics.shadow_update = None;
        cfg.graphics.shadow_distance = None;
        cfg.graphics.shadow_cascades = None;
        cfg.graphics.anisotropy = None;
        cfg.graphics.render_scale = None;

        self.relabel_preset_dependents(ctx);
        // The master row's own label carries the Auto(tier) suffix.
        let label = quality_preset::preset_label(preset, &self.gpu_profile);
        if let Some(id) = cmd.value_label {
            set_label_content(ctx, id, &label);
        }
        true
    }

    // Refresh the rows a preset change re-derives from the init-captured
    // value-label ids; the menu's HitRegions are drained after init, so they
    // cannot be re-queried here.
    fn relabel_preset_dependents(&self, ctx: &mut PipelineContext) {
        for key in SettingKey::QUALITY_TOGGLES {
            let on = gsys::quality_toggle_on(&self.post_config, key).unwrap_or(false);
            self.relabel_option(ctx, key, on as usize);
        }
        self.relabel_option(
            ctx,
            SettingKey::RenderScale,
            settings::render_scale_index(self.render_scale),
        );
        for key in settings::QUALITY_CYCLE_KEYS {
            if let Some(idx) = gsys::quality_cycle_index(&self.post_config, key) {
                self.relabel_option(ctx, key, idx);
            }
        }
        for (key, idx) in [
            (
                SettingKey::ShadowMapSize,
                settings::shadow_resolution_index(self.shadow_map_size),
            ),
            (
                SettingKey::ShadowUpdate,
                settings::shadow_update_index(self.shadow_update),
            ),
            (
                SettingKey::ShadowDistance,
                settings::shadow_distance_index(self.shadow_distance),
            ),
            (
                SettingKey::ShadowCascades,
                settings::shadow_cascades_index(self.shadow_cascades),
            ),
            (
                SettingKey::Anisotropy,
                settings::anisotropy_index(self.anisotropy),
            ),
        ] {
            self.relabel_option(ctx, key, idx);
        }
    }

    // Set a captured cycle row's value label to its option at `index`.
    fn relabel_option(&self, ctx: &mut PipelineContext, key: SettingKey, index: usize) {
        if let Some(text) = settings::options(key).and_then(|o| o.get(index).copied()) {
            set_cached_row_label(&self.cycle_value_labels, ctx, key, text);
        }
    }

    // Flip a quality feature on the stored config, persist it, and rebuild the
    // affected render resources live (a no-op backend keeps the choice for the
    // next launch).
    pub(super) fn apply_quality_toggle(
        &mut self,
        ctx: &mut PipelineContext,
        ops: &mut RenderOps,
        cfg: &mut Settings,
        cmd: &SettingCommand,
        opts: RowOptions,
    ) -> &'static str {
        let key = cmd.setting;
        let cur = gsys::quality_toggle_on(&self.post_config, key).unwrap_or(false);
        let next = settings::cycle(cur as usize, opts.len(), cmd.op);
        let on = next == 1;
        gsys::set_quality_toggle(&mut self.post_config, key, on);
        match key {
            SettingKey::Ssao => cfg.graphics.ssao = Some(on),
            SettingKey::Ssr => cfg.graphics.ssr = Some(on),
            SettingKey::RayTracedReflections => cfg.graphics.ray_traced_reflections = Some(on),
            SettingKey::Ssgi => cfg.graphics.ssgi = Some(on),
            SettingKey::AutoExposure => cfg.graphics.auto_exposure = Some(on),
            other => tracing::warn!("SettingsSystem: no persisted field for {other:?}"),
        }
        self.opt_out_of_preset(ctx, cfg);
        let quality = gsys::derive_quality_settings(&self.post_config);
        record_quality_apply(ops, quality);
        // Auto-exposure overwrites the backend's live exposure each frame while
        // it runs, so its copy freezes at the last adapted value once toggled
        // off. Re-push the static params (the authored / slider EV) so exposure
        // reverts; on a toggle-on the AE loop overwrites it next frame.
        if key == SettingKey::AutoExposure {
            let params = self.post_process;
            ops.record(move |backend| backend.update_post_process(params));
        }
        opts[next]
    }

    // Cycle a quality knob on the stored config, persist it, and rebuild the
    // affected effect live through the same path as the toggles (the sub-tunable
    // travels in the feature's settings payload).
    pub(super) fn apply_quality_cycle(
        &mut self,
        ctx: &mut PipelineContext,
        ops: &mut RenderOps,
        cfg: &mut Settings,
        cmd: &SettingCommand,
        opts: RowOptions,
    ) -> &'static str {
        let key = cmd.setting;
        let cur = gsys::quality_cycle_index(&self.post_config, key).unwrap_or(0);
        let next = settings::cycle(cur, opts.len(), cmd.op);
        gsys::set_quality_cycle(&mut self.post_config, key, next);
        let post = &self.post_config;
        match key {
            SettingKey::AaMode => cfg.graphics.aa_mode = Some(post.aa_mode),
            SettingKey::SsgiResolution => cfg.graphics.ssgi_resolution = Some(post.ssgi_resolution),
            SettingKey::SsgiRays => cfg.graphics.ssgi_rays = Some(post.ssgi_rays),
            SettingKey::SsgiSteps => cfg.graphics.ssgi_steps = Some(post.ssgi_steps),
            SettingKey::ReflectionBlurResolution => {
                cfg.graphics.reflection_blur_resolution = Some(post.reflection_blur_resolution)
            }
            other => tracing::warn!("SettingsSystem: no persisted field for {other:?}"),
        }
        self.opt_out_of_preset(ctx, cfg);
        let quality = gsys::derive_quality_settings(&self.post_config);
        record_quality_apply(ops, quality);
        // The AA mode also drives the composite FXAA flag, which rides
        // PostProcessParams rather than the rebuild above.
        if key == SettingKey::AaMode {
            self.post_process.fxaa = self.post_config.aa_mode.fxaa_flag();
            let params = self.post_process;
            ops.record(move |backend| backend.update_post_process(params));
        }
        opts[next]
    }

    // An explicit per-row quality change opts the master preset out to Custom
    // (no ceiling clamps the user's choice) and relabels the master row.
    pub(super) fn opt_out_of_preset(&mut self, ctx: &mut PipelineContext, cfg: &mut Settings) {
        self.quality_preset = quality_preset::QualityPreset::Custom;
        cfg.graphics.quality_preset = Some(self.quality_preset);
        set_cached_row_label(
            &self.cycle_value_labels,
            ctx,
            SettingKey::GraphicsQuality,
            self.quality_preset.name(),
        );
    }
}

// Record a live quality rebuild. A failure is logged, and running out of device
// memory raises the replay's memory pressure for the streaming valve.
pub(super) fn record_quality_apply(ops: &mut RenderOps, quality: QualitySettings) {
    ops.record_with(move |backend, out| {
        if let Err(e) = backend.apply_quality_settings(quality) {
            tracing::error!("SettingsSystem: quality rebuild failed: {e}");
            out.memory_pressure |= matches!(e, RenderError::OutOfDeviceMemory(_));
        }
    });
}

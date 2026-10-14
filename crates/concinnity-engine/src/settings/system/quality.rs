// The quality-governed settings rows: the master "Graphics Quality" preset, the
// feature toggles, and the cycle quality knobs. A per-row change opts the preset
// out to Custom.

use concinnity_core::components::{SettingCommand, SettingOp};
use concinnity_core::ecs::PipelineContext;

use super::SettingsState;
use super::apply::RowOptions;
use super::rows::{set_cached_row_label, set_label_content};
use crate::config::Settings;
use crate::settings;
use crate::settings::SettingKey;
use concinnity_core::settings::QualityPreset;
use concinnity_core::settings::quality::{preset_at, preset_index, preset_label, resolve_ceiling};
use concinnity_core::settings::rows::{
    QUALITY_CYCLES, QUALITY_TOGGLES, QualityCycle, QualityToggle,
};

impl SettingsState {
    // A preset is a performance ceiling over the world's authored look (it never
    // enables a feature the world did not author), so picking a tier or Auto
    // clears the per-row quality overrides and re-resolves the preset-governed
    // group from the world's baseline under the new ceiling, exactly as the next
    // launch will. Custom resolves to the no-op ceiling (the world's look). See
    // `concinnity_core::settings::quality`.
    pub(super) fn apply_quality_preset(
        &mut self,
        ctx: &mut PipelineContext,
        cfg: &mut Settings,
        cmd: &SettingCommand,
    ) -> bool {
        let opts = settings::options(SettingKey::GraphicsQuality).unwrap_or(&[]);
        let cur = preset_index(self.quality_preset);
        let next = settings::cycle(cur, opts.len(), cmd.op);
        let preset = preset_at(next);
        self.quality_preset = preset;
        let ceiling = resolve_ceiling(preset, &self.gpu_profile);

        cfg.graphics.quality_preset = Some(preset);
        for row in &QUALITY_TOGGLES {
            *(row.persisted.get_mut)(&mut cfg.graphics) = None;
        }
        for row in &QUALITY_CYCLES {
            (row.clear)(&mut cfg.graphics);
        }
        cfg.graphics.shadow_map_size = None;
        cfg.graphics.shadow_update = None;
        cfg.graphics.shadow_distance = None;
        cfg.graphics.shadow_cascades = None;
        cfg.graphics.anisotropy = None;
        cfg.graphics.render_scale = None;
        self.resolved
            .set_quality(self.resolve_quality(&cfg.graphics, &ceiling));

        self.relabel_preset_dependents(ctx);
        // The master row's own label carries the Auto(tier) suffix.
        let label = preset_label(preset, &self.gpu_profile);
        if let Some(id) = cmd.value_label {
            set_label_content(ctx, id, &label);
        }
        true
    }

    // Refresh the rows a preset change re-derives from the init-captured
    // value-label ids; the menu's HitRegions are drained after init, so they
    // cannot be re-queried here.
    fn relabel_preset_dependents(&self, ctx: &mut PipelineContext) {
        for row in &QUALITY_TOGGLES {
            let on = (row.get)(&self.resolved.graphics);
            self.relabel_option(ctx, row.key, on as usize);
        }
        self.relabel_option(
            ctx,
            SettingKey::RenderScale,
            settings::render_scale_index(self.resolved.graphics.anti_aliasing.upscale_quality),
        );
        for row in &QUALITY_CYCLES {
            self.relabel_option(ctx, row.key, (row.index)(&self.resolved.graphics));
        }
        for (key, idx) in [
            (
                SettingKey::ShadowMapSize,
                settings::shadow_resolution_index(self.resolved.graphics.shadows.map_size),
            ),
            (
                SettingKey::ShadowUpdate,
                settings::shadow_update_index(self.resolved.graphics.shadows.update),
            ),
            (
                SettingKey::ShadowDistance,
                settings::shadow_distance_index(self.resolved.graphics.shadows.distance),
            ),
            (
                SettingKey::ShadowCascades,
                settings::shadow_cascades_index(self.resolved.graphics.shadows.cascades),
            ),
            (
                SettingKey::Anisotropy,
                settings::anisotropy_index(self.resolved.graphics.texture_filtering.anisotropy),
            ),
        ] {
            self.relabel_option(ctx, key, idx);
        }
    }

    // Set a captured cycle row's value label to its option at `index`.
    fn relabel_option(&self, ctx: &mut PipelineContext, key: SettingKey, index: usize) {
        let Some(text) = settings::options(key).and_then(|o| o.get(index).copied()) else {
            return;
        };
        if settings::setting_available(key, &self.capabilities, self.gpu_profile.vendor) {
            set_cached_row_label(&self.cycle_value_labels, ctx, key, text);
        } else {
            let text = settings::unsupported_label(text);
            set_cached_row_label(&self.cycle_value_labels, ctx, key, &text);
        }
    }

    // Flip a quality feature on the stored config and persist it; the batch's
    // plan rebuilds the affected render resources.
    pub(super) fn apply_quality_toggle(
        &mut self,
        ctx: &mut PipelineContext,
        cfg: &mut Settings,
        row: &QualityToggle,
        opts: RowOptions,
        op: SettingOp,
    ) -> &'static str {
        let cur = (row.get)(&self.resolved.graphics);
        let next = settings::cycle(cur as usize, opts.len(), op);
        let on = next == 1;
        (row.set)(&mut self.resolved.graphics, on);
        *(row.persisted.get_mut)(&mut cfg.graphics) = Some(on);
        self.opt_out_of_preset(ctx, cfg);
        opts[next]
    }

    // Cycle a quality knob on the stored config and persist it; the batch's
    // plan rebuilds the affected effect.
    pub(super) fn apply_quality_cycle(
        &mut self,
        ctx: &mut PipelineContext,
        cfg: &mut Settings,
        row: &QualityCycle,
        opts: RowOptions,
        op: SettingOp,
    ) -> &'static str {
        let cur = (row.index)(&self.resolved.graphics);
        let next = settings::cycle(cur, opts.len(), op);
        (row.set)(&mut self.resolved.graphics, next);
        (row.persist)(&self.resolved.graphics, &mut cfg.graphics);
        self.opt_out_of_preset(ctx, cfg);
        opts[next]
    }

    // An explicit per-row quality change opts the master preset out to Custom
    // (no ceiling clamps the user's choice) and relabels the master row.
    pub(super) fn opt_out_of_preset(&mut self, ctx: &mut PipelineContext, cfg: &mut Settings) {
        self.quality_preset = QualityPreset::Custom;
        cfg.graphics.quality_preset = Some(self.quality_preset);
        set_cached_row_label(
            &self.cycle_value_labels,
            ctx,
            SettingKey::GraphicsQuality,
            self.quality_preset.name(),
        );
    }
}

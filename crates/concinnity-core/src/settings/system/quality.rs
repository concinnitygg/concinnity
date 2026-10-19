// The quality-governed settings rows: the master "Graphics Quality" preset, the
// feature toggles, and the cycle quality knobs. A per-row change opts the preset
// out to Custom.

use crate::components::{SettingCommand, SettingOp};
use crate::ecs::PipelineContext;

use super::SettingsState;
use super::apply::RowOptions;
use super::rows::{set_cached_row_label, set_label_content};
use crate::platform::CompiledPlatforms;
use crate::settings::quality::{preset_label, resolve_ceiling};
use crate::settings::rows::{QUALITY_CYCLES, QUALITY_TOGGLES, QualityCycle, QualityToggle};
use crate::settings::{self, QualityPreset, SettingKey, StoredSettings, levels, registry};

impl SettingsState {
    // A preset is a performance ceiling over the world's authored look (it never
    // enables a feature the world did not author), so picking a tier or Auto
    // clears the per-row quality overrides and re-resolves the preset-governed
    // group from the world's baseline under the new ceiling, exactly as the next
    // launch will. Custom resolves to the no-op ceiling (the world's look). See
    // `crate::settings::quality`.
    pub(super) fn apply_quality_preset(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        cfg: &mut StoredSettings,
        cmd: &SettingCommand,
    ) -> bool {
        let opts = settings::options(SettingKey::GraphicsQuality).unwrap_or(&[]);
        let cur = levels::GRAPHICS_QUALITY.index_of(self.quality_preset);
        let next = registry::cycle(cur, opts.len(), cmd.op);
        let preset = levels::GRAPHICS_QUALITY.at(next);
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
    fn relabel_preset_dependents(&self, ctx: &mut PipelineContext<'_>) {
        for row in &QUALITY_TOGGLES {
            let on = (row.get)(&self.resolved.graphics);
            self.relabel_option(ctx, row.key, usize::from(on));
        }
        self.relabel_option(
            ctx,
            SettingKey::RenderScale,
            levels::RENDER_SCALE.index_of(self.resolved.graphics.anti_aliasing.upscale_quality),
        );
        for row in &QUALITY_CYCLES {
            self.relabel_option(ctx, row.key, (row.index)(&self.resolved.graphics));
        }
        for (key, idx) in [
            (
                SettingKey::ShadowMapSize,
                levels::SHADOW_RESOLUTION.index_of(self.resolved.graphics.shadows.map_size),
            ),
            (
                SettingKey::ShadowUpdate,
                levels::SHADOW_UPDATE.index_of(self.resolved.graphics.shadows.update),
            ),
            (
                SettingKey::ShadowDistance,
                levels::SHADOW_DISTANCE.index_of(self.resolved.graphics.shadows.distance),
            ),
            (
                SettingKey::ShadowCascades,
                levels::SHADOW_CASCADES.index_of(self.resolved.graphics.shadows.cascades),
            ),
            (
                SettingKey::Anisotropy,
                levels::ANISOTROPY.index_of(self.resolved.graphics.texture_filtering.anisotropy),
            ),
        ] {
            self.relabel_option(ctx, key, idx);
        }
    }

    // Set a captured cycle row's value label to its option at `index`.
    fn relabel_option(&self, ctx: &mut PipelineContext<'_>, key: SettingKey, index: usize) {
        let Some(text) = settings::options(key).and_then(|o| o.get(index).copied()) else {
            return;
        };
        let compiled = ctx.resource::<CompiledPlatforms>().map_or(&[][..], |c| c.0);
        if registry::setting_available(key, &self.capabilities, self.gpu_profile.vendor, compiled) {
            set_cached_row_label(&self.cycle_value_labels, ctx, key, text);
        } else {
            let text = registry::unsupported_label(text);
            set_cached_row_label(&self.cycle_value_labels, ctx, key, &text);
        }
    }

    // Flip a quality feature on the stored config and persist it; the batch's
    // plan rebuilds the affected render resources.
    pub(super) fn apply_quality_toggle(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        cfg: &mut StoredSettings,
        row: &QualityToggle,
        opts: RowOptions,
        op: SettingOp,
    ) -> &'static str {
        let cur = (row.get)(&self.resolved.graphics);
        let next = registry::cycle(usize::from(cur), opts.len(), op);
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
        ctx: &mut PipelineContext<'_>,
        cfg: &mut StoredSettings,
        row: &QualityCycle,
        opts: RowOptions,
        op: SettingOp,
    ) -> &'static str {
        let cur = (row.index)(&self.resolved.graphics);
        let next = registry::cycle(cur, opts.len(), op);
        (row.set)(&mut self.resolved.graphics, next);
        (row.persist)(&self.resolved.graphics, &mut cfg.graphics);
        self.opt_out_of_preset(ctx, cfg);
        opts[next]
    }

    // An explicit per-row quality change opts the master preset out to Custom
    // (no ceiling clamps the user's choice) and relabels the master row.
    pub(super) fn opt_out_of_preset(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        cfg: &mut StoredSettings,
    ) {
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

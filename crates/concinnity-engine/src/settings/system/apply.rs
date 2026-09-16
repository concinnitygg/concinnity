// The per-frame SettingCommand drain. Each command goes to the handler for its
// row family, which changes the value, applies it (live to the backend where the
// feature supports it, persist-only where a restart is required), refreshes the
// row's value label, and records the change in the batch's settings snapshot.

use concinnity_core::components::{
    AudioCommand, AudioTarget, ControlsCommand, SettingCommand, SettingOp, Sprite, WindowMode,
};
use concinnity_core::ecs::FrameRateCap;
use concinnity_core::ecs::PipelineContext;
use concinnity_core::render::ops::RenderOps;
use concinnity_core::window::display_mode;

use super::SettingsState;
use super::rows::{set_label_content, set_rows_grayed, set_sprite_x};
use crate::config::{GraphicsSettings, Settings};
use crate::gfx::system as gsys;
use crate::settings;
use crate::settings::SliderTarget;

// A cycle row's option labels.
pub(super) type RowOptions = &'static [&'static str];

// An Off/On row backed by one `SettingsState` bool and its persisted override.
pub(super) struct BoolRow {
    pub(super) key: &'static str,
    pub(super) state: fn(&mut SettingsState) -> &mut bool,
    pub(super) persisted: fn(&mut GraphicsSettings) -> &mut Option<bool>,
}

// None of these rows is governed by the quality preset. vsync applies live and
// the stats-HUD toggles publish through HudPrefs; the rest are restart-required.
pub(super) static BOOL_ROWS: [BoolRow; 8] = [
    BoolRow {
        key: "vsync",
        state: |s| &mut s.vsync,
        persisted: |g| &mut g.vsync,
    },
    BoolRow {
        key: "perf_stats",
        state: |s| &mut s.perf_stats,
        persisted: |g| &mut g.perf_stats,
    },
    BoolRow {
        key: "show_fps",
        state: |s| &mut s.show_fps,
        persisted: |g| &mut g.show_fps,
    },
    BoolRow {
        key: "show_vram",
        state: |s| &mut s.show_vram,
        persisted: |g| &mut g.show_vram,
    },
    BoolRow {
        key: "occlusion_two_pass",
        state: |s| &mut s.occlusion_two_pass,
        persisted: |g| &mut g.occlusion_two_pass,
    },
    BoolRow {
        key: "temporal_upscaling",
        state: |s| &mut s.temporal_upscaling,
        persisted: |g| &mut g.temporal_upscaling,
    },
    BoolRow {
        key: "hdr_display",
        state: |s| &mut s.hdr_display,
        persisted: |g| &mut g.hdr_display,
    },
    BoolRow {
        key: "hdr_pq",
        state: |s| &mut s.hdr_pq,
        persisted: |g| &mut g.hdr_pq,
    },
];

impl SettingsState {
    pub(super) fn apply_setting_commands(
        &mut self,
        ctx: &mut PipelineContext,
        ops: &mut RenderOps,
    ) {
        // Clone the commands out of the queue so the ctx borrow is released
        // before the handlers, which need &mut ctx.
        let setting_cmds: Vec<SettingCommand> = match ctx.events::<SettingCommand>() {
            Some(events) => events.read(&mut self.setting_cmd_cursor).cloned().collect(),
            None => Vec::new(),
        };
        // One settings snapshot serves the whole command batch: loaded lazily
        // (from the in-memory cache after the first change, so a
        // queued-but-unflushed background write is never re-read stale from
        // disk), mutated by the commands below, then queued once to the
        // background writer -- the render thread never blocks on settings disk
        // I/O.
        let mut cfg = self.settings_cache.take();
        let mut cfg_dirty = false;
        let tree = ctx
            .resource::<concinnity_host::store::paths::StateTree>()
            .cloned();
        for mut cmd in setting_cmds {
            let cfg = cfg.get_or_insert_with(|| Settings::load(tree.as_ref()));
            if !self.slider_step_fraction(ctx, &mut cmd) {
                continue;
            }
            cfg_dirty |= self.apply_setting_command(ctx, ops, cfg, &cmd);
        }
        // Hand the batch's snapshot to the background writer (spawned on the
        // first persisted change) and keep it as the cache the next change
        // starts from.
        if let Some(cfg) = cfg {
            if cfg_dirty {
                self.settings_writer
                    .get_or_insert_with(|| super::writer::SettingsWriter::spawn(tree.clone()))
                    .save(cfg.clone());
            }
            self.settings_cache = Some(cfg);
        }
    }

    // A Next/Prev on a slider row steps its value by `SLIDER_STEP_FRACTION` of the
    // range (a focused row's Left/Right), rewriting the op into the SetFraction a
    // drag sends. The handle's position carries the current fraction: it is placed
    // from the persisted value at init and moved on every change since. Returns
    // false when the slider has no captured track, so the command is skipped.
    fn slider_step_fraction(&self, ctx: &PipelineContext, cmd: &mut SettingCommand) -> bool {
        if !matches!(cmd.op, SettingOp::Next | SettingOp::Prev)
            || settings::slider(&cmd.setting).is_none()
        {
            return true;
        }
        let Some(cur) = self
            .sliders
            .iter()
            .find(|s| s.key == cmd.setting)
            .and_then(|s| {
                let hx = ctx
                    .query::<Sprite>()
                    .find(|sp| sp.asset_id == s.handle_id)
                    .map(|sp| sp.x)?;
                let travel = (s.track_w - s.handle_w).max(f32::EPSILON);
                Some(((hx - s.track_x) / travel).clamp(0.0, 1.0))
            })
        else {
            return false;
        };
        let step = if matches!(cmd.op, SettingOp::Prev) {
            -settings::SLIDER_STEP_FRACTION
        } else {
            settings::SLIDER_STEP_FRACTION
        };
        cmd.op = SettingOp::SetFraction((cur + step).clamp(0.0, 1.0));
        true
    }

    // Dispatch one command to its row family's handler. Returns whether the
    // command changed the batch's settings snapshot.
    fn apply_setting_command(
        &mut self,
        ctx: &mut PipelineContext,
        ops: &mut RenderOps,
        cfg: &mut Settings,
        cmd: &SettingCommand,
    ) -> bool {
        match cmd.op {
            SettingOp::Rebind(key) => return self.apply_key_rebind(ctx, ops, cfg, cmd, key),
            SettingOp::RebindButton(button) => {
                return self.apply_pad_rebind(ctx, cfg, cmd, button);
            }
            SettingOp::SetFraction(frac) => return self.apply_slider(ctx, ops, cfg, cmd, frac),
            SettingOp::Next | SettingOp::Prev | SettingOp::SetIndex(_) => {}
        }
        if cmd.setting == "graphics_quality" {
            return self.apply_quality_preset(ctx, ops, cfg, cmd);
        }
        if cmd.setting == "resolution" {
            return self.apply_resolution(ctx, ops, cfg, cmd);
        }
        let Some(opts) = settings::options(&cmd.setting) else {
            tracing::warn!("SettingsSystem: unknown setting '{}'", cmd.setting);
            return false;
        };
        let new_text = match BOOL_ROWS.iter().find(|row| row.key == cmd.setting) {
            Some(row) => Some(self.apply_bool_row(ctx, ops, cfg, row, opts, cmd.op)),
            None => match cmd.setting.as_str() {
                "master_volume" | "music_volume" | "sfx_volume" | "voice_volume" => {
                    apply_volume(ctx, cfg, cmd, opts)
                }
                "fps_cap" | "window_mode" | "render_scale" | "upscale_backend" => {
                    self.apply_display_row(ctx, ops, cfg, cmd, opts)
                }
                key if settings::is_quality_toggle(key) => {
                    Some(self.apply_quality_toggle(ctx, ops, cfg, cmd, opts))
                }
                key if gsys::is_quality_cycle(key) => {
                    Some(self.apply_quality_cycle(ctx, ops, cfg, cmd, opts))
                }
                "shadow_map_size" | "anisotropy" | "shadow_update" | "shadow_distance"
                | "shadow_cascades" => self.apply_shadow_row(ctx, ops, cfg, cmd, opts),
                "frames_in_flight" | "texture_quality" => self.apply_system_row(cfg, cmd, opts),
                other => {
                    tracing::warn!("SettingsSystem: unknown setting '{other}'");
                    None
                }
            },
        };
        let Some(text) = new_text else {
            return false;
        };
        if let Some(label_id) = cmd.value_label {
            set_label_content(ctx, label_id, text);
        }
        true
    }

    // Apply a slider's value live, move its handle, refresh its label, and persist
    // only on the commit frame (drag release).
    fn apply_slider(
        &mut self,
        ctx: &mut PipelineContext,
        ops: &mut RenderOps,
        cfg: &mut Settings,
        cmd: &SettingCommand,
        frac: f32,
    ) -> bool {
        let Some(slider) = settings::slider(&cmd.setting) else {
            tracing::warn!("SettingsSystem: unknown slider '{}'", cmd.setting);
            return false;
        };
        let value = slider.value_at(frac);
        let stored = (slider.apply)(value);
        match &slider.target {
            SliderTarget::PostProcess { field, .. } => {
                *(field.get_mut)(&mut self.post_process) = stored;
                let params = self.post_process;
                ops.record(move |backend| backend.update_post_process(params));
            }
            // The stored config is what a later rebuild re-derives from; the live
            // push mutates the backend's settings without one.
            SliderTarget::PostConfig { field, .. } => {
                *(field.get_mut)(&mut self.post_config) = stored;
                let quality = gsys::derive_quality_settings(&self.post_config);
                ops.record(move |backend| backend.update_quality_params(quality));
            }
            SliderTarget::Ambient { .. } => {
                self.ambient_intensity = stored;
                let params = self.post_process;
                ops.record(move |backend| backend.update_post_process(params));
                ops.record(move |backend| backend.set_ambient_intensity(stored));
            }
            SliderTarget::Controls { command, .. } => {
                ctx.events_mut::<ControlsCommand>().send(command(stored));
            }
        }
        if let Some(s) = self.sliders.iter().find(|s| s.key == cmd.setting) {
            let hx = s.track_x + frac.clamp(0.0, 1.0) * (s.track_w - s.handle_w).max(0.0);
            set_sprite_x(ctx, s.handle_id, hx);
        }
        if let Some(label_id) = cmd.value_label {
            set_label_content(ctx, label_id, &(slider.format)(value));
        }
        if cmd.persist {
            slider.persist(cfg, value);
        }
        cmd.persist
    }

    // The Resolution row cycles the enumerated display modes rather than a static
    // option list. Fullscreen-only: the backend holds the display to the chosen
    // mode while the window is fullscreen, and the row is grayed and inert in the
    // other modes, so the window is never resized here. Independent of the
    // quality preset.
    fn apply_resolution(
        &mut self,
        ctx: &mut PipelineContext,
        ops: &mut RenderOps,
        cfg: &mut Settings,
        cmd: &SettingCommand,
    ) -> bool {
        if self.display_modes.is_empty() {
            return false;
        }
        let cur = display_mode::index_of(&self.display_modes, self.effective_resolution());
        let next = settings::cycle(cur, self.display_modes.len(), cmd.op);
        let mode = self.display_modes[next];
        self.resolution = Some(mode);
        ops.record(move |backend| backend.set_display_mode(mode));
        cfg.graphics.resolution = Some([mode.width, mode.height, mode.refresh_hz]);
        if let Some(label_id) = cmd.value_label {
            set_label_content(ctx, label_id, &mode.label());
        }
        true
    }

    fn apply_bool_row(
        &mut self,
        ctx: &mut PipelineContext,
        ops: &mut RenderOps,
        cfg: &mut Settings,
        row: &BoolRow,
        opts: RowOptions,
        op: SettingOp,
    ) -> &'static str {
        let next = settings::cycle(*(row.state)(self) as usize, opts.len(), op);
        let on = next == 1;
        *(row.state)(self) = on;
        *(row.persisted)(&mut cfg.graphics) = Some(on);
        match row.key {
            "vsync" => ops.record(move |backend| backend.set_vsync(on)),
            // The stats master grays or restores its two sub-rows.
            "perf_stats" => set_rows_grayed(ctx, &self.perf_sub_row_labels, !on),
            _ => {}
        }
        opts[next]
    }

    // Shadow resolution and anisotropy are restart-required (the shadow map array
    // and the scene sampler are built once at init), so they persist and relabel
    // only. The cadence, distance, and cascade count are read each frame, so they
    // also push live. Every change opts the preset out to Custom.
    fn apply_shadow_row(
        &mut self,
        ctx: &mut PipelineContext,
        ops: &mut RenderOps,
        cfg: &mut Settings,
        cmd: &SettingCommand,
        opts: RowOptions,
    ) -> Option<&'static str> {
        let next = match cmd.setting.as_str() {
            "shadow_map_size" => {
                let cur = settings::shadow_resolution_index(self.shadow_map_size);
                let next = settings::cycle(cur, opts.len(), cmd.op);
                self.shadow_map_size = settings::shadow_resolution_at(next);
                cfg.graphics.shadow_map_size = Some(self.shadow_map_size);
                next
            }
            "anisotropy" => {
                let cur = settings::anisotropy_index(self.anisotropy);
                let next = settings::cycle(cur, opts.len(), cmd.op);
                self.anisotropy = settings::anisotropy_at(next);
                cfg.graphics.anisotropy = Some(self.anisotropy);
                next
            }
            "shadow_update" => {
                let cur = settings::shadow_update_index(self.shadow_update);
                let next = settings::cycle(cur, opts.len(), cmd.op);
                self.shadow_update = settings::shadow_update_at(next);
                let update = self.shadow_update;
                ops.record(move |backend| backend.set_shadow_update(update));
                cfg.graphics.shadow_update = Some(self.shadow_update);
                next
            }
            "shadow_distance" => {
                let cur = settings::shadow_distance_index(self.shadow_distance);
                let next = settings::cycle(cur, opts.len(), cmd.op);
                self.shadow_distance = settings::shadow_distance_at(next);
                let distance = self.shadow_distance;
                ops.record(move |backend| backend.set_shadow_distance(distance));
                cfg.graphics.shadow_distance = Some(self.shadow_distance);
                next
            }
            "shadow_cascades" => {
                let cur = settings::shadow_cascades_index(self.shadow_cascades);
                let next = settings::cycle(cur, opts.len(), cmd.op);
                self.shadow_cascades = settings::shadow_cascades_at(next);
                let count = self.shadow_cascades;
                ops.record(move |backend| backend.set_shadow_cascades(count));
                cfg.graphics.shadow_cascades = Some(self.shadow_cascades);
                next
            }
            other => {
                tracing::warn!("SettingsSystem: unknown shadow setting '{other}'");
                return None;
            }
        };
        self.opt_out_of_preset(ctx, cfg);
        Some(opts[next])
    }

    // The frame-rate cap and window mode apply live; render scale and the
    // upscaler are restart-required (the upscaler and render targets are built
    // once at init). Only render scale is governed by the quality preset.
    fn apply_display_row(
        &mut self,
        ctx: &mut PipelineContext,
        ops: &mut RenderOps,
        cfg: &mut Settings,
        cmd: &SettingCommand,
        opts: RowOptions,
    ) -> Option<&'static str> {
        let next = match cmd.setting.as_str() {
            "fps_cap" => {
                let cur = settings::fps_cap_index(self.fps_cap);
                let next = settings::cycle(cur, opts.len(), cmd.op);
                self.fps_cap = settings::fps_cap_at(next);
                // No backend call: the App-level pacer reads the republished cap
                // before the next step, and the change re-bases its deadline.
                ctx.insert_resource(FrameRateCap(self.fps_cap));
                cfg.graphics.fps_cap = Some(self.fps_cap);
                next
            }
            "window_mode" => {
                let cur = settings::window_mode_index(self.window_args.mode);
                let next = settings::cycle(cur, opts.len(), cmd.op);
                let mode = settings::window_mode_at(next);
                self.window_args.mode = mode;
                ops.record(move |backend| backend.set_window_mode(mode));
                // Borderless and fullscreen leave the window at the display size,
                // so returning to windowed re-applies the remembered size.
                if mode == WindowMode::Windowed {
                    let (w, h) = (self.window_args.width, self.window_args.height);
                    ops.record(move |backend| backend.set_window_size(w, h));
                }
                // The Resolution row only applies in fullscreen.
                set_rows_grayed(
                    ctx,
                    &self.resolution_row_labels,
                    mode != WindowMode::Fullscreen,
                );
                cfg.graphics.window_mode = Some(mode);
                next
            }
            "render_scale" => {
                let cur = settings::render_scale_index(self.render_scale);
                let next = settings::cycle(cur, opts.len(), cmd.op);
                self.render_scale = settings::render_scale_at(next);
                cfg.graphics.render_scale = Some(self.render_scale);
                self.opt_out_of_preset(ctx, cfg);
                next
            }
            "upscale_backend" => {
                // Skip upscalers this GPU vendor does not offer (DLSS NVIDIA-only,
                // XeSS Intel-only); Auto and FSR3 are always available, so the
                // loop terminates.
                let cur = settings::upscale_backend_index(self.upscale_backend);
                let mut next = settings::cycle(cur, opts.len(), cmd.op);
                while !settings::upscale_backend_available(
                    settings::upscale_backend_at(next),
                    self.gpu_profile.vendor,
                ) {
                    next = settings::cycle(next, opts.len(), cmd.op);
                }
                self.upscale_backend = settings::upscale_backend_at(next);
                cfg.graphics.upscale_backend = Some(self.upscale_backend);
                next
            }
            other => {
                tracing::warn!("SettingsSystem: unknown display setting '{other}'");
                return None;
            }
        };
        Some(opts[next])
    }

    // Restart-required (the ring buffers, cull pipeline, and streaming pool are
    // sized once at init) and independent of the quality preset.
    fn apply_system_row(
        &mut self,
        cfg: &mut Settings,
        cmd: &SettingCommand,
        opts: RowOptions,
    ) -> Option<&'static str> {
        let next = match cmd.setting.as_str() {
            "frames_in_flight" => {
                let cur = settings::frames_in_flight_index(self.frames_in_flight as u32);
                let next = settings::cycle(cur, opts.len(), cmd.op);
                self.frames_in_flight = settings::frames_in_flight_at(next) as usize;
                cfg.graphics.frames_in_flight = Some(self.frames_in_flight as u32);
                next
            }
            // One row drives both the streaming pool cap and the per-frame upload
            // budget.
            "texture_quality" => {
                let cur = settings::texture_quality_index(self.texture_cap);
                let next = settings::cycle(cur, opts.len(), cmd.op);
                let (cap, budget) = settings::texture_quality_at(next);
                self.texture_cap = cap;
                self.texture_budget = budget;
                cfg.graphics.texture_cap = Some(cap);
                cfg.graphics.texture_budget = Some(budget);
                next
            }
            other => {
                tracing::warn!("SettingsSystem: unknown system setting '{other}'");
                return None;
            }
        };
        Some(opts[next])
    }
}

// Cycle a volume row's gain, persist it, and hand it to AudioSystem as an
// AudioCommand drained this same tick. A world with no audio has no AudioSystem
// to drain it; the persisted value then applies at the next audio init.
fn apply_volume(
    ctx: &mut PipelineContext,
    cfg: &mut Settings,
    cmd: &SettingCommand,
    opts: RowOptions,
) -> Option<&'static str> {
    let (stored, target) = match cmd.setting.as_str() {
        "master_volume" => (&mut cfg.audio.master_volume, AudioTarget::Master),
        "music_volume" => (&mut cfg.audio.music_volume, AudioTarget::Music),
        "sfx_volume" => (&mut cfg.audio.sfx_volume, AudioTarget::Sfx),
        "voice_volume" => (&mut cfg.audio.voice_volume, AudioTarget::Voice),
        other => {
            tracing::warn!("SettingsSystem: unknown volume setting '{other}'");
            return None;
        }
    };
    let cur = settings::volume_index(stored.unwrap_or(settings::DEFAULT_VOLUME));
    let next = settings::cycle(cur, opts.len(), cmd.op);
    let gain = settings::volume_at(next);
    *stored = Some(gain);
    ctx.events_mut::<AudioCommand>()
        .send(AudioCommand { target, gain });
    Some(opts[next])
}

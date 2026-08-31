// The per-frame SettingCommand drain: one arm per settings row. Cycles the
// value, applies it (live to the backend where the feature supports it,
// persist-only where a restart is required), refreshes the row's value label,
// and persists the change through the background writer. Moved verbatim from
// the GraphicsSystem frame step; the state fields keep their names.

use super::SettingsState;
use super::rows::{set_cached_row_label, set_label_content, set_rows_grayed, set_sprite_x};
use crate::components::{SettingCommand, SettingOp, WindowMode};
use crate::ecs::PipelineContext;
use crate::gfx::graphics_system as gsys;
use crate::gfx::ops::RenderOps;
use crate::gfx::settings;

impl SettingsState {
    pub(super) fn apply_setting_commands(
        &mut self,
        ctx: &mut PipelineContext,
        ops: &mut RenderOps,
    ) {
        // apply graphics settings changes UiInputSystem sent last tick:
        // cycle the setting, apply it to the backend, refresh the value
        // label, and persist the new value. Clone the commands out of
        // the queue so the ctx borrow is released before the loop body,
        // which needs &mut ctx (label/sprite updates, ControlsCommand /
        // AudioCommand sends).
        let setting_cmds: Vec<SettingCommand> = match ctx.events::<SettingCommand>() {
            Some(events) => events.read(&mut self.setting_cmd_cursor).cloned().collect(),
            None => Vec::new(),
        };
        // One settings snapshot serves the whole command batch: loaded
        // lazily (from the in-memory cache after the first change, so a
        // queued-but-unflushed background write is never re-read stale
        // from disk), mutated by the commands below, then queued once to
        // the background writer -- the render thread never blocks on
        // settings disk I/O.
        let mut cfg = self.settings_cache.take();
        let mut cfg_dirty = false;
        let tree = ctx
            .resource::<concinnity_host::store::paths::StateTree>()
            .cloned();
        for mut cmd in setting_cmds {
            let cfg = cfg.get_or_insert_with(|| crate::config::Settings::load(tree.as_ref()));
            // A Next/Prev on a slider row steps its value by a twentieth of
            // the range (a focused row's Left/Right), rewriting the op so the
            // SetFraction arm below applies + persists it. The handle's
            // position carries the current fraction: it is placed from the
            // persisted value at init and moved on every change since.
            if matches!(cmd.op, SettingOp::Next | SettingOp::Prev)
                && settings::slider_range(&cmd.setting).is_some()
            {
                let Some(cur) = self
                    .sliders
                    .iter()
                    .find(|s| s.key == cmd.setting)
                    .and_then(|s| {
                        let hx = ctx
                            .query::<crate::components::Sprite>()
                            .find(|sp| sp.asset_id == s.handle_id)
                            .map(|sp| sp.x)?;
                        let travel = (s.track_w - s.handle_w).max(f32::EPSILON);
                        Some(((hx - s.track_x) / travel).clamp(0.0, 1.0))
                    })
                else {
                    continue;
                };
                let step = match cmd.op {
                    SettingOp::Prev => -settings::SLIDER_STEP_FRACTION,
                    _ => settings::SLIDER_STEP_FRACTION,
                };
                cmd.op = SettingOp::SetFraction((cur + step).clamp(0.0, 1.0));
            }
            // InputKey-rebind settings (Controls tab) take a Rebind op: bind
            // the named action to the captured key, swapping with whatever
            // action held it, push the map to the backend, persist, and
            // refresh the affected row label(s). Handled first; the
            // slider + cycle settings below take SetFraction / Next / Prev.
            if let SettingOp::Rebind(key) = cmd.op {
                let Some(action) = crate::gfx::keymap::Bindable::from_setting_key(&cmd.setting)
                else {
                    tracing::warn!("GraphicsSystem: unknown rebind '{}'", cmd.setting);
                    continue;
                };
                // The action (if any) that currently holds the new key,
                // captured before the swap so its label is refreshed too.
                let victim = self.keymap.action_for_key(key).filter(|&a| a != action);
                self.keymap.rebind(action, key);
                let keymap = self.keymap;
                ops.record(move |backend| backend.set_keymap(&keymap));
                cfg.controls.keymap = Some(self.keymap);
                cfg_dirty = true;
                // Refresh the rebound row label and any swap victim's,
                // reading the registry by direct field access (disjoint
                // from the `backend` borrow).
                for act in [Some(action), victim].into_iter().flatten() {
                    if let Some(value_id) = self
                        .rebind_rows
                        .iter()
                        .find(|r| r.action == act)
                        .map(|r| r.value_id)
                    {
                        let name = self.keymap.get(act).display_name();
                        set_label_content(ctx, value_id, name);
                    }
                }
                continue;
            }
            // Gamepad rebind rows (`pad_*` settings) take a RebindButton op:
            // the same bind-with-swap flow as the key rebinds above, but the
            // live map travels to InputSystem as a ControlsCommand instead of
            // a backend push (the gamepad is polled engine-side).
            if let SettingOp::RebindButton(button) = cmd.op {
                let Some(action) = crate::components::GamepadAction::from_setting_key(&cmd.setting)
                else {
                    tracing::warn!("GraphicsSystem: unknown gamepad rebind '{}'", cmd.setting);
                    continue;
                };
                let victim = self
                    .gamepad_map
                    .action_for_button(button)
                    .filter(|&a| a != action);
                self.gamepad_map.rebind(action, button);
                ctx.events_mut::<crate::components::ControlsCommand>().send(
                    crate::components::ControlsCommand {
                        gamepad_map: Some(self.gamepad_map),
                        ..Default::default()
                    },
                );
                cfg.controls.gamepad_map = Some(self.gamepad_map);
                cfg_dirty = true;
                for act in [Some(action), victim].into_iter().flatten() {
                    if let Some(value_id) = self
                        .pad_rebind_rows
                        .iter()
                        .find(|r| r.action == act)
                        .map(|r| r.value_id)
                    {
                        let name = self.gamepad_map.get(act).display_name();
                        set_label_content(ctx, value_id, name);
                    }
                }
                continue;
            }
            // Slider settings (continuous) take a SetFraction op: apply
            // the value live to the post-process params, move the handle,
            // refresh the value label, and persist only on the commit
            // frame (drag release). Handled here; the cycle settings
            // below take Next/Prev.
            if let SettingOp::SetFraction(frac) = cmd.op {
                let Some(value) = settings::slider_value_at(&cmd.setting, frac) else {
                    tracing::warn!("GraphicsSystem: unknown slider '{}'", cmd.setting);
                    continue;
                };
                // Track geometry for the handle, copied out so the
                // `self.sliders` borrow ends before mutating self below.
                let geom = self
                    .sliders
                    .iter()
                    .find(|s| s.key == cmd.setting)
                    .map(|s| (s.handle_id, s.track_x, s.track_w, s.handle_w));
                // Apply the value to the live render param. The clamp /
                // EV-to-multiplier transform lives in
                // `settings::slider_apply_value` (shared with the
                // persisted re-apply at init, so they cannot diverge).
                let stored = settings::slider_apply_value(&cmd.setting, value);
                let is_qparam = settings::is_quality_param_slider(&cmd.setting);
                match cmd.setting.as_str() {
                    "exposure" => self.post_process.exposure = stored,
                    "bloom_intensity" => self.post_process.bloom_intensity = stored,
                    "bloom_threshold" => self.post_process.bloom_threshold = stored,
                    "bloom_knee" => self.post_process.bloom_knee = stored,
                    "vignette" => self.post_process.vignette = stored,
                    "lut_strength" => self.post_process.lut_strength = stored,
                    // Per-feature sub-quality sliders live on the stored
                    // PostProcessConfig (the source of truth a later rebuild
                    // re-derives from); the live apply below mutates the
                    // backend's stored settings without a rebuild.
                    "ssao_radius" => self.post_config.ssao_radius = stored,
                    "ssao_intensity" => self.post_config.ssao_intensity = stored,
                    "ssr_intensity" => self.post_config.ssr_intensity = stored,
                    "ssr_max_distance" => self.post_config.ssr_max_distance = stored,
                    "ssgi_intensity" => self.post_config.ssgi_intensity = stored,
                    "ssgi_max_distance" => self.post_config.ssgi_max_distance = stored,
                    "auto_exposure_min_ev" => self.post_config.auto_exposure_min_ev = stored,
                    "auto_exposure_max_ev" => self.post_config.auto_exposure_max_ev = stored,
                    "auto_exposure_speed" => self.post_config.auto_exposure_speed = stored,
                    _ => {}
                }
                // Apply live. The sub-quality sliders mutate the backend's
                // stored *Settings via update_quality_params (re-read into a
                // per-frame uniform, no pass rebuild). The post-process
                // sliders push PostProcessParams. The controls sliders are not
                // render params, so they skip both (handled below); the
                // ambient re-push through update_post_process is harmless.
                if is_qparam {
                    let quality = gsys::derive_quality_settings(&self.post_config);
                    ops.record(move |backend| backend.update_quality_params(quality));
                } else if !settings::is_controls_slider(&cmd.setting) {
                    {
                        let params = self.post_process;
                        ops.record(move |backend| backend.update_post_process(params));
                    }
                }
                // Ambient (IBL) scale lives in LightUniforms, not
                // PostProcessParams, so it takes a dedicated setter.
                if cmd.setting == "ambient_intensity" {
                    self.ambient_intensity = stored;
                    ops.record(move |backend| backend.set_ambient_intensity(stored));
                }
                // The controls sliders take effect on the camera / input
                // sampling, not the renderer: hand the new value across as a
                // ControlsCommand read this same tick (live, no restart).
                // Each carries only the field it changed.
                let controls_cmd = match cmd.setting.as_str() {
                    "mouse_sensitivity" => Some(crate::components::ControlsCommand {
                        mouse_sensitivity: Some(stored),
                        ..Default::default()
                    }),
                    "fov" => Some(crate::components::ControlsCommand {
                        fov_y_degrees: Some(stored),
                        ..Default::default()
                    }),
                    "gamepad_look_sensitivity" => Some(crate::components::ControlsCommand {
                        gamepad_look_sensitivity: Some(stored),
                        ..Default::default()
                    }),
                    "gamepad_deadzone" => Some(crate::components::ControlsCommand {
                        gamepad_deadzone: Some(stored),
                        ..Default::default()
                    }),
                    _ => None,
                };
                if let Some(controls_cmd) = controls_cmd {
                    ctx.events_mut::<crate::components::ControlsCommand>()
                        .send(controls_cmd);
                }
                // Move the handle to the new fraction.
                if let Some((handle_id, track_x, track_w, handle_w)) = geom {
                    let hx = track_x + frac.clamp(0.0, 1.0) * (track_w - handle_w).max(0.0);
                    set_sprite_x(ctx, handle_id, hx);
                }
                // Refresh the value label.
                if let Some(label_id) = cmd.value_label {
                    set_label_content(
                        ctx,
                        label_id,
                        &settings::format_slider_value(&cmd.setting, value),
                    );
                }
                // Persist only on release (the in-progress frames apply
                // live but skip the disk write).
                if cmd.persist {
                    match cmd.setting.as_str() {
                        "exposure" => cfg.graphics.exposure_ev = Some(value),
                        "bloom_intensity" => cfg.graphics.bloom_intensity = Some(value),
                        "bloom_threshold" => cfg.graphics.bloom_threshold = Some(value),
                        "bloom_knee" => cfg.graphics.bloom_knee = Some(value),
                        "vignette" => cfg.graphics.vignette = Some(value),
                        "lut_strength" => cfg.graphics.lut_strength = Some(value),
                        "ambient_intensity" => cfg.graphics.ambient_intensity = Some(value),
                        "ssao_radius" => cfg.graphics.ssao_radius = Some(value),
                        "ssao_intensity" => cfg.graphics.ssao_intensity = Some(value),
                        "ssr_intensity" => cfg.graphics.ssr_intensity = Some(value),
                        "ssr_max_distance" => cfg.graphics.ssr_max_distance = Some(value),
                        "ssgi_intensity" => cfg.graphics.ssgi_intensity = Some(value),
                        "ssgi_max_distance" => cfg.graphics.ssgi_max_distance = Some(value),
                        "auto_exposure_min_ev" => cfg.graphics.auto_exposure_min_ev = Some(value),
                        "auto_exposure_max_ev" => cfg.graphics.auto_exposure_max_ev = Some(value),
                        "auto_exposure_speed" => cfg.graphics.auto_exposure_speed = Some(value),
                        // Persist the applied values (what the camera / input
                        // sampling read), not the 1..100 UI values.
                        "mouse_sensitivity" => cfg.controls.mouse_sensitivity = Some(stored),
                        "gamepad_look_sensitivity" => {
                            cfg.controls.gamepad_look_sensitivity = Some(stored)
                        }
                        "gamepad_deadzone" => cfg.controls.gamepad_deadzone = Some(stored),
                        // FOV persists the clamped degrees (a graphics
                        // preference, stored alongside the look sliders).
                        "fov" => cfg.graphics.fov = Some(stored),
                        _ => {}
                    }
                    cfg_dirty = true;
                }
                continue;
            }
            // Master "Graphics Quality" preset row. A preset is a
            // performance ceiling over the world's authored look (it never
            // enables a feature the world did not author), so picking a
            // tier / Auto clears the per-row quality overrides and
            // re-derives the toggles + render scale from the world's
            // authored config under the new ceiling: raising a preset
            // restores the world's features, lowering it clamps them off.
            // Custom resolves to the no-op ceiling (the world's look).
            // Render scale is restart-required, so it only persists +
            // relabels here. See gfx/quality_preset.rs.
            if cmd.setting == "graphics_quality" {
                use crate::gfx::quality_preset;
                let opts = settings::options("graphics_quality").unwrap_or(&[]);
                let cur = quality_preset::preset_index(self.quality_preset);
                let next = settings::cycle(cur, opts.len(), cmd.op);
                let preset = quality_preset::preset_at(next);
                self.quality_preset = preset;
                let ceiling = quality_preset::resolve_ceiling(preset, &self.gpu_profile);

                // Re-derive the live quality toggles from the world
                // baseline under the new ceiling (force off where
                // disallowed; never turn on).
                self.post_config = self.authored_post_config.clone();
                for (key, allowed) in [
                    ("ssao", ceiling.ssao),
                    ("ssr", ceiling.ssr),
                    ("ray_traced_reflections", ceiling.ray_traced_reflections),
                    ("ssgi", ceiling.ssgi),
                    ("auto_exposure", ceiling.auto_exposure),
                ] {
                    if !allowed {
                        gsys::set_quality_toggle(&mut self.post_config, key, false);
                    }
                }
                // And clamp the cycle quality knobs (AA mode + SSGI gather
                // + reflection blur) under the ceiling (overrides cleared,
                // so clamp every one).
                for key in settings::QUALITY_CYCLE_KEYS {
                    gsys::clamp_quality_cycle(&mut self.post_config, key, &ceiling, false);
                }
                // The composite FXAA flag rides PostProcessParams (pushed
                // by update_post_process below), so refresh it from the
                // re-derived AA mode before that push.
                self.post_process.fxaa = self.post_config.aa_mode.fxaa_flag();
                let quality = gsys::derive_quality_settings(&self.post_config);
                ops.record(move |backend| backend.apply_quality_settings(quality));
                // Auto-exposure may have flipped off; re-push the static
                // post-process params so exposure reverts (mirrors the
                // quality-toggle arm below).
                {
                    let params = self.post_process;
                    ops.record(move |backend| backend.update_post_process(params));
                }
                // Restart-required: update the live render scale for the
                // row label only (the upscaler + targets are sized at init,
                // so it takes effect at the next launch).
                self.render_scale = quality_preset::more_aggressive_upscale(
                    self.authored_post_config.upscale_quality,
                    ceiling.min_upscale,
                );
                // Re-derive the shadow knobs from the authored baselines
                // under the new ceiling. The cadence is live (the scheduler
                // reads it each frame); the resolution is restart-required,
                // so it only updates the row label below.
                self.shadow_map_size = self.authored_shadow_map_size.min(ceiling.shadow_map_size);
                self.shadow_update =
                    quality_preset::clamp_shadow_update(self.authored_shadow_update, &ceiling);
                {
                    let update = self.shadow_update;
                    ops.record(move |backend| backend.set_shadow_update(update));
                }
                // Shadow distance: live (the cascade-split math reads it
                // each frame), so re-derive from the authored baseline and
                // push it to the backend.
                self.shadow_distance = self.authored_shadow_distance.min(ceiling.shadow_distance);
                {
                    let distance = self.shadow_distance;
                    ops.record(move |backend| backend.set_shadow_distance(distance));
                }
                // Shadow cascade count: live (the per-frame split + schedule
                // read it), so re-derive from the authored baseline and push.
                self.shadow_cascades = self.authored_shadow_cascades.min(ceiling.shadow_cascades);
                {
                    let count = self.shadow_cascades;
                    ops.record(move |backend| backend.set_shadow_cascades(count));
                }
                // Anisotropy: restart-required, so re-derive from the
                // authored baseline for the row label only (the sampler is
                // built at init; the new degree takes effect next launch).
                self.anisotropy = self.authored_anisotropy.min(ceiling.anisotropy);

                // Persist the preset and drop the per-row quality overrides,
                // so the next launch re-resolves them from the world +
                // ceiling exactly as this live re-derive did.
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
                cfg_dirty = true;

                // Refresh the dependent rows (quality toggles + render
                // scale) from the init-captured value-label ids -- the
                // menu's HitRegions are drained by UiInputSystem after
                // init, so they cannot be re-queried here.
                for key in settings::QUALITY_TOGGLE_KEYS {
                    let on = gsys::quality_toggle_on(&self.post_config, key).unwrap_or(false);
                    if let Some(text) =
                        settings::options(key).and_then(|o| o.get(on as usize).copied())
                    {
                        set_cached_row_label(&self.cycle_value_labels, ctx, key, text);
                    }
                }
                if let Some(text) = settings::options("render_scale").and_then(|o| {
                    o.get(settings::render_scale_index(self.render_scale))
                        .copied()
                }) {
                    set_cached_row_label(&self.cycle_value_labels, ctx, "render_scale", text);
                }
                for key in settings::QUALITY_CYCLE_KEYS {
                    if let Some(text) = gsys::quality_cycle_index(&self.post_config, key)
                        .and_then(|idx| settings::options(key).and_then(|o| o.get(idx).copied()))
                    {
                        set_cached_row_label(&self.cycle_value_labels, ctx, key, text);
                    }
                }
                // And the shadow + anisotropy rows (their state lives on
                // self, not the post_config, so they relabel from the live
                // fields).
                for key in [
                    "shadow_map_size",
                    "shadow_update",
                    "shadow_distance",
                    "shadow_cascades",
                    "anisotropy",
                ] {
                    let idx = match key {
                        "shadow_map_size" => {
                            settings::shadow_resolution_index(self.shadow_map_size)
                        }
                        "shadow_distance" => settings::shadow_distance_index(self.shadow_distance),
                        "shadow_cascades" => settings::shadow_cascades_index(self.shadow_cascades),
                        "anisotropy" => settings::anisotropy_index(self.anisotropy),
                        _ => settings::shadow_update_index(self.shadow_update),
                    };
                    if let Some(text) = settings::options(key).and_then(|o| o.get(idx).copied()) {
                        set_cached_row_label(&self.cycle_value_labels, ctx, key, text);
                    }
                }
                // The master row's own label carries the Auto(tier) suffix
                // and is updated through the event-carried value-label id.
                let label = quality_preset::preset_label(preset, &self.gpu_profile);
                if let Some(id) = cmd.value_label {
                    set_label_content(ctx, id, &label);
                }
                continue;
            }
            // The Resolution row: a dynamic dropdown whose option list is
            // the enumerated display-mode list, not the static registry.
            // Fullscreen-only: the backend holds the display to the
            // chosen mode while the window is fullscreen (restoring the
            // desktop mode on exit); in the other modes the row is
            // grayed + inert (a windowed size comes from the window
            // itself, borderless covers the display), so the window is
            // never resized here. Independent of the quality preset (a
            // user/hardware preference, like window mode).
            if cmd.setting == "resolution" {
                if self.display_modes.is_empty() {
                    continue;
                }
                // The chosen mode, else the display's own (read inline,
                // not via effective_resolution, so the borrow stays
                // field-local alongside the live backend).
                let effective = self.resolution.or(self.current_mode).unwrap_or(
                    crate::gfx::display_mode::DisplayMode {
                        width: self.window_args.width,
                        height: self.window_args.height,
                        refresh_hz: 0,
                    },
                );
                let cur = crate::gfx::display_mode::index_of(&self.display_modes, effective);
                let next = settings::cycle(cur, self.display_modes.len(), cmd.op);
                let mode = self.display_modes[next];
                self.resolution = Some(mode);
                ops.record(move |backend| backend.set_display_mode(mode));
                cfg.graphics.resolution = Some([mode.width, mode.height, mode.refresh_hz]);
                cfg_dirty = true;
                if let Some(label_id) = cmd.value_label {
                    set_label_content(ctx, label_id, &mode.label());
                }
                continue;
            }
            let Some(opts) = settings::options(&cmd.setting) else {
                tracing::warn!("GraphicsSystem: unknown setting '{}'", cmd.setting);
                continue;
            };
            // Apply per setting: cycle the value, apply it (live for
            // window/vsync; render_scale is restart-required so it only
            // persists), then persist and refresh the value label.
            let new_text: Option<&str> = match cmd.setting.as_str() {
                "vsync" => {
                    let next = settings::cycle(self.vsync as usize, opts.len(), cmd.op);
                    self.vsync = next == 1;
                    {
                        let on = self.vsync;
                        ops.record(move |backend| backend.set_vsync(on));
                    }
                    cfg.graphics.vsync = Some(self.vsync);
                    Some(opts[next])
                }
                // Stats-HUD display toggles: live via the HudPrefs resource
                // (published each frame from these fields). The master
                // additionally grays / restores the two sub-rows.
                "perf_stats" => {
                    let next = settings::cycle(self.perf_stats as usize, opts.len(), cmd.op);
                    self.perf_stats = next == 1;
                    cfg.graphics.perf_stats = Some(self.perf_stats);
                    set_rows_grayed(ctx, &self.perf_sub_row_labels, !self.perf_stats);
                    Some(opts[next])
                }
                "show_fps" => {
                    let next = settings::cycle(self.show_fps as usize, opts.len(), cmd.op);
                    self.show_fps = next == 1;
                    cfg.graphics.show_fps = Some(self.show_fps);
                    Some(opts[next])
                }
                "show_vram" => {
                    let next = settings::cycle(self.show_vram as usize, opts.len(), cmd.op);
                    self.show_vram = next == 1;
                    cfg.graphics.show_vram = Some(self.show_vram);
                    Some(opts[next])
                }
                "fps_cap" => {
                    let cur = settings::fps_cap_index(self.fps_cap);
                    let next = settings::cycle(cur, opts.len(), cmd.op);
                    self.fps_cap = settings::fps_cap_at(next);
                    // Live with no backend call: the App-level pacer reads
                    // the republished cap before the next step, and the
                    // value change re-bases its running deadline so
                    // changing the cap never leaves one stale long wait.
                    // Independent of the preset (no Custom flip).
                    ctx.insert_resource(crate::ecs::FrameRateCap(self.fps_cap));
                    cfg.graphics.fps_cap = Some(self.fps_cap);
                    Some(opts[next])
                }
                "window_mode" => {
                    let cur = settings::window_mode_index(self.window_args.mode);
                    let next = settings::cycle(cur, opts.len(), cmd.op);
                    let mode = settings::window_mode_at(next);
                    self.window_args.mode = mode;
                    ops.record(move |backend| backend.set_window_mode(mode));
                    // Returning to windowed: re-apply the remembered
                    // windowed size, since borderless/fullscreen left the
                    // window at the display size (no-op while fullscreen
                    // is still animating; each backend guards that).
                    if mode == WindowMode::Windowed {
                        {
                            let (w, h) = (self.window_args.width, self.window_args.height);
                            ops.record(move |backend| backend.set_window_size(w, h));
                        }
                    }
                    // The Resolution row only applies in fullscreen, so
                    // the new mode grays it out or restores it (the
                    // matching inertness rides the DisabledSettingRows
                    // publish below).
                    set_rows_grayed(
                        ctx,
                        &self.resolution_row_labels,
                        mode != WindowMode::Fullscreen,
                    );
                    cfg.graphics.window_mode = Some(mode);
                    Some(opts[next])
                }
                "render_scale" => {
                    // Restart-required: persist + display only; the
                    // upscaler and render targets are sized once at init.
                    let cur = settings::render_scale_index(self.render_scale);
                    let next = settings::cycle(cur, opts.len(), cmd.op);
                    self.render_scale = settings::render_scale_at(next);
                    cfg.graphics.render_scale = Some(self.render_scale);
                    self.opt_out_of_preset(ctx, cfg);
                    Some(opts[next])
                }
                "upscale_backend" => {
                    // Restart-required: persist + display only; the
                    // upscaler is selected + built once at init. Independent
                    // of the quality preset (a user / hardware preference,
                    // not a tier), so no Custom-flip and no live backend
                    // call. The cycle skips upscalers this GPU vendor does
                    // not offer (DLSS NVIDIA-only, XeSS Intel-only); Auto /
                    // FSR3 are always available, so the loop terminates.
                    let mut next = settings::cycle(
                        settings::upscale_backend_index(self.upscale_backend),
                        opts.len(),
                        cmd.op,
                    );
                    while !settings::upscale_backend_available(
                        settings::upscale_backend_at(next),
                        self.gpu_profile.vendor,
                    ) {
                        next = settings::cycle(next, opts.len(), cmd.op);
                    }
                    self.upscale_backend = settings::upscale_backend_at(next);
                    cfg.graphics.upscale_backend = Some(self.upscale_backend);
                    Some(opts[next])
                }
                "master_volume" | "music_volume" | "sfx_volume" | "voice_volume" => {
                    // Live: cycle the gain, persist it, and hand it to
                    // AudioSystem (which owns the audio engine) as an
                    // AudioCommand it drains this same tick -- GraphicsSystem
                    // runs first, so the change applies this frame. A world
                    // with no audio simply has no AudioSystem to drain it;
                    // the persisted value then applies at the next audio init.
                    let (stored, target) = match cmd.setting.as_str() {
                        "master_volume" => (
                            &mut cfg.audio.master_volume,
                            crate::components::AudioTarget::Master,
                        ),
                        "music_volume" => (
                            &mut cfg.audio.music_volume,
                            crate::components::AudioTarget::Music,
                        ),
                        "sfx_volume" => (
                            &mut cfg.audio.sfx_volume,
                            crate::components::AudioTarget::Sfx,
                        ),
                        _ => (
                            &mut cfg.audio.voice_volume,
                            crate::components::AudioTarget::Voice,
                        ),
                    };
                    let cur = settings::volume_index(stored.unwrap_or(settings::DEFAULT_VOLUME));
                    let next = settings::cycle(cur, opts.len(), cmd.op);
                    let gain = settings::volume_at(next);
                    *stored = Some(gain);
                    ctx.events_mut::<crate::components::AudioCommand>()
                        .send(crate::components::AudioCommand { target, gain });
                    Some(opts[next])
                }
                // Quality-feature toggles: flip the matching field on the
                // stored config, persist the bool, then apply live by
                // rebuilding the affected render resources (Metal; a
                // no-op backend keeps the choice for the next launch).
                key if settings::is_quality_toggle(key) => {
                    let cur = gsys::quality_toggle_on(&self.post_config, key).unwrap_or(false);
                    let next = settings::cycle(cur as usize, opts.len(), cmd.op);
                    let on = next == 1;
                    gsys::set_quality_toggle(&mut self.post_config, key, on);
                    match key {
                        "ssao" => cfg.graphics.ssao = Some(on),
                        "ssr" => cfg.graphics.ssr = Some(on),
                        "ray_traced_reflections" => cfg.graphics.ray_traced_reflections = Some(on),
                        "ssgi" => cfg.graphics.ssgi = Some(on),
                        "auto_exposure" => cfg.graphics.auto_exposure = Some(on),
                        _ => {}
                    }
                    self.opt_out_of_preset(ctx, cfg);
                    let quality = gsys::derive_quality_settings(&self.post_config);
                    ops.record(move |backend| backend.apply_quality_settings(quality));
                    // Auto-exposure overwrites the backend's live exposure
                    // each frame while it runs; once it is toggled off, the
                    // backend's copy is frozen at the last adapted value.
                    // Re-push the static post-process params (this side's
                    // `post_process.exposure` is the authored / slider EV,
                    // untouched by auto-exposure) so exposure reverts. A
                    // toggle-on is harmless: the AE loop overwrites it next
                    // frame.
                    if key == "auto_exposure" {
                        {
                            let params = self.post_process;
                            ops.record(move |backend| backend.update_post_process(params));
                        }
                    }
                    Some(opts[next])
                }
                // Cycle quality knobs (SSGI gather sub-quality dropdowns):
                // cycle the value on the stored config, persist it, flip
                // the preset to Custom, then rebuild the affected effect
                // live (Metal; a no-op backend keeps the choice for the
                // next launch). Rides the same apply_quality_settings path
                // as the toggles -- the sub-tunable travels in the feature's
                // settings payload, so no new backend method is needed.
                key if gsys::is_quality_cycle(key) => {
                    let cur = gsys::quality_cycle_index(&self.post_config, key).unwrap_or(0);
                    let next = settings::cycle(cur, opts.len(), cmd.op);
                    gsys::set_quality_cycle(&mut self.post_config, key, next);
                    match key {
                        "aa_mode" => cfg.graphics.aa_mode = Some(self.post_config.aa_mode),
                        "ssgi_resolution" => {
                            cfg.graphics.ssgi_resolution = Some(self.post_config.ssgi_resolution)
                        }
                        "ssgi_rays" => cfg.graphics.ssgi_rays = Some(self.post_config.ssgi_rays),
                        "ssgi_steps" => cfg.graphics.ssgi_steps = Some(self.post_config.ssgi_steps),
                        "reflection_blur_resolution" => {
                            cfg.graphics.reflection_blur_resolution =
                                Some(self.post_config.reflection_blur_resolution)
                        }
                        _ => {}
                    }
                    self.opt_out_of_preset(ctx, cfg);
                    let quality = gsys::derive_quality_settings(&self.post_config);
                    ops.record(move |backend| backend.apply_quality_settings(quality));
                    // The AA mode also drives the composite FXAA flag,
                    // which rides PostProcessParams rather than the
                    // QualitySettings rebuild above. Refresh it and push it
                    // live (the TAA pass itself rebuilt via the call above).
                    if key == "aa_mode" {
                        self.post_process.fxaa = self.post_config.aa_mode.fxaa_flag();
                        {
                            let params = self.post_process;
                            ops.record(move |backend| backend.update_post_process(params));
                        }
                    }
                    Some(opts[next])
                }
                // Display-output / upscaling preference toggles. Restart-
                // required: persist + display only (the swapchain format /
                // render targets are sized once at init, so it applies at
                // the next launch). Independent of the quality preset, so
                // no Custom-flip and no live backend call.
                key @ ("temporal_upscaling" | "hdr_display" | "hdr_pq") => {
                    let cur = match key {
                        "temporal_upscaling" => self.temporal_upscaling,
                        "hdr_display" => self.hdr_display,
                        _ => self.hdr_pq,
                    };
                    let next = settings::cycle(cur as usize, opts.len(), cmd.op);
                    let on = next == 1;
                    match key {
                        "temporal_upscaling" => {
                            self.temporal_upscaling = on;
                            cfg.graphics.temporal_upscaling = Some(on);
                        }
                        "hdr_display" => {
                            self.hdr_display = on;
                            cfg.graphics.hdr_display = Some(on);
                        }
                        _ => {
                            self.hdr_pq = on;
                            cfg.graphics.hdr_pq = Some(on);
                        }
                    }
                    Some(opts[next])
                }
                // Shadow resolution: restart-required (the shadow map array
                // is sized once at init), so persist + display only; the
                // new size takes effect at the next launch.
                "shadow_map_size" => {
                    let cur = settings::shadow_resolution_index(self.shadow_map_size);
                    let next = settings::cycle(cur, opts.len(), cmd.op);
                    self.shadow_map_size = settings::shadow_resolution_at(next);
                    cfg.graphics.shadow_map_size = Some(self.shadow_map_size);
                    self.opt_out_of_preset(ctx, cfg);
                    Some(opts[next])
                }
                // Anisotropic filtering: restart-required (the scene
                // sampler is built once at init), so persist + display only;
                // the new degree takes effect at the next launch.
                "anisotropy" => {
                    let cur = settings::anisotropy_index(self.anisotropy);
                    let next = settings::cycle(cur, opts.len(), cmd.op);
                    self.anisotropy = settings::anisotropy_at(next);
                    cfg.graphics.anisotropy = Some(self.anisotropy);
                    self.opt_out_of_preset(ctx, cfg);
                    Some(opts[next])
                }
                // Shadow re-render cadence: live -- the cascade scheduler
                // reads the policy each frame, so it applies on the next draw.
                "shadow_update" => {
                    let cur = settings::shadow_update_index(self.shadow_update);
                    let next = settings::cycle(cur, opts.len(), cmd.op);
                    self.shadow_update = settings::shadow_update_at(next);
                    {
                        let update = self.shadow_update;
                        ops.record(move |backend| backend.set_shadow_update(update));
                    }
                    cfg.graphics.shadow_update = Some(self.shadow_update);
                    self.opt_out_of_preset(ctx, cfg);
                    Some(opts[next])
                }
                // Shadow distance: live -- the per-frame cascade-split math
                // reads it, so it applies on the next draw.
                "shadow_distance" => {
                    let cur = settings::shadow_distance_index(self.shadow_distance);
                    let next = settings::cycle(cur, opts.len(), cmd.op);
                    self.shadow_distance = settings::shadow_distance_at(next);
                    {
                        let distance = self.shadow_distance;
                        ops.record(move |backend| backend.set_shadow_distance(distance));
                    }
                    cfg.graphics.shadow_distance = Some(self.shadow_distance);
                    self.opt_out_of_preset(ctx, cfg);
                    Some(opts[next])
                }
                // Shadow cascade count: live -- the per-frame split + schedule
                // read it, so it applies on the next draw.
                "shadow_cascades" => {
                    let cur = settings::shadow_cascades_index(self.shadow_cascades);
                    let next = settings::cycle(cur, opts.len(), cmd.op);
                    self.shadow_cascades = settings::shadow_cascades_at(next);
                    {
                        let count = self.shadow_cascades;
                        ops.record(move |backend| backend.set_shadow_cascades(count));
                    }
                    cfg.graphics.shadow_cascades = Some(self.shadow_cascades);
                    self.opt_out_of_preset(ctx, cfg);
                    Some(opts[next])
                }
                // System / streaming restart rows. Restart-required (the
                // ring buffers / cull pipeline / streaming pool are sized
                // once at init), so persist + display only; independent of
                // the quality preset, so no Custom-flip and no live call.
                "frames_in_flight" => {
                    let cur = settings::frames_in_flight_index(self.frames_in_flight as u32);
                    let next = settings::cycle(cur, opts.len(), cmd.op);
                    self.frames_in_flight = settings::frames_in_flight_at(next) as usize;
                    cfg.graphics.frames_in_flight = Some(self.frames_in_flight as u32);
                    Some(opts[next])
                }
                "occlusion_two_pass" => {
                    let next =
                        settings::cycle(self.occlusion_two_pass as usize, opts.len(), cmd.op);
                    self.occlusion_two_pass = next == 1;
                    cfg.graphics.occlusion_two_pass = Some(self.occlusion_two_pass);
                    Some(opts[next])
                }
                // One row drives both the streaming pool cap and the
                // per-frame upload budget.
                "texture_quality" => {
                    let cur = settings::texture_quality_index(self.texture_cap);
                    let next = settings::cycle(cur, opts.len(), cmd.op);
                    let (cap, budget) = settings::texture_quality_at(next);
                    self.texture_cap = cap;
                    self.texture_budget = budget;
                    cfg.graphics.texture_cap = Some(cap);
                    cfg.graphics.texture_budget = Some(budget);
                    Some(opts[next])
                }
                _ => None,
            };
            if let Some(text) = new_text {
                cfg_dirty = true;
                if let Some(label_id) = cmd.value_label {
                    set_label_content(ctx, label_id, text);
                }
            }
        }
        // Hand the batch's snapshot to the background writer (spawned on
        // the first persisted change) and keep it as the cache the next
        // change starts from. Field access, not a &mut self helper: the
        // `backend` borrow above lives to the end of this scope.
        if let Some(cfg) = cfg {
            if cfg_dirty {
                self.settings_writer
                    .get_or_insert_with(|| super::writer::SettingsWriter::spawn(tree.clone()))
                    .save(cfg.clone());
            }
            self.settings_cache = Some(cfg);
        }
    }

    // An explicit per-row quality change opts the master preset out to Custom
    // (no ceiling clamps the user's choice) and relabels the master row.
    fn opt_out_of_preset(&mut self, ctx: &mut PipelineContext, cfg: &mut crate::config::Settings) {
        self.quality_preset = crate::gfx::quality_preset::QualityPreset::Custom;
        cfg.graphics.quality_preset = Some(self.quality_preset);
        set_cached_row_label(
            &self.cycle_value_labels,
            ctx,
            "graphics_quality",
            self.quality_preset.name(),
        );
    }
}

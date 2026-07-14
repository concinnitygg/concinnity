// GraphicsSystem one-time setup: backend creation, draw-list build, and the
// shader / texture / streaming wiring performed on the first tick.

use crate::assets::{
    BlockType, Camera3D, Decal, DirectionalLight, GlassPanel, GraphicsConfig, HitRegion, Material,
    Model, ParticleEmitter, PointLight, PostProcessConfig, PostProcessResolve, SdfVolume,
    ShaderKind, ShaderStage, ShaderStageExt, SkinnedMeshGeometry, StreamingConfig, TextLabel,
    VolumetricFog, VoxelWorld, WaterSurface, Window,
};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::{
    draw_list::{self, MaterialEntry},
    lights, skinning, text,
};
use std::time::Instant;

use super::helpers::*;
use super::*;

impl GraphicsSystem {
    pub(super) fn run_init(&mut self, ctx: &mut PipelineContext) {
        // Persisted settings-menu choices override the world's authored defaults
        // below (each field is None when the user never changed that setting).
        let user_graphics = self.persisted_settings().graphics;

        // Detect the GPU before the backend is built so the auto-config quality
        // ceiling can influence the render targets / effect pipelines sized at
        // backend init. Held on self for later (e.g. the menu's preset label).
        self.gpu_profile = self.detect_gpu_profile();
        // Resolve the master quality preset. An ephemeral `CN_QUALITY_PRESET`
        // env override wins first and is never persisted, so a test / CI / GPU
        // smoke can force a preset (e.g. `custom` for no clamp) without touching
        // settings.bin. Otherwise the persisted choice; `None` there = never
        // configured (a first launch, or a settings file written before the
        // preset existed): seed `Auto` and persist once, which records the
        // detection without baking any per-field value (the per-field overrides
        // keep their `None = world default` meaning). `Auto` re-resolves from the
        // detected tier each launch; `Custom` / an unclassified GPU impose no
        // ceiling.
        use crate::gfx::quality_preset::QualityPreset;
        let env_preset = std::env::var("CN_QUALITY_PRESET")
            .ok()
            .and_then(|s| QualityPreset::parse(&s));
        let active_preset = env_preset
            .or(user_graphics.quality_preset)
            .unwrap_or_else(|| {
                self.seed_first_launch_preset();
                QualityPreset::Auto
            });
        // Hold the resolved preset as the live value the settings-menu master
        // row cycles (and that an individual quality-row change flips to Custom).
        self.quality_preset = active_preset;
        let quality_ceiling =
            crate::gfx::quality_preset::resolve_ceiling(active_preset, &self.gpu_profile);
        tracing::info!(
            "auto-config: GPU tier {:?}, quality preset {:?}",
            self.gpu_profile.tier,
            active_preset,
        );

        if let Some(w) = ctx.drain::<Window>().into_iter().next() {
            self.window_args = w;
        }
        // Capture the DebugHud chip ids (cursor, camera, sys, passes stack
        // order) so the frame step can anchor them to the top-right of the
        // window. Passes is last because it grows/shrinks with the frame's step
        // count, so keeping it at the bottom leaves the fixed-height chips
        // unshifted. The DebugHud component is queried (not drained) by its
        // system, so it is still present here; absent fields are skipped.
        self.debug_hud_chips = ctx
            .query::<crate::assets::DebugHud>()
            .next()
            .map(|d| {
                [d.mouse_label, d.camera_label, d.sys_label, d.passes_label]
                    .into_iter()
                    .flatten()
                    .collect()
            })
            .unwrap_or_default();
        // Capture the StatHud chip ids (fps, vram, ram, ev, edr strip order) so
        // the frame step can pack them tight from the top-left. Like DebugHud
        // the component is queried (not drained), so it is still present here.
        self.stat_hud_chips = ctx
            .query::<crate::assets::StatHud>()
            .next()
            .map(|s| {
                [
                    s.fps_label,
                    s.vram_label,
                    s.ram_label,
                    s.ev_label,
                    s.edr_label,
                ]
                .into_iter()
                .flatten()
                .collect()
            })
            .unwrap_or_default();
        if let Some(m) = user_graphics.window_mode {
            self.window_args.mode = m;
        }
        // The chosen fullscreen display mode. Fullscreen-only: it never feeds
        // the windowed size, which stays the world's authored `Window` value.
        if let Some([w, h, hz]) = user_graphics.resolution {
            self.resolution = Some(crate::gfx::display_mode::DisplayMode {
                width: w,
                height: h,
                refresh_hz: hz,
            });
        }

        if let Some(c) = ctx.drain::<GraphicsConfig>().into_iter().next() {
            let args = c;
            self.frames_in_flight = args.frames_in_flight as usize;
            self.vsync = args.vsync;
            self.fps_cap = args.fps_cap;
            self.clear_color = args.clear_color;
            self.max_frames = args.max_frames;
            self.shadow_map_size = args.shadow_map_size;
            self.shadow_update = args.shadow_update;
            self.shadow_distance = args.shadow_distance;
            self.shadow_cascades = args.shadow_cascades;
            self.anisotropy = args.anisotropy;
        }
        // A persisted vsync choice overrides the world's value. Applied outside
        // the GraphicsConfig block (unconditional), matching window_mode /
        // resolution, so it wins over both the authored value and the default.
        if let Some(v) = user_graphics.vsync {
            self.vsync = v;
        }
        // A persisted frame-rate cap overrides the world's value (0 = unlimited),
        // applied live by the render-step pacer. Independent of the quality preset,
        // like vsync, so no ceiling clamp.
        if let Some(v) = user_graphics.fps_cap {
            self.fps_cap = v;
        }
        // Stats-HUD display toggles (None = shown, the default, so an existing
        // settings file keeps the FPS / VRAM chips visible). Independent of the
        // quality preset, like vsync / fps_cap.
        if let Some(v) = user_graphics.perf_stats {
            self.perf_stats = v;
        }
        if let Some(v) = user_graphics.show_fps {
            self.show_fps = v;
        }
        if let Some(v) = user_graphics.show_vram {
            self.show_vram = v;
        }

        // Shadow quality knobs (GraphicsConfig-sourced). Snapshot the world's
        // authored values as the baseline a live preset change re-clamps from,
        // then apply any persisted override and otherwise clamp under the preset
        // ceiling (an explicit override wins, like the quality toggles below). The
        // resolution is restart-required -- the shadow map array is sized from
        // `self.shadow_map_size` at backend init below -- while the cadence is read
        // by the cascade scheduler each frame.
        self.authored_shadow_map_size = self.shadow_map_size;
        self.authored_shadow_update = self.shadow_update;
        match user_graphics.shadow_map_size {
            Some(v) => self.shadow_map_size = v,
            None => {
                self.shadow_map_size = crate::gfx::quality_preset::clamp_shadow_map_size(
                    self.shadow_map_size,
                    &quality_ceiling,
                )
            }
        }
        match user_graphics.shadow_update {
            Some(v) => self.shadow_update = v,
            None => {
                self.shadow_update = crate::gfx::quality_preset::clamp_shadow_update(
                    self.shadow_update,
                    &quality_ceiling,
                )
            }
        }
        // Shadow distance (GraphicsConfig-sourced, live -- the per-frame cascade
        // split reads it). Same baseline / override / ceiling-clamp shape as the
        // shadow knobs above.
        self.authored_shadow_distance = self.shadow_distance;
        match user_graphics.shadow_distance {
            Some(v) => self.shadow_distance = v,
            None => {
                self.shadow_distance = crate::gfx::quality_preset::clamp_shadow_distance(
                    self.shadow_distance,
                    &quality_ceiling,
                )
            }
        }
        // Shadow cascade count (GraphicsConfig-sourced, live -- the per-frame split
        // + schedule read it). Same baseline / override / ceiling-clamp shape.
        self.authored_shadow_cascades = self.shadow_cascades;
        match user_graphics.shadow_cascades {
            Some(v) => self.shadow_cascades = v,
            None => {
                self.shadow_cascades = crate::gfx::quality_preset::clamp_shadow_cascades(
                    self.shadow_cascades,
                    &quality_ceiling,
                )
            }
        }
        // Anisotropy (GraphicsConfig-sourced, restart-required -- the scene sampler
        // is built from `self.anisotropy` at backend init below). Same baseline /
        // override / ceiling-clamp shape as the shadow knobs above.
        self.authored_anisotropy = self.anisotropy;
        match user_graphics.anisotropy {
            Some(v) => self.anisotropy = v,
            None => {
                self.anisotropy =
                    crate::gfx::quality_preset::clamp_anisotropy(self.anisotropy, &quality_ceiling)
            }
        }
        // Frames-in-flight (ring-buffer depth): a persisted override clamped to the
        // 1..3 the backends support, applied unconditionally like vsync. Restart-
        // required (the ring buffers are sized at backend init below), independent
        // of the quality preset.
        if let Some(v) = user_graphics.frames_in_flight {
            self.frames_in_flight = (v as usize).clamp(1, 3);
        }

        // Resolve post-process tunables. The first declared PostProcessConfig
        // wins; with none declared the renderer uses the stack defaults. The
        // AA mode resolves into a TAA gate (threaded alongside the params) and
        // the composite `fxaa` flag inside `post_process` (refreshed below once
        // the override + ceiling clamp have settled the final mode).
        let post_config = ctx.drain::<PostProcessConfig>().into_iter().next();
        let mut post_process = post_config
            .as_ref()
            .map(|c| c.resolve())
            .unwrap_or(crate::gfx::render_types::PostProcessParams::DEFAULT);
        // A persisted exposure choice (the Exposure slider) overrides the
        // world's `exposure_ev`. Stored as EV (stops); convert to the linear
        // multiplier the shaders use, clamped like `PostProcessConfig::resolve`.
        // Applied live at runtime and re-applied here each launch so a persisted
        // value survives a restart.
        // Persisted slider choices override the world's values, re-applied here
        // each launch so they survive a restart. The transform / clamp is shared
        // with the live drag-apply via `settings::slider_apply_value`, so the
        // value re-applied at launch matches the value applied at drag time.
        use crate::gfx::settings::slider_apply_value;
        if let Some(ev) = user_graphics.exposure_ev {
            post_process.exposure = slider_apply_value("exposure", ev);
        }
        if let Some(v) = user_graphics.bloom_intensity {
            post_process.bloom_intensity = slider_apply_value("bloom_intensity", v);
        }
        if let Some(v) = user_graphics.bloom_threshold {
            post_process.bloom_threshold = slider_apply_value("bloom_threshold", v);
        }
        if let Some(v) = user_graphics.vignette {
            post_process.vignette = slider_apply_value("vignette", v);
        }
        if let Some(v) = user_graphics.lut_strength {
            post_process.lut_strength = slider_apply_value("lut_strength", v);
        }
        if let Some(v) = user_graphics.bloom_knee {
            post_process.bloom_knee = slider_apply_value("bloom_knee", v);
        }
        // Keep a copy as the live source of truth for the slider settings to
        // read at init and mutate at runtime (PostProcessParams is Copy, so the
        // value is still passed into the backend below).
        self.post_process = post_process;
        // Ambient (IBL) scale: the world's `PostProcessConfig.ambient_intensity`
        // overridden by any persisted choice. It rides `LightUniforms`, not
        // `PostProcessParams`, so it is held here and pushed to the backend once
        // after it is built (the world value is already seeded at backend init,
        // so this only matters for a persisted override). Clamped like
        // `PostProcessConfig::ambient_intensity`.
        let world_ambient = post_config
            .as_ref()
            .map(|c| c.ambient_intensity())
            .unwrap_or(1.0);
        self.ambient_intensity = slider_apply_value(
            "ambient_intensity",
            user_graphics.ambient_intensity.unwrap_or(world_ambient),
        );
        // Quality-feature toggles: the world's config overlaid with the user's
        // persisted choices, stored as the source of truth for the Quality-group
        // rows. A runtime toggle flips a field here, re-derives the per-feature
        // settings, and rebuilds the affected GPU resources. The overlay applies
        // only when the world declares a config: overriding a feature is
        // meaningless without its tunables, and the upscaler / ambient resolution
        // below intentionally keys off `post_config.is_some()` (synthesizing a
        // config would wrongly engage the upscaler). The stored copy is still
        // defaulted when absent so a runtime toggle has a config to flip.
        self.post_config = post_config.clone().unwrap_or_default();
        // The pristine world baseline, before the user overrides + preset ceiling
        // below. A live preset change re-clamps the quality toggles from this, so
        // raising a preset restores the world's features (a ceiling never enables
        // anything the world did not author, so re-clamping the baseline is exact).
        self.authored_post_config = self.post_config.clone();
        if post_config.is_some() {
            if let Some(v) = user_graphics.ssao {
                super::set_quality_toggle(&mut self.post_config, "ssao", v);
            }
            if let Some(v) = user_graphics.ssr {
                super::set_quality_toggle(&mut self.post_config, "ssr", v);
            }
            if let Some(v) = user_graphics.ray_traced_reflections {
                super::set_quality_toggle(&mut self.post_config, "ray_traced_reflections", v);
            }
            if let Some(v) = user_graphics.ssgi {
                super::set_quality_toggle(&mut self.post_config, "ssgi", v);
            }
            if let Some(v) = user_graphics.auto_exposure {
                super::set_quality_toggle(&mut self.post_config, "auto_exposure", v);
            }
            // AA mode + SSGI gather + reflection blur sub-quality overrides
            // (cycle dropdowns), alongside the boolean toggles above.
            if let Some(v) = user_graphics.aa_mode {
                self.post_config.aa_mode = v;
            }
            if let Some(v) = user_graphics.ssgi_resolution {
                self.post_config.ssgi_resolution = v;
            }
            if let Some(v) = user_graphics.ssgi_rays {
                self.post_config.ssgi_rays = v;
            }
            if let Some(v) = user_graphics.ssgi_steps {
                self.post_config.ssgi_steps = v;
            }
            if let Some(v) = user_graphics.reflection_blur_resolution {
                self.post_config.reflection_blur_resolution = v;
            }
            // Per-feature sub-quality slider overrides (look-tuning, applied live
            // via update_quality_params). Clamped through the shared
            // `slider_apply_value` so the value re-applied at launch matches the
            // value applied at drag time. Not preset-governed, so no ceiling clamp.
            use crate::gfx::settings::slider_apply_value as sav;
            if let Some(v) = user_graphics.ssao_radius {
                self.post_config.ssao_radius = sav("ssao_radius", v);
            }
            if let Some(v) = user_graphics.ssao_intensity {
                self.post_config.ssao_intensity = sav("ssao_intensity", v);
            }
            if let Some(v) = user_graphics.ssr_intensity {
                self.post_config.ssr_intensity = sav("ssr_intensity", v);
            }
            if let Some(v) = user_graphics.ssr_max_distance {
                self.post_config.ssr_max_distance = sav("ssr_max_distance", v);
            }
            if let Some(v) = user_graphics.ssgi_intensity {
                self.post_config.ssgi_intensity = sav("ssgi_intensity", v);
            }
            if let Some(v) = user_graphics.ssgi_max_distance {
                self.post_config.ssgi_max_distance = sav("ssgi_max_distance", v);
            }
            if let Some(v) = user_graphics.auto_exposure_min_ev {
                self.post_config.auto_exposure_min_ev = sav("auto_exposure_min_ev", v);
            }
            if let Some(v) = user_graphics.auto_exposure_max_ev {
                self.post_config.auto_exposure_max_ev = sav("auto_exposure_max_ev", v);
            }
            if let Some(v) = user_graphics.auto_exposure_speed {
                self.post_config.auto_exposure_speed = sav("auto_exposure_speed", v);
            }
        }
        // Apply the active quality preset as a performance ceiling over the
        // world's authored toggles: where the ceiling disallows a feature, force
        // it off -- but only for a toggle the user did not explicitly override (an
        // explicit choice wins), and never turning a feature on. A no-op under
        // Custom / an unclassified GPU (the ceiling permits everything). Runs
        // after the user-override overlay and before the per-feature derivation
        // below, so the backend builds against the clamped config.
        if post_config.is_some() {
            let clamp = |cfg: &mut crate::assets::PostProcessConfig,
                         key: &str,
                         overridden: bool,
                         allowed: bool| {
                if !overridden && !allowed {
                    super::set_quality_toggle(cfg, key, false);
                }
            };
            clamp(
                &mut self.post_config,
                "ssao",
                user_graphics.ssao.is_some(),
                quality_ceiling.ssao,
            );
            clamp(
                &mut self.post_config,
                "ssr",
                user_graphics.ssr.is_some(),
                quality_ceiling.ssr,
            );
            clamp(
                &mut self.post_config,
                "ray_traced_reflections",
                user_graphics.ray_traced_reflections.is_some(),
                quality_ceiling.ray_traced_reflections,
            );
            clamp(
                &mut self.post_config,
                "ssgi",
                user_graphics.ssgi.is_some(),
                quality_ceiling.ssgi,
            );
            clamp(
                &mut self.post_config,
                "auto_exposure",
                user_graphics.auto_exposure.is_some(),
                quality_ceiling.auto_exposure,
            );
            // Clamp the cycle quality knobs (SSGI gather + reflection blur) under
            // the ceiling too (coarser resolution / fewer rays / steps), skipping
            // any the user explicitly overrode.
            let cycle_overridden = |key: &str| match key {
                "aa_mode" => user_graphics.aa_mode.is_some(),
                "ssgi_resolution" => user_graphics.ssgi_resolution.is_some(),
                "ssgi_rays" => user_graphics.ssgi_rays.is_some(),
                "ssgi_steps" => user_graphics.ssgi_steps.is_some(),
                "reflection_blur_resolution" => user_graphics.reflection_blur_resolution.is_some(),
                _ => false,
            };
            for key in crate::gfx::settings::QUALITY_CYCLE_KEYS {
                super::clamp_quality_cycle(
                    &mut self.post_config,
                    key,
                    &quality_ceiling,
                    cycle_overridden(key),
                );
            }
        }
        // Per-feature settings, derived from the overlaid config. Each is the
        // init-time gate the backend builds against; the same derivation feeds a
        // live rebuild (`derive_quality_settings`). RT reflections need an
        // RT-capable GPU, falling back to SSR where ray tracing is unavailable.
        // RT takes precedence over SSR where both are on (the graph builder picks
        // `RtReflections`), reusing the same SSR pre-pass G-buffer + resolve
        // target.
        let taa_enabled = self.post_config.aa_mode.taa_enabled();
        // The composite FXAA flag follows the final (overridden + ceiling-clamped)
        // AA mode. resolve() seeded `post_process.fxaa` from the authored mode
        // before the override/clamp above, so refresh both the local copy passed
        // to the backend ctor and the live `self.post_process` here.
        post_process.fxaa = if self.post_config.aa_mode.fxaa_enabled() {
            1.0
        } else {
            0.0
        };
        self.post_process.fxaa = post_process.fxaa;
        let ssao_settings = self.post_config.ssao_settings();
        let ssr_settings = self.post_config.ssr_settings();
        let rt_reflection_settings = self.post_config.rt_reflection_settings();
        let reflection_blur_scale = self.post_config.reflection_blur_divisor();
        let ssgi_settings = self.post_config.ssgi_settings();
        // The authored `exposure_ev` becomes an additive bias on the adapted EV
        // when auto-exposure is on; otherwise the static path bakes it into
        // `post_process.exposure` (resolve()) and the bias here is unused.
        let auto_exposure_settings = self.post_config.auto_exposure_settings();
        let auto_exposure_bias_ev = self.post_config.exposure_ev;
        // Display-output / upscaling preferences: the world's value overridden by
        // any persisted settings-menu choice. Restart-required (the swapchain
        // format + render targets are sized once at init), so they are read here,
        // passed to the backend ctor below, and held on self for the settings rows
        // to display + cycle. Independent of the quality preset (a user choice,
        // not a tier), so they never clamp under the ceiling or flip it to Custom.
        // HDR display output is additionally gated on the platform advertising an
        // HDR-capable surface (else it warns and falls back to the SDR composite).
        self.hdr_display = user_graphics
            .hdr_display
            .unwrap_or_else(|| post_config.as_ref().map(|c| c.hdr_display).unwrap_or(false));
        self.hdr_pq = user_graphics
            .hdr_pq
            .unwrap_or_else(|| post_config.as_ref().map(|c| c.hdr_pq).unwrap_or(false));
        self.temporal_upscaling = user_graphics.temporal_upscaling.unwrap_or_else(|| {
            post_config
                .as_ref()
                .map(|c| c.temporal_upscaling)
                .unwrap_or(false)
        });
        let hdr_display = self.hdr_display;
        let hdr_pq = self.hdr_pq;
        let temporal_upscaling = self.temporal_upscaling;
        // Two-pass Hi-Z occlusion + texture-streaming quality: also restart-class
        // and independent of the preset, resolved here (before the value-label sync
        // below) from the world's config overridden by any persisted choice.
        // `occlusion_two_pass` is gated on the bindless GPU-cull path being active
        // (the cull pipeline must exist). The texture pool size + per-frame upload
        // budget come from the StreamingConfig, drained here so the override lands
        // before the streamer is built later; the pool only bites where the world
        // declares streaming.
        self.occlusion_two_pass = user_graphics.occlusion_two_pass.unwrap_or_else(|| {
            post_config
                .as_ref()
                .map(|c| c.occlusion_two_pass)
                .unwrap_or(false)
        });
        let occlusion_two_pass = self.occlusion_two_pass;
        let mut streaming_config = ctx.drain::<StreamingConfig>().into_iter().next();
        if let Some(sc) = streaming_config.as_mut() {
            if let Some(v) = user_graphics.texture_cap {
                sc.texture_cap = v;
            }
            if let Some(v) = user_graphics.texture_budget {
                sc.texture_budget = v;
            }
        }
        self.texture_cap = streaming_config
            .as_ref()
            .map(|c| c.texture_cap)
            .unwrap_or(96);
        self.texture_budget = streaming_config
            .as_ref()
            .map(|c| c.texture_budget)
            .unwrap_or(4);
        // Render-scale (upscaling quality): the world's choice overridden by any
        // persisted settings-menu choice. Restart-required -- the upscaler and
        // render targets are sized from this once, here. `self.render_scale` is
        // kept for the settings row to display and cycle.
        let world_quality = post_config
            .as_ref()
            .map(|c| c.upscale_quality)
            .unwrap_or_default();
        // A persisted render-scale choice wins; otherwise the world's choice,
        // clamped under the preset ceiling (the more aggressive of the two, so a
        // weak-tier ceiling forces more upscaling but never less).
        self.render_scale = match user_graphics.render_scale {
            Some(v) => v,
            None => crate::gfx::quality_preset::more_aggressive_upscale(
                world_quality,
                quality_ceiling.min_upscale,
            ),
        };
        let upscale_scale = if post_config.is_some() {
            self.render_scale.scale()
        } else {
            1.0
        };
        // Upscaler backend (Auto / FSR3 / DLSS / XeSS): the persisted choice wins,
        // else the world's value. Restart-required (the upscaler is selected +
        // built once at init); independent of the quality preset, so no ceiling
        // clamp. Resolved here (ahead of the value-label sync) so the settings row
        // shows the live value. DirectX / Vulkan honour it; Metal uses MetalFX.
        self.upscale_backend = user_graphics.upscale_backend.unwrap_or_else(|| {
            post_config
                .as_ref()
                .map(|c| c.upscale_backend)
                .unwrap_or_default()
        });

        // Set each settings value label to its live value before the first
        // render, so a persisted/authored choice shows instead of the build's
        // placeholder. HitRegions are still present here: GraphicsSystem.init
        // runs before UiInputSystem.init, which drains them.
        let (vsync, mode, scale) = (self.vsync, self.window_args.mode, self.render_scale);
        let fps_cap_val = self.fps_cap;
        // Stats-HUD display toggles for the value-label sync (copies, so the
        // closure below does not borrow self while ctx is borrowed mutably).
        let (perf_stats_val, show_fps_val, show_vram_val) =
            (self.perf_stats, self.show_fps, self.show_vram);
        // Display-group toggle states for the value-label sync (copies, so the
        // closure below does not borrow self while ctx is borrowed mutably).
        let (display_upscaling, display_hdr, display_pq) =
            (self.temporal_upscaling, self.hdr_display, self.hdr_pq);
        // Upscaler-backend selection for the value-label sync (a copy, same
        // reason as the display tuple above).
        let upscale_backend_sel = self.upscale_backend;
        // Shadow knob states for the value-label sync (copies, same reason).
        let (shadow_size, shadow_update_val) = (self.shadow_map_size, self.shadow_update);
        let shadow_distance_val = self.shadow_distance;
        let shadow_cascades_val = self.shadow_cascades;
        let anisotropy_val = self.anisotropy;
        // System / streaming restart-row states for the value-label sync (copies).
        // `occlusion_two_pass` is already a local above.
        let (frames_in_flight_n, texture_cap_n) = (self.frames_in_flight as u32, self.texture_cap);
        // Audio / controls value labels read from the persisted settings store
        // (with the baseline default when unset); their owning systems apply the
        // value at their own init.
        let user_settings = self.persisted_settings();
        let master_volume = user_settings
            .audio
            .master_volume
            .unwrap_or(crate::gfx::settings::DEFAULT_MASTER_VOLUME);
        // Movement key map: a persisted rebind set overrides the engine default.
        // Pushed to the backend after it is built (below) and used to sync the
        // Controls-tab rebind row labels (`init_rebind_rows`).
        self.keymap = user_settings.controls.keymap.unwrap_or_default();
        // Snapshot of the resolved quality toggles for the value-label arm below
        // (a copy, matching the other snapshot locals, so the closure does not
        // borrow self while ctx is borrowed mutably).
        let quality_cfg = self.post_config.clone();
        sync_setting_value_labels(ctx, |key| match key {
            "vsync" => Some(vsync as usize),
            "fps_cap" => Some(crate::gfx::settings::fps_cap_index(fps_cap_val)),
            "window_mode" => Some(crate::gfx::settings::window_mode_index(mode)),
            // "resolution" is a dynamic dropdown; its label is set from the
            // enumerated mode list after the backend is built.
            "render_scale" => Some(crate::gfx::settings::render_scale_index(scale)),
            "upscale_backend" => Some(crate::gfx::settings::upscale_backend_index(
                upscale_backend_sel,
            )),
            "master_volume" => Some(crate::gfx::settings::master_volume_index(master_volume)),
            // Display-output / upscaling toggles (Off/On), held on self.
            "temporal_upscaling" => Some(display_upscaling as usize),
            "hdr_display" => Some(display_hdr as usize),
            "hdr_pq" => Some(display_pq as usize),
            // Stats-HUD display toggles (Off/On), held on self.
            "perf_stats" => Some(perf_stats_val as usize),
            "show_fps" => Some(show_fps_val as usize),
            "show_vram" => Some(show_vram_val as usize),
            // Shadow quality knobs (resolution restart-required, cadence live).
            "shadow_map_size" => Some(crate::gfx::settings::shadow_resolution_index(shadow_size)),
            "shadow_update" => Some(crate::gfx::settings::shadow_update_index(shadow_update_val)),
            "shadow_distance" => Some(crate::gfx::settings::shadow_distance_index(
                shadow_distance_val,
            )),
            "shadow_cascades" => Some(crate::gfx::settings::shadow_cascades_index(
                shadow_cascades_val,
            )),
            "anisotropy" => Some(crate::gfx::settings::anisotropy_index(anisotropy_val)),
            // System / streaming restart rows.
            "frames_in_flight" => Some(crate::gfx::settings::frames_in_flight_index(
                frames_in_flight_n,
            )),
            "occlusion_two_pass" => Some(occlusion_two_pass as usize),
            "texture_quality" => Some(crate::gfx::settings::texture_quality_index(texture_cap_n)),
            // mouse_sensitivity is a slider now, synced by `init_sliders`.
            // Quality toggles: index 0 = Off, 1 = On, matching OFF_ON_OPTIONS.
            key if crate::gfx::settings::is_quality_toggle(key) => {
                super::quality_toggle_on(&quality_cfg, key).map(|on| on as usize)
            }
            // SSGI gather sub-quality dropdowns.
            key if super::is_quality_cycle(key) => super::quality_cycle_index(&quality_cfg, key),
            _ => None,
        });
        // The master "Graphics Quality" row carries the resolved tier under Auto
        // (e.g. "Auto (High)"), which the static option table cannot express, so
        // it is set directly after the generic sync above writes the bare name.
        let preset_label =
            crate::gfx::quality_preset::preset_label(active_preset, &self.gpu_profile);
        set_setting_row_label(ctx, "graphics_quality", &preset_label);
        // Capture the slider rows and sync each handle + value label to its live
        // value (e.g. the persisted/authored exposure). Like the cycle-row sync
        // above, this runs before UiInputSystem drains the HitRegions.
        self.init_sliders(ctx);
        // Capture the rebind rows and sync each value label to the live bound
        // key (persisted or default). Like the slider sync, before UiInputSystem
        // drains the HitRegions.
        self.init_rebind_rows(ctx);
        // Capture each cycle row's value-label id, so a preset change can relabel
        // its dependent rows (and a quality-row change the master row) at runtime,
        // when the HitRegions are gone. Also before UiInputSystem drains them.
        self.init_cycle_value_labels(ctx);
        // Capture the show_fps / show_vram row labels and apply the initial
        // gray-out from the resolved "Display performance stats" master toggle.
        // Before UiInputSystem drains the HitRegions / ScrollPanels.
        self.capture_perf_sub_rows(ctx);
        // Capture the Resolution row's labels and apply the initial gray-out
        // from the resolved window mode (the row only applies in fullscreen).
        self.capture_resolution_row(ctx);
        // Capture each ScrollPanel's per-element clip bands for the draw path,
        // before UiInputSystem drains the panels (init order: graphics first).
        self.init_clip_rects(ctx);
        // Upscaler backend selector, resolved above (persisted choice over the
        // world's `PostProcessConfig.upscale_backend`) and held on self for the
        // settings row. Honoured by the DirectX and Vulkan backends (FSR3 / DLSS /
        // XeSS); Metal always uses MetalFX, so it ignores the selector.
        let upscale_backend = self.upscale_backend;
        // Infinite-world chunk streaming. The first declared VoxelWorld wins;
        // with none declared, no chunks stream. BlockTypes are drained here so
        // the runtime can resolve the VoxelWorld palette to chunk-mesh data.
        let voxel_world = ctx.drain::<VoxelWorld>().into_iter().next();
        let block_types: std::collections::HashMap<AssetId, BlockType> = ctx
            .drain::<BlockType>()
            .into_iter()
            .map(|bt| (bt.asset_id, bt))
            .collect();

        // Whether the blob payloads came from files on disk (`cn run`) rather
        // than an in-memory build (`cn debug`). Captured before the blobs are
        // released; the streaming subsystem uses it to pick a disk-backed
        // payload source so streamed bytes need not stay RAM-resident.
        let blob_disk_backed = ctx.blob.disk_backed();

        // Snapshot each ProceduralMesh before `load_mesh_geometry` drains
        // them, so the world.jsonl hot-reload pass can diff a freshly parsed
        // on-disk entry against the init state and re-run the generator when
        // they differ. A `None` here (hot-reload off) keeps the captured set
        // empty so the reload pass has nothing to inspect on `cn run`. Names
        // come from the interner so the reload log can read "regenerated
        // 'box_mesh'" instead of an opaque id.
        let proc_mesh_args_snapshot: std::collections::HashMap<
            AssetId,
            (String, crate::assets::ProceduralMesh),
        > = if crate::app::dev_flags::enabled() {
            let name_table = crate::ecs::asset_id::name_table();
            ctx.query::<crate::assets::ProceduralMesh>()
                .filter_map(|pm| {
                    let name = name_table.get(pm.asset_id.0 as usize).cloned()?;
                    Some((pm.asset_id, (name, pm.clone())))
                })
                .collect()
        } else {
            std::collections::HashMap::new()
        };

        let (mesh_geometry, mesh_sources, always_resident_meshes, component_mesh_handles) =
            match draw_list::load_mesh_geometry(ctx) {
                Some(m) => m,
                None => {
                    self.failed = true;
                    return;
                }
            };

        // Load the SkinnedMesh resource table and decode each entry's geometry
        // payload now, before the shared blob is released. The placement,
        // material references, capsule, and spawn reserve travel in the baked
        // `data_bytes`; the vertex/index geometry + skeleton in the compiled
        // payload. The table index IS the mesh's `SkinnedMeshHandle`, which keys
        // the whole animation correlation web.
        // Decoded geometry for one SkinnedMesh: its handle, interned name id,
        // the baked data, its vertices, LOD0 indices, the bind-pose skeleton,
        // and LOD alternates.
        type SkinnedGeometry = (
            crate::ecs::SkinnedMeshHandle,
            AssetId,
            crate::assets::SkinnedMesh,
            Vec<crate::gfx::mesh_payload::SkinnedVertex>,
            Vec<u16>,
            Vec<crate::assets::JointDef>,
            Vec<(f32, Vec<u16>)>,
        );
        let skinned_table = ctx
            .resource::<crate::resource::SkinnedMeshTable>()
            .cloned()
            .unwrap_or_default();
        let mut skinned_geometry: Vec<SkinnedGeometry> = Vec::new();
        let mut skinned_blob_indices: Vec<u32> = Vec::new();
        // Interned name -> handle, published for the debug WS animation
        // commands, which address a mesh by its typed name.
        let mut skinned_name_index: std::collections::HashMap<
            AssetId,
            crate::ecs::SkinnedMeshHandle,
        > = std::collections::HashMap::new();
        for (handle, entry) in skinned_table.0.iter().enumerate() {
            let handle = crate::ecs::SkinnedMeshHandle(handle as u32);
            let (name_id, sm): (u32, crate::assets::SkinnedMesh) =
                match postcard::from_bytes(&entry.data_bytes) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!(
                            "GraphicsSystem: SkinnedMesh handle {} baked data failed to decode: {}",
                            handle.index(),
                            e
                        );
                        self.failed = true;
                        return;
                    }
                };
            let name_id = AssetId(name_id);
            skinned_name_index.insert(name_id, handle);
            let locator = match &entry.payload {
                Some(l) => l.clone(),
                None => {
                    tracing::error!(
                        "GraphicsSystem: SkinnedMesh handle {} has no compiled payload",
                        handle.index()
                    );
                    self.failed = true;
                    return;
                }
            };
            skinned_blob_indices.push(locator.blob_index);
            let bytes = match ctx.read_payload(&locator) {
                Ok(b) => b.to_vec(),
                Err(e) => {
                    tracing::error!(
                        "GraphicsSystem: failed to read SkinnedMesh handle {} payload: {:?}",
                        handle.index(),
                        e
                    );
                    self.failed = true;
                    return;
                }
            };
            match crate::gfx::mesh_payload::deserialise_skinned_with_lods(&bytes) {
                Ok((v, idx, payload_joints, alternates)) => {
                    let skeleton = crate::geometry::payload_joints_to_defs(payload_joints);
                    skinned_geometry.push((handle, name_id, sm, v, idx, skeleton, alternates));
                }
                Err(e) => {
                    tracing::error!("GraphicsSystem: malformed SkinnedMesh payload: {}", e);
                    self.failed = true;
                    return;
                }
            }
        }
        // Publish the name index before AnimationSystem inits (it runs after
        // GraphicsSystem) so debug WS animation commands can resolve a typed
        // mesh name to the handle keying the correlation web.
        ctx.insert_resource(crate::gfx::skinned_mesh_map::SkinnedMeshNameIndex(
            skinned_name_index,
        ));

        // drain Model components into a name-keyed map for Prop lookup
        let models = ctx.drain::<Model>();
        let model_map: std::collections::HashMap<AssetId, Vec<crate::assets::SubMeshRef>> =
            models.into_iter().map(|m| (m.asset_id, m.meshes)).collect();

        // decode Room payloads before shaders/textures are read; all payloads
        // live in the same blob and must be consumed before it is released
        let (room_geometry, room_blob_indices) = match draw_list::load_room_geometry(ctx) {
            Some(r) => r,
            None => {
                self.failed = true;
                return;
            }
        };

        let mut shaders = ctx.drain::<ShaderStage>();
        let find_shader = |shaders: &mut Vec<ShaderStage>, kind: ShaderKind| {
            shaders
                .iter()
                .position(|s| s.kind == kind)
                .map(|i| shaders.remove(i))
        };

        let vert_instanced_shader = find_shader(&mut shaders, ShaderKind::VertexInstanced);

        let vert_shader = match find_shader(&mut shaders, ShaderKind::Vertex) {
            Some(s) => s,
            None => {
                tracing::error!(
                    "GraphicsSystem: no vertex ShaderStage found -- add one to world.jsonl"
                );
                self.failed = true;
                return;
            }
        };
        let frag_shader = match find_shader(&mut shaders, ShaderKind::Fragment) {
            Some(s) => s,
            None => {
                tracing::error!(
                    "GraphicsSystem: no fragment ShaderStage found -- add one to world.jsonl"
                );
                self.failed = true;
                return;
            }
        };

        let vert_locator = match &vert_shader.locator {
            Some(l) => l.clone(),
            None => {
                tracing::error!(
                    "GraphicsSystem: vertex ShaderStage '{}' has no compiled payload",
                    vert_shader.source
                );
                self.failed = true;
                return;
            }
        };
        let frag_locator = match &frag_shader.locator {
            Some(l) => l.clone(),
            None => {
                tracing::error!(
                    "GraphicsSystem: fragment ShaderStage '{}' has no compiled payload",
                    frag_shader.source
                );
                self.failed = true;
                return;
            }
        };

        // instanced vertex shader is optional; required only when at least
        // one InstancedProp is in the world (which we don't know yet).
        let vert_instanced_locator = vert_instanced_shader
            .as_ref()
            .and_then(|s| s.locator.clone());

        // Capture every world-loaded ShaderStage's resolved on-disk source
        // path so the asset hot-reload watcher can recompile + rebuild
        // pipelines on a `.metal` / `.hlsl` / `.glsl` save. Stages whose
        // current-platform source is the embedded GLSL fallback (or whose
        // declaration uses a non-platform-compatible extension) carry no
        // file to watch and are skipped; the inline GLSL path keeps
        // rendering at whatever was baked in.
        let mut shader_stage_source_map = super::hot_reload_sources::ShaderStageSourceMap::new();
        if crate::app::dev_flags::enabled() {
            let mut capture = |stage_opt: Option<&ShaderStage>, kind: ShaderKind| {
                let Some(stage) = stage_opt else {
                    return;
                };
                let Some(raw) = stage.current_platform_source() else {
                    return;
                };
                // Engine-bundled built-ins are served from `include_str!`-baked
                // source by `concinnity_cook::shader::compile_shader`, not from
                // disk, and a separate watcher in `crate::metal::hot_reload`
                // already covers them via `src/metal/shaders/`. Skip them
                // here so the asset watcher does not redundantly subscribe to
                // a path it cannot meaningfully reload.
                if crate::build::shader::builtin_shader_source(&raw).is_some() {
                    return;
                }
                let resolved = crate::assets::shader_stage::resolve_runtime_source_path(&raw);
                shader_stage_source_map.entries.push(
                    super::hot_reload_sources::ShaderStageSourceEntry {
                        kind,
                        resolved_path: resolved,
                    },
                );
            };
            capture(Some(&vert_shader), ShaderKind::Vertex);
            capture(Some(&frag_shader), ShaderKind::Fragment);
            capture(vert_instanced_shader.as_ref(), ShaderKind::VertexInstanced);
        }

        // drain textures and record a name->slot mapping for Prop texture lookup.
        // Under `cn debug` we also capture the file-backed source paths into
        // `asset_source_map` so the hot-reload watcher can re-decode + re-upload
        // when the user saves a texture on disk. Procedural textures (generator
        // non-empty) and source-less assets carry no source file and are
        // omitted from the map.
        // The shared texture pool comes from the blob's resource stream: cook
        // assigned each texture a dense `TextureHandle` (== its pool slot) and the
        // runtime loaded them into a `TextureTable`. Reading the table by handle
        // replaces draining a `Texture` component column and scanning names.
        let texture_table = ctx
            .resource::<crate::resource::TextureTable>()
            .cloned()
            .unwrap_or_default();
        // Dev-only source catalogue (present under `cn debug`) so the hot-reload
        // watcher can map a texture handle back to the file that backs it.
        let texture_sources = ctx.resource::<crate::resource::TextureSources>().cloned();
        let mut texture_locators = Vec::with_capacity(texture_table.len());
        let mut asset_source_map = super::hot_reload_sources::TextureSourceMap::new();
        // Name -> pool slot, built only under `cn debug` for the runtime
        // spawn-by-name path (`WorldReloadState`). The shipped runtime resolves
        // every texture by handle and never needs it.
        let mut texture_name_to_slot: std::collections::HashMap<AssetId, usize> =
            std::collections::HashMap::new();
        let capture_sources = crate::app::dev_flags::enabled();
        for (slot, entry) in texture_table.0.iter().enumerate() {
            match &entry.payload {
                Some(l) => {
                    texture_locators.push(l.clone());
                    if capture_sources
                        && let Some(info) = texture_sources.as_ref().and_then(|s| s.0.get(slot))
                    {
                        texture_name_to_slot.insert(AssetId(info.name_id), slot);
                        if !info.source.is_empty() {
                            asset_source_map.push_texture(
                                info.source.clone(),
                                info.image_index,
                                slot,
                            );
                        }
                    }
                }
                None => {
                    tracing::error!(
                        "GraphicsSystem: Texture has no compiled payload -- did the build succeed?"
                    );
                    self.failed = true;
                    return;
                }
            }
        }

        // drain Materials and build a name -> (albedo_slot, normal_map_slot, gpu uniforms) map.
        // Materials have no payload; all data lives in their args.
        //
        // Every texture -- whether referenced as an albedo, a normal map, an
        // emissive/ORM map, or a terrain secondary -- lives once in the shared
        // pool at its drain slot, so a normal-map reference resolves to the same
        // slot as any other reference to that texture (no separate normal pool).
        // A material with no normal map carries `NO_NORMAL_MAP_SLOT`, which the
        // backend maps to the pool's flat-normal fallback.
        //
        // Albedo-region textures (albedo, secondary albedo, emissive, packed
        // ORM) carry a cook-assigned `TextureHandle` whose value is the texture's
        // declaration-order drain slot -- i.e. its slot in this pool -- so the
        // handle indexes `texture_locators` directly, no name->slot scan. A
        // handle past the pool is a resolution error (cook validates the
        // reference exists, so this only guards a corrupt build).
        let texture_count = texture_table.len();
        let albedo_slot_of = |handle: crate::ecs::TextureHandle| -> Option<usize> {
            let slot = handle.index();
            (slot < texture_count).then_some(slot)
        };

        // Materials are a DATA resource: each `MaterialTable` entry's `data_bytes`
        // is the cook-validated Material, serialized. Decode each in handle order
        // into the per-object GPU uniforms + resolved texture slots the draw list
        // indexes by `MaterialHandle` (the table is dense, so index == handle).
        let material_table = ctx
            .resource::<crate::resource::MaterialTable>()
            .cloned()
            .unwrap_or_default();
        let mut material_map: std::collections::HashMap<crate::ecs::MaterialHandle, MaterialEntry> =
            std::collections::HashMap::with_capacity(material_table.len());
        for (material_handle, entry) in material_table.0.iter().enumerate() {
            let mat: Material = match postcard::from_bytes(&entry.data_bytes) {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!(
                        "GraphicsSystem: Material handle {} failed to decode: {}",
                        material_handle,
                        e
                    );
                    self.failed = true;
                    return;
                }
            };
            let albedo_slot = match mat.albedo {
                None => 0,
                Some(handle) => match albedo_slot_of(handle) {
                    Some(slot) => slot,
                    None => {
                        tracing::error!(
                            "GraphicsSystem: Material {} references out-of-range albedo texture handle {} (only {} textures)",
                            material_handle,
                            handle.index(),
                            texture_count
                        );
                        self.failed = true;
                        return;
                    }
                },
            };

            // A normal map is a texture in the shared pool at its own drain slot,
            // addressed by its cook-assigned `TextureHandle` just like the albedo
            // region; unset selects the flat-normal fallback. A handle past the
            // pool is a resolution error (cook validated the reference exists).
            let normal_map_slot = match mat.normal_map {
                None => crate::gfx::render_types::NO_NORMAL_MAP_SLOT,
                Some(handle) => match albedo_slot_of(handle) {
                    Some(slot) => slot,
                    None => {
                        tracing::error!(
                            "GraphicsSystem: Material {} references out-of-range normal_map texture handle {} (only {} textures)",
                            material_handle,
                            handle.index(),
                            texture_count
                        );
                        self.failed = true;
                        return;
                    }
                },
            };

            // Optional secondary albedo/normal pair for the terrain shader's
            // slope-blending mode. Same handle resolution as the primary pair
            // (both index the shared pool by `TextureHandle`); the fallback slots
            // fall through when either is unset and the shader's
            // `terrain_blend > 0` gate is what controls whether the secondary
            // actually gets sampled.
            let albedo_secondary_slot: u32 = match mat.albedo_secondary {
                None => 0,
                Some(handle) => match albedo_slot_of(handle) {
                    Some(slot) => slot as u32,
                    None => {
                        tracing::error!(
                            "GraphicsSystem: Material {} references out-of-range albedo_secondary texture handle {} (only {} textures)",
                            material_handle,
                            handle.index(),
                            texture_count
                        );
                        self.failed = true;
                        return;
                    }
                },
            };
            // Secondary normal for the terrain slope-blend layer, sampled only
            // when `terrain_blend > 0`. Resolves to the texture's shared-pool
            // slot; unset selects the flat-normal fallback (index `texture_count`,
            // one past the last real texture) so a slope layer without its own
            // normal map perturbs nothing.
            let normal_secondary_slot: u32 = match mat.normal_secondary {
                None => texture_count as u32,
                Some(handle) => match albedo_slot_of(handle) {
                    Some(slot) => slot as u32,
                    None => {
                        tracing::error!(
                            "GraphicsSystem: Material {} references out-of-range normal_secondary texture handle {} (only {} textures)",
                            material_handle,
                            handle.index(),
                            texture_count
                        );
                        self.failed = true;
                        return;
                    }
                },
            };

            // Emissive + packed-ORM maps live in the albedo region of the
            // bindless pool, so their `TextureHandle` indexes the pool directly
            // like the primary/secondary albedo. Slot 0 (unset) is the sentinel
            // the shader gates on to keep the scalar fallback.
            let emissive_map_slot: u32 = match mat.emissive_map {
                None => 0,
                Some(handle) => match albedo_slot_of(handle) {
                    Some(slot) => slot as u32,
                    None => {
                        tracing::error!(
                            "GraphicsSystem: Material {} references out-of-range emissive_map texture handle {} (only {} textures)",
                            material_handle,
                            handle.index(),
                            texture_count
                        );
                        self.failed = true;
                        return;
                    }
                },
            };
            let orm_map_slot: u32 = match mat.orm_map {
                None => 0,
                Some(handle) => match albedo_slot_of(handle) {
                    Some(slot) => slot as u32,
                    None => {
                        tracing::error!(
                            "GraphicsSystem: Material {} references out-of-range orm_map texture handle {} (only {} textures)",
                            material_handle,
                            handle.index(),
                            texture_count
                        );
                        self.failed = true;
                        return;
                    }
                },
            };

            let uniforms = crate::gfx::render_types::MaterialUniforms {
                roughness: mat.roughness,
                metallic: mat.metallic,
                macro_variation: mat.macro_variation,
                terrain_blend: mat.terrain_blend,
                tint: mat.tint,
                _pad2: 0.0,
                emissive: mat.emissive_factor,
                secondary_blend_sharpness: mat.secondary_blend_sharpness,
                albedo_secondary_index: albedo_secondary_slot,
                normal_secondary_index: normal_secondary_slot,
                emissive_map_index: emissive_map_slot,
                orm_map_index: orm_map_slot,
                opacity: mat.opacity,
                transparent: u32::from(mat.transparent),
                see_through: u32::from(mat.see_through),
            };
            material_map.insert(
                crate::ecs::MaterialHandle(material_handle as u32),
                (albedo_slot, normal_map_slot, uniforms),
            );
        }

        // Build skinned draw objects, the shared skinned vertex/index buffers,
        // and bind-pose skeletons from the decoded SkinnedMesh geometry. Runs
        // after the material map so SkinnedMesh material references resolve.
        let mut skinned_vertices: Vec<crate::gfx::mesh_payload::SkinnedVertex> = Vec::new();
        let mut skinned_indices: Vec<u16> = Vec::new();
        let mut skinned_draw_objects: Vec<crate::gfx::render_types::SkinnedDrawObject> = Vec::new();
        // One entry per authored skinned mesh: its handle, interned name id,
        // the skinned index of its (visible) template draw object, and its
        // bind-pose skeleton. The template index is recorded explicitly rather
        // than inferred from position because pre-reserved instance copies
        // interleave the draw-object list (template, copies, template, ...).
        // (handle, name id, draw index, skeleton, authored model, optional
        // capsule) per skinned mesh; drives the SkeletonPose + CharacterRig
        // publish after the geometry upload below.
        type SkinnedSkeletonEntry = (
            crate::ecs::SkinnedMeshHandle,
            AssetId,
            usize,
            skinning::Skeleton,
            [[f32; 4]; 4],
            Option<crate::assets::CharacterCapsule>,
        );
        let mut skinned_skeletons: Vec<SkinnedSkeletonEntry> = Vec::new();
        // `(template_index, instance_index)` pairs seeding the backend skinned
        // instance pool: each instance is a hidden bind-pose copy reserved from
        // SkinnedMesh.max_instances.
        let mut skinned_pool_reservations: Vec<(usize, usize)> = Vec::new();
        // Asset hot-reload (`cn debug` only) needs the per-slot vertex region
        // + joint count so it can reject size + shape changes before pushing
        // to the backend. SkinnedMesh is 1:1 with its draw slot (no Prop
        // fan-out), so one entry per asset.
        let mut skinned_mesh_source_map = super::hot_reload_sources::SkinnedMeshSourceMap::new();
        for (handle, name_id, sm, verts, idxs, joint_defs, lod_alts) in &skinned_geometry {
            let (texture_slot, normal_map_slot, material) = if let Some(mat_id) = sm.material {
                match material_map.get(&mat_id) {
                    Some(&(s, n, u)) => (s, n, u),
                    None => {
                        tracing::error!(
                            "GraphicsSystem: SkinnedMesh '{}' references unknown material {}",
                            name_id,
                            mat_id.index()
                        );
                        self.failed = true;
                        return;
                    }
                }
            } else if let Some(tex_id) = sm.texture {
                let slot = tex_id.index();
                (
                    if slot < texture_count { slot } else { 0 },
                    crate::gfx::render_types::NO_NORMAL_MAP_SLOT,
                    crate::gfx::render_types::MaterialUniforms::DEFAULT,
                )
            } else {
                (
                    0,
                    crate::gfx::render_types::NO_NORMAL_MAP_SLOT,
                    crate::gfx::render_types::MaterialUniforms::DEFAULT,
                )
            };

            let base = skinned_vertices.len() as u16;
            let index_offset = skinned_indices.len();
            skinned_vertices.extend_from_slice(verts);
            skinned_indices.extend(idxs.iter().map(|i| i + base));

            // LOD alternates share this slot's vertex region. The runtime
            // skinned IB is u16, so each alternate's mesh-relative indices
            // are rebased onto the same `base` as LOD0, identical to how
            // the shadow / velocity / SSAO / SSR pre-passes already consume
            // the IB.
            let mut lod_slices: Vec<crate::gfx::render_types::LodSlice> =
                Vec::with_capacity(lod_alts.len());
            for (switch_distance, alt_idx) in lod_alts {
                let alt_offset = skinned_indices.len();
                skinned_indices.extend(alt_idx.iter().map(|i| i + base));
                lod_slices.push(crate::gfx::render_types::LodSlice {
                    index_offset: alt_offset,
                    index_count: alt_idx.len(),
                    switch_distance: *switch_distance,
                });
            }

            let skeleton = crate::assets::build_skeleton_from_joint_defs(joint_defs);
            let joint_count = skeleton.len().min(crate::gfx::render_types::MAX_JOINTS);

            // Bind-pose (object-space) AABB over this mesh's vertices. The
            // GPU-driven skinned fold pads + transforms it per frame for culling.
            let (local_bb_min, local_bb_max) = if verts.is_empty() {
                ([0.0; 3], [0.0; 3])
            } else {
                let mut lo = [f32::INFINITY; 3];
                let mut hi = [f32::NEG_INFINITY; 3];
                for v in verts.iter() {
                    for a in 0..3 {
                        lo[a] = lo[a].min(v.pos[a]);
                        hi[a] = hi[a].max(v.pos[a]);
                    }
                }
                (lo, hi)
            };

            let skinned_index = skinned_draw_objects.len();
            skinned_draw_objects.push(crate::gfx::render_types::SkinnedDrawObject {
                vertex_base: base,
                vertex_count: verts.len(),
                index_offset,
                index_count: idxs.len(),
                model: sm.model_matrix(),
                texture_slot,
                normal_map_slot,
                material,
                visible: true,
                joint_count,
                local_bb_min,
                local_bb_max,
                lod_alternates: lod_slices,
            });
            if capture_sources && !sm.source.is_empty() {
                skinned_mesh_source_map.entries.push(
                    super::hot_reload_sources::SkinnedMeshSourceEntry {
                        source: sm.source.clone(),
                        skinned_index,
                        vertex_base: base,
                        vertex_count: verts.len(),
                        index_count: idxs.len(),
                        joint_count,
                    },
                );
            }
            // Pre-reserve runtime spawn copies: append `max_instances` hidden
            // bind-pose duplicates of this mesh, each with its OWN vertex region
            // in the shared skinned buffer. They must not share a region because
            // the GPU skin fold writes the deformed buffer keyed by global vertex
            // index, so two live instances at one region would clobber each
            // other's pose. A runtime skinned spawn reveals one of these without
            // growing any GPU buffer; a despawn returns it to the pool.
            for _ in 0..sm.max_instances {
                // The shared skinned index buffer is u16, so a copy's vertex
                // region must fit there. Stop reserving (and warn) once the next
                // copy would overflow rather than truncating into a neighbour.
                let copy_base_usize = skinned_vertices.len();
                if copy_base_usize + verts.len() > u16::MAX as usize + 1 {
                    let reserved = skinned_draw_objects.len() - skinned_index - 1;
                    tracing::warn!(
                        "GraphicsSystem: SkinnedMesh '{}' reserved {} of {} requested instances; \
                         the u16-indexed skinned vertex buffer is full",
                        name_id,
                        reserved,
                        sm.max_instances
                    );
                    break;
                }
                let copy_base = copy_base_usize as u16;
                let copy_index_offset = skinned_indices.len();
                skinned_vertices.extend_from_slice(verts);
                skinned_indices.extend(idxs.iter().map(|i| i + copy_base));
                let mut copy_lods: Vec<crate::gfx::render_types::LodSlice> =
                    Vec::with_capacity(lod_alts.len());
                for (switch_distance, alt_idx) in lod_alts {
                    let alt_offset = skinned_indices.len();
                    skinned_indices.extend(alt_idx.iter().map(|i| i + copy_base));
                    copy_lods.push(crate::gfx::render_types::LodSlice {
                        index_offset: alt_offset,
                        index_count: alt_idx.len(),
                        switch_distance: *switch_distance,
                    });
                }
                let copy_skinned_index = skinned_draw_objects.len();
                skinned_draw_objects.push(crate::gfx::render_types::SkinnedDrawObject {
                    vertex_base: copy_base,
                    vertex_count: verts.len(),
                    index_offset: copy_index_offset,
                    index_count: idxs.len(),
                    model: sm.model_matrix(),
                    texture_slot,
                    normal_map_slot,
                    material,
                    // Hidden until a runtime spawn claims it.
                    visible: false,
                    joint_count,
                    local_bb_min,
                    local_bb_max,
                    lod_alternates: copy_lods,
                });
                skinned_pool_reservations.push((skinned_index, copy_skinned_index));
            }

            skinned_skeletons.push((
                *handle,
                *name_id,
                skinned_index,
                skeleton,
                sm.model_matrix(),
                sm.capsule.clone(),
            ));
        }

        // read all payloads before releasing -- they may share a blob
        let vert_bytes = match ctx.read_payload(&vert_locator) {
            Ok(b) => b.to_vec(),
            Err(e) => {
                tracing::error!("GraphicsSystem: failed to read vertex shader: {:?}", e);
                self.failed = true;
                return;
            }
        };
        let frag_bytes = match ctx.read_payload(&frag_locator) {
            Ok(b) => b.to_vec(),
            Err(e) => {
                tracing::error!("GraphicsSystem: failed to read fragment shader: {:?}", e);
                self.failed = true;
                return;
            }
        };

        // The DirectX backend engages its bindless main pass only when the
        // main-shader override is empty (it then uses its embedded bindless
        // pipeline + the embedded default for any legacy/streamed fallback). The
        // built-in default ShaderStage compiles to non-empty DXBC, which would
        // pin every built-in world to the legacy per-draw path. When the world's
        // main shader IS the built-in default, hand DX empty bytes so it takes
        // the bindless path, matching Vulkan (whose default payload is already
        // empty) and Metal (whose `default.metal` drives its own bindless pass).
        // Custom-shader worlds keep their compiled bytes and the legacy path.
        // Metal loads its metallib from these bytes, so it is left untouched.
        #[cfg(backend_dx)]
        let (vert_bytes, frag_bytes) = {
            let is_builtin_main = |src: Option<String>| {
                matches!(
                    src.as_deref(),
                    Some("default_vert.hlsl") | Some("default_frag.hlsl") | Some("default.metal")
                )
            };
            if is_builtin_main(vert_shader.current_platform_source())
                && is_builtin_main(frag_shader.current_platform_source())
            {
                (Vec::new(), Vec::new())
            } else {
                (vert_bytes, frag_bytes)
            }
        };
        // The shadow shader is engine-internal now (compiled from
        // `shadow_map.metal`), so there is no per-world shadow payload. The
        // DX / Vulkan constructors still take a shadow byte slice pending their
        // own internal-shadow migration; Metal ignores it.
        let shadow_bytes: Vec<u8> = Vec::new();
        let vert_instanced_bytes = if let Some(ref locator) = vert_instanced_locator {
            match ctx.read_payload(locator) {
                Ok(b) => b.to_vec(),
                Err(e) => {
                    tracing::error!(
                        "GraphicsSystem: failed to read instanced vertex shader: {:?}",
                        e
                    );
                    self.failed = true;
                    return;
                }
            }
        } else {
            Vec::new()
        };

        let mut texture_data: Vec<(u32, u32, Vec<u8>)> = Vec::new();
        // Raw compiled texture payloads, kept past blob release so the
        // asset-streaming subsystem can re-decode them off the main thread.
        // Left empty when the blobs are disk-backed: the streamer then re-reads
        // each payload from its blob file instead of holding a RAM copy.
        let mut texture_payloads: Vec<Vec<u8>> = Vec::new();
        for locator in &texture_locators {
            let tex_bytes = match ctx.read_payload(locator) {
                Ok(b) => b.to_vec(),
                Err(e) => {
                    tracing::error!("GraphicsSystem: failed to read texture payload: {:?}", e);
                    self.failed = true;
                    return;
                }
            };
            match crate::build::texture::deserialise(&tex_bytes) {
                Ok(t) => texture_data.push(t),
                Err(e) => {
                    tracing::error!("GraphicsSystem: malformed texture payload: {}", e);
                    self.failed = true;
                    return;
                }
            }
            if !blob_disk_backed {
                texture_payloads.push(tex_bytes);
            }
        }

        // Read the first EnvironmentMap from its resource table and capture its payload.
        // The runtime supports at most one IBL environment per world;
        // additional declarations are logged and ignored. Under `cn debug` we also
        // capture the resolved HDR source path + sizing knobs into
        // `environment_map_source` so the hot-reload watcher knows what to
        // subscribe to and the reload helper can re-run the convolutions with
        // matching dimensions. Procedural `generator` declarations have no
        // file to watch and are skipped.
        let env_map_table = ctx
            .resource::<crate::resource::EnvironmentMapTable>()
            .cloned()
            .unwrap_or_default();
        if env_map_table.len() > 1 {
            tracing::warn!(
                "GraphicsSystem: {} EnvironmentMaps declared; only the first is used",
                env_map_table.len()
            );
        }
        let mut env_map_bytes: Option<Vec<u8>> = None;
        let mut environment_map_source: Option<super::hot_reload_sources::EnvironmentMapSource> =
            None;
        // The runtime uses handle 0; a compiled EnvironmentMap always carries a
        // payload (procedural generators bake one too), so a `None` locator here
        // means simply "no EnvironmentMap declared".
        if let Some(locator) = env_map_table.locator(0) {
            match ctx.read_payload(&locator) {
                Ok(b) => env_map_bytes = Some(b.to_vec()),
                Err(e) => {
                    tracing::error!(
                        "GraphicsSystem: failed to read EnvironmentMap payload: {:?}",
                        e
                    );
                    self.failed = true;
                    return;
                }
            }
        }
        if capture_sources
            && let Some(info) = ctx
                .resource::<crate::resource::EnvironmentMapSources>()
                .and_then(|s| s.0.clone())
        {
            environment_map_source = Some(super::hot_reload_sources::EnvironmentMapSource {
                resolved_path: crate::build::environment_map::resolve_source_path(&info.source),
                prefilter_face_size: info.prefilter_face_size,
                irradiance_face_size: info.irradiance_face_size,
                prefilter_samples: info.prefilter_samples,
                prefilter_clamp: info.prefilter_clamp,
            });
        }

        // Read the first ColorLut from its resource table and capture its payload.
        // At most one colour-grading LUT per world; extras are logged and ignored.
        // Under `cn debug` we also capture the resolved source path (from the dev
        // source catalogue) into `color_lut_source` so the hot-reload watcher knows
        // what to subscribe to and the reload helper knows where to re-read the LUT.
        let color_lut_table = ctx
            .resource::<crate::resource::ColorLutTable>()
            .cloned()
            .unwrap_or_default();
        if color_lut_table.len() > 1 {
            tracing::warn!(
                "GraphicsSystem: {} ColorLuts declared; only the first is used",
                color_lut_table.len()
            );
        }
        let mut color_lut_bytes: Option<Vec<u8>> = None;
        let mut color_lut_source: Option<super::hot_reload_sources::ColorLutSource> = None;
        // Handle 0 is the sole LUT the renderer applies; a compiled ColorLut always
        // carries a payload, so a `None` locator means "no ColorLut declared".
        if let Some(locator) = color_lut_table.locator(0) {
            match ctx.read_payload(&locator) {
                Ok(b) => color_lut_bytes = Some(b.to_vec()),
                Err(e) => {
                    tracing::error!("GraphicsSystem: failed to read ColorLut payload: {:?}", e);
                    self.failed = true;
                    return;
                }
            }
        }
        if capture_sources
            && let Some(src) = ctx
                .resource::<crate::resource::ColorLutSources>()
                .and_then(|c| c.0.clone())
        {
            color_lut_source = Some(super::hot_reload_sources::ColorLutSource {
                resolved_path: crate::build::color_lut::resolve_source_path(&src),
            });
        }

        // Read Font resources from the FontTable; deserialise each atlas + metrics
        // for text rendering. A font's `FontHandle` is its atlas slot (dense 0..N),
        // so the handle indexes both `loaded_fonts` and the leading `text_atlas_data`
        // slots. `size_px` now rides in the payload (it left the component column).
        let font_table = ctx
            .resource::<crate::resource::FontTable>()
            .cloned()
            .unwrap_or_default();
        let mut text_atlas_data: Vec<(u32, u32, Vec<u8>)> = Vec::new();
        for (slot, entry) in font_table.0.iter().enumerate() {
            let locator = match &entry.payload {
                Some(l) => l.clone(),
                None => {
                    tracing::error!(
                        "GraphicsSystem: Font handle {} has no compiled payload -- did the build succeed?",
                        slot
                    );
                    self.failed = true;
                    return;
                }
            };
            let bytes = match ctx.read_payload(&locator) {
                Ok(b) => b.to_vec(),
                Err(e) => {
                    tracing::error!(
                        "GraphicsSystem: failed to read Font handle {} payload: {:?}",
                        slot,
                        e
                    );
                    self.failed = true;
                    return;
                }
            };
            match crate::build::font::deserialise(&bytes) {
                Ok((aw, ah, supersample, size_px, rgba, metrics)) => {
                    let metrics_map: std::collections::HashMap<
                        u32,
                        crate::build::font::GlyphMetrics,
                    > = metrics.into_iter().map(|m| (m.char_code, m)).collect();
                    let size_px = size_px as f32;
                    self.loaded_fonts.insert(
                        crate::ecs::FontHandle(slot as u32),
                        text::LoadedFont {
                            atlas_slot: slot,
                            cap_px: text::derive_cap_px(&metrics_map, size_px),
                            metrics: metrics_map,
                            atlas_w: aw,
                            atlas_h: ah,
                            size_px,
                            supersample: (supersample.max(1)) as f32,
                        },
                    );
                    text_atlas_data.push((aw, ah, rgba));
                }
                Err(e) => {
                    tracing::error!("GraphicsSystem: malformed Font payload: {}", e);
                    self.failed = true;
                    return;
                }
            }
        }

        // Sprite textures ride the text-atlas pool: each distinct Texture a
        // Sprite references is decoded and appended after the font atlases,
        // drawn by the same pipeline (positive vertex mode = RGBA quad). A
        // Story's stage images are gathered too: the story system swaps them
        // onto the stage sprites at runtime, so they must be resident even
        // though no sprite references them yet. A texture that cannot be
        // resolved demotes its sprite to the solid tint fill, warned rather
        // than fatal.
        let sprite_texture_ids: Vec<crate::ecs::TextureHandle> = {
            let mut ids: Vec<crate::ecs::TextureHandle> = ctx
                .query::<crate::assets::Sprite>()
                .filter_map(|s| s.texture)
                .collect();
            for story in ctx.query::<crate::assets::Story>() {
                let stages = story.nodes.iter().flat_map(|n| {
                    n.pages
                        .iter()
                        .map(|p| &p.stage)
                        .chain(std::iter::once(&n.choice_stage))
                });
                for stage in stages {
                    for image in [&stage.bg, &stage.left, &stage.center, &stage.right]
                        .into_iter()
                        .flatten()
                    {
                        ids.push(image.texture);
                    }
                }
            }
            ids.sort_unstable_by_key(|id| id.0);
            ids.dedup();
            ids
        };
        for tex_id in sprite_texture_ids {
            // The texture handle is the texture's declaration-order pool slot,
            // so it indexes the locator table directly.
            let Some(locator) = texture_locators.get(tex_id.index()).cloned() else {
                tracing::warn!(
                    "GraphicsSystem: Sprite references unknown texture {:?}; drawing its tint",
                    tex_id
                );
                continue;
            };
            match ctx.read_payload(&locator) {
                Ok(bytes) => match crate::build::texture::deserialise(bytes) {
                    Ok((w, h, rgba)) => {
                        self.sprite_texture_slots
                            .insert(tex_id, text_atlas_data.len());
                        text_atlas_data.push((w, h, rgba));
                    }
                    Err(e) => {
                        tracing::warn!("GraphicsSystem: sprite texture {:?}: {}", tex_id, e)
                    }
                },
                Err(e) => tracing::warn!(
                    "GraphicsSystem: sprite texture {:?} payload read failed: {:?}",
                    tex_id,
                    e
                ),
            }
        }

        // Indirect-ambient multiplier from PostProcessConfig, folded into the
        // shared LightUniforms so every backend's main pass scales its IBL /
        // flat-fallback ambient by it. 1.0 (the default) is a no-op.
        let ambient_intensity = post_config
            .as_ref()
            .map(|c| c.ambient_intensity())
            .unwrap_or(1.0);
        let light_uniforms = lights::build_light_uniforms(
            ctx.drain::<DirectionalLight>(),
            ctx.drain::<PointLight>(),
            ambient_intensity,
        );

        let font_blob_indices: Vec<u32> = font_table.blob_indices().into_iter().collect();

        // AudioSystem inits after GraphicsSystem and reads audio-clip payloads
        // from the `AudioClipTable`, so any blob a clip lives in must survive this
        // release sweep.
        let audio_blobs = ctx
            .resource::<crate::resource::AudioClipTable>()
            .map(|table| table.blob_indices())
            .unwrap_or_default();
        // SdfVolume payloads are drained later in this same init pass (see
        // the `sdf_volumes` block below), so the release sweep here must
        // also leave their blobs resident. Without this gate, any world
        // whose SDF shader bytes happen to land alone in a blob shows
        // "failed to read fragment shader payload: FileIo; skipping" at
        // runtime and the SDF surface never draws.
        let sdf_blobs = crate::assets::sdf_volume::sdf_volume_blob_indices(ctx);
        // PhysicsSystem inits after GraphicsSystem and reads the baked
        // heightfield collider grid from a heightfield ProceduralMesh's
        // payload, so those blobs must also survive this sweep.
        let terrain_blobs = crate::assets::procedural_mesh::heightfield_blob_indices(ctx);
        let mut released = std::collections::HashSet::new();
        for idx in std::iter::once(vert_locator.blob_index)
            .chain(std::iter::once(frag_locator.blob_index))
            .chain(vert_instanced_locator.iter().map(|l| l.blob_index))
            .chain(texture_locators.iter().map(|l| l.blob_index))
            .chain(room_blob_indices)
            .chain(font_blob_indices)
            .chain(skinned_blob_indices)
        {
            if !audio_blobs.contains(&idx)
                && !sdf_blobs.contains(&idx)
                && !terrain_blobs.contains(&idx)
                && released.insert(idx)
            {
                ctx.release_blob(idx);
            }
        }

        // InstancedProp components are drained because every instance becomes a
        // baked DrawObject; there is no per-frame update path yet. Drain before
        // taking Prop references because drain shifts the underlying Vec.
        let instanced_props = ctx.drain::<crate::assets::InstancedProp>();

        // Entities to render, in Prop-column order, so each gets a RenderHandle +
        // GlobalTransform attached below. Enumerated through the Transform column
        // (the decomposition gives every prop a Transform in Prop order); the Prop
        // column itself was drained by the decomposition pass at load.
        let prop_entities: Vec<crate::ecs::Entity> = ctx
            .query_with_entity::<crate::assets::Transform>()
            .map(|(entity, _)| entity)
            .collect();

        // Build the draw-list inputs from each entity's per-instance components:
        // renderer fields from MeshRenderer/ModelRenderer, world matrices from
        // Transform/Parent. `items` / `world_mats` are column-aligned with
        // `prop_entities`.
        let resolved = draw_list::resolve_world_matrices(ctx);
        let entity_name: std::collections::HashMap<crate::ecs::Entity, AssetId> = ctx
            .resource::<crate::ecs::decompose::EntityByName>()
            .map(|n| n.0.iter().map(|(&id, &e)| (e, id)).collect())
            .unwrap_or_default();
        let mut items = Vec::with_capacity(prop_entities.len());
        let mut world_mats = Vec::with_capacity(prop_entities.len());
        for &entity in &prop_entities {
            let asset_id = entity_name.get(&entity).copied().unwrap_or_default();
            items.push(draw_list::decomposed_renderable_item(ctx, entity, asset_id));
            world_mats.push(
                resolved
                    .get(&entity)
                    .copied()
                    .unwrap_or(draw_list::IDENTITY4),
            );
        }

        let (
            all_vertices,
            all_indices,
            mut draw_objects,
            instanced_clusters,
            prop_draw_indices,
            mesh_handle_to_draws,
        ) = match draw_list::build_draw_list(draw_list::DrawListInputs {
            items: &items,
            instanced_props: &instanced_props,
            world_mats: &world_mats,
            model_map: &model_map,
            mesh_geometry: &mesh_geometry,
            room_geometry: &room_geometry,
            texture_count,
            material_map: &material_map,
            always_resident_meshes: &always_resident_meshes,
        }) {
            Some(d) => d,
            None => {
                self.failed = true;
                return;
            }
        };

        // Give each prop entity a RenderHandle (its GPU draw slots) and a
        // GlobalTransform (its init world matrix), so the per-frame push reads
        // these. prop_entities is column-aligned with prop_draw_indices and
        // world_mats; prop_draw_indices is consumed here and then dropped.
        for (i, &entity) in prop_entities.iter().enumerate() {
            let draws: Vec<u32> = prop_draw_indices[i]
                .iter()
                .map(|&slot| slot as u32)
                .collect();
            ctx.insert(entity, crate::assets::RenderHandle { draws });
            ctx.insert(entity, crate::assets::GlobalTransform(world_mats[i]));
        }

        // Asset hot-reload mesh map: cross-reference the file-backed source
        // metadata captured at drain time with the per-Mesh draw indices
        // build_draw_list just produced. A Mesh without any draws (referenced
        // by nothing) carries no entry; the watcher would still fire on the
        // .glb change but the reload helper has nothing to push to.
        let mut mesh_source_map = super::hot_reload_sources::MeshSourceMap::new();
        if capture_sources {
            for (handle, meta) in &mesh_sources {
                if let Some(draws) = mesh_handle_to_draws.get(handle) {
                    if draws.is_empty() {
                        continue;
                    }
                    mesh_source_map
                        .entries
                        .push(super::hot_reload_sources::MeshSourceEntry {
                            source: meta.source.clone(),
                            primitive_index: meta.primitive_index,
                            lod_levels: meta.lod_levels,
                            lod_distances: meta.lod_distances.clone(),
                            draw_indices: draws.clone(),
                        });
                }
            }
        }

        // Procedural-mesh hot-reload map: same cross-reference, but the
        // "source" is the JSONL `args` object captured pre-drain rather than
        // a file path. A procedural mesh that no Prop references carries no
        // draws and is omitted; a JSONL save changing its args would be
        // observable only through a future system that introspects the args
        // map directly, which we deliberately do not maintain.
        let mut procedural_mesh_source_map =
            super::hot_reload_sources::ProceduralMeshSourceMap::new();
        if capture_sources {
            for (asset_id, (name, args)) in &proc_mesh_args_snapshot {
                let Some(handle) = component_mesh_handles.get(asset_id) else {
                    continue;
                };
                if let Some(draws) = mesh_handle_to_draws.get(handle) {
                    if draws.is_empty() {
                        continue;
                    }
                    procedural_mesh_source_map.entries.push(
                        super::hot_reload_sources::ProceduralMeshSourceEntry {
                            name: name.clone(),
                            args: args.clone(),
                            draw_indices: draws.clone(),
                        },
                    );
                }
            }
        }

        // A geometry-less world (e.g. text-only) is valid: the backend is
        // initialised with empty geometry buffers and only the text path runs.

        // Per-texture-slot draw positions, captured before `draw_objects` is
        // moved into the backend. The streaming subsystem scores each texture
        // by the camera's distance to the nearest draw that samples it. Albedo
        // and normal maps share one pool, so a draw contributes its position to
        // both the texture it samples as albedo and the one it samples as a
        // normal map (`NO_NORMAL_MAP_SLOT` = no normal map, scored by neither).
        let texture_centers: Vec<Vec<[f32; 3]>> = {
            let mut centers = vec![Vec::new(); texture_data.len()];
            for obj in &draw_objects {
                let pos = draw_object_position(obj);
                if let Some(slot) = centers.get_mut(obj.texture_slot) {
                    slot.push(pos);
                }
                if obj.normal_map_slot != crate::gfx::render_types::NO_NORMAL_MAP_SLOT
                    && let Some(slot) = centers.get_mut(obj.normal_map_slot)
                {
                    slot.push(pos);
                }
            }
            centers
        };

        // Per-streamed-mesh data, also captured before `draw_objects` moves
        // into the backend. Only static, frustum-cullable draws stream; skybox,
        // rooms, and dynamic props (sentinel AABB) stay resident so structural
        // geometry never pops in. Each payload is a copy of the draw's region
        // of the shared vertex/index buffers, scored by its AABB centre.
        let (mesh_stream_draw_indices, mesh_centers, mesh_payloads) = {
            let mut draw_indices: Vec<usize> = Vec::new();
            let mut centers: Vec<Vec<[f32; 3]>> = Vec::new();
            let mut payloads: Vec<crate::app::mesh_stream::DecodedMesh> = Vec::new();
            for (draw_idx, obj) in draw_objects.iter().enumerate() {
                if !obj.cullable() {
                    continue;
                }
                let vstart = obj.vertex_offset / std::mem::size_of::<Vertex>();
                let vend = vstart + obj.vertex_count;
                let iend = obj.index_offset + obj.index_count;
                if vend > all_vertices.len() || iend > all_indices.len() {
                    // Build-time offsets should always be in range; skip
                    // defensively rather than risk an out-of-bounds slice.
                    continue;
                }
                draw_indices.push(draw_idx);
                centers.push(vec![draw_object_position(obj)]);
                // Indices are stored mesh-relative (0-based): the sub-allocator
                // places the mesh's vertices anywhere on upload, and upload_mesh
                // rebases the indices onto whatever vertex region it chose.
                // mesh-relative index is global - vbase; each per-mesh region
                // fits in u16 (the build-time splitter enforces this), so we
                // narrow back here for DecodedMesh's per-mesh u16 indices.
                let vbase = vstart as u32;
                payloads.push(crate::app::mesh_stream::DecodedMesh {
                    vertices: all_vertices[vstart..vend].to_vec(),
                    indices: all_indices[obj.index_offset..iend]
                        .iter()
                        .map(|&i| (i - vbase) as u16)
                        .collect(),
                });
            }
            (draw_indices, centers, payloads)
        };

        // Mesh streaming and LOD alternates don't yet cooperate: upload_mesh
        // writes only LOD0 to its newly-allocated region, but obj.lod_alternates
        // still carries the build-time offsets for LOD1..N. Once another stream
        // upload reuses those byte ranges, active_lod() returns offsets that
        // point at unrelated geometry and the draw renders garbage / nothing
        // (the obelisks vanish past their first LOD switch_distance). Until
        // upload_mesh learns to stream every LOD, strip the alternates from
        // every streamable draw so active_lod() always returns LOD0.
        if streaming_config.is_some() && !mesh_payloads.is_empty() {
            for &draw_idx in &mesh_stream_draw_indices {
                if let Some(obj) = draw_objects.get_mut(draw_idx) {
                    obj.lod_alternates.clear();
                }
            }
        }

        // Shrinkable seed VRAM (Metal + DirectX + Vulkan). By default
        // `build_draw_list` bakes every streamed mesh into the shared
        // vertex/index buffers, sizing them for the whole streamed set, so
        // streaming reuses space but never shrinks GPU memory. When the residency
        // cap is smaller than the streamed set, compact the resident geometry and
        // reserve a smaller seed headroom -- sized to the cap-many largest meshes
        // -- for the streamed meshes, which are placed into it on upload
        // (tolerating a transient alloc miss while freed regions await their
        // retire frame). Done before `init_backend` so the GPU buffers are born
        // small and the RT acceleration structure (built over resident draws
        // inside init) sees the final offsets. Gated to the backends whose
        // `seed_mesh_streaming` seeds the sub-allocators with the headroom block.
        #[cfg(any(backend_metal, backend_dx, backend_vk))]
        let mut all_vertices = all_vertices;
        #[cfg(any(backend_metal, backend_dx, backend_vk))]
        let mut all_indices = all_indices;
        #[cfg(any(backend_metal, backend_dx, backend_vk))]
        let mut instanced_clusters = instanced_clusters;
        let mesh_seed_region: Option<crate::gfx::mesh_seed::MeshSeedRegion> = {
            #[cfg(any(backend_metal, backend_dx, backend_vk))]
            {
                match streaming_config.as_ref() {
                    Some(cfg) if !mesh_payloads.is_empty() => {
                        let sizes: Vec<(u64, u64)> = mesh_payloads
                            .iter()
                            .map(|m| {
                                (
                                    (m.vertices.len() * std::mem::size_of::<Vertex>()) as u64,
                                    (m.indices.len() * std::mem::size_of::<u32>()) as u64,
                                )
                            })
                            .collect();
                        match crate::gfx::mesh_seed::plan_seed_bytes(&sizes, cfg.mesh_cap()) {
                            Some((seed_vtx, seed_idx)) => {
                                let mut streamed = vec![false; draw_objects.len()];
                                for &idx in &mesh_stream_draw_indices {
                                    if let Some(s) = streamed.get_mut(idx) {
                                        *s = true;
                                    }
                                }
                                let region = crate::gfx::mesh_seed::compact_for_streaming(
                                    &mut all_vertices,
                                    &mut all_indices,
                                    &mut draw_objects,
                                    &mut instanced_clusters,
                                    &streamed,
                                    seed_vtx,
                                    seed_idx,
                                );
                                tracing::info!(
                                    "GraphicsSystem: shrinkable seed VRAM -- {} streamed mesh(es), cap {}, seed headroom {} KiB vtx + {} KiB idx",
                                    mesh_stream_draw_indices.len(),
                                    cfg.mesh_cap(),
                                    seed_vtx / 1024,
                                    seed_idx / 1024,
                                );
                                Some(region)
                            }
                            None => None,
                        }
                    }
                    _ => None,
                }
            }
            #[cfg(not(any(backend_metal, backend_dx, backend_vk)))]
            {
                None
            }
        };

        let draw_object_count = draw_objects.len();
        let cluster_count = instanced_clusters.len();
        let total_instances: usize = instanced_clusters.iter().map(|c| c.instances.len()).sum();

        // Build projected-decal records from the world's `Decal` components.
        // Resolved here (rather than per-frame) because the decal set is built
        // at init and never grows: each record carries the resolved texture
        // slot and pre-inverted model matrix the fragment shader needs. The
        // Decal components are drained because the runtime keeps no per-frame
        // update path for them.
        let decal_records = {
            let decals: Vec<Decal> = ctx.drain::<Decal>();
            let refs: Vec<&Decal> = decals.iter().collect();
            crate::gfx::decal::build_decal_records(&refs, texture_count)
        };
        let decal_count = decal_records.len();

        // Build particle-emitter records from the world's `ParticleEmitter`
        // components. Same drain-at-init pattern as decals: each record carries
        // the clamped emitter tunables and the resolved texture slot. The
        // backend allocates one persistent GPU pool per record at init.
        let particle_records = {
            let emitters: Vec<ParticleEmitter> = ctx.drain::<ParticleEmitter>();
            let refs: Vec<&ParticleEmitter> = emitters.iter().collect();
            crate::gfx::particles::build_particle_records(&refs, texture_count)
        };
        let particle_count = particle_records.len();

        // Drain transparent water surfaces. The Metal backend builds a
        // tessellated grid + per-surface uniforms per record at init;
        // DirectX / Vulkan ignore the slice for now.
        let water_surfaces: Vec<WaterSurface> = ctx.drain::<WaterSurface>();

        // Drain translucent glass panels. Every backend builds a world-space
        // quad + per-panel uniforms per record at init and draws them in the
        // shared transparent pass (`metal/glass.rs`, `directx/glass.rs`,
        // `vulkan/glass.rs`).
        let glass_panels: Vec<GlassPanel> = ctx.drain::<GlassPanel>();

        // Drain raymarched SDF volumes and pull the compiled-payload
        // fragment-shader source bytes for each. Each backend wraps the bytes
        // with the engine-shipped helpers + template and compiles a per-volume
        // pipeline at init. Volumes whose payload read fails are dropped with a
        // logged warning rather than failing the whole world build.
        let sdf_volumes: Vec<(SdfVolume, Vec<u8>, String)> = {
            let raw: Vec<SdfVolume> = ctx.drain::<SdfVolume>();
            let name_table = crate::ecs::asset_id::name_table();
            let mut out = Vec::with_capacity(raw.len());
            for v in raw {
                let asset_id = v.asset_id;
                let label = name_table
                    .get(asset_id.0 as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("sdf_volume_{}", asset_id.0));
                let locator = match v.locator.as_ref() {
                    Some(l) => l.clone(),
                    None => {
                        tracing::warn!(
                            "SdfVolume '{}': no payload locator (fragment shader \
                             never compiled); skipping",
                            label
                        );
                        continue;
                    }
                };
                match ctx.read_payload(&locator) {
                    Ok(bytes) => {
                        let owned = bytes.to_vec();
                        out.push((v, owned, label));
                    }
                    Err(e) => {
                        tracing::warn!(
                            "SdfVolume '{}': failed to read fragment shader \
                             payload: {:?}; skipping",
                            label,
                            e
                        );
                    }
                }
            }
            out
        };

        // Resolve the world's `VolumetricFog`. The first declared instance
        // wins; later ones are silently dropped (one homogeneous medium is
        // all the fog pass models). `None` means the renderer skips the
        // fog pass; an asset with `enabled = false` also yields `None`.
        let fog_settings = {
            let fogs: Vec<VolumetricFog> = ctx.drain::<VolumetricFog>();
            fogs.into_iter().find(|f| f.enabled).map(|f| {
                crate::gfx::volumetric_fog::FogSettings::resolve(
                    f.color,
                    f.density,
                    f.height_falloff,
                    f.height_reference,
                    f.max_distance,
                    f.phase_g,
                    f.ambient,
                )
            })
        };
        let fog_enabled = fog_settings.is_some();
        // Seed the hot-reload dedupe state. Subsequent reload_volumetric_fog
        // calls compare resolved JSONL settings against this and only push
        // (and log) on a real change.
        self.last_fog_settings = fog_settings;

        // The CLI `--validation` flag (via `dev_flags`) drives the DirectX /
        // Vulkan debug layers; unset falls back to the build profile. Metal is
        // unaffected here: its validation layer is enabled by the CLI re-execing
        // with `MTL_DEBUG_LAYER`, not through this flag.
        let validation = crate::app::dev_flags::validation().unwrap_or(cfg!(debug_assertions));
        // Shader hot-reload is opted in by `cn debug` (sets the static flag
        // in `crate::app::dev_flags` before world build). Production `cn run`
        // leaves it off; the backend then never spawns the filesystem watcher
        // and shader sources stay strictly include_str!-baked.
        let hot_reload = crate::app::dev_flags::enabled();
        // Worst-case resident chunk count for the streaming VoxelWorld (0 for a
        // non-voxel world). Threaded into the backend so its GPU-cull buffers
        // reserve a chunk record region at init; resident chunks fold into the
        // indirect path each frame. The VoxelWorld is consumed later by
        // `setup_voxel_world_streaming`, so borrow it here.
        let n_chunk_max = voxel_world.as_ref().map_or(
            0,
            crate::gfx::graphics_system::streaming::chunk_reserve_count,
        );

        // Reflection-probe auto-seed geometry. Computed here, before `draw_objects`
        // moves into the backend: when the world declares no `ReflectionProbe`, surface-
        // voxelise the static geometry so a watertight single-mesh interior is detected
        // (object AABBs alone would read it as a solid block). Budget-gated, so a heavy
        // import keeps coarse AABB occupancy. `None` -> the backend's own AABB auto-seed.
        let auto_seed_geometry_probes = if ctx
            .query::<crate::assets::ReflectionProbe>()
            .next()
            .is_some()
        {
            None
        } else {
            gather_auto_seed_triangles(&draw_objects, &all_vertices, &all_indices).and_then(
                |tris| {
                    let occupancy: Vec<([f32; 3], [f32; 3])> = draw_objects
                        .iter()
                        .map(|o| (o.bb_min, o.bb_max))
                        .filter(|(mn, mx)| mn.iter().chain(mx).all(|c| c.is_finite()))
                        .collect();
                    let (mn, mx) =
                        crate::gfx::reflection_probe::fold_world_bounds(occupancy.iter().copied())?;
                    Some(
                        crate::gfx::reflection_probe::auto_seed_probes_with_geometry(
                            mn, mx, &occupancy, &tris,
                        ),
                    )
                },
            )
        };

        // Planar reflection plane budget: there is no world-authored value, so the
        // engine capacity is the baseline, scaled down under the quality preset /
        // GPU tier ceiling. A lower tier renders fewer full render-res mirror passes
        // (VRAM + GPU savings); reflectors past the budget take the probe cube.
        // Restart-required like anisotropy above -- the mirror targets are allocated
        // once at backend init below.
        let planar_reflection_planes = quality_ceiling.planar_reflection_planes as usize;

        // Assemble the backend construction inputs, derive the world's render
        // requirements from them (a world with no 3D content drops every
        // scene-scoped feature before any backend resource is sized), and
        // hand the result to the compile-time-selected backend.
        use crate::gfx::backend_init::{
            BackendInit, MediaPayloads, PostSettings, SceneData, ShaderBytes, ShadowParams, WorldFx,
        };
        let mut backend_init = BackendInit {
            window: &self.window_args,
            validation,
            frames_in_flight: self.frames_in_flight,
            vsync: self.vsync,
            clear_color: self.clear_color,
            hot_reload,
            scene: SceneData {
                vertices: &all_vertices,
                indices: &all_indices,
                draw_objects,
                instanced_clusters,
                // Skinned draw-object count, to size the backend's GPU-cull
                // buffers for the merged total at init. The skinned geometry is
                // uploaded later by `upload_skinned` (which consumes
                // `skinned_draw_objects`).
                n_skinned: skinned_draw_objects.len(),
                n_chunk_max,
            },
            shaders: ShaderBytes {
                vert: &vert_bytes,
                frag: &frag_bytes,
                shadow: &shadow_bytes,
                vert_instanced: &vert_instanced_bytes,
            },
            media: MediaPayloads {
                textures: &texture_data,
                text_atlases: text_atlas_data,
                env_map_bytes: env_map_bytes.as_deref(),
                color_lut_bytes: color_lut_bytes.as_deref(),
            },
            light_uniforms,
            shadows: ShadowParams {
                map_size: self.shadow_map_size,
                update: self.shadow_update,
                distance: self.shadow_distance,
                cascades: self.shadow_cascades,
            },
            anisotropy: self.anisotropy,
            planar_planes: planar_reflection_planes,
            post: PostSettings {
                post_process,
                taa_enabled,
                ssao: ssao_settings,
                ssr: ssr_settings,
                ssgi: ssgi_settings,
                rt_reflections: rt_reflection_settings,
                reflection_blur_scale,
                auto_exposure: auto_exposure_settings,
                auto_exposure_bias_ev,
                hdr_display,
                hdr_pq,
                temporal_upscaling,
                upscale_scale,
                upscale_backend,
                occlusion_two_pass,
            },
            fx: WorldFx {
                decals: decal_records,
                particles: particle_records,
                fog: fog_settings,
                water_surfaces,
                glass_panels,
                sdf_volumes,
            },
            requirements: Default::default(),
        };
        backend_init.resolve_requirements();
        // Live editor swap: the rebuilt world may carry a render backend
        // transplanted out of the pre-edit world (a `PendingBackend` resource,
        // published by the `cn editor` live SAVE). Reuse it instead of building a
        // new one -- so the edit applies without recreating the OS window or
        // re-initialising the GPU device -- but only when the backend can hot-swap
        // AND the swapchain-level config (pixel format / frames-in-flight / EDR)
        // is unchanged. An incapable backend (DirectX / Vulkan, whose default
        // `hot_swap_config` is `None`) or a swapchain change routes to a full
        // rebuild instead: the transplanted backend is idled + dropped and a new
        // window is created (a rare one-frame flash). Deciding via `hot_swap_config`
        // up front leaves `backend_init` intact for that rebuild path.
        let reuse_backend = match ctx
            .resources
            .remove::<crate::ecs::PendingBackend>()
            .map(|p| p.0)
        {
            Some(backend) if backend.hot_swap_config() == Some(backend_init.swapchain_config()) => {
                Some(backend)
            }
            Some(backend) => {
                backend.wait_idle();
                None
            }
            None => None,
        };
        // Tests inject a mock backend factory through `test_hooks`; production
        // always routes to the compile-time-selected real backend. Inline (not
        // a method) because `backend_init` still borrows `self.window_args`.
        #[cfg(test)]
        let built = match reuse_backend {
            Some(mut backend) => match backend.reload_world(backend_init) {
                Ok(()) => {
                    tracing::info!(
                        "GraphicsSystem: reused live backend (world reloaded in place, window kept)"
                    );
                    Some(backend)
                }
                Err(e) => {
                    tracing::error!("GraphicsSystem: reload_world failed: {e}");
                    None
                }
            },
            None => match self.test_hooks.as_mut() {
                Some(hooks) => (hooks.backend_factory)(backend_init),
                None => concinnity_device::init_backend(backend_init),
            },
        };
        #[cfg(not(test))]
        let built = match reuse_backend {
            Some(mut backend) => match backend.reload_world(backend_init) {
                Ok(()) => {
                    tracing::info!(
                        "GraphicsSystem: reused live backend (world reloaded in place, window kept)"
                    );
                    Some(backend)
                }
                Err(e) => {
                    tracing::error!("GraphicsSystem: reload_world failed: {e}");
                    None
                }
            },
            None => concinnity_device::init_backend(backend_init),
        };
        self.backend = built;

        if self.backend.is_none() {
            self.failed = true;
            return;
        }

        // Apply a persisted or authored non-windowed window mode at startup. The
        // window is always created as a standard titled window, so a Borderless
        // or Fullscreen choice (set in the settings menu and persisted across
        // launches) has to be applied here; otherwise the app would always start
        // windowed regardless of the saved mode. No-op for Windowed and in
        // embedded mode (the backend owns no window there).
        if self.window_args.mode != crate::assets::WindowMode::Windowed
            && let Some(backend) = self.backend.as_deref_mut()
        {
            backend.set_window_mode(self.window_args.mode);
        }

        // The Resolution row's mode list: the display's enumerated modes when
        // the backend can provide them, else the static preset fallback. The
        // list is published once for UiInputSystem to seed the row's dropdown.
        // A persisted mode choice is handed to the backend, which holds the
        // display to it whenever the window is in fullscreen; with no choice
        // the row shows the display's own mode. The row's value label is
        // dynamic (built from the list, not the static registry), so it is set
        // here rather than in the generic sync above.
        let chosen = self.resolution;
        if let Some(backend) = self.backend.as_deref_mut() {
            let raw = backend.display_modes();
            self.display_modes = if raw.is_empty() {
                crate::gfx::display_mode::fallback_modes()
            } else {
                crate::gfx::display_mode::normalize(raw)
            };
            self.current_mode = backend.current_display_mode();
            if let Some(mode) = chosen {
                backend.set_display_mode(mode);
            }
        }
        ctx.insert_resource(crate::ecs::DisplayModes(self.display_modes.clone()));
        // The resolved frame-rate cap (world value or persisted override) for
        // the App-level pacer; the settings row's live change republishes it.
        ctx.insert_resource(crate::ecs::FrameRateCap(self.fps_cap));
        let idx =
            crate::gfx::display_mode::index_of(&self.display_modes, self.effective_resolution());
        if let Some(m) = self.display_modes.get(idx) {
            set_setting_row_label(ctx, "resolution", &m.label());
        }

        // Reflection probes: hand the backend the declared `ReflectionProbe`
        // placements (Metal bakes a cube per probe; an empty list auto-seeds from
        // the scene bounds). Pushed once here, after construction; DX/VK no-op.
        if let Some(backend) = self.backend.as_deref_mut() {
            let declared: Vec<crate::gfx::reflection_probe::ProbePlacement> = ctx
                .drain::<crate::assets::ReflectionProbe>()
                .into_iter()
                .map(|p| {
                    crate::gfx::reflection_probe::ProbePlacement::from_center_extents(
                        p.position,
                        p.half_extents,
                    )
                })
                .collect();
            // Declared probes win; otherwise the geometry-aware auto-seed (when the scene
            // was small enough to gather); otherwise an empty list, which lets the backend
            // run its own coarse-AABB auto-seed (the unchanged path for heavy imports).
            let placements = if !declared.is_empty() {
                declared
            } else {
                auto_seed_geometry_probes.unwrap_or_default()
            };
            backend.set_reflection_probes(&placements);
        }

        // World.jsonl path for the Prop transform reload pass. The dev host
        // (`cn debug` / `cn editor`) resolves the world path -- world.jsonl
        // discovery is authoring I/O in `concinnity-cook`, which the runtime does
        // not link -- and hands it in via `dev_flags`. Embedded preview / WS-
        // driven runs leave it None and the file watcher has no `.jsonl` to
        // subscribe to.
        let world_jsonl_path: Option<String> = if capture_sources {
            crate::app::dev_flags::world_jsonl_path()
        } else {
            None
        };

        // Asset hot-reload state. Built only when `cn debug` opted in
        // (`capture_sources`) and the world declared at least one file-backed
        // asset (texture, ColorLut, EnvironmentMap, Mesh, SkinnedMesh, or
        // world.jsonl). The constructor spawns a `notify` watcher over the
        // parent directories of every captured source path; `step` polls the
        // shared atomic at frame start.
        if capture_sources
            && (!asset_source_map.is_empty()
                || color_lut_source.is_some()
                || environment_map_source.is_some()
                || !mesh_source_map.is_empty()
                || !skinned_mesh_source_map.is_empty()
                || !procedural_mesh_source_map.is_empty()
                || !shader_stage_source_map.is_empty()
                || world_jsonl_path.is_some())
        {
            tracing::info!(
                "asset hot-reload: captured {} file-backed texture source(s), {} \
                 ColorLut source(s), {} EnvironmentMap source(s), {} Mesh \
                 source(s), {} SkinnedMesh source(s), {} ProceduralMesh source(s), \
                 {} ShaderStage source(s), and world.jsonl path = {:?}",
                asset_source_map.len(),
                color_lut_source.as_ref().map(|_| 1).unwrap_or(0),
                environment_map_source.as_ref().map(|_| 1).unwrap_or(0),
                mesh_source_map.len(),
                skinned_mesh_source_map.len(),
                procedural_mesh_source_map.len(),
                shader_stage_source_map.len(),
                world_jsonl_path
            );
            self.pending_hot_reload_sources = Some(super::hot_reload_sources::HotReloadSources {
                map: asset_source_map,
                color_lut: color_lut_source,
                environment_map: environment_map_source,
                meshes: mesh_source_map,
                skinned_meshes: skinned_mesh_source_map,
                procedural_meshes: procedural_mesh_source_map,
                shader_stages: shader_stage_source_map,
                world_jsonl_path,
            });
            // The texture-name -> slot map for runtime decal / emitter spawn
            // (`cn debug`), which resolves a Texture asset name to its live pool
            // slot. Captured only when hot-reload is on, so a `cn run` skips the
            // clone cost.
            self.world_reload = Some(super::WorldReloadState {
                texture_name_to_slot: texture_name_to_slot.clone(),
            });
        }

        // Upload skinned geometry to the backend and publish one SkeletonPose
        // per skinned mesh for AnimationSystem to drive. The poses are published
        // regardless of backend so the system graph is identical.
        if !skinned_skeletons.is_empty() {
            if let Some(backend) = self.backend.as_deref_mut() {
                // Metal uses `vert_bytes` + `frag_bytes` and sources the shadow
                // shader internally; `shadow_bytes` is empty (engine-internal
                // shadow). DX/VK compile their vertex/shadow paths inline.
                if let Err(e) = backend.upload_skinned(
                    &skinned_vertices,
                    &skinned_indices,
                    std::mem::take(&mut skinned_draw_objects),
                    &vert_bytes,
                    &frag_bytes,
                    &shadow_bytes,
                ) {
                    tracing::error!("GraphicsSystem: skinned geometry upload failed: {}", e);
                    self.failed = true;
                    return;
                }
                // Seed the backend's skinned instance pool with the hidden copies
                // reserved above, so a runtime skinned spawn can claim one.
                backend.seed_skinned_instance_pool(std::mem::take(&mut skinned_pool_reservations));
            }
            let skinned_count = skinned_skeletons.len();
            for (handle, name_id, template_index, skeleton, model, capsule) in skinned_skeletons {
                let entity = ctx.components.spawn();
                ctx.insert(
                    entity,
                    crate::assets::SkeletonPose::new(handle, template_index, skeleton),
                );
                // Register the template under its mesh name so a runtime
                // SpawnRequest can resolve it to this entity, the same way the
                // static spawn path resolves a named placement. The spawn then
                // clones this template's skeleton + pose into a pooled slot.
                if let Some(by_name) = ctx.resource_mut::<crate::ecs::decompose::EntityByName>() {
                    by_name.0.insert(name_id, entity);
                }
                // A mesh with a capsule gets a character rig: PhysicsSystem
                // (init runs later this tick) creates the kinematic capsule
                // from it, and the render transform follows it each frame.
                if let Some(capsule) = capsule {
                    ctx.push(crate::assets::CharacterRig::new(
                        handle,
                        template_index,
                        model,
                        capsule.half_height.max(0.05),
                        capsule.radius.max(0.05),
                    ));
                }
            }
            tracing::info!("GraphicsSystem: {} skinned mesh(es) ready", skinned_count);
        }

        self.setup_texture_streaming(
            streaming_config.clone(),
            texture_payloads,
            &texture_locators,
            blob_disk_backed,
            texture_centers,
        );
        self.setup_mesh_streaming(
            streaming_config,
            mesh_payloads,
            mesh_centers,
            mesh_stream_draw_indices,
            blob_disk_backed,
            mesh_seed_region,
        );
        self.setup_voxel_world_streaming(voxel_world, &block_types, &material_map);

        // Decide cursor handling. A plain first-person world (Camera3D, no UI)
        // captures the cursor at startup as before. A Camera3D world that also
        // has UI (a MainMenu's HitRegion / KeyBinding) is "menu mode": capture
        // is driven per-frame in `run_step` by whether a menu view is active,
        // so Escape can pause the camera into the menu and back. A UI-only
        // world (no camera) stays free-cursor.
        let has_ui = ctx.query::<HitRegion>().next().is_some()
            || ctx.query::<crate::assets::KeyBinding>().next().is_some();
        let has_camera = ctx.query::<Camera3D>().next().is_some();
        self.menu_mode = has_camera && has_ui;
        let mut device_caps = crate::gfx::backend::DeviceCapabilities::ALL;
        if let Some(backend) = self.backend.as_deref_mut() {
            // Capability flags drive the settings-menu gating below.
            device_caps = backend.capabilities();
            // Detected GPU performance profile, logged once at init so the
            // classified tier is verifiable on each device.
            let gpu = backend.gpu_profile();
            tracing::info!(
                "GPU profile: vendor={:?} tier={:?} memory_budget={} MB unified={} discrete={}",
                gpu.vendor,
                gpu.tier,
                gpu.memory_budget_bytes / (1 << 20),
                gpu.unified_memory,
                gpu.discrete,
            );
            backend.set_menu_mode(self.menu_mode);
            // Push the effective ambient scale (world value or persisted
            // override). The backend already seeds the world value at its own
            // init, so this is the path that applies a persisted Ambient-slider
            // choice; idempotent when there is no override. No-op on DX/VK.
            backend.set_ambient_intensity(self.ambient_intensity);
            // Push the movement key map (the persisted rebinds, or the default).
            // The backend decodes physical keys through it; idempotent with its
            // own default seed when there is no override.
            backend.set_keymap(&self.keymap);
            if has_camera && !has_ui {
                backend.capture_cursor();
            }
        }
        self.caps = device_caps;
        // Gray out + disable settings rows whose feature the device cannot
        // provide (e.g. ray-traced reflections on a GPU without hardware ray
        // tracing). Runs while the menu HitRegions / TextLabels / ScrollPanels
        // are still present (GraphicsSystem.init runs before UiInputSystem drains
        // them); the value-label sync above already set each row's live value.
        self.apply_capability_gating(ctx);

        self.setup_scene_reel(ctx);

        // Hand the overlay build inputs assembled above (font atlases, sprite
        // slots, HUD chip ids, clip bands) to OverlaySystem, which shapes the
        // draw list from them each frame before this system submits it.
        ctx.insert_resource(crate::gfx::overlay::OverlayAssets {
            fonts: std::mem::take(&mut self.loaded_fonts),
            sprite_texture_slots: std::mem::take(&mut self.sprite_texture_slots),
            debug_hud_chips: std::mem::take(&mut self.debug_hud_chips),
            stat_hud_chips: std::mem::take(&mut self.stat_hud_chips),
            clip_rects: std::mem::take(&mut self.clip_rects),
            initial_viewport: self
                .backend
                .as_ref()
                .map(|b| b.logical_size())
                .unwrap_or((0.0, 0.0)),
        });

        // Hand the resolved settings snapshot to SettingsSystem, which owns the
        // live SettingCommand / SceneCommand drain against the backend. This
        // system resolves the values (world config + persisted overrides +
        // device capabilities) at init; SettingsState owns them from here (the
        // row bookkeeping is moved out, the render config is copied -- this
        // system never re-reads its copies after init).
        ctx.insert_resource(crate::gfx::settings_system::SettingsState {
            keymap: self.keymap,
            rebind_rows: std::mem::take(&mut self.rebind_rows),
            sliders: std::mem::take(&mut self.sliders),
            cycle_value_labels: std::mem::take(&mut self.cycle_value_labels),
            post_process: self.post_process,
            post_config: self.post_config.clone(),
            authored_post_config: self.authored_post_config.clone(),
            ambient_intensity: self.ambient_intensity,
            quality_preset: self.quality_preset,
            gpu_profile: self.gpu_profile,
            render_scale: self.render_scale,
            upscale_backend: self.upscale_backend,
            temporal_upscaling: self.temporal_upscaling,
            hdr_display: self.hdr_display,
            hdr_pq: self.hdr_pq,
            shadow_map_size: self.shadow_map_size,
            shadow_update: self.shadow_update,
            shadow_distance: self.shadow_distance,
            shadow_cascades: self.shadow_cascades,
            anisotropy: self.anisotropy,
            authored_shadow_map_size: self.authored_shadow_map_size,
            authored_shadow_update: self.authored_shadow_update,
            authored_shadow_distance: self.authored_shadow_distance,
            authored_shadow_cascades: self.authored_shadow_cascades,
            authored_anisotropy: self.authored_anisotropy,
            vsync: self.vsync,
            fps_cap: self.fps_cap,
            perf_stats: self.perf_stats,
            show_fps: self.show_fps,
            show_vram: self.show_vram,
            perf_sub_row_labels: std::mem::take(&mut self.perf_sub_row_labels),
            window_args: self.window_args.clone(),
            display_modes: std::mem::take(&mut self.display_modes),
            resolution: self.resolution,
            current_mode: self.current_mode,
            resolution_row_labels: std::mem::take(&mut self.resolution_row_labels),
            frames_in_flight: self.frames_in_flight,
            occlusion_two_pass: self.occlusion_two_pass,
            texture_cap: self.texture_cap,
            texture_budget: self.texture_budget,
            settings_cache: None,
            settings_writer: None,
            scene_cmd_cursor: crate::ecs::EventCursor::default(),
            setting_cmd_cursor: crate::ecs::EventCursor::default(),
        });

        // Hand the overlay build inputs assembled above (font atlases, sprite
        // slots, HUD chip ids, clip bands) to OverlaySystem, which shapes the
        // draw list from them each frame before this system submits it.
        ctx.insert_resource(crate::gfx::overlay::OverlayAssets {
            fonts: std::mem::take(&mut self.loaded_fonts),
            sprite_texture_slots: std::mem::take(&mut self.sprite_texture_slots),
            debug_hud_chips: std::mem::take(&mut self.debug_hud_chips),
            stat_hud_chips: std::mem::take(&mut self.stat_hud_chips),
            clip_rects: std::mem::take(&mut self.clip_rects),
            initial_viewport: self
                .backend
                .as_ref()
                .map(|b| b.logical_size())
                .unwrap_or((0.0, 0.0)),
        });

        // Hand the streaming pools built above to StreamingSystem: it drives
        // them each frame (against the parked backend) and publishes the
        // camera-relative view GraphicsSystem draws with. `frame_count` starts
        // at 0 in lockstep with this system's own frame clock (both tick once
        // per world step), so eviction retire-frames match the draw's frame.
        ctx.insert_resource(crate::gfx::streaming_system::StreamingState {
            texture_streamer: self.texture_streamer.take(),
            mesh_streamer: self.mesh_streamer.take(),
            mesh_stream_draw_indices: std::mem::take(&mut self.mesh_stream_draw_indices),
            chunk_stream: self.chunk_stream.take(),
            frame_count: 0,
            frames_in_flight: self.frames_in_flight,
        });

        // Init-time wiring is done: park the backend in the world's shared
        // slot, where each per-step user (this system's frame encode,
        // InputSystem's poll) takes and returns it.
        ctx.insert_resource(crate::ecs::ActiveRenderBackend(self.backend.take()));

        let start = Instant::now();
        self.start_time = Some(start);
        // Hand the SceneReel to the shared slot SettingsSystem jumps and this
        // system ticks. `epoch` shares this system's start clock so a jump's
        // fade timing matches the render clock.
        ctx.insert_resource(crate::ecs::ActiveSceneReel {
            reel: self.reel.take(),
            epoch: start,
        });
        tracing::info!(
            "GraphicsSystem: ready ({}x{} \"{}\", {} frames in flight, {} draw objects, {} instanced clusters ({} instances total), {} decals, {} particle emitter(s), fog={})",
            self.window_args.width,
            self.window_args.height,
            self.window_args.title,
            self.frames_in_flight,
            draw_object_count,
            cluster_count,
            total_instances,
            decal_count,
            particle_count,
            if fog_enabled { "on" } else { "off" },
        );
    }
}

// Set the value TextLabel of every `setting:<key>` HitRegion to the live value
// of that setting. `current_index` maps a setting key to the index of its
// active option (None for an unknown key). Runs once at init, before any
// system drains the HitRegions.
fn sync_setting_value_labels(
    ctx: &mut PipelineContext,
    current_index: impl Fn(&str) -> Option<usize>,
) {
    // (setting key, value-label id) for each settings row.
    let rows: Vec<(String, AssetId)> = ctx
        .query::<HitRegion>()
        .filter_map(|r| {
            let rest = r.action.strip_prefix("setting:")?;
            let key = rest.split(':').next()?;
            Some((key.to_string(), r.label?))
        })
        .collect();

    for (key, label_id) in rows {
        let (Some(opts), Some(idx)) = (crate::gfx::settings::options(&key), current_index(&key))
        else {
            continue;
        };
        if let Some(text) = opts.get(idx).copied() {
            for l in ctx.query_mut::<TextLabel>() {
                if l.asset_id == label_id {
                    l.content = text.to_string();
                    break;
                }
            }
        }
    }
}

// Set the value label of the settings row bound to `key` to `text` directly,
// for a label that is not one of the row's static `options` (the master preset
// row's "Auto (High)", or the live "Custom" flip when a quality row changes).
fn set_setting_row_label(ctx: &mut PipelineContext, key: &str, text: &str) {
    let label_id = ctx.query::<HitRegion>().find_map(|r| {
        let row_key = r.action.strip_prefix("setting:")?.split(':').next()?;
        (row_key == key).then_some(r.label).flatten()
    });
    if let Some(id) = label_id {
        for l in ctx.query_mut::<TextLabel>() {
            if l.asset_id == id {
                l.content = text.to_string();
                break;
            }
        }
    }
}

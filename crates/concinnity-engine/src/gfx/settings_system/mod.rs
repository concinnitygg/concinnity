// src/gfx/settings_system/mod.rs
//
// SettingsSystem: applies the runtime command batches UiInputSystem produces
// -- SettingCommand (settings-menu changes: graphics toggles, sliders, key
// rebinds, volume) and SceneCommand (imperative scene jumps) -- recording
// their backend effects into the frame's op queue, owns the in-memory
// settings snapshot + the background disk writer, and publishes the per-frame
// HUD-preference state:
//   mod.rs    system + state + scene jumps + HUD-state publish
//   apply.rs  the SettingCommand drain (one arm per settings row)
//   rows.rs   row helpers shared with GraphicsSystem's init-time captures
//   writer.rs background disk writer for settings changes
//
// Scheduled after SpawnSystem and before GraphicsSystem, so a change's
// recorded op lands on the backend before this frame's draw (visible the same
// frame, as it was when the drain called the backend directly) and the
// HUD-preference resources are fresh for StatHud / UiInput later this tick.
// The state is resolved by GraphicsSystem's init (world config + persisted
// overrides + backend capabilities) and parked here as the `SettingsState`
// resource; each step takes it and puts it back, so the state and the
// `PipelineContext` are never borrowed together.

use crate::assets::SceneCommand;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};
use crate::gfx::ops::RenderOps;
use crate::gfx::scene_flow;

mod apply;
pub(crate) mod rows;
#[cfg(test)]
mod tests;
pub(crate) mod writer;

// The live settings state: every value the settings menu displays and cycles,
// with the authored baselines a preset change re-clamps from, the row
// bookkeeping captured at init, and the persistence machinery. Field meanings
// match the settings-menu rows they back; see the drain arms in `apply.rs`.
pub(crate) struct SettingsState {
    // Live gameplay movement key map (the source of truth for the Controls-tab
    // rebind rows), pushed to the backend on each rebind (with a swap).
    pub(crate) keymap: crate::gfx::keymap::KeyMap,
    pub(crate) rebind_rows: Vec<crate::gfx::graphics_system::RebindViz>,
    // Live gamepad action -> button map (the source of truth for the gamepad
    // rebind rows), carried to InputSystem via ControlsCommand on each rebind.
    pub(crate) gamepad_map: crate::assets::GamepadMap,
    pub(crate) pad_rebind_rows: Vec<crate::gfx::graphics_system::PadRebindViz>,
    pub(crate) sliders: Vec<crate::gfx::graphics_system::SliderViz>,
    // Cycle rows' setting key -> value-label id, captured at init, so a change
    // can relabel a row other than the one clicked.
    pub(crate) cycle_value_labels: std::collections::HashMap<String, AssetId>,
    // Live post-process parameters (bloom / exposure / vignette / LUT blend),
    // the source of truth for slider settings.
    pub(crate) post_process: crate::gfx::render_types::PostProcessParams,
    // The world's resolved PostProcessConfig with the user's persisted
    // quality-toggle overrides applied: the source of truth for the
    // Quality-group toggles and cycle knobs.
    pub(crate) post_config: crate::assets::PostProcessConfig,
    // The authored baseline a live preset change re-clamps from.
    pub(crate) authored_post_config: crate::assets::PostProcessConfig,
    // Live ambient (IBL) light scale (lives in the backend's LightUniforms,
    // so it takes a dedicated setter).
    pub(crate) ambient_intensity: f32,
    // The live master "Graphics Quality" preset; an explicit per-row change
    // flips it to Custom.
    pub(crate) quality_preset: crate::gfx::quality_preset::QualityPreset,
    pub(crate) gpu_profile: crate::gfx::backend::GpuProfile,
    // Restart-required display state (persist + relabel only).
    pub(crate) render_scale: crate::assets::UpscaleQuality,
    pub(crate) upscale_backend: crate::assets::UpscalerBackend,
    pub(crate) temporal_upscaling: bool,
    pub(crate) hdr_display: bool,
    pub(crate) hdr_pq: bool,
    // Shadow knobs (live) and their authored baselines.
    pub(crate) shadow_map_size: u32,
    pub(crate) shadow_update: crate::assets::ShadowUpdate,
    pub(crate) shadow_distance: u32,
    pub(crate) shadow_cascades: u32,
    pub(crate) anisotropy: u32,
    pub(crate) authored_shadow_map_size: u32,
    pub(crate) authored_shadow_update: crate::assets::ShadowUpdate,
    pub(crate) authored_shadow_distance: u32,
    pub(crate) authored_shadow_cascades: u32,
    pub(crate) authored_anisotropy: u32,
    pub(crate) vsync: bool,
    // Frame-rate cap; a change republishes the `FrameRateCap` resource the
    // App-level pacer reads.
    pub(crate) fps_cap: u32,
    // Stats-HUD display toggles + the captured sub-row labels the master
    // toggle grays.
    pub(crate) perf_stats: bool,
    pub(crate) show_fps: bool,
    pub(crate) show_vram: bool,
    pub(crate) perf_sub_row_labels: Vec<(AssetId, [f32; 3])>,
    // Window mode + authored size (the windowed size restored on mode return).
    pub(crate) window_args: crate::assets::Window,
    // The Resolution row's mode list, the user's chosen fullscreen mode, the
    // display's own mode at init, and the row labels grayed outside
    // fullscreen.
    pub(crate) display_modes: Vec<crate::gfx::display_mode::DisplayMode>,
    pub(crate) resolution: Option<crate::gfx::display_mode::DisplayMode>,
    pub(crate) current_mode: Option<crate::gfx::display_mode::DisplayMode>,
    pub(crate) resolution_row_labels: Vec<(AssetId, [f32; 3])>,
    // System / streaming restart rows (persist + display only).
    pub(crate) frames_in_flight: usize,
    pub(crate) occlusion_two_pass: bool,
    pub(crate) texture_cap: u32,
    pub(crate) texture_budget: u32,
    // In-memory copy of the persisted settings store, loaded once on the first
    // settings change and mutated in place from then on, so a queued (not yet
    // flushed) background write is never re-read stale from disk.
    pub(crate) settings_cache: Option<crate::config::Settings>,
    // Background disk writer for settings changes; spawned on the first
    // persisted change so an unchanged session never starts the thread.
    pub(crate) settings_writer: Option<writer::SettingsWriter>,
    // Cursors into the SceneCommand / SettingCommand queues.
    pub(crate) scene_cmd_cursor: crate::ecs::EventCursor,
    pub(crate) setting_cmd_cursor: crate::ecs::EventCursor,
    // Last-published HUD state, so `publish_hud_state` only re-inserts the
    // resources (rebuilding the disabled-rows set) when the inputs actually
    // change rather than every frame. `None` until the first publish.
    pub(crate) published_hud_prefs: Option<crate::ecs::HudPrefs>,
    pub(crate) published_disabled_inputs: Option<(bool, bool)>,
}

#[derive(Debug, Default)]
pub struct SettingsSystem;

impl SettingsSystem {
    pub fn new() -> Self {
        Self
    }

    // Park the state back in its slot at the end of a step that took it.
    fn park(ctx: &mut PipelineContext, state: SettingsState) {
        match ctx.resources.get_mut::<SettingsSlot>() {
            Some(slot) => slot.0 = Some(state),
            None => {
                ctx.resources.insert(SettingsSlot(Some(state)));
            }
        }
    }
}

// The parked `SettingsState` slot: each step takes the value out (so `ctx`
// stays freely borrowable) and puts it back, reusing the slot's allocation.
// `None` only while a step has it taken.
pub(crate) struct SettingsSlot(pub(crate) Option<SettingsState>);

impl std::fmt::Debug for SettingsState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SettingsState")
            .field("quality_preset", &self.quality_preset)
            .finish()
    }
}

impl System for SettingsSystem {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // No parked state (graphics init has not succeeded) or op queue:
        // nothing to apply against; the queued commands wait in retention.
        let Some(mut state) = ctx
            .resources
            .get_mut::<SettingsSlot>()
            .and_then(|slot| slot.0.take())
        else {
            return StepResult::Continue;
        };
        let Some(mut queues) = crate::ecs::ActiveRenderQueues::take(ctx.resources) else {
            Self::park(ctx, state);
            return StepResult::Continue;
        };
        state.apply_scene_commands(ctx, &mut queues.ops);
        state.apply_setting_commands(ctx, &mut queues.ops);
        crate::ecs::ActiveRenderQueues::put(ctx.resources, queues);
        state.publish_hud_state(ctx);
        Self::park(ctx, state);
        StepResult::Continue
    }
}

impl SettingsState {
    // Apply any imperative scene jumps sent by UiInputSystem last tick, copied
    // out of the event queue so the borrow is released before the jump runs.
    // The flow lives in the shared `ActiveSceneFlow` resource (GraphicsSystem
    // ticks its fades later this same tick, so a jump's fade starts on this
    // frame); the jump's fade / visibility effects are recorded through the
    // same `SceneOp` recorder the fade tick uses, then queued as one op.
    fn apply_scene_commands(&mut self, ctx: &mut PipelineContext, ops: &mut RenderOps) {
        let scene_cmds: Vec<SceneCommand> = match ctx.events::<SceneCommand>() {
            Some(events) => events.read(&mut self.scene_cmd_cursor).cloned().collect(),
            None => Vec::new(),
        };
        if scene_cmds.is_empty() {
            return;
        }
        // Source scene-jump visibility from the per-entity components,
        // snapshotting once for the whole command batch (jumps are rare edges,
        // so a local snapshot is fine here).
        let mut scratch = crate::gfx::graphics_system::scene::SceneVisibilityScratch::default();
        crate::gfx::graphics_system::scene::refresh_visibility_snapshot(ctx, &mut scratch);
        let Some(slot) = ctx.resources.get_mut::<crate::ecs::ActiveSceneFlow>() else {
            return;
        };
        let elapsed = slot.epoch.elapsed().as_secs_f32();
        let mut scene_ops: Vec<crate::gfx::snapshot::SceneOp> = Vec::new();
        for cmd in scene_cmds {
            let mut recorder = crate::gfx::snapshot::SceneOpRecorder(&mut scene_ops);
            scene_flow::jump_to_scene(
                &mut slot.flow,
                &scratch.visibility,
                elapsed,
                cmd.scene,
                &cmd.transition,
                &mut recorder,
            );
        }
        if !scene_ops.is_empty() {
            ops.record(move |backend| {
                for op in scene_ops {
                    match op {
                        crate::gfx::snapshot::SceneOp::SetFade(fade) => backend.set_fade(fade),
                        crate::gfx::snapshot::SceneOp::Visibility { draw_idx, visible } => {
                            backend.update_visibility(draw_idx, visible)
                        }
                    }
                }
            });
        }
    }

    // Publish the stats-HUD state for the systems that run after this one each
    // tick. Done AFTER the settings drain so a "Display performance stats"
    // toggle this frame is reflected the same frame -- the visibility
    // (StatHudSystem reads HudPrefs) and the inert/grayed sub-rows
    // (UiInputSystem reads DisabledSettingRows) stay in lockstep with the
    // gray-out applied in the drain.
    fn publish_hud_state(&mut self, ctx: &mut PipelineContext) {
        // Both resources are pure functions of a few settings fields and persist
        // in the resource map once inserted, so republish only when the inputs
        // change -- steady-state frames skip the HashSet + String allocations the
        // disabled-rows set would otherwise churn every frame.
        let prefs = crate::ecs::HudPrefs {
            show_fps: self.perf_stats && self.show_fps,
            show_vram: self.perf_stats && self.show_vram,
        };
        if self.published_hud_prefs != Some(prefs) {
            ctx.insert_resource(prefs);
            self.published_hud_prefs = Some(prefs);
        }

        // The Resolution row only applies in fullscreen (windowed sizes come from
        // the window, borderless covers the display), so it is inert in the other
        // modes. The disabled-rows set is fully determined by these two inputs.
        let is_fullscreen = self.window_args.mode == crate::assets::WindowMode::Fullscreen;
        let inputs = (self.perf_stats, is_fullscreen);
        if self.published_disabled_inputs != Some(inputs) {
            let mut disabled_rows: std::collections::HashSet<String> = if self.perf_stats {
                std::collections::HashSet::new()
            } else {
                ["show_fps", "show_vram"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            };
            if !is_fullscreen {
                disabled_rows.insert("resolution".to_string());
            }
            ctx.insert_resource(crate::ecs::DisabledSettingRows(disabled_rows));
            self.published_disabled_inputs = Some(inputs);
        }
    }
}

//! SettingsSystem: applies the runtime command batches UiInputSystem produces
//! -- SettingCommand (settings-menu changes: graphics toggles, sliders, key
//! rebinds, volume) and SceneCommand (imperative scene jumps) -- recording
//! their backend effects into the frame's op queue, owns the in-memory
//! settings snapshot + the background disk writer, and publishes the per-frame
//! HUD-preference state:
//!   mod.rs      system + state + scene jumps + HUD-state publish
//!   apply.rs    the SettingCommand drain and its per-family row handlers
//!   quality.rs  the quality preset, feature toggle, and quality knob rows
//!   rebind.rs   the key and gamepad rebind rows
//!   rows.rs     row helpers shared with GraphicsSystem's init-time captures
//!   writer.rs   background disk writer for settings changes
//!
//! Scheduled after SpawnSystem and before GraphicsSystem, so a change's
//! recorded op lands on the backend before this frame's draw (visible the same
//! frame, as it was when the drain called the backend directly) and the
//! HUD-preference resources are fresh for UiInput later this tick.
//! The state is resolved by GraphicsSystem's init (world config + persisted
//! overrides + backend capabilities) and parked here as the `SettingsState`
//! resource; each step takes it and puts it back, so the state and the
//! `PipelineContext` are never borrowed together.

use concinnity_core::components::GamepadMap;
use concinnity_core::components::PlanetDetail;
use concinnity_core::components::SceneCommand;
use concinnity_core::components::Window;
use concinnity_core::components::WindowMode;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{EventCursor, HudPrefs, PipelineContext, Resources, StepResult, System};
use concinnity_core::input::keymap;
use concinnity_core::render::backend;
use concinnity_core::render::error::RenderError;
use concinnity_core::render::ops::RenderOps;
use concinnity_core::render::scene_flow;
use concinnity_core::render::snapshot;
use concinnity_core::settings::apply::{ApplyPlan, plan_apply};
use concinnity_core::settings::quality::resolve_ceiling;
use concinnity_core::settings::{
    QualityCeiling, QualityPreset, ResolvedGraphics, ResolvedSettings, SettingKey, SettingsLayers,
    UserGraphicsSettings, WorldSettings,
};
use concinnity_core::window::display_mode;

mod apply;
mod quality;
mod rebind;
pub(crate) mod rows;
#[cfg(test)]
mod tests;
pub(crate) mod writer;

// The live settings state: every value the settings menu displays and cycles,
// with the authored baselines a preset change re-clamps from, the row
// bookkeeping captured at init, and the persistence machinery. Field meanings
// match the settings-menu rows they back; see the row handlers in `apply.rs`
// and its siblings.
pub(crate) struct SettingsState {
    // Live gameplay movement key map (the source of truth for the Controls-tab
    // rebind rows), pushed to the backend on each rebind (with a swap).
    pub(crate) keymap: keymap::KeyMap,
    pub(crate) rebind_rows: Vec<crate::gfx::system::RebindViz>,
    // Live gamepad action -> button map (the source of truth for the gamepad
    // rebind rows), carried to InputSystem via ControlsCommand on each rebind.
    pub(crate) gamepad_map: GamepadMap,
    pub(crate) pad_rebind_rows: Vec<crate::gfx::system::PadRebindViz>,
    pub(crate) sliders: Vec<crate::gfx::system::SliderViz>,
    // Cycle rows' setting key -> value-label id, captured at init, so a change
    // can relabel a row other than the one clicked.
    pub(crate) cycle_value_labels: std::collections::HashMap<SettingKey, AssetId>,
    // The world's authored graphics values a preset change or authoring edit
    // re-resolves from, and the live resolved values the rows display and cycle.
    pub(crate) authored: WorldSettings,
    pub(crate) resolved: ResolvedSettings,
    // The live master "Graphics Quality" preset; an explicit per-row change
    // flips it to Custom.
    pub(crate) quality_preset: QualityPreset,
    pub(crate) gpu_profile: backend::GpuProfile,
    // What the probed device can run, which clamps every resolution.
    pub(crate) capabilities: backend::DeviceCapabilities,
    // The captured stats sub-row labels the "Display performance stats" master
    // toggle grays.
    pub(crate) perf_sub_row_labels: Vec<(AssetId, [f32; 3])>,
    // Window mode + authored size (the windowed size restored on mode return).
    pub(crate) window_args: Window,
    // The Resolution row's mode list, the user's chosen fullscreen mode, the
    // display's own mode at init, and the row labels grayed outside
    // fullscreen.
    pub(crate) display_modes: Vec<display_mode::DisplayMode>,
    pub(crate) resolution: Option<display_mode::DisplayMode>,
    pub(crate) current_mode: Option<display_mode::DisplayMode>,
    pub(crate) resolution_row_labels: Vec<(AssetId, [f32; 3])>,
    // The persisted graphics overrides as they stood at init, the fallback for
    // `persisted_graphics` until a settings change loads `settings_cache`.
    pub(crate) persisted_graphics: UserGraphicsSettings,
    // This launch's overrides (`--dlss-preset`), layered over the user's
    // choices on every resolve and never persisted.
    pub(crate) session: UserGraphicsSettings,
    // Whether the backend built the fog pass at init. A world that started with
    // fog off cannot be handed fog live; that edit rebuilds instead.
    pub(crate) fog_built: bool,
    // In-memory copy of the persisted settings store, loaded once on the first
    // settings change and mutated in place from then on, so a queued (not yet
    // flushed) background write is never re-read stale from disk.
    pub(crate) settings_cache: Option<crate::config::Settings>,
    // Background disk writer for settings changes; spawned on the first
    // persisted change so an unchanged session never starts the thread.
    pub(crate) settings_writer: Option<writer::SettingsWriter>,
    // Cursors into the SceneCommand / SettingCommand queues.
    pub(crate) scene_cmd_cursor: EventCursor,
    pub(crate) setting_cmd_cursor: EventCursor,
    // Last-published HUD state, so `publish_hud_state` only re-inserts the
    // resources (rebuilding the disabled-rows set) when the inputs actually
    // change rather than every frame. `None` until the first publish.
    pub(crate) published_hud_prefs: Option<HudPrefs>,
    pub(crate) published_disabled_inputs: Option<(bool, bool)>,
}

#[derive(Debug, Default)]
pub(crate) struct SettingsSystem;

impl SettingsSystem {
    pub(crate) fn new() -> Self {
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

// How wide a planet's finest terrain cells get, in meters: the resolved
// setting once graphics init has parked the state, else the section default.
pub(crate) fn planet_finest_cell_size(resources: &Resources) -> u32 {
    resources
        .get::<SettingsSlot>()
        .and_then(|slot| slot.0.as_ref())
        .map_or(PlanetDetail::default().finest_cell_size, |state| {
            state.resolved.graphics.planet_detail.finest_cell_size
        })
}

// Record `plan` against `graphics` when it reaches the renderer. A failed
// rebuild is logged, and running out of device memory raises the replay's memory
// pressure for the streaming valve.
pub(crate) fn record_plan(ops: &mut RenderOps, graphics: &ResolvedGraphics, plan: ApplyPlan) {
    if !plan.touches_renderer() {
        return;
    }
    let graphics = graphics.clone();
    ops.record_with(move |backend, out| {
        if let Err(e) = backend.apply_settings(&graphics, &plan) {
            tracing::error!("SettingsSystem: settings apply failed: {e}");
            out.memory_pressure |= matches!(e, RenderError::OutOfDeviceMemory(_));
        }
    });
}

impl SettingsState {
    // The state before init resolves anything: an unconfigured world's graphics,
    // the `Auto` preset, and no rows captured.
    pub(crate) fn new() -> Self {
        Self {
            keymap: keymap::KeyMap::default(),
            rebind_rows: Vec::new(),
            gamepad_map: GamepadMap::default(),
            pad_rebind_rows: Vec::new(),
            sliders: Vec::new(),
            cycle_value_labels: std::collections::HashMap::new(),
            authored: WorldSettings::default(),
            resolved: ResolvedSettings::default(),
            quality_preset: QualityPreset::Auto,
            gpu_profile: backend::GpuProfile::UNKNOWN,
            capabilities: backend::DeviceCapabilities::ALL,
            perf_sub_row_labels: Vec::new(),
            window_args: Window::default(),
            display_modes: Vec::new(),
            resolution: None,
            current_mode: None,
            resolution_row_labels: Vec::new(),
            persisted_graphics: UserGraphicsSettings::default(),
            session: UserGraphicsSettings::default(),
            fog_built: false,
            settings_cache: None,
            settings_writer: None,
            scene_cmd_cursor: EventCursor::default(),
            setting_cmd_cursor: EventCursor::default(),
            published_hud_prefs: None,
            published_disabled_inputs: None,
        }
    }

    // A neutral live state for tests: no persisted overrides, no preset ceiling
    // (`Custom`), the engine defaults everywhere else. Tests set the few fields
    // they exercise.
    #[cfg(test)]
    pub(crate) fn for_tests() -> Self {
        let shadow_cadence = concinnity_core::render::backend_init::ShadowCadence {
            update: Default::default(),
            distance: 200,
            cascades: 4,
        };
        let mut state = Self {
            quality_preset: QualityPreset::Custom,
            fog_built: true,
            ..Self::new()
        };
        state.authored.graphics.shadows.map_size = 2048;
        state.authored.graphics.shadows.update = shadow_cadence.update;
        state.authored.graphics.shadows.distance = shadow_cadence.distance;
        state.authored.graphics.shadows.cascades = shadow_cadence.cascades;
        state.authored.graphics.texture_filtering.anisotropy = 8;
        let r = &mut state.resolved;
        let g = &mut r.graphics;
        g.shadows.map_size = 2048;
        g.shadows.update = shadow_cadence.update;
        g.shadows.distance = shadow_cadence.distance;
        g.shadows.cascades = shadow_cadence.cascades;
        g.texture_filtering.anisotropy = 8;
        g.frame_pacing.vsync = true;
        g.frame_pacing.fps_cap = 0;
        g.frame_pacing.frames_in_flight = 2;
        g.culling.two_pass_occlusion = false;
        r.texture_cap = 0;
        r.texture_budget = 0;
        state
    }

    // The persisted graphics overrides in force: the in-memory store once a
    // settings change has loaded it, otherwise the snapshot init took from disk.
    pub(crate) fn persisted_graphics(&self) -> &UserGraphicsSettings {
        match &self.settings_cache {
            Some(cache) => &cache.graphics,
            None => &self.persisted_graphics,
        }
    }

    // The active quality preset's performance ceiling on this GPU.
    pub(crate) fn ceiling(&self) -> QualityCeiling {
        resolve_ceiling(self.quality_preset, &self.gpu_profile)
    }

    // The resolver's layers: the authored baseline, `ceiling`, the user's
    // `user` choices and this launch's overrides, clamped to what the probed
    // device can run.
    fn layers<'a>(
        &'a self,
        user: &'a UserGraphicsSettings,
        ceiling: &'a QualityCeiling,
    ) -> SettingsLayers<'a> {
        SettingsLayers {
            session: Some(&self.session),
            capabilities: &self.capabilities,
            ..SettingsLayers::new(&self.authored, ceiling, user)
        }
    }

    // Every setting, resolved under `user`'s choices and `ceiling`.
    pub(crate) fn resolve(
        &self,
        user: &UserGraphicsSettings,
        ceiling: &QualityCeiling,
    ) -> ResolvedSettings {
        self.layers(user, ceiling).resolve()
    }

    // Record the backend work that moving the resolved settings from `before`
    // to their current values takes: one plan, in one op.
    pub(crate) fn record_change(&self, before: &ResolvedSettings, ops: &mut RenderOps) {
        let after = &self.resolved;
        let plan = plan_apply(&after.diff(before), &before.graphics, &after.graphics);
        record_plan(ops, &after.graphics, plan);
    }

    // The preset-governed group, resolved under `user`'s choices and `ceiling`.
    pub(crate) fn resolve_quality(
        &self,
        user: &UserGraphicsSettings,
        ceiling: &QualityCeiling,
    ) -> ResolvedGraphics {
        self.layers(user, ceiling).resolve_quality()
    }

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
        let mut scratch = crate::gfx::system::scene::SceneVisibilityScratch::default();
        crate::gfx::system::scene::refresh_visibility_snapshot(ctx, &mut scratch);
        let now = ctx
            .resource::<concinnity_core::ecs::FrameTime>()
            .copied()
            .unwrap_or_default()
            .elapsed;
        let Some(slot) = ctx.resources.get_mut::<crate::ecs::ActiveSceneFlow>() else {
            return;
        };
        let elapsed = slot.elapsed(now);
        let mut scene_ops: Vec<snapshot::SceneOp> = Vec::new();
        for cmd in scene_cmds {
            let mut recorder = snapshot::SceneOpRecorder(&mut scene_ops);
            if let Err(rejection) = scene_flow::jump_to_scene(
                &mut slot.flow,
                &scratch.visibility,
                elapsed,
                cmd.scene,
                cmd.transition,
                &mut recorder,
            ) {
                tracing::warn!("scene jump to {} rejected: {rejection:?}", cmd.scene.0);
            }
        }
        if !scene_ops.is_empty() {
            ops.record(move |backend| {
                for op in scene_ops {
                    match op {
                        snapshot::SceneOp::SetFade(fade) => backend.set_fade(fade),
                        snapshot::SceneOp::Visibility { draw_idx, visible } => {
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
        // change -- steady-state frames skip the HashSet allocation the
        // disabled-rows set would otherwise churn every frame.
        let prefs = HudPrefs {
            show_fps: self.resolved.perf_stats && self.resolved.show_fps,
            show_vram: self.resolved.perf_stats && self.resolved.show_vram,
        };
        if self.published_hud_prefs != Some(prefs) {
            ctx.insert_resource(prefs);
            self.published_hud_prefs = Some(prefs);
        }

        // The Resolution row only applies in fullscreen (windowed sizes come from
        // the window, borderless covers the display), so it is inert in the other
        // modes. The disabled-rows set is fully determined by these two inputs.
        let is_fullscreen = self.window_args.mode == WindowMode::Fullscreen;
        let inputs = (self.resolved.perf_stats, is_fullscreen);
        if self.published_disabled_inputs != Some(inputs) {
            let mut disabled_rows = std::collections::HashSet::new();
            if !self.resolved.perf_stats {
                disabled_rows.extend([SettingKey::ShowFps, SettingKey::ShowVram]);
            }
            if !is_fullscreen {
                disabled_rows.insert(SettingKey::Resolution);
            }
            ctx.insert_resource(crate::ecs::DisabledSettingRows(disabled_rows));
            self.published_disabled_inputs = Some(inputs);
        }
    }
}

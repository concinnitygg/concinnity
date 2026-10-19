//! SettingsSystem: applies the runtime command batches UiInputSystem produces
//! (SettingCommand for settings-menu changes, SceneCommand for imperative scene
//! jumps), recording their backend effects into the frame's op queue, edits
//! the [`StoredSettings`](crate::settings::StoredSettings) resource the host
//! persists, and publishes the per-frame HUD-preference state:
//!   mod.rs      system + state + scene jumps + HUD-state publish
//!   init.rs     the state graphics init resolves, and its init-time captures
//!   apply.rs    the SettingCommand drain and its per-family row handlers
//!   quality.rs  the quality preset, feature toggle, and quality knob rows
//!   rebind.rs   the key and gamepad rebind rows
//!   rows.rs     row helpers shared with the init-time captures
//!
//! Runs before the renderer, so a change's recorded op lands on the backend
//! before this frame's draw. The state is resolved by graphics init (world
//! config, persisted overrides, backend capabilities) and parked as the
//! [`SettingsSlot`] resource; each step takes it and puts it back, so the state
//! and the `PipelineContext` are never borrowed together.

use crate::ecs::{
    EventCursor, HudPrefs, Parked, PipelineContext, Resources, StepResult, System, World,
};
use crate::render::render_mode::RenderMode;
use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::components::{
    GamepadAction, GamepadMap, PlanetDetail, SceneCommand, StreamingConfig, Window, WindowMode,
};
use crate::ecs::asset_id::AssetId;
use crate::input::keymap;
use crate::platform::GraphicsApiStatus;
use crate::render::error::RenderError;
use crate::render::ops::RenderOps;
use crate::render::queues::ActiveRenderQueues;
use crate::render::{backend, scene_flow, snapshot};
use crate::settings::apply::{ApplyPlan, plan_apply};
use crate::settings::quality::resolve_ceiling;
use crate::settings::{
    GraphicsApiTrial, QualityCeiling, QualityPreset, ResolvedGraphics, ResolvedSettings,
    SettingKey, SettingsLayers, SettingsRevision, UserGraphicsSettings, WorldSettings,
};
use crate::window::display_mode;

mod apply;
mod init;
#[cfg(test)]
mod init_rows_tests;
#[cfg(test)]
mod init_tests;
#[cfg(any(test, feature = "test-support"))]
mod inspect;
mod persist;
mod quality;
mod rebind;
mod rows;
#[cfg(test)]
mod tests;

pub use init::{DisplayReport, SettingsInit, gate_unavailable_rows};
#[cfg(any(test, feature = "test-support"))]
pub use inspect::SliderRow;
pub use persist::SettingsStore;
pub use rows::DISABLED_ROW_COLOR;

// One key-rebind row's runtime bookkeeping: the action it rebinds and the value
// `TextLabel` showing its bound key. Built at init from the row's `key_*`
// rebind HitRegion.
pub(crate) struct RebindViz {
    pub(crate) action: keymap::Bindable,
    pub(crate) value_id: AssetId,
}

// One gamepad-rebind row's runtime bookkeeping, mirroring `RebindViz`.
pub(crate) struct PadRebindViz {
    pub(crate) action: GamepadAction,
    pub(crate) value_id: AssetId,
}

// One slider row's runtime bookkeeping: the setting it controls, the track
// geometry it maps a fraction onto, and the handle Sprite + value TextLabel it
// drives. Built at init from the row's drag HitRegion and the handle Sprite's
// width.
pub(crate) struct SliderViz {
    pub(crate) key: SettingKey,
    pub(crate) track_x: f32,
    pub(crate) track_w: f32,
    pub(crate) handle_w: f32,
    pub(crate) handle_id: AssetId,
    pub(crate) value_id: AssetId,
}

/// The live settings state: every value the settings menu displays and cycles,
/// with the authored baselines a preset change re-clamps from and the row
/// bookkeeping captured at init.
pub struct SettingsState {
    // Live gameplay movement key map (the source of truth for the Controls-tab
    // rebind rows), pushed to the backend on each rebind (with a swap).
    pub(crate) keymap: keymap::KeyMap,
    pub(crate) rebind_rows: Vec<RebindViz>,
    // Live gamepad action -> button map (the source of truth for the gamepad
    // rebind rows), carried to the input poll via ControlsCommand on each rebind.
    pub(crate) gamepad_map: GamepadMap,
    pub(crate) pad_rebind_rows: Vec<PadRebindViz>,
    pub(crate) sliders: Vec<SliderViz>,
    // Cycle rows' setting key -> value-label id, captured at init, so a change
    // can relabel a row other than the one clicked.
    pub(crate) cycle_value_labels: hashbrown::HashMap<SettingKey, AssetId>,
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
    // The user's graphics overrides the resolver layers: the stored ones at
    // init, kept in step with every saved change.
    pub(crate) persisted_graphics: UserGraphicsSettings,
    // This launch's overrides (`--dlss-preset`), layered over the user's
    // choices on every resolve and never persisted.
    pub(crate) session: UserGraphicsSettings,
    // Whether the backend built the fog pass at init. A world that started with
    // fog off cannot be handed fog live; that edit rebuilds instead.
    pub(crate) fog_built: bool,
    // Cursors into the SceneCommand / SettingCommand queues.
    pub(crate) scene_cmd_cursor: EventCursor,
    pub(crate) setting_cmd_cursor: EventCursor,
    // Last-published HUD state, so `publish_hud_state` only re-inserts the
    // resources (rebuilding the disabled-rows set) when the inputs actually
    // change rather than every frame. `None` until the first publish.
    pub(crate) published_hud_prefs: Option<HudPrefs>,
    pub(crate) published_disabled_inputs: Option<(bool, bool)>,
    // Steps run since graphics init parked this state, so a newly chosen
    // graphics API is confirmed only once a frame has drawn on it.
    pub(crate) steps_since_init: u32,
    // Whether this launch's graphics API has been confirmed, or needed none.
    pub(crate) graphics_api_settled: bool,
}

/// Applies the settings menu's command batches against the parked
/// [`SettingsState`], and writes the store through the host's
/// [`SettingsStore`] whenever it changes.
#[derive(Debug, Default)]
pub struct SettingsSystem {
    store: Option<Box<dyn SettingsStore>>,
    // The revision last written; a world starts at zero, unchanged.
    written: SettingsRevision,
}

impl SettingsSystem {
    /// The system for a world that runs with a window.
    pub fn for_world(world: &World) -> Option<Self> {
        RenderMode::renders_in(world).then(Self::new)
    }

    /// The system, persisting nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Write the store through `store` whenever it changes.
    pub fn with_store(mut self, store: Box<dyn SettingsStore>) -> Self {
        self.store = Some(store);
        self
    }
}

impl SettingsSystem {
    // Apply this tick's commands against the parked state. With no parked
    // state (graphics init has not succeeded) or op queue the commands wait
    // in retention.
    fn apply(&mut self, ctx: &mut PipelineContext) {
        let Some(mut state) = SettingsSlot::take(ctx.resources) else {
            return;
        };
        let Some(mut queues) = ActiveRenderQueues::take(ctx.resources) else {
            SettingsSlot::put(ctx.resources, state);
            return;
        };
        state.apply_scene_commands(ctx, &mut queues.ops);
        state.apply_setting_commands(ctx, &mut queues.ops);
        ActiveRenderQueues::put(ctx.resources, queues);
        state.publish_hud_state(ctx);
        state.confirm_graphics_api(ctx);
        SettingsSlot::put(ctx.resources, state);
    }
}

/// The parked [`SettingsState`], published by graphics init.
pub type SettingsSlot = Parked<SettingsState>;

impl core::fmt::Debug for SettingsState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SettingsState")
            .field("quality_preset", &self.quality_preset)
            .finish()
    }
}

impl System for SettingsSystem {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        self.apply(ctx);
        // Written in the same step that made the change, so a tick that stops
        // the run still persists it.
        self.persist(ctx);
        StepResult::Continue
    }
}

/// How wide a planet's finest terrain cells get, in meters: the resolved
/// setting once graphics init has parked the state, else the section default.
pub fn planet_finest_cell_size(resources: &Resources) -> u32 {
    SettingsSlot::get(resources).map_or(PlanetDetail::default().finest_cell_size, |state| {
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
            cycle_value_labels: hashbrown::HashMap::new(),
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
            scene_cmd_cursor: EventCursor::default(),
            setting_cmd_cursor: EventCursor::default(),
            published_hud_prefs: None,
            published_disabled_inputs: None,
            steps_since_init: 0,
            graphics_api_settled: false,
        }
    }

    // A neutral live state for tests: no persisted overrides, no preset ceiling
    // (`Custom`), the engine defaults everywhere else. Tests set the few fields
    // they exercise.
    #[cfg(test)]
    pub(crate) fn for_tests() -> Self {
        let shadow_cadence = crate::render::backend_init::ShadowCadence {
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

    /// The live resolved settings.
    pub fn resolved(&self) -> &ResolvedSettings {
        &self.resolved
    }

    /// The window as configured: authored size, persisted mode.
    pub fn window(&self) -> &Window {
        &self.window_args
    }

    /// The mode a freshly built window switches to. A window opens as a
    /// standard titled window, so `None` when that is the configured mode.
    pub fn startup_window_mode(&self) -> Option<WindowMode> {
        (self.window_args.mode != WindowMode::Windowed).then_some(self.window_args.mode)
    }

    /// Overwrite `config`'s texture cap and budget with the resolved ones,
    /// which win over the authored values.
    pub fn apply_to_streaming(&self, config: &mut StreamingConfig) {
        config.texture_cap = self.resolved.texture_cap;
        config.texture_budget = self.resolved.texture_budget;
    }

    /// The live master quality preset.
    pub fn quality_preset(&self) -> QualityPreset {
        self.quality_preset
    }

    /// The live key map.
    pub fn keymap(&self) -> &keymap::KeyMap {
        &self.keymap
    }

    /// The user's chosen fullscreen display mode, if any.
    pub fn resolution(&self) -> Option<display_mode::DisplayMode> {
        self.resolution
    }

    /// The user's graphics overrides the resolver layers.
    pub fn persisted_graphics(&self) -> &UserGraphicsSettings {
        &self.persisted_graphics
    }

    /// Record whether the backend built the fog pass.
    pub fn set_fog_built(&mut self, built: bool) {
        self.fog_built = built;
    }

    /// Adopt what the built backend can run as the authority from here on.
    pub fn set_capabilities(&mut self, caps: backend::DeviceCapabilities) {
        self.capabilities = caps;
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
    // The flow lives in the shared `ActiveSceneFlow` resource (the renderer
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
        let mut scratch = scene_flow::SceneVisibilityScratch::default();
        scene_flow::refresh_visibility_snapshot(ctx, &mut scratch);
        let now = ctx
            .resource::<crate::ecs::FrameTime>()
            .copied()
            .unwrap_or_default()
            .elapsed;
        let Some(slot) = ctx.resources.get_mut::<scene_flow::ActiveSceneFlow>() else {
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
    // tick. Done after the settings drain so a "Display performance stats"
    // toggle this frame is reflected the same frame: the visibility
    // (StatHudSystem reads HudPrefs) and the inert/grayed sub-rows
    // (UiInputSystem reads DisabledSettingRows) stay in lockstep with the
    // gray-out applied in the drain.
    fn publish_hud_state(&mut self, ctx: &mut PipelineContext) {
        // Both resources are pure functions of a few settings fields and persist
        // in the resource map once inserted, so republish only when the inputs
        // change: steady-state frames skip the HashSet allocation the
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
            let mut disabled_rows = hashbrown::HashSet::new();
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

// The steps after graphics init before a newly chosen graphics API counts as
// working: the first runs before this launch has drawn anything.
const GRAPHICS_API_CONFIRM_STEPS: u32 = 2;

impl SettingsState {
    // Clear the persisted trial on a newly chosen graphics API once a frame has
    // drawn on it, so the next launch keeps the choice.
    fn confirm_graphics_api(&mut self, ctx: &mut PipelineContext) {
        if self.graphics_api_settled {
            return;
        }
        self.steps_since_init = self.steps_since_init.saturating_add(1);
        if self.steps_since_init < GRAPHICS_API_CONFIRM_STEPS {
            return;
        }
        self.graphics_api_settled = true;
        if !ctx
            .resource::<GraphicsApiStatus>()
            .is_some_and(|status| status.confirm_pending)
        {
            return;
        }
        // Only the launch marks a trial `Trying`; an `Untried` one is a newer
        // choice the next launch must still try.
        self.edit_stored(ctx, |_, _, cfg| {
            let trying = cfg.graphics.graphics_api_trial == Some(GraphicsApiTrial::Trying);
            if trying {
                cfg.graphics.graphics_api_trial = None;
            }
            trying
        });
    }
}

// SettingsSystem unit tests: drive the SettingCommand drain against the
// recording backend and a bare World. No GPU device is created; a persisted
// change is the StoredSettings resource the drain edits and the revision it
// bumps.

use crate::components::{
    AaMode, AudioCommand, ControlsCommand, GamepadAction, GamepadButton, GamepadMap,
    IndirectTechnique, InputKey, SettingCommand, SettingOp, ShadowUpdate, Sprite, StreamingConfig,
    TextLabel, UpscalerBackend, WindowMode,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{FrameRateCap, HudPrefs, StepResult, System, World};
use crate::input::keymap::{Bindable, KeyMap};
use crate::render::backend::init::ShadowCadence;
use crate::render::backend::{GpuProfile, GpuVendor};
use crate::render::output::dlss::DlssPreset;
use crate::render::temporal::history_reset::{HistoryResetCauses, PendingHistoryReset};
use crate::settings::UserGraphicsSettings;
use crate::settings::system::EventCursor;
use crate::settings::system::RenderOps;
use crate::settings::system::Window;
use crate::window::display_mode::DisplayMode;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use super::{PadRebindViz, RebindViz, SettingsState, SliderViz};
use crate::render::backend::recording::{
    Call, RecordingBackend, SharedRecording, recording_backend,
};
use crate::settings::quality::QualityPreset;
use crate::settings::resolve::{ResolvedSettings, WorldSettings};
use crate::settings::{self, SettingKey, SettingsRevision, StoredSettings, levels, registry};

// The value label ids the fixture wires up, so a test can assert a row relabeled
// without standing up a whole menu.
const VALUE_LABEL: AssetId = AssetId(1);
const HANDLE: AssetId = AssetId(2);
const QUALITY_LABEL: AssetId = AssetId(3);
const SUB_ROW_LABEL: AssetId = AssetId(4);
const REBIND_LABEL: AssetId = AssetId(5);
const VICTIM_LABEL: AssetId = AssetId(6);
const RESOLUTION_LABEL: AssetId = AssetId(7);
const TOGGLE_LABEL: AssetId = AssetId(8);
const PAD_REBIND_LABEL: AssetId = AssetId(9);
const PAD_VICTIM_LABEL: AssetId = AssetId(10);

// The authored color the fixture's gray-able rows start at, so a restore is
// distinguishable from a gray.
const LIT: [f32; 3] = [1.0, 1.0, 1.0];

struct Fixture {
    world: World,
    state: SettingsState,
    backend: RecordingBackend,
    calls: SharedRecording,
}

impl Fixture {
    fn new() -> Self {
        Self::with_profile(GpuProfile::UNKNOWN)
    }

    fn with_profile(gpu_profile: GpuProfile) -> Self {
        let (calls, backend) = recording_backend();

        let cycle_value_labels = [
            (SettingKey::GraphicsQuality, QUALITY_LABEL),
            (SettingKey::RenderScale, VALUE_LABEL),
        ]
        .into_iter()
        .collect();

        let shadow_cadence = ShadowCadence {
            update: ShadowUpdate::EveryFrame,
            distance: levels::SHADOW_DISTANCE.at(0),
            cascades: levels::SHADOW_CASCADES.at(0),
        };
        let mut authored = WorldSettings::default();
        let g = &mut authored.graphics;
        g.shadows.map_size = levels::SHADOW_RESOLUTION.at(0);
        g.shadows.update = shadow_cadence.update;
        g.shadows.distance = shadow_cadence.distance;
        g.shadows.cascades = shadow_cadence.cascades;
        g.texture_filtering.anisotropy = levels::ANISOTROPY.at(0);
        let mut graphics = authored.graphics.clone();
        graphics.anti_aliasing.upscale_quality = levels::RENDER_SCALE.at(0);
        graphics.anti_aliasing.upscaler = levels::UPSCALE_BACKEND.at(0);
        graphics.frame_pacing.vsync = false;
        graphics.frame_pacing.fps_cap = levels::FPS_CAP.at(0);
        graphics.frame_pacing.frames_in_flight = levels::FRAMES_IN_FLIGHT.at(0);
        graphics.display.hdr = false;
        graphics.display.pq = false;
        graphics.dlss_preset = levels::DLSS_PRESET.at(0);
        graphics.culling.two_pass_occlusion = false;
        let resolved = ResolvedSettings {
            graphics,
            perf_stats: true,
            show_fps: true,
            show_vram: false,
            texture_cap: levels::TEXTURE_QUALITY.at(0).0,
            texture_budget: levels::TEXTURE_QUALITY.at(0).1,
        };
        let state = SettingsState {
            keymap: KeyMap::default(),
            rebind_rows: vec![
                RebindViz {
                    action: Bindable::Forward,
                    value_id: REBIND_LABEL,
                },
                RebindViz {
                    action: Bindable::Backward,
                    value_id: VICTIM_LABEL,
                },
            ],
            gamepad_map: GamepadMap::default(),
            pad_rebind_rows: vec![
                PadRebindViz {
                    action: GamepadAction::Jump,
                    value_id: PAD_REBIND_LABEL,
                },
                PadRebindViz {
                    action: GamepadAction::Sprint,
                    value_id: PAD_VICTIM_LABEL,
                },
            ],
            sliders: vec![SliderViz {
                key: SettingKey::Exposure,
                track_x: 100.0,
                track_w: 200.0,
                handle_w: 20.0,
                handle_id: HANDLE,
                value_id: VALUE_LABEL,
            }],
            cycle_value_labels,
            authored,
            resolved,
            quality_preset: QualityPreset::Custom,
            gpu_profile,
            capabilities: crate::render::backend::DeviceCapabilities::ALL,
            perf_sub_row_labels: vec![(SUB_ROW_LABEL, LIT)],
            window_args: Window::default(),
            display_modes: Vec::new(),
            resolution: None,
            current_mode: None,
            resolution_row_labels: vec![(RESOLUTION_LABEL, LIT)],
            persisted_graphics: UserGraphicsSettings::default(),
            session: UserGraphicsSettings::default(),
            fog_built: true,
            scene_cmd_cursor: EventCursor::default(),
            setting_cmd_cursor: EventCursor::default(),
            published_hud_prefs: None,
            published_disabled_inputs: None,
            steps_since_init: 0,
            graphics_api_settled: false,
        };

        let mut fixture = Self {
            world: World::new(),
            state,
            backend,
            calls,
        };
        fixture.world.insert_resource(StoredSettings::default());
        {
            let mut ctx = fixture.world.context();
            for (id, content) in [
                (VALUE_LABEL, "value"),
                (QUALITY_LABEL, "quality"),
                (SUB_ROW_LABEL, "sub"),
                (REBIND_LABEL, "rebind"),
                (VICTIM_LABEL, "victim"),
                (RESOLUTION_LABEL, "resolution"),
                (TOGGLE_LABEL, "toggle"),
                (PAD_REBIND_LABEL, "pad_rebind"),
                (PAD_VICTIM_LABEL, "pad_victim"),
            ] {
                ctx.push_identified(
                    id,
                    TextLabel {
                        content: content.to_string(),
                        color: LIT,
                        ..Default::default()
                    },
                );
            }
            ctx.push_identified(
                HANDLE,
                Sprite {
                    x: 0.0,
                    ..Default::default()
                },
            );
        }
        fixture
    }

    // Queue `cmds` and run one drain over them, replaying the recorded ops
    // onto the mock backend so the call assertions keep working.
    fn apply(&mut self, cmds: Vec<SettingCommand>) {
        {
            let mut ctx = self.world.context();
            let events = ctx.events_mut::<SettingCommand>();
            for cmd in cmds {
                events.send(cmd);
            }
        }
        let mut ctx = self.world.context();
        let mut ops = RenderOps::default();
        self.state.apply_setting_commands(&mut ctx, &mut ops);
        ops.replay(&mut self.backend);
    }

    // Cycle a row forward once, routing the value label back to the row.
    fn next(&mut self, setting: SettingKey) {
        self.apply(vec![cycle(setting, SettingOp::Next)]);
    }

    // The stored settings after a persisted change.
    fn persisted(&self) -> StoredSettings {
        assert!(self.revision() > 0, "a change was persisted");
        self.world
            .resource::<StoredSettings>()
            .cloned()
            .expect("the store is published")
    }

    // How many persisted changes the drain has made.
    fn revision(&self) -> u64 {
        self.world.resource::<SettingsRevision>().map_or(0, |r| r.0)
    }

    fn saw(&self, call: &Call) -> bool {
        self.calls.lock().saw(call)
    }

    fn label(&mut self, id: AssetId) -> String {
        self.world
            .context()
            .get_by_id::<TextLabel>(id)
            .map(|l| l.content.clone())
            .expect("label present")
    }

    fn label_color(&mut self, id: AssetId) -> [f32; 3] {
        self.world
            .context()
            .get_by_id::<TextLabel>(id)
            .map(|l| l.color)
            .expect("label present")
    }

    // Every ControlsCommand the drain has sent, oldest first.
    fn sent_controls(&mut self) -> Vec<ControlsCommand> {
        self.world
            .context()
            .events::<ControlsCommand>()
            .map(|e| e.read(&mut EventCursor::default()).copied().collect())
            .unwrap_or_default()
    }
}

// A cycle-row command carrying the fixture's value label.
fn cycle(setting: SettingKey, op: SettingOp) -> SettingCommand {
    SettingCommand {
        setting,
        op,
        value_label: Some(VALUE_LABEL),
        persist: true,
    }
}

// A slider command at `frac` of the track; `persist` marks the drag-release
// frame.
fn drag(setting: SettingKey, frac: f32, persist: bool) -> SettingCommand {
    SettingCommand {
        setting,
        op: SettingOp::SetFraction(frac),
        value_label: Some(VALUE_LABEL),
        persist,
    }
}

// Cycling vsync flips the live value, pushes it to the backend, relabels the
// row, and persists the new value.
#[test]
fn vsync_cycles_live_and_persists() {
    let mut f = Fixture::new();
    f.next(SettingKey::Vsync);

    assert!(
        f.state.resolved.graphics.frame_pacing.vsync,
        "the row cycled Off -> On"
    );
    assert!(f.saw(&Call::SetVsync(true)), "applied live");
    assert_eq!(f.label(VALUE_LABEL), "On");
    assert_eq!(f.persisted().graphics.vsync, Some(true));
}

// Prev cycles the other way, wrapping around the option list.
#[test]
fn cycling_prev_wraps_the_option_list() {
    let mut f = Fixture::new();
    f.apply(vec![cycle(SettingKey::Vsync, SettingOp::Prev)]);
    assert!(
        f.state.resolved.graphics.frame_pacing.vsync,
        "Off wraps back to the last option"
    );
}

// A dropdown pick jumps straight to an option index rather than stepping.
#[test]
fn set_index_jumps_to_the_chosen_option() {
    let mut f = Fixture::new();
    f.apply(vec![cycle(
        SettingKey::ShadowCascades,
        SettingOp::SetIndex(2),
    )]);
    assert_eq!(
        f.state.resolved.graphics.shadows.cascades,
        levels::SHADOW_CASCADES.at(2)
    );
}

// A rebind binds the action to the captured key, pushes the map to the backend,
// persists it, and relabels both the rebound row and the action it took the key
// from (which inherits the rebound action's old key).
#[test]
fn rebind_swaps_the_victim_and_relabels_both_rows() {
    let mut f = Fixture::new();
    let forward_key = f.state.keymap.get(Bindable::Forward);
    let backward_key = f.state.keymap.get(Bindable::Backward);

    f.apply(vec![SettingCommand {
        setting: SettingKey::KeyRebind(Bindable::Forward),
        op: SettingOp::Rebind(backward_key),
        value_label: None,
        persist: true,
    }]);

    assert_eq!(f.state.keymap.get(Bindable::Forward), backward_key);
    assert_eq!(
        f.state.keymap.get(Bindable::Backward),
        forward_key,
        "the victim takes the rebound action's old key"
    );
    assert!(f.saw(&Call::SetKeymap));
    assert_eq!(f.label(REBIND_LABEL), backward_key.display_name());
    assert_eq!(f.label(VICTIM_LABEL), forward_key.display_name());
    assert_eq!(f.persisted().controls.keymap, Some(f.state.keymap));
}

// Rebinding to an unbound key relabels only the rebound row: there is no victim
// to swap with.
#[test]
fn rebind_to_a_free_key_has_no_victim() {
    let mut f = Fixture::new();
    assert!(
        f.state.keymap.action_for_key(InputKey::Q).is_none(),
        "Q is unbound in the default map"
    );
    f.apply(vec![SettingCommand {
        setting: SettingKey::KeyRebind(Bindable::Forward),
        op: SettingOp::Rebind(InputKey::Q),
        value_label: None,
        persist: true,
    }]);

    assert_eq!(f.state.keymap.get(Bindable::Forward), InputKey::Q);
    assert_eq!(f.label(REBIND_LABEL), InputKey::Q.display_name());
    assert_eq!(f.label(VICTIM_LABEL), "victim", "no victim was relabeled");
}

// A key rebind on a setting that is not a key rebind row is ignored.
#[test]
fn rebind_of_a_non_rebind_setting_is_ignored() {
    let mut f = Fixture::new();
    f.apply(vec![SettingCommand {
        setting: SettingKey::Vsync,
        op: SettingOp::Rebind(InputKey::Q),
        value_label: None,
        persist: true,
    }]);

    assert!(!f.saw(&Call::SetKeymap));
    assert_eq!(f.state.keymap, KeyMap::default());
}

// A slider applies live on every drag frame but only writes to the store on the
// commit (release) frame.
#[test]
fn slider_applies_live_and_persists_only_on_release() {
    let mut f = Fixture::new();
    f.apply(vec![drag(SettingKey::Exposure, 1.0, false)]);

    let mid_drag = f.state.resolved.graphics.post_process().exposure;
    assert!(f.saw(&Call::UpdatePostProcess), "applied live mid-drag");
    assert!(f.revision() == 0, "an in-progress drag never writes");

    f.apply(vec![drag(SettingKey::Exposure, 1.0, true)]);
    assert_eq!(
        f.state.resolved.graphics.post_process().exposure,
        mid_drag,
        "same value applied"
    );
    assert!(
        f.persisted().graphics.exposure_ev.is_some(),
        "the release frame writes"
    );
}

// The slider persists the authored EV and the composite applies it as a linear
// multiplier.
#[test]
fn exposure_slider_persists_ev_and_applies_the_multiplier() {
    let mut f = Fixture::new();
    // Fraction 1.0 is the top of the EV range.
    f.apply(vec![drag(SettingKey::Exposure, 1.0, true)]);

    let ev = f.persisted().graphics.exposure_ev.expect("persisted");
    let exposure = registry::slider(SettingKey::Exposure).expect("exposure is a slider");
    assert_eq!(
        f.state.resolved.graphics.post_process().exposure,
        concinnity_core::math::exp2(exposure.apply(ev)),
        "the live param is 2^ev of the clamped EV"
    );
    assert_eq!(f.label(VALUE_LABEL), (exposure.format)(ev));
}

// The slider's handle slides to the dragged fraction along its track.
#[test]
fn slider_moves_the_handle_along_its_track() {
    let mut f = Fixture::new();
    f.apply(vec![drag(SettingKey::Exposure, 0.5, false)]);

    let handle_x = f
        .world
        .context()
        .get_by_id::<Sprite>(HANDLE)
        .map(|s| s.x)
        .expect("handle present");
    // track_x 100 + 0.5 * (track_w 200 - handle_w 20).
    assert_eq!(handle_x, 190.0);
}

// A fraction outside the track clamps rather than sliding the handle off it.
#[test]
fn slider_clamps_an_out_of_range_fraction() {
    let mut f = Fixture::new();
    f.apply(vec![drag(SettingKey::Exposure, 2.0, false)]);

    let handle_x = f
        .world
        .context()
        .get_by_id::<Sprite>(HANDLE)
        .map(|s| s.x)
        .expect("handle present");
    assert_eq!(handle_x, 280.0, "pinned to the track's right end");
}

// A Next/Prev on a slider row (a focused row's Left/Right pulse) steps the
// value a twentieth of the range from the handle's current position, clamped
// at the ends, applying + persisting like a released drag.
#[test]
fn slider_steps_by_next_and_prev() {
    let mut f = Fixture::new();
    let handle_x = |f: &mut Fixture| {
        f.world
            .context()
            .get_by_id::<Sprite>(HANDLE)
            .map(|s| s.x)
            .expect("handle present")
    };
    // Place the handle mid-track (fraction 0.5 = x 190 on the 180px travel).
    f.apply(vec![drag(SettingKey::Exposure, 0.5, true)]);
    assert_eq!(handle_x(&mut f), 190.0);

    f.apply(vec![cycle(SettingKey::Exposure, SettingOp::Next)]);
    assert!(
        (handle_x(&mut f) - 199.0).abs() < 1.0e-3,
        "Next steps +5% of the travel"
    );
    assert!(f.persisted().graphics.exposure_ev.is_some());

    f.apply(vec![cycle(SettingKey::Exposure, SettingOp::Prev)]);
    assert!(
        (handle_x(&mut f) - 190.0).abs() < 1.0e-3,
        "Prev steps back down"
    );

    // Stepping far past the bottom clamps at the track's left end.
    for _ in 0..25 {
        f.apply(vec![cycle(SettingKey::Exposure, SettingOp::Prev)]);
    }
    assert_eq!(handle_x(&mut f), 100.0);
}

// A fraction sent to a setting that is not a slider is ignored.
#[test]
fn a_fraction_on_a_non_slider_is_ignored() {
    let mut f = Fixture::new();
    f.apply(vec![drag(SettingKey::Vsync, 0.5, true)]);

    assert!(f.calls.lock().calls.is_empty());
    assert_eq!(f.revision(), 0);
}

// Every row with static options applies a Next and persists it, so no setting
// misses its handler or its table row.
#[test]
fn every_option_row_applies_and_persists() {
    for key in SettingKey::ALL {
        if settings::options(key).is_none() {
            continue;
        }
        let mut f = Fixture::new();
        f.next(key);
        assert!(f.revision() > 0, "{key:?} persisted");
    }
}

// A stepper op on a slider or rebind row, and a rebind op on a slider, are
// ignored.
#[test]
fn an_op_outside_its_row_kind_is_ignored() {
    let mut f = Fixture::new();
    f.apply(vec![
        cycle(SettingKey::Exposure, SettingOp::SetIndex(1)),
        cycle(SettingKey::KeyRebind(Bindable::Jump), SettingOp::Next),
        cycle(SettingKey::Fov, SettingOp::Rebind(InputKey::Q)),
    ]);

    assert!(f.calls.lock().calls.is_empty());
    assert_eq!(f.revision(), 0);
    assert_eq!(f.state.keymap, KeyMap::default());
}

// Every table-driven Off/On row flips its state field and persisted override on
// Next, and back on Prev.
#[test]
fn bool_rows_flip_state_and_persisted_value_both_ways() {
    for row in &super::apply::BOOL_ROWS {
        let mut f = Fixture::new();
        let before = *(row.state)(&mut f.state);
        let stored = |f: &mut Fixture| *(row.persisted)(&mut f.persisted().graphics);

        f.next(row.key);
        assert_eq!(*(row.state)(&mut f.state), !before, "{:?} Next", row.key);
        assert_eq!(stored(&mut f), Some(!before), "{:?} Next persists", row.key);

        f.apply(vec![cycle(row.key, SettingOp::Prev)]);
        assert_eq!(*(row.state)(&mut f.state), before, "{:?} Prev", row.key);
        assert_eq!(stored(&mut f), Some(before), "{:?} Prev persists", row.key);
        assert_eq!(f.revision(), 2, "{:?} persists each change", row.key);
    }
}

// A sub-quality slider rides the quality-params push (no pass rebuild) rather
// than the post-process push.
#[test]
fn quality_param_slider_updates_quality_params() {
    let mut f = Fixture::new();
    f.apply(vec![drag(SettingKey::SsaoRadius, 0.5, true)]);

    assert!(f.saw(&Call::UpdateQualityParams));
    assert!(
        !f.saw(&Call::UpdatePostProcess),
        "a quality param is not a post-process param"
    );
    assert!(f.persisted().graphics.ssao_radius.is_some());
}

// Ambient scale lives in the light uniforms, so it takes its own setter on top
// of the post-process push.
// A bloom drag reaches the composite: the plan pushes the post-process
// tunables, and the backend reads the new intensity.
#[test]
fn a_bloom_drag_pushes_the_composite() {
    let mut f = Fixture::new();
    f.state.resolved.graphics.bloom.intensity = 0.0;
    f.apply(vec![drag(SettingKey::BloomIntensity, 1.0, false)]);
    assert!(f.state.resolved.graphics.bloom.intensity > 0.0);
    assert!(f.saw(&Call::UpdatePostProcess));
}

#[test]
fn ambient_slider_takes_the_dedicated_setter() {
    let mut f = Fixture::new();
    f.apply(vec![drag(SettingKey::AmbientIntensity, 0.75, true)]);

    let applied = f.state.resolved.graphics.ambient_intensity();
    assert!(f.saw(&Call::SetAmbientIntensity(applied)));
    assert!(f.persisted().graphics.ambient_intensity.is_some());
}

// Mouse sensitivity is a camera preference, not a render param: it travels as a
// ControlsCommand and persists the radians/pixel value the camera reads, not the
// 1..100 UI value.
#[test]
fn mouse_sensitivity_slider_sends_a_controls_command() {
    let mut f = Fixture::new();
    f.apply(vec![drag(SettingKey::MouseSensitivity, 1.0, true)]);

    let sent: Vec<ControlsCommand> = f
        .world
        .context()
        .events::<ControlsCommand>()
        .map(|e| e.read(&mut EventCursor::default()).copied().collect())
        .unwrap_or_default();
    assert_eq!(sent.len(), 1);
    assert!(sent[0].mouse_sensitivity.is_some());
    assert!(sent[0].fov_y_degrees.is_none());
    assert!(
        !f.saw(&Call::UpdatePostProcess),
        "sensitivity is not a render param"
    );

    let stored = sent[0].mouse_sensitivity.unwrap();
    assert_eq!(
        f.persisted().controls.mouse_sensitivity,
        Some(stored),
        "the radians/pixel value is what persists"
    );
}

// A saved change lands in the published store, so a world started later in the
// process seeds its cameras from the new value.
#[test]
fn a_saved_change_reaches_the_stored_controls() {
    let mut f = Fixture::new();
    f.apply(vec![drag(SettingKey::Fov, 0.0, true)]);
    let controls = f.persisted().controls_command();
    assert_eq!(controls.fov_y_degrees, Some(50.0));
}

// FOV is likewise a camera preference; it persists the clamped degrees.
#[test]
fn fov_slider_sends_a_controls_command() {
    let mut f = Fixture::new();
    f.apply(vec![drag(SettingKey::Fov, 0.0, true)]);

    let sent: Vec<ControlsCommand> = f
        .world
        .context()
        .events::<ControlsCommand>()
        .map(|e| e.read(&mut EventCursor::default()).copied().collect())
        .unwrap_or_default();
    assert_eq!(sent.len(), 1);
    assert!(sent[0].fov_y_degrees.is_some());
    assert!(sent[0].mouse_sensitivity.is_none());
    assert_eq!(f.persisted().graphics.fov, sent[0].fov_y_degrees);
}

// A gamepad rebind binds the action to the captured button, hands the new map
// to the input poll as a ControlsCommand, persists it, and relabels both the
// rebound row and the action it took the button from.
#[test]
fn pad_rebind_swaps_the_victim_and_relabels_both_rows() {
    use crate::components::GamepadAction;
    let mut f = Fixture::new();
    let jump_button = f.state.gamepad_map.get(GamepadAction::Jump);
    let sprint_button = f.state.gamepad_map.get(GamepadAction::Sprint);

    f.apply(vec![SettingCommand {
        setting: SettingKey::PadRebind(GamepadAction::Jump),
        op: SettingOp::RebindButton(sprint_button),
        value_label: None,
        persist: true,
    }]);

    assert_eq!(f.state.gamepad_map.get(GamepadAction::Jump), sprint_button);
    assert_eq!(
        f.state.gamepad_map.get(GamepadAction::Sprint),
        jump_button,
        "the victim takes the rebound action's old button"
    );
    let sent = f.sent_controls();
    assert_eq!(sent.len(), 1);
    assert_eq!(
        sent[0].gamepad_map,
        Some(f.state.gamepad_map),
        "the live map travels to the input poll"
    );
    assert_eq!(f.label(PAD_REBIND_LABEL), sprint_button.display_name());
    assert_eq!(f.label(PAD_VICTIM_LABEL), jump_button.display_name());
    assert_eq!(
        f.persisted().controls.gamepad_map,
        Some(f.state.gamepad_map)
    );
}

// A button rebind naming a keyboard (or unknown) action is ignored.
#[test]
fn pad_rebind_of_a_non_gamepad_action_is_ignored() {
    let mut f = Fixture::new();
    f.apply(vec![SettingCommand {
        setting: SettingKey::KeyRebind(Bindable::Forward),
        op: SettingOp::RebindButton(GamepadButton::North),
        value_label: None,
        persist: true,
    }]);

    assert_eq!(f.state.gamepad_map, GamepadMap::default());
    assert!(f.revision() == 0, "nothing persisted");
}

// The gamepad sliders travel as ControlsCommands and persist the applied
// values: radians/second for the look sensitivity, the deflection fraction for
// the deadzone (not their UI-scale values).
#[test]
fn gamepad_sliders_send_controls_commands_and_persist_applied_values() {
    let mut f = Fixture::new();
    f.apply(vec![
        drag(SettingKey::GamepadLookSensitivity, 1.0, true),
        drag(SettingKey::GamepadDeadzone, 0.5, true),
    ]);

    let sent = f.sent_controls();
    assert_eq!(sent.len(), 2);
    let rate = sent[0].gamepad_look_sensitivity.expect("sensitivity sent");
    assert!(
        (rate - 6.0).abs() < 1e-5,
        "full track is the max rate: {rate}"
    );
    let dz = sent[1].gamepad_deadzone.expect("deadzone sent");
    assert!(
        (dz - 0.2).abs() < 1e-5,
        "mid track of 0..40% stores 0.2: {dz}"
    );
    assert!(
        !f.saw(&Call::UpdatePostProcess),
        "the gamepad sliders are not render params"
    );
    let cfg = f.persisted();
    assert_eq!(cfg.controls.gamepad_look_sensitivity, Some(rate));
    assert_eq!(cfg.controls.gamepad_deadzone, Some(dz));
}

// The frame-rate cap applies through the republished resource the runtime-level
// pacer reads, with no backend call.
#[test]
fn fps_cap_publishes_the_frame_rate_cap_resource() {
    let mut f = Fixture::new();
    f.next(SettingKey::FpsCap);

    let published = f
        .world
        .context()
        .resource::<FrameRateCap>()
        .map(|c| c.0)
        .expect("cap published");
    assert_eq!(published, f.state.resolved.graphics.frame_pacing.fps_cap);
    assert_eq!(
        f.persisted().graphics.fps_cap,
        Some(f.state.resolved.graphics.frame_pacing.fps_cap)
    );
}

// Volumes are owned by audio playback, so each change travels as an
// AudioCommand it drains this same tick, addressed to the row's target.
#[test]
fn volume_rows_send_targeted_audio_commands() {
    use crate::components::AudioTarget;
    let mut f = Fixture::new();
    f.next(SettingKey::MasterVolume);
    f.next(SettingKey::VoiceVolume);

    let sent: Vec<AudioCommand> = f
        .world
        .context()
        .events::<AudioCommand>()
        .map(|e| e.read(&mut EventCursor::default()).copied().collect())
        .unwrap_or_default();
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0].target, AudioTarget::Master);
    assert_eq!(f.persisted().audio.master_volume, Some(sent[0].gain));
    assert_eq!(sent[1].target, AudioTarget::Voice);
    assert_eq!(f.persisted().audio.voice_volume, Some(sent[1].gain));
}

// The "Display performance stats" master grays its sub-rows when switched off
// and restores their authored colors when switched back on.
#[test]
fn perf_stats_master_grays_and_restores_the_sub_rows() {
    let mut f = Fixture::new();
    f.next(SettingKey::PerfStats);

    assert!(!f.state.resolved.perf_stats, "cycled On -> Off");
    assert_eq!(
        f.label_color(SUB_ROW_LABEL),
        super::rows::DISABLED_ROW_COLOR
    );

    f.next(SettingKey::PerfStats);
    assert!(f.state.resolved.perf_stats);
    assert_eq!(
        f.label_color(SUB_ROW_LABEL),
        LIT,
        "the authored color returns"
    );
}

// Leaving fullscreen grays the Resolution row (a windowed size comes from the
// window itself), and returning to windowed re-applies the remembered size the
// borderless/fullscreen modes left behind.
#[test]
fn window_mode_grays_resolution_and_restores_the_windowed_size() {
    let mut f = Fixture::new();
    f.state.window_args.mode = WindowMode::Fullscreen;
    f.state.window_args.width = 800;
    f.state.window_args.height = 600;

    // Fullscreen -> Windowed (the option order wraps Fullscreen back to index 0).
    f.apply(vec![cycle(SettingKey::WindowMode, SettingOp::Next)]);
    assert_eq!(f.state.window_args.mode, WindowMode::Windowed);
    assert!(f.saw(&Call::SetWindowMode(WindowMode::Windowed)));
    assert!(
        f.saw(&Call::SetWindowSize(800, 600)),
        "the remembered windowed size is re-applied"
    );
    assert_eq!(
        f.label_color(RESOLUTION_LABEL),
        super::rows::DISABLED_ROW_COLOR,
        "Resolution is inert outside fullscreen"
    );
}

// Entering fullscreen restores the Resolution row and does not resize the window.
#[test]
fn entering_fullscreen_restores_the_resolution_row() {
    let mut f = Fixture::new();
    f.apply(vec![cycle(
        SettingKey::WindowMode,
        SettingOp::SetIndex(levels::WINDOW_MODE.index_of(WindowMode::Fullscreen)),
    )]);

    assert_eq!(f.state.window_args.mode, WindowMode::Fullscreen);
    assert_eq!(f.label_color(RESOLUTION_LABEL), LIT);
    assert!(
        !f.calls
            .lock()
            .calls
            .iter()
            .any(|c| matches!(c, Call::SetWindowSize(..))),
        "only a return to windowed re-applies the size"
    );
}

// The Resolution row cycles the enumerated display-mode list and holds the
// display to the chosen mode.
#[test]
fn resolution_cycles_the_enumerated_display_modes() {
    let mut f = Fixture::new();
    let modes = [
        DisplayMode {
            width: 1920,
            height: 1080,
            refresh_hz: 60,
        },
        DisplayMode {
            width: 2560,
            height: 1440,
            refresh_hz: 165,
        },
    ];
    f.state.display_modes = modes.to_vec();
    f.state.current_mode = Some(modes[0]);

    f.next(SettingKey::Resolution);

    assert_eq!(f.state.resolution, Some(modes[1]));
    assert!(f.saw(&Call::SetDisplayMode(modes[1])));
    assert_eq!(f.label(VALUE_LABEL), modes[1].label());
    assert_eq!(
        f.persisted().graphics.resolution,
        Some([2560, 1440, 165]),
        "the mode persists as its three components"
    );
}

// With no enumerable modes (a backend that cannot list them) the row is inert.
#[test]
fn resolution_without_enumerated_modes_is_inert() {
    let mut f = Fixture::new();
    f.next(SettingKey::Resolution);

    assert!(f.state.resolution.is_none());
    assert!(f.calls.lock().calls.is_empty());
}

// A quality toggle flips the live feature, rebuilds it on the backend, and opts
// the master preset out to Custom (no ceiling clamps an explicit choice).
#[test]
fn quality_toggle_flips_the_master_preset_to_custom() {
    let mut f = Fixture::with_profile(GpuProfile::UNKNOWN);
    f.state.quality_preset = QualityPreset::High;
    f.next(SettingKey::Ssao);

    assert!(f.saw(&Call::ApplyQualitySettings));
    assert_eq!(f.state.quality_preset, QualityPreset::Custom);
    assert_eq!(f.label(QUALITY_LABEL), QualityPreset::Custom.name());
    let cfg = f.persisted();
    assert_eq!(cfg.graphics.quality_preset, Some(QualityPreset::Custom));
    assert!(cfg.graphics.ssao.is_some());
}

// Toggling auto-exposure off re-pushes the static post-process params, so the
// exposure the auto loop froze reverts to the authored / slider value.
#[test]
fn auto_exposure_toggle_repushes_the_post_process_params() {
    let mut f = Fixture::new();
    f.next(SettingKey::AutoExposure);

    assert!(f.saw(&Call::ApplyQualitySettings));
    assert!(f.saw(&Call::UpdatePostProcess), "exposure reverts");
}

// A cycle quality knob rebuilds through the same path as a toggle and likewise
// flips the preset to Custom.
#[test]
fn quality_cycle_knob_rebuilds_and_flips_to_custom() {
    let mut f = Fixture::new();
    f.next(SettingKey::SsgiRays);

    assert!(f.saw(&Call::ApplyQualitySettings));
    assert_eq!(f.state.quality_preset, QualityPreset::Custom);
    assert!(f.persisted().graphics.ssgi_rays.is_some());
}

// A quality rebuild that runs out of device memory raises the replay's memory
// pressure for the streaming valve; any other failure only logs.
#[test]
fn quality_rebuild_out_of_memory_raises_memory_pressure() {
    use crate::render::backend::error::RenderError;

    for (error, pressure) in [
        (
            RenderError::OutOfDeviceMemory("quality targets".to_string()),
            true,
        ),
        (RenderError::Other("pipeline rejected".to_string()), false),
    ] {
        let (calls, mut backend) = recording_backend();
        calls.lock().fail_quality = Some(error);
        let mut queue = RenderOps::default();
        let mut plan = settings::apply::ApplyPlan::default();
        plan.rebuild.insert(settings::apply::RebuildFeature::Ssao);
        super::record_plan(&mut queue, &settings::ResolvedGraphics::default(), plan);

        let outcome = queue.replay(&mut backend);
        assert_eq!(outcome.memory_pressure, pressure);
        assert!(calls.lock().saw(&Call::ApplyQualitySettings));
    }
}

// The AA mode also drives the composite FXAA flag, which rides the post-process
// params rather than the quality rebuild.
#[test]
fn aa_mode_cycle_refreshes_the_composite_fxaa_flag() {
    let mut f = Fixture::new();
    f.apply(vec![cycle(
        SettingKey::AaMode,
        SettingOp::SetIndex(levels::AA_MODE.index_of(AaMode::Fxaa)),
    )]);

    assert_eq!(f.state.resolved.graphics.anti_aliasing.mode, AaMode::Fxaa);
    assert_eq!(f.state.resolved.graphics.post_process().fxaa, 1.0);
    assert!(f.saw(&Call::UpdatePostProcess));

    f.apply(vec![cycle(
        SettingKey::AaMode,
        SettingOp::SetIndex(levels::AA_MODE.index_of(AaMode::Off)),
    )]);
    assert_eq!(f.state.resolved.graphics.post_process().fxaa, 0.0);
}

// The live shadow knobs (cadence, distance, cascades) reach the backend the same
// frame and flip the preset to Custom.
#[test]
fn live_shadow_knobs_push_to_the_backend() {
    let mut f = Fixture::new();

    f.next(SettingKey::ShadowUpdate);
    assert!(f.saw(&Call::SetShadowCadence(
        f.state.resolved.graphics.shadow_cadence()
    )));
    assert_eq!(
        f.state.resolved.graphics.shadows.update,
        levels::SHADOW_UPDATE.at(1)
    );

    f.next(SettingKey::ShadowDistance);
    assert!(f.saw(&Call::SetShadowCadence(
        f.state.resolved.graphics.shadow_cadence()
    )));

    f.next(SettingKey::ShadowCascades);
    assert!(f.saw(&Call::SetShadowCadence(
        f.state.resolved.graphics.shadow_cadence()
    )));

    assert_eq!(f.state.quality_preset, QualityPreset::Custom);
    let cfg = f.persisted();
    assert!(cfg.graphics.shadow_update.is_some());
    assert!(cfg.graphics.shadow_distance.is_some());
    assert!(cfg.graphics.shadow_cascades.is_some());
}

// The restart-required rows persist and relabel without touching the backend:
// the resources they size are built once at init.
#[test]
fn restart_required_rows_persist_without_a_backend_call() {
    let mut f = Fixture::new();
    for key in [
        SettingKey::RenderScale,
        SettingKey::ShadowMapSize,
        SettingKey::Anisotropy,
        SettingKey::TemporalUpscaling,
        SettingKey::HdrDisplay,
        SettingKey::HdrPq,
        SettingKey::FramesInFlight,
        SettingKey::OcclusionTwoPass,
        SettingKey::TextureQuality,
        SettingKey::UpscaleBackend,
        SettingKey::DlssPreset,
    ] {
        f.next(key);
    }

    assert!(
        f.calls.lock().calls.is_empty(),
        "no restart-required row applies live"
    );
    let cfg = f.persisted();
    assert!(cfg.graphics.render_scale.is_some());
    assert!(cfg.graphics.shadow_map_size.is_some());
    assert!(cfg.graphics.anisotropy.is_some());
    assert_eq!(cfg.graphics.temporal_upscaling, Some(true));
    assert_eq!(cfg.graphics.hdr_display, Some(true));
    assert_eq!(cfg.graphics.hdr_pq, Some(true));
    assert!(cfg.graphics.frames_in_flight.is_some());
    assert_eq!(cfg.graphics.occlusion_two_pass, Some(true));
    assert!(cfg.graphics.texture_cap.is_some());
    assert!(cfg.graphics.texture_budget.is_some());
    assert!(cfg.graphics.upscale_backend.is_some());
    assert_eq!(cfg.graphics.dlss_preset, Some(DlssPreset::K));
}

// The ceiling-governed restart rows still opt the master preset out to Custom.
#[test]
fn render_scale_flips_the_master_preset_to_custom() {
    let mut f = Fixture::new();
    f.state.quality_preset = QualityPreset::High;
    f.next(SettingKey::RenderScale);

    assert_eq!(f.state.quality_preset, QualityPreset::Custom);
    assert_eq!(f.label(QUALITY_LABEL), QualityPreset::Custom.name());
}

// The upscaler cycle skips backends this GPU vendor does not offer, so a
// non-NVIDIA / non-Intel device never lands on DLSS or XeSS.
#[test]
fn upscale_backend_cycle_skips_unavailable_vendors() {
    let mut f = Fixture::with_profile(GpuProfile::UNKNOWN);
    assert_eq!(f.state.gpu_profile.vendor, GpuVendor::Other);

    // Cycling the whole list only ever lands on the always-available options.
    for _ in 0..settings::options(SettingKey::UpscaleBackend).unwrap().len() * 2 {
        f.next(SettingKey::UpscaleBackend);
        assert!(
            registry::upscale_backend_available(
                f.state.resolved.graphics.anti_aliasing.upscaler,
                GpuVendor::Other
            ),
            "landed on an unavailable upscaler: {:?}",
            f.state.resolved.graphics.anti_aliasing.upscaler
        );
    }
}

// An NVIDIA device can reach DLSS, which the unknown-vendor cycle above skips.
#[test]
fn upscale_backend_cycle_reaches_dlss_on_nvidia() {
    let mut profile = GpuProfile::UNKNOWN;
    profile.vendor = GpuVendor::Nvidia;
    let mut f = Fixture::with_profile(profile);

    let mut seen = false;
    for _ in 0..settings::options(SettingKey::UpscaleBackend).unwrap().len() {
        f.next(SettingKey::UpscaleBackend);
        seen |= f.state.resolved.graphics.anti_aliasing.upscaler == UpscalerBackend::Dlss;
    }
    assert!(seen, "DLSS is reachable on an NVIDIA device");
}

// The DLSS preset row steps through every preset, persists the choice for
// the next launch, and changes nothing live, so it leaves the temporal history
// alone.
#[test]
fn dlss_preset_cycles_and_persists_without_resetting_history() {
    let mut f = Fixture::new();
    for want in [
        DlssPreset::K,
        DlssPreset::L,
        DlssPreset::M,
        DlssPreset::Default,
    ] {
        f.next(SettingKey::DlssPreset);
        assert_eq!(f.state.resolved.graphics.dlss_preset, want);
    }
    assert_eq!(
        f.persisted().graphics.dlss_preset,
        Some(DlssPreset::Default)
    );
    assert!(!PendingHistoryReset::take(&mut f.world.context()).any());
}

// A live quality change reports a history reset for the next frame.
#[test]
fn a_live_quality_change_requests_a_history_reset() {
    let mut f = Fixture::new();
    f.next(SettingKey::AaMode);
    assert_eq!(
        PendingHistoryReset::take(&mut f.world.context()),
        HistoryResetCauses::SETTINGS_CHANGE
    );
}

// The master preset is a ceiling over the world's authored look: picking a tier
// clears the per-row overrides and re-derives every governed row from the
// authored baseline under the new ceiling.
#[test]
fn graphics_quality_preset_clears_overrides_and_re_derives_the_rows() {
    let mut f = Fixture::new();
    // An authored look with the expensive features on, then a user override that
    // turned one off: picking a preset must re-derive from the authored baseline,
    // not from the override.
    f.state.authored.graphics.ambient_occlusion.enabled = true;
    f.state.authored.graphics.reflections.screen_space = true;
    f.state.resolved.graphics.ambient_occlusion.enabled = false;
    f.state
        .cycle_value_labels
        .insert(SettingKey::Ssao, TOGGLE_LABEL);

    f.apply(vec![SettingCommand {
        setting: SettingKey::GraphicsQuality,
        op: SettingOp::SetIndex(levels::GRAPHICS_QUALITY.index_of(QualityPreset::Ultra)),
        value_label: Some(QUALITY_LABEL),
        persist: true,
    }]);

    assert_eq!(f.state.quality_preset, QualityPreset::Ultra);
    assert!(
        f.state.resolved.graphics.ambient_occlusion.enabled,
        "the authored feature is restored, not the cleared override"
    );
    assert!(
        f.saw(&Call::ApplyQualitySettings),
        "the restored feature rebuilds"
    );
    assert_eq!(f.label(TOGGLE_LABEL), "On", "the dependent row relabeled");
    assert_eq!(
        f.label(QUALITY_LABEL),
        settings::quality::preset_label(QualityPreset::Ultra, &f.state.gpu_profile),
    );

    // Every per-row override is dropped, so the next launch re-resolves them from
    // the world + ceiling exactly as this live re-derive did.
    let cfg = f.persisted();
    assert_eq!(cfg.graphics.quality_preset, Some(QualityPreset::Ultra));
    assert_eq!(cfg.graphics.ssao, None);
    assert_eq!(cfg.graphics.ssr, None);
    assert_eq!(cfg.graphics.aa_mode, None);
    assert_eq!(cfg.graphics.shadow_map_size, None);
    assert_eq!(cfg.graphics.render_scale, None);
    assert_eq!(cfg.graphics.anisotropy, None);
}

// Look tuning is not preset-governed: a preset change keeps a persisted
// sub-quality slider, live and on disk, as the next launch will.
#[test]
fn graphics_quality_preset_keeps_the_look_tuning_sliders() {
    let mut f = Fixture::new();
    f.state.authored.graphics.ambient_occlusion.radius = 0.5;
    f.state.resolved.graphics.ambient_occlusion.radius = 1.5;
    f.world
        .resource_mut::<StoredSettings>()
        .unwrap()
        .graphics
        .ssao_radius = Some(1.5);

    f.apply(vec![SettingCommand {
        setting: SettingKey::GraphicsQuality,
        op: SettingOp::SetIndex(levels::GRAPHICS_QUALITY.index_of(QualityPreset::High)),
        value_label: Some(QUALITY_LABEL),
        persist: true,
    }]);

    assert_eq!(f.state.resolved.graphics.ambient_occlusion.radius, 1.5);
    assert_eq!(f.persisted().graphics.ssao_radius, Some(1.5));
}

// A Low preset clamps the authored look off: the ceiling never enables a feature
// the world did not author, but it does force expensive ones off.
// On a device without ray tracing a preset change resolves the RT row off and
// labels it unsupported, even with a stored choice to turn it on.
#[test]
fn a_preset_change_marks_an_unsupported_row() {
    let mut f = Fixture::new();
    f.state.capabilities = crate::render::backend::DeviceCapabilities {
        ray_tracing: false,
        ..crate::render::backend::DeviceCapabilities::ALL
    };
    f.state.authored.graphics.reflections.ray_traced = true;
    f.state.persisted_graphics.ray_traced_reflections = Some(true);
    f.state
        .cycle_value_labels
        .insert(SettingKey::RayTracedReflections, TOGGLE_LABEL);

    f.apply(vec![SettingCommand {
        setting: SettingKey::GraphicsQuality,
        op: SettingOp::SetIndex(levels::GRAPHICS_QUALITY.index_of(QualityPreset::Ultra)),
        value_label: Some(QUALITY_LABEL),
        persist: true,
    }]);

    assert!(!f.state.resolved.graphics.reflections.ray_traced);
    assert_eq!(f.label(TOGGLE_LABEL), "Off (not supported)");
}

#[test]
fn low_preset_clamps_the_authored_features_off() {
    let mut f = Fixture::new();
    f.state.authored.graphics.ambient_occlusion.enabled = true;
    f.state.authored.graphics.indirect_lighting.technique = IndirectTechnique::Ssgi;
    let authored = f.state.authored.graphics.clone();
    f.state.resolved.graphics.adopt_quality(&authored);

    f.apply(vec![SettingCommand {
        setting: SettingKey::GraphicsQuality,
        op: SettingOp::SetIndex(levels::GRAPHICS_QUALITY.index_of(QualityPreset::Low)),
        value_label: Some(QUALITY_LABEL),
        persist: true,
    }]);

    let ceiling = settings::quality::resolve_ceiling(QualityPreset::Low, &f.state.gpu_profile);
    assert!(!ceiling.ssgi, "the Low ceiling disallows SSGI");
    assert_eq!(
        f.state.resolved.graphics.indirect_lighting.technique,
        IndirectTechnique::Ibl,
        "the authored feature is clamped off"
    );
}

// A preset change coarsens the finest planet terrain cells live, and the
// streaming system reads the resolved size from the parked state.
#[test]
fn a_preset_change_reaches_the_planet_cell_size() {
    let mut f = Fixture::new();
    assert_eq!(f.state.resolved.graphics.planet_detail.finest_cell_size, 1);
    f.apply(vec![SettingCommand {
        setting: SettingKey::GraphicsQuality,
        op: SettingOp::SetIndex(levels::GRAPHICS_QUALITY.index_of(QualityPreset::Low)),
        value_label: Some(QUALITY_LABEL),
        persist: true,
    }]);
    assert_eq!(f.state.resolved.graphics.planet_detail.finest_cell_size, 4);

    let mut world = World::new();
    assert_eq!(
        super::planet_finest_cell_size(world.context().resources),
        1,
        "the section default before graphics init"
    );
    world.insert_resource(super::SettingsSlot::new(f.state));
    assert_eq!(super::planet_finest_cell_size(world.context().resources), 4);
}

// One revision serves a whole batch: every command's change rides the same
// write, and the next batch starts from the store the first left.
#[test]
fn a_batch_persists_once_and_carries_the_store_forward() {
    let mut f = Fixture::new();
    f.apply(vec![
        cycle(SettingKey::Vsync, SettingOp::Next),
        cycle(SettingKey::OcclusionTwoPass, SettingOp::Next),
    ]);
    assert_eq!(f.revision(), 1, "one write for the batch");
    let stored = f.persisted();
    assert_eq!(stored.graphics.vsync, Some(true));
    assert_eq!(stored.graphics.occlusion_two_pass, Some(true));

    f.next(SettingKey::ShowVram);
    let cfg = f.persisted();
    assert_eq!(
        cfg.graphics.vsync,
        Some(true),
        "the earlier change survives"
    );
    assert_eq!(cfg.graphics.occlusion_two_pass, Some(true));
    assert_eq!(cfg.graphics.show_vram, Some(true));
}

// A drain with nothing queued touches neither the backend nor the store, so an
// unchanged session never starts the writer thread.
#[test]
fn an_empty_drain_persists_nothing() {
    let mut f = Fixture::new();
    f.apply(Vec::new());

    assert!(f.calls.lock().calls.is_empty());
    assert_eq!(f.revision(), 0);
}

// The stats-HUD readouts are gated by their master toggle, so turning the master
// off hides both regardless of the sub-toggles' own values.
#[test]
fn hud_prefs_publish_under_the_master_toggle() {
    let mut f = Fixture::new();
    f.state.resolved.show_fps = true;
    f.state.resolved.show_vram = true;

    f.state.publish_hud_state(&mut f.world.context());
    let prefs = *f.world.context().resource::<HudPrefs>().unwrap();
    assert!(prefs.show_fps);
    assert!(prefs.show_vram);

    f.state.resolved.perf_stats = false;
    f.state.publish_hud_state(&mut f.world.context());
    let prefs = *f.world.context().resource::<HudPrefs>().unwrap();
    assert!(!prefs.show_fps, "the master gates the sub-readout");
    assert!(!prefs.show_vram);
}

// The rows the drain grayed are published as inert, so the gray-out and the
// input inertness stay in lockstep: the sub-rows under a disabled master, and
// Resolution outside fullscreen.
#[test]
fn disabled_rows_publish_alongside_the_gray_out() {
    let mut f = Fixture::new();
    f.state.window_args.mode = WindowMode::Fullscreen;
    f.state.publish_hud_state(&mut f.world.context());
    let rows = f
        .world
        .context()
        .resource::<crate::ecs::DisabledSettingRows>()
        .map(|r| r.0.clone())
        .unwrap();
    assert!(rows.is_empty(), "every row is live in fullscreen");

    f.state.resolved.perf_stats = false;
    f.state.window_args.mode = WindowMode::Windowed;
    f.state.publish_hud_state(&mut f.world.context());
    let rows = f
        .world
        .context()
        .resource::<crate::ecs::DisabledSettingRows>()
        .map(|r| r.0.clone())
        .unwrap();
    assert!(rows.contains(&SettingKey::ShowFps));
    assert!(rows.contains(&SettingKey::ShowVram));
    assert!(rows.contains(&SettingKey::Resolution));
}

// With no parked state (graphics init never succeeded) the step is a no-op and
// the queued commands wait in retention.
#[test]
fn step_without_a_parked_state_is_a_noop() {
    let mut f = Fixture::new();
    let mut sys = super::SettingsSystem::new();

    assert_eq!(sys.step(&mut f.world.context()), StepResult::Continue);
    assert!(
        f.world.context().resource::<HudPrefs>().is_none(),
        "nothing published without a state"
    );
}

// With a state but no backend (the editor transplanted it away) the state is
// parked again untouched rather than dropped.
#[test]
fn step_without_a_backend_puts_the_state_back() {
    let mut f = Fixture::new();
    let mut sys = super::SettingsSystem::new();
    f.world.insert_resource(super::SettingsSlot::new(f.state));

    assert_eq!(sys.step(&mut f.world.context()), StepResult::Continue);
    assert!(
        super::SettingsSlot::get(f.world.context().resources).is_some(),
        "the state is parked again for the next tick"
    );
}

// The full step drains into the parked op queue, publishes the HUD state, and
// parks both the queue and the state again for the next tick; replaying the
// queue applies the drained command to the backend.
#[test]
fn step_drains_into_the_op_queue_and_reparks() {
    let mut f = Fixture::new();
    let mut sys = super::SettingsSystem::new();
    {
        let mut ctx = f.world.context();
        ctx.events_mut::<SettingCommand>()
            .send(cycle(SettingKey::Vsync, SettingOp::Next));
    }
    f.world
        .insert_resource(crate::render::frame::queues::ActiveRenderQueues::new(
            crate::render::frame::queues::RenderQueues {
                ops: RenderOps::default(),
                slots: crate::render::slots::render_slots::RenderSlots::new(0, true, &[]),
            },
        ));
    f.world.insert_resource(super::SettingsSlot::new(f.state));

    assert_eq!(sys.step(&mut f.world.context()), StepResult::Continue);

    assert!(f.world.context().resource::<HudPrefs>().is_some());
    assert!(super::SettingsSlot::get(f.world.context().resources).is_some());
    let mut queues =
        crate::render::frame::queues::ActiveRenderQueues::take(f.world.context().resources)
            .expect("the op queue is parked again");
    queues.ops.replay(&mut f.backend);
    assert!(
        f.calls.lock().saw(&Call::SetVsync(true)),
        "the drained command reaches the backend at replay"
    );
}

// A trial graphics API is kept once a frame has drawn on it: the trial clears
// only after the first steps, only when the launch is trying one, and is saved
// once. A newer choice still untried survives for the next launch to try.
#[test]
fn a_drawn_frame_confirms_the_trial_graphics_api() {
    use crate::platform::{GraphicsApiStatus, Platform};
    use crate::settings::GraphicsApiTrial;
    let run = |trial: GraphicsApiTrial, confirm_pending: bool| {
        let mut world = World::new();
        let mut stored = StoredSettings::default();
        stored.graphics.graphics_api_trial = Some(trial);
        world.insert_resource(stored);
        world.insert_resource(GraphicsApiStatus {
            current: Platform::Vulkan,
            reverted: None,
            confirm_pending,
        });
        let mut state = SettingsState::for_tests();
        let mut trials = Vec::new();
        for _ in 0..3 {
            state.confirm_graphics_api(&mut world.context());
            let trial = world
                .resource::<StoredSettings>()
                .and_then(|s| s.graphics.graphics_api_trial);
            trials.push(trial);
        }
        let revision = world.resource::<SettingsRevision>().map_or(0, |r| r.0);
        (trials, revision)
    };
    let trying = Some(GraphicsApiTrial::Trying);
    let untried = Some(GraphicsApiTrial::Untried);
    assert_eq!(
        run(GraphicsApiTrial::Trying, true),
        (vec![trying, None, None], 1)
    );
    assert_eq!(
        run(GraphicsApiTrial::Trying, false),
        (vec![trying, trying, trying], 0)
    );
    assert_eq!(
        run(GraphicsApiTrial::Untried, true),
        (vec![untried, untried, untried], 0)
    );
}

// A host that published no store still gets the change live, and nothing is
// marked for writing. Debug builds flag the missing store.
#[test]
#[cfg_attr(
    debug_assertions,
    should_panic(expected = "the host publishes StoredSettings")
)]
fn a_change_without_a_published_store_applies_live_only() {
    let mut f = Fixture::new();
    let _ = f.world.context().resources.remove::<StoredSettings>();
    let vsync = f.state.resolved.graphics.frame_pacing.vsync;
    f.next(SettingKey::Vsync);
    assert_ne!(f.state.resolved.graphics.frame_pacing.vsync, vsync);
    assert_eq!(f.revision(), 0);
    assert!(f.world.resource::<StoredSettings>().is_none());
}

// A window opens titled, so only another configured mode is applied after it.
#[test]
fn only_a_non_windowed_mode_is_applied_at_startup() {
    let mut state = SettingsState::for_tests();
    assert_eq!(state.startup_window_mode(), None);
    state.window_args.mode = WindowMode::Borderless;
    assert_eq!(state.startup_window_mode(), Some(WindowMode::Borderless));
}

// The resolved texture cap and budget replace whatever the world authored.
#[test]
fn resolved_texture_limits_win_over_the_authored_streaming_config() {
    let mut state = SettingsState::for_tests();
    state.resolved.texture_cap = 512;
    state.resolved.texture_budget = 8;
    let mut config = StreamingConfig {
        texture_cap: 64,
        texture_budget: 2,
        ..Default::default()
    };
    state.apply_to_streaming(&mut config);
    assert_eq!((config.texture_cap, config.texture_budget), (512, 8));
}

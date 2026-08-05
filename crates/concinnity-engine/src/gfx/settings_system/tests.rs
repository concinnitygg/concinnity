// SettingsSystem unit tests: drive the SettingCommand drain against the
// recording mock backend (gfx::mock_backend) and a hand-assembled
// PipelineContext. No GPU device is created and the on-disk settings store is
// never read or written: the state is seeded with an in-memory settings cache
// and a writer whose sink captures the persisted snapshots instead.

use std::sync::{Arc, Mutex};

use crate::assets::{
    AudioCommand, ControlsCommand, IndirectLighting, Key, SettingCommand, SettingOp, ShadowUpdate,
    Sprite, TextLabel, WindowMode,
};
use crate::blob::BlobData;
use crate::config::Settings;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{ComponentStorage, PipelineContext, Resources, System};
use crate::gfx::backend::{GpuProfile, GpuVendor};
use crate::gfx::display_mode::DisplayMode;
use crate::gfx::graphics_system::{RebindViz, SliderViz};
use crate::gfx::keymap::{Bindable, KeyMap};
use crate::gfx::mock_backend::{Call, MockBackend, MockState, recording_backend};
use crate::gfx::profile::FrameProfile;
use crate::gfx::quality_preset::QualityPreset;
use crate::gfx::settings;

use super::SettingsState;
use super::writer::SettingsWriter;

// The value label ids the fixture wires up, so a test can assert a row relabelled
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

// Owns the storage a PipelineContext borrows from. Held separately from the
// state + backend so all three can be borrowed at once in `apply`.
struct World {
    components: ComponentStorage,
    blob: BlobData,
    profile: FrameProfile,
    resources: Resources,
    scratch: crate::ecs::Arena,
}

impl World {
    fn ctx(&mut self) -> PipelineContext<'_> {
        PipelineContext {
            components: &mut self.components,
            blob: &mut self.blob,
            profile: &mut self.profile,
            resources: &mut self.resources,
            frame: crate::ecs::FrameContext::new(&self.scratch),
        }
    }
}

struct Fixture {
    world: World,
    state: SettingsState,
    backend: MockBackend,
    calls: Arc<Mutex<MockState>>,
    // Every snapshot the writer's sink was handed, newest last.
    saved: Arc<Mutex<Vec<Settings>>>,
}

impl Fixture {
    fn new() -> Self {
        Self::with_profile(GpuProfile::UNKNOWN)
    }

    fn with_profile(gpu_profile: GpuProfile) -> Self {
        let (calls, backend) = recording_backend();
        let saved: Arc<Mutex<Vec<Settings>>> = Arc::default();
        let sink_log = Arc::clone(&saved);
        // Seeding both the cache and the writer keeps the drain off the disk: it
        // never falls back to `Settings::load` or spawns the real writer.
        let writer = SettingsWriter::with_sink(move |cfg| {
            sink_log.lock().unwrap().push(cfg.clone());
            Ok(())
        });

        let mut components = ComponentStorage::default();
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
            components.push_typed(TextLabel {
                asset_id: id,
                content: content.to_string(),
                color: LIT,
                ..Default::default()
            });
        }
        components.push_typed(Sprite {
            asset_id: HANDLE,
            x: 0.0,
            ..Default::default()
        });

        let cycle_value_labels = [
            ("graphics_quality".to_string(), QUALITY_LABEL),
            ("render_scale".to_string(), VALUE_LABEL),
        ]
        .into_iter()
        .collect();

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
            gamepad_map: crate::assets::GamepadMap::default(),
            pad_rebind_rows: vec![
                crate::gfx::graphics_system::PadRebindViz {
                    action: crate::assets::GamepadAction::Jump,
                    value_id: PAD_REBIND_LABEL,
                },
                crate::gfx::graphics_system::PadRebindViz {
                    action: crate::assets::GamepadAction::Sprint,
                    value_id: PAD_VICTIM_LABEL,
                },
            ],
            sliders: vec![SliderViz {
                key: "exposure".to_string(),
                track_x: 100.0,
                track_w: 200.0,
                handle_w: 20.0,
                handle_id: HANDLE,
                value_id: VALUE_LABEL,
            }],
            cycle_value_labels,
            post_process: crate::gfx::render_types::PostProcessParams::DEFAULT,
            post_config: Default::default(),
            authored_post_config: Default::default(),
            ambient_intensity: 1.0,
            quality_preset: QualityPreset::Custom,
            gpu_profile,
            render_scale: settings::render_scale_at(0),
            upscale_backend: settings::upscale_backend_at(0),
            temporal_upscaling: false,
            hdr_display: false,
            hdr_pq: false,
            shadow_map_size: settings::shadow_resolution_at(0),
            shadow_update: ShadowUpdate::EveryFrame,
            shadow_distance: settings::shadow_distance_at(0),
            shadow_cascades: settings::shadow_cascades_at(0),
            anisotropy: settings::anisotropy_at(0),
            authored_shadow_map_size: settings::shadow_resolution_at(0),
            authored_shadow_update: ShadowUpdate::EveryFrame,
            authored_shadow_distance: settings::shadow_distance_at(0),
            authored_shadow_cascades: settings::shadow_cascades_at(0),
            authored_anisotropy: settings::anisotropy_at(0),
            vsync: false,
            fps_cap: settings::fps_cap_at(0),
            perf_stats: true,
            show_fps: true,
            show_vram: false,
            perf_sub_row_labels: vec![(SUB_ROW_LABEL, LIT)],
            window_args: Default::default(),
            display_modes: Vec::new(),
            resolution: None,
            current_mode: None,
            resolution_row_labels: vec![(RESOLUTION_LABEL, LIT)],
            frames_in_flight: settings::frames_in_flight_at(0) as usize,
            occlusion_two_pass: false,
            texture_cap: settings::texture_quality_at(0).0,
            texture_budget: settings::texture_quality_at(0).1,
            settings_cache: Some(Settings::default()),
            settings_writer: Some(writer),
            scene_cmd_cursor: Default::default(),
            setting_cmd_cursor: Default::default(),
            published_hud_prefs: None,
            published_disabled_inputs: None,
        };

        Fixture {
            world: World {
                components,
                blob: BlobData::new(vec![Some(Vec::new())]),
                profile: FrameProfile::default(),
                resources: Resources::new(),
                scratch: crate::ecs::Arena::with_capacity(64 * 1024),
            },
            state,
            backend,
            calls,
            saved,
        }
    }

    // Queue `cmds` and run one drain over them.
    fn apply(&mut self, cmds: Vec<SettingCommand>) {
        {
            let mut ctx = self.world.ctx();
            let events = ctx.events_mut::<SettingCommand>();
            for cmd in cmds {
                events.send(cmd);
            }
        }
        let mut ctx = self.world.ctx();
        self.state
            .apply_setting_commands(&mut ctx, &mut self.backend);
    }

    // Cycle a row forward once, routing the value label back to the row.
    fn next(&mut self, setting: &str) {
        self.apply(vec![cycle(setting, SettingOp::Next)]);
    }

    // The settings snapshot the drain persisted last. The writer is dropped
    // first so its thread drains the queue before the log is read.
    fn persisted(&mut self) -> Settings {
        self.state.settings_writer = None;
        let saved = self.saved.lock().unwrap();
        saved.last().cloned().expect("a snapshot was persisted")
    }

    fn saw(&self, call: &Call) -> bool {
        self.calls.lock().unwrap().saw(call)
    }

    fn label(&mut self, id: AssetId) -> String {
        self.world
            .ctx()
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .map(|l| l.content.clone())
            .expect("label present")
    }

    fn label_color(&mut self, id: AssetId) -> [f32; 3] {
        self.world
            .ctx()
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .map(|l| l.color)
            .expect("label present")
    }

    // Every ControlsCommand the drain has sent, oldest first.
    fn sent_controls(&mut self) -> Vec<ControlsCommand> {
        self.world
            .ctx()
            .events::<ControlsCommand>()
            .map(|e| {
                e.read(&mut Default::default())
                    .into_iter()
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }
}

// A cycle-row command carrying the fixture's value label.
fn cycle(setting: &str, op: SettingOp) -> SettingCommand {
    SettingCommand {
        setting: setting.to_string(),
        op,
        value_label: Some(VALUE_LABEL),
        persist: true,
    }
}

// A slider command at `frac` of the track; `persist` marks the drag-release
// frame.
fn drag(setting: &str, frac: f32, persist: bool) -> SettingCommand {
    SettingCommand {
        setting: setting.to_string(),
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
    f.next("vsync");

    assert!(f.state.vsync, "the row cycled Off -> On");
    assert!(f.saw(&Call::SetVsync(true)), "applied live");
    assert_eq!(f.label(VALUE_LABEL), "On");
    assert_eq!(f.persisted().graphics.vsync, Some(true));
}

// Prev cycles the other way, wrapping around the option list.
#[test]
fn cycling_prev_wraps_the_option_list() {
    let mut f = Fixture::new();
    f.apply(vec![cycle("vsync", SettingOp::Prev)]);
    assert!(f.state.vsync, "Off wraps back to the last option");
}

// A dropdown pick jumps straight to an option index rather than stepping.
#[test]
fn set_index_jumps_to_the_chosen_option() {
    let mut f = Fixture::new();
    f.apply(vec![cycle("shadow_cascades", SettingOp::SetIndex(2))]);
    assert_eq!(f.state.shadow_cascades, settings::shadow_cascades_at(2));
}

// An unknown setting key is ignored: nothing is applied, nothing is persisted.
#[test]
fn unknown_setting_is_ignored() {
    let mut f = Fixture::new();
    f.next("not_a_setting");

    assert!(f.calls.lock().unwrap().calls.is_empty(), "nothing applied");
    assert!(
        f.saved.lock().unwrap().is_empty(),
        "an unknown key never persists"
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
        setting: Bindable::Forward.setting_key().to_string(),
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
        f.state.keymap.action_for_key(Key::Q).is_none(),
        "Q is unbound in the default map"
    );
    f.apply(vec![SettingCommand {
        setting: Bindable::Forward.setting_key().to_string(),
        op: SettingOp::Rebind(Key::Q),
        value_label: None,
        persist: true,
    }]);

    assert_eq!(f.state.keymap.get(Bindable::Forward), Key::Q);
    assert_eq!(f.label(REBIND_LABEL), Key::Q.display_name());
    assert_eq!(f.label(VICTIM_LABEL), "victim", "no victim was relabelled");
}

// A rebind naming an action that does not exist is ignored.
#[test]
fn rebind_of_an_unknown_action_is_ignored() {
    let mut f = Fixture::new();
    f.apply(vec![SettingCommand {
        setting: "not_an_action".to_string(),
        op: SettingOp::Rebind(Key::Q),
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
    f.apply(vec![drag("exposure", 1.0, false)]);

    let mid_drag = f.state.post_process.exposure;
    assert!(f.saw(&Call::UpdatePostProcess), "applied live mid-drag");
    assert!(
        f.saved.lock().unwrap().is_empty(),
        "an in-progress drag never writes"
    );

    f.apply(vec![drag("exposure", 1.0, true)]);
    assert_eq!(
        f.state.post_process.exposure, mid_drag,
        "same value applied"
    );
    assert!(
        f.persisted().graphics.exposure_ev.is_some(),
        "the release frame writes"
    );
}

// The slider stores the transformed render value but persists the authored one
// (exposure persists as EV, applies as a linear multiplier).
#[test]
fn exposure_slider_persists_ev_and_applies_the_multiplier() {
    let mut f = Fixture::new();
    // Fraction 1.0 is the top of the EV range.
    f.apply(vec![drag("exposure", 1.0, true)]);

    let ev = f.persisted().graphics.exposure_ev.expect("persisted");
    assert_eq!(
        f.state.post_process.exposure,
        settings::slider_apply_value("exposure", ev),
        "the live param is the EV mapped through the apply transform"
    );
    assert_eq!(
        f.label(VALUE_LABEL),
        settings::format_slider_value("exposure", ev)
    );
}

// The slider's handle slides to the dragged fraction along its track.
#[test]
fn slider_moves_the_handle_along_its_track() {
    let mut f = Fixture::new();
    f.apply(vec![drag("exposure", 0.5, false)]);

    let handle_x = f
        .world
        .ctx()
        .query::<Sprite>()
        .find(|s| s.asset_id == HANDLE)
        .map(|s| s.x)
        .expect("handle present");
    // track_x 100 + 0.5 * (track_w 200 - handle_w 20).
    assert_eq!(handle_x, 190.0);
}

// A fraction outside the track clamps rather than sliding the handle off it.
#[test]
fn slider_clamps_an_out_of_range_fraction() {
    let mut f = Fixture::new();
    f.apply(vec![drag("exposure", 2.0, false)]);

    let handle_x = f
        .world
        .ctx()
        .query::<Sprite>()
        .find(|s| s.asset_id == HANDLE)
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
            .ctx()
            .query::<Sprite>()
            .find(|s| s.asset_id == HANDLE)
            .map(|s| s.x)
            .expect("handle present")
    };
    // Place the handle mid-track (fraction 0.5 = x 190 on the 180px travel).
    f.apply(vec![drag("exposure", 0.5, true)]);
    assert_eq!(handle_x(&mut f), 190.0);

    f.apply(vec![cycle("exposure", SettingOp::Next)]);
    assert!(
        (handle_x(&mut f) - 199.0).abs() < 1.0e-3,
        "Next steps +5% of the travel"
    );
    assert!(f.persisted().graphics.exposure_ev.is_some());

    f.apply(vec![cycle("exposure", SettingOp::Prev)]);
    assert!(
        (handle_x(&mut f) - 190.0).abs() < 1.0e-3,
        "Prev steps back down"
    );

    // Stepping far past the bottom clamps at the track's left end.
    for _ in 0..25 {
        f.apply(vec![cycle("exposure", SettingOp::Prev)]);
    }
    assert_eq!(handle_x(&mut f), 100.0);
}

// An unknown slider key is ignored.
#[test]
fn unknown_slider_is_ignored() {
    let mut f = Fixture::new();
    f.apply(vec![drag("not_a_slider", 0.5, true)]);

    assert!(f.calls.lock().unwrap().calls.is_empty());
    assert!(f.saved.lock().unwrap().is_empty());
}

// A sub-quality slider rides the quality-params push (no pass rebuild) rather
// than the post-process push.
#[test]
fn quality_param_slider_updates_quality_params() {
    let mut f = Fixture::new();
    f.apply(vec![drag("ssao_radius", 0.5, true)]);

    assert!(f.saw(&Call::UpdateQualityParams));
    assert!(
        !f.saw(&Call::UpdatePostProcess),
        "a quality param is not a post-process param"
    );
    assert!(f.persisted().graphics.ssao_radius.is_some());
}

// Ambient scale lives in the light uniforms, so it takes its own setter on top
// of the post-process push.
#[test]
fn ambient_slider_takes_the_dedicated_setter() {
    let mut f = Fixture::new();
    f.apply(vec![drag("ambient_intensity", 0.25, true)]);

    let applied = f.state.ambient_intensity;
    assert!(f.saw(&Call::SetAmbientIntensity(applied)));
    assert!(f.persisted().graphics.ambient_intensity.is_some());
}

// Mouse sensitivity is a camera preference, not a render param: it travels as a
// ControlsCommand and persists the radians/pixel value the camera reads, not the
// 1..100 UI value.
#[test]
fn mouse_sensitivity_slider_sends_a_controls_command() {
    let mut f = Fixture::new();
    f.apply(vec![drag("mouse_sensitivity", 1.0, true)]);

    let sent: Vec<ControlsCommand> = f
        .world
        .ctx()
        .events::<ControlsCommand>()
        .map(|e| {
            e.read(&mut Default::default())
                .into_iter()
                .cloned()
                .collect()
        })
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

// FOV is likewise a camera preference; it persists the clamped degrees.
#[test]
fn fov_slider_sends_a_controls_command() {
    let mut f = Fixture::new();
    f.apply(vec![drag("fov", 0.0, true)]);

    let sent: Vec<ControlsCommand> = f
        .world
        .ctx()
        .events::<ControlsCommand>()
        .map(|e| {
            e.read(&mut Default::default())
                .into_iter()
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(sent.len(), 1);
    assert!(sent[0].fov_y_degrees.is_some());
    assert!(sent[0].mouse_sensitivity.is_none());
    assert_eq!(f.persisted().graphics.fov, sent[0].fov_y_degrees);
}

// A gamepad rebind binds the action to the captured button, hands the new map
// to InputSystem as a ControlsCommand, persists it, and relabels both the
// rebound row and the action it took the button from.
#[test]
fn pad_rebind_swaps_the_victim_and_relabels_both_rows() {
    use crate::assets::GamepadAction;
    let mut f = Fixture::new();
    let jump_button = f.state.gamepad_map.get(GamepadAction::Jump);
    let sprint_button = f.state.gamepad_map.get(GamepadAction::Sprint);

    f.apply(vec![SettingCommand {
        setting: GamepadAction::Jump.setting_key().to_string(),
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
        "the live map travels to InputSystem"
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
        setting: Bindable::Forward.setting_key().to_string(),
        op: SettingOp::RebindButton(crate::assets::GamepadButton::North),
        value_label: None,
        persist: true,
    }]);

    assert_eq!(f.state.gamepad_map, crate::assets::GamepadMap::default());
    assert!(f.saved.lock().unwrap().is_empty(), "nothing persisted");
}

// The gamepad sliders travel as ControlsCommands and persist the applied
// values: radians/second for the look sensitivity, the deflection fraction for
// the deadzone (not their UI-scale values).
#[test]
fn gamepad_sliders_send_controls_commands_and_persist_applied_values() {
    let mut f = Fixture::new();
    f.apply(vec![
        drag("gamepad_look_sensitivity", 1.0, true),
        drag("gamepad_deadzone", 0.5, true),
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

// The frame-rate cap applies through the republished resource the App-level
// pacer reads, with no backend call.
#[test]
fn fps_cap_publishes_the_frame_rate_cap_resource() {
    let mut f = Fixture::new();
    f.next("fps_cap");

    let published = f
        .world
        .ctx()
        .resource::<crate::ecs::FrameRateCap>()
        .map(|c| c.0)
        .expect("cap published");
    assert_eq!(published, f.state.fps_cap);
    assert_eq!(f.persisted().graphics.fps_cap, Some(f.state.fps_cap));
}

// Master volume is owned by AudioSystem, so the change travels as an
// AudioCommand it drains this same tick.
#[test]
fn master_volume_sends_an_audio_command() {
    let mut f = Fixture::new();
    f.next("master_volume");

    let sent: Vec<AudioCommand> = f
        .world
        .ctx()
        .events::<AudioCommand>()
        .map(|e| {
            e.read(&mut Default::default())
                .into_iter()
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(sent.len(), 1);
    assert_eq!(
        f.persisted().audio.master_volume,
        Some(sent[0].master_volume)
    );
}

// The "Display performance stats" master grays its sub-rows when switched off
// and restores their authored colors when switched back on.
#[test]
fn perf_stats_master_grays_and_restores_the_sub_rows() {
    let mut f = Fixture::new();
    f.next("perf_stats");

    assert!(!f.state.perf_stats, "cycled On -> Off");
    assert_eq!(
        f.label_color(SUB_ROW_LABEL),
        super::rows::DISABLED_ROW_COLOR
    );

    f.next("perf_stats");
    assert!(f.state.perf_stats);
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
    f.apply(vec![cycle("window_mode", SettingOp::Next)]);
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
        "window_mode",
        SettingOp::SetIndex(settings::window_mode_index(WindowMode::Fullscreen)),
    )]);

    assert_eq!(f.state.window_args.mode, WindowMode::Fullscreen);
    assert_eq!(f.label_color(RESOLUTION_LABEL), LIT);
    assert!(
        !f.calls
            .lock()
            .unwrap()
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

    f.next("resolution");

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
    f.next("resolution");

    assert!(f.state.resolution.is_none());
    assert!(f.calls.lock().unwrap().calls.is_empty());
}

// A quality toggle flips the live feature, rebuilds it on the backend, and opts
// the master preset out to Custom (no ceiling clamps an explicit choice).
#[test]
fn quality_toggle_flips_the_master_preset_to_custom() {
    let mut f = Fixture::with_profile(GpuProfile::UNKNOWN);
    f.state.quality_preset = QualityPreset::High;
    f.next("ssao");

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
    f.next("auto_exposure");

    assert!(f.saw(&Call::ApplyQualitySettings));
    assert!(f.saw(&Call::UpdatePostProcess), "exposure reverts");
}

// A cycle quality knob rebuilds through the same path as a toggle and likewise
// flips the preset to Custom.
#[test]
fn quality_cycle_knob_rebuilds_and_flips_to_custom() {
    let mut f = Fixture::new();
    f.next("ssgi_rays");

    assert!(f.saw(&Call::ApplyQualitySettings));
    assert_eq!(f.state.quality_preset, QualityPreset::Custom);
    assert!(f.persisted().graphics.ssgi_rays.is_some());
}

// The AA mode also drives the composite FXAA flag, which rides the post-process
// params rather than the quality rebuild.
#[test]
fn aa_mode_cycle_refreshes_the_composite_fxaa_flag() {
    let mut f = Fixture::new();
    f.apply(vec![cycle(
        "aa_mode",
        SettingOp::SetIndex(settings::aa_mode_index(crate::assets::AaMode::Fxaa)),
    )]);

    assert_eq!(f.state.post_config.aa_mode, crate::assets::AaMode::Fxaa);
    assert_eq!(f.state.post_process.fxaa, 1.0);
    assert!(f.saw(&Call::UpdatePostProcess));

    f.apply(vec![cycle(
        "aa_mode",
        SettingOp::SetIndex(settings::aa_mode_index(crate::assets::AaMode::Off)),
    )]);
    assert_eq!(f.state.post_process.fxaa, 0.0);
}

// The live shadow knobs (cadence, distance, cascades) reach the backend the same
// frame and flip the preset to Custom.
#[test]
fn live_shadow_knobs_push_to_the_backend() {
    let mut f = Fixture::new();

    f.next("shadow_update");
    assert!(f.saw(&Call::SetShadowUpdate));
    assert_eq!(f.state.shadow_update, settings::shadow_update_at(1));

    f.next("shadow_distance");
    assert!(f.saw(&Call::SetShadowDistance(f.state.shadow_distance)));

    f.next("shadow_cascades");
    assert!(f.saw(&Call::SetShadowCascades(f.state.shadow_cascades)));

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
        "render_scale",
        "shadow_map_size",
        "anisotropy",
        "temporal_upscaling",
        "hdr_display",
        "hdr_pq",
        "frames_in_flight",
        "occlusion_two_pass",
        "texture_quality",
        "upscale_backend",
    ] {
        f.next(key);
    }

    assert!(
        f.calls.lock().unwrap().calls.is_empty(),
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
}

// The ceiling-governed restart rows still opt the master preset out to Custom.
#[test]
fn render_scale_flips_the_master_preset_to_custom() {
    let mut f = Fixture::new();
    f.state.quality_preset = QualityPreset::High;
    f.next("render_scale");

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
    for _ in 0..settings::options("upscale_backend").unwrap().len() * 2 {
        f.next("upscale_backend");
        assert!(
            settings::upscale_backend_available(f.state.upscale_backend, GpuVendor::Other),
            "landed on an unavailable upscaler: {:?}",
            f.state.upscale_backend
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
    for _ in 0..settings::options("upscale_backend").unwrap().len() {
        f.next("upscale_backend");
        seen |= f.state.upscale_backend == crate::assets::UpscalerBackend::Dlss;
    }
    assert!(seen, "DLSS is reachable on an NVIDIA device");
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
    f.state.authored_post_config.ssao = true;
    f.state.authored_post_config.ssr = true;
    f.state.post_config.ssao = false;
    f.state
        .cycle_value_labels
        .insert("ssao".to_string(), TOGGLE_LABEL);

    f.apply(vec![SettingCommand {
        setting: "graphics_quality".to_string(),
        op: SettingOp::SetIndex(crate::gfx::quality_preset::preset_index(
            QualityPreset::Ultra,
        )),
        value_label: Some(QUALITY_LABEL),
        persist: true,
    }]);

    assert_eq!(f.state.quality_preset, QualityPreset::Ultra);
    assert!(
        f.state.post_config.ssao,
        "the authored feature is restored, not the cleared override"
    );
    assert!(f.saw(&Call::ApplyQualitySettings));
    assert!(f.saw(&Call::UpdatePostProcess));
    assert!(f.saw(&Call::SetShadowUpdate));
    assert!(f.saw(&Call::SetShadowDistance(f.state.shadow_distance)));
    assert!(f.saw(&Call::SetShadowCascades(f.state.shadow_cascades)));
    assert_eq!(f.label(TOGGLE_LABEL), "On", "the dependent row relabelled");
    assert_eq!(
        f.label(QUALITY_LABEL),
        crate::gfx::quality_preset::preset_label(QualityPreset::Ultra, &f.state.gpu_profile),
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

// A Low preset clamps the authored look off: the ceiling never enables a feature
// the world did not author, but it does force expensive ones off.
#[test]
fn low_preset_clamps_the_authored_features_off() {
    let mut f = Fixture::new();
    f.state.authored_post_config.ssao = true;
    f.state.authored_post_config.indirect_lighting = IndirectLighting::Ssgi;
    f.state.post_config = f.state.authored_post_config.clone();

    f.apply(vec![SettingCommand {
        setting: "graphics_quality".to_string(),
        op: SettingOp::SetIndex(crate::gfx::quality_preset::preset_index(QualityPreset::Low)),
        value_label: Some(QUALITY_LABEL),
        persist: true,
    }]);

    let ceiling =
        crate::gfx::quality_preset::resolve_ceiling(QualityPreset::Low, &f.state.gpu_profile);
    assert!(!ceiling.ssgi, "the Low ceiling disallows SSGI");
    assert_eq!(
        f.state.post_config.indirect_lighting,
        IndirectLighting::Ibl,
        "the authored feature is clamped off"
    );
}

// One snapshot serves a whole batch: every command's change rides the same
// write, and the in-memory cache carries it forward so a later change never
// re-reads a stale value from disk.
#[test]
fn a_batch_persists_once_and_carries_the_cache_forward() {
    let mut f = Fixture::new();
    f.apply(vec![
        cycle("vsync", SettingOp::Next),
        cycle("occlusion_two_pass", SettingOp::Next),
    ]);

    let cached = f.state.settings_cache.clone().expect("cache retained");
    assert_eq!(cached.graphics.vsync, Some(true));
    assert_eq!(cached.graphics.occlusion_two_pass, Some(true));

    // A second batch starts from the cache, so the first batch's values survive.
    f.next("show_vram");
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

    assert!(f.calls.lock().unwrap().calls.is_empty());
    assert!(f.saved.lock().unwrap().is_empty());
}

// The stats-HUD readouts are gated by their master toggle, so turning the master
// off hides both regardless of the sub-toggles' own values.
#[test]
fn hud_prefs_publish_under_the_master_toggle() {
    let mut f = Fixture::new();
    f.state.show_fps = true;
    f.state.show_vram = true;

    f.state.publish_hud_state(&mut f.world.ctx());
    let prefs = *f.world.ctx().resource::<crate::ecs::HudPrefs>().unwrap();
    assert!(prefs.show_fps);
    assert!(prefs.show_vram);

    f.state.perf_stats = false;
    f.state.publish_hud_state(&mut f.world.ctx());
    let prefs = *f.world.ctx().resource::<crate::ecs::HudPrefs>().unwrap();
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
    f.state.publish_hud_state(&mut f.world.ctx());
    let rows = f
        .world
        .ctx()
        .resource::<crate::ecs::DisabledSettingRows>()
        .map(|r| r.0.clone())
        .unwrap();
    assert!(rows.is_empty(), "every row is live in fullscreen");

    f.state.perf_stats = false;
    f.state.window_args.mode = WindowMode::Windowed;
    f.state.publish_hud_state(&mut f.world.ctx());
    let rows = f
        .world
        .ctx()
        .resource::<crate::ecs::DisabledSettingRows>()
        .map(|r| r.0.clone())
        .unwrap();
    assert!(rows.contains("show_fps"));
    assert!(rows.contains("show_vram"));
    assert!(rows.contains("resolution"));
}

// With no parked state (graphics init never succeeded) the step is a no-op and
// the queued commands wait in retention.
#[test]
fn step_without_a_parked_state_is_a_noop() {
    let mut f = Fixture::new();
    let mut sys = super::SettingsSystem::new();

    assert_eq!(
        sys.step(&mut f.world.ctx()),
        crate::ecs::StepResult::Continue
    );
    assert!(
        f.world.ctx().resource::<crate::ecs::HudPrefs>().is_none(),
        "nothing published without a state"
    );
}

// With a state but no backend (the editor transplanted it away) the state is
// parked again untouched rather than dropped.
#[test]
fn step_without_a_backend_puts_the_state_back() {
    let mut f = Fixture::new();
    let mut sys = super::SettingsSystem::new();
    f.world.resources.insert(f.state);

    assert_eq!(
        sys.step(&mut f.world.ctx()),
        crate::ecs::StepResult::Continue
    );
    assert!(
        f.world.ctx().resources.get_mut::<SettingsState>().is_some(),
        "the state is parked again for the next tick"
    );
}

// The full step drains against the parked backend, publishes the HUD state, and
// parks both the backend and the state again for the next tick.
#[test]
fn step_drains_against_the_parked_backend_and_reparks() {
    let mut f = Fixture::new();
    let mut sys = super::SettingsSystem::new();
    {
        let mut ctx = f.world.ctx();
        ctx.events_mut::<SettingCommand>()
            .send(cycle("vsync", SettingOp::Next));
    }
    crate::ecs::ActiveRenderBackend::put(&mut f.world.resources, Box::new(f.backend));
    f.world.resources.insert(f.state);

    assert_eq!(
        sys.step(&mut f.world.ctx()),
        crate::ecs::StepResult::Continue
    );

    assert!(
        f.calls.lock().unwrap().saw(&Call::SetVsync(true)),
        "the command reached the backend"
    );
    assert!(f.world.ctx().resource::<crate::ecs::HudPrefs>().is_some());
    assert!(f.world.ctx().resources.get_mut::<SettingsState>().is_some());
    assert!(
        crate::ecs::ActiveRenderBackend::take(&mut f.world.resources).is_some(),
        "the backend is parked again"
    );
}

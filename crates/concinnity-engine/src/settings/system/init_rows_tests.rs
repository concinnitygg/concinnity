// The settings menu as graphics init leaves it: every row's value label synced
// to the live value of the state init resolves, the rows a runtime change
// relabels captured, and the rows the device cannot provide grayed.

use crate::components::{
    GraphicsConfig, HitRegion, InputKey, ScrollPanel, ScrollRow, SettingCommand, SettingOp,
    ShadowUpdate, Sprite, TextLabel, UiAction,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Ref, World};
use crate::input::keymap::{Bindable, KeyMap};
use crate::launch::LaunchRequest;
use crate::render::backend::{DeviceCapabilities, GpuProfile, GpuTier, GpuVendor};
use crate::render::frame::ops::RenderOps;
use crate::render::output::dlss::DlssPreset;
use crate::settings::{QualityPreset, SettingKey, SettingsRevision, StoredSettings};
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;

use super::init_tests::{Launch, effects, profile_at};
use super::{DISABLED_ROW_COLOR, SettingsState, gate_unavailable_rows};

// The authored row color every label starts at, so a gray is a visible change.
const LIT: [f32; 3] = [0.9, 0.9, 0.9];

const PLACEHOLDER: &str = "<placeholder>";

fn act(action: serde_json::Value) -> UiAction {
    serde_json::from_value(action).unwrap()
}

fn placeholder_label(world: &mut World, id: AssetId) {
    world.push_identified(
        id,
        TextLabel {
            content: PLACEHOLDER.to_string(),
            color: LIT,
            ..Default::default()
        },
    );
}

// One settings row: the setting-action HitRegion the menu builds plus the
// value TextLabel it points at.
fn push_row(world: &mut World, key: &str, verb: &str, label: AssetId) {
    world.push(HitRegion {
        action: Some(act(
            serde_json::json!({"setting": {"key": key, "verb": verb}}),
        )),
        label: Some(Ref::new(label)),
        ..Default::default()
    });
    placeholder_label(world, label);
}

// A slider row: a drag region spanning x = 0..100 with a 10-wide handle.
fn push_slider(world: &mut World, key: &str, handle: AssetId, label: AssetId) {
    world.push(HitRegion {
        action: Some(act(
            serde_json::json!({"setting": {"key": key, "verb": "drag"}}),
        )),
        x: 0.0,
        width: 100.0,
        drag_handle: Some(Ref::new(handle)),
        label: Some(Ref::new(label)),
        ..Default::default()
    });
    world.push_identified(
        handle,
        Sprite {
            width: 10.0,
            ..Default::default()
        },
    );
    placeholder_label(world, label);
}

fn label_text(world: &mut World, id: AssetId) -> String {
    world
        .context()
        .get_by_id::<TextLabel>(id)
        .expect("label present")
        .content
        .clone()
}

fn label_color(world: &mut World, id: AssetId) -> [f32; 3] {
    world
        .context()
        .get_by_id::<TextLabel>(id)
        .expect("label present")
        .color
}

fn handle_x(world: &mut World, id: AssetId) -> f32 {
    world
        .context()
        .get_by_id::<Sprite>(id)
        .expect("handle sprite present")
        .x
}

// Whether each region bound to `key` is disabled.
fn disabled_regions(world: &World, key: SettingKey) -> Vec<bool> {
    world
        .query::<HitRegion>()
        .filter(|r| matches!(r.action, Some(UiAction::Setting { key: k, .. }) if k == key))
        .map(|r| r.disabled)
        .collect()
}

// The state `launch` resolves, with the rows in `world` synced to it.
fn synced(world: &mut World, launch: &Launch) -> SettingsState {
    let mut state = launch.state();
    state.sync_menu_labels(&mut world.context(), &launch.stored);
    state
}

// Every row's value label shows its live value, and the master preset row
// carries the tier `Auto` resolved to, which the static option table cannot
// express.
#[test]
fn settings_rows_show_their_live_values_at_init() {
    let mut stored = StoredSettings::default();
    let g = &mut stored.graphics;
    g.vsync = Some(true);
    g.fps_cap = Some(60);
    g.shadow_map_size = Some(4096);
    g.shadow_update = Some(ShadowUpdate::EveryFrame);
    g.shadow_distance = Some(160);
    g.shadow_cascades = Some(3);
    g.anisotropy = Some(16);
    g.frames_in_flight = Some(3);
    g.occlusion_two_pass = Some(true);
    g.texture_cap = Some(384);
    g.hdr_display = Some(true);
    g.ssao = Some(true);
    g.ssgi_rays = Some(4);

    let mut world = World::new();
    let rows: Vec<(&str, &str, AssetId)> = vec![
        ("vsync", "next", AssetId(100)),
        ("fps_cap", "next", AssetId(101)),
        ("window_mode", "next", AssetId(102)),
        ("render_scale", "next", AssetId(103)),
        ("master_volume", "next", AssetId(104)),
        ("temporal_upscaling", "next", AssetId(105)),
        ("hdr_display", "next", AssetId(106)),
        ("hdr_pq", "next", AssetId(107)),
        ("perf_stats", "next", AssetId(108)),
        ("show_fps", "next", AssetId(109)),
        ("show_vram", "next", AssetId(110)),
        ("shadow_map_size", "next", AssetId(111)),
        ("shadow_update", "next", AssetId(112)),
        ("shadow_distance", "next", AssetId(113)),
        ("shadow_cascades", "next", AssetId(114)),
        ("anisotropy", "next", AssetId(115)),
        ("frames_in_flight", "next", AssetId(116)),
        ("occlusion_two_pass", "next", AssetId(117)),
        ("texture_quality", "next", AssetId(118)),
        ("ssao", "next", AssetId(119)),
        ("ssgi_rays", "open", AssetId(120)),
        ("graphics_quality", "next", AssetId(121)),
    ];
    for &(key, verb, label) in &rows {
        push_row(&mut world, key, verb, label);
    }
    let launch = Launch::new(effects(|_| {}))
        .stored(stored)
        .on(profile_at(GpuTier::MidDiscrete));
    let live = synced(&mut world, &launch);

    assert_eq!(label_text(&mut world, AssetId(100)), "On", "stored vsync");
    assert_eq!(label_text(&mut world, AssetId(101)), "60");
    assert_eq!(label_text(&mut world, AssetId(102)), "Windowed");
    assert_eq!(label_text(&mut world, AssetId(106)), "On", "hdr_display");
    assert_eq!(label_text(&mut world, AssetId(107)), "Off", "hdr_pq unset");
    assert_eq!(label_text(&mut world, AssetId(111)), "4096");
    assert_eq!(label_text(&mut world, AssetId(114)), "3");
    assert_eq!(label_text(&mut world, AssetId(115)), "16x");
    assert_eq!(label_text(&mut world, AssetId(116)), "3");
    assert_eq!(label_text(&mut world, AssetId(117)), "On", "occlusion");
    assert_eq!(label_text(&mut world, AssetId(119)), "On", "quality toggle");
    assert_eq!(label_text(&mut world, AssetId(120)), "4", "cycle dropdown");
    assert_eq!(label_text(&mut world, AssetId(121)), "Auto (High)");
    for &(key, _, label) in &rows {
        assert_ne!(
            label_text(&mut world, label),
            PLACEHOLDER,
            "{key} still shows the build placeholder"
        );
    }
    // The cycle rows' value labels are captured for the runtime relabel.
    assert_eq!(
        live.cycle_value_label(SettingKey::SsgiRays),
        Some(AssetId(120))
    );
    assert_eq!(
        live.cycle_value_label(SettingKey::Vsync),
        Some(AssetId(100))
    );
}

// Each slider's handle sits at the value's fraction along the track and its
// label shows the formatted value. The exposure slider stores 2^EV, so the
// sync recovers the EV to place the handle.
#[test]
fn slider_rows_sync_their_handle_and_label_to_the_live_value() {
    let mut stored = StoredSettings::default();
    stored.graphics.quality_preset = Some(QualityPreset::Custom);
    stored.graphics.exposure_ev = Some(0.0);
    stored.graphics.vignette = Some(0.5);
    let mut world = World::new();
    push_slider(&mut world, "exposure", AssetId(200), AssetId(201));
    push_slider(&mut world, "vignette", AssetId(202), AssetId(203));
    // A drag region missing its handle and label is skipped.
    world.push(HitRegion {
        action: Some(act(
            serde_json::json!({"setting": {"key": "exposure", "verb": "drag"}}),
        )),
        ..Default::default()
    });
    let live = synced(&mut world, &Launch::new(effects(|_| {})).stored(stored));

    // Exposure spans -4..+4 EV, so 0 EV is mid-track: 0 + 0.5 * (100 - 10).
    assert_eq!(handle_x(&mut world, AssetId(200)), 45.0);
    assert_eq!(label_text(&mut world, AssetId(201)), "+0.0 EV");
    assert_eq!(handle_x(&mut world, AssetId(202)), 45.0);
    assert_ne!(label_text(&mut world, AssetId(203)), PLACEHOLDER);

    let rows = live.slider_rows();
    assert!(rows.iter().any(|(k, _)| *k == SettingKey::Vignette));
    let exposure = rows
        .iter()
        .find(|(k, _)| *k == SettingKey::Exposure)
        .expect("the exposure slider is captured");
    assert_eq!(exposure.1, [0.0, 100.0, 10.0]);
}

// Every graphics slider recovers a live value, so no row shows the build
// placeholder.
#[test]
fn every_owned_slider_key_recovers_a_live_value() {
    let keys = [
        "exposure",
        "bloom_intensity",
        "bloom_threshold",
        "bloom_knee",
        "vignette",
        "lut_strength",
        "ambient_intensity",
        "ssao_radius",
        "ssao_intensity",
        "ssr_intensity",
        "ssr_max_distance",
        "ssgi_intensity",
        "ssgi_max_distance",
        "auto_exposure_min_ev",
        "auto_exposure_max_ev",
        "auto_exposure_speed",
    ];
    let label = |i: usize| AssetId(301 + u32::saturating_from(i) * 2);
    let mut world = World::new();
    for (i, key) in keys.iter().enumerate() {
        push_slider(
            &mut world,
            key,
            AssetId(300 + u32::saturating_from(i) * 2),
            label(i),
        );
    }
    let live = synced(&mut world, &Launch::new(effects(|_| {})));

    for (i, key) in keys.iter().enumerate() {
        assert_ne!(
            label_text(&mut world, label(i)),
            PLACEHOLDER,
            "{key} never synced to a live value"
        );
    }
    assert_eq!(live.slider_rows().len(), keys.len());
}

// Each rebind row shows its bound key (the stored rebind, or the default), and
// the rows are captured for the live rebind drain.
#[test]
fn rebind_rows_show_their_bound_keys_at_init() {
    let mut stored = StoredSettings::default();
    stored.controls.keymap = Some(KeyMap {
        forward: InputKey::Up,
        ..Default::default()
    });
    let mut world = World::new();
    for (i, action) in Bindable::ALL.iter().enumerate() {
        let label = AssetId(400 + u32::saturating_from(i));
        world.push(HitRegion {
            action: Some(act(
                serde_json::json!({"setting": {"key": action.setting_key(), "verb": "rebind"}}),
            )),
            label: Some(Ref::new(label)),
            ..Default::default()
        });
        placeholder_label(&mut world, label);
    }
    let live = synced(
        &mut world,
        &Launch::new(GraphicsConfig::default()).stored(stored),
    );

    assert_eq!(
        label_text(&mut world, AssetId(400)),
        InputKey::Up.display_name(),
        "the stored forward rebind shows on its row"
    );
    for i in 1..Bindable::ALL.len() {
        assert_ne!(
            label_text(&mut world, AssetId(400 + u32::saturating_from(i))),
            PLACEHOLDER,
            "row {i} never synced to its default binding"
        );
    }
    assert_eq!(live.rebind_row_count(), Bindable::ALL.len());
    assert_eq!(
        live.keymap().forward,
        InputKey::Up,
        "the stored map is live"
    );
}

// A row the device cannot honor is disabled and marked unsupported, and the
// gray expands from its value label to every element of its scroll row.
#[test]
fn a_capability_gated_row_grays_out_its_whole_scroll_row() {
    let mut world = World::new();
    push_row(&mut world, "upscale_backend", "next", AssetId(600));
    world.push_identified(
        AssetId(601),
        TextLabel {
            content: "Upscaler".to_string(),
            color: LIT,
            ..Default::default()
        },
    );
    push_row(&mut world, "vsync", "next", AssetId(602));
    world.push(ScrollPanel {
        rows: vec![
            ScrollRow {
                elements: vec![Ref::new(AssetId(600)), Ref::new(AssetId(601))],
                ..Default::default()
            },
            ScrollRow {
                elements: vec![Ref::new(AssetId(602))],
                ..Default::default()
            },
        ],
        ..Default::default()
    });
    synced(&mut world, &Launch::new(effects(|_| {})));
    let caps = DeviceCapabilities {
        selectable_upscaler: false,
        ..DeviceCapabilities::ALL
    };
    gate_unavailable_rows(&mut world.context(), &caps, GpuProfile::UNKNOWN.vendor);

    assert_eq!(label_color(&mut world, AssetId(600)), DISABLED_ROW_COLOR);
    assert_eq!(
        label_color(&mut world, AssetId(601)),
        DISABLED_ROW_COLOR,
        "the gray expands to the row's name label, not just its value"
    );
    assert_eq!(label_color(&mut world, AssetId(602)), LIT);
    assert!(label_text(&mut world, AssetId(600)).ends_with(" (not supported)"));
    assert!(!label_text(&mut world, AssetId(602)).contains("not supported"));
    assert_eq!(
        disabled_regions(&world, SettingKey::UpscaleBackend),
        vec![true]
    );
}

// The perf-stats master toggle grays its two sub-rows when off, and the
// Resolution row grays outside fullscreen; the captured labels carry the
// authored colors a restore reads back.
#[test]
fn master_toggles_gray_the_rows_they_govern() {
    let mut stored = StoredSettings::default();
    stored.graphics.perf_stats = Some(false);
    let mut world = World::new();
    push_row(&mut world, "show_fps", "next", AssetId(700));
    push_row(&mut world, "show_vram", "next", AssetId(701));
    push_row(&mut world, "resolution", "open", AssetId(702));
    push_row(&mut world, "vsync", "next", AssetId(703));
    world.push(ScrollPanel {
        rows: [700, 701, 702, 703]
            .into_iter()
            .map(|id| ScrollRow {
                elements: vec![Ref::new(AssetId(id))],
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    });
    let live = synced(
        &mut world,
        &Launch::new(GraphicsConfig::default()).stored(stored),
    );

    assert_eq!(label_color(&mut world, AssetId(700)), DISABLED_ROW_COLOR);
    assert_eq!(label_color(&mut world, AssetId(701)), DISABLED_ROW_COLOR);
    assert_eq!(
        label_color(&mut world, AssetId(702)),
        DISABLED_ROW_COLOR,
        "the Resolution row grays outside fullscreen"
    );
    assert_eq!(label_color(&mut world, AssetId(703)), LIT);
    assert!(live.perf_sub_row_labels().contains(&(AssetId(700), LIT)));
    assert!(live.resolution_row_labels().contains(&(AssetId(702), LIT)));
}

// `--dlss-preset` outranks the stored preset for the launch: the state builds
// with it and the row shows it, while the store keeps the saved choice until
// the row itself is cycled.
#[test]
fn a_forced_dlss_preset_reaches_the_menu_unpersisted() {
    const ROW_LABEL: AssetId = AssetId(700);
    let mut saved = StoredSettings::default();
    saved.graphics.dlss_preset = Some(DlssPreset::K);
    // An NVIDIA GPU, where the DLSS preset row is live.
    let nvidia = GpuProfile {
        vendor: GpuVendor::Nvidia,
        ..GpuProfile::UNKNOWN
    };
    let launch = |forced: Option<DlssPreset>| {
        let mut launch = Launch::new(effects(|_| {}))
            .stored(saved.clone())
            .on(nvidia);
        launch.request = LaunchRequest {
            dlss_preset: forced,
            ..Default::default()
        };
        let mut world = World::new();
        push_row(&mut world, "dlss_preset", "next", ROW_LABEL);
        let state = synced(&mut world, &launch);
        gate_unavailable_rows(
            &mut world.context(),
            &DeviceCapabilities::ALL,
            GpuVendor::Nvidia,
        );
        (world, state)
    };
    let shown = |world: &mut World, live: &SettingsState| {
        (
            (
                live.resolved().graphics.dlss_preset,
                live.persisted_graphics().dlss_preset,
            ),
            label_text(world, ROW_LABEL),
        )
    };

    let (mut world, live) = launch(None);
    assert_eq!(
        shown(&mut world, &live),
        ((DlssPreset::K, Some(DlssPreset::K)), "K".to_string())
    );

    let (mut world, mut live) = launch(Some(DlssPreset::M));
    assert_eq!(
        shown(&mut world, &live),
        ((DlssPreset::M, Some(DlssPreset::K)), "M".to_string())
    );

    // Cycling the row steps from the forced value and saves the preset alone.
    world.insert_resource(saved.clone());
    world.events_mut::<SettingCommand>().send(SettingCommand {
        setting: SettingKey::DlssPreset,
        op: SettingOp::Next,
        value_label: Some(ROW_LABEL),
        persist: true,
    });
    live.apply_setting_commands(&mut world.context(), &mut RenderOps::default());
    assert_eq!(live.resolved().graphics.dlss_preset, DlssPreset::Default);
    assert_eq!(label_text(&mut world, ROW_LABEL), "Default");
    let mut expected = saved;
    expected.graphics.dlss_preset = Some(DlssPreset::Default);
    assert_eq!(world.resource::<StoredSettings>(), Some(&expected));
    assert!(
        world
            .resource::<SettingsRevision>()
            .is_some_and(|r| r.0 > 0)
    );
}

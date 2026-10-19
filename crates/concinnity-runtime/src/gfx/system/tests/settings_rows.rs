use super::*;
use concinnity_core::components::{InputKey, ScrollPanel, ScrollRow, ShadowUpdate};
use concinnity_core::input::keymap::{Bindable, KeyMap};
use concinnity_core::settings::system::DISABLED_ROW_COLOR;

// The color of the TextLabel with `id`.
fn label_color(world: &mut World, id: AssetId) -> [f32; 3] {
    world
        .context()
        .get_by_id::<TextLabel>(id)
        .expect("label present")
        .color
}

// Every settings row's value label is synced to its live value before the first
// render, so a persisted or authored choice shows instead of the build's
// placeholder. The master preset row carries the resolved tier under `Auto`,
// which the static option table cannot express.
#[test]
fn settings_rows_show_their_live_values_at_init() {
    let mut settings = concinnity_core::settings::StoredSettings::default();
    settings.graphics.vsync = Some(true);
    settings.graphics.fps_cap = Some(60);
    settings.graphics.shadow_map_size = Some(4096);
    settings.graphics.shadow_update = Some(ShadowUpdate::EveryFrame);
    settings.graphics.shadow_distance = Some(160);
    settings.graphics.shadow_cascades = Some(3);
    settings.graphics.anisotropy = Some(16);
    settings.graphics.frames_in_flight = Some(3);
    settings.graphics.occlusion_two_pass = Some(true);
    settings.graphics.texture_cap = Some(384);
    settings.graphics.hdr_display = Some(true);
    settings.graphics.ssao = Some(true);
    settings.graphics.ssgi_rays = Some(4);

    // A classified GPU, so the Auto preset resolves to a named tier below.
    let (_state, hooks) = recording_hooks_with(profile_at(GpuTier::MidDiscrete));
    let mut b = effects_scene(Default::default());
    // (key, verb, label id) for a spread across every value-label arm: booleans,
    // discrete numeric levels, enums, a quality toggle, and a cycle dropdown.
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
        push_settings_row(&mut b, key, verb, label);
    }
    let mut world = b.build();
    world.insert_resource(settings);
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    assert_eq!(
        label_text(&mut world, AssetId(100)),
        "On",
        "persisted vsync"
    );
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
    assert_eq!(
        label_text(&mut world, AssetId(121)),
        "Auto (High)",
        "the master row carries the tier the Auto preset resolved to, \
         which the static option table cannot express"
    );
    // Every non-dynamic row's label moved off the build's placeholder.
    for &(key, _, label) in &rows {
        if key == "not_a_setting" {
            continue;
        }
        assert_ne!(
            label_text(&mut world, label),
            "<placeholder>",
            "{key} still shows the build placeholder"
        );
    }
    // The cycle rows' value labels are captured for the runtime relabel, since
    // the HitRegions are drained right after init.
    let live = settings_state(&world);
    assert_eq!(
        live.cycle_value_label(SettingKey::SsgiRays),
        Some(AssetId(120))
    );
    assert_eq!(
        live.cycle_value_label(SettingKey::Vsync),
        Some(AssetId(100))
    );
}

// Each slider row's handle position and value label are synced to the live value
// at init: the handle sits at the value's fraction along the track, and the label
// shows the formatted value. The exposure slider stores 2^EV, so the sync must
// recover the EV to place the handle.
#[test]
fn slider_rows_sync_their_handle_and_label_to_the_live_value() {
    let mut settings = concinnity_core::settings::StoredSettings::default();
    settings.graphics.quality_preset = Some(QualityPreset::Custom);
    settings.graphics.exposure_ev = Some(0.0);
    settings.graphics.vignette = Some(0.5);
    let (_state, hooks) = recording_hooks();

    let mut b = effects_scene(Default::default());
    // A drag region spanning x = 0..100 with a 10-wide handle: the handle travels
    // the 90 units between the track's start and its own right edge.
    for (key, handle, label) in [
        ("exposure", AssetId(200), AssetId(201)),
        ("vignette", AssetId(202), AssetId(203)),
    ] {
        b.push(HitRegion {
            action: act(serde_json::json!({"setting": {"key": key, "verb": "drag"}})),
            x: 0.0,
            width: 100.0,
            drag_handle: Some(Ref::new(handle)),
            label: Some(Ref::new(label)),
            ..Default::default()
        });
        b.push_identified(
            handle,
            Sprite {
                width: 10.0,
                ..Default::default()
            },
        );
        b.push_identified(
            label,
            TextLabel {
                content: "<placeholder>".to_string(),
                ..Default::default()
            },
        );
    }
    // A drag region missing its handle / label is skipped rather than panicking.
    b.push(HitRegion {
        action: act(serde_json::json!({"setting": {"key": "exposure", "verb": "drag"}})),
        ..Default::default()
    });
    let mut world = b.build();
    world.insert_resource(settings);
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let handle_x = |world: &mut World, id: AssetId| {
        world
            .context()
            .get_by_id::<Sprite>(id)
            .expect("handle sprite present")
            .x
    };
    // Exposure spans -4..+4 EV, so 0 EV is mid-track: 0 + 0.5 * (100 - 10).
    assert_eq!(handle_x(&mut world, AssetId(200)), 45.0);
    assert_eq!(label_text(&mut world, AssetId(201)), "+0.0 EV");
    // Vignette spans 0..1, so 0.5 is also mid-track.
    assert_eq!(handle_x(&mut world, AssetId(202)), 45.0);
    assert_ne!(label_text(&mut world, AssetId(203)), "<placeholder>");

    // The captured rows are handed to SettingsSystem for the live drag drain.
    let live = settings_state(&world);
    let rows = live.slider_rows();
    assert!(rows.iter().any(|(k, _)| *k == SettingKey::Vignette));
    let exposure = rows
        .iter()
        .find(|(k, _)| *k == SettingKey::Exposure)
        .expect("the exposure slider is captured");
    assert_eq!(exposure.1, [0.0, 100.0, 10.0]);
}

// Every slider this system owns recovers a live value, so no row silently shows
// the build placeholder. Sliders backed by the persisted store (mouse
// sensitivity, FOV) are excluded: reading them would hit the on-disk settings
// file, which these tests never touch.
#[test]
fn every_owned_slider_key_recovers_a_live_value() {
    let (_state, hooks) = recording_hooks();
    let mut b = effects_scene(Default::default());
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
    for (i, key) in keys.iter().enumerate() {
        let handle = AssetId(300 + i as u32 * 2);
        let label = AssetId(301 + i as u32 * 2);
        b.push(HitRegion {
            action: act(serde_json::json!({"setting": {"key": key, "verb": "drag"}})),
            x: 0.0,
            width: 100.0,
            drag_handle: Some(Ref::new(handle)),
            label: Some(Ref::new(label)),
            ..Default::default()
        });
        b.push_identified(
            handle,
            Sprite {
                width: 10.0,
                ..Default::default()
            },
        );
        b.push_identified(
            label,
            TextLabel {
                content: "<placeholder>".to_string(),
                ..Default::default()
            },
        );
    }
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    for (i, key) in keys.iter().enumerate() {
        let label = AssetId(301 + i as u32 * 2);
        assert_ne!(
            label_text(&mut world, label),
            "<placeholder>",
            "{key} never synced to a live value"
        );
    }
    assert_eq!(settings_state(&world).slider_rows().len(), keys.len());
}

// Each Controls-tab rebind row's value label is synced to the live bound key
// (the persisted rebind, or the engine default), and the rows are captured for
// the live rebind drain.
#[test]
fn rebind_rows_show_their_bound_keys_at_init() {
    let mut settings = concinnity_core::settings::StoredSettings::default();
    settings.controls.keymap = Some(KeyMap {
        forward: InputKey::Up,
        ..Default::default()
    });
    let (_state, hooks) = recording_hooks();

    let mut b = scene_builder();
    for (i, action) in Bindable::ALL.iter().enumerate() {
        let label = AssetId(400 + i as u32);
        b.push(HitRegion {
            action: act(
                serde_json::json!({"setting": {"key": action.setting_key(), "verb": "rebind"}}),
            ),
            label: Some(Ref::new(label)),
            ..Default::default()
        });
        b.push_identified(
            label,
            TextLabel {
                content: "<placeholder>".to_string(),
                ..Default::default()
            },
        );
    }
    let mut world = b.build();
    world.insert_resource(settings);
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    assert_eq!(
        label_text(&mut world, AssetId(400)),
        InputKey::Up.display_name(),
        "the persisted forward rebind shows on its row"
    );
    for i in 1..Bindable::ALL.len() {
        assert_ne!(
            label_text(&mut world, AssetId(400 + i as u32)),
            "<placeholder>",
            "row {i} never synced to its default binding"
        );
    }
    let live = settings_state(&world);
    assert_eq!(live.rebind_row_count(), Bindable::ALL.len());
    assert_eq!(
        live.keymap().forward,
        InputKey::Up,
        "the persisted map is the live one"
    );
}

// Every element a ScrollPanel lists is mapped to its panel's content band, so
// the draw path can scissor scroll rows to the panel and an off-band row does
// not bleed over the chrome. The bands are handed to OverlaySystem at init.
#[test]
fn scroll_panel_rows_clip_their_elements_to_the_panel_band() {
    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(ScrollPanel {
        x: 10.0,
        y: 20.0,
        width: 300.0,
        height: 400.0,
        rows: vec![
            ScrollRow {
                elements: vec![Ref::new(AssetId(500)), Ref::new(AssetId(501))],
                ..Default::default()
            },
            ScrollRow {
                elements: vec![Ref::new(AssetId(502))],
                ..Default::default()
            },
        ],
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let clips = &world
        .resource::<concinnity_core::render::overlay::OverlayAssets>()
        .expect("OverlayAssets parked at init")
        .clip_rects;
    let band = [10.0, 20.0, 300.0, 400.0];
    for id in [500, 501, 502] {
        assert_eq!(
            clips.get(&AssetId(id)),
            Some(&band),
            "element {id} clips to its panel's band"
        );
    }
    assert!(
        clips.get(&AssetId(503)).is_none(),
        "an unlisted element is unclipped"
    );
}

// A settings row whose feature the backend cannot honor is grayed out and made
// inert, and the gray expands from its value label to the whole scroll row
// (background, name, value, steppers) so the row reads as unavailable as a unit.
// Driven here by a device that reports a fixed upscaler, which grays the
// upscaler-selector row.
#[test]
fn a_capability_gated_row_grays_out_its_whole_scroll_row() {
    let (state, hooks) = recording_hooks();
    state.lock().caps = backend::DeviceCapabilities {
        selectable_upscaler: false,
        ..backend::DeviceCapabilities::ALL
    };
    let mut b = effects_scene(Default::default());
    // The gated row: name + value labels, both listed in one scroll row.
    push_settings_row(&mut b, "upscale_backend", "next", AssetId(600));
    b.push_identified(
        AssetId(601),
        TextLabel {
            content: "Upscaler".to_string(),
            color: LIT,
            ..Default::default()
        },
    );
    // An ungated row alongside it, which must stay lit.
    push_settings_row(&mut b, "vsync", "next", AssetId(602));
    b.push(ScrollPanel {
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
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    assert_eq!(
        label_color(&mut world, AssetId(600)),
        DISABLED,
        "the gated row's value label"
    );
    assert_eq!(
        label_color(&mut world, AssetId(601)),
        DISABLED,
        "the gray expands to the row's name label, not just its value"
    );
    assert_eq!(
        label_color(&mut world, AssetId(602)),
        LIT,
        "an available row stays lit"
    );
    assert!(
        label_text(&mut world, AssetId(600)).ends_with(" (not supported)"),
        "the gated row's value reads as unsupported"
    );
    assert!(!label_text(&mut world, AssetId(602)).contains("not supported"));
    // A disabled region is dropped by UiInputSystem, so it never hovers or fires.
    let disabled: Vec<bool> = world
        .context()
        .query::<HitRegion>()
        .filter(|r| {
            matches!(
                r.action,
                Some(UiAction::Setting {
                    key: SettingKey::UpscaleBackend,
                    ..
                })
            )
        })
        .map(|r| r.disabled)
        .collect();
    assert_eq!(disabled, vec![true]);
}

// The DLSS preset row is live wherever the upscaler row could pick DLSS (an
// NVIDIA GPU on a selectable-upscaler backend), whichever upscaler built, and
// grayed everywhere else.
#[test]
fn the_dlss_preset_row_follows_whether_dlss_is_selectable() {
    let preset_row_disabled = |vendor| {
        let mut profile = GpuProfile::UNKNOWN;
        profile.vendor = vendor;
        let (_, hooks) = recording_hooks_with(profile);
        let mut b = effects_scene(Default::default());
        push_settings_row(&mut b, "dlss_preset", "next", AssetId(600));
        let mut world = b.build();
        let gs = init_graphics(&mut world, hooks);
        assert!(!gs.failed);
        let disabled: Vec<bool> = world
            .context()
            .query::<HitRegion>()
            .filter(|r| {
                matches!(
                    r.action,
                    Some(UiAction::Setting {
                        key: SettingKey::DlssPreset,
                        ..
                    })
                )
            })
            .map(|r| r.disabled)
            .collect();
        disabled
    };
    assert_eq!(preset_row_disabled(GpuVendor::Nvidia), vec![false]);
    assert_eq!(preset_row_disabled(GpuVendor::Amd), vec![true]);
}

// The muted gray a disabled settings row is recolored to.
const DISABLED: [f32; 3] = DISABLED_ROW_COLOR;

// The master "Display performance stats" toggle grays its two sub-rows when it
// is off, and the Resolution row grays outside fullscreen (windowed sizes come
// from the window itself, so the row does not apply).
#[test]
fn master_toggles_gray_the_rows_they_govern() {
    let mut settings = concinnity_core::settings::StoredSettings::default();
    settings.graphics.perf_stats = Some(false);
    let (_state, hooks) = recording_hooks();

    let mut b = scene_builder();
    push_settings_row(&mut b, "show_fps", "next", AssetId(700));
    push_settings_row(&mut b, "show_vram", "next", AssetId(701));
    push_settings_row(&mut b, "resolution", "open", AssetId(702));
    push_settings_row(&mut b, "vsync", "next", AssetId(703));
    b.push(ScrollPanel {
        rows: [700, 701, 702, 703]
            .into_iter()
            .map(|id| ScrollRow {
                elements: vec![Ref::new(AssetId(id))],
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    });
    let mut world = b.build();
    world.insert_resource(settings);
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    assert_eq!(
        label_color(&mut world, AssetId(700)),
        DISABLED,
        "show_fps grays under the master toggle"
    );
    assert_eq!(label_color(&mut world, AssetId(701)), DISABLED, "show_vram");
    assert_eq!(
        label_color(&mut world, AssetId(702)),
        DISABLED,
        "the Resolution row grays outside fullscreen"
    );
    assert_eq!(
        label_color(&mut world, AssetId(703)),
        LIT,
        "an unrelated row"
    );
    // The captured labels carry the authored colors the restore reads back.
    let live = settings_state(&world);
    assert!(live.perf_sub_row_labels().contains(&(AssetId(700), LIT)));
    assert!(live.resolution_row_labels().contains(&(AssetId(702), LIT)));
}

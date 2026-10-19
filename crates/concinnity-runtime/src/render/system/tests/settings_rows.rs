use super::*;
use concinnity_core::components::{ScrollPanel, ScrollRow};
use concinnity_core::settings::system::DISABLED_ROW_COLOR;

// The color of the TextLabel with `id`.
fn label_color(world: &mut World, id: AssetId) -> [f32; 3] {
    world
        .context()
        .get_by_id::<TextLabel>(id)
        .expect("label present")
        .color
}

// Init syncs the world's settings rows to the state it resolved before the
// first render: the stored vsync choice shows, the master preset row names the
// tier the detected GPU resolved `Auto` to, and the parked state holds the
// captured rows.
#[test]
fn settings_rows_show_their_live_values_at_init() {
    let mut settings = concinnity_core::settings::StoredSettings::default();
    settings.graphics.vsync = Some(true);
    let (_state, hooks) = recording_hooks_with(profile_at(GpuTier::MidDiscrete));
    let mut b = effects_scene(ResolvedGraphics::default());
    push_settings_row(&mut b, "vsync", "next", AssetId(100));
    push_settings_row(&mut b, "graphics_quality", "next", AssetId(121));
    let mut world = b.build();
    world.insert_resource(settings);
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    assert_eq!(label_text(&mut world, AssetId(100)), "On", "stored vsync");
    assert_eq!(label_text(&mut world, AssetId(121)), "Auto (High)");
    assert_eq!(
        settings_state(&world).cycle_value_label(SettingKey::Vsync),
        Some(AssetId(100))
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

// The built backend's capabilities gate the settings rows: a device that
// reports a fixed upscaler grays the upscaler-selector row and makes it inert.
#[test]
fn the_backend_capabilities_gate_the_settings_rows() {
    let (state, hooks) = recording_hooks();
    state.lock().caps = backend::DeviceCapabilities {
        selectable_upscaler: false,
        ..backend::DeviceCapabilities::ALL
    };
    let mut b = effects_scene(ResolvedGraphics::default());
    push_settings_row(&mut b, "upscale_backend", "next", AssetId(600));
    push_settings_row(&mut b, "vsync", "next", AssetId(602));
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    assert_eq!(
        label_color(&mut world, AssetId(600)),
        DISABLED,
        "the gated row's value label"
    );
    assert_eq!(
        label_color(&mut world, AssetId(602)),
        LIT,
        "an available row stays lit"
    );
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
        let mut b = effects_scene(ResolvedGraphics::default());
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

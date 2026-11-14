// The rows a launch flag fixes for the run: which flag fixes which row, and the
// menu gray-out that shows them inert with the flag's value.

use crate::components::{
    AaMode, GraphicsConfig, HitRegion, TextLabel, UpscaleQuality, UpscalerBackend, WindowMode,
};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;
use crate::ecs::reference::Ref;
use crate::launch::{LaunchArgs, LaunchRequest};
use crate::render::output::dlss::DlssPreset;
use crate::settings::{QualityPreset, SettingKey};
use alloc::string::ToString;
use alloc::vec::Vec;
use concinnity_core::platform::Platform;

use super::init_tests::Launch;
use super::rows::DISABLED_ROW_COLOR;

fn launched(args: LaunchArgs) -> super::SettingsState {
    let mut launch = Launch::new(GraphicsConfig::default());
    launch.request = LaunchRequest {
        args,
        ..Default::default()
    };
    launch.state()
}

fn fixed(args: LaunchArgs) -> Vec<SettingKey> {
    let state = launched(args);
    SettingKey::ALL
        .into_iter()
        .filter(|&key| state.fixed_by_launch(key))
        .collect()
}

// A launch without flags fixes nothing.
#[test]
fn a_plain_launch_fixes_no_row() {
    assert!(fixed(LaunchArgs::NONE).is_empty());
}

// Each settings flag fixes its own row.
#[test]
fn each_flag_fixes_its_own_row() {
    let cases = [
        (
            LaunchArgs {
                vsync: Some(true),
                ..LaunchArgs::NONE
            },
            SettingKey::Vsync,
        ),
        (
            LaunchArgs {
                limit_fps: Some(30),
                ..LaunchArgs::NONE
            },
            SettingKey::FpsCap,
        ),
        (
            LaunchArgs {
                window_mode: Some(WindowMode::Borderless),
                ..LaunchArgs::NONE
            },
            SettingKey::WindowMode,
        ),
        (
            LaunchArgs {
                graphics_api: Some(Platform::Vulkan),
                ..LaunchArgs::NONE
            },
            SettingKey::GraphicsApi,
        ),
        (
            LaunchArgs {
                upscaling: Some(false),
                ..LaunchArgs::NONE
            },
            SettingKey::TemporalUpscaling,
        ),
        (
            LaunchArgs {
                rt_reflections: Some(true),
                ..LaunchArgs::NONE
            },
            SettingKey::RayTracedReflections,
        ),
        (
            LaunchArgs {
                dlss_preset: Some(DlssPreset::K),
                ..LaunchArgs::NONE
            },
            SettingKey::DlssPreset,
        ),
        (
            LaunchArgs {
                aa_mode: Some(AaMode::Fxaa),
                ..LaunchArgs::NONE
            },
            SettingKey::AaMode,
        ),
        (
            LaunchArgs {
                upscaler: Some(UpscalerBackend::Fsr3),
                ..LaunchArgs::NONE
            },
            SettingKey::UpscaleBackend,
        ),
        (
            LaunchArgs {
                render_scale: Some(UpscaleQuality::Performance),
                ..LaunchArgs::NONE
            },
            SettingKey::RenderScale,
        ),
        (
            LaunchArgs {
                ssao: Some(false),
                ..LaunchArgs::NONE
            },
            SettingKey::Ssao,
        ),
        (
            LaunchArgs {
                ssr: Some(true),
                ..LaunchArgs::NONE
            },
            SettingKey::Ssr,
        ),
        (
            LaunchArgs {
                ssgi: Some(false),
                ..LaunchArgs::NONE
            },
            SettingKey::Ssgi,
        ),
    ];
    for (args, key) in cases {
        assert_eq!(fixed(args), [key]);
    }
}

// A preset given at launch fixes the preset row and every row it governs, and
// leaves the rest free.
#[test]
fn a_launch_preset_fixes_every_row_it_governs() {
    let keys = fixed(LaunchArgs {
        graphics_quality: Some(QualityPreset::Low),
        ..LaunchArgs::NONE
    });
    let expected: Vec<SettingKey> = SettingKey::ALL
        .into_iter()
        .filter(|&key| key == SettingKey::GraphicsQuality || key.preset_governed())
        .collect();
    assert_eq!(keys, expected);
}

// A fixed row is disabled and grayed with its value left as the flag set it;
// a free row stays live.
#[test]
fn fixed_rows_are_grayed_and_inert() {
    let mut world = World::new();
    for (id, content) in [(1, "On"), (2, "60")] {
        world.push_identified(
            AssetId(id),
            TextLabel {
                content: content.to_string(),
                ..Default::default()
            },
        );
    }
    for (key, label) in [("vsync", 1), ("fps_cap", 2)] {
        world.push(HitRegion {
            action: Some(
                serde_json::from_value(
                    serde_json::json!({"setting": {"key": key, "verb": "next"}}),
                )
                .unwrap(),
            ),
            label: Some(Ref::new(AssetId(label))),
            ..Default::default()
        });
    }
    let state = launched(LaunchArgs {
        vsync: Some(true),
        ..LaunchArgs::NONE
    });

    state.gate_launch_fixed_rows(&mut world.context());

    let disabled: Vec<bool> = world.query::<HitRegion>().map(|r| r.disabled).collect();
    assert_eq!(disabled, [true, false]);
    let label = |world: &mut World, id| {
        world
            .context()
            .get_by_id::<TextLabel>(AssetId(id))
            .cloned()
            .unwrap()
    };
    let vsync = label(&mut world, 1);
    assert_eq!(vsync.content, "On");
    assert_eq!(vsync.color, DISABLED_ROW_COLOR);
    assert_ne!(label(&mut world, 2).color, DISABLED_ROW_COLOR);
}

// Regraying the SSR row after a preset change or an RT toggle keeps a
// launch-fixed SSR row grayed even with ray-traced reflections off.
#[test]
fn a_launch_fixed_ssr_row_stays_grayed_when_regrayed() {
    let authored = [0.8, 0.8, 0.8];
    let mut world = World::new();
    world.push_identified(
        AssetId(1),
        TextLabel {
            color: authored,
            ..Default::default()
        },
    );
    let mut state = launched(LaunchArgs {
        ssr: Some(false),
        ..LaunchArgs::NONE
    });
    state.ssr_row_labels = alloc::vec![(AssetId(1), authored)];
    state.resolved.graphics.reflections.ray_traced = false;

    state.gray_superseded_ssr_row(&mut world.context());

    let color = world
        .context()
        .get_by_id::<TextLabel>(AssetId(1))
        .map(|label| label.color);
    assert_eq!(color, Some(DISABLED_ROW_COLOR));
}

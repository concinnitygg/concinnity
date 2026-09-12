// src/editor/hook/tests/edit/lighting_tests.rs
//
// The Lighting panel's actions (`hook/edit/lighting.rs`): the seeding an open
// performs, the commit an apply makes and what it does with text that will not
// parse, the immediate commit a bool toggle makes, the missing singleton an add
// row appends, and the focus it yields when it is not frontmost.

use concinnity_core::components::FrameInput;
use concinnity_core::components::Sprite;
use concinnity_core::ecs::World;

use crate::debug_hook::DebugHook;

use crate::editor::hook::tests::fixtures::hook;

use crate::editor::inject;

use crate::editor::panels::lighting;
use crate::editor::panels::lighting_panel;
use crate::editor::panels::list_panel;

use crate::editor::panels::registry::PanelKey;

use crate::editor::widget;

// Lighting panel

fn sun_entry() -> serde_json::Value {
    serde_json::json!({"name": "sun", "type": "DirectionalLight", "args": {
        "direction": [-0.35, 0.85, 0.35], "color": [1.0, 0.96, 0.86], "intensity": 2.2
    }})
}

fn fog_entry(enabled: bool) -> serde_json::Value {
    serde_json::json!({"name": "fog", "type": "VolumetricFog", "args": {
        "enabled": enabled, "density": 0.02
    }})
}

// Global binding indices pinned by `lighting::SECTIONS` declaration order.
const SUN_INTENSITY: usize = 4;
fn fog_enabled_binding() -> usize {
    lighting::section_base(1)
}
fn fog_density_binding() -> usize {
    lighting::section_base(1) + 2
}

// The View panel's Lighting row (registry order: Assets, Preview, Templates,
// Lighting) opens the panel through the real tick + click route and seeds its
// controls from the entries.
#[test]
fn lighting_opens_via_the_view_panel_and_seeds() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    world.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    let mut h = hook(vec![sun_entry()]);
    h.view_open = true;
    let vp = [1280.0, 720.0];
    let vo = h.origin(PanelKey::View, vp);
    let row = list_panel::row_rect(vo, 200.0, 3);
    // Click the row's interior (clear of the edge resize band).
    assert!(h.try_panel_press(
        PanelKey::View,
        row[0] + row[2] * 0.5,
        row[1] + 5.0,
        vp,
        &mut world
    ));
    assert!(h.lighting_open, "the Lighting row opens the panel");
    assert_eq!(
        widget::field_text(&world, lighting_panel::input(SUN_INTENSITY)),
        "2.2",
        "the sun intensity control seeds from the entry"
    );
    h.tick(&mut world);
    let bg = world
        .query::<Sprite>()
        .find(|s| s.asset_id == lighting_panel::PANEL_BG)
        .unwrap();
    assert!(bg.visible, "the panel renders once open");
}

// Apply captures the text controls and commits through validation; the entry's
// args update and the live preview rebuild is requested.
#[test]
fn lighting_apply_commits_sun_intensity() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let mut h = hook(vec![sun_entry()]);
    h.lighting_open = true;
    h.seed_lighting(&mut world);
    widget::seed_field(&mut world, lighting_panel::input(SUN_INTENSITY), "5.5");
    h.apply_lighting(&mut world);
    assert_eq!(h.lighting_status, None);
    assert!(
        h.dirty && h.rebuild_preview,
        "commit marks the world changed"
    );
    let args = h.entries[0]["args"].as_object().unwrap();
    assert_eq!(args["intensity"].as_f64().unwrap() as f32, 5.5);
    assert_eq!(
        args["direction"][1].as_f64().unwrap() as f32,
        0.85,
        "untouched fields keep their authored values"
    );
}

// Unparseable text falls back to the authored value (the same coerce rule the
// edit form uses), so Apply never corrupts an entry.
#[test]
fn lighting_apply_with_unparseable_text_keeps_the_authored_value() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let mut h = hook(vec![sun_entry()]);
    h.lighting_open = true;
    h.seed_lighting(&mut world);
    widget::seed_field(&mut world, lighting_panel::input(SUN_INTENSITY), "garbage");
    h.apply_lighting(&mut world);
    assert_eq!(h.lighting_status, None);
    let args = h.entries[0]["args"].as_object().unwrap();
    assert_eq!(args["intensity"].as_f64().unwrap() as f32, 2.2);
}

// A checkbox toggle commits immediately (live preview on the click) without
// capturing the text controls, so an in-progress typed edit stays pending.
#[test]
fn lighting_bool_toggle_commits_immediately_and_keeps_typed_text() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let mut h = hook(vec![fog_entry(false)]);
    h.lighting_open = true;
    h.seed_lighting(&mut world);
    // An in-progress density edit, not yet applied.
    widget::seed_field(
        &mut world,
        lighting_panel::input(fog_density_binding()),
        "0.5",
    );
    h.toggle_lighting_bool(fog_enabled_binding());
    let args = h.entries[0]["args"].as_object().unwrap();
    assert_eq!(args["enabled"], serde_json::Value::Bool(true));
    assert_eq!(
        args["density"].as_f64().unwrap() as f32,
        0.02,
        "the pending text edit is not committed by the toggle"
    );
    assert_eq!(
        widget::field_text(&world, lighting_panel::input(fog_density_binding())),
        "0.5",
        "the typed text stays in its control"
    );
    assert!(h.dirty, "the toggle itself is committed");
}

// The add row appends the missing singleton with default args; its fields then
// derive and seed.
#[test]
fn lighting_add_row_appends_the_missing_singleton() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let mut h = hook(vec![sun_entry()]);
    h.lighting_open = true;
    assert_eq!(h.lighting_present(), vec![true, false, false, false]);
    h.add_lighting_section(1, &mut world);
    assert_eq!(h.entries.len(), 2);
    assert_eq!(h.entries[1]["type"], "VolumetricFog");
    assert!(h.dirty);
    assert!(h.lighting_present()[1]);
    // A second add is a no-op (the section binds the existing entry).
    h.add_lighting_section(1, &mut world);
    assert_eq!(h.entries.len(), 2);
}

// Lighting fields assert keyboard focus only while the panel is frontmost, so
// its inputs never fight another panel's focused field.
#[test]
fn lighting_focus_yields_when_not_frontmost() {
    let mut h = hook(vec![sun_entry()]);
    h.lighting_open = true;
    h.lighting_focus = Some(SUN_INTENSITY);
    h.focus_panel(PanelKey::Lighting);
    let d = h.lighting_data();
    assert_eq!(
        h.make_lighting_view(&d, [0.0, 0.0]).focus,
        Some(SUN_INTENSITY)
    );
    h.focus_panel(PanelKey::Assets);
    assert_eq!(h.make_lighting_view(&d, [0.0, 0.0]).focus, None);
}

//! The Planet panel's actions (`hook/edit/planet.rs`): the texts an open seeds,
//! the commit Apply makes from them, the row buttons and reference pickers
//! that commit on the click, and the brush settings and pick the tool strip
//! moves.

use concinnity_engine::components::{FrameInput, UiEvent, UiValue};
use concinnity_engine::ecs::World;

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::{hook, set_input};
use crate::editor::inject;
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::planet_tools::{Setting, Tool};
use crate::frame_hook::FrameHook;

fn opened() -> (World, EditorHook) {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    set_input(
        &mut world,
        FrameInput {
            viewport: [1280.0, 720.0],
            ..Default::default()
        },
    );
    let mut h = hook(vec![
        serde_json::json!({"type": "WaterSurface", "args": {"$id": "sea"}}),
        serde_json::json!({"type": "WaterSurface", "args": {"$id": "lake"}}),
        serde_json::json!({"type": "Planet", "args": {
            "seed": 3,
            "sea": {"level": 0.0, "water": "sea"},
            "layers": [{"grass": "meadow"}, {"grass": "dune"}]
        }}),
    ]);
    registry::panel(PanelKey::Planet).toggle(&mut h, &mut world);
    h.tick(&mut world);
    (world, h)
}

fn send(world: &mut World, h: &mut EditorHook, name: &str, value: Option<UiValue>) {
    world.events_mut::<UiEvent>().send(UiEvent {
        name: name.into(),
        element: None,
        value,
    });
    h.read_ui_events(world);
}

fn row_of(h: &EditorHook, key: &str) -> usize {
    let (_, rows) = h.planet.view.as_ref().unwrap();
    rows.iter()
        .position(|r| r.field().is_some_and(|f| f.key == key))
        .unwrap()
}

fn planet_args(h: &EditorHook) -> serde_json::Value {
    h.entries[2]["args"].clone()
}

// An open seeds each number box from the planet; Apply writes what was typed
// as one edit.
#[test]
fn apply_commits_the_typed_boxes() {
    let (mut world, mut h) = opened();
    let seed = row_of(&h, "seed");
    assert_eq!(h.planet.texts[seed], "3");
    h.planet.texts[seed] = "9".into();
    send(&mut world, &mut h, "editor.planet.apply", None);
    assert_eq!(planet_args(&h)["seed"], 9);
    assert!(h.can_undo() && !h.planet.touched);
}

// A row's button and a reference picker commit on the click, leaving typed
// boxes pending.
#[test]
fn row_buttons_and_pickers_commit_on_the_click() {
    let (mut world, mut h) = opened();
    let seed = row_of(&h, "seed");
    // The seed row's reroll is its rightmost button.
    let reroll = i32::try_from(seed * 2).unwrap();
    send(
        &mut world,
        &mut h,
        "editor.planet.button",
        Some(UiValue::Int(reroll)),
    );
    assert_ne!(planet_args(&h)["seed"], 3, "a new seed is drawn");

    h.tick(&mut world);
    let water = row_of(&h, "sea.water");
    let row = i32::try_from(water).unwrap();
    send(
        &mut world,
        &mut h,
        "editor.planet.ref",
        Some(UiValue::Int(row)),
    );
    assert_eq!(planet_args(&h)["sea"]["water"], "lake");
}

// The strip's slider moves the brush, and the pick cycles through the
// planet's layers.
#[test]
fn the_strip_moves_the_brush_and_the_pick() {
    let (mut world, mut h) = opened();
    send(
        &mut world,
        &mut h,
        "editor.planet.tool",
        Some(UiValue::Int(4)),
    );
    assert_eq!(h.planet_tool.tools.active, Some(Tool::PaintGrass));
    send(
        &mut world,
        &mut h,
        "editor.planet.brush.strength",
        Some(UiValue::Float(1.0)),
    );
    assert_eq!(h.planet_tool.tools.brush.position(Setting::Strength), 1.0);
    h.tick(&mut world);
    let pick = |h: &EditorHook| h.make_planet_view().unwrap().strip.pick.to_string();
    assert!(pick(&h).starts_with("meadow"), "{}", pick(&h));
    send(&mut world, &mut h, "editor.planet.pick", None);
    h.tick(&mut world);
    assert!(pick(&h).starts_with("dune"), "{}", pick(&h));
}

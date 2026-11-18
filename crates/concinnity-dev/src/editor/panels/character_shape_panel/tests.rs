use super::super::character_shape::{self, Rows};
use super::*;
use concinnity_cook::compile::character::builtin_schema;
use concinnity_engine::components::{Button, UiState};
use concinnity_engine::ecs::asset_id::AssetId;

fn names(list: &[&str]) -> Vec<String> {
    list.iter().map(ToString::to_string).collect()
}

fn layout(presets: &[&str], has_shape: bool) -> ShapeLayout {
    let derived = character_shape::derive_rows(
        builtin_schema::humanoid(),
        &names(&["jaw+", "jaw-", "muscle"]),
        &names(&["thigh_l", "thigh_r"]),
    );
    let Rows { sections, sliders } = derived.clone();
    ShapeLayout {
        rows: character_shape::rows(&derived, presets.len(), has_shape),
        sections,
        sliders,
        presets: names(presets),
    }
}

fn built(world: &mut World, layout: &ShapeLayout) -> (BuiltPanel, ShapeIds) {
    let mut ui = UiBuilder::new(world, AssetId(9), None);
    let mut panel = build(&mut ui).unwrap();
    let ids = build_rows(world, &mut panel, layout);
    (panel, ids)
}

fn event(name: &str, value: Option<UiValue>) -> UiEvent {
    UiEvent {
        name: name.into(),
        element: None,
        value,
    }
}

fn state(world: &World, name: &str) -> Option<UiValue> {
    world.resource::<UiState>().unwrap().get(name).cloned()
}

#[test]
fn events_name_their_slider_or_preset() {
    assert_eq!(
        action(&event("editor.shape.change.2", Some(UiValue::Float(0.5)))),
        Some(ShapeAction::Change {
            slider: 2,
            value: 0.5
        })
    );
    assert_eq!(
        action(&event("editor.shape.commit.0", Some(UiValue::Float(-1.0)))),
        Some(ShapeAction::Commit {
            slider: 0,
            value: -1.0
        })
    );
    assert_eq!(
        action(&event(PRESET, Some(UiValue::Int(1)))),
        Some(ShapeAction::Preset(1))
    );
    assert_eq!(action(&event(RESET, None)), Some(ShapeAction::Reset));
    assert_eq!(
        action(&event(RANDOMIZE, None)),
        Some(ShapeAction::Randomize)
    );
    assert_eq!(action(&event(ADD, None)), Some(ShapeAction::Add));
    assert_eq!(action(&event("editor.shape.change.1", None)), None);
}

// One slider per slider row, each bound to its own value; the preset rows
// and headings build alongside them.
#[test]
fn rows_build_a_slider_per_slider_row() {
    let mut world = World::new();
    let layout = layout(&["slim", "heavy"], true);
    let (_, ids) = built(&mut world, &layout);
    let sliders: Vec<_> = world.query::<ValueSlider>().collect();
    assert_eq!(sliders.len(), layout.sliders.len());
    assert_eq!(ids.layout, layout);
    let captions: Vec<String> = world
        .query::<Button>()
        .map(|b| b.label.clone())
        .filter(|l| l.starts_with("apply"))
        .collect();
    assert_eq!(captions, ["apply slim", "apply heavy"]);
    let jaw = sliders
        .iter()
        .find(|s| s.on_commit == "editor.shape.commit.0")
        .unwrap();
    assert_eq!((jaw.min, jaw.max), (-1.0, 1.0), "jaw is bipolar");
}

// The title names the shape, the values reach each slider's state, and the
// buttons show only while there is a shape to act on.
#[test]
fn sync_names_the_shape_and_writes_the_values() {
    let mut world = World::new();
    let layout = layout(&[], true);
    let (panel, ids) = built(&mut world, &layout);
    let values = vec![0.25, 0.5, -0.5];
    let view = ShapeView {
        shape: Some("body_shape"),
        status: None,
        values: &values,
    };
    sync(&view, &ids, &panel, &mut world);
    assert_eq!(
        state(&world, TITLE),
        Some(UiValue::Text("Character Shape: body_shape".into()))
    );
    assert_eq!(
        state(&world, "editor.shape.value.0"),
        Some(UiValue::Float(0.25))
    );
    assert_eq!(
        state(&world, "editor.shape.value.2"),
        Some(UiValue::Float(-0.5))
    );
    let buttons = |world: &World| {
        world
            .get_by_id::<Layout>(panel.slots[BUTTONS_SLOT])
            .unwrap()
            .visible
    };
    assert!(buttons(&world));
    let none = ShapeView {
        shape: None,
        status: Some("Select a SkinnedMesh"),
        values: &[],
    };
    sync(&none, &ids, &panel, &mut world);
    assert!(!buttons(&world), "no shape, no buttons");
    assert_eq!(
        state(&world, STATUS),
        Some(UiValue::Text("Select a SkinnedMesh".into()))
    );
}

// Without a shape the column holds the add row alone.
#[test]
fn a_mesh_without_a_shape_offers_the_add_row() {
    let mut world = World::new();
    let (_, _) = built(&mut world, &layout(&[], false));
    assert_eq!(world.query::<ValueSlider>().count(), 0);
    assert!(
        world
            .query::<Button>()
            .any(|b| b.label == "+ Add CharacterShape")
    );
}

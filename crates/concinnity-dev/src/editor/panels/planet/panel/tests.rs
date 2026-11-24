use super::*;
use concinnity_engine::components::{Layout, UiState};
use serde_json::json;

use crate::editor::panels::planet;
use crate::editor::planet_tools::{Tool, Tools};

fn planet_rows(covers: usize) -> Vec<Row> {
    let covers: Vec<_> = (0..covers).map(|_| json!({"material": "sand"})).collect();
    let args = json!({
        "seed": 3,
        "sea": {"level": 0.0, "water": "sea"},
        "covers": covers
    });
    planet::rows(Some(&planet::working(args.as_object().unwrap())))
}

fn built(world: &mut World, rows: &[Row]) -> (BuiltPanel, PlanetIds) {
    let mut ui = UiBuilder::new(world, AssetId(9), None);
    let mut panel = build(&mut ui).unwrap();
    let ids = build_rows(world, &mut panel, rows);
    (panel, ids)
}

fn event(name: &str, value: Option<UiValue>) -> UiEvent {
    UiEvent {
        name: name.into(),
        element: None,
        value,
    }
}

fn position(rows: &[Row], key: &str) -> usize {
    rows.iter()
        .position(|r| r.field().is_some_and(|f| f.key == key))
        .unwrap()
}

fn text_of(world: &World, name: &str) -> Option<String> {
    match world.resource::<UiState>()?.get(name) {
        Some(UiValue::Text(t)) => Some(t.clone()),
        _ => None,
    }
}

fn shown(world: &World, id: AssetId) -> bool {
    world.get_by_id::<Layout>(id).unwrap().visible
}

#[test]
fn events_name_their_row_or_tool() {
    assert_eq!(
        action(&event(ROW_BUTTON, Some(UiValue::Int(7)))),
        Some(PlanetAction::RowButton { row: 3, slot: 1 })
    );
    assert_eq!(
        action(&event(CYCLE_REF, Some(UiValue::Int(4)))),
        Some(PlanetAction::CycleRef(4))
    );
    assert_eq!(action(&event(APPLY, None)), Some(PlanetAction::Apply));
    assert_eq!(action(&event(ADD_SEA, None)), Some(PlanetAction::AddSea));
    assert_eq!(
        action(&event("editor.planet.tool", Some(UiValue::Int(1)))),
        Some(PlanetAction::Strip(StripAction::Toggle(Tool::Lower)))
    );
    assert_eq!(action(&event(CYCLE_REF, None)), None);
}

// Each text row builds a text field and each reference row a picker; the
// rows build again only when what they lay out changes.
#[test]
fn rows_build_their_controls_and_rebuild_on_a_new_layout() {
    let mut world = World::new();
    let rows = planet_rows(1);
    let (_, ids) = built(&mut world, &rows);
    let seed = position(&rows, "seed");
    assert!(ids.texts[seed].is_some(), "a number row types");
    let water = position(&rows, "sea.water");
    assert!(ids.refs[water].is_some(), "a reference row picks");
    assert!(ids.texts[water].is_none());
    assert!(ids.lays_out(&planet_rows(1)));
    assert!(!ids.lays_out(&planet_rows(2)), "a new cover adds rows");
    let mut cycled = planet_rows(1);
    if let RowKind::Field { field, .. } = &mut cycled[water].kind {
        field.variant_idx += 1;
    }
    assert!(ids.lays_out(&cycled), "a picked value is not a new layout");
}

// The texts, the focus, the pickers' names and the title follow the hook's
// model.
#[test]
fn sync_writes_the_texts_pickers_and_title() {
    let mut world = World::new();
    let rows = planet_rows(1);
    let (panel, ids) = built(&mut world, &rows);
    let seed = position(&rows, "seed");
    let mut texts = vec![String::new(); rows.len()];
    texts[seed] = "42".into();
    let tools = Tools::default();
    let view = PlanetView {
        rows: &rows,
        texts: &texts,
        focus: Some(seed),
        status: Some("bad"),
        dirty: true,
        strip: StripView {
            tools: &tools,
            pick: "",
        },
    };
    sync(&view, &ids, &panel, &mut world);
    let field = world
        .get_by_id::<TextField>(ids.texts[seed].unwrap())
        .unwrap();
    assert_eq!(field.content, "42");
    assert!(field.focused);
    assert_eq!(text_of(&world, TITLE).as_deref(), Some("Planet *"));
    assert_eq!(text_of(&world, STATUS).as_deref(), Some("bad"));
    let water = position(&rows, "sea.water");
    assert_eq!(
        text_of(&world, ids.refs[water].as_ref().unwrap()).as_deref(),
        Some("sea")
    );

    // Typing into the field reads back, with its focus.
    if let Some(f) = world.get_mut_by_id::<TextField>(ids.texts[seed].unwrap()) {
        f.content = "7".into();
    }
    assert_eq!(observe(&world, &ids, &mut texts), Some(seed));
    assert_eq!(texts[seed], "7");
}

// The brush rows show only under a picked tool, a paint tool adding its
// pick.
#[test]
fn the_brush_rows_follow_the_picked_tool() {
    let mut world = World::new();
    let rows = planet_rows(0);
    let (panel, ids) = built(&mut world, &rows);
    let mut tools = Tools::default();
    let sync_with = |world: &mut World, tools: &Tools| {
        let view = PlanetView {
            rows: &rows,
            texts: &[],
            focus: None,
            status: None,
            dirty: false,
            strip: StripView {
                tools,
                pick: "meadow   1/2",
            },
        };
        sync(&view, &ids, &panel, world);
    };
    let (brush, pick) = (panel.slots[BRUSH_SLOT], panel.slots[PICK_SLOT]);
    sync_with(&mut world, &tools);
    assert!(!shown(&world, brush), "no tool, no brush");
    tools.toggle(Tool::Raise);
    sync_with(&mut world, &tools);
    assert!(shown(&world, brush) && !shown(&world, pick));
    tools.toggle(Tool::PaintGrass);
    sync_with(&mut world, &tools);
    assert!(shown(&world, pick), "a paint tool shows its pick");
}

#[test]
fn a_world_without_a_planet_offers_to_add_one() {
    let mut world = World::new();
    let rows = planet::rows(None);
    let (_, ids) = built(&mut world, &rows);
    assert_eq!(ids.shapes.len(), 1);
    assert_eq!(ids.shapes[0].control, Control::Add(ADD_PLANET));
}

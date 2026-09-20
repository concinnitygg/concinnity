//! The Map panel's actions (`hook/edit/map.rs`): the working entry list turned
//! into the world's map, the keys the cards are addressed by, and the pan that
//! carries the canvas.

use concinnity_core::components::TextLabel;
use concinnity_core::ecs::World;
use serde_json::json;

use crate::editor::asset_handle::AssetHandle;
use crate::editor::behavior::chart::ChartIds;
use crate::editor::behavior::graph::{Card, Chart};
use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::{behavior, drag_input, entry, entry_with_args, hook};
use crate::editor::panels::registry::{self, PanelKey};

fn map_session(entries: Vec<serde_json::Value>) -> (EditorHook, World) {
    let mut world = injected_world();
    let mut h = hook(entries);
    registry::panel(PanelKey::Map).toggle(&mut h, &mut world);
    (h, world)
}

fn injected_world() -> World {
    let p = registry::panel(PanelKey::Map);
    crate::test_support::injected_world(&p.sprite_ids(), &p.label_ids(), &[])
}

fn card<'a>(chart: &'a Chart, title: &str) -> &'a Card {
    chart
        .cards
        .iter()
        .find(|c| c.title == title)
        .unwrap_or_else(|| panic!("no `{title}` card in {:?}", titles(chart)))
}

fn titles(chart: &Chart) -> Vec<&str> {
    chart.cards.iter().map(|c| c.title.as_str()).collect()
}

// A world whose menu opens a scene: the picture the panel exists to draw.
fn menu_world() -> Vec<serde_json::Value> {
    vec![
        entry("bistro", "Scene"),
        entry_with_args(
            "main",
            "MainMenu",
            json!({"initial": true, "items": [{"label": "Start", "action": "scene:bistro"}]}),
        ),
    ]
}

// The entry list is what the map reads, so an authored menu and the scene it
// opens reach the panel as two cards and the move between them.
#[test]
fn the_working_entries_become_the_worlds_map() {
    let (h, _world) = map_session(menu_world());
    let chart = h.map_chart();
    assert_eq!(titles(&chart), ["main", "bistro"]);
    assert_eq!(card(&chart, "main").column, 0, "the world starts there");
    assert_eq!(card(&chart, "bistro").column, 1);
    let start = &chart.wires[0];
    assert_eq!(start.label.as_deref(), Some("Start"));
}

// A card is addressed by the key its entry holds this session, not by where the
// entry sits: that is what makes a click on one a selection later.
#[test]
fn a_card_carries_the_key_of_the_entry_declaring_its_place() {
    let (h, _world) = map_session(menu_world());
    let key = h.entries.key_at(0).unwrap();
    assert_eq!(
        card(&h.map_chart(), "bistro").handle,
        Some(AssetHandle::Entry(key))
    );
}

// An entry declaring no `$id` is still a place, under the label the build gives
// it -- which is the name the rest of the editor calls it by too.
#[test]
fn an_anonymous_entry_maps_under_the_label_the_build_gives_it() {
    let (h, _world) = map_session(vec![json!({"type": "Scene", "args": {}})]);
    let chart = h.map_chart();
    assert_eq!(titles(&chart), ["Scene#0"]);
    assert_eq!(
        card(&chart, "Scene#0").handle,
        Some(AssetHandle::Entry(h.entries.key_at(0).unwrap()))
    );
}

// An entry the registry cannot type is no place and no content of one: a build
// would not place it either.
#[test]
fn an_entry_of_no_registered_type_is_left_out() {
    let (h, _world) = map_session(vec![
        entry("bistro", "Scene"),
        json!({"type": "Nonsense", "args": {"$id": "x", "scene": "bistro"}}),
    ]);
    let chart = h.map_chart();
    assert_eq!(titles(&chart), ["bistro"]);
    assert_eq!(card(&chart, "bistro").detail, "scene");
}

fn scenes(count: usize) -> Vec<serde_json::Value> {
    (0..count)
        .map(|i| entry(&format!("scene_{i}"), "Scene"))
        .collect()
}

// The wheel moves the canvas along whichever axis has anywhere to go, and stops
// at the map's edge rather than running off into empty canvas.
#[test]
fn the_wheel_pans_the_canvas_and_stops_at_the_maps_edge() {
    let (mut h, mut world) = map_session(scenes(20));
    let p = registry::panel(PanelKey::Map);

    p.scroll(&mut h, &mut world, 1.0);
    assert!(h.map.pan[1] > 0.0, "a column of places pans down");
    assert_eq!(h.map.pan[0], 0.0, "there is only the one column");

    for _ in 0..64 {
        p.scroll(&mut h, &mut world, 1.0);
    }
    let bottom = h.map.pan;
    p.scroll(&mut h, &mut world, 1.0);
    assert_eq!(h.map.pan, bottom, "the map's foot is the end of the pan");

    for _ in 0..128 {
        p.scroll(&mut h, &mut world, -1.0);
    }
    assert_eq!(h.map.pan, [0.0, 0.0]);
}

// A press grabs the canvas and it follows the cursor until the button comes up.
#[test]
fn a_press_on_the_canvas_drags_the_map_under_the_cursor() {
    let (mut h, mut world) = map_session(scenes(20));
    let p = registry::panel(PanelKey::Map);
    let vp = [1280.0, 720.0];
    let o = h.origin(PanelKey::Map, vp);
    let grab = [o[0] + 60.0, o[1] + 120.0];

    assert!(p.press(&mut h, &mut world, grab[0], grab[1], o));
    h.drive_map_pan(&drag_input([grab[0], grab[1] - 40.0], true));
    assert_eq!(h.map.pan[1], 40.0, "the map followed the cursor");

    h.drive_map_pan(&drag_input([grab[0], grab[1] - 80.0], false));
    assert_eq!(h.map.pan[1], 40.0, "the release left it where it was");
    assert!(h.map.pan_drag.is_none());
}

// The whole point of the id refactor: two panels drawing charts at once do not
// blank each other's cards.
#[test]
fn the_map_and_the_behavior_chart_draw_at_once() {
    let mut world_entries = menu_world();
    world_entries.push(behavior(
        "greeter",
        json!({"on": "start", "do": [{"show": {"target": "self"}}]}),
    ));
    let mut h = hook(world_entries);
    let behavior = registry::panel(PanelKey::Behavior);
    let map = registry::panel(PanelKey::Map);
    let mut world = crate::test_support::injected_world(
        &[behavior.sprite_ids(), map.sprite_ids()].concat(),
        &[behavior.label_ids(), map.label_ids()].concat(),
        &[],
    );
    for id in behavior.field_ids() {
        world.push_identified(id.0, concinnity_core::components::TextInput::default());
    }
    behavior.toggle(&mut h, &mut world);
    map.toggle(&mut h, &mut world);
    h.behavior.mode = crate::editor::behavior::panel::ViewMode::Chart;

    let vp = [1600.0, 900.0];
    map.draw(&h, &mut world, h.origin(PanelKey::Map, vp), [-1.0, -1.0]);
    behavior.draw(
        &h,
        &mut world,
        h.origin(PanelKey::Behavior, vp),
        [-1.0, -1.0],
    );

    let drawn = |ids: ChartIds| -> Vec<String> {
        (0..4)
            .map(|i| world.get_by_id::<TextLabel>(ids.card_title(i)).unwrap())
            .filter(|l| l.visible && !l.content.is_empty())
            .map(|l| l.content.clone())
            .collect()
    };
    assert!(
        drawn(ChartIds::of(PanelKey::Map)).contains(&"main".to_string()),
        "the map lost its cards to the behavior chart",
    );
    assert!(
        !drawn(ChartIds::of(PanelKey::Behavior)).is_empty(),
        "the behavior chart drew nothing, so the map proves nothing",
    );
}

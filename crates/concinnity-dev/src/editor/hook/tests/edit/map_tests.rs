//! The Map panel's actions (`hook/edit/map.rs`): the working entry list turned
//! into the world's map, the keys the cards are addressed by, and the pan that
//! carries the canvas.

use concinnity_engine::components::{Canvas, UiEvent, UiValue};
use concinnity_engine::ecs::World;
use serde_json::json;

use crate::editor::asset_handle::AssetHandle;
use crate::editor::behavior::chart;
use crate::editor::behavior::graph::{Card, Chart};
use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::{
    active, behavior, built_world, entry, entry_target, entry_with_args, hook, select, selected,
};
use crate::editor::map;
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::ui::{BuiltPanel, BuiltUi};

fn map_session(entries: Vec<serde_json::Value>) -> (EditorHook, World) {
    let mut world = built_world(&[PanelKey::Map]);
    let mut h = hook(entries);
    registry::toggle(PanelKey::Map, &mut h, &mut world);
    (h, world)
}

fn built(world: &World) -> BuiltPanel {
    world
        .resource::<BuiltUi>()
        .and_then(|b| b.panel(PanelKey::Map))
        .cloned()
        .unwrap()
}

fn drive(h: &mut EditorHook) {
    let chart = h.map_chart();
    h.drive_map(&chart);
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

fn canvas(h: &EditorHook) -> [f32; 2] {
    map::panel::canvas(h.effective_size(PanelKey::Map))
}

// Press the card standing for `title`, as its canvas reports it, returning
// whether the panel took the press.
fn click_card(h: &mut EditorHook, world: &mut World, title: &str) -> bool {
    let chart = h.map_chart();
    let index = chart.cards.iter().position(|c| c.title == title).unwrap();
    let event = UiEvent {
        name: map::panel::CARD_EVENT.into(),
        element: None,
        value: Some(UiValue::Int(i32::try_from(index).unwrap())),
    };
    registry::panel(PanelKey::Map).on_event(h, world, &event)
}

// A world whose menu opens a scene: the picture the panel exists to draw.
fn menu_world() -> Vec<serde_json::Value> {
    vec![
        entry("bistro", "Scene"),
        entry_with_args(
            "main",
            "MainMenu",
            json!({"initial": true, "items": [{"label": "Start", "action": {"scene": "bistro"}}]}),
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

// Where the user pans the canvas is where the map is, written back to a fresh
// canvas. The canvas keeps the pan within the map itself.
#[test]
fn the_canvas_pan_is_the_maps() {
    let (mut h, mut world) = map_session(scenes(20));
    let p = registry::panel(PanelKey::Map);
    let mut panel = built(&world);
    p.sync(&mut h, &mut world, &mut panel);
    world
        .get_mut_by_id::<Canvas>(map::panel::canvas_of(&panel))
        .unwrap()
        .pan = [0.0, 9000.0];

    p.observe(&mut h, &mut world, &panel);
    assert_eq!(h.map.pan, [0.0, 9000.0], "the canvas's pan was taken in");

    let mut fresh = built_world(&[PanelKey::Map]);
    let mut fresh_panel = built(&fresh);
    p.sync(&mut h, &mut fresh, &mut fresh_panel);
    assert_eq!(
        map::panel::observed(&fresh, &fresh_panel).unwrap().0,
        h.map.pan
    );
}

// The chart is the canvas's shapes, rebuilt when the entries change.
#[test]
fn the_chart_is_written_as_the_canvas_shapes() {
    let (mut h, mut world) = map_session(menu_world());
    let p = registry::panel(PanelKey::Map);
    let mut panel = built(&world);
    p.sync(&mut h, &mut world, &mut panel);
    let shapes = world
        .resource::<concinnity_engine::components::UiState>()
        .unwrap()
        .shapes(map::panel::SHAPES)
        .to_vec();
    let tags: Vec<u32> = shapes.iter().filter_map(|s| s.tag).collect();
    assert_eq!(tags, [0, 1], "a pressable card per place");
    assert!(shapes.iter().any(|s| s.text == "Start"), "the move's words");

    h.entries.push(entry("cellar", "Scene"));
    p.sync(&mut h, &mut world, &mut panel);
    let cards = world
        .resource::<concinnity_engine::components::UiState>()
        .unwrap()
        .shapes(map::panel::SHAPES)
        .iter()
        .filter(|s| s.tag.is_some())
        .count();
    assert_eq!(cards, 3, "the new place drew");
}

// The feature the addressing was for: a card is the place it stands for, so
// clicking one selects that place and opens it the way its Assets row does.
#[test]
fn a_click_on_a_place_selects_it_and_opens_it_for_editing() {
    let (mut h, mut world) = map_session(menu_world());
    assert!(click_card(&mut h, &mut world, "bistro"));
    assert_eq!(selected(&h), ["bistro"]);
    assert_eq!(h.form.target, entry_target(&h, 0));
    assert_eq!(h.form.selected_type.as_deref(), Some("Scene"));
}

// Shift behaves as it does over a row or in the viewport: it adds a place to
// the selection, and takes it away again.
#[test]
fn a_shift_click_adds_a_place_to_the_selection_and_takes_it_away_again() {
    let (mut h, mut world) = map_session(menu_world());
    click_card(&mut h, &mut world, "bistro");
    h.shift_held = true;
    click_card(&mut h, &mut world, "main");
    assert_eq!(selected(&h), ["bistro", "main"]);
    assert_eq!(active(&h).as_deref(), Some("main"));

    click_card(&mut h, &mut world, "main");
    assert_eq!(selected(&h), ["bistro"]);
}

// A move onto a place no entry declares draws a card saying so, and there is
// nothing behind it to select -- but the press is still the panel's, so it
// cannot fall through to the world behind.
#[test]
fn a_click_on_a_place_the_world_does_not_declare_selects_nothing() {
    let (mut h, mut world) = map_session(vec![entry_with_args(
        "main",
        "MainMenu",
        json!({"initial": true, "items": [{"label": "Start", "action": {"scene": "nowhere"}}]}),
    )]);
    assert!(click_card(&mut h, &mut world, "nowhere"));
    assert!(h.selection.is_empty(), "{:?}", selected(&h));
}

// The screen a menu opens has no authored line, so it is selected under the
// identity the build gives it -- the same handle its Assets row carries.
#[test]
fn a_click_on_a_place_the_build_generates_selects_it_under_that_identity() {
    let (mut h, mut world) = map_session(vec![entry_with_args(
        "main",
        "MainMenu",
        json!({"initial": true, "items": [{"label": "Settings", "action": "settings"}]}),
    )]);
    let generated = h
        .map_chart()
        .cards
        .iter()
        .find(|c| matches!(c.handle, Some(AssetHandle::Generated(_))))
        .expect("the settings screen the menu opens")
        .title
        .clone();

    assert!(click_card(&mut h, &mut world, &generated));
    assert_eq!(
        h.selection.active(),
        Some(&AssetHandle::Generated(generated))
    );
}

// A behavior's move leaves from the world rather than from a place, so the map
// draws one arrow out of the world however many behaviors make it. Which of
// them lead here is asked of the place, and the panel that shows a behavior
// opens on the first.
#[test]
fn a_ctrl_click_on_a_place_reaches_the_behaviors_that_send_the_world_to_it() {
    let (mut h, mut world) = map_session(vec![
        entry("bistro", "Scene"),
        behavior(
            "opener",
            json!({"on": "start", "do": [{"scene": {"scene": "bistro"}}]}),
        ),
        behavior("idler", json!({"on": "tick", "do": []})),
    ]);
    h.ctrl_held = true;
    assert!(click_card(&mut h, &mut world, "bistro"));

    assert_eq!(
        selected(&h),
        ["opener"],
        "the one that sends the world here"
    );
    assert!(h.open[PanelKey::Behavior], "the panel showing one opened");
    assert_eq!(h.behavior_data().name, "opener");
}

// A place nothing sends the world to has nothing to reach, and says so rather
// than quietly emptying the selection.
#[test]
fn a_ctrl_click_on_a_place_no_behavior_reaches_leaves_the_selection_alone() {
    let (mut h, mut world) = map_session(menu_world());
    click_card(&mut h, &mut world, "main");
    h.ctrl_held = true;
    assert!(click_card(&mut h, &mut world, "bistro"));

    assert_eq!(selected(&h), ["main"]);
    assert!(!h.open[PanelKey::Behavior], "nothing to open");
}

// The other direction: what the Assets panel or the viewport selected is what
// the map lights up, so the two surfaces agree on where the session is.
#[test]
fn a_selection_made_elsewhere_lights_up_its_card() {
    let (mut h, _world) = map_session(menu_world());
    select(&mut h, &["bistro"]);
    let chart = h.map_chart();
    let at = chart.cards.iter().position(|c| c.title == "bistro");
    assert_eq!(h.map_selected(&chart), at);
}

// A selection standing for no place at all leaves the map with nothing to
// light: the map draws where a world can be, not everything in it.
#[test]
fn a_selection_naming_no_place_lights_up_nothing() {
    let mut entries = menu_world();
    entries.push(entry_with_args(
        "floor",
        "Prop",
        json!({"mesh": "floor_mesh"}),
    ));
    let (mut h, _world) = map_session(entries);
    select(&mut h, &["floor"]);
    let chart = h.map_chart();
    assert_eq!(h.map_selected(&chart), None);
}

// A place selected off the canvas is panned to, so selecting one in the Assets
// panel says where it sits in the world rather than nowhere.
#[test]
fn a_selection_off_the_canvas_pans_the_map_to_it() {
    let (mut h, _world) = map_session(scenes(20));
    drive(&mut h);
    assert_eq!(h.map.pan, [0.0, 0.0]);

    select(&mut h, &["scene_19"]);
    drive(&mut h);
    assert!(
        h.map.pan[1] > 0.0,
        "the canvas did not follow the selection"
    );
    let chart = h.map_chart();
    assert_eq!(
        chart::pan_to(card(&chart, "scene_19"), canvas(&h), h.map.pan, &chart),
        h.map.pan,
        "it stopped short of the place it was following",
    );
}

// And it moves no further than it must: a place already on the canvas leaves
// the pan alone, as does one the map does not draw.
#[test]
fn a_selection_already_in_view_leaves_the_canvas_where_it_is() {
    let mut entries = scenes(20);
    entries.push(entry_with_args(
        "floor",
        "Prop",
        json!({"mesh": "floor_mesh"}),
    ));
    let (mut h, _world) = map_session(entries);
    select(&mut h, &["scene_19"]);
    drive(&mut h);
    let at = h.map.pan;

    select(&mut h, &["scene_18"]);
    drive(&mut h);
    assert_eq!(
        h.map.pan, at,
        "a neighbor already in view re-centered the map"
    );

    select(&mut h, &["floor"]);
    drive(&mut h);
    assert_eq!(h.map.pan, at, "a prop is inside a place, not one of them");
}

// Opening the panel is a fresh look at the world, so the canvas says where the
// world starts rather than resuming wherever it was left.
#[test]
fn reopening_the_panel_puts_the_canvas_back_on_where_the_world_starts() {
    let (mut h, mut world) = map_session(scenes(20));
    drive(&mut h);
    h.map.pan = [0.0, 200.0];

    registry::close(PanelKey::Map, &mut h, &mut world);
    registry::toggle(PanelKey::Map, &mut h, &mut world);
    drive(&mut h);
    assert_eq!(h.map.pan, [0.0, 0.0]);
}

// The panel opens closed and stays out of the way until it is asked for: a
// closed Map is one more thing that must not touch the canvas or the selection
// every other surface shares.
#[test]
fn a_closed_panel_drives_nothing() {
    let mut h = hook(scenes(20));
    assert!(
        !h.open[PanelKey::Map],
        "the Map panel is closed until it is opened"
    );
    select(&mut h, &["scene_19"]);
    drive(&mut h);
    assert_eq!(h.map.pan, [0.0, 0.0]);
    assert!(!h.map.rooted);
    assert_eq!(selected(&h), ["scene_19"], "the selection is not the map's");
}

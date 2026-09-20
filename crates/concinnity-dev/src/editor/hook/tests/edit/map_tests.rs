//! The Map panel's actions (`hook/edit/map.rs`): the working entry list turned
//! into the world's map, the keys the cards are addressed by, and the pan that
//! carries the canvas.

use concinnity_core::components::TextLabel;
use concinnity_core::ecs::World;
use serde_json::json;

use crate::editor::asset_handle::AssetHandle;
use crate::editor::behavior::chart::{self, ChartIds};
use crate::editor::behavior::graph::{Card, Chart};
use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::{
    active, behavior, drag_input, entry, entry_target, entry_with_args, hook, select, selected,
};
use crate::editor::map;
use crate::editor::panels::registry::{self, PanelKey};

const VP: [f32; 2] = [1280.0, 720.0];

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

fn canvas(h: &EditorHook) -> [f32; 2] {
    map::panel::canvas(h.effective_size(PanelKey::Map))
}

// The point a click on the card standing for `title` lands on.
fn card_center(h: &EditorHook, title: &str) -> [f32; 2] {
    let chart = h.map_chart();
    let o = h.origin(PanelKey::Map, VP);
    let band = map::panel::band(o, h.effective_size(PanelKey::Map));
    let r = chart::card_rect(card(&chart, title), band, h.map.pan);
    [r[0] + r[2] * 0.5, r[1] + r[3] * 0.5]
}

// Press the card standing for `title`, as the routing does, returning whether
// the panel took the press.
fn click_card(h: &mut EditorHook, world: &mut World, title: &str) -> bool {
    let at = card_center(h, title);
    let o = h.origin(PanelKey::Map, VP);
    registry::panel(PanelKey::Map).press(h, world, at[0], at[1], o)
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
    // Clear of the one column of cards, so the press grabs the canvas rather
    // than selecting a place.
    let grab = [o[0] + 400.0, o[1] + 120.0];

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
        json!({"initial": true, "items": [{"label": "Start", "action": "scene:nowhere"}]}),
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

// The other direction: what the Assets panel or the viewport selected is what
// the map lights up, so the two surfaces agree on where the session is.
#[test]
fn a_selection_made_elsewhere_lights_up_its_card() {
    let (mut h, _world) = map_session(menu_world());
    select(&mut h, &["bistro"]);
    let chart = h.map_chart();
    let at = chart.cards.iter().position(|c| c.title == "bistro");
    assert_eq!(h.make_map_view(&chart, [-1.0, -1.0]).selected, at);
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
    assert_eq!(h.make_map_view(&chart, [-1.0, -1.0]).selected, None);
}

// A place selected off the canvas is panned to, so selecting one in the Assets
// panel says where it sits in the world rather than nowhere.
#[test]
fn a_selection_off_the_canvas_pans_the_map_to_it() {
    let (mut h, _world) = map_session(scenes(20));
    h.drive_map();
    assert_eq!(h.map.pan, [0.0, 0.0]);

    select(&mut h, &["scene_19"]);
    h.drive_map();
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
    h.drive_map();
    let at = h.map.pan;

    select(&mut h, &["scene_18"]);
    h.drive_map();
    assert_eq!(
        h.map.pan, at,
        "a neighbor already in view re-centered the map"
    );

    select(&mut h, &["floor"]);
    h.drive_map();
    assert_eq!(h.map.pan, at, "a prop is inside a place, not one of them");
}

// Opening the panel is a fresh look at the world, so the canvas says where the
// world starts rather than resuming wherever it was left.
#[test]
fn reopening_the_panel_puts_the_canvas_back_on_where_the_world_starts() {
    let (mut h, mut world) = map_session(scenes(20));
    let p = registry::panel(PanelKey::Map);
    for _ in 0..8 {
        p.scroll(&mut h, &mut world, 1.0);
    }
    assert!(h.map.pan[1] > 0.0, "nothing moved, so nothing is proven");

    p.close(&mut h, &mut world);
    p.toggle(&mut h, &mut world);
    h.drive_map();
    assert_eq!(h.map.pan, [0.0, 0.0]);
}

// An edit that shrinks the map must not strand the canvas past the last place
// it drew.
#[test]
fn a_map_that_shrank_under_the_canvas_brings_it_back_to_the_last_place() {
    let (mut h, mut world) = map_session(scenes(20));
    let p = registry::panel(PanelKey::Map);
    for _ in 0..16 {
        p.scroll(&mut h, &mut world, 1.0);
    }
    let deep = h.map.pan;

    for _ in 3..20 {
        h.entries.remove(3);
    }
    h.drive_map();
    assert!(
        h.map.pan[1] < deep[1],
        "the canvas stayed past the map's foot"
    );
    let chart = h.map_chart();
    assert_eq!(chart::clamp_pan(h.map.pan, &chart, canvas(&h)), h.map.pan);
}

// The panel opens closed and stays out of the way until it is asked for: a
// closed Map is one more thing that must not touch the canvas or the selection
// every other surface shares.
#[test]
fn a_closed_panel_drives_nothing() {
    let mut h = hook(scenes(20));
    assert!(!h.map.open, "the Map panel is closed until it is opened");
    select(&mut h, &["scene_19"]);
    h.drive_map();
    assert_eq!(h.map.pan, [0.0, 0.0]);
    assert!(!h.map.rooted);
    assert_eq!(selected(&h), ["scene_19"], "the selection is not the map's");
}

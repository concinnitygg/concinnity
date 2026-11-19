//! The right-click "Create here" menu (`hook/drive/create_menu.rs`): the
//! positioned entry a type row creates at the surface point, the instance a
//! prefab row creates, and the dismissal rules -- the menu closing itself
//! closes it here, a claimed region does not reach it.

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_engine::components::{FrameInput, UiEvent, UiValue};
use concinnity_engine::ecs::World;
use concinnity_host::thread::asset_id;

use crate::frame_hook::FrameHook;

use crate::editor::hook::tests::fixtures::{active, click_at, hook, pick_world, set_input};
use crate::editor::hook::{EditorHook, declared_id, entry_type};

use crate::editor::panels::registry::PanelKey;

use crate::editor::sim;

// A right-press tick at `pos` (the create-menu opener).
fn right_click_at(world: &mut World, h: &mut EditorHook, pos: [f32; 2]) {
    set_input(
        world,
        FrameInput {
            viewport: [1280.0, 720.0],
            mouse_x: pos[0],
            mouse_y: pos[1],
            right_click: true,
            ..Default::default()
        },
    );
    h.tick(world);
}

// Pick the create-menu row whose item matches `pick`, as the menu reports it
// (its heading is row 0).
fn click_menu_row(
    world: &mut World,
    h: &mut EditorHook,
    pick: &crate::editor::create_menu::MenuItem,
) {
    let items = h.create_menu_items();
    let row = items
        .iter()
        .position(|i| i == pick)
        .unwrap_or_else(|| panic!("{pick:?} not in {items:?}"));
    world.events_mut::<UiEvent>().send(UiEvent {
        name: "editor.create.pick".into(),
        element: None,
        value: Some(UiValue::Int(i32::try_from(row + 1).unwrap())),
    });
    h.read_ui_events(world);
    // A frame with no press lists what the pick changed.
    set_input(
        world,
        FrameInput {
            viewport: [1280.0, 720.0],
            ..Default::default()
        },
    );
    h.tick(world);
}

// A viewport right press opens the create menu with the world position under
// the cursor captured; picking a type row creates one entry there through the
// shared path (one undo step) and selects it.
#[test]
fn right_click_creates_a_positioned_type_at_the_surface_point() {
    asset_id::begin_build();
    let ground = asset_id::intern("ground");
    let mut world = pick_world(
        [0.0; 3],
        vec![(ground, [-20.0, -1.0, -20.0], [20.0, 0.0, 20.0])],
    );
    let mut h = hook(vec![serde_json::json!({
        "type": "Prop",
        "args": { "$id": "ground", "mesh": "demo", "position": [0.0, -0.5, 0.0] }
    })]);

    right_click_at(&mut world, &mut h, [640.0, 500.0]);
    let menu = h.create_menu.as_ref().expect("the right press opens it");
    assert!(
        menu.pos[1].abs() < 1e-3,
        "captured the ground-top landing point: {:?}",
        menu.pos
    );
    let before = h.entries.len();

    click_menu_row(
        &mut world,
        &mut h,
        &crate::editor::create_menu::MenuItem::Type("PointLight"),
    );
    assert!(h.create_menu.is_none(), "picking a row closes the menu");
    assert_eq!(h.entries.len(), before + 1);
    let created = &h.entries[before];
    assert_eq!(entry_type(created), Some(RegisteredType::PointLight));
    assert!(
        created["args"]["position"][1].as_f64().unwrap().abs() < 1e-3,
        "{created}"
    );
    assert_eq!(declared_id(created), None, "a placed asset is anonymous");
    assert_eq!(active(&h).as_deref(), Some("PointLight#0"));
    assert!(h.dirty);
    h.undo();
    assert_eq!(h.entries.len(), before, "one undo removes the creation");
}

// The Prefab section unfolds from its header row, and a prefab row creates a
// Prop instancing it at the captured point.
#[test]
fn prefab_rows_create_a_prop_instance() {
    asset_id::begin_build();
    let mut world = pick_world([0.0; 3], vec![]);
    let mut h = hook(vec![serde_json::json!({
        "type": "Prefab", "args": {"$id": "door_set"}
    })]);

    right_click_at(&mut world, &mut h, [640.0, 500.0]);
    assert!(h.create_menu.is_some());
    click_menu_row(
        &mut world,
        &mut h,
        &crate::editor::create_menu::MenuItem::PrefabHeader { open: false },
    );
    assert!(
        h.create_menu.as_ref().unwrap().prefabs_open,
        "the header row unfolds the section"
    );
    click_menu_row(
        &mut world,
        &mut h,
        &crate::editor::create_menu::MenuItem::Prefab("door_set".to_string()),
    );
    assert!(h.create_menu.is_none());
    let created = h.entries.last().unwrap();
    assert_eq!(entry_type(created), Some(RegisteredType::Prop));
    assert_eq!(created["args"]["prefab"], "door_set");
    assert_eq!(declared_id(created), None, "a placed instance is anonymous");
    assert_eq!(active(&h).as_deref(), Some("Prop#0"));
}

// The menu closing itself (a click away) closes it here without creating
// anything, and a press while it is open reaches no viewport pick; play mode
// and presses over the top bar / a floating panel never open it.
#[test]
fn the_create_menu_dismisses_on_click_away_and_respects_claimed_regions() {
    asset_id::begin_build();
    let mut world = pick_world([0.0; 3], vec![]);
    let mut h = hook(Vec::new());

    right_click_at(&mut world, &mut h, [640.0, 500.0]);
    assert!(h.create_menu.is_some());
    let before = h.entries.len();
    click_at(&mut world, &mut h, [30.0, 700.0]);
    assert!(h.selection.is_empty(), "the open menu takes the press");
    crate::editor::create_menu::show(&mut world, false);
    h.tick(&mut world);
    assert!(h.create_menu.is_none(), "a click away dismisses");
    assert_eq!(h.entries.len(), before, "and creates nothing");

    // The top bar's region never opens it.
    right_click_at(&mut world, &mut h, [200.0, 10.0]);
    assert!(h.create_menu.is_none());

    // A right press over an open floating panel belongs to that panel.
    let o = h.origin(PanelKey::Preview, [1280.0, 720.0]);
    right_click_at(&mut world, &mut h, [o[0] + 10.0, o[1] + 10.0]);
    assert!(h.create_menu.is_none());

    // Play mode keeps the viewport for the running world.
    h.sim.state = sim::SimState::Playing;
    right_click_at(&mut world, &mut h, [640.0, 500.0]);
    assert!(h.create_menu.is_none());
}

// The rows are read only while the menu is open, and a fresh world does not
// close it.
#[test]
fn the_closed_menu_reads_no_rows_and_a_fresh_world_keeps_it_open() {
    let mut h = hook(Vec::new());
    let mut world = World::new();
    crate::editor::install::editor_ui(&mut world);
    h.drive_create_menu(&mut world, true);
    assert!(h.create_items.is_none(), "closed, nothing read");
    h.create_menu = Some(crate::editor::hook::drive::create_menu::CreateMenu {
        pos: [0.0; 3],
        prefabs_open: false,
    });
    h.observe_create_menu(&world);
    assert!(h.create_menu.is_some(), "a fresh world is no close");
    h.drive_create_menu(&mut world, true);
    assert!(h.create_items.is_some());
}

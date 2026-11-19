//! The floating panels' stacking and pointer feedback (`hook/layout.rs`): the
//! focus order a press changes, the draw layers a publish ranks, the close
//! button, the resize cursor over a panel's edge, and what a tick shows.

use concinnity_engine::components::{FrameInput, Layout, ResizeEdges, UiEvent, UiState, UiValue};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use super::fixtures::{entry, hook, seed_tree};
use crate::editor::hook::{EditorHook, FormTarget};
use crate::editor::install;
use crate::frame_hook::FrameHook;
use concinnity_engine::protocol::CursorShape;

use crate::editor::panels::{asset_tree, assets_panel};
use crate::editor::ui::{BuiltUi, CLOSE_EVENT};

use crate::editor::panels::registry::PanelKey;

use crate::editor::hook::tests::fixtures::entry_target;

// Focusing a panel moves it to the front of the stack (drawn on top, first
// clicked) without duplicating it.
#[test]
fn focusing_a_panel_moves_it_to_the_front() {
    let mut h = hook(Vec::new());
    let panels = h.panel_order.len();
    // The default order is the registry's: the Worlds panel
    // frontmost (a session that opens on it must see it over everything), the
    // palette under it, then the Template detail (over the Templates list it
    // spawns from).
    assert_eq!(h.panel_order.last().copied(), Some(PanelKey::Worlds));
    h.focus_panel(PanelKey::Assets);
    assert_eq!(h.panel_order.last().copied(), Some(PanelKey::Assets));
    assert_eq!(h.panel_order.len(), panels, "no duplicates");
    // Re-focusing the frontmost is a no-op.
    h.focus_panel(PanelKey::Assets);
    assert_eq!(h.panel_order.last().copied(), Some(PanelKey::Assets));
    assert_eq!(h.panel_order.len(), panels);
}

// The focus order sets the ranks the engine stacks the panels by, frontmost
// highest, and the top bar is a front root, which stays above them all.
#[test]
fn the_focus_order_ranks_the_panels_under_the_top_bar() {
    let mut h = hook(Vec::new());
    h.open[PanelKey::Map] = true;
    h.open[PanelKey::Behavior] = true;
    h.focus_panel(PanelKey::Behavior);
    h.focus_panel(PanelKey::Map);
    let mut world = World::new();
    install::editor_ui(&mut world);
    h.sync_built_panels(&mut world, true, [1280.0, 720.0]);
    let built = world.resource::<BuiltUi>().unwrap();
    let rank = |key| {
        world
            .get_by_id::<Layout>(built.panel(key).unwrap().root)
            .unwrap()
            .rank
    };
    assert!(
        rank(PanelKey::Map) > rank(PanelKey::Behavior),
        "the frontmost panel outranks the others"
    );
    let bar = built.top_bar.as_ref().unwrap();
    assert!(
        world.get_by_id::<Layout>(bar.root).unwrap().front,
        "the top bar stays in front of every panel"
    );
}

// Over a resizable panel's right or bottom edge the cursor says which way a
// drag sizes it; over its body, or nowhere near it, it is the arrow.
#[test]
fn the_cursor_shows_the_edge_a_drag_would_size() {
    let mut h = hook(Vec::new());
    let mut world = World::new();
    install::editor_ui(&mut world);
    place_root(
        &mut h,
        &mut world,
        PanelKey::Behavior,
        [100.0, 100.0, 400.0, 300.0],
    );
    let at = |h: &EditorHook, world: &World, x, y| h.hover_cursor(world, [x, y]);
    assert_eq!(at(&h, &world, 498.0, 200.0), CursorShape::ResizeEW);
    assert_eq!(at(&h, &world, 300.0, 398.0), CursorShape::ResizeNS);
    assert_eq!(at(&h, &world, 498.0, 398.0), CursorShape::ResizeNWSE);
    assert_eq!(at(&h, &world, 300.0, 200.0), CursorShape::Default);
    assert_eq!(at(&h, &world, 20.0, 20.0), CursorShape::Default);

    // A panel in front whose body covers the edge keeps the arrow.
    place_root(
        &mut h,
        &mut world,
        PanelKey::Map,
        [450.0, 150.0, 300.0, 200.0],
    );
    assert_eq!(at(&h, &world, 498.0, 200.0), CursorShape::Default);

    // While a drag sizes a panel the cursor holds its shape, wherever the
    // pointer has got to.
    let root = root_of(&world, PanelKey::Behavior);
    if let Some(l) = world.get_mut_by_id::<Layout>(root) {
        l.resizing = Some(ResizeEdges {
            right: true,
            bottom: false,
        });
    }
    assert_eq!(at(&h, &world, 20.0, 20.0), CursorShape::ResizeEW);
}

fn root_of(world: &World, key: PanelKey) -> AssetId {
    world
        .resource::<BuiltUi>()
        .and_then(|b| b.panel(key))
        .unwrap()
        .root
}

// Show panel `key` laid out at `rect`, in front of those placed before it.
fn place_root(h: &mut EditorHook, world: &mut World, key: PanelKey, rect: [f32; 4]) {
    h.open[key] = true;
    h.focus_panel(key);
    let root = root_of(world, key);
    if let Some(l) = world.get_mut_by_id::<Layout>(root) {
        l.visible = true;
        l.rect = Some(rect);
    }
}

// The close button in the edit form's title bar closes the form.
#[test]
fn edit_form_close_button_closes_the_form() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let mut world = World::new();
    install::editor_ui(&mut world);
    world.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    h.open[PanelKey::Assets] = true;
    let target = entry_target(&h, 0);
    h.open_form("PointLight".to_string(), target);
    assert!(h.form_open());
    world.events_mut::<UiEvent>().send(UiEvent {
        name: CLOSE_EVENT.into(),
        element: None,
        value: Some(UiValue::Int(i32::try_from(PanelKey::Edit.index()).unwrap())),
    });
    h.tick(&mut world);
    assert!(!h.form_open(), "the close button closed the form");
}

// A tick shows the open Assets panel in each of its states: the tree, the type
// picker in its place, the row menu, and the edit form beside it.
#[test]
fn tick_shows_the_open_panel_in_every_state() {
    let mut world = World::new();
    install::editor_ui(&mut world);
    world.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        mouse_x: 1200.0,
        mouse_y: 300.0,
        ..Default::default()
    });
    let mut h = hook(vec![entry("a", "PointLight"), entry("b", "Decal")]);
    h.open[PanelKey::Assets] = true;
    seed_tree(&mut h, Vec::new());
    let built = |w: &World| {
        w.resource::<BuiltUi>()
            .unwrap()
            .panel(PanelKey::Assets)
            .cloned()
            .unwrap()
    };
    let shown = |w: &World| w.get_by_id::<Layout>(built(w).root).unwrap().visible;
    let state = |w: &World| w.resource::<UiState>().unwrap().clone();

    // Tree: the panel shows, its first row the World group header.
    h.tick(&mut world);
    assert!(shown(&world), "the panel shows");
    let tree = state(&world).list(assets_panel::TREE).to_vec();
    assert_eq!(tree[0].text, asset_tree::WORLD_GROUP);
    assert!(tree[0].expandable && tree[0].expanded);

    // Type picker: its list shows in the tree's place.
    h.assets.picker_open = true;
    h.tick(&mut world);
    assert!(!state(&world).list(assets_panel::PICKER).is_empty());

    // Row menu: Delete is offered for the "a" row, the tree's second.
    h.assets.picker_open = false;
    h.tick(&mut world);
    world.events_mut::<UiEvent>().send(UiEvent {
        name: assets_panel::CONTEXT.into(),
        element: None,
        value: Some(UiValue::Int(1)),
    });
    h.tick(&mut world);
    let s = state(&world);
    assert_eq!(s.get(assets_panel::MENU_OPEN), Some(&UiValue::Bool(true)));
    assert_eq!(s.list(assets_panel::MENU_ITEMS)[0].text, "Delete");

    // Form open: the edit panel shows alongside the tree.
    h.assets.row_menu = None;
    h.open_form("PointLight".to_string(), FormTarget::New);
    h.tick(&mut world);
    let form_shown = |w: &World| {
        let root = w
            .resource::<BuiltUi>()
            .unwrap()
            .panel(PanelKey::Edit)
            .unwrap()
            .root;
        w.get_by_id::<Layout>(root).unwrap().visible
    };
    assert!(form_shown(&world), "the form shows");
    assert!(shown(&world), "the tree stays beside the form");

    // Closing the panel and the form hides both.
    h.open[PanelKey::Assets] = false;
    h.form.close();
    h.tick(&mut world);
    assert!(!shown(&world));
    assert!(!form_shown(&world));
}

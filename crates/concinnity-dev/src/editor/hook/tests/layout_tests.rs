//! The floating panels' geometry and stacking (`hook/layout.rs`): a title-bar
//! drag that moves a panel and clamps it on screen, the focus order a press
//! changes, the draw layers a publish ranks, the close button, what a hidden
//! panel stops answering, each region's scroll, and where the secondary panels
//! park while a drag is in flight.

use concinnity_engine::components::{FrameInput, Layout, Sprite, UiEvent, UiState, UiValue};
use concinnity_engine::ecs::World;

use super::fixtures::{
    behavior, behavior_session, close_rect_of, entry, hook, seed_tree, select_behavior, set_input,
    title_rect_of, world_with_fields, world_with_input,
};
use crate::editor::behavior::panel::{BehaviorAction, ViewMode};
use crate::editor::hook::{Drag, FormTarget};
use crate::editor::{behavior, hud, inject};
use crate::frame_hook::FrameHook;

use crate::editor::panels::{asset_tree, assets_panel, form_panel};
use crate::editor::ui::BuiltUi;

use crate::editor::panels::registry::PanelKey;

use crate::editor::hook::tests::fixtures::entry_target;

// Holding a panel's title bar drags it; the origin follows the cursor by the
// grab offset and hard-stops at the window edges. Release ends the drag.
#[test]
fn title_bar_drag_moves_and_clamps_a_panel() {
    let mut h = hook(Vec::new());
    h.open[PanelKey::Map] = true;
    let vp = [1280.0, 720.0];
    let start = h.origin(PanelKey::Map, vp);
    // Press on the title bar, 10 px in from its corner.
    let mut world = world_with_input(FrameInput {
        viewport: vp,
        mouse_x: start[0] + 10.0,
        mouse_y: start[1] + 10.0,
        left_click: true,
        left_button_down: true,
        ..Default::default()
    });
    h.tick(&mut world);
    assert!(h.drag.is_some(), "the title press starts a drag");

    // Hold and move: the origin follows, preserving the grab offset.
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: 400.0,
            mouse_y: 150.0,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert_eq!(h.origin(PanelKey::Map, vp), [390.0, 140.0]);

    // Drag far past the top-left corner: the panel hard-stops at the left edge
    // and at the top bar's lower edge, never sliding under the bar.
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: -500.0,
            mouse_y: -500.0,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert_eq!(
        h.origin(PanelKey::Map, vp),
        [0.0, hud::BAR_H],
        "never partially off screen or under the top bar"
    );

    // Release ends the drag; the panel stays where it was dropped.
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            left_button_down: false,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert!(h.drag.is_none(), "release ends the drag");
    assert_eq!(h.origin(PanelKey::Map, vp), [0.0, hud::BAR_H]);
}

// The edit-form panel drags by its own title bar, independent of the Assets
// panel.
#[test]
fn edit_panel_drags_by_its_title_bar() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let mut world = World::new();
    inject::editor_hud(&mut world);
    h.open[PanelKey::Assets] = true;
    let target = entry_target(&h, 0);
    h.open_form(&mut world, "PointLight".to_string(), target);
    let vp = [1280.0, 720.0];
    let fo = h.origin(PanelKey::Edit, vp);
    world.add_component(FrameInput {
        viewport: vp,
        mouse_x: fo[0] + 12.0,
        mouse_y: fo[1] + 8.0,
        left_click: true,
        left_button_down: true,
        ..Default::default()
    });
    h.tick(&mut world);
    assert!(h.drag.is_some(), "the form title press starts a drag");
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: 112.0,
            mouse_y: 208.0,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert_eq!(h.origin(PanelKey::Edit, vp), [100.0, 200.0]);
    assert_eq!(
        h.origin(PanelKey::Assets, vp),
        assets_panel::default_origin(vp[0]),
        "the Assets panel did not move"
    );
}

// Focusing a panel moves it to the front of the stack (drawn on top, first
// clicked) without duplicating it.
#[test]
fn focusing_a_panel_moves_it_to_the_front() {
    let mut h = hook(Vec::new());
    let panels = h.panel_order.len();
    // Default order matches the injected draw order: the Worlds panel
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

// The published HUD layers rank the panels by focus (frontmost highest) and pin
// the top bar above them all, so the renderer occludes overlaps cleanly.
#[test]
fn publish_layers_ranks_panels_below_the_top_bar() {
    let mut h = hook(Vec::new());
    h.focus_panel(PanelKey::Edit); // Edit -> frontmost
    let layers = h.compute_layers(None);
    let layer = |id| *layers.get(&id).expect("id mapped");
    let edit = layer(form_panel::EDIT_BG);
    let behavior = layer(behavior::panel::PANEL_BG);
    assert!(edit > behavior, "the frontmost panel outranks the others");
    assert!(
        layer(hud::SAVE_BUTTON) > edit,
        "the top bar sits above every panel"
    );
    // A panel's text input shares its panel's layer (it must not sink below it).
    assert_eq!(layer(form_panel::NAME_INPUT), edit);
}

// A press on a shown panel brings it to the front and (on its title bar) starts
// a drag.
#[test]
fn a_panel_press_brings_it_to_the_front() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let mut world = world_with_fields();
    h.open[PanelKey::Map] = true;
    let vp = [1280.0, 720.0];
    let t = title_rect_of(&h, PanelKey::Map, vp);
    // The title bar's interior (clear of the corner / edge resize band) drags.
    let claimed = h.try_panel_press(
        PanelKey::Map,
        t[0] + t[2] * 0.5,
        t[1] + t[3] * 0.5,
        vp,
        &mut world,
    );
    assert!(claimed, "the press was claimed by the Map panel");
    assert_eq!(h.panel_order.last().copied(), Some(PanelKey::Map));
    assert!(h.drag.is_some(), "a title-bar press starts a drag");
}

// The X in the edit form's title bar closes the form: the hook routes it before
// the title-bar drag, so it closes rather than starting a drag.
#[test]
fn edit_form_title_bar_x_closes_the_form() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let mut world = world_with_fields();
    h.open[PanelKey::Assets] = true;
    let target = entry_target(&h, 0);
    h.open_form(&mut world, "PointLight".to_string(), target);
    assert!(h.form_open());
    let vp = [1280.0, 720.0];
    let x = form_panel::close_rect(h.origin(PanelKey::Edit, vp), form_panel::EDIT_W);
    let claimed = h.try_panel_press(PanelKey::Edit, x[0] + 5.0, x[1] + 5.0, vp, &mut world);
    assert!(claimed, "the X press was claimed");
    assert!(!h.form_open(), "the X closed the form");
    assert!(h.drag.is_none(), "the X did not start a drag");
}

// Every floating panel's title-bar X closes it: the press is checked before the
// title drag, so it closes rather than starting a drag.
#[test]
fn every_panel_title_bar_x_closes_it() {
    let vp = [1280.0, 720.0];
    let mut world = world_with_fields();

    let mut h = hook(Vec::new());
    h.open[PanelKey::Map] = true;
    let ax = close_rect_of(&h, PanelKey::Map, vp);
    assert!(h.try_panel_press(PanelKey::Map, ax[0] + 5.0, ax[1] + 5.0, vp, &mut world));
    assert!(
        !h.open[PanelKey::Map] && h.drag.is_none(),
        "Map X closed it, no drag"
    );
}

// A panel toggled off (its View checkbox unticked) is not interactive: a press
// where it would be falls through instead of being claimed.
#[test]
fn a_hidden_panel_is_not_interactive() {
    let mut h = hook(Vec::new());
    let vp = [1280.0, 720.0];
    let mut world = world_with_fields();
    // The Map panel starts hidden: its press falls through until it is
    // opened.
    let at = title_rect_of(&h, PanelKey::Map, vp);
    assert!(!h.try_panel_press(PanelKey::Map, at[0] + 5.0, at[1] + 5.0, vp, &mut world));
    h.open[PanelKey::Map] = true;
    assert!(h.try_panel_press(PanelKey::Map, at[0] + 5.0, at[1] + 5.0, vp, &mut world));
}

// An open Behavior panel claims only presses that land on it. Its chart views
// used to answer for the whole screen, which left every other panel unable to
// be moved, closed, or brought forward while one of them was showing.
#[test]
fn the_behavior_panel_leaves_the_other_panels_pressable_in_every_view() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": [{"hide": {"target": "self"}}]}),
    )]);
    let vp = [1280.0, 720.0];
    h.open[PanelKey::Map] = true;
    // Behavior sits in front of the panels the press has to reach.
    h.focus_panel(PanelKey::Behavior);

    for mode in [ViewMode::Outline, ViewMode::Chart, ViewMode::Overview] {
        h.behavior.mode = mode;
        let title = title_rect_of(&h, PanelKey::Map, vp);
        let (mx, my) = (title[0] + 5.0, title[1] + 5.0);
        assert!(
            !h.try_panel_press(PanelKey::Behavior, mx, my, vp, &mut world),
            "{mode:?}: Behavior swallowed a press meant for the Map"
        );
        h.drag = None;
        assert!(
            h.try_panel_press(PanelKey::Map, mx, my, vp, &mut world),
            "{mode:?}: the Map never saw the press"
        );
        h.drag = None;
        h.focus_panel(PanelKey::Behavior);
    }
}

// A tick shows the open Assets panel in each of its states: the tree, the type
// picker in its place, the row menu, and the edit form beside it.
#[test]
fn tick_shows_the_open_panel_in_every_state() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
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
    h.open_form(&mut world, "PointLight".to_string(), FormTarget::New);
    h.tick(&mut world);
    assert!(
        world
            .get_by_id::<Sprite>(form_panel::EDIT_BG)
            .unwrap()
            .visible,
        "the form shows"
    );
    assert!(shown(&world), "the tree stays beside the form");

    // Closing the panel and the form hides both.
    h.open[PanelKey::Assets] = false;
    h.form.close();
    h.tick(&mut world);
    assert!(!shown(&world));
    assert!(
        !world
            .get_by_id::<Sprite>(form_panel::EDIT_BG)
            .unwrap()
            .visible
    );
}

#[test]
fn drive_drag_parks_each_secondary_panel() {
    let vp = [1280.0, 720.0];
    let held = FrameInput {
        left_button_down: true,
        mouse_x: 220.0,
        mouse_y: 160.0,
        ..Default::default()
    };

    let mut h = hook(Vec::new());
    h.drag = Some(Drag {
        key: PanelKey::View,
        grab: [10.0, 10.0],
    });
    h.drive_drag(&held, vp);
    assert!(
        h.positions[PanelKey::View].is_some(),
        "the View panel follows the cursor"
    );

    let mut h = hook(Vec::new());
    h.drag = Some(Drag {
        key: PanelKey::Templates,
        grab: [10.0, 10.0],
    });
    h.drive_drag(&held, vp);
    assert!(
        h.positions[PanelKey::Templates].is_some(),
        "the Templates panel follows"
    );

    let mut h = hook(Vec::new());
    h.templates.detail = Some(0);
    h.drag = Some(Drag {
        key: PanelKey::TemplateDetail,
        grab: [10.0, 10.0],
    });
    h.drive_drag(&held, vp);
    assert!(
        h.positions[PanelKey::TemplateDetail].is_some(),
        "the Template detail panel follows"
    );
}

// An open overlay draws above its own panel but still below the panel in front
// of it. This is what lets an opaque backing occlude what it covers, instead of
// each panel blanking the elements it happens to sit over.
#[test]
fn an_open_overlay_layers_above_its_panel_and_below_the_one_in_front() {
    let (mut h, mut world) = behavior_session(vec![behavior(
        "chase",
        serde_json::json!({"on": "tick", "do": [{"save": {}}]}),
    )]);
    // Nothing open: the palette sits at its panel's own layer.
    let flat = h.compute_layers(None);
    let panel_layer = flat[&behavior::panel::PANEL_BG];
    assert_eq!(flat[&behavior::panel::DROP_BG], panel_layer);

    select_behavior(&mut h, &mut world, "do");
    h.apply_behavior_action(BehaviorAction::Palette, &mut world, [0.0, 0.0]);
    let open = h.compute_layers(None);
    for id in behavior::panel::palette_ids() {
        assert!(
            open[&id] > open[&behavior::panel::PANEL_BG],
            "{id:?} does not draw above its own panel",
        );
        assert!(
            open[&id] > open[&behavior::panel::row_label(0)],
            "{id:?} does not draw above the rows it covers",
        );
    }
    // A panel focused in front still clears the whole band, overlay included.
    h.form.selected_type = Some("PointLight".into());
    h.focus_panel(PanelKey::Edit);
    let stacked = h.compute_layers(None);
    let front = stacked[&form_panel::EDIT_BG];
    for id in behavior::panel::palette_ids() {
        assert!(stacked[&id] < front, "{id:?} escaped above the front panel");
    }
}

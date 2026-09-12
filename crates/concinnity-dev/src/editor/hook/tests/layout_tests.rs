// src/editor/hook/tests/layout_tests.rs
//
// The floating panels' geometry and stacking (`hook/layout.rs`): a title-bar
// drag that moves a panel and clamps it on screen, the focus order a press
// changes, the draw layers a publish ranks, the close button, what a hidden
// panel stops answering, each region's scroll, and where the secondary panels
// park while a drag is in flight.

use concinnity_core::components::FrameInput;
use concinnity_core::components::Sprite;
use concinnity_core::components::TextInput;
use concinnity_core::components::TextLabel;
use concinnity_core::ecs::World;
use concinnity_host::thread::asset_id;

use super::fixtures::{
    behavior, behavior_session, close_rect_of, entry, hook, seed_tree, select_behavior, set_input,
    title_rect_of, world_with_fields, world_with_input,
};
use crate::debug_hook::DebugHook;
use crate::editor::behavior;
use crate::editor::behavior::panel::{BehaviorAction, ViewMode};
use crate::editor::hook::{Drag, FormTarget};
use crate::editor::hud;
use crate::editor::inject;

use crate::editor::panels::form_panel;

use crate::editor::panels::panel;
use crate::editor::panels::preview;
use crate::editor::panels::registry::PanelKey;

use crate::editor::widget;

// Holding a panel's title bar drags it; the origin follows the cursor by the
// grab offset and hard-stops at the window edges. Release ends the drag.
#[test]
fn title_bar_drag_moves_and_clamps_the_assets_panel() {
    let mut h = hook(Vec::new());
    h.panel_open = true;
    let vp = [1280.0, 720.0];
    let start = h.origin(PanelKey::Assets, vp);
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
    assert_eq!(h.origin(PanelKey::Assets, vp), [390.0, 140.0]);

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
        h.origin(PanelKey::Assets, vp),
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
    assert_eq!(h.origin(PanelKey::Assets, vp), [0.0, hud::BAR_H]);
}

// The Preview panel drags by its own title bar, clamped to the window's far
// corner by its own (smaller) footprint.
#[test]
fn title_bar_drag_moves_and_clamps_the_preview_panel() {
    let mut h = hook(Vec::new());
    let vp = [1280.0, 720.0];
    let start = h.origin(PanelKey::Preview, vp);
    let mut world = world_with_input(FrameInput {
        viewport: vp,
        mouse_x: start[0] + 5.0,
        mouse_y: start[1] + 5.0,
        left_click: true,
        left_button_down: true,
        ..Default::default()
    });
    h.tick(&mut world);
    assert!(h.drag.is_some());
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: 5000.0,
            mouse_y: 5000.0,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    let size = preview::size();
    assert_eq!(
        h.origin(PanelKey::Preview, vp),
        [vp[0] - size[0], vp[1] - size[1]],
        "stops flush with the bottom-right corner"
    );
}

// The edit-form panel drags by its own title bar, independent of the Assets
// panel.
#[test]
fn edit_panel_drags_by_its_title_bar() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let mut world = World::new();
    inject::editor_hud(&mut world);
    h.panel_open = true;
    h.open_form(&mut world, "PointLight".to_string(), FormTarget::Entry(0));
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
        panel::default_origin(vp[0]),
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
    let layers = h.compute_layers();
    let layer = |id| *layers.get(&id).expect("id mapped");
    let edit = layer(form_panel::EDIT_BG);
    let assets = layer(panel::PANEL_BG);
    let preview = layer(preview::PANEL_BG);
    assert!(
        edit > assets && edit > preview,
        "the frontmost panel outranks the others"
    );
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
    h.panel_open = true;
    let vp = [1280.0, 720.0];
    let po = h.origin(PanelKey::Assets, vp);
    let t = widget::title_rect(po, panel::PANEL_W);
    // The title bar's interior (clear of the corner / edge resize band) drags.
    let claimed = h.try_panel_press(
        PanelKey::Assets,
        t[0] + t[2] * 0.5,
        t[1] + t[3] * 0.5,
        vp,
        &mut world,
    );
    assert!(claimed, "the press was claimed by the Assets panel");
    assert_eq!(h.panel_order.last().copied(), Some(PanelKey::Assets));
    assert!(h.drag.is_some(), "a title-bar press starts a drag");
}

// The X in the edit form's title bar closes the form: the hook routes it before
// the title-bar drag, so it closes rather than starting a drag.
#[test]
fn edit_form_title_bar_x_closes_the_form() {
    let mut h = hook(vec![entry("lamp", "PointLight")]);
    let mut world = world_with_fields();
    h.panel_open = true;
    h.open_form(&mut world, "PointLight".to_string(), FormTarget::Entry(0));
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

    // Preview starts shown; its X hides it.
    let mut h = hook(Vec::new());
    let px = close_rect_of(&h, PanelKey::Preview, vp);
    assert!(h.try_panel_press(PanelKey::Preview, px[0] + 5.0, px[1] + 5.0, vp, &mut world));
    assert!(
        !h.preview_open && h.drag.is_none(),
        "Preview X closed it, no drag"
    );

    // Assets.
    let mut h = hook(Vec::new());
    h.panel_open = true;
    let ax = close_rect_of(&h, PanelKey::Assets, vp);
    assert!(h.try_panel_press(PanelKey::Assets, ax[0] + 5.0, ax[1] + 5.0, vp, &mut world));
    assert!(
        !h.panel_open && h.drag.is_none(),
        "Assets X closed it, no drag"
    );

    // View.
    let mut h = hook(Vec::new());
    h.view_open = true;
    let vx = close_rect_of(&h, PanelKey::View, vp);
    assert!(h.try_panel_press(PanelKey::View, vx[0] + 5.0, vx[1] + 5.0, vp, &mut world));
    assert!(
        !h.view_open && h.drag.is_none(),
        "View X closed it, no drag"
    );

    // Templates.
    let mut h = hook(Vec::new());
    h.templates_open = true;
    let tx = close_rect_of(&h, PanelKey::Templates, vp);
    assert!(h.try_panel_press(
        PanelKey::Templates,
        tx[0] + 5.0,
        tx[1] + 5.0,
        vp,
        &mut world
    ));
    assert!(
        !h.templates_open && h.drag.is_none(),
        "Templates X closed it, no drag"
    );
}

// A panel toggled off (its View checkbox unticked) is not interactive: a press
// where it would be falls through instead of being claimed.
#[test]
fn a_hidden_panel_is_not_interactive() {
    let mut h = hook(Vec::new());
    let vp = [1280.0, 720.0];
    let mut world = world_with_fields();
    // Preview starts shown: a title-bar press is claimed (starts a drag).
    let pt = title_rect_of(&h, PanelKey::Preview, vp);
    assert!(h.try_panel_press(PanelKey::Preview, pt[0] + 5.0, pt[1] + 5.0, vp, &mut world));
    // Hidden: the same press falls through.
    h.drag = None;
    h.preview_open = false;
    assert!(!h.try_panel_press(PanelKey::Preview, pt[0] + 5.0, pt[1] + 5.0, vp, &mut world));
    // The View panel starts hidden: its press falls through until it is opened.
    let vt = title_rect_of(&h, PanelKey::View, vp);
    assert!(!h.try_panel_press(PanelKey::View, vt[0] + 5.0, vt[1] + 5.0, vp, &mut world));
    h.view_open = true;
    assert!(h.try_panel_press(PanelKey::View, vt[0] + 5.0, vt[1] + 5.0, vp, &mut world));
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
    h.view_open = true;
    // Behavior sits in front of the panels the press has to reach.
    h.focus_panel(PanelKey::Behavior);

    for mode in [ViewMode::Outline, ViewMode::Chart, ViewMode::Overview] {
        h.behavior_mode = mode;
        for key in [PanelKey::Preview, PanelKey::View] {
            let title = title_rect_of(&h, key, vp);
            let (mx, my) = (title[0] + 5.0, title[1] + 5.0);
            assert!(
                !h.try_panel_press(PanelKey::Behavior, mx, my, vp, &mut world),
                "{mode:?}: Behavior swallowed a press meant for {key:?}"
            );
            h.drag = None;
            assert!(
                h.try_panel_press(key, mx, my, vp, &mut world),
                "{mode:?}: {key:?} never saw the press"
            );
            h.drag = None;
            h.focus_panel(PanelKey::Behavior);
        }
    }
}

// The Templates panel drags by its own title bar and comes to the front on a
// press, like the other floating panels.
#[test]
fn templates_panel_press_drags_and_focuses() {
    let mut h = hook(Vec::new());
    h.templates_open = true;
    let vp = [1280.0, 720.0];
    let mut world = world_with_fields();
    let t = title_rect_of(&h, PanelKey::Templates, vp);
    // The title bar's interior (clear of the corner / edge resize band) drags.
    assert!(h.try_panel_press(
        PanelKey::Templates,
        t[0] + t[2] * 0.5,
        t[1] + t[3] * 0.5,
        vp,
        &mut world
    ));
    assert!(h.drag.is_some(), "a title-bar press starts a drag");
    assert_eq!(h.panel_order.last().copied(), Some(PanelKey::Templates));
}

// Drive `tick` against a fully injected HUD world in each panel body state,
// exercising the real `panel::apply` layout path (not just the pure hit-test /
// action logic the other tests cover).
#[test]
fn tick_lays_out_the_open_panel_in_every_state() {
    let sprite_visible = |w: &World, id: asset_id::AssetId| {
        w.query::<Sprite>()
            .find(|s| s.asset_id == id)
            .unwrap()
            .visible
    };
    let label = |w: &World, id: asset_id::AssetId| {
        w.query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .unwrap()
            .clone()
    };

    let mut world = World::new();
    inject::editor_hud(&mut world);
    world.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        mouse_x: 1200.0,
        mouse_y: 300.0,
        ..Default::default()
    });
    let mut h = hook(vec![entry("a", "PointLight"), entry("b", "Decal")]);
    h.panel_open = true;
    seed_tree(&mut h, Vec::new());

    // Tree: panel drawn, first row is the World group header.
    h.tick(&mut world);
    assert!(sprite_visible(&world, panel::PANEL_BG), "panel bg shown");
    let row0 = label(&world, panel::name_label(0));
    assert!(
        row0.visible && row0.content.starts_with("- World"),
        "first row is the World group header, got {:?}",
        row0.content
    );

    // Type picker: the solid backing and the search field show.
    h.picker_open = true;
    h.tick(&mut world);
    assert!(
        sprite_visible(&world, panel::PICKER_BG),
        "picker backing shown"
    );
    assert!(
        world
            .query::<TextInput>()
            .find(|t| t.asset_id == panel::SEARCH_INPUT)
            .unwrap()
            .visible
    );

    // Row menu: the Delete popup shows over the "a" row.
    h.picker_open = false;
    h.row_menu = Some("a".to_string());
    h.tick(&mut world);
    assert!(sprite_visible(&world, panel::MENU_BG), "row menu shown");
    assert_eq!(label(&world, panel::MENU_DELETE_LABEL).content, "Delete");

    // Form open: the edit panel shows alongside the browse list, with its
    // title bar, name heading, and confirm button.
    h.row_menu = None;
    h.open_form(&mut world, "PointLight".to_string(), FormTarget::New);
    h.tick(&mut world);
    assert!(
        sprite_visible(&world, form_panel::APPLY_BG),
        "confirm button shown"
    );
    assert_eq!(
        label(&world, form_panel::TITLE_LABEL).content,
        "New PointLight"
    );
    assert_eq!(label(&world, form_panel::APPLY_LABEL).content, "Add");
    assert!(
        world
            .query::<TextInput>()
            .find(|t| t.asset_id == form_panel::NAME_INPUT)
            .unwrap()
            .visible,
        "the name heading shows"
    );
    assert!(
        label(&world, panel::name_label(0)).visible,
        "the tree stays visible beside the form"
    );

    // Closing the panel + form blanks both.
    h.panel_open = false;
    h.close_form();
    h.tick(&mut world);
    assert!(!sprite_visible(&world, panel::PANEL_BG), "panel bg hidden");
    assert!(
        !sprite_visible(&world, form_panel::EDIT_BG),
        "form panel hidden"
    );
}

#[test]
fn scroll_moves_each_regions_offset() {
    let world = world_with_fields();
    // A tree longer than the row window, so its scroll can advance.
    let mut h = hook(
        (0..20)
            .map(|i| entry(&format!("log{i}"), "Logger"))
            .collect(),
    );
    seed_tree(&mut h, Vec::new());
    h.row_menu = Some("log0".to_string());
    h.scroll_tree(1.0, &world);
    assert!(h.tree_scroll > 0, "a closed picker scrolls the tree");
    assert!(h.row_menu.is_none(), "scrolling dismisses an open row menu");
    h.scroll_tree(-1.0, &world);
    assert_eq!(h.tree_scroll, 0, "scrolling back up clamps at the top");

    // An open picker scrolls its own option list instead of the tree.
    h.picker_open = true;
    let before = h.tree_scroll;
    h.scroll_tree(1.0, &world);
    assert_eq!(
        h.tree_scroll, before,
        "the tree stays put while the picker is open"
    );
    assert!(h.picker_scroll > 0, "the picker's own list scrolled");
    h.picker_open = false;

    // A picked template detail scrolls its own asset list.
    h.open_template = Some(0);
    h.scroll_template_list(1.0);
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
        h.positions[PanelKey::View.index()].is_some(),
        "the View panel follows the cursor"
    );

    let mut h = hook(Vec::new());
    h.drag = Some(Drag {
        key: PanelKey::Templates,
        grab: [10.0, 10.0],
    });
    h.drive_drag(&held, vp);
    assert!(
        h.positions[PanelKey::Templates.index()].is_some(),
        "the Templates panel follows"
    );

    let mut h = hook(Vec::new());
    h.open_template = Some(0);
    h.drag = Some(Drag {
        key: PanelKey::TemplateDetail,
        grab: [10.0, 10.0],
    });
    h.drive_drag(&held, vp);
    assert!(
        h.positions[PanelKey::TemplateDetail.index()].is_some(),
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
    let flat = h.compute_layers();
    let panel_layer = flat[&behavior::panel::PANEL_BG];
    assert_eq!(flat[&behavior::panel::DROP_BG], panel_layer);

    select_behavior(&mut h, &mut world, "do");
    h.apply_behavior_action(BehaviorAction::Palette, &mut world, [0.0, 0.0]);
    let open = h.compute_layers();
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
    h.focus_panel(PanelKey::Preview);
    let stacked = h.compute_layers();
    let front = stacked[&preview::PANEL_BG];
    for id in behavior::panel::palette_ids() {
        assert!(stacked[&id] < front, "{id:?} escaped above the front panel");
    }
}

// src/editor/hook/modal_tests.rs
//
// Tests for the confirmation dialog: the open / press / close flow, the
// screen-modal press and wheel lockout, and its place in the draw layers.

use super::*;
use crate::components::{Sprite, TextLabel};

const VP: [f32; 2] = [1280.0, 720.0];

fn hook() -> EditorHook {
    EditorHook::new("unused.jsonl".to_string(), Vec::new())
}

fn world_with_input(input: FrameInput) -> World {
    let mut world = World::new();
    world.add_component(input);
    world
}

fn plain(label: &str) -> modal::Button {
    modal::Button {
        label: label.to_string(),
        danger: false,
        action: modal::Action::Dismiss,
    }
}

fn confirm_buttons() -> Vec<modal::Button> {
    vec![
        plain("Cancel"),
        modal::Button {
            label: "Delete".to_string(),
            danger: true,
            action: modal::Action::Dismiss,
        },
    ]
}

fn click_at(x: f32, y: f32) -> FrameInput {
    FrameInput {
        left_click: true,
        mouse_x: x,
        mouse_y: y,
        viewport: VP,
        ..Default::default()
    }
}

#[test]
fn open_truncates_to_the_button_pool() {
    let mut h = hook();
    h.open_modal("m", vec![plain("a"), plain("b"), plain("c"), plain("d")]);
    assert_eq!(h.modal.as_ref().unwrap().buttons.len(), modal::MAX_BUTTONS);
}

#[test]
fn a_button_press_closes_the_dialog_and_a_press_elsewhere_does_not() {
    let mut h = hook();
    let mut world = World::new();
    // No dialog: the press is not claimed and routing continues.
    assert!(!h.route_modal_click(&click_at(5.0, 5.0), VP, &mut world));
    h.open_modal("Delete the world?", confirm_buttons());
    // A press on the dimmed screen and one on the dialog's own chrome are
    // both swallowed and change nothing: a click-away is not a cancel.
    let p = modal::panel_rect(VP);
    assert!(h.route_modal_click(&click_at(5.0, 5.0), VP, &mut world));
    assert!(h.route_modal_click(&click_at(p[0] + 2.0, p[1] + 2.0), VP, &mut world));
    assert!(h.modal.is_some());
    // A button press runs its action and closes the dialog.
    let r = modal::button_rect(p, 2, 1);
    assert!(h.route_modal_click(&click_at(r[0] + 2.0, r[1] + 2.0), VP, &mut world));
    assert!(h.modal.is_none());
}

// While the dialog is up, a press that would reach the top bar or a panel's
// close button is swallowed: the dialog locks out everything behind it.
#[test]
fn an_open_dialog_swallows_presses_to_the_bar_and_panels() {
    let mut h = hook();
    h.open_modal("m", confirm_buttons());
    let bar = hud::layout(VP[0]);
    let view = [bar.view[0] + 2.0, bar.view[1] + 2.0];
    let mut world = world_with_input(click_at(view[0], view[1]));
    h.tick(&mut world);
    assert!(!h.view_open, "the View chip never saw the press");

    assert!(h.preview_open, "Preview starts shown");
    let o = h.origin(PanelKey::Preview, VP);
    let title = widget::title_rect(o, registry::panel(PanelKey::Preview).size(&h)[0]);
    let close = widget::close_rect(title);
    let mut world = world_with_input(click_at(close[0] + 2.0, close[1] + 2.0));
    h.tick(&mut world);
    assert!(h.preview_open, "the close button never saw the press");
    assert!(
        h.modal.is_some(),
        "a press off the buttons keeps the dialog up"
    );

    // The same top-bar press lands once the dialog is gone.
    h.modal = None;
    let mut world = world_with_input(click_at(view[0], view[1]));
    h.tick(&mut world);
    assert!(h.view_open);
}

// While the dialog is up the wheel is swallowed too: a scrollable panel under
// the cursor does not move.
#[test]
fn an_open_dialog_swallows_the_wheel() {
    let mut h = hook();
    h.panel_open = true;
    h.preview_open = false;
    // Enough tree rows that the Assets panel can scroll.
    h.tree_groups = vec![TreeGroup {
        label: asset_tree::WORLD_GROUP.to_string(),
        assets: (0..40)
            .map(|i| asset_tree::TreeAsset {
                name: format!("a{i}"),
                asset_type: "Prop".to_string(),
                badge: asset_tree::Badge::Authored,
                promote: None,
            })
            .collect(),
    }];
    h.tree_unfolded = vec![0];
    h.tree_stale = false;
    let o = h.origin(PanelKey::Assets, VP);
    let s = h.effective_size(PanelKey::Assets);
    let input = FrameInput {
        scroll_delta: 2.0,
        mouse_x: o[0] + s[0] * 0.5,
        mouse_y: o[1] + s[1] - 10.0,
        viewport: VP,
        ..Default::default()
    };
    h.open_modal("m", confirm_buttons());
    let mut world = world_with_input(input.clone());
    h.tick(&mut world);
    assert_eq!(
        h.tree_scroll, 0,
        "the wheel is swallowed while the dialog is up"
    );
    h.modal = None;
    let mut world = world_with_input(input);
    h.tick(&mut world);
    assert_eq!(h.tree_scroll, 1, "the same wheel scrolls once it closes");
}

// The dialog's elements draw above every panel, the top bar, and the toast
// stack while open, and drop out of the layer map when it closes.
#[test]
fn dialog_layers_sit_above_all_other_chrome() {
    let mut h = hook();
    h.notifier.push(notify::Level::Info, "toast");
    h.open_modal("m", confirm_buttons());
    let layers = h.compute_layers();
    let dialog = layers[&modal::all_sprite_ids()[0]];
    assert!(dialog > TOP_BAR_LAYER);
    assert!(dialog > layers[&toast_overlay::all_sprite_ids()[0]]);
    for id in modal::all_sprite_ids()
        .into_iter()
        .chain(modal::all_label_ids())
    {
        assert_eq!(layers[&id], dialog);
    }
    h.modal = None;
    assert!(!h.compute_layers().contains_key(&modal::all_sprite_ids()[0]));
}

// The draw pass shows the dialog only while it is open and the HUD is shown;
// the hidden pass blanks the elements but keeps the state.
#[test]
fn draw_shows_while_open_and_hides_otherwise() {
    let mut h = hook();
    let mut world = World::new();
    for id in modal::all_sprite_ids() {
        world.add_component(Sprite {
            asset_id: id,
            ..Default::default()
        });
    }
    for id in modal::all_label_ids() {
        world.add_component(TextLabel {
            asset_id: id,
            ..Default::default()
        });
    }
    h.drive_modal_draw(&mut world, VP, true, [0.0, 0.0]);
    assert!(
        world.query::<Sprite>().all(|s| !s.visible),
        "closed draws nothing"
    );
    h.open_modal("m", confirm_buttons());
    h.drive_modal_draw(&mut world, VP, true, [0.0, 0.0]);
    assert!(world.query::<Sprite>().any(|s| s.visible));
    h.drive_modal_draw(&mut world, VP, false, [0.0, 0.0]);
    assert!(world.query::<Sprite>().all(|s| !s.visible));
    assert!(world.query::<TextLabel>().all(|l| !l.visible));
    assert!(h.modal.is_some(), "hiding keeps the dialog's state");
}

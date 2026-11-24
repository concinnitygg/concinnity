//! Tests for the confirmation dialog: the open / press / close flow, the
//! presses the hook leaves to it while it is open, what is typed into a
//! prompt, and its place in the draw layers.

use concinnity_engine::components::{FrameInput, Layout, TextField, UiEvent, UiValue};
use concinnity_engine::ecs::World;

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::{VP, press_modal_check};
use crate::editor::install;
use crate::editor::modal::{self, Check};
use crate::editor::panels::registry::PanelKey;
use crate::editor::ui::BuiltUi;
use crate::frame_hook::FrameHook;

fn hook() -> EditorHook {
    EditorHook::new("unused.jsonl".to_string(), Vec::new())
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

fn press(h: &mut EditorHook, world: &mut World, i: i32) {
    world.events_mut::<UiEvent>().send(UiEvent {
        name: "editor.dialog.button".into(),
        element: None,
        value: Some(UiValue::Int(i)),
    });
    h.read_ui_events(world);
}

// A menu is a popup, which draws over the dialog, so opening a dialog closes
// the Display and create menus rather than leave them over it.
#[test]
fn opening_a_dialog_closes_the_menus() {
    let mut h = hook();
    h.display_menu_open = true;
    h.create_menu = Some(crate::editor::hook::drive::create_menu::CreateMenu {
        pos: [0.0; 3],
        prefabs_open: false,
    });
    h.open_modal("m", confirm_buttons());
    assert!(!h.display_menu_open && h.create_menu.is_none());
}

#[test]
fn open_truncates_to_the_button_pool() {
    let mut h = hook();
    h.open_modal("m", vec![plain("a"), plain("b"), plain("c"), plain("d")]);
    assert_eq!(h.modal.as_ref().unwrap().buttons.len(), modal::MAX_BUTTONS);
}

// A button the dialog reports closes it; a slot it does not use changes
// nothing.
#[test]
fn a_button_press_closes_the_dialog() {
    let mut h = hook();
    let mut world = World::new();
    h.open_modal("Delete the world?", confirm_buttons());
    press(&mut h, &mut world, 2);
    assert!(h.modal.is_some(), "an unused slot is no button");
    press(&mut h, &mut world, 1);
    assert!(h.modal.is_none());
}

// The checkbox flips only while it may change.
#[test]
fn the_checkbox_flips_unless_disabled() {
    let mut h = hook();
    let mut world = World::new();
    let check = |enabled| Check {
        caption: "Delete its files".into(),
        on: false,
        enabled,
        note: None,
    };
    h.open_modal_with_check("m", confirm_buttons(), check(true));
    press_modal_check(&mut h, &mut world);
    assert!(h.modal.as_ref().unwrap().check.as_ref().unwrap().on);
    h.open_modal_with_check("m", confirm_buttons(), check(false));
    press_modal_check(&mut h, &mut world);
    assert!(!h.modal.as_ref().unwrap().check.as_ref().unwrap().on);
}

// While the dialog is up the hook routes no press: a click on a panel's close
// button leaves the panel, and the press is left for the dialog to take.
#[test]
fn an_open_dialog_leaves_every_press_to_itself() {
    let mut h = hook();
    h.open_modal("m", confirm_buttons());
    assert!(h.open[PanelKey::Preview], "Preview starts shown");
    let o = h.origin(PanelKey::Preview, VP);
    let mut world = World::new();
    world.add_component(FrameInput {
        left_click: true,
        mouse_x: o[0] + 10.0,
        mouse_y: o[1] + 10.0,
        viewport: VP,
        ..Default::default()
    });
    h.tick(&mut world);
    assert!(h.open[PanelKey::Preview]);
    assert!(h.modal.is_some());
    let input = world.query::<FrameInput>().last().unwrap();
    assert!(input.left_click, "not withheld from the dialog");
}

// A world built for the editor, its window `VP` big.
fn editor_world() -> World {
    let mut world = World::new();
    install::editor_ui(&mut world);
    world.add_component(FrameInput {
        viewport: VP,
        ..Default::default()
    });
    world
}

// What is typed into a prompt's field is the dialog's text, which a fresh
// world's field shows again.
#[test]
fn a_prompts_text_follows_its_field() {
    let mut h = hook();
    let mut world = editor_world();
    h.open_prompt("Name it", confirm_buttons());
    h.tick(&mut world);
    let field = world
        .resource::<BuiltUi>()
        .and_then(|b| b.dialog.as_ref())
        .map(|d| d.root)
        .unwrap();
    assert!(world.get_by_id::<Layout>(field).unwrap().visible);
    let typed = world.query_mut::<TextField>().find(|f| f.focused).unwrap();
    typed.content = "gym".into();
    h.tick(&mut world);
    assert_eq!(h.modal.as_ref().unwrap().text, "gym");

    let mut fresh = editor_world();
    h.tick(&mut fresh);
    let shown = fresh.query::<TextField>().find(|f| f.focused).unwrap();
    assert_eq!(
        shown.content, "gym",
        "the text is the hook's, not the world's"
    );
}

// The dialog is a modal root, which the engine stacks in front of every panel
// and the top bar while it shows, and hides once it closes.
#[test]
fn the_dialog_is_a_modal_root_shown_while_open() {
    let mut h = hook();
    let mut world = editor_world();
    let root = world
        .resource::<BuiltUi>()
        .and_then(|b| b.dialog.as_ref())
        .unwrap()
        .root;
    h.open_modal("m", confirm_buttons());
    h.tick(&mut world);
    let dialog = |world: &World| world.get_by_id::<Layout>(root).unwrap().clone();
    assert!(dialog(&world).modal && dialog(&world).visible);
    h.modal = None;
    h.tick(&mut world);
    assert!(!dialog(&world).visible);
}

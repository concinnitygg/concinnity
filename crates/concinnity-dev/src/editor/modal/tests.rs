use super::*;
use concinnity_engine::components::UiState;

fn built(world: &mut World) -> BuiltDialog {
    build(&mut UiBuilder::new(world, AssetId(9), None)).unwrap()
}

fn dialog(field: bool, check: Option<Check>) -> Dialog {
    Dialog {
        message: "Delete it?".into(),
        buttons: vec![
            Button {
                label: "Cancel".into(),
                danger: false,
                action: Action::Dismiss,
            },
            Button {
                label: "Delete".into(),
                danger: true,
                action: Action::Dismiss,
            },
        ],
        field,
        text: "half".into(),
        check,
        shown: false,
    }
}

fn layout(world: &World, id: AssetId) -> &Layout {
    world.get_by_id::<Layout>(id).unwrap()
}

#[test]
fn events_name_the_button_or_the_checkbox() {
    let press = UiEvent {
        name: PRESS.into(),
        element: None,
        value: Some(UiValue::Int(2)),
    };
    assert_eq!(event(&press), Some(DialogEvent::Press(2)));
    let check = UiEvent {
        name: CHECK.into(),
        element: None,
        value: Some(UiValue::Bool(true)),
    };
    assert_eq!(event(&check), Some(DialogEvent::Check));
}

// The dialog is a modal root, hidden until a dialog is open.
#[test]
fn the_dialog_is_a_modal_root_shown_only_while_open() {
    let mut world = World::new();
    let d = built(&mut world);
    assert!(layout(&world, d.root).modal);
    assert!(!layout(&world, d.root).visible);
    sync(&mut world, &d, Some(&dialog(false, None)));
    assert!(layout(&world, d.root).visible);
    sync(&mut world, &d, None);
    assert!(!layout(&world, d.root).visible);
}

// The message, the buttons in use (the destructive one red), and the field
// and checkbox rows only when the dialog carries them.
#[test]
fn sync_shows_the_message_buttons_and_optional_controls() {
    let mut world = World::new();
    let d = built(&mut world);
    sync(&mut world, &d, Some(&dialog(false, None)));
    assert_eq!(
        world.resource::<UiState>().unwrap().get(MESSAGE),
        Some(&UiValue::Text("Delete it?".into()))
    );
    let button =
        |world: &World, i: usize| world.get_by_id::<UiButton>(d.buttons[i]).unwrap().clone();
    assert_eq!(button(&world, 1).label, "Delete");
    assert_eq!(button(&world, 1).style.background, DANGER_TINT);
    assert!(!button(&world, 2).visible, "the third slot is unused");
    assert!(!layout(&world, d.field_row).visible);
    assert!(!layout(&world, d.check_row).visible);
    let plain = layout(&world, d.root).height;

    let check = Check {
        caption: "Delete its files".into(),
        on: true,
        enabled: false,
        note: Some("the files are shared".into()),
    };
    sync(&mut world, &d, Some(&dialog(true, Some(check))));
    assert!(layout(&world, d.field_row).visible);
    assert!(layout(&world, d.check_row).visible);
    assert_ne!(
        layout(&world, d.root).height,
        plain,
        "taller by its controls"
    );
    let field = world.get_by_id::<TextField>(d.field).unwrap();
    assert_eq!(field.content, "half");
    assert!(field.focused, "a prompt's field owns the keyboard");
    let check = world.get_by_id::<Checkbox>(d.check).unwrap();
    assert!(check.disabled);
    assert_eq!(check.note, "the files are shared");
    assert_eq!(typed(&world, &d), Some("half"));
}

#[test]
fn a_disabled_check_keeps_its_state() {
    let mut check = Check {
        caption: String::new(),
        on: false,
        enabled: false,
        note: None,
    };
    check.toggle();
    assert!(!check.on);
    check.enabled = true;
    check.toggle();
    assert!(check.on);
}

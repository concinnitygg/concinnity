use super::*;
use concinnity_engine::components::{UiState, UiValue};

fn built(world: &mut World) -> TopBar {
    build(&mut UiBuilder::new(world, AssetId(9), None)).unwrap()
}

fn state(dirty: bool, sim: SimState) -> HudState {
    HudState {
        dirty,
        undo: false,
        redo: true,
        view_open: true,
        display_open: false,
        sim,
        visible: true,
    }
}

fn button(world: &World, id: AssetId) -> &Button {
    world.get_by_id::<Button>(id).unwrap()
}

#[test]
fn each_chip_names_its_action() {
    for (name, expected) in [
        (SAVE, HudAction::Save),
        (UNDO, HudAction::Undo),
        (REDO, HudAction::Redo),
        (VIEW, HudAction::ToggleView),
        (DISPLAY, HudAction::ToggleDisplay),
        (PLAY, HudAction::PlayPause),
        (STEP, HudAction::Step),
        (STOP, HudAction::Stop),
    ] {
        let event = UiEvent {
            name: name.into(),
            element: None,
            value: None,
        };
        assert_eq!(action(&event), Some(expected));
    }
}

// A clean Save and an empty history take no clicks; unsaved edits light Save,
// a run arms Stop and turns Play into Pause, and an open panel presses View.
#[test]
fn sync_follows_the_edits_history_and_transport() {
    let mut world = World::new();
    let bar = built(&mut world);
    sync(&mut world, &bar, state(false, SimState::Stopped));
    assert!(button(&world, bar.save).disabled);
    assert!(button(&world, bar.undo).disabled);
    assert!(!button(&world, bar.redo).disabled);
    assert!(button(&world, bar.stop).disabled);

    sync(&mut world, &bar, state(true, SimState::Playing));
    let save = button(&world, bar.save);
    assert!(!save.disabled);
    assert_eq!(save.style.background, SAVE_TINT_ACTIVE);
    assert_eq!(button(&world, bar.stop).style.background, STOP_TINT_ARMED);
    let ui = world.resource::<UiState>().unwrap();
    assert_eq!(ui.get(PLAY_LABEL), Some(&UiValue::Text("Pause".into())));
    assert!(state::get_bool(&world, VIEW_OPEN));
    assert!(state::get_bool(&world, PLAYING));
}

// F1 hides the whole bar.
#[test]
fn a_hidden_hud_hides_the_bar() {
    let mut world = World::new();
    let bar = built(&mut world);
    let mut hidden = state(false, SimState::Stopped);
    hidden.visible = false;
    sync(&mut world, &bar, hidden);
    assert!(!world.get_by_id::<Layout>(bar.root).unwrap().visible);
}

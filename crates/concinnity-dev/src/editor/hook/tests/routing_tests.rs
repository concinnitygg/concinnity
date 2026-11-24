//! What a frame's input reaches (`hook/routing.rs`): the state a session opens
//! in, Escape handing the cursor back, F1 hiding the HUD, a drag that crosses a
//! control without triggering it, and the panel actions each apply path
//! consumes. What those actions then do to the world is asserted beside the
//! module that does it.

use concinnity_engine::components::FrameInput;

use super::fixtures::{entry, hook, world_with_fields, world_with_input};

use crate::frame_hook::FrameHook;

use crate::editor::panels::form::{self, FormField};
use crate::editor::panels::form_panel::FormAction;

use crate::editor::panels::assets_panel::PanelAction;
use crate::editor::panels::registry::PanelKey;

use crate::editor::sim;

#[test]
fn starts_in_edit_mode_with_hud_shown() {
    let h = hook(Vec::new());
    assert_eq!(
        h.sim.state,
        sim::SimState::Stopped,
        "editor holds the cursor at launch"
    );
    assert!(h.hud_visible, "HUD shown at launch");
    // Assets / View / Templates start closed; Preview starts shown.
    assert!(!h.open[PanelKey::Assets] && !h.open[PanelKey::View] && !h.open[PanelKey::Templates]);
    assert!(
        h.open[PanelKey::Preview],
        "the Preview panel is shown at launch"
    );
    assert!(!h.assets.picker_open);
}

#[test]
fn tick_escape_returns_cursor_to_editor() {
    let mut h = hook(Vec::new());
    h.sim.state = sim::SimState::Playing;
    let mut world = world_with_input(FrameInput {
        escape: true,
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    h.tick(&mut world);
    assert_eq!(
        h.sim.state,
        sim::SimState::Paused,
        "Escape pauses play mode"
    );
}

#[test]
fn tick_f1_toggles_hud_visibility() {
    let mut h = hook(Vec::new());
    let mut world = world_with_input(FrameInput {
        hud_toggle: true,
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    assert!(h.hud_visible);
    h.tick(&mut world);
    assert!(!h.hud_visible, "first F1 hides the HUD");
    h.tick(&mut world);
    assert!(h.hud_visible, "second F1 shows it again");
}

#[test]
fn apply_form_focus_toggle_and_consume() {
    let world = world_with_fields();
    let mut h = hook(Vec::new());
    h.form.fields = vec![FormField {
        key: "on".into(),
        kind: form::FieldKind::Bool,
        initial: String::new(),
        boolval: false,
        variants: Vec::new(),
        variant_idx: 0,
        choices: Vec::new(),
        sparse: false,
        unset: false,
    }];
    h.apply_form(FormAction::ToggleField(0), &world);
    assert!(h.form.fields[0].boolval, "the bool field flipped");
}

#[test]
fn apply_panel_toggles_the_picker() {
    let mut h = hook(Vec::new());
    h.apply_panel(PanelAction::TogglePicker);
    assert!(h.assets.picker_open);
    assert!(
        h.assets.search_focus,
        "the picker types into the search field"
    );
    h.apply_panel(PanelAction::TogglePicker);
    assert!(!h.assets.picker_open, "a second toggle closes the picker");
}

#[test]
fn apply_panel_pick_option_opens_the_add_form() {
    let mut h = hook(vec![entry("log", "Logger")]);

    h.assets.picker_open = true;
    h.apply_panel(PanelAction::PickOption(0));
    assert!(h.form_open(), "a picker pick opens the add form");
    assert!(!h.assets.picker_open, "and closes the picker behind it");

    // A pick with the picker already closed is a no-op: there is no option
    // list to index into.
    h.form.close();
    h.apply_panel(PanelAction::PickOption(0));
    assert!(!h.form_open());
}

#[test]
fn confirm_form_without_a_selected_type_just_closes() {
    let mut h = hook(Vec::new());
    h.form.selected_type = None;
    h.confirm_form();
    assert!(!h.form_open());
}

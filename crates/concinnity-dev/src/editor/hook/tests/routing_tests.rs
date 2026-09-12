// src/editor/hook/tests/routing_tests.rs
//
// What a frame's input reaches (`hook/routing.rs`): the state a session opens
// in, Escape handing the cursor back, F1 hiding the HUD, a drag that crosses a
// control without triggering it, and the panel actions each apply path
// consumes. What those actions then do to the world is asserted beside the
// module that does it.

use concinnity_core::components::FrameInput;

use super::fixtures::{entry, hook, set_input, world_with_fields, world_with_input};

use crate::debug_hook::DebugHook;

use crate::editor::panels::form::{self, FormField};
use crate::editor::panels::form_panel::{FormAction, FormFocus};

use crate::editor::panels::panel::PanelAction;
use crate::editor::panels::preview;
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
    assert!(!h.panel_open && !h.view_open && !h.templates_open);
    assert!(h.preview_open, "the Preview panel is shown at launch");
    assert!(!h.picker_open);
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

// While a drag is in progress the press's click must not also resolve to a
// control underneath on later frames -- e.g. dragging the Assets panel across
// the Preview checkbox must not toggle play mode.
#[test]
fn dragging_does_not_trigger_controls_it_crosses() {
    let mut h = hook(Vec::new());
    h.panel_open = true;
    let vp = [1280.0, 720.0];
    let start = h.origin(PanelKey::Assets, vp);
    let mut world = world_with_input(FrameInput {
        viewport: vp,
        mouse_x: start[0] + 10.0,
        mouse_y: start[1] + 10.0,
        left_click: true,
        left_button_down: true,
        ..Default::default()
    });
    h.tick(&mut world);
    // Cross the Preview panel's capture row with the button still held and a
    // stray click edge (e.g. from event coalescing).
    let pv = h.origin(PanelKey::Preview, vp);
    set_input(
        &mut world,
        FrameInput {
            viewport: vp,
            mouse_x: pv[0] + 10.0,
            mouse_y: pv[1] + preview::size()[1] - 5.0,
            left_click: true,
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(&mut world);
    assert!(!h.sim.playing(), "the drag swallowed the click");
    assert!(h.drag.is_some(), "still dragging");
}

#[test]
fn apply_form_focus_toggle_and_consume() {
    let mut world = world_with_fields();
    let mut h = hook(Vec::new());
    h.form_fields = vec![FormField {
        key: "on".into(),
        kind: form::FieldKind::Bool,
        initial: String::new(),
        boolval: false,
        variants: Vec::new(),
        variant_idx: 0,
    }];
    h.apply_form(FormAction::FocusField(0), &mut world);
    assert!(matches!(h.form_focus, FormFocus::Field(0)));
    h.apply_form(FormAction::FocusName, &mut world);
    assert!(matches!(h.form_focus, FormFocus::Name));
    h.apply_form(FormAction::ToggleField(0), &mut world);
    assert!(h.form_fields[0].boolval, "the bool field flipped");
    // A click that hits no control is swallowed without side effects.
    h.apply_form(FormAction::Consume, &mut world);
    assert!(h.form_fields[0].boolval);
}

#[test]
fn apply_panel_toggles_the_picker_and_consumes() {
    let mut world = world_with_fields();
    let mut h = hook(Vec::new());
    h.apply_panel(PanelAction::TogglePicker, &mut world);
    assert!(h.picker_open);
    assert!(h.search_focus, "the picker types into the search field");
    h.apply_panel(PanelAction::TogglePicker, &mut world);
    assert!(!h.picker_open, "a second toggle closes the picker");
    h.apply_panel(PanelAction::Consume, &mut world);
}

#[test]
fn apply_panel_pick_option_opens_the_add_form() {
    let mut world = world_with_fields();
    let mut h = hook(vec![entry("log", "Logger")]);

    h.picker_open = true;
    h.apply_panel(PanelAction::PickOption(0), &mut world);
    assert!(h.form_open(), "a picker pick opens the add form");
    assert!(!h.picker_open, "and closes the picker behind it");

    // A pick with the picker already closed is a no-op: there is no option
    // list to index into.
    h.close_form();
    h.apply_panel(PanelAction::PickOption(0), &mut world);
    assert!(!h.form_open());
}

#[test]
fn confirm_form_without_a_selected_type_just_closes() {
    let mut world = world_with_fields();
    let mut h = hook(Vec::new());
    h.selected_type = None;
    h.confirm_form(&mut world);
    assert!(!h.form_open());
}

//! The editor "Preview" panel: a small floating panel holding the controls that
//! affect how the running world is previewed: the play checkbox (the simulation
//! transport's Play / Pause, mirroring the top-bar chip), the fly-camera
//! checkbox (navigate the frozen world; also the F key), the world-axes
//! checkbox (the origin axis lines drawn in the viewport), the two snap rows
//! (grid / angle snapping for gizmo drags; the checkbox toggles, the step
//! button beside it cycles the step), the align-to-surface checkbox, and the
//! drop-to-floor button. Escape leaves either camera mode. Each control shows
//! UI state the hook keeps in step with its model and sends the event the hook
//! acts on.

use concinnity_engine::components::{Binding, Button, Checkbox, LayoutChild, UiEvent};
use concinnity_engine::ecs::World;

use super::registry::PanelKey;
use crate::editor::ui::build::{BODY_PAD, emit};
use crate::editor::ui::{BuiltPanel, PanelSpec, ROW_H, UiBuilder, panel_height, state};
use crate::editor::viewport::snap::SnapSettings;

const PREVIEW_W: f32 = 232.0;
const ROWS: usize = 7;
// The snap rows' step buttons.
const STEP_W: f32 = 64.0;

// The state each checkbox shows, which is also the event it sends.
const PLAY: &str = "editor.preview.play";
const FLY: &str = "editor.preview.fly";
const AXES: &str = "editor.preview.axes";
const SNAP_MOVE: &str = "editor.preview.snap_move";
const SNAP_ROTATE: &str = "editor.preview.snap_rotate";
const ALIGN: &str = "editor.preview.align";
// The text each step button shows, which is also the event it sends.
const SNAP_MOVE_STEP: &str = "editor.preview.snap_move_step";
const SNAP_ROTATE_STEP: &str = "editor.preview.snap_rotate_step";
const DROP_FLOOR: &str = "editor.preview.drop_floor";

// Where the panel sits until the user drags it: the window's top-left, below
// the top bar (clear of its buttons and the Assets panel's default anchor).
pub(crate) fn default_origin() -> [f32; 2] {
    [8.0, crate::editor::hud::body_top()]
}

// The panel's fixed footprint, for the hook's drag clamp.
pub(crate) fn size() -> [f32; 2] {
    [PREVIEW_W, panel_height(ROW_H * ROWS as f32)]
}

// What the user asked of the panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreviewAction {
    // The play checkbox: run / pause the world (and the cursor follows).
    TogglePlay,
    // The fly checkbox: toggle the edit-mode fly camera.
    ToggleFly,
    // The world-axes checkbox: show / hide the origin axis lines.
    ToggleAxes,
    // The snap rows: the checkbox toggles the family, the step button cycles
    // the step through the presets.
    ToggleSnapMove,
    CycleSnapMoveStep,
    ToggleSnapRotate,
    CycleSnapRotateStep,
    // The align row: orient drag-out drops to the struck surface's normal.
    ToggleAlign,
    // The action row: drop the selection onto the surface below it.
    DropToFloor,
}

// The action a Preview event asks for, if `event` is one.
pub(crate) fn action(event: &UiEvent) -> Option<PreviewAction> {
    Some(match event.name.as_str() {
        PLAY => PreviewAction::TogglePlay,
        FLY => PreviewAction::ToggleFly,
        AXES => PreviewAction::ToggleAxes,
        SNAP_MOVE => PreviewAction::ToggleSnapMove,
        SNAP_MOVE_STEP => PreviewAction::CycleSnapMoveStep,
        SNAP_ROTATE => PreviewAction::ToggleSnapRotate,
        SNAP_ROTATE_STEP => PreviewAction::CycleSnapRotateStep,
        ALIGN => PreviewAction::ToggleAlign,
        DROP_FLOOR => PreviewAction::DropToFloor,
        _ => return None,
    })
}

// The panel's row states, in row order.
#[derive(Debug, Clone, Copy, PartialEq)]
#[expect(clippy::struct_excessive_bools, reason = "independent preview toggles")]
pub(crate) struct PreviewState {
    pub playing: bool,
    pub fly: bool,
    pub axes: bool,
    pub snap: SnapSettings,
    pub align: bool,
}

fn check(caption: &str, name: &str) -> Checkbox {
    Checkbox {
        label: caption.to_string(),
        checked: Some(Binding::State(name.to_string())),
        on_change: name.to_string(),
        ..Default::default()
    }
}

// A snap row: its checkbox, and the button that cycles its step.
fn snap_row(
    ui: &mut UiBuilder<'_>,
    caption: &str,
    toggle: &str,
    step: &str,
) -> Option<LayoutChild> {
    let check = ui.checkbox(check(caption, toggle), ROW_H)?;
    let button = Button {
        text_binding: Some(Binding::State(step.to_string())),
        action: Some(emit(step, None)),
        ..Default::default()
    };
    let step = ui.button_with(button, [STEP_W, ROW_H - 6.0])?;
    ui.row(vec![check, step], ROW_H)
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let mut rows = vec![
        ui.checkbox(check("Play (Ctrl+P)", PLAY), ROW_H)?,
        ui.checkbox(check("Fly camera (F)", FLY), ROW_H)?,
        ui.checkbox(check("World axes", AXES), ROW_H)?,
    ];
    rows.push(snap_row(ui, "Snap move", SNAP_MOVE, SNAP_MOVE_STEP)?);
    rows.push(snap_row(ui, "Snap rotate", SNAP_ROTATE, SNAP_ROTATE_STEP)?);
    rows.push(ui.checkbox(check("Align drop to surface", ALIGN), ROW_H)?);
    let drop = ui.button(
        "Drop to floor (Ctrl+Down)",
        DROP_FLOOR,
        None,
        [PREVIEW_W - 2.0 * BODY_PAD, ROW_H - 4.0],
    )?;
    rows.push(drop);
    let spec = PanelSpec {
        key: PanelKey::Preview,
        title: "Preview",
        title_state: None,
        size: size(),
        min_size: None,
    };
    ui.panel(&spec, rows)
}

// Show the panel's state.
pub(crate) fn sync(s: PreviewState, world: &mut World) {
    state::set_bool(world, PLAY, s.playing);
    state::set_bool(world, FLY, s.fly);
    state::set_bool(world, AXES, s.axes);
    state::set_bool(world, SNAP_MOVE, s.snap.translate.enabled);
    state::set_bool(world, SNAP_ROTATE, s.snap.rotate.enabled);
    state::set_bool(world, ALIGN, s.align);
    let (mv, rot) = (s.snap.translate.step, s.snap.rotate.step);
    state::set_text(world, SNAP_MOVE_STEP, format_args!("{mv} m"));
    state::set_text(world, SNAP_ROTATE_STEP, format_args!("{rot} deg"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::{UiState, UiValue};
    use concinnity_engine::ecs::asset_id::AssetId;

    fn event(name: &str) -> UiEvent {
        UiEvent {
            name: name.into(),
            element: None,
            value: None,
        }
    }

    #[test]
    fn every_control_names_its_action() {
        assert_eq!(action(&event(PLAY)), Some(PreviewAction::TogglePlay));
        assert_eq!(
            action(&event(SNAP_MOVE_STEP)),
            Some(PreviewAction::CycleSnapMoveStep)
        );
        assert_eq!(action(&event(DROP_FLOOR)), Some(PreviewAction::DropToFloor));
        assert_eq!(action(&event("editor.view.0")), None);
    }

    #[test]
    fn the_panel_holds_six_checkboxes_and_three_buttons_besides_close() {
        let mut world = World::new();
        let mut ui = UiBuilder::new(&mut world, AssetId(9), None);
        build(&mut ui).unwrap();
        assert_eq!(world.query::<Checkbox>().count(), 6);
        assert_eq!(world.query::<Button>().count(), 4);
    }

    #[test]
    fn sync_shows_the_steps_with_their_units() {
        let mut world = World::new();
        let s = PreviewState {
            playing: true,
            fly: false,
            axes: true,
            snap: SnapSettings::default(),
            align: false,
        };
        sync(s, &mut world);
        let state = world.resource::<UiState>().unwrap();
        assert_eq!(state.get(PLAY), Some(&UiValue::Bool(true)));
        let step = s.snap.translate.step;
        assert_eq!(
            state.get(SNAP_MOVE_STEP),
            Some(&UiValue::Text(format!("{step} m")))
        );
    }
}

// src/editor/preview.rs
//
// The editor "Preview" panel: a small floating panel holding the controls that
// affect how the running world is previewed: the play checkbox (the simulation
// transport's Play / Pause, mirroring the top-bar chip), the fly-camera
// checkbox (navigate the frozen world; also the F key), the world-axes
// checkbox (the origin axis lines drawn in the viewport), and the two snap
// rows (grid / angle snapping for gizmo drags; the row toggles, its value
// strip cycles the step). Escape leaves either camera mode. Like the rest of
// the editor HUD it is plain `Sprite` / `TextLabel` components at reserved ids
// (injected by `inject.rs`), driven each frame by the editor hook. The title
// bar, close button, and row draw come from the shared `list_panel`; this
// module only names the ids, width, and its row actions.

use super::list_panel::{self, Row};
use super::registry::{self, PanelKey};
use super::snap::SnapSettings;
use super::widget::{self, point_in};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::Preview);
// Named ids the cross-module tests reference (injection ordering / visibility);
// the shipping paths derive every id from `BASE` through `list_panel`.
#[cfg(test)]
pub(crate) const PANEL_BG: AssetId = list_panel::panel_bg(BASE);
#[cfg(test)]
pub(crate) const ROW_BG: AssetId = list_panel::row_bg(BASE, 0);
#[cfg(test)]
pub(crate) const CHECK_BOX: AssetId = list_panel::check_box(BASE, 0);

const PREVIEW_W: f32 = 232.0;
// The capture row, the fly row, the world-axes row, the two snap rows, the
// align-to-surface row, and the drop-to-floor action row.
const ROWS: usize = 7;
// The snap rows' indices: their clicks split into toggle vs step-cycle.
const SNAP_MOVE_ROW: usize = 3;
const SNAP_ROTATE_ROW: usize = 4;
const ALIGN_ROW: usize = 5;
const DROP_FLOOR_ROW: usize = 6;

// Where the panel sits until the user drags it: the window's top-left, below
// the top bar (clear of its buttons and the Assets panel's default anchor).
pub(crate) fn default_origin() -> [f32; 2] {
    [8.0, super::hud::body_top()]
}

// The panel's fixed footprint, for the hook's drag clamp.
pub(crate) fn size() -> [f32; 2] {
    list_panel::size(PREVIEW_W, ROWS)
}

// The panel outer rect (title bar + the capture row).
pub(crate) fn panel_rect(o: [f32; 2]) -> [f32; 4] {
    list_panel::panel_rect(o, PREVIEW_W, ROWS)
}

// A resolved Preview-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreviewAction {
    // The play checkbox: run / pause the world (and the cursor follows).
    TogglePlay,
    // The fly checkbox: toggle the edit-mode fly camera.
    ToggleFly,
    // The world-axes checkbox: show / hide the origin axis lines.
    ToggleAxes,
    // The snap rows: the row toggles the family, its value strip cycles the
    // step through the presets.
    ToggleSnapMove,
    CycleSnapMoveStep,
    ToggleSnapRotate,
    CycleSnapRotateStep,
    // The align row: orient drag-out drops to the struck surface's normal.
    ToggleAlign,
    // The action row: drop the selection onto the surface below it.
    DropToFloor,
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Resolve a click at `(mx, my)` against the panel at origin `o`. `None` means the
// click missed the panel. Title-bar presses never reach this: the hook intercepts
// them first to start a drag (the shared routing owns the title-bar geometry).
pub(crate) fn hit_test(mx: f32, my: f32, o: [f32; 2]) -> Option<PreviewAction> {
    let on_value = |i| point_in(mx, my, list_panel::value_rect(o, PREVIEW_W, i));
    match list_panel::hit_row(mx, my, o, PREVIEW_W, ROWS) {
        Some(0) => return Some(PreviewAction::TogglePlay),
        Some(1) => return Some(PreviewAction::ToggleFly),
        Some(2) => return Some(PreviewAction::ToggleAxes),
        Some(SNAP_MOVE_ROW) => {
            return Some(if on_value(SNAP_MOVE_ROW) {
                PreviewAction::CycleSnapMoveStep
            } else {
                PreviewAction::ToggleSnapMove
            });
        }
        Some(SNAP_ROTATE_ROW) => {
            return Some(if on_value(SNAP_ROTATE_ROW) {
                PreviewAction::CycleSnapRotateStep
            } else {
                PreviewAction::ToggleSnapRotate
            });
        }
        Some(ALIGN_ROW) => return Some(PreviewAction::ToggleAlign),
        Some(DROP_FLOOR_ROW) => return Some(PreviewAction::DropToFloor),
        Some(_) | None => {}
    }
    point_in(mx, my, panel_rect(o)).then_some(PreviewAction::Consume)
}

// The panel's row states, in row order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PreviewState {
    pub playing: bool,
    pub fly: bool,
    pub axes: bool,
    pub snap: SnapSettings,
    pub align: bool,
}

// Position + show the panel at origin `o`, colouring each checkbox by state.
pub(crate) fn apply(world: &mut World, o: [f32; 2], state: PreviewState, mouse: [f32; 2]) {
    let rows = [
        Row::checkbox("Play (Ctrl+P)", state.playing),
        Row::checkbox("Fly camera (F)", state.fly),
        Row::checkbox("World axes", state.axes),
        Row::checkbox("Snap move", state.snap.translate.enabled)
            .with_value(format!("{} m", state.snap.translate.step)),
        Row::checkbox("Snap rotate", state.snap.rotate.enabled)
            .with_value(format!("{} deg", state.snap.rotate.step)),
        Row::checkbox("Align drop to surface", state.align),
        Row::label("Drop to floor (Ctrl+Down)"),
    ];
    list_panel::apply(world, BASE, o, size(), "Preview", &rows, mouse);
}

// Hide every panel element (the F1-hidden pass).
pub(crate) fn hide_all(world: &mut World) {
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &[]);
}

// Every panel sprite / label id, for injection and the hidden pass.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    list_panel::all_sprite_ids(BASE, ROWS, true)
}
pub(crate) fn all_label_ids() -> Vec<AssetId> {
    list_panel::all_label_ids(BASE, ROWS, true)
}

#[cfg(test)]
mod tests {
    use super::super::widget;
    use super::list_panel::title_label;
    use super::*;
    use crate::components::{Sprite, TextLabel};

    fn injected_world() -> World {
        let mut world = World::new();
        for id in all_sprite_ids() {
            world.add_component(Sprite {
                asset_id: id,
                ..Default::default()
            });
        }
        for id in all_label_ids() {
            world.add_component(TextLabel {
                asset_id: id,
                ..Default::default()
            });
        }
        world
    }

    #[test]
    fn hit_test_resolves_the_play_row_and_swallows_the_rest() {
        let o = default_origin();
        let row = list_panel::row_rect(o, PREVIEW_W, 0);
        assert_eq!(
            hit_test(row[0] + 10.0, row[1] + 10.0, o),
            Some(PreviewAction::TogglePlay)
        );
        let fly_row = list_panel::row_rect(o, PREVIEW_W, 1);
        assert_eq!(
            hit_test(fly_row[0] + 10.0, fly_row[1] + 10.0, o),
            Some(PreviewAction::ToggleFly)
        );
        let axes_row = list_panel::row_rect(o, PREVIEW_W, 2);
        assert_eq!(
            hit_test(axes_row[0] + 10.0, axes_row[1] + 10.0, o),
            Some(PreviewAction::ToggleAxes)
        );
        let t = widget::title_rect(o, PREVIEW_W);
        assert_eq!(
            hit_test(t[0] + 5.0, t[1] + 5.0, o),
            Some(PreviewAction::Consume),
            "a title click that was not a drag is swallowed"
        );
        assert_eq!(hit_test(900.0, 500.0, o), None, "a miss falls through");
    }

    #[test]
    fn snap_rows_split_into_toggle_and_step_cycle() {
        let o = default_origin();
        let row = list_panel::row_rect(o, PREVIEW_W, SNAP_MOVE_ROW);
        assert_eq!(
            hit_test(row[0] + 10.0, row[1] + 10.0, o),
            Some(PreviewAction::ToggleSnapMove),
            "the row body toggles"
        );
        let v = list_panel::value_rect(o, PREVIEW_W, SNAP_MOVE_ROW);
        assert_eq!(
            hit_test(v[0] + 5.0, v[1] + 10.0, o),
            Some(PreviewAction::CycleSnapMoveStep),
            "the value strip cycles the step"
        );
        let rot = list_panel::row_rect(o, PREVIEW_W, SNAP_ROTATE_ROW);
        assert_eq!(
            hit_test(rot[0] + 10.0, rot[1] + 10.0, o),
            Some(PreviewAction::ToggleSnapRotate)
        );
        let rv = list_panel::value_rect(o, PREVIEW_W, SNAP_ROTATE_ROW);
        assert_eq!(
            hit_test(rv[0] + 5.0, rv[1] + 10.0, o),
            Some(PreviewAction::CycleSnapRotateStep)
        );
    }

    #[test]
    fn apply_shows_the_snap_steps_in_the_value_strips() {
        let mut world = injected_world();
        apply(
            &mut world,
            default_origin(),
            PreviewState {
                playing: false,
                fly: false,
                axes: false,
                snap: SnapSettings::default(),
                align: false,
            },
            [0.0, 0.0],
        );
        let value = |i| {
            world
                .query::<TextLabel>()
                .find(|l| l.asset_id == list_panel::value_label(BASE, i))
                .cloned()
                .unwrap()
        };
        let mv = value(SNAP_MOVE_ROW);
        assert!(mv.visible && mv.content == "0.5 m", "{}", mv.content);
        let rot = value(SNAP_ROTATE_ROW);
        assert!(rot.visible && rot.content == "15 deg", "{}", rot.content);
        let plain = value(0);
        assert!(!plain.visible, "a value-less row hides its value label");
    }

    #[test]
    fn apply_shows_heading_and_play_state() {
        let mut world = injected_world();
        let o = default_origin();
        let off_state = PreviewState {
            playing: false,
            fly: false,
            axes: false,
            snap: SnapSettings::default(),
            align: false,
        };
        apply(&mut world, o, off_state, [0.0, 0.0]);
        let title = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == title_label(BASE))
            .unwrap();
        assert!(title.visible && title.content == "Preview");
        let off = world
            .query::<Sprite>()
            .find(|s| s.asset_id == CHECK_BOX)
            .cloned()
            .unwrap();
        apply(
            &mut world,
            o,
            PreviewState {
                playing: true,
                ..off_state
            },
            [0.0, 0.0],
        );
        let on = world
            .query::<Sprite>()
            .find(|s| s.asset_id == CHECK_BOX)
            .unwrap();
        assert_ne!(off.tint, on.tint, "the checkbox greens while playing");
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        apply(
            &mut world,
            default_origin(),
            PreviewState {
                playing: true,
                fly: true,
                axes: true,
                snap: SnapSettings::default(),
                align: false,
            },
            [0.0, 0.0],
        );
        hide_all(&mut world);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}

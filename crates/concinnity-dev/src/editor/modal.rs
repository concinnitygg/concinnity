// src/editor/modal.rs
//
// The confirmation dialog: a centered panel with a wrapped message, an optional
// name field, and a row of two or three buttons over a translucent dim covering
// the whole screen.
// Pure geometry and draw; the open / press / close flow lives in
// `hook/modal_drive.rs`. Not a registered panel: the dialog has no title bar,
// drag, focus rank, or View toggle, and while open it is screen-modal -- the
// hook routes every press and wheel to it before anything else, and its draw
// layer sits above all other chrome.

use super::registry::ID_BASE;
use super::widget::{self, point_in};
use super::{hud, theme};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Reserved id family: the next free block after the toast stack's (0xA000).
const BASE: u32 = ID_BASE + 0xB000;

const DIM: AssetId = AssetId(BASE);
const PANEL_BG: AssetId = AssetId(BASE + 1);
const MESSAGE: AssetId = AssetId(BASE + 2);
pub(crate) const NAME_INPUT: AssetId = AssetId(BASE + 3);

pub(crate) const MAX_BUTTONS: usize = 3;
const fn button_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x10 + i as u32)
}
const fn button_label(i: usize) -> AssetId {
    AssetId(BASE + 0x20 + i as u32)
}

const PANEL_W: f32 = 380.0;
const PAD: f32 = 14.0;
// The message box's height in wrapped lines; longer text ends in an ellipsis.
const MSG_LINES: u32 = 3;
const BTN_W: f32 = 100.0;
const BTN_H: f32 = 28.0;
const BTN_GAP: f32 = 10.0;
// The name field a prompt dialog carries under its message.
const FIELD_H: f32 = 28.0;

// The screen-wide dim behind the dialog, so the chrome under it reads inactive.
const DIM_TINT: [f32; 4] = [0.02, 0.02, 0.03, 0.55];
// The destructive-action red, matching the editor's other confirm chips.
const DANGER_TINT: [f32; 4] = [0.68, 0.26, 0.28, 1.0];
const DANGER_HOVER_TINT: [f32; 4] = [0.80, 0.32, 0.34, 1.0];

// One dialog button. `danger` draws it in the destructive red so the intent
// is unmistakable; `action` is what pressing it does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Button {
    pub(crate) label: String,
    pub(crate) danger: bool,
    pub(crate) action: Action,
}

// What pressing a dialog button does; the hook runs it as the dialog closes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Action {
    // Close without acting (a Cancel / No button).
    Dismiss,
    // A Worlds-panel decision the dialog was guarding.
    Worlds(super::worlds::WorldsConfirm),
    // Name the untitled world the session is on, with whatever the dialog's
    // field holds, and save it there.
    NameWorld,
}

// The dialog's footprint. A prompt is taller by the field it carries.
pub(crate) fn size(field: bool) -> [f32; 2] {
    let field_h = match field {
        true => FIELD_H + PAD,
        false => 0.0,
    };
    [
        PANEL_W,
        PAD + MSG_LINES as f32 * widget::LINE_H + field_h + PAD + BTN_H + PAD,
    ]
}

// The dialog's rect: centered in the viewport, clamped fully on screen below
// the top bar for tiny windows.
pub(crate) fn panel_rect(vp: [f32; 2], field: bool) -> [f32; 4] {
    let s = size(field);
    let centered = [(vp[0] - s[0]) * 0.5, (vp[1] - s[1]) * 0.5];
    widget::outer_rect(widget::clamp_origin(centered, s, vp, hud::BAR_H), s)
}

fn message_rect(p: [f32; 4]) -> [f32; 4] {
    [
        p[0] + PAD,
        p[1] + PAD,
        p[2] - 2.0 * PAD,
        MSG_LINES as f32 * widget::LINE_H,
    ]
}

// The prompt's name field, between the message and the buttons.
fn field_rect(p: [f32; 4]) -> [f32; 4] {
    let m = message_rect(p);
    [m[0], m[1] + m[3] + PAD, m[2], FIELD_H]
}

// Button `i` of `count`, in caller order left to right along the dialog's
// bottom edge, the last flush with the right padding.
pub(crate) fn button_rect(p: [f32; 4], count: usize, i: usize) -> [f32; 4] {
    let right = p[0] + p[2] - PAD;
    let x = right - (count - i) as f32 * BTN_W - (count - 1 - i) as f32 * BTN_GAP;
    [x, p[1] + p[3] - PAD - BTN_H, BTN_W, BTN_H]
}

// The button under `(mx, my)`, or `None` for a press anywhere else -- which
// the dialog swallows without acting (a click-away is not a cancel).
pub(crate) fn hit_button(
    mx: f32,
    my: f32,
    vp: [f32; 2],
    count: usize,
    field: bool,
) -> Option<usize> {
    let p = panel_rect(vp, field);
    (0..count.min(MAX_BUTTONS)).find(|&i| point_in(mx, my, button_rect(p, count, i)))
}

fn button_tint(danger: bool, hovered: bool) -> [f32; 4] {
    match (danger, hovered) {
        (true, true) => DANGER_HOVER_TINT,
        (true, false) => DANGER_TINT,
        (false, true) => theme::HOVER_TINT,
        (false, false) => theme::BUTTON_TINT,
    }
}

pub(crate) fn apply(
    world: &mut World,
    vp: [f32; 2],
    message: &str,
    buttons: &[Button],
    field: bool,
    mouse: [f32; 2],
) {
    widget::place_sprite(world, DIM, [0.0, 0.0, vp[0], vp[1]], DIM_TINT, true);
    let p = panel_rect(vp, field);
    widget::place_panel(world, PANEL_BG, p);
    widget::place_message(world, MESSAGE, message_rect(p), message, theme::LABEL, true);
    // A prompt's field always owns the keyboard: the dialog is screen-modal, so
    // nothing else can be typing into anything.
    match field {
        true => widget::show_field(world, NAME_INPUT, field_rect(p), true),
        false => widget::hide_field(world, NAME_INPUT),
    }
    for slot in 0..MAX_BUTTONS {
        match buttons.get(slot) {
            Some(b) => {
                let r = button_rect(p, buttons.len(), slot);
                let hovered = point_in(mouse[0], mouse[1], r);
                widget::place_rounded(
                    world,
                    button_bg(slot),
                    r,
                    button_tint(b.danger, hovered),
                    theme::CONTROL_RADIUS,
                    true,
                );
                widget::place_center_label(
                    world,
                    button_label(slot),
                    [r[0] + r[2] * 0.5, r[1] + r[3] * 0.5 - theme::TEXT_HALF],
                    &widget::clip_text(&b.label, 12),
                    theme::LABEL,
                    true,
                );
            }
            None => {
                widget::set_sprite_visible(world, button_bg(slot), false);
                widget::set_label_visible(world, button_label(slot), false);
            }
        }
    }
}

pub(crate) fn hide(world: &mut World) {
    widget::set_sprite_visible(world, DIM, false);
    widget::set_sprite_visible(world, PANEL_BG, false);
    widget::set_label_visible(world, MESSAGE, false);
    widget::hide_field(world, NAME_INPUT);
    for slot in 0..MAX_BUTTONS {
        widget::set_sprite_visible(world, button_bg(slot), false);
        widget::set_label_visible(world, button_label(slot), false);
    }
}

pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    [DIM, PANEL_BG]
        .into_iter()
        .chain((0..MAX_BUTTONS).map(button_bg))
        .collect()
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    std::iter::once(MESSAGE)
        .chain((0..MAX_BUTTONS).map(button_label))
        .collect()
}

pub(crate) fn all_field_ids() -> Vec<AssetId> {
    vec![NAME_INPUT]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Sprite, TextAlign, TextLabel};

    const VP: [f32; 2] = [1280.0, 720.0];

    fn world_with_elements() -> World {
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
        for id in all_field_ids() {
            world.add_component(crate::components::TextInput {
                asset_id: id,
                ..Default::default()
            });
        }
        world
    }

    fn plain(label: &str) -> Button {
        Button {
            label: label.to_string(),
            danger: false,
            action: Action::Dismiss,
        }
    }

    fn danger(label: &str) -> Button {
        Button {
            label: label.to_string(),
            danger: true,
            action: Action::Dismiss,
        }
    }

    fn sprite(world: &World, id: AssetId) -> Sprite {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == id)
            .cloned()
            .unwrap()
    }

    fn label(world: &World, id: AssetId) -> TextLabel {
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .cloned()
            .unwrap()
    }

    #[test]
    fn panel_centers_and_clamps_below_the_top_bar() {
        let p = panel_rect(VP, false);
        let s = size(false);
        assert_eq!(p[0], (VP[0] - s[0]) * 0.5);
        assert_eq!(p[1], (VP[1] - s[1]) * 0.5);
        assert_eq!((p[2], p[3]), (s[0], s[1]));
        // A viewport too short to center in pins the dialog below the bar.
        let tiny = panel_rect([500.0, 120.0], false);
        assert!(tiny[1] >= hud::BAR_H);
        assert!(tiny[0] >= 0.0);
    }

    #[test]
    fn buttons_lay_out_left_to_right_inside_the_dialog() {
        let p = panel_rect(VP, false);
        for count in [2, 3] {
            let rects: Vec<[f32; 4]> = (0..count).map(|i| button_rect(p, count, i)).collect();
            for w in rects.windows(2) {
                assert!(w[0][0] + w[0][2] < w[1][0], "no overlap, caller order");
            }
            let last = rects[count - 1];
            assert_eq!(
                last[0] + last[2],
                p[0] + p[2] - PAD,
                "the last button is flush with the right padding"
            );
            for r in &rects {
                assert!(
                    r[0] >= p[0] && r[0] + r[2] <= p[0] + p[2],
                    "inside the dialog"
                );
                assert!(r[1] + r[3] <= p[1] + p[3], "inside the dialog");
                let m = message_rect(p);
                assert!(r[1] >= m[1] + m[3], "below the message area");
            }
        }
    }

    #[test]
    fn hit_button_resolves_each_button_and_misses_the_rest() {
        let p = panel_rect(VP, false);
        for count in [2, 3] {
            for i in 0..count {
                let r = button_rect(p, count, i);
                assert_eq!(
                    hit_button(r[0] + 2.0, r[1] + 2.0, VP, count, false),
                    Some(i)
                );
            }
            // The message area, the dimmed screen, and the gap between two
            // buttons all miss.
            assert_eq!(hit_button(p[0] + 2.0, p[1] + 2.0, VP, count, false), None);
            assert_eq!(hit_button(5.0, 5.0, VP, count, false), None);
            let r1 = button_rect(p, count, 1);
            assert_eq!(hit_button(r1[0] - 1.0, r1[1] + 2.0, VP, count, false), None);
        }
    }

    #[test]
    fn apply_dims_the_whole_screen_behind_the_dialog() {
        let mut world = world_with_elements();
        apply(
            &mut world,
            VP,
            "Delete?",
            &[plain("Cancel"), danger("Delete")],
            false,
            [0.0, 0.0],
        );
        let dim = sprite(&world, DIM);
        assert!(dim.visible);
        assert_eq!(
            (dim.x, dim.y, dim.width, dim.height),
            (0.0, 0.0, VP[0], VP[1])
        );
        assert!(dim.tint[3] < 1.0, "translucent, not a blackout");
        assert!(sprite(&world, PANEL_BG).visible);
    }

    #[test]
    fn danger_styling_marks_the_flagged_button_only() {
        let mut world = world_with_elements();
        apply(
            &mut world,
            VP,
            "Discard changes?",
            &[plain("Cancel"), danger("Discard")],
            false,
            [0.0, 0.0],
        );
        assert_eq!(sprite(&world, button_bg(0)).tint, theme::BUTTON_TINT);
        assert_eq!(sprite(&world, button_bg(1)).tint, DANGER_TINT);
        let l = label(&world, button_label(1));
        assert!(l.visible && l.content == "Discard");
        assert_eq!(l.align, TextAlign::Center);
    }

    #[test]
    fn hover_highlights_the_button_under_the_cursor() {
        let p = panel_rect(VP, false);
        let buttons = [plain("Cancel"), danger("Discard")];
        let r0 = button_rect(p, 2, 0);
        let r1 = button_rect(p, 2, 1);

        let mut world = world_with_elements();
        apply(
            &mut world,
            VP,
            "m",
            &buttons,
            false,
            [r0[0] + 2.0, r0[1] + 2.0],
        );
        assert_eq!(sprite(&world, button_bg(0)).tint, theme::HOVER_TINT);
        assert_eq!(sprite(&world, button_bg(1)).tint, DANGER_TINT);

        apply(
            &mut world,
            VP,
            "m",
            &buttons,
            false,
            [r1[0] + 2.0, r1[1] + 2.0],
        );
        assert_eq!(sprite(&world, button_bg(0)).tint, theme::BUTTON_TINT);
        assert_eq!(sprite(&world, button_bg(1)).tint, DANGER_HOVER_TINT);
    }

    #[test]
    fn a_two_button_dialog_hides_the_third_slot_and_bounds_the_message() {
        let mut world = world_with_elements();
        apply(
            &mut world,
            VP,
            "a long message that must wrap inside the dialog",
            &[plain("No"), plain("Yes")],
            false,
            [0.0, 0.0],
        );
        assert!(!sprite(&world, button_bg(2)).visible);
        assert!(!label(&world, button_label(2)).visible);
        let m = label(&world, MESSAGE);
        assert!(m.visible);
        assert_eq!(m.max_lines, MSG_LINES);
        assert_eq!(m.wrap_width, PANEL_W - 2.0 * PAD);
    }

    #[test]
    fn hide_blanks_every_element() {
        let mut world = world_with_elements();
        apply(
            &mut world,
            VP,
            "m",
            &[plain("No"), plain("Yes")],
            false,
            [0.0, 0.0],
        );
        hide(&mut world);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
        assert!(
            world
                .query::<crate::components::TextInput>()
                .all(|t| !t.visible && !t.focused)
        );
    }

    // A prompt carries a focused name field between its message and its
    // buttons, and grows to make room for it; a plain dialog carries none.
    #[test]
    fn a_prompt_shows_a_focused_name_field_and_a_plain_dialog_does_not() {
        let mut world = world_with_elements();
        apply(
            &mut world,
            VP,
            "Name this world",
            &[plain("Save")],
            true,
            [0.0, 0.0],
        );
        let field = world
            .query::<crate::components::TextInput>()
            .find(|t| t.asset_id == NAME_INPUT)
            .cloned()
            .unwrap();
        assert!(field.visible && field.focused);
        let p = panel_rect(VP, true);
        let m = label(&world, MESSAGE);
        assert!(field.y > m.y, "the field sits under the message");
        let button = button_rect(p, 1, 0);
        assert!(field.y + field.height <= button[1], "and above the buttons");
        assert!(size(true)[1] > size(false)[1], "the prompt is taller");

        apply(&mut world, VP, "Delete?", &[plain("No")], false, [0.0, 0.0]);
        assert!(
            world
                .query::<crate::components::TextInput>()
                .all(|t| !t.visible && !t.focused)
        );
    }

    #[test]
    fn id_lists_cover_every_slot_without_repeats() {
        let sprites = all_sprite_ids();
        let labels = all_label_ids();
        let mut all: Vec<AssetId> = sprites.iter().chain(labels.iter()).copied().collect();
        let n = all.len();
        all.sort_by_key(|id| id.0);
        all.dedup();
        assert_eq!(all.len(), n, "no duplicate reserved ids");
        assert_eq!(sprites.len(), 2 + MAX_BUTTONS);
        assert_eq!(labels.len(), 1 + MAX_BUTTONS);
    }
}

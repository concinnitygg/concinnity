//! A checkbox for the editor HUD: a box tinted by its state and a caption
//! beside it, with an optional dim note underneath. A disabled checkbox keeps
//! its state, draws dimmed, and its note says why. Plain `Sprite` /
//! `TextLabel` components at reserved ids, placed each frame by the owner,
//! which also routes the press.

use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;

use super::theme;
use super::widget::{self, place_rounded, point_in};

const BOX_SIZE: f32 = 16.0;
const LABEL_INSET: f32 = BOX_SIZE + 8.0;
const ROW_H: f32 = 22.0;
// A disabled box, whatever its state.
const DISABLED_TINT: [f32; 4] = [0.22, 0.22, 0.25, 1.0];

// The height a checkbox takes: its row, then a line for the note.
pub(crate) const HEIGHT: f32 = ROW_H + widget::LINE_H;

// The reserved ids one checkbox draws with.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CheckIds {
    pub box_bg: AssetId,
    pub caption: AssetId,
    pub note: AssetId,
}

impl CheckIds {
    pub(crate) fn sprites(&self) -> [AssetId; 1] {
        [self.box_bg]
    }

    pub(crate) fn labels(&self) -> [AssetId; 2] {
        [self.caption, self.note]
    }
}

// One checkbox's state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Check {
    pub(crate) caption: String,
    pub(crate) on: bool,
    pub(crate) enabled: bool,
    pub(crate) note: Option<String>,
}

impl Check {
    // Flip the state; a disabled checkbox stays as it is.
    pub(crate) fn toggle(&mut self) {
        if self.enabled {
            self.on = !self.on;
        }
    }
}

// The row a press toggles: the box and its caption, across `rect`'s width.
pub(crate) fn row_rect(rect: [f32; 4]) -> [f32; 4] {
    [rect[0], rect[1], rect[2], ROW_H]
}

fn box_rect(rect: [f32; 4]) -> [f32; 4] {
    [
        rect[0],
        rect[1] + (ROW_H - BOX_SIZE) * 0.5,
        BOX_SIZE,
        BOX_SIZE,
    ]
}

pub(crate) fn hit(mx: f32, my: f32, rect: [f32; 4]) -> bool {
    point_in(mx, my, row_rect(rect))
}

// Draw `check` in `rect` (its width, `HEIGHT` tall). An enabled box that is
// off highlights under the cursor.
pub(crate) fn place(
    world: &mut World,
    ids: CheckIds,
    rect: [f32; 4],
    check: &Check,
    mouse: [f32; 2],
) {
    let hovered = check.enabled && hit(mouse[0], mouse[1], rect);
    let tint = match (check.enabled, check.on, hovered) {
        (false, _, _) => DISABLED_TINT,
        (true, true, _) => theme::CHECK_ON_TINT,
        (true, false, true) => theme::HOVER_TINT,
        (true, false, false) => theme::CHECK_OFF_TINT,
    };
    place_rounded(world, ids.box_bg, box_rect(rect), tint, 4.0, true);
    let color = match check.enabled {
        true => theme::LABEL,
        false => theme::LABEL_DIM,
    };
    let text_x = rect[0] + LABEL_INSET;
    let text_w = (rect[2] - LABEL_INSET).max(0.0);
    widget::place_message(
        world,
        ids.caption,
        [
            text_x,
            rect[1] + ROW_H * 0.5 - theme::TEXT_HALF,
            text_w,
            widget::LINE_H,
        ],
        &check.caption,
        color,
        true,
    );
    match &check.note {
        Some(note) => widget::place_message(
            world,
            ids.note,
            [text_x, rect[1] + ROW_H, text_w, widget::LINE_H],
            note,
            theme::LABEL_DIM,
            true,
        ),
        None => widget::set_label_visible(world, ids.note, false),
    }
}

pub(crate) fn hide(world: &mut World, ids: CheckIds) {
    widget::hide_all(world, &ids.sprites(), &ids.labels(), &[]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::{Sprite, TextLabel};

    const IDS: CheckIds = CheckIds {
        box_bg: AssetId(1),
        caption: AssetId(2),
        note: AssetId(3),
    };
    const RECT: [f32; 4] = [100.0, 100.0, 300.0, HEIGHT];

    fn world() -> World {
        crate::test_support::injected_world(&IDS.sprites(), &IDS.labels(), &[])
    }

    fn check(on: bool, enabled: bool, note: Option<&str>) -> Check {
        Check {
            caption: "Also delete its files".to_string(),
            on,
            enabled,
            note: note.map(str::to_string),
        }
    }

    fn tint(world: &World) -> [f32; 4] {
        world.get_by_id::<Sprite>(IDS.box_bg).unwrap().tint
    }

    fn label(world: &World, id: AssetId) -> TextLabel {
        world.get_by_id::<TextLabel>(id).cloned().unwrap()
    }

    #[test]
    fn a_disabled_check_keeps_its_state() {
        let mut c = check(false, true, None);
        c.toggle();
        assert!(c.on);
        let mut c = check(false, false, Some("why"));
        c.toggle();
        assert!(!c.on);
    }

    #[test]
    fn the_box_tint_follows_state_and_availability() {
        let mut world = world();
        let away = [0.0, 0.0];
        place(&mut world, IDS, RECT, &check(true, true, None), away);
        assert_eq!(tint(&world), theme::CHECK_ON_TINT);
        place(&mut world, IDS, RECT, &check(false, true, None), away);
        assert_eq!(tint(&world), theme::CHECK_OFF_TINT);
        let over = [RECT[0] + 4.0, RECT[1] + 4.0];
        place(&mut world, IDS, RECT, &check(false, true, None), over);
        assert_eq!(tint(&world), theme::HOVER_TINT);
        place(&mut world, IDS, RECT, &check(true, true, None), over);
        assert_eq!(tint(&world), theme::CHECK_ON_TINT, "on stays readable");
        place(
            &mut world,
            IDS,
            RECT,
            &check(true, false, Some("why")),
            over,
        );
        assert_eq!(tint(&world), DISABLED_TINT, "no hover on a disabled box");
        assert_eq!(label(&world, IDS.caption).color, theme::LABEL_DIM);
    }

    #[test]
    fn the_note_shows_under_the_caption_only_when_set() {
        let mut world = world();
        place(
            &mut world,
            IDS,
            RECT,
            &check(false, false, Some("why")),
            [0.0, 0.0],
        );
        let (caption, note) = (label(&world, IDS.caption), label(&world, IDS.note));
        assert!(note.visible && note.content == "why");
        assert!(note.y > caption.y);
        place(&mut world, IDS, RECT, &check(false, true, None), [0.0, 0.0]);
        assert!(!label(&world, IDS.note).visible);
        hide(&mut world, IDS);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }

    #[test]
    fn a_press_on_the_row_hits_and_the_note_does_not() {
        assert!(hit(RECT[0] + 200.0, RECT[1] + 5.0, RECT));
        assert!(!hit(RECT[0] + 30.0, RECT[1] + ROW_H + 5.0, RECT));
    }
}

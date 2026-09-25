//! Rows a type adds to the add / edit form beyond its schema fields: a
//! heading or note, a checkbox, or a choice cycled on click. The form lists
//! them after the fields, in the same scrolling window and control pool, and
//! reports a press on one by the id its type gave it. What a row means is the
//! type's business (`hook/form_extras.rs`); this is how one looks and where a
//! click on it lands.

use concinnity_core::ecs::World;

use crate::editor::theme;
use crate::editor::widget::{self, place_rounded, point_in};

// What a row carries on its right.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExtraControl {
    // Nothing: the caption alone, a heading or a note.
    Label,
    Check {
        on: bool,
        enabled: bool,
    },
    // Advances to the next option on click.
    Choice {
        options: Vec<String>,
        selected: usize,
    },
}

// One row. `detail` is dim text in the control column, after a checkbox's box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExtraRow {
    pub(crate) id: usize,
    pub(crate) caption: String,
    pub(crate) indent: bool,
    pub(crate) control: ExtraControl,
    pub(crate) detail: Option<String>,
}

impl ExtraRow {
    pub(crate) fn label(caption: impl Into<String>, detail: Option<String>) -> Self {
        Self {
            id: 0,
            caption: caption.into(),
            indent: false,
            control: ExtraControl::Label,
            detail,
        }
    }

    pub(crate) fn indented(mut self) -> Self {
        self.indent = true;
        self
    }

    // Whether a click on the row does anything.
    pub(crate) fn pressable(&self) -> bool {
        match &self.control {
            ExtraControl::Label => false,
            ExtraControl::Check { enabled, .. } => *enabled,
            ExtraControl::Choice { options, .. } => options.len() > 1,
        }
    }
}

// The rects of the form slot a row is drawn into.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Slot {
    pub(crate) row: [f32; 4],
    pub(crate) control: [f32; 4],
    pub(crate) toggle: [f32; 4],
}

// The slot's pooled elements the row draws with.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SlotIds {
    pub(crate) caption: concinnity_core::ecs::asset_id::AssetId,
    pub(crate) control: concinnity_core::ecs::asset_id::AssetId,
    pub(crate) text: concinnity_core::ecs::asset_id::AssetId,
}

const PAD: f32 = 10.0;
const INDENT: f32 = 14.0;
const LABEL: [f32; 3] = [0.90, 0.90, 0.92];
const LABEL_DIM: [f32; 3] = [0.60, 0.60, 0.66];
const CHECK_ON: [f32; 4] = [0.30, 0.66, 0.34, 1.0];
const CHECK_OFF: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const CHECK_ON_DISABLED: [f32; 4] = [0.24, 0.40, 0.27, 1.0];
const CHECK_OFF_DISABLED: [f32; 4] = [0.22, 0.22, 0.25, 1.0];

// Whether a click at (`mx`, `my`) presses `row` drawn in `slot`: a checkbox
// anywhere on its control column, a choice on its button.
pub(crate) fn hits(row: &ExtraRow, slot: &Slot, mx: f32, my: f32) -> bool {
    row.pressable() && point_in(mx, my, slot.control)
}

// Draw `row` into `slot`.
pub(crate) fn place(world: &mut World, row: &ExtraRow, slot: &Slot, ids: SlotIds, mouse: [f32; 2]) {
    let indent = if row.indent { INDENT } else { 0.0 };
    // An indented caption-only row is a note under the row above it.
    let note = row.control == ExtraControl::Label && row.indent;
    widget::place_left_label(
        world,
        ids.caption,
        [
            slot.row[0] + PAD + indent,
            slot.row[1] + slot.row[3] * 0.5 - theme::TEXT_HALF,
        ],
        &row.caption,
        if note { LABEL_DIM } else { LABEL },
        true,
    );
    let text_x = match &row.control {
        ExtraControl::Label => slot.control[0],
        ExtraControl::Check { on, enabled } => {
            let tint = match (on, enabled) {
                (true, true) => CHECK_ON,
                (false, true) => CHECK_OFF,
                (true, false) => CHECK_ON_DISABLED,
                (false, false) => CHECK_OFF_DISABLED,
            };
            place_rounded(world, ids.control, slot.toggle, tint, 4.0, true);
            slot.toggle[0] + slot.toggle[2] + 8.0
        }
        ExtraControl::Choice { options, selected } => {
            let c = slot.control;
            let hover = row.pressable() && point_in(mouse[0], mouse[1], c);
            let tint = if hover {
                theme::HOVER_TINT
            } else {
                theme::BUTTON_TINT
            };
            place_rounded(world, ids.control, c, tint, theme::CONTROL_RADIUS, true);
            let value = options.get(*selected).map(String::as_str).unwrap_or("");
            widget::place_left_label(
                world,
                ids.text,
                [c[0] + 8.0, c[1] + c[3] * 0.5 - theme::TEXT_HALF],
                value,
                LABEL,
                true,
            );
            return;
        }
    };
    if let Some(detail) = &row.detail {
        widget::place_left_label(
            world,
            ids.text,
            [
                text_x,
                slot.control[1] + slot.control[3] * 0.5 - theme::TEXT_HALF,
            ],
            detail,
            LABEL_DIM,
            true,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::{Sprite, TextLabel};
    use concinnity_core::ecs::asset_id::AssetId;

    fn check(on: bool, enabled: bool) -> ExtraRow {
        ExtraRow {
            id: 7,
            caption: "vertex".to_string(),
            indent: true,
            control: ExtraControl::Check { on, enabled },
            detail: Some("a.hlsl".to_string()),
        }
    }

    fn slot() -> Slot {
        Slot {
            row: [0.0, 100.0, 400.0, 34.0],
            control: [190.0, 103.0, 200.0, 26.0],
            toggle: [190.0, 103.0, 22.0, 22.0],
        }
    }

    const IDS: SlotIds = SlotIds {
        caption: AssetId(1),
        control: AssetId(2),
        text: AssetId(3),
    };

    #[test]
    fn only_an_enabled_check_or_a_real_choice_is_pressable() {
        assert!(check(false, true).pressable());
        assert!(!check(true, false).pressable());
        assert!(!ExtraRow::label("Stages", None).pressable());
        let choice = |n: usize| ExtraRow {
            control: ExtraControl::Choice {
                options: (0..n).map(|i| i.to_string()).collect(),
                selected: 0,
            },
            ..check(false, true)
        };
        assert!(choice(2).pressable() && !choice(1).pressable());
    }

    #[test]
    fn a_press_lands_on_the_control_column_only() {
        let row = check(false, true);
        assert!(hits(&row, &slot(), 300.0, 110.0));
        assert!(!hits(&row, &slot(), 20.0, 110.0), "the caption column");
        assert!(!hits(&check(false, false), &slot(), 300.0, 110.0));
    }

    #[test]
    fn a_check_draws_its_box_and_its_detail_after_it() {
        let mut world = World::new();
        world.push_identified(IDS.caption, TextLabel::default());
        world.push_identified(IDS.control, Sprite::default());
        world.push_identified(IDS.text, TextLabel::default());
        place(&mut world, &check(true, false), &slot(), IDS, [0.0, 0.0]);
        let sprite = world.get_by_id::<Sprite>(IDS.control).unwrap();
        assert!(sprite.visible);
        assert_eq!(sprite.tint, CHECK_ON_DISABLED);
        let text = world.get_by_id::<TextLabel>(IDS.text).unwrap();
        assert_eq!(text.content, "a.hlsl");
        assert!(text.x > slot().toggle[0] + slot().toggle[2]);
        let caption = world.get_by_id::<TextLabel>(IDS.caption).unwrap();
        assert_eq!(caption.content, "vertex");
    }
}

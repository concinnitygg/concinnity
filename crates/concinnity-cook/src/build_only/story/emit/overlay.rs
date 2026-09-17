use concinnity_core::components::StoryCommand;
use concinnity_core::gfx::overlay::UI_REFERENCE_SIZE;

use super::choices::{CHOICE_BOX_COLOR, CHOICE_BOX_RADIUS};
use super::names::StoryNames;
use super::widgets::{LabelStyle, hidden_label, hit_region, label, rounded_sprite};
use crate::authoring::spec::asset::ui_action;
use crate::build_only::ui_spec::sprite;

// Slot rows the save / load overlay shows at once. The story scrolls this
// fixed window over its larger set of logical slots (the auto-save resumed by
// Continue is separate), so each row's click action carries its row index, not
// a fixed slot number.
pub(super) const VISIBLE_SLOTS: usize = 5;

// Overlay furniture, declared after the rest of the stage so the dim draws
// over it: the shared full-canvas dim, the backlog history text, and the
// save/load slot rows. All hidden by rendering nothing (zero alpha, empty
// content); the story system fills them per overlay.
pub(super) fn emit_overlay(names: &StoryNames) -> Vec<serde_json::Value> {
    let (win_w, win_h) = (UI_REFERENCE_SIZE[0], UI_REFERENCE_SIZE[1]);
    let stage = &names.stage;
    let mut out = vec![
        sprite(&stage.dim, 0.0, 0.0, win_w, win_h, [0.02, 0.02, 0.04, 0.0]),
        label(
            &stage.history,
            &names.font_dialog,
            "",
            LabelStyle {
                x: 100.0,
                y: 70.0,
                color: [0.92, 0.92, 0.92],
                ..LabelStyle::default()
            },
        ),
        label(
            &stage.slot_title,
            &names.font_menu,
            "",
            LabelStyle {
                x: win_w / 2.0,
                y: 160.0,
                color: [1.0, 0.92, 0.78],
                align: Some("center"),
                ..LabelStyle::default()
            },
        ),
    ];
    for (i, row) in stage.slots.iter().enumerate() {
        let y = 230.0 + i as f32 * 80.0;
        out.push(rounded_sprite(
            &row.row_box,
            (280.0, y, win_w - 560.0, 56.0),
            [
                CHOICE_BOX_COLOR[0],
                CHOICE_BOX_COLOR[1],
                CHOICE_BOX_COLOR[2],
                0.0,
            ],
            CHOICE_BOX_RADIUS,
        ));
        out.push(hidden_label(
            &row.button.label,
            &names.font_menu,
            win_w / 2.0,
            y + 14.0,
            [0.92, 0.92, 0.92],
            None,
        ));
        out.push(hit_region(
            &row.button.region,
            (280.0, y, win_w - 560.0, 56.0),
            Some(&row.button.label),
            &ui_action::story(StoryCommand::Slot(i)),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_slot_rows_carry_their_row_index() {
        let out = emit_overlay(&StoryNames::new("s", true, 0));
        assert_eq!(out[0]["name"], "s_stage_dim");
        assert_eq!(out[0]["args"]["tint"][3], 0.0);
        let regions: Vec<_> = out.iter().filter(|e| e["type"] == "HitRegion").collect();
        assert_eq!(regions.len(), VISIBLE_SLOTS);
        for (i, region) in regions.iter().enumerate() {
            assert_eq!(region["name"], format!("s_stage_slot{}_btn", i));
            assert_eq!(region["args"]["action"], format!("story:slot:{}", i));
        }
    }
}

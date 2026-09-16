use concinnity_core::gfx::overlay::UI_REFERENCE_SIZE;

use super::names::StoryNames;
use super::widgets::{hidden_label, rounded_sprite};
use crate::authoring::spec::{asset, spec_to_value};

// Choice option rows: each option gets its own rounded box behind the label
// so the menu stands apart from the dialog box's dark backdrop. The color
// must match the story system's shown tint (it re-tints the boxes to show
// and hide them at runtime).
pub(super) const CHOICE_BOX_COLOR: [f32; 3] = [0.16, 0.20, 0.35];
pub(in crate::build_only::story) const CHOICE_BOX_RADIUS: f32 = 10.0;

// Choice furniture, sized for the widest menu in the story: one rounded box
// + label per option, hidden until the story system reaches a choice (each
// box is re-tinted visible with its slot). The buttons stay hit-active the
// whole time; the story system ignores a choose action outside a menu (and
// an advance inside one), so the overlap with the full-canvas advance region
// resolves by mode.
pub(super) fn emit_choice_furniture(names: &StoryNames) -> Vec<serde_json::Value> {
    let (win_w, win_h) = (UI_REFERENCE_SIZE[0], UI_REFERENCE_SIZE[1]);
    let options = &names.stage.options;
    let y0 = win_h / 2.0 - options.len() as f32 * 30.0;
    let mut out = Vec::new();
    for (ci, row) in options.iter().enumerate() {
        let y = y0 + ci as f32 * 60.0;
        out.push(rounded_sprite(
            &row.row_box,
            (280.0, y, win_w - 560.0, 44.0),
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
            y + 8.0,
            [0.92, 0.92, 0.92],
            None,
        ));
        out.push(spec_to_value(
            &asset::hit_region(
                row.button.region.as_str(),
                [280.0, y, win_w - 560.0, 44.0],
                format!("story:choose:{}", ci),
            )
            .set("label", row.button.label.as_str())
            .set("hover_color", [1.0f32, 0.85, 0.3])
            .set("hover_scale", 1.06f32),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_story_without_choices_emits_no_furniture() {
        assert!(emit_choice_furniture(&StoryNames::new("s", true, 0)).is_empty());
    }

    #[test]
    fn each_option_gets_a_box_label_and_region() {
        let out = emit_choice_furniture(&StoryNames::new("s", true, 3));
        assert_eq!(out.len(), 9);
        for (ci, row) in out.chunks(3).enumerate() {
            assert_eq!(row[0]["name"], format!("s_stage_opt{}_box", ci));
            assert_eq!(row[0]["args"]["corner_radius"], CHOICE_BOX_RADIUS);
            assert_eq!(row[1]["name"], format!("s_stage_opt{}_lbl", ci));
            assert_eq!(row[2]["args"]["action"], format!("story:choose:{}", ci));
            assert_eq!(row[2]["args"]["label"], row[1]["name"]);
        }
    }
}

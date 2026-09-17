use concinnity_core::components::StoryCommand;

use super::names::StoryNames;
use super::stage::DIALOG_BOX;
use super::widgets::{hidden_label, hit_region_fit};
use crate::authoring::spec::asset::ui_action;

// The quick row: small always-clickable controls along the dialog box's
// bottom edge (Log / Auto / Skip / Save). The story system fills each label's
// text in page mode and clears it elsewhere; the hit regions stay active the
// whole time and out-of-mode commands are ignored, like the choice buttons.
pub(super) fn emit_quick_row(names: &StoryNames) -> Vec<serde_json::Value> {
    // In `QUICK_KEYS` order.
    let actions = [
        StoryCommand::ToggleLog,
        StoryCommand::ToggleAuto,
        StoryCommand::ToggleSkip,
        StoryCommand::OpenSave,
    ];
    let quick_y = DIALOG_BOX.1 + DIALOG_BOX.3 - 38.0;
    let quick_w = 80.0;
    let quick_x0 = DIALOG_BOX.0 + DIALOG_BOX.2 - 30.0 - actions.len() as f32 * 90.0;
    let mut out = Vec::new();
    for (i, (button, action)) in names.stage.quick.iter().zip(actions).enumerate() {
        let x = quick_x0 + i as f32 * 90.0;
        out.push(hidden_label(
            &button.label,
            &names.font_dialog,
            x + quick_w / 2.0,
            quick_y + 2.0,
            [0.75, 0.75, 0.75],
            Some("bottom"),
        ));
        out.push(hit_region_fit(
            &button.region,
            (x, quick_y, quick_w, 30.0),
            Some(&button.label),
            &ui_action::story(action),
            Some("bottom"),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_hidden_bottom_fit_controls_fire_the_story_actions() {
        let out = emit_quick_row(&StoryNames::new("s", true, 0));
        assert_eq!(out.len(), 8);
        let (labels, regions): (Vec<_>, Vec<_>) =
            out.iter().partition(|e| e["type"] == "TextLabel");
        assert!(
            labels
                .iter()
                .all(|l| l["args"]["visible"] == false && l["args"]["fit"] == "bottom")
        );
        assert!(regions.iter().all(|r| r["args"]["fit"] == "bottom"));
        let actions: Vec<_> = regions
            .iter()
            .map(|r| r["args"]["action"].as_str().unwrap())
            .collect();
        assert_eq!(
            actions,
            ["story:log", "story:auto", "story:skip", "story:save"]
        );
        assert_eq!(regions[0]["args"]["label"], "s_stage_qlog_lbl");
    }
}

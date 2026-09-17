use concinnity_core::components::StoryCommand;
use concinnity_core::gfx::overlay::UI_REFERENCE_SIZE;

use super::names::StoryNames;
use super::widgets::{LabelStyle, hit_region, label, rounded_sprite_fit, screen, stage_sprite};
use crate::authoring::spec::asset::ui_action;

// The fixed dialog box the stage's name plate and dialog text sit on: nearly
// flush with the canvas bottom, tall enough for the name plate to sit inside
// against the box's dark backdrop.
pub(super) const DIALOG_BOX: (f32, f32, f32, f32) = (140.0, 500.0, 1000.0, 210.0);
pub(in crate::build_only::story) const DIALOG_BOX_RADIUS: f32 = 14.0;

// The stage: one screen the story system drives, from its backdrop through
// the advance marker. Sprites and labels are placeholders here; the system
// fills text, swaps textures, and toggles visibility page by page.
// Declaration order is draw order.
pub(super) fn emit_stage(names: &StoryNames) -> Vec<serde_json::Value> {
    let (win_w, win_h) = (UI_REFERENCE_SIZE[0], UI_REFERENCE_SIZE[1]);
    let stage = &names.stage;
    let mut out = vec![
        screen(&stage.screen, names.title.is_none()),
        stage_sprite(
            &stage.bg,
            [0.0, 0.0, win_w, win_h],
            [0.05, 0.06, 0.09, 1.0],
            true,
        ),
    ];
    for portrait in [&stage.left, &stage.center, &stage.right] {
        out.push(stage_sprite(
            portrait,
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            false,
        ));
    }
    // The dialog box and everything on it are bottom-anchored (fit scale, but
    // pinned to the window bottom) so the box hugs the bottom edge at any
    // aspect ratio instead of floating above the letterbox margin.
    out.push(rounded_sprite_fit(
        &stage.dialog_box,
        DIALOG_BOX,
        [0.0, 0.0, 0.0, 0.55],
        DIALOG_BOX_RADIUS,
        Some("bottom"),
    ));
    // The name plate sits inside the dialog box so the speaker reads against
    // its dark backdrop, with the dialogue below it.
    out.push(label(
        &stage.name_label,
        &names.font_menu,
        "",
        LabelStyle {
            x: DIALOG_BOX.0 + 30.0,
            y: DIALOG_BOX.1 + 14.0,
            color: [1.0, 1.0, 1.0],
            fit: Some("bottom"),
            ..LabelStyle::default()
        },
    ));
    out.push(label(
        &stage.text_label,
        &names.font_dialog,
        "",
        LabelStyle {
            x: DIALOG_BOX.0 + 30.0,
            y: DIALOG_BOX.1 + 58.0,
            color: [1.0, 0.95, 0.85],
            fit: Some("bottom"),
            ..LabelStyle::default()
        },
    ));
    out.push(hit_region(
        &stage.advance,
        (0.0, 0.0, win_w, win_h),
        None,
        &ui_action::story(StoryCommand::Advance),
    ));
    // Space and Enter both advance the dialogue (in addition to a click). Each
    // is its own KeyBinding; the UI fires whichever key was pressed.
    for (name, key) in stage.advance_keys.iter().zip(["Space", "Enter"]) {
        out.push(serde_json::json!({
            "name": name,
            "type": "KeyBinding",
            "args": { "key": key, "action": ui_action::story(StoryCommand::Advance) }
        }));
    }
    // The advance marker: a small rounded square at the dialog box's lower
    // right that the story system pulses while a fully revealed page waits
    // for a click.
    out.push(rounded_sprite_fit(
        &stage.marker,
        (
            DIALOG_BOX.0 + DIALOG_BOX.2 - 50.0,
            DIALOG_BOX.1 + DIALOG_BOX.3 - 70.0,
            14.0,
            14.0,
        ),
        [1.0, 0.95, 0.85, 0.0],
        4.0,
        Some("bottom"),
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stage_is_initial_only_without_a_title_screen() {
        let with_title = emit_stage(&StoryNames::new("s", true, 0));
        assert_eq!(with_title[0]["name"], "s_stage");
        assert_eq!(with_title[0]["args"]["initial"], false);
        let without = emit_stage(&StoryNames::new("s", false, 0));
        assert_eq!(without[0]["args"]["initial"], true);
    }

    #[test]
    fn space_and_enter_both_advance() {
        let out = emit_stage(&StoryNames::new("s", true, 0));
        let keys: Vec<_> = out.iter().filter(|e| e["type"] == "KeyBinding").collect();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0]["name"], "s_advance_key");
        assert_eq!(keys[0]["args"]["key"], "Space");
        assert_eq!(keys[1]["args"]["key"], "Enter");
        assert!(keys.iter().all(|k| k["args"]["action"] == "story:advance"));
        assert_eq!(out.last().unwrap()["name"], "s_stage_marker");
    }
}

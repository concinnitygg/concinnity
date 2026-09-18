use concinnity_core::components::StoryCommand;
use concinnity_core::gfx::overlay::UI_REFERENCE_SIZE;

use super::names::StoryNames;
use super::widgets::{LabelStyle, button, label, screen};
use crate::authoring::spec::asset::ui_action;
use crate::build_only::ui_spec::sprite;

// The ending screen, shown by the story system when the last node runs out
// of pages. Its button returns to the title screen, or restarts the story
// when there is none.
pub(super) fn emit_ending_screen(names: &StoryNames) -> Vec<serde_json::Value> {
    let (win_w, win_h) = (UI_REFERENCE_SIZE[0], UI_REFERENCE_SIZE[1]);
    let ending = &names.ending;
    let mut out = vec![
        screen(&ending.screen, false),
        sprite(&ending.bg, 0.0, 0.0, win_w, win_h, [0.03, 0.03, 0.05, 1.0]),
        label(
            &ending.fin,
            &names.font_title,
            "~ fin ~",
            LabelStyle {
                x: win_w / 2.0,
                y: 260.0,
                color: [0.95, 0.88, 0.7],
                align: Some("center"),
                ..LabelStyle::default()
            },
        ),
    ];
    let (back_label, back_action) = match &names.title {
        Some(title) => ("Back to title", ui_action::screen_show(&title.screen)),
        None => ("Restart", ui_action::story(StoryCommand::Start)),
    };
    out.extend(button(
        &ending.back,
        &names.font_menu,
        back_label,
        (win_w / 2.0 - 160.0, 490.0, 320.0),
        &back_action,
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn back_action(title_screen: bool) -> String {
        let out = emit_ending_screen(&StoryNames::new("s", title_screen, 0));
        let back = out.last().unwrap();
        assert_eq!(back["args"]["$id"], "s_ending_back_btn");
        back["args"]["action"].as_str().unwrap().to_string()
    }

    #[test]
    fn back_returns_to_the_title_screen_or_restarts() {
        assert_eq!(back_action(true), "screen:show:s_title");
        assert_eq!(back_action(false), "story:start");
    }
}

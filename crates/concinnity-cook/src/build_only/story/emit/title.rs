use concinnity_core::components::StoryCommand;
use concinnity_core::gfx::overlay::UI_REFERENCE_SIZE;

use super::media::MediaAssets;
use super::names::StoryNames;
use super::widgets::{LabelStyle, label, screen, textured_cover_sprite, title_button};
use crate::authoring::spec::asset::ui_action;
use crate::build_only::story::model::Story;
use crate::build_only::ui_spec::sprite;

// Multiplied into the title backdrop image so the light menu text keeps its
// contrast on a bright photo. The gray value dims the image to this fraction
// of its brightness; alpha stays opaque so the backdrop still fully covers.
const TITLE_BACKDROP_DIM: [f32; 4] = [0.35, 0.35, 0.35, 1.0];

// The title menu, empty when the import has no title screen.
pub(super) fn emit_title_screen(
    names: &StoryNames,
    story: &Story,
    media: &mut MediaAssets,
) -> Vec<serde_json::Value> {
    let Some(title) = &names.title else {
        return Vec::new();
    };
    let (win_w, win_h) = (UI_REFERENCE_SIZE[0], UI_REFERENCE_SIZE[1]);
    let mut out = vec![screen(&title.screen, true)];
    // The menu backdrop: a full-bleed image when the frontmatter set one,
    // else a flat dark fill.
    match &story.background {
        Some(path) => {
            let texture = media.image(path);
            // Darken the backdrop image (its tint multiplies the texture) so
            // the light title and menu text stay readable over a bright photo.
            // The image still shows through; the flat-fill case below is
            // already dark enough to need no dimming.
            out.push(textured_cover_sprite(
                &title.bg,
                [0.0, 0.0, win_w, win_h],
                &texture,
                TITLE_BACKDROP_DIM,
            ));
        }
        None => out.push(sprite(
            &title.bg,
            0.0,
            0.0,
            win_w,
            win_h,
            [0.05, 0.06, 0.12, 1.0],
        )),
    }
    out.push(label(
        &title.heading,
        &names.font_title,
        &story.title,
        LabelStyle {
            x: win_w / 2.0,
            y: 180.0,
            color: [1.0, 0.92, 0.78],
            align: Some("center"),
            ..LabelStyle::default()
        },
    ));
    // The menu buttons at default contiguous positions; the story re-lays
    // them out at runtime (Continue and Load only when a save exists,
    // Settings only when a settings screen exists), so their hit regions
    // follow their labels. Rows follow `TITLE_BUTTON_KEYS`.
    let rows = [
        ("Start", 400.0, ui_action::story(StoryCommand::Start)),
        ("Continue", 452.0, ui_action::story(StoryCommand::Continue)),
        ("Load", 504.0, ui_action::story(StoryCommand::OpenLoad)),
        (
            "Settings",
            556.0,
            ui_action::story(StoryCommand::OpenSettings),
        ),
        ("Quit", 608.0, ui_action::quit()),
    ];
    for (button, (text, y, action)) in title.buttons.iter().zip(rows) {
        out.extend(title_button(button, &names.font_menu, text, y, &action));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn story(background: Option<&str>) -> Story {
        Story {
            title: "T".to_string(),
            background: background.map(str::to_string),
            ..Story::default()
        }
    }

    #[test]
    fn no_title_screen_emits_nothing() {
        let names = StoryNames::new("s", false, 0);
        let mut media = MediaAssets::new("s");
        assert!(emit_title_screen(&names, &story(Some("m.png")), &mut media).is_empty());
        assert!(media.entries().is_empty());
    }

    #[test]
    fn without_a_background_the_backdrop_is_a_flat_fill() {
        let names = StoryNames::new("s", true, 0);
        let mut media = MediaAssets::new("s");
        let out = emit_title_screen(&names, &story(None), &mut media);
        let bg = &out[1]["args"];
        assert_eq!(out[1]["args"]["$id"], "s_title_bg");
        assert!(bg.get("texture").is_none());
        assert_eq!(bg["tint"][3], 1.0);
        assert!(media.entries().is_empty());
    }

    #[test]
    fn a_background_becomes_a_dimmed_cover_sprite() {
        let names = StoryNames::new("s", true, 0);
        let mut media = MediaAssets::new("s");
        let out = emit_title_screen(&names, &story(Some("m.png")), &mut media);
        let bg = &out[1]["args"];
        assert_eq!(bg["texture"], "s_img0");
        assert_eq!(bg["fit"], "cover");
        assert_eq!(bg["tint"][0], TITLE_BACKDROP_DIM[0]);
        assert_eq!(media.entries()[0]["args"]["source"], "m.png");
    }

    #[test]
    fn the_five_title_buttons_follow_their_labels() {
        let names = StoryNames::new("s", true, 0);
        let out = emit_title_screen(&names, &story(None), &mut MediaAssets::new("s"));
        let regions: Vec<_> = out.iter().filter(|e| e["type"] == "HitRegion").collect();
        assert_eq!(regions.len(), 5);
        for region in &regions {
            assert_eq!(region["args"]["follow_label"], true);
        }
        assert_eq!(regions[1]["args"]["$id"], "s_title_continue_btn");
        assert_eq!(regions[1]["args"]["action"], "story:continue");
        assert_eq!(regions[4]["args"]["action"], "quit");
    }
}

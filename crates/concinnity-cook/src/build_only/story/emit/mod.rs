use self::choices::emit_choice_furniture;
use self::ending::emit_ending_screen;
use self::graph::{compile_nodes, story_entry};
use self::media::MediaAssets;
use self::names::{StoryNames, check_screen_names};
use self::overlay::emit_overlay;
use self::quick_row::emit_quick_row;
use self::stage::emit_stage;
use self::title::emit_title_screen;
use self::widgets::font;
use super::model::{ImageDims, Story};
use crate::build_only::membership::scope_to_screen;

pub(super) mod choices;
mod ending;
mod graph;
mod media;
mod names;
mod overlay;
mod quick_row;
pub(super) mod stage;
mod title;
mod widgets;

#[cfg(test)]
mod tests;

// One screen's generated elements, each naming the screen it belongs to.
fn scoped(mut group: Vec<serde_json::Value>, screen: &str) -> Vec<serde_json::Value> {
    scope_to_screen(&mut group, screen);
    group
}

const TITLE_FONT_PX: u32 = 56;
const MENU_FONT_PX: u32 = 28;
const DIALOG_FONT_PX: u32 = 22;

// Emit the runtime assets for one parsed story: the compiled Story graph
// plus the stage scaffolding the story system drives at runtime. The whole
// story plays inside one stage screen whose labels and sprites are mutated
// page by page; the title and ending screens stay build-generated. Stages are
// emitted in draw order. `prefix` is the sanitized import name; every generated
// name starts with it. `image_dims` reads an image file's pixel size
// (portrait layout needs the aspect ratio); tests stub it so emission stays
// free of file IO.
pub(crate) fn emit_story(
    prefix: &str,
    story: &Story,
    title_screen: bool,
    text_speed: f32,
    image_dims: ImageDims,
) -> Result<Vec<serde_json::Value>, String> {
    let max_choices = story
        .nodes
        .iter()
        .map(|n| n.choices.len())
        .max()
        .unwrap_or(0);
    let names = StoryNames::new(prefix, title_screen, max_choices);
    let mut media = MediaAssets::new(prefix);

    let mut out = vec![
        font(&names.font_title, TITLE_FONT_PX),
        font(&names.font_menu, MENU_FONT_PX),
        font(&names.font_dialog, DIALOG_FONT_PX),
    ];
    // The title backdrop claims its texture name before the graph's images.
    let title_screen_name = names.title.as_ref().map(|t| t.screen.clone());
    out.extend(scoped(
        emit_title_screen(&names, story, &mut media),
        title_screen_name.as_deref().unwrap_or(""),
    ));
    let nodes = compile_nodes(story, &mut media, image_dims)?;
    out.push(story_entry(&names, story, nodes, text_speed));
    for group in [
        emit_stage(&names),
        emit_quick_row(&names),
        emit_choice_furniture(&names),
        emit_overlay(&names),
    ] {
        out.extend(scoped(group, &names.stage.screen));
    }
    out.extend(scoped(emit_ending_screen(&names), &names.ending.screen));
    out.extend(media.entries());

    check_screen_names(&names)?;
    Ok(out)
}

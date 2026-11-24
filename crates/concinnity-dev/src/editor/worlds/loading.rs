//! The start screen's loading cover: the black field and caption that stand
//! over the preview area while the world the sidebar picked is compiled. The
//! editor's own window comes up first and the listing with it, so the wait is
//! spent on a screen the user can already read and click, and only the render
//! side of it is covered.
//!
//! Styled after the engine's own scene-loading overlay
//! (`concinnity-engine/src/defaults/loading.rs`): a black field with the word
//! over it. There is no progress bar, because there is no progress to report --
//! the compile is one call, and a bar frozen mid-sweep reads as a hang.

use concinnity_engine::components::TextAlign;

use crate::editor::theme;
use crate::editor::ui::shapes::{Shape, Shapes};

// The same black the shots hand over through, so the cover gives way to the
// opening fade without a step in between.
const COVER_TINT: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

// Write the caption a world's name earns into `out`, or the bare word while
// nothing is named.
pub(crate) fn caption(name: Option<&str>, out: &mut String) {
    out.clear();
    out.push_str("Loading");
    if let Some(name) = name {
        out.push(' ');
        out.push_str(name);
    }
}

// Cover `area` (the window the sidebar does not stand on) and caption it in
// its middle.
pub(crate) fn push(shapes: &mut Shapes<'_>, area: [f32; 4], caption: &str) {
    shapes.push(Shape {
        rect: area,
        fill: COVER_TINT,
        text: caption,
        text_color: theme::LABEL_DIM,
        align: TextAlign::Center,
        ..Shape::default()
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cover_fills_its_area_and_names_what_is_loading() {
        let mut text = String::new();
        caption(Some("bistro"), &mut text);
        let mut list = Vec::new();
        let mut shapes = Shapes::new(&mut list);
        push(&mut shapes, [280.0, 0.0, 1000.0, 720.0], &text);
        shapes.finish();

        let cover = &list[0];
        assert_eq!(cover.rect, [280.0, 0.0, 1000.0, 720.0]);
        assert_eq!(
            cover.fill, COVER_TINT,
            "opaque, so nothing half-built shows"
        );
        assert_eq!(cover.text, "Loading bistro");
        assert_eq!(cover.align, TextAlign::Center);
    }

    #[test]
    fn an_unnamed_world_still_says_what_is_happening() {
        let mut text = "stale".to_string();
        caption(None, &mut text);
        assert_eq!(text, "Loading");
    }
}

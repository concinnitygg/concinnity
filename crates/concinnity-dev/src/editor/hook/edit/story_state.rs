//! EditorHook: the Story panel's session state, beside its actions in `story.rs`.

use crate::editor::text_area::TextArea;
use crate::editor::text_area::markers::GutterMarker;

// Shown state, the loaded source in its text area, whether that area holds the
// keyboard (while the panel is frontmost), the source path shown in the
// header, the last parse / IO error, and the gutter marker a parse error pins
// to its line.
#[derive(Debug, Default)]
pub(in crate::editor::hook) struct StoryState {
    pub(in crate::editor::hook) open: bool,
    pub(in crate::editor::hook) area: TextArea,
    pub(in crate::editor::hook) focus: bool,
    pub(in crate::editor::hook) path: String,
    pub(in crate::editor::hook) status: Option<String>,
    pub(in crate::editor::hook) markers: Vec<GutterMarker>,
}

impl StoryState {
    // Drop the source read out of the world being left. Shown state is not the
    // world's.
    pub(in crate::editor::hook) fn reset_for_world(&mut self) {
        self.area = TextArea::default();
        self.focus = false;
        self.path = String::new();
        self.status = None;
        self.markers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_for_world_keeps_shown_state() {
        let mut s = StoryState {
            open: true,
            area: TextArea::from_text("a\nb"),
            focus: true,
            path: "story.md".into(),
            status: Some("bad".into()),
            markers: crate::editor::panels::story::error_marker("line 1: bad")
                .into_iter()
                .collect(),
        };
        s.area.type_char('x');
        s.reset_for_world();
        assert_eq!(s.area.text(), "");
        assert!(!s.area.is_dirty());
        assert!(!s.focus);
        assert_eq!((s.path.as_str(), s.status.as_deref()), ("", None));
        assert!(s.markers.is_empty());
        assert!(s.open);
    }
}

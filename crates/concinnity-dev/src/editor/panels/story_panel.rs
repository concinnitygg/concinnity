//! The Story panel: an editor over the Markdown source of the world's
//! `StoryImport` (see `story.rs` for the starter text and `hook/edit/story.rs`
//! for the actions). The header carries the source path and the Apply button,
//! a status line shows parse / IO errors, and the body is a code text area,
//! whose editor the hook lends the panel each frame. A world with no
//! `StoryImport` shows a "+ Create story" button instead.

use concinnity_engine::components::{
    Button, LineMark, TextAlign, TextArea, TextLabel, TextLanguage, UiEvent,
};
use concinnity_engine::ecs::World;
use concinnity_engine::presentation::text_edit::TextEditor;

use super::registry::PanelKey;
use crate::editor::text_area::markers::{GutterMarker, Severity};
use crate::editor::theme;
use crate::editor::ui::{BuiltPanel, PanelSpec, ROW_H, UiBuilder, panel_height, state};

// The default (and minimum) width; the user can widen the panel past this.
pub(crate) const STORY_W: f32 = 620.0;
const HEADER_H: f32 = 32.0;
const BTN_W: f32 = 70.0;
// The text area's height at the default panel size.
const AREA_H: f32 = 440.0;

const PATH_COLOR: [f32; 3] = [0.62, 0.66, 0.76];
const ADD_LABEL: [f32; 3] = [0.55, 0.80, 0.60];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];

// The state the panel shows, and the events its controls send.
const TITLE: &str = "editor.story.title";
const PATH: &str = "editor.story.path";
const STATUS: &str = "editor.story.status";
const APPLY: &str = "editor.story.apply";
const SAVE: &str = "editor.story.save";
const CREATE: &str = "editor.story.create";

// The panel's slots.
const AREA_SLOT: usize = 0;
const APPLY_SLOT: usize = 1;
const CREATE_SLOT: usize = 2;
const STATUS_SLOT: usize = 3;

// What the user asked of the panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StoryAction {
    // Write the starter story and add its StoryImport entry.
    Create,
    // Validate, write the source file, and refresh the live preview.
    Apply,
}

// The action a Story event asks for, if `event` is one.
pub(crate) fn action(event: &UiEvent) -> Option<StoryAction> {
    match event.name.as_str() {
        CREATE => Some(StoryAction::Create),
        APPLY | SAVE => Some(StoryAction::Apply),
        _ => None,
    }
}

// Where the panel sits until the user drags it: centered below the top bar
// (clear of the left default stack and the right Assets anchor).
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [(vw - STORY_W) * 0.5, crate::editor::hud::body_top() + 8.0]
}

// The panel's default (and minimum) footprint.
pub(crate) fn size() -> [f32; 2] {
    [STORY_W, panel_height(HEADER_H + ROW_H + AREA_H)]
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let path = ui.bound_label(PATH, PATH_COLOR, TextAlign::Left)?;
    let apply = ui.button("Apply", APPLY, None, [BTN_W, HEADER_H - 8.0])?;
    let apply_id = apply.item?.id();
    let header = ui.row(vec![path, apply], HEADER_H)?;
    let status = ui.bound_label(STATUS, ERROR_LABEL, TextAlign::Left)?;
    let status_id = status.item?.id();
    let create = ui.button("+ Create story", CREATE, None, [STORY_W - 16.0, ROW_H])?;
    let create_id = create.item?.id();
    let area = ui.text_area(TextArea {
        language: TextLanguage::Markdown,
        on_save: SAVE.to_string(),
        ..Default::default()
    })?;
    let area_id = area.item?.id();
    let spec = PanelSpec {
        key: PanelKey::Story,
        title: "Story",
        title_state: Some(TITLE),
        size: size(),
        min_size: Some(size()),
    };
    let mut panel = ui.panel(&spec, vec![header, status, create, area])?;
    panel.slots = vec![area_id, apply_id, create_id, status_id];
    Some(panel)
}

// What the panel shows this frame.
pub(crate) struct StoryView<'a> {
    // Whether the text area holds the keyboard.
    pub focus: bool,
    // Whether the source has edits not yet applied.
    pub dirty: bool,
    // The story source path shown in the header; empty in create mode.
    pub path: &'a str,
    // Parse / IO error from the last Apply (or load).
    pub status: Option<&'a str>,
    // Lines the last Apply's parse error names.
    pub markers: &'a [GutterMarker],
    // No StoryImport in the world: show the create button instead of the text.
    pub create: bool,
}

fn mark_color(severity: Severity) -> [f32; 3] {
    match severity {
        Severity::Error => theme::LOG_ERROR,
        Severity::Warning => theme::LOG_WARN,
    }
}

// Show the path, status, and mode, and the area's focus and marks.
pub(crate) fn sync(view: &StoryView<'_>, world: &mut World, built: &BuiltPanel) {
    let title = if view.dirty { "Story *" } else { "Story" };
    state::set_text(world, TITLE, format_args!("{title}"));
    state::set_text(world, PATH, format_args!("{}", view.path));
    state::set_text(world, STATUS, format_args!("{}", view.status.unwrap_or("")));
    let slot = |i: usize| built.slots.get(i).copied();
    if let Some(area) = slot(AREA_SLOT).and_then(|id| world.get_mut_by_id::<TextArea>(id)) {
        area.visible = !view.create;
        area.focused = view.focus && !view.create;
        let marks = view.markers.iter().map(|m| LineMark {
            line: m.line,
            color: mark_color(m.severity),
        });
        if !area.marks.iter().copied().eq(marks.clone()) {
            area.marks.clear();
            area.marks.extend(marks);
        }
    }
    if let Some(b) = slot(APPLY_SLOT).and_then(|id| world.get_mut_by_id::<Button>(id)) {
        b.visible = !view.create;
    }
    if let Some(b) = slot(CREATE_SLOT).and_then(|id| world.get_mut_by_id::<Button>(id)) {
        b.visible = view.create;
        b.style.text = ADD_LABEL;
    }
    if let Some(l) = slot(STATUS_SLOT).and_then(|id| world.get_mut_by_id::<TextLabel>(id)) {
        l.wrap_width = STORY_W - 16.0;
        l.max_lines = 2;
    }
}

// Whether the text area holds the keyboard, as the widget last showed it.
pub(crate) fn area_focused(world: &World, built: &BuiltPanel) -> bool {
    built
        .slots
        .get(AREA_SLOT)
        .and_then(|&id| world.get_by_id::<TextArea>(id))
        .is_some_and(|a| a.focused)
}

// Swap `editor` with the text area's: lending it to the area, or taking it
// back.
pub(crate) fn swap_editor(editor: &mut TextEditor, world: &mut World, built: &BuiltPanel) -> bool {
    let Some(area) = built
        .slots
        .get(AREA_SLOT)
        .and_then(|&id| world.get_mut_by_id::<TextArea>(id))
    else {
        return false;
    };
    std::mem::swap(editor, &mut *area.editor);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::ecs::asset_id::AssetId;

    fn built(world: &mut World) -> BuiltPanel {
        let mut ui = UiBuilder::new(world, AssetId(9), None);
        build(&mut ui).unwrap()
    }

    fn view(create: bool) -> StoryView<'static> {
        StoryView {
            focus: true,
            dirty: true,
            path: "story.md",
            status: None,
            markers: &[],
            create,
        }
    }

    #[test]
    fn create_mode_shows_the_create_button_alone() {
        let mut world = World::new();
        let panel = built(&mut world);
        sync(&view(true), &mut world, &panel);
        let area = world.get_by_id::<TextArea>(panel.slots[AREA_SLOT]).unwrap();
        assert!(!area.visible && !area.focused);
        assert!(
            world
                .get_by_id::<Button>(panel.slots[CREATE_SLOT])
                .unwrap()
                .visible
        );
        sync(&view(false), &mut world, &panel);
        let area = world.get_by_id::<TextArea>(panel.slots[AREA_SLOT]).unwrap();
        assert!(area.visible && area.focused);
        assert!(
            !world
                .get_by_id::<Button>(panel.slots[CREATE_SLOT])
                .unwrap()
                .visible
        );
    }

    #[test]
    fn an_error_marker_marks_its_line() {
        let mut world = World::new();
        let panel = built(&mut world);
        let markers = [GutterMarker {
            line: 4,
            column: 0,
            severity: Severity::Error,
            message: "bad".into(),
        }];
        let v = StoryView {
            markers: &markers,
            ..view(false)
        };
        sync(&v, &mut world, &panel);
        let area = world.get_by_id::<TextArea>(panel.slots[AREA_SLOT]).unwrap();
        assert_eq!(area.marks.len(), 1);
        assert_eq!(area.marks[0].line, 4);
    }

    #[test]
    fn the_editor_lends_and_comes_back_whole() {
        let mut world = World::new();
        let panel = built(&mut world);
        let mut editor = TextEditor::from_text("# hi");
        assert!(swap_editor(&mut editor, &mut world, &panel));
        let area = world.get_by_id::<TextArea>(panel.slots[AREA_SLOT]).unwrap();
        assert_eq!(area.editor.text(), "# hi");
        assert!(swap_editor(&mut editor, &mut world, &panel));
        assert_eq!(editor.text(), "# hi");
    }

    #[test]
    fn save_applies() {
        let event = UiEvent {
            name: SAVE.into(),
            element: None,
            value: None,
        };
        assert_eq!(action(&event), Some(StoryAction::Apply));
    }
}

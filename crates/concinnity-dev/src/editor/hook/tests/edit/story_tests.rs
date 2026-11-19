//! The Story panel's actions (`hook/edit/story.rs`): the source's editor lent
//! to the text area and taken back, the area's focus, the save shortcut and
//! Apply validating before they write, the status a missing file reports, and
//! the starter file a create writes along with the import that names it.

use concinnity_engine::components::{FrameInput, TextArea, UiEvent};
use concinnity_engine::ecs::World;

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::hook;
use crate::editor::install;
use crate::editor::panels::registry::PanelKey;
use crate::editor::panels::story;
use crate::editor::ui::BuiltUi;
use crate::frame_hook::FrameHook;
use concinnity_engine::presentation::text_edit::TextEditor;

fn story_import(source: &str) -> serde_json::Value {
    serde_json::json!({"type": "StoryImport", "args": {"$id": "tale", "source": source}})
}

// A hook with the story panel open, frontmost, focused, and holding `text`
// (no file IO: the area edits in memory until Apply).
fn story_session(text: &str) -> (EditorHook, World) {
    story_session_at("unused.md", text)
}

fn story_session_at(source: &str, text: &str) -> (EditorHook, World) {
    let mut world = World::new();
    install::editor_ui(&mut world);
    let mut h = hook(vec![story_import(source)]);
    h.viewport = [1280.0, 720.0];
    h.open[PanelKey::Story] = true;
    h.story.area = TextEditor::from_text(text);
    h.story.focus = true;
    h.focus_panel(PanelKey::Story);
    (h, world)
}

// The hook lends the source's editor to the panel's text area for the frame
// and takes it back whole at the next one, so the area edits the hook's text.
#[test]
fn the_editor_is_lent_to_the_text_area_and_taken_back() {
    let (mut h, mut world) = story_session("one");
    world.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    h.tick(&mut world);
    h.tick(&mut world);
    let area = world.query::<TextArea>().next().unwrap();
    assert_eq!(area.editor.text(), "one", "lent for the frame");
    assert!(area.visible && area.focused);
    h.reclaim_built(&mut world);
    assert_eq!(h.story.area.text(), "one", "taken back whole");
}

// The area holds the keyboard only while focused and the panel is frontmost,
// and a press the engine gave the area, or took from it, reaches the hook.
#[test]
fn focus_follows_the_area_and_the_front_panel() {
    let (mut h, mut world) = story_session("x");
    assert!(h.story_view().focus);
    h.focus_panel(PanelKey::Console);
    assert!(!h.story_view().focus, "a panel in front takes the keyboard");
    h.focus_panel(PanelKey::Story);
    world.add_component(FrameInput {
        viewport: [1280.0, 720.0],
        ..Default::default()
    });
    h.tick(&mut world);
    h.tick(&mut world);
    for area in world.query_mut::<TextArea>() {
        area.focused = false;
    }
    let built = world
        .resource::<BuiltUi>()
        .unwrap()
        .panel(PanelKey::Story)
        .unwrap()
        .clone();
    h.observe_story(&world, &built);
    assert!(!h.story.focus, "a press elsewhere blurred it");
}

// Apply validates with the real story parser before writing: a broken story
// shows on the status line and the file is untouched; a valid one writes and
// refreshes the live preview without touching the world.jsonl dirty flag.
#[test]
fn story_apply_validates_then_writes() {
    let tree = concinnity_testing::TempTree::new();
    let path = tree.write("tale.md", story::STARTER_STORY);
    let src = path.to_string_lossy().to_string();

    let mut h = hook(vec![story_import(&src)]);
    h.open[PanelKey::Story] = true;
    h.load_story();
    assert_eq!(h.story.status, None);
    assert_eq!(h.story.path, src);
    assert_eq!(h.story.area.text(), story::STARTER_STORY);

    // Break the story (no frontmatter): Apply rejects and writes nothing.
    h.story.area = TextEditor::from_text("just prose, no frontmatter");
    h.story.area.type_char('!');
    h.apply_story();
    assert!(h.story.status.is_some(), "parse failure shown");
    assert!(
        h.story.area.is_dirty(),
        "a rejected apply leaves the edit unapplied"
    );
    assert!(!h.rebuild_preview);
    let on_disk = std::fs::read_to_string(&path).unwrap();
    assert_eq!(on_disk, story::STARTER_STORY, "file untouched");

    // An error naming its line pins a gutter marker there.
    let broken = story::STARTER_STORY.replace("The story begins here.", "[x](#nowhere) prose");
    h.story.area = TextEditor::from_text(&broken);
    h.apply_story();
    let marked: Vec<usize> = h.story.markers.iter().map(|m| m.line).collect();
    let line = broken.lines().position(|l| l.contains("#nowhere")).unwrap();
    assert_eq!(marked, [line]);
    assert_eq!(
        h.story.area.caret().line,
        line,
        "the caret jumps to the error"
    );

    // A valid edit writes and requests the preview rebuild; the world.jsonl
    // dirty flag stays clear (no entry changed).
    h.load_story();
    assert!(h.story.markers.is_empty(), "a reload clears the markers");
    let last = h.story.area.line_count() - 1;
    h.story.area.go_to(last, 0);
    for c in "And they lived on.\n".chars() {
        h.story.area.type_char(c);
    }
    h.apply_story();
    assert_eq!(h.story.status, None);
    assert!(!h.story.area.is_dirty(), "applied text is saved text");
    assert!(h.rebuild_preview, "preview refresh requested");
    assert!(!h.dirty, "no world.jsonl change");
    let on_disk = std::fs::read_to_string(&path).unwrap();
    assert!(on_disk.ends_with("The story begins here.\nAnd they lived on.\n"));
}

// The save shortcut applies like the Apply button.
#[test]
fn the_save_shortcut_applies() {
    let tree = concinnity_testing::TempTree::new();
    let path = tree.write("tale.md", story::STARTER_STORY);
    let (mut h, mut world) = story_session_at(&path.to_string_lossy(), "");
    h.load_story();
    h.story.area.go_to(0, 0);
    h.story.area.type_char('\n');
    world.events_mut::<UiEvent>().send(UiEvent {
        name: "editor.story.save".into(),
        element: None,
        value: None,
    });
    h.read_ui_events(&mut world);
    let on_disk = std::fs::read_to_string(&path).unwrap();
    assert!(on_disk.starts_with("\n---"), "{on_disk:?}");
    assert!(!h.story.area.is_dirty());
}

// A missing source file loads as an empty editable story with the error shown.
#[test]
fn story_load_missing_file_shows_status() {
    let mut h = hook(vec![story_import("/no/such/dir/story.md")]);
    h.open[PanelKey::Story] = true;
    h.load_story();
    assert!(h.story.status.is_some());
    assert_eq!(h.story.area.text(), "");
}

// Create writes the starter file, adds the StoryImport entry (a normal world
// edit), and loads it for editing. In a working directory of its own: the new
// file lands relative to the project root.
#[test]
fn story_create_writes_starter_and_adds_the_import() {
    let state = crate::test_support::in_temp_cwd();

    let mut h = hook(Vec::new());
    h.open[PanelKey::Story] = true;
    h.load_story();
    assert!(h.story_view().create, "no import: create mode");
    h.create_story();

    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["type"], "StoryImport");
    assert_eq!(h.entries[0]["args"]["source"], "story.md");
    assert!(h.dirty, "the new entry is a world edit");
    assert_eq!(h.story.area.text(), story::STARTER_STORY);
    assert!(h.story.focus, "the new story is ready to type into");
    assert!(!h.story_view().create);
    let written = std::fs::read_to_string(state.tree().join("story.md")).unwrap();
    assert_eq!(written, story::STARTER_STORY);
    // A second create is a no-op while an import exists.
    h.create_story();
    assert_eq!(h.entries.len(), 1);
}

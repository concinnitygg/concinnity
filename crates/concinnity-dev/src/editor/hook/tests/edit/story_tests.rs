//! The Story panel's actions (`hook/edit/story.rs`): key events reaching the
//! text area only while it holds the keyboard, presses focusing it, the save
//! shortcut and Apply validating before they write, the status a missing file
//! reports, and the starter file a create writes along with the import that
//! names it.

use concinnity_core::components::{FrameInput, InputKey, KeyEvent, KeyMods, KeyPress};
use concinnity_core::ecs::World;
use concinnity_core::window::clipboard::Clipboard;

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::hook;
use crate::editor::inject;
use crate::editor::panels::registry::PanelKey;
use crate::editor::panels::story;
use crate::editor::panels::story_panel::{self, StoryAction};
use crate::editor::text_area::TextArea;

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
    inject::editor_hud(&mut world);
    let mut h = hook(vec![story_import(source)]);
    h.viewport = [1280.0, 720.0];
    h.story.open = true;
    h.story.area = TextArea::from_text(text);
    h.story.focus = true;
    h.focus_panel(PanelKey::Story);
    (h, world)
}

fn events(events: Vec<KeyEvent>) -> FrameInput {
    FrameInput {
        key_events: events,
        viewport: [1280.0, 720.0],
        ..Default::default()
    }
}

fn chord(key: InputKey) -> KeyEvent {
    let mods = if cfg!(target_os = "macos") {
        KeyMods::CMD
    } else {
        KeyMods::CTRL
    };
    KeyEvent::Press(KeyPress::new(key, mods))
}

// A burst of keystrokes landing in one frame all reach the text, in order,
// and the heading marks the unapplied edit.
#[test]
fn a_frame_of_keys_edits_the_focused_story() {
    let (mut h, mut world) = story_session("    one");
    h.story.area.go_to(0, 7);
    let input = events(vec![
        KeyEvent::press(InputKey::Enter),
        KeyEvent::Text('t'),
        KeyEvent::Text('w'),
        KeyEvent::Text('o'),
    ]);
    h.story_keys(&mut world, &input);
    assert_eq!(h.story.area.text(), "    one\n    two");
    assert!(h.make_story_view([0.0, 0.0]).area.is_dirty());
}

// Keys reach the area only while it is focused and the panel is frontmost.
#[test]
fn keys_need_focus_and_the_front_panel() {
    let (mut h, mut world) = story_session("x");
    let typed = events(vec![KeyEvent::Text('y')]);
    h.story.focus = false;
    h.story_keys(&mut world, &typed);
    assert_eq!(h.story.area.text(), "x", "an unfocused area ignores typing");
    h.story.focus = true;
    h.focus_panel(PanelKey::Console);
    h.story_keys(&mut world, &typed);
    assert_eq!(
        h.story.area.text(),
        "x",
        "a panel in front takes the keyboard"
    );
    assert!(!h.make_story_view([0.0, 0.0]).focus);
    h.focus_panel(PanelKey::Story);
    h.story_keys(&mut world, &typed);
    assert_eq!(h.story.area.text(), "yx");
}

// A press on the text area focuses it and places the caret; a press on the
// panel chrome blurs it.
#[test]
fn presses_focus_and_blur_the_area() {
    let (mut h, _world) = story_session("first\nsecond");
    h.story.focus = false;
    let o = h.origin(PanelKey::Story, h.viewport);
    let s = h.effective_size(PanelKey::Story);
    let r = story_panel::area_rect(o, s);
    let second_row = r[1] + 1.5 * crate::editor::code_font::LINE_H;
    h.apply_story_action(StoryAction::Text, r[0] + r[2] - 20.0, second_row);
    assert!(h.story.focus);
    assert_eq!(h.story.area.caret().line, 1);
    h.apply_story_action(StoryAction::Consume, 0.0, 0.0);
    assert!(!h.story.focus);
}

// Copy and paste work with no window: the internal clipboard stands in for the
// system one.
#[test]
fn copy_and_paste_fall_back_to_the_internal_clipboard() {
    let (mut h, mut world) = story_session("abc");
    let input = events(vec![chord(InputKey::A), chord(InputKey::C)]);
    h.story_keys(&mut world, &input);
    assert_eq!(h.text_clipboard.text().as_deref(), Some("abc"));
    let input = events(vec![KeyEvent::press(InputKey::End), chord(InputKey::V)]);
    h.story_keys(&mut world, &input);
    assert_eq!(h.story.area.text(), "abcabc");
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
    h.story.open = true;
    h.load_story();
    assert_eq!(h.story.status, None);
    assert_eq!(h.story.path, src);
    assert_eq!(h.story.area.text(), story::STARTER_STORY);

    // Break the story (no frontmatter): Apply rejects and writes nothing.
    h.story.area = TextArea::from_text("just prose, no frontmatter");
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
    h.story.area = TextArea::from_text(&broken);
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
    h.story.focus = true;
    h.story.area.go_to(0, 0);
    let input = events(vec![KeyEvent::press(InputKey::Enter), chord(InputKey::S)]);
    h.story_keys(&mut world, &input);
    let on_disk = std::fs::read_to_string(&path).unwrap();
    assert!(on_disk.starts_with("\n---"), "{on_disk:?}");
    assert!(!h.story.area.is_dirty());
}

// A missing source file loads as an empty editable story with the error shown.
#[test]
fn story_load_missing_file_shows_status() {
    let mut h = hook(vec![story_import("/no/such/dir/story.md")]);
    h.story.open = true;
    h.load_story();
    assert!(h.story.status.is_some());
    assert_eq!(h.story.area.text(), "");
}

// Create writes the starter file, adds the StoryImport entry (a normal world
// edit), and loads it for editing. Serialized via the cwd lock: the new file
// lands relative to the project root.
#[test]
fn story_create_writes_starter_and_adds_the_import() {
    let _guard = crate::test_support::lock();
    let tree = concinnity_testing::TempTree::new();
    let old = std::env::current_dir().unwrap();
    std::env::set_current_dir(tree.path()).unwrap();

    let mut h = hook(Vec::new());
    h.story.open = true;
    h.load_story();
    assert!(
        h.make_story_view([0.0, 0.0]).create,
        "no import: create mode"
    );
    h.create_story();

    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["type"], "StoryImport");
    assert_eq!(h.entries[0]["args"]["source"], "story.md");
    assert!(h.dirty, "the new entry is a world edit");
    assert_eq!(h.story.area.text(), story::STARTER_STORY);
    assert!(h.story.focus, "the new story is ready to type into");
    assert!(!h.make_story_view([0.0, 0.0]).create);
    let written = std::fs::read_to_string(tree.join("story.md")).unwrap();
    assert_eq!(written, story::STARTER_STORY);
    // A second create is a no-op while an import exists.
    h.create_story();
    assert_eq!(h.entries.len(), 1);

    std::env::set_current_dir(old).unwrap();
}

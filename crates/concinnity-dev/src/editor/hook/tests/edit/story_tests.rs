// src/editor/hook/tests/edit/story_tests.rs
//
// The Story panel's actions (`hook/edit/story.rs`): the line model's splits and
// joins, the caret's navigation and what it commits on the way, the validation
// an apply runs before writing, the status a missing file reports, and the
// starter file a create writes along with the import that names it.

use concinnity_core::components::InputKey;
use concinnity_core::ecs::World;

use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::{hook, story_key_input};

use crate::editor::inject;

use crate::editor::panels::registry::PanelKey;
use crate::editor::panels::story;
use crate::editor::panels::story_panel;

use crate::editor::widget;

// Story panel

fn story_import(source: &str) -> serde_json::Value {
    serde_json::json!({"name": "tale", "type": "StoryImport", "args": {"source": source}})
}

// A hook + injected world with the story loaded from `lines` (no file IO: the
// line editor operates purely on the loaded lines until Apply).
fn story_session(lines: &[&str]) -> (EditorHook, World) {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let mut h = hook(vec![story_import("unused.md")]);
    h.story_open = true;
    h.story_lines = lines.iter().map(|s| s.to_string()).collect();
    h.story_focus = true;
    h.focus_panel(PanelKey::Story);
    h.seed_story_line(&mut world);
    (h, world)
}

fn line_caret(world: &mut World, caret: usize) {
    let t = widget::input_mut(world, story_panel::LINE_INPUT).unwrap();
    t.caret = caret;
}

// Enter splits the current line at the caret; Backspace at column 0 joins it
// back, blurring the control for one frame so the text system does not also
// eat a character.
#[test]
fn story_enter_splits_and_backspace_joins() {
    let (mut h, mut world) = story_session(&["hello world"]);
    line_caret(&mut world, 5);
    h.story_keys(&mut world, &story_key_input(InputKey::Enter));
    assert_eq!(h.story_lines, ["hello", " world"]);
    assert_eq!(h.story_line, 1);
    let input = widget::input(&world, story_panel::LINE_INPUT).unwrap();
    assert_eq!(input.content, " world");
    assert_eq!(input.caret, 0, "the caret starts the new line");

    h.story_keys(&mut world, &story_key_input(InputKey::Backspace));
    assert_eq!(h.story_lines, ["hello world"]);
    assert_eq!(h.story_line, 0);
    let input = widget::input(&world, story_panel::LINE_INPUT).unwrap();
    assert_eq!(input.caret, 5, "the caret sits at the join point");
    assert!(h.story_blur, "the control blurs for the join frame");
    assert!(
        !h.make_story_view([0.0, 0.0]).focus,
        "the view yields focus that frame"
    );
    // The next key frame clears the blur.
    h.story_keys(&mut world, &story_key_input(InputKey::Left));
    assert!(!h.story_blur);
}

// Up / Down commit the edited line and move; typed text is never lost.
#[test]
fn story_up_down_commit_and_navigate() {
    let (mut h, mut world) = story_session(&["one", "two", "three"]);
    widget::seed_field(&mut world, story_panel::LINE_INPUT, "ONE edited");
    h.story_keys(&mut world, &story_key_input(InputKey::Down));
    assert_eq!(h.story_lines[0], "ONE edited", "moving commits the edit");
    assert_eq!(h.story_line, 1);
    let input = widget::input(&world, story_panel::LINE_INPUT).unwrap();
    assert_eq!(input.content, "two");
    h.story_keys(&mut world, &story_key_input(InputKey::Up));
    assert_eq!(h.story_line, 0);
    // Up at the first line stays put.
    h.story_keys(&mut world, &story_key_input(InputKey::Up));
    assert_eq!(h.story_line, 0);
}

// Apply validates with the real story parser before writing: a broken story
// shows on the status line and the file is untouched; a valid one writes and
// refreshes the live preview without touching the world.jsonl dirty flag.
#[test]
fn story_apply_validates_then_writes() {
    let tree = concinnity_testing::TempTree::new();
    let path = tree.write("tale.md", story::STARTER_STORY);
    let src = path.to_string_lossy().to_string();

    let mut world = World::new();
    inject::editor_hud(&mut world);
    let mut h = hook(vec![story_import(&src)]);
    h.story_open = true;
    h.load_story(&mut world);
    assert_eq!(h.story_status, None);
    assert_eq!(h.story_path, src);
    assert!(h.story_lines.len() > 5, "the starter story loaded");

    // Break the story (no frontmatter): Apply rejects and writes nothing.
    h.story_lines = vec!["just prose, no frontmatter".to_string()];
    h.story_line = 0;
    h.seed_story_line(&mut world);
    h.apply_story(&mut world);
    assert!(h.story_status.is_some(), "parse failure shown");
    assert!(!h.rebuild_preview);
    let on_disk = std::fs::read_to_string(&path).unwrap();
    assert_eq!(on_disk, story::STARTER_STORY, "file untouched");

    // A valid edit writes and requests the preview rebuild; the world.jsonl
    // dirty flag stays clear (no entry changed).
    h.story_lines = story::lines_of(story::STARTER_STORY);
    let last = h.story_lines.len() - 1;
    h.story_line = last;
    h.seed_story_line(&mut world);
    widget::seed_field(&mut world, story_panel::LINE_INPUT, "And they lived on.");
    h.apply_story(&mut world);
    assert_eq!(h.story_status, None);
    assert!(h.rebuild_preview, "preview refresh requested");
    assert!(!h.dirty, "no world.jsonl change");
    let on_disk = std::fs::read_to_string(&path).unwrap();
    assert!(on_disk.contains("And they lived on."));
    assert!(on_disk.ends_with('\n'));
}

// A missing source file loads as an empty editable story with the error shown.
#[test]
fn story_load_missing_file_shows_status() {
    let mut world = World::new();
    inject::editor_hud(&mut world);
    let mut h = hook(vec![story_import("/no/such/dir/story.md")]);
    h.story_open = true;
    h.load_story(&mut world);
    assert!(h.story_status.is_some());
    assert_eq!(h.story_lines, [""]);
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

    let mut world = World::new();
    inject::editor_hud(&mut world);
    let mut h = hook(Vec::new());
    h.story_open = true;
    h.load_story(&mut world);
    assert!(
        h.make_story_view([0.0, 0.0]).create,
        "no import: create mode"
    );
    h.create_story(&mut world);

    assert_eq!(h.entries.len(), 1);
    assert_eq!(h.entries[0]["type"], "StoryImport");
    assert_eq!(h.entries[0]["args"]["source"], "story.md");
    assert!(h.dirty, "the new entry is a world edit");
    assert!(h.story_lines.len() > 5, "the starter story is loaded");
    assert!(!h.make_story_view([0.0, 0.0]).create);
    let written = std::fs::read_to_string(tree.join("story.md")).unwrap();
    assert_eq!(written, story::STARTER_STORY);
    // A second create is a no-op while an import exists.
    h.create_story(&mut world);
    assert_eq!(h.entries.len(), 1);

    std::env::set_current_dir(old).unwrap();
}

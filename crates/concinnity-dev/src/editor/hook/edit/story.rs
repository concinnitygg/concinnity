//! EditorHook: the Story panel's actions. The panel is a code text area over
//! the Markdown source file of the world's first `StoryImport` entry. Because
//! the cook reads story sources from disk, Apply (or the save shortcut)
//! VALIDATES the text with the real story parser and then writes the file (the
//! one editor action that persists outside SAVE -- an in-memory preview of a
//! file-backed source is impossible), then refreshes the live preview. The
//! area takes the frame's key events and pointer while it holds focus and the
//! panel is frontmost.

use concinnity_core::components::FrameInput;
use concinnity_core::ecs::World;

use crate::editor::hook::{EditorHook, entry_type, short_status};
use crate::editor::notify;
use crate::editor::panels::registry::PanelKey;
use crate::editor::panels::story;
use crate::editor::panels::story_panel::{self, StoryAction, StoryView};
use crate::editor::text_area::keys::Platform;
use crate::editor::text_area::layout::{Geometry, Metrics};
use crate::editor::text_area::{TextArea, clipboard};

impl EditorHook {
    // The `entries` index of the first StoryImport (the panel's subject).
    pub(super) fn story_import_index(&self) -> Option<usize> {
        self.entries
            .iter()
            .position(|e| entry_type(e) == Some("StoryImport"))
    }

    // The import's source path, if an import with a source exists.
    pub(super) fn story_source(&self) -> Option<String> {
        let idx = self.story_import_index()?;
        self.entries[idx]
            .get("args")
            .and_then(|a| a.get("source"))
            .and_then(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from)
    }

    // (Re)load the story from its source file: on panel open, after Create,
    // and never implicitly in between (in-progress edits are not clobbered).
    pub(in crate::editor::hook) fn load_story(&mut self) {
        self.story.status = None;
        self.story.markers.clear();
        self.story.focus = false;
        let text = match self.story_source() {
            Some(path) => {
                let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                    self.story.status = Some(short_status(&format!("{path}: {e}")));
                    String::new()
                });
                self.story.path = path;
                text
            }
            None => {
                self.story.path = String::new();
                String::new()
            }
        };
        self.story.area = TextArea::from_text(&text);
    }

    // Whether the text area holds the keyboard: focused by a press and the
    // panel frontmost.
    pub(in crate::editor::hook) fn story_typing(&self) -> bool {
        self.story.open && self.story.focus && self.panel_order.last() == Some(&PanelKey::Story)
    }

    // The text area laid out in the panel as it stands, with the view it shows
    // pushed into the area so its scrolling keeps the caret on screen.
    fn story_geometry(&mut self) -> Geometry {
        let vp = self.viewport;
        let o = self.origin(PanelKey::Story, vp);
        let s = self.effective_size(PanelKey::Story);
        let g = story_panel::area_geometry(o, s, &self.story.area, Metrics::code());
        self.story.area.set_view(g.view());
        g
    }

    pub(in crate::editor::hook) fn scroll_story(&mut self, delta: f32) {
        self.story_geometry();
        self.story.area.wheel(delta, self.shift_held);
    }

    pub(in crate::editor::hook) fn make_story_view(&self, mouse: [f32; 2]) -> StoryView<'_> {
        StoryView {
            area: &self.story.area,
            focus: self.story_typing(),
            path: &self.story.path,
            status: self.story.status.as_deref(),
            markers: &self.story.markers,
            create: self.story_import_index().is_none(),
            mouse,
        }
    }

    // Route a resolved Story-panel click at `(mx, my)`.
    pub(in crate::editor::hook) fn apply_story_action(
        &mut self,
        action: StoryAction,
        mx: f32,
        my: f32,
    ) {
        match action {
            StoryAction::Text => {
                let g = self.story_geometry();
                let now = self.clock.elapsed().as_secs_f64();
                self.story.area.press_at(&g, mx, my, self.shift_held, now);
                self.story.focus = true;
            }
            StoryAction::Create => self.create_story(),
            StoryAction::Apply => self.apply_story(),
            // A click on panel chrome blurs the text area.
            StoryAction::Consume => self.story.focus = false,
        }
    }

    // The per-frame input while the panel is frontmost: a held press keeps
    // dragging, and while the area holds the keyboard every key event of the
    // frame edits it, in order. The save shortcut applies.
    pub(in crate::editor::hook) fn story_keys(&mut self, world: &mut World, input: &FrameInput) {
        if self.story_import_index().is_none() {
            return;
        }
        let g = self.story_geometry();
        let area = &mut self.story.area;
        if area.pointer_busy() {
            area.pointer(&g, input.mouse_x, input.mouse_y, input.left_button_down);
        }
        if !self.story_typing() || input.key_events.is_empty() {
            return;
        }
        let clipboard = clipboard::system_or(world, &mut self.text_clipboard);
        let response =
            self.story
                .area
                .handle_events(&input.key_events, Platform::current(), clipboard);
        if response.save {
            self.apply_story();
        }
    }

    // Validate the story with the real parser and, only when it parses, write
    // the source file and refresh the live preview. The write persists to disk
    // immediately (unlike entry edits, which SAVE persists): the cook can only
    // read story sources from disk, so there is no in-memory-only preview for
    // them. world.jsonl is untouched, so the SAVE flag is not set.
    pub(in crate::editor::hook) fn apply_story(&mut self) {
        let Some(path) = self.story_source() else {
            return;
        };
        self.story.status = None;
        self.story.markers.clear();
        let content = self.story.area.text();
        if let Err(e) = concinnity_cook::build_only::validate_story_source(&content) {
            self.story.status = Some(short_status(&e));
            if let Some(marker) = story::error_marker(&e) {
                self.story.area.go_to(marker.line, 0);
                self.story.markers.push(marker);
            }
            self.notifier
                .error_with(&format!("Story rejected: {e}"), notify::Action::OpenConsole);
            return;
        }
        if let Err(e) = std::fs::write(&path, content) {
            self.story.status = Some(short_status(&format!("{path}: {e}")));
            self.notifier.error_with(
                &format!("Story write failed: {path}: {e}"),
                notify::Action::OpenConsole,
            );
            return;
        }
        self.notifier.success(&format!("Applied story {path}"));
        self.story.area.mark_saved();
        // The cook reads the story from disk, so nothing in the entry list
        // describes what changed: only a rebuild picks the new source up.
        self.require_rebuild();
    }

    // Write the starter story to a fresh file and add its StoryImport entry,
    // then load it for editing. The entry addition is a normal world edit
    // (dirty + live preview); the file itself is created immediately.
    pub(in crate::editor::hook) fn create_story(&mut self) {
        if self.story_import_index().is_some() {
            return;
        }
        let path = free_story_path();
        if let Err(e) = std::fs::write(&path, story::STARTER_STORY) {
            self.story.status = Some(short_status(&format!("{path}: {e}")));
            return;
        }
        let name = self.unique_name("story");
        self.entries.push(serde_json::json!({
            "type": "StoryImport", "args": { "$id": name, "source": path },
        }));
        self.mark_changed();
        self.load_story();
        self.story.focus = true;
    }
}

// The first unused `story.md` / `story_<n>.md` name in the project root (the
// directory story sources resolve against).
fn free_story_path() -> String {
    if !std::path::Path::new("story.md").exists() {
        return "story.md".to_string();
    }
    let mut i = 1;
    loop {
        let candidate = format!("story_{i}.md");
        if !std::path::Path::new(&candidate).exists() {
            return candidate;
        }
        i += 1;
    }
}

// src/editor/hook/story_edit.rs
//
// EditorHook: the Story panel's actions. The panel is a line editor over the
// Markdown source file of the world's first `StoryImport` entry (`story.rs`
// owns the text model). Because the cook reads story sources from disk, Apply
// VALIDATES the joined text with the real story parser and then writes the
// file (the one editor action that persists outside SAVE -- an in-memory
// preview of a file-backed source is impossible), then refreshes the live
// preview. Line editing rides the engine's single-line `TextInput`: the hook
// handles Enter (split), Up / Down (navigate), and Backspace at column 0
// (join) from the frame's captured key, delivered only while the panel is the
// frontmost open panel.

use super::*;
use crate::assets::Key;

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
    pub(super) fn load_story(&mut self, world: &mut World) {
        self.story_status = None;
        self.story_touched = false;
        self.story_line = 0;
        self.story_scroll = 0;
        self.story_focus = false;
        match self.story_source() {
            Some(path) => {
                match std::fs::read_to_string(&path) {
                    Ok(content) => self.story_lines = story::lines_of(&content),
                    Err(e) => {
                        self.story_lines = story::lines_of("");
                        self.story_status = Some(short_status(&format!("{path}: {e}")));
                    }
                }
                self.story_path = path;
            }
            None => {
                self.story_lines = story::lines_of("");
                self.story_path = String::new();
            }
        }
        self.seed_story_line(world);
    }

    // Seed the edit control with the current line. The control's injected
    // default caps text at the form fields' length; a story line has no such
    // limit, so the cap is lifted here (0 = unlimited).
    pub(super) fn seed_story_line(&mut self, world: &mut World) {
        let text = self
            .story_lines
            .get(self.story_line)
            .cloned()
            .unwrap_or_default();
        widget::seed_field(world, story_panel::LINE_INPUT, &text);
        if let Some(t) = widget::input_mut(world, story_panel::LINE_INPUT) {
            t.max_len = 0;
        }
    }

    // Fold the edit control's live text back into the current line.
    pub(super) fn commit_story_line(&mut self, world: &World) {
        if let Some(line) = self.story_lines.get_mut(self.story_line) {
            let text = widget::field_text(world, story_panel::LINE_INPUT);
            if *line != text {
                self.story_touched = true;
                *line = text;
            }
        }
    }

    // Move the edit line to `i` (committing the old line first) and keep it
    // inside the visible window.
    pub(super) fn set_story_line(&mut self, world: &mut World, i: usize) {
        self.commit_story_line(world);
        self.story_line = i.min(self.story_lines.len().saturating_sub(1));
        self.seed_story_line(world);
        self.story_focus = true;
        self.ensure_story_visible();
    }

    fn story_rows_shown(&self) -> usize {
        story_panel::visible_rows(self.effective_size(PanelKey::Story)[1])
    }

    fn ensure_story_visible(&mut self) {
        let rows_shown = self.story_rows_shown();
        if self.story_line < self.story_scroll {
            self.story_scroll = self.story_line;
        } else if self.story_line >= self.story_scroll + rows_shown {
            self.story_scroll = self.story_line + 1 - rows_shown;
        }
    }

    pub(super) fn scroll_story(&mut self, delta: f32) {
        let max = self
            .story_lines
            .len()
            .saturating_sub(self.story_rows_shown());
        self.story_scroll = scroll_step(self.story_scroll, delta, max);
    }

    pub(super) fn make_story_view(&self, mouse: [f32; 2]) -> StoryView<'_> {
        StoryView {
            lines: &self.story_lines,
            scroll: self.story_scroll,
            current: self.story_line,
            // Focus is asserted only while frontmost (see the lighting panel's
            // matching guard) and not in the one-frame blur after a line join.
            focus: self.story_focus
                && !self.story_blur
                && self.panel_order.last() == Some(&PanelKey::Story),
            path: &self.story_path,
            status: self.story_status.as_deref(),
            create: self.story_import_index().is_none(),
            dirty: self.story_touched,
            mouse,
        }
    }

    // Route a resolved Story-panel click.
    pub(super) fn apply_story_action(&mut self, action: StoryAction, world: &mut World) {
        match action {
            StoryAction::Line(i) => self.set_story_line(world, i),
            StoryAction::Create => self.create_story(world),
            StoryAction::Apply => self.apply_story(world),
            // A click on panel chrome blurs the edit line.
            StoryAction::Consume => self.story_focus = false,
        }
    }

    // The per-frame editing keys, while the panel is frontmost and its edit
    // line focused: Enter splits at the caret, Up / Down move lines, and
    // Backspace at column 0 joins with the previous line.
    pub(super) fn story_keys(&mut self, world: &mut World, input: &FrameInput) {
        self.story_blur = false;
        if !self.story_focus || self.story_import_index().is_none() {
            return;
        }
        let caret = widget::input(world, story_panel::LINE_INPUT)
            .map(|t| t.caret)
            .unwrap_or(0);
        match input.captured_key {
            Some(Key::Enter) => {
                self.commit_story_line(world);
                self.story_line = story::split_line(&mut self.story_lines, self.story_line, caret);
                self.seed_story_line(world);
                self.set_story_caret(world, 0);
                self.ensure_story_visible();
            }
            Some(Key::Up) if self.story_line > 0 => {
                self.set_story_line(world, self.story_line - 1);
            }
            Some(Key::Down) if self.story_line + 1 < self.story_lines.len() => {
                self.set_story_line(world, self.story_line + 1);
            }
            Some(Key::Backspace) if caret == 0 => {
                self.commit_story_line(world);
                if let Some((line, caret)) =
                    story::join_with_previous(&mut self.story_lines, self.story_line)
                {
                    self.story_line = line;
                    self.seed_story_line(world);
                    self.set_story_caret(world, caret);
                    self.ensure_story_visible();
                    // The text system processes this same Backspace after the
                    // tick; blurring the control for this frame keeps it from
                    // also deleting the character before the join point.
                    self.story_blur = true;
                }
            }
            _ => {}
        }
    }

    fn set_story_caret(&self, world: &mut World, caret: usize) {
        if let Some(t) = widget::input_mut(world, story_panel::LINE_INPUT) {
            t.caret = caret;
        }
    }

    // Validate the joined story with the real parser and, only when it parses,
    // write the source file and refresh the live preview. The write persists
    // to disk immediately (unlike entry edits, which SAVE persists): the cook
    // can only read story sources from disk, so there is no in-memory-only
    // preview for them. world.jsonl is untouched, so the SAVE flag is not set.
    pub(super) fn apply_story(&mut self, world: &mut World) {
        self.commit_story_line(world);
        let Some(path) = self.story_source() else {
            return;
        };
        self.story_status = None;
        let content = story::join_lines(&self.story_lines);
        if let Err(e) = concinnity_cook::world::validate_story_source(&content) {
            self.story_status = Some(short_status(&e));
            self.notifier
                .error_with(&format!("Story rejected: {e}"), notify::Action::OpenConsole);
            return;
        }
        if let Err(e) = std::fs::write(&path, content) {
            self.story_status = Some(short_status(&format!("{path}: {e}")));
            self.notifier.error_with(
                &format!("Story write failed: {path}: {e}"),
                notify::Action::OpenConsole,
            );
            return;
        }
        self.notifier.success(&format!("Applied story {path}"));
        self.story_touched = false;
        self.rebuild_preview = true;
    }

    // Write the starter story to a fresh file and add its StoryImport entry,
    // then load it for editing. The entry addition is a normal world edit
    // (dirty + live preview); the file itself is created immediately.
    pub(super) fn create_story(&mut self, world: &mut World) {
        if self.story_import_index().is_some() {
            return;
        }
        let path = free_story_path();
        if let Err(e) = std::fs::write(&path, story::STARTER_STORY) {
            self.story_status = Some(short_status(&format!("{path}: {e}")));
            return;
        }
        let name = self.unique_name("story");
        self.entries.push(serde_json::json!({
            "name": name, "type": "StoryImport", "args": { "source": path },
        }));
        self.mark_changed();
        self.load_story(world);
        self.story_focus = true;
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

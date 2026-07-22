// src/editor/hook/console_edit.rs
//
// EditorHook: the Console panel's actions. The backtick toggle (with a
// one-frame focus blur so the opening keypress never types into the command
// line), the log window's pinned-tail scroll, the /del ghost autocomplete, and
// the command dispatch: /add and /del mutate the working entries exactly like
// their panel counterparts, the build command compiles the in-memory entries
// on a worker thread (in-process -- a separate `cn build` process would leave
// this editor's name table empty), and everything reports through the shared
// log sink.

use super::*;
use crate::assets::Key;
use std::sync::atomic::Ordering;

impl EditorHook {
    // Open / close the console (the View row, the title X, and backtick all
    // funnel here). Opening focuses a cleared command line and lifts the
    // injected form-field length cap (inline JSON runs long); closing
    // releases focus.
    pub(super) fn toggle_console(&mut self, world: &mut World) {
        self.console_open = !self.console_open;
        if self.console_open {
            self.console_focus = true;
            self.console_pinned = true;
            widget::seed_field(world, console_panel::INPUT, "");
            if let Some(t) = widget::input_mut(world, console_panel::INPUT) {
                t.max_len = 0;
            }
            self.focus_panel(PanelKey::Console);
        } else {
            self.console_focus = false;
        }
    }

    // The backtick edge: toggle unless play mode owns the keyboard or another
    // text field is being typed into (a backtick there is just a character).
    // The console's own focus does not stand in the way, so backtick still
    // closes an open console mid-typing.
    pub(super) fn drive_console_toggle(&mut self, input: &FrameInput, world: &mut World) {
        if input.captured_key != Some(Key::Backtick) || input.ctrl {
            return;
        }
        if self.world_capture || self.non_console_text_focus() {
            return;
        }
        self.toggle_console(world);
        if self.console_open {
            // The same keypress delivers a '`' typed_char after this tick;
            // one unfocused frame keeps it out of the fresh command line.
            self.console_blur = true;
        }
    }

    // The visible log window under the current scroll: (lines, total, first).
    // Pinned follows the tail, so new lines auto-scroll into view.
    pub(super) fn console_window(&self) -> (Vec<console::ConsoleLine>, usize, usize) {
        let total = self.console_sink.len();
        let max_first = total.saturating_sub(console_panel::LINE_POOL);
        let first = if self.console_pinned {
            max_first
        } else {
            self.console_scroll.min(max_first)
        };
        (
            self.console_sink.window(first, console_panel::LINE_POOL),
            total,
            first,
        )
    }

    pub(super) fn make_console_view<'a>(
        &self,
        lines: &'a [console::ConsoleLine],
        total: usize,
        first: usize,
        ghost: &'a str,
        mouse: [f32; 2],
    ) -> ConsoleView<'a> {
        ConsoleView {
            lines,
            total,
            first,
            // Focus is asserted only while frontmost (see the other panels'
            // matching guard) and not in the one-frame blur after a backtick
            // open.
            focus: self.console_focus
                && !self.console_blur
                && self.panel_order.last() == Some(&PanelKey::Console),
            ghost,
            mouse,
        }
    }

    // The /del name argument's inline completion suffix for the current input
    // text, matched against the authored entry names.
    pub(super) fn console_ghost(&self, world: &World) -> String {
        let text = widget::field_text(world, console_panel::INPUT);
        console::del_ghost(&text, self.entries.iter().filter_map(entry_name)).unwrap_or_default()
    }

    // Step the log window; landing on the last line re-pins it to the tail.
    pub(super) fn scroll_console(&mut self, delta: f32) {
        let max = self
            .console_sink
            .len()
            .saturating_sub(console_panel::LINE_POOL);
        let cur = if self.console_pinned {
            max
        } else {
            self.console_scroll.min(max)
        };
        let next = scroll_step(cur, delta, max);
        self.console_scroll = next;
        self.console_pinned = next >= max;
    }

    pub(super) fn apply_console_action(&mut self, action: ConsoleAction, _world: &mut World) {
        match action {
            ConsoleAction::FocusInput => self.console_focus = true,
            // A click on panel chrome blurs the command line.
            ConsoleAction::Consume => self.console_focus = false,
        }
    }

    // The per-frame editing keys while the command line is focused: Enter
    // submits, Tab (or Right at the caret's end) accepts the ghost.
    pub(super) fn console_keys(&mut self, world: &mut World, input: &FrameInput) {
        if !self.console_focus {
            return;
        }
        match input.captured_key {
            Some(Key::Enter) => {
                let line = widget::field_text(world, console_panel::INPUT);
                widget::seed_field(world, console_panel::INPUT, "");
                let line = line.trim();
                if !line.is_empty() {
                    self.run_console_line(line);
                }
            }
            Some(Key::Tab) => self.accept_console_ghost(world),
            Some(Key::Right) => {
                let at_end = widget::input(world, console_panel::INPUT)
                    .map(|t| t.caret >= t.content.chars().count())
                    .unwrap_or(false);
                if at_end {
                    self.accept_console_ghost(world);
                }
            }
            _ => {}
        }
    }

    fn accept_console_ghost(&mut self, world: &mut World) {
        let text = widget::field_text(world, console_panel::INPUT);
        if let Some(ghost) = console::del_ghost(&text, self.entries.iter().filter_map(entry_name)) {
            widget::focus_field_with(world, console_panel::INPUT, &format!("{text}{ghost}"));
        }
    }

    // Echo and dispatch one submitted line. Submitting re-pins the log to the
    // tail so the response is visible.
    pub(super) fn run_console_line(&mut self, line: &str) {
        self.console_sink
            .push(console::Severity::Command, &format!("> {line}"));
        self.console_pinned = true;
        match console::parse_command(line) {
            // The echo above is the whole behaviour of a bare line.
            Ok(console::Command::Echo(_)) => {}
            Ok(console::Command::Add { target, name }) => {
                self.console_add(&target, name.as_deref());
            }
            Ok(console::Command::Del { name }) => self.console_del(&name),
            Ok(console::Command::Cook) => self.console_build(),
            Ok(console::Command::Help) => {
                for l in console::help_lines() {
                    self.console_sink.info(&l);
                }
            }
            Err(e) => self.console_sink.error(&e),
        }
    }

    // /add: resolve the target exactly as `cn add` does and append the entries
    // to the working list (renamed to stay unique), like the Import panel's
    // Add. Scene entries defer reading their file to the cook, so a missing
    // path is caught here instead of surfacing as a preview-rebuild log.
    fn console_add(&mut self, target: &str, name: Option<&str>) {
        if crate::authoring::is_path_like(target) && !std::path::Path::new(target).is_file() {
            self.console_sink.error(&format!("{target}: no such file"));
            return;
        }
        let mut new_entries = match crate::authoring::resolve_add_target(target) {
            Ok(entries) => entries,
            Err(e) => {
                self.console_sink.error(&e.to_string());
                return;
            }
        };
        if let Some(n) = name {
            crate::authoring::apply_name_override(&mut new_entries, n);
        }
        // An `.hdr` into a world that already has a lighting environment
        // retargets it rather than appending a second one the runtime would
        // ignore, matching `cn add <file>`.
        if let Some((name, source)) =
            crate::authoring::try_retarget_environment_map(&mut self.entries, &new_entries)
        {
            self.console_sink.info(&format!("{name} now uses {source}"));
            self.mark_changed();
            return;
        }
        for entry in &mut new_entries {
            let base = entry_name(entry).unwrap_or("asset").to_string();
            let unique = self.unique_from(&base);
            entry["name"] = serde_json::Value::String(unique.clone());
            let ty = entry_type(entry).unwrap_or("?").to_string();
            self.entries.push(entry.clone());
            self.console_sink.info(&format!("added '{unique}' ({ty})"));
        }
        self.mark_changed();
    }

    // /del: remove the authored entry by name, with the same open-form index
    // fixups as the browse list's Delete.
    fn console_del(&mut self, name: &str) {
        let Some(idx) = self
            .entries
            .iter()
            .position(|e| entry_name(e) == Some(name))
        else {
            self.console_sink
                .error(&format!("no authored asset named '{name}'"));
            return;
        };
        let ty = entry_type(&self.entries[idx]).unwrap_or("?").to_string();
        self.entries.remove(idx);
        self.mark_changed();
        match self.form_target {
            FormTarget::Entry(e) if e == idx => self.close_form(),
            FormTarget::Entry(e) if e > idx => self.form_target = FormTarget::Entry(e - 1),
            _ => {}
        }
        self.row_menu = None;
        self.console_sink.info(&format!("removed '{name}' ({ty})"));
    }

    // The build command ("cook" in user-facing text): compile the in-memory
    // entries and write the blobs on a worker thread, reporting start /
    // finish / errors through the sink so the frame loop never stalls. One
    // build at a time; SAVE also stands down while one is writing blobs.
    fn console_build(&mut self) {
        if self.console_build_running.swap(true, Ordering::SeqCst) {
            self.console_sink.warn("cook already running");
            return;
        }
        let content = match crate::world::write_world_jsonl(&self.entries) {
            Ok(c) => c,
            Err(e) => {
                self.console_build_running.store(false, Ordering::SeqCst);
                self.console_sink.error(&format!("cook failed: {e}"));
                return;
            }
        };
        let sink = self.console_sink.clone();
        let running = self.console_build_running.clone();
        sink.info("cook started");
        std::thread::spawn(move || {
            let start = std::time::Instant::now();
            match crate::authoring::build_world_str_to_disk(&content) {
                Ok(()) => sink.info(&format!(
                    "cook finished in {:.1}s",
                    start.elapsed().as_secs_f32()
                )),
                Err(e) => sink.error(&format!("cook failed: {e}")),
            }
            running.store(false, Ordering::SeqCst);
        });
    }
}

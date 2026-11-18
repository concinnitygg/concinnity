//! `EditorHook`: the Console panel's actions. The backtick toggle (with a
//! one-frame focus blur so the opening keypress never types into the command
//! line), the log window's pinned-tail scroll, the /del ghost autocomplete, and
//! the command dispatch: /add and /del mutate the working entries exactly like
//! their panel counterparts, the build command compiles the in-memory entries
//! on a worker thread (in-process -- a separate `cn cook` process would leave
//! this editor's name table empty), and everything reports through the shared
//! log sink.

use concinnity_cook::authoring::world::{entry_handle, find_entry, set_entry_id};
use concinnity_engine::components::{FrameInput, InputKey, TextField};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;
use std::sync::atomic::Ordering;

use crate::editor::build_failure;
use crate::editor::entry_list::build_text;
use crate::editor::hook::{EditorHook, declared_id, entry_type_name};
use crate::editor::panels::console;
use crate::editor::panels::console_panel::{self, ConsoleView};
use crate::editor::panels::registry::PanelKey;
use crate::editor::ui::{BuiltPanel, BuiltUi, field};
use crate::editor::viewport::snap;

impl EditorHook {
    // The command line's id in `world`.
    pub(in crate::editor::hook) fn console_input(world: &World) -> Option<AssetId> {
        world
            .resource::<BuiltUi>()
            .and_then(|b| b.panel(PanelKey::Console))
            .and_then(console_panel::input)
    }

    // Open / close the console (the View row, the title X, and backtick all
    // funnel here). Opening focuses a cleared command line and lifts the
    // field's length cap (inline JSON runs long); closing releases focus.
    pub(in crate::editor::hook) fn toggle_console(&mut self, world: &mut World) {
        self.open[PanelKey::Console] = !self.open[PanelKey::Console];
        if self.open[PanelKey::Console] {
            self.console.focus = true;
            self.console.pinned = true;
            let input = Self::console_input(world);
            field::set_text(world, input, "");
            if let Some(t) = input.and_then(|id| world.get_mut_by_id::<TextField>(id)) {
                t.max_len = 0;
            }
            self.focus_panel(PanelKey::Console);
        } else {
            self.console.focus = false;
        }
    }

    // The backtick edge: toggle unless play mode owns the keyboard or another
    // text field is being typed into (a backtick there is just a character).
    // The console's own focus does not stand in the way, so backtick still
    // closes an open console mid-typing.
    pub(in crate::editor::hook) fn drive_console_toggle(
        &mut self,
        input: &FrameInput,
        world: &mut World,
    ) {
        if !input.pressed_fresh(InputKey::Backtick) || input.ctrl {
            return;
        }
        if self.sim.playing() || self.non_console_text_focus() {
            return;
        }
        self.toggle_console(world);
        if self.open[PanelKey::Console] {
            // The same keypress queues a '`' character too; one unfocused
            // frame keeps it out of the fresh command line.
            self.console.blur = true;
        }
    }

    // What the console shows this frame. The command line holds focus only
    // while the console is frontmost (see the other panels' matching guard)
    // and not in the one-frame blur after a backtick open.
    pub(in crate::editor::hook) fn console_view<'a>(&self, ghost: &'a str) -> ConsoleView<'a> {
        ConsoleView {
            pushed: self.console_sink.pushed(),
            pinned: self.console.pinned,
            focus: self.console.focus
                && !self.console.blur
                && self.panel_order.last() == Some(&PanelKey::Console),
            ghost,
        }
    }

    // Take in what the user did to the console since last frame: a click that
    // focused or left the command line (its focus differs from what was last
    // shown), and a scroll away from the log's tail or back onto it.
    pub(in crate::editor::hook) fn observe_console(&mut self, world: &World, built: &BuiltPanel) {
        let focused = console_panel::input(built)
            .and_then(|id| world.get_by_id::<TextField>(id))
            .is_some_and(|f| f.focused);
        if focused != self.console_view("").focus {
            self.console.focus = focused;
        }
        self.console.pinned = console_panel::at_tail(world, built);
    }

    // The /del completion for the command line as it stands, borrowed from the
    // authored name it completes to.
    pub(in crate::editor::hook) fn console_ghost_now(&self, world: &World) -> &str {
        let line = Self::console_input(world)
            .and_then(|id| world.get_by_id::<TextField>(id))
            .map_or("", |f| f.content.as_str());
        console::del_ghost(line, self.entries.iter().filter_map(declared_id)).unwrap_or_default()
    }

    // The /del name argument's inline completion suffix for the current input
    // text, matched against the authored entry names.
    #[cfg(test)]
    pub(in crate::editor::hook) fn console_ghost(&self, world: &World) -> String {
        let text = field::text(world, Self::console_input(world));
        console::del_ghost(&text, self.entries.iter().filter_map(declared_id))
            .unwrap_or_default()
            .to_string()
    }

    // The per-frame editing keys while the command line is focused: Enter
    // submits, Tab (or Right at the caret's end) accepts the ghost.
    pub(in crate::editor::hook) fn console_keys(&mut self, world: &mut World, input: &FrameInput) {
        if !self.console.focus {
            return;
        }
        let field = Self::console_input(world);
        for press in input.key_presses() {
            match press.key {
                // A held Enter submits once.
                InputKey::Enter if !press.repeat => {
                    let line = field::text(world, field);
                    field::set_text(world, field, "");
                    let line = line.trim().to_string();
                    if !line.is_empty() {
                        self.run_console_line(world, &line);
                    }
                }
                InputKey::Tab => self.accept_console_ghost(world),
                InputKey::Right => {
                    let at_end = field
                        .and_then(|id| world.get_by_id::<TextField>(id))
                        .is_some_and(|t| t.caret >= t.content.chars().count());
                    if at_end {
                        self.accept_console_ghost(world);
                    }
                }
                _ => {}
            }
        }
    }

    fn accept_console_ghost(&self, world: &mut World) {
        let field = Self::console_input(world);
        let text = field::text(world, field);
        if let Some(ghost) = console::del_ghost(&text, self.entries.iter().filter_map(declared_id))
        {
            field::set_text(world, field, &format!("{text}{ghost}"));
        }
    }

    // Echo and dispatch one submitted line. Submitting re-pins the log to the
    // tail so the response is visible.
    pub(in crate::editor::hook) fn run_console_line(&mut self, world: &mut World, line: &str) {
        self.console_sink
            .push(console::Severity::Command, &format!("> {line}"));
        self.console.pinned = true;
        match console::parse_command(line) {
            // The echo above is the whole behavior of a bare line.
            Ok(console::Command::Echo(_)) => {}
            Ok(console::Command::Add { target, name }) => {
                self.console_add(&target, name.as_deref());
            }
            Ok(console::Command::Del { name }) => self.console_del(&name),
            Ok(console::Command::Cook) => self.console_build(),
            Ok(console::Command::Snap(cmd)) => self.console_snap(cmd),
            Ok(console::Command::Dup) => {
                let made = self.duplicate_selection();
                self.console_sink.info(&format!("duplicated {made}"));
            }
            Ok(console::Command::Floor) => {
                let moved = self.drop_selection_to_floor(world);
                self.console_sink.info(&format!("dropped {moved}"));
            }
            Ok(console::Command::Select(cmd)) => self.console_select(cmd, world),
            Ok(console::Command::Export { name, bake }) => {
                self.console_export(name.as_deref(), bake);
            }
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
            crate::authoring::apply_id_override(&mut new_entries, n);
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
            if let Some(base) = declared_id(entry).map(str::to_string) {
                set_entry_id(entry, &self.unique_from(&base));
            }
            let ty = entry_type_name(entry).unwrap_or("?").to_string();
            self.entries.push(entry.clone());
            let handle = entry_handle(&self.entries, self.entries.len() - 1).unwrap_or_default();
            self.console_sink.info(&format!("added '{handle}' ({ty})"));
        }
        self.mark_changed();
    }

    // /del: remove the authored entry by its handle (a `$id`, or the label of
    // an anonymous entry), with the same open-form index fixups as the browse
    // list's Delete.
    fn console_del(&mut self, name: &str) {
        let Some(idx) = find_entry(&self.entries, name) else {
            self.console_sink
                .error(&format!("no authored asset named '{name}'"));
            return;
        };
        let ty = entry_type_name(&self.entries[idx])
            .unwrap_or("?")
            .to_string();
        self.remove_entry_at(idx);
        self.assets.row_menu = None;
        self.console_sink.info(&format!("removed '{name}' ({ty})"));
    }

    // /snap: adjust the gizmo snap settings (session state, mirrored by the
    // Preview panel rows) and report the resulting state.
    fn console_snap(&mut self, cmd: console::SnapCmd) {
        use console::{SnapCmd, SnapSet};
        fn set(snap: &mut snap::Snap, s: SnapSet) {
            match s {
                SnapSet::Enable(on) => snap.enabled = on,
                SnapSet::Step(step) => {
                    snap.step = step;
                    snap.enabled = true;
                }
            }
        }
        match cmd {
            SnapCmd::Status => {}
            SnapCmd::All(on) => {
                self.snap.translate.enabled = on;
                self.snap.rotate.enabled = on;
            }
            SnapCmd::Move(s) => set(&mut self.snap.translate, s),
            SnapCmd::Rotate(s) => set(&mut self.snap.rotate, s),
        }
        self.console_sink.info(&self.snap.describe());
    }

    // The build command ("cook" in user-facing text): compile the in-memory
    // entries and write the blobs on a worker thread, reporting start /
    // finish / errors through the sink so the frame loop never stalls. One
    // build at a time, shared with /export.
    fn console_build(&self) {
        if self.console_build_running.swap(true, Ordering::SeqCst) {
            self.console_sink.warn("cook already running");
            return;
        }
        let content = match build_text(&self.entries) {
            Ok(c) => c,
            Err(e) => {
                self.console_build_running.store(false, Ordering::SeqCst);
                self.console_sink.error(&format!("cook failed: {e}"));
                return;
            }
        };
        let sink = self.console_sink.clone();
        let toasts = self.notifier.clone();
        sink.info("cook started");
        self.spawn_cook_worker("Cooking", content, move |outcome, secs| match outcome {
            Ok(()) => {
                sink.info(&format!("cook finished in {secs:.1}s"));
                toasts.success(&format!("Cook finished in {secs:.1}s"));
            }
            Err(e) => {
                let failure = build_failure::report("Cook", &e);
                for line in &failure.lines {
                    sink.error(line);
                }
                toasts.error_with(&failure.toast, failure.action);
            }
        });
    }
}

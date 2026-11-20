//! `EditorHook`: the command palette's drive. Ctrl+K toggles it (with a one-frame
//! focus blur so the opening press never types into the query); the item list
//! is built from the providers on open; the query is mirrored off the field
//! once a frame, re-ranking through `palette::matches`. A committed row acts
//! through the editor's existing paths only: panel toggles, the selection (plus
//! the framing glide), the Behavior panel's open path, console dispatch, and
//! the Display menu's state. The palette closes on commit, on Escape, and on a
//! click outside it.

use concinnity_engine::components::{FrameInput, InputKey};
use concinnity_engine::ecs::World;

use crate::editor::behavior::navigate;
use crate::editor::hook::EditorHook;
use crate::editor::palette::panel::{self, PaletteView};
use crate::editor::palette::{PaletteAction, providers};
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::ui::BuiltPanel;

impl EditorHook {
    // Toggle the palette on Ctrl+K or Cmd+K unless play mode owns the keyboard;
    // the chord types nothing, so other fields do not block it.
    pub(in crate::editor::hook) fn drive_palette_toggle(&mut self, input: &FrameInput) {
        if !input.pressed_fresh(InputKey::K) || !(input.ctrl || input.cmd) || self.sim.playing() {
            return;
        }
        self.toggle_palette();
        if self.open[PanelKey::Palette] {
            // The same keypress may queue a typed character too; one unfocused
            // frame keeps it out of the fresh query.
            self.palette.blur = true;
        }
    }

    // Open / close (the shortcut and the title X both funnel here). Opening
    // rebuilds the item list from the current world, clears the query, and
    // fronts the panel.
    pub(in crate::editor::hook) fn toggle_palette(&mut self) {
        if self.open[PanelKey::Palette] {
            self.close_palette();
            return;
        }
        self.open[PanelKey::Palette] = true;
        // The asset provider reads the cooked tree; bring it up to date first.
        self.refresh_tree_if_needed();
        self.palette.items = providers::all_items(&self.assets.groups);
        self.palette.query = String::new();
        self.palette.rerank();
        self.focus_panel(PanelKey::Palette);
    }

    pub(in crate::editor::hook) fn close_palette(&mut self) {
        self.open[PanelKey::Palette] = false;
    }

    // Take in what was typed into the query.
    pub(in crate::editor::hook) fn observe_palette(&mut self, world: &World, built: &BuiltPanel) {
        let Some(typed) = panel::typed(world, built) else {
            return;
        };
        if typed != self.palette.query {
            self.palette.query = typed.to_string();
            self.palette.rerank();
        }
    }

    pub(in crate::editor::hook) fn make_palette_view(&self) -> PaletteView<'_> {
        PaletteView {
            items: &self.palette.items,
            matches: &self.palette.matches,
            query: &self.palette.query,
            pick: self.palette.pick,
            // Focus is asserted only while frontmost (matching the other
            // panels' guard) and not in the one-frame blur after the open.
            focus: self.open[PanelKey::Palette]
                && !self.palette.blur
                && self.panel_order.last() == Some(&PanelKey::Palette),
            revision: self.palette.revision,
            moved: self.palette.moved,
        }
    }

    // The per-frame editing keys: Up / Down move the highlight, Enter commits.
    // Typing goes to the field; Escape closes through the global escape drive.
    pub(in crate::editor::hook) fn palette_keys(&mut self, world: &mut World, input: &FrameInput) {
        for press in input.key_presses() {
            let key = press.key;
            match key {
                InputKey::Enter if !press.repeat => self.commit_palette(world),
                InputKey::Up | InputKey::Down => {
                    let delta = if key == InputKey::Up { -1 } else { 1 };
                    let total = self.palette.matches.len();
                    let Some(at) = navigate::step(Some(self.palette.pick), delta, total) else {
                        continue;
                    };
                    self.palette.highlight(at);
                }
                _ => {}
            }
        }
    }

    // A click on a result row commits it.
    pub(in crate::editor::hook) fn pick_palette_row(&mut self, row: usize, world: &mut World) {
        if let Some(&at) = self.palette.matches.get(row) {
            self.commit_palette_item(at, world);
        }
    }

    // Enter: a query carrying a command line with arguments dispatches as
    // typed; anything else commits the highlighted row.
    fn commit_palette(&mut self, world: &mut World) {
        let query = self.palette.query.trim().to_string();
        if query.starts_with('/') && query.contains(char::is_whitespace) {
            self.close_palette();
            self.dispatch_palette_command(world, &query);
            return;
        }
        let Some(&at) = self.palette.matches.get(self.palette.pick) else {
            return;
        };
        self.commit_palette_item(at, world);
    }

    fn commit_palette_item(&mut self, at: usize, world: &mut World) {
        let item = self.palette.items[at].clone();
        self.palette.note_recent(item.label);
        self.apply_palette_action(item.action, world);
    }

    fn apply_palette_action(&mut self, action: PaletteAction, world: &mut World) {
        match action {
            PaletteAction::OpenPanel(key) => {
                self.close_palette();
                if !registry::is_open(key, self) {
                    registry::toggle(key, self, world);
                }
                self.focus_panel(key);
            }
            PaletteAction::SelectEntity(name) => {
                self.close_palette();
                self.select_named(&name);
                self.pick_last = None;
                self.focus_ui_on(&name);
                self.frame_selection(self.viewport, world);
            }
            PaletteAction::OpenAsset(name) => {
                self.close_palette();
                self.open_behavior_named(&name, world);
            }
            // Stays open: the reseeded query re-ranks on next frame's sample.
            PaletteAction::CommandMode(name) => {
                self.palette.query = format!("/{name} ");
                self.palette.rerank();
            }
            PaletteAction::RunCommand(line) => {
                self.close_palette();
                self.dispatch_palette_command(world, &line);
            }
            PaletteAction::SetOption(row) => {
                self.close_palette();
                self.apply_display_row(row);
            }
        }
    }

    // Dispatch through the console so the palette needs no command logic of
    // its own; the console opens first so the reply is visible.
    fn dispatch_palette_command(&mut self, world: &mut World, line: &str) {
        if !self.open[PanelKey::Console] {
            self.toggle_console(world);
        }
        self.run_console_line(world, line);
    }
}

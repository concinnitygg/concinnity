// src/editor/hook/palette_edit.rs
//
// EditorHook: the command palette's drive. Ctrl+K toggles it (with a one-frame
// focus blur so the opening press never types into the query); the item list
// is built from the providers on open; the query is mirrored off the field
// once a frame, re-ranking through `palette::matches`. A committed row acts
// through the editor's existing paths only: panel toggles, the selection (plus
// the framing glide), the Behavior panel's open path, console dispatch, and
// the Display menu's state. The palette closes on commit, on Escape, and on a
// click outside it.

use super::*;
use crate::assets::Key;
use crate::editor::behavior::navigate;
use crate::editor::palette::{PaletteAction, providers};

// How many result rows a wheel step or arrow keeps in view.
const WINDOW: usize = palette_panel::ROW_POOL;

impl EditorHook {
    // The Ctrl+K edge: toggle unless play mode owns the keyboard. Other text
    // fields do not stand in the way (Ctrl+K types nothing), so the palette
    // opens from anywhere in edit mode.
    pub(super) fn drive_palette_toggle(&mut self, input: &FrameInput, world: &mut World) {
        if input.captured_key != Some(Key::K) || !input.ctrl || self.sim.playing() {
            return;
        }
        self.toggle_palette(world);
        if self.palette_open {
            // The same keypress may deliver a typed_char after this tick; one
            // unfocused frame keeps it out of the fresh query.
            self.palette_blur = true;
        }
    }

    // Open / close (the shortcut and the title X both funnel here). Opening
    // rebuilds the item list from the current world, clears the query, and
    // fronts the panel.
    pub(super) fn toggle_palette(&mut self, world: &mut World) {
        if self.palette_open {
            self.close_palette();
            return;
        }
        self.palette_open = true;
        // The asset provider reads the cooked tree; bring it up to date first.
        self.refresh_tree_if_needed();
        self.palette_items = providers::all_items(&self.tree_groups);
        self.palette_query = String::new();
        widget::seed_field(world, palette_panel::INPUT, "");
        self.rerank_palette();
        self.focus_panel(PanelKey::Palette);
    }

    pub(super) fn close_palette(&mut self) {
        self.palette_open = false;
    }

    // Read the query off its field. Mirrored onto the hook because the data a
    // press and a draw resolve against is built without world access.
    pub(super) fn sample_palette_query(&mut self, world: &World) {
        if !self.palette_open {
            return;
        }
        let typed = widget::field_text(world, palette_panel::INPUT);
        if typed != self.palette_query {
            self.palette_query = typed;
            self.rerank_palette();
        }
    }

    // A changed query is a different list, so the highlight starts again at
    // its best answer.
    fn rerank_palette(&mut self) {
        self.palette_matches = palette::matches(
            &self.palette_items,
            &self.palette_recent,
            &self.palette_query,
        );
        self.palette_pick = 0;
        self.palette_scroll = 0;
    }

    // A click outside an open palette dismisses it and claims the press, so
    // dismissal can never also pick or drag something underneath.
    pub(super) fn route_palette_dismiss(&mut self, input: &FrameInput, vp: [f32; 2]) -> bool {
        if !self.palette_open {
            return false;
        }
        let o = self.origin(PanelKey::Palette, vp);
        let over = point_in(
            input.mouse_x,
            input.mouse_y,
            widget::outer_rect(o, palette_panel::size()),
        );
        if over {
            return false;
        }
        self.close_palette();
        true
    }

    pub(super) fn make_palette_view(&self, mouse: [f32; 2]) -> PaletteView<'_> {
        let rows = self
            .palette_matches
            .iter()
            .skip(self.palette_scroll)
            .take(WINDOW)
            .map(|&at| {
                let item = &self.palette_items[at];
                palette_panel::PaletteRow {
                    caption: &item.label,
                    hint: &item.hint,
                    tag: item.category.tag(),
                }
            })
            .collect();
        PaletteView {
            rows,
            selected: self.palette_pick,
            scroll: self.palette_scroll,
            total: self.palette_matches.len(),
            // Focus is asserted only while frontmost (matching the other
            // panels' guard) and not in the one-frame blur after the open.
            focus: self.palette_open
                && !self.palette_blur
                && self.panel_order.last() == Some(&PanelKey::Palette),
            mouse,
        }
    }

    pub(super) fn scroll_palette(&mut self, delta: f32) {
        let max = self.palette_matches.len().saturating_sub(WINDOW);
        self.palette_scroll = scroll_step(self.palette_scroll, delta, max);
    }

    // The per-frame editing keys: Up / Down move the highlight, Enter commits.
    // Typing goes to the field; Escape closes through the global escape drive.
    pub(super) fn palette_keys(&mut self, world: &mut World, input: &FrameInput) {
        match input.captured_key {
            Some(Key::Enter) => self.commit_palette(world),
            Some(key @ (Key::Up | Key::Down)) => {
                let delta = if key == Key::Up { -1 } else { 1 };
                let total = self.palette_matches.len();
                let Some(at) = navigate::step(Some(self.palette_pick), delta, total) else {
                    return;
                };
                self.palette_pick = at;
                self.palette_scroll = navigate::scroll_to(at, self.palette_scroll, WINDOW);
            }
            _ => {}
        }
    }

    pub(super) fn apply_palette_hit(&mut self, hit: PaletteHit, world: &mut World) {
        match hit {
            // The input holds focus while the palette is open; nothing to do.
            PaletteHit::FocusInput | PaletteHit::Consume => {}
            PaletteHit::Row(slot) => {
                if let Some(&at) = self.palette_matches.get(self.palette_scroll + slot) {
                    self.commit_palette_item(at, world);
                }
            }
        }
    }

    // Enter: a query carrying a command line with arguments dispatches as
    // typed; anything else commits the highlighted row.
    fn commit_palette(&mut self, world: &mut World) {
        let query = self.palette_query.trim().to_string();
        if query.starts_with('/') && query.contains(char::is_whitespace) {
            self.close_palette();
            self.dispatch_palette_command(world, &query);
            return;
        }
        let Some(&at) = self.palette_matches.get(self.palette_pick) else {
            return;
        };
        self.commit_palette_item(at, world);
    }

    fn commit_palette_item(&mut self, at: usize, world: &mut World) {
        let item = self.palette_items[at].clone();
        self.note_recent(item.label);
        self.apply_palette_action(item.action, world);
    }

    fn note_recent(&mut self, label: String) {
        self.palette_recent.retain(|l| l != &label);
        self.palette_recent.insert(0, label);
        self.palette_recent.truncate(palette::RECENT_CAP);
    }

    fn apply_palette_action(&mut self, action: PaletteAction, world: &mut World) {
        match action {
            PaletteAction::OpenPanel(key) => {
                self.close_palette();
                if !registry::panel(key).is_open(self) {
                    registry::panel(key).toggle(self, world);
                }
                self.focus_panel(key);
            }
            PaletteAction::SelectEntity(name) => {
                self.close_palette();
                self.selection.replace(name.clone());
                self.pick_last = None;
                self.focus_ui_on(&name, world);
                self.frame_selection(self.viewport, world);
            }
            PaletteAction::OpenAsset(name) => {
                self.close_palette();
                self.open_behavior_named(&name, world);
            }
            // Stays open: the reseeded query re-ranks on next frame's sample.
            PaletteAction::CommandMode(name) => {
                widget::focus_field_with(world, palette_panel::INPUT, &format!("/{name} "));
            }
            PaletteAction::RunCommand(line) => {
                self.close_palette();
                self.dispatch_palette_command(world, &line);
            }
            PaletteAction::SetOption(row) => {
                self.close_palette();
                match row {
                    view_menu::MenuRow::Mode(m) => self.view_mode = m,
                    view_menu::MenuRow::Heading(_) => {}
                    view_menu::MenuRow::Flag(f, _) => self.show_flags = self.show_flags.toggled(f),
                    view_menu::MenuRow::Billboards => self.show_billboards = !self.show_billboards,
                    view_menu::MenuRow::Extent(c, _) => {
                        self.extent_show = self.extent_show.toggled(c);
                    }
                }
            }
        }
    }

    // Dispatch through the console so the palette needs no command logic of
    // its own; the console opens first so the reply is visible.
    fn dispatch_palette_command(&mut self, world: &mut World, line: &str) {
        if !self.console_open {
            self.toggle_console(world);
        }
        self.run_console_line(world, line);
    }

    // Open the Behavior panel on the named behavior. A name the authored
    // entries do not carry (a generated behavior) has no ordinal to open, so
    // it falls back to the edit form like any other asset.
    fn open_behavior_named(&mut self, name: &str, world: &mut World) {
        let ordinal = self
            .behavior_entries()
            .iter()
            .position(|&i| entry_name(&self.entries[i]) == Some(name));
        let Some(ordinal) = ordinal else {
            self.selection.replace(name.to_string());
            self.focus_ui_on(name, world);
            return;
        };
        self.behavior_index = ordinal;
        self.behavior_open = true;
        self.open_behavior(world);
        self.focus_panel(PanelKey::Behavior);
    }
}

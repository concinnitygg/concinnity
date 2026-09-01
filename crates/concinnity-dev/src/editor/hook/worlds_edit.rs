// src/editor/hook/worlds_edit.rs
//
// EditorHook: the Worlds panel's actions. Opening a world retargets the whole
// session -- the path SAVE writes, the working entries and their history, the
// hot-reload watcher, and the per-world session state -- then asks for the
// preview rebuild that swaps the compiled world under the live backend, the
// same machinery every other rebuild takes. `+` retargets the same way onto an
// empty world that is not on disk yet, so the editor comes up in full on it and
// the first SAVE asks for a name; deleting removes the file and the session
// store's entry for it. Both switches run behind the confirmation dialog
// whenever the open world has unsaved edits.

use super::*;
use std::path::Path;

impl EditorHook {
    // Re-read the project's worlds. The listing changes only when the panel
    // acts on it (or when it opens), so the draw and press paths resolve
    // against this rather than reading the directory every frame.
    pub(super) fn refresh_worlds(&mut self) {
        let open = Path::new(&self.world_path);
        self.worlds_rows = world_files::list(
            crate::project::worlds_dir().as_deref(),
            crate::project::content_root().as_deref(),
        )
        .into_iter()
        .map(|w| WorldRow {
            open: w.path == open,
            name: w.name,
            path: w.path.to_string_lossy().into_owned(),
        })
        .collect();
        let max = self.worlds_max_scroll();
        self.worlds_scroll = self.worlds_scroll.min(max);
    }

    // Show the panel, with a fresh listing.
    pub(super) fn open_worlds_panel(&mut self) {
        self.worlds_open = true;
        self.worlds_status = None;
        self.worlds_scroll = 0;
        self.worlds_menu = None;
        self.refresh_worlds();
    }

    pub(super) fn make_worlds_view<'a>(&'a self, mouse: [f32; 2]) -> WorldsView<'a> {
        WorldsView {
            rows: &self.worlds_rows,
            // The window follows the viewport, so a listing scrolled to its end
            // in a short window is pulled back when the window grows.
            scroll: self.worlds_scroll.min(self.worlds_max_scroll()),
            layout: self.worlds_layout(),
            selected: self.worlds_row_of(self.worlds_selected.as_ref()),
            previewing: self.worlds_row_of(self.worlds_preview.as_ref()),
            menu: self.worlds_row_of(self.worlds_menu.as_ref()),
            status: self.worlds_status.as_deref(),
            mouse,
        }
    }

    pub(super) fn scroll_worlds(&mut self, delta: f32) {
        let max = self.worlds_max_scroll();
        self.worlds_scroll = scroll_step(self.worlds_scroll, delta, max);
    }

    // How far the listing scrolls: the rows it holds past the window the
    // current presentation shows.
    fn worlds_max_scroll(&self) -> usize {
        let window = self.worlds_layout().rows();
        self.worlds_rows.len().saturating_sub(window)
    }

    // Route a resolved Worlds-panel click. Every action but opening the menu
    // closes it, so no press leaves it standing over a list it no longer
    // describes.
    pub(super) fn apply_worlds_action(&mut self, action: WorldsAction, world: &mut World) {
        let opening_menu = matches!(action, WorldsAction::OpenMenu(_));
        match action {
            WorldsAction::New => self.request_world(WorldTarget::Untitled, world),
            WorldsAction::Select(i) => self.select_world(i),
            WorldsAction::Open(i) if self.start_mode => self.open_from_start(i, world),
            WorldsAction::Open(i) => {
                if let Some(row) = self.worlds_rows.get(i) {
                    let target = WorldTarget::Open(row.path.clone());
                    self.request_world(target, world);
                }
            }
            WorldsAction::OpenMenu(i) => {
                self.worlds_menu = self.worlds_rows.get(i).map(|r| r.path.clone());
            }
            WorldsAction::Delete(i) => self.confirm_delete_world(i),
            WorldsAction::CloseMenu | WorldsAction::Consume => {}
        }
        if !opening_menu {
            self.worlds_menu = None;
        }
    }

    // Guard a switch behind the confirmation dialog while the open world has
    // edits that are not on disk; a clean world switches straight away.
    fn request_world(&mut self, target: WorldTarget, world: &mut World) {
        // Nothing is open to be dirty on the start screen: the world it shows
        // is a preview the session never edited.
        if !self.dirty || self.start_mode {
            self.go_to_world(target, world);
            return;
        }
        let name = session_store::world_key(&self.world_path);
        self.open_modal(
            &format!("'{name}' has unsaved changes. Save them before switching worlds?"),
            vec![
                modal::Button {
                    label: "Discard".to_string(),
                    danger: true,
                    action: modal::Action::Worlds(WorldsConfirm::Discard(target.clone())),
                },
                modal::Button {
                    label: "Cancel".to_string(),
                    danger: false,
                    action: modal::Action::Dismiss,
                },
                modal::Button {
                    label: "Save".to_string(),
                    danger: false,
                    action: modal::Action::Worlds(WorldsConfirm::Save(target)),
                },
            ],
        );
    }

    // Carry out a dialog button's Worlds decision.
    pub(super) fn apply_worlds_confirm(&mut self, confirm: WorldsConfirm, world: &mut World) {
        match confirm {
            WorldsConfirm::Delete(path) => self.delete_world(&path),
            WorldsConfirm::Save(target) => {
                self.save();
                // A failed save leaves the world dirty and nothing switches:
                // the edits the dialog was guarding are still only in memory.
                if !self.dirty {
                    self.go_to_world(target, world);
                }
            }
            WorldsConfirm::Discard(target) => self.go_to_world(target, world),
        }
    }

    pub(super) fn go_to_world(&mut self, target: WorldTarget, world: &mut World) {
        // Whatever the start screen was showing is being replaced, so its
        // attract camera stops, handing that world its own pose back in case a
        // failed compile leaves it standing.
        self.stop_cinematic(world);
        match target {
            WorldTarget::Open(path) => self.open_world(path),
            WorldTarget::Untitled => self.new_untitled_world(),
        }
    }

    // Start on an empty world that has no file yet. The session retargets as it
    // would onto any other world, so the top bar and the panels come straight
    // up on it; only the naming is deferred, to the first SAVE.
    fn new_untitled_world(&mut self) {
        self.retarget(crate::editor::unsaved_world_path(), Vec::new(), Adopt::No);
        self.untitled = true;
    }

    // Open an existing world file. A file that will not parse is reported on
    // the status line and nothing is retargeted, so the session keeps the world
    // it has.
    fn open_world(&mut self, path: String) {
        match world_files::read_entries(Path::new(&path)) {
            Ok(entries) => self.retarget(path, entries, Adopt::No),
            Err(e) => self.worlds_status = Some(e),
        }
    }

    // Whether a name prompt is open, so the shortcuts that would otherwise
    // fire on a keystroke stand down while it is being typed into.
    pub(super) fn naming_world(&self) -> bool {
        self.modal.as_ref().is_some_and(|m| m.field)
    }

    // Ask what to call the untitled world the session is on. `rejected` carries
    // back why the last attempt was turned down, so the dialog reopens saying
    // so rather than dismissing the work.
    pub(super) fn prompt_world_name(&mut self, rejected: Option<String>) {
        let message = rejected.unwrap_or_else(|| "Name this world to save it.".to_string());
        self.open_prompt(
            &message,
            vec![
                modal::Button {
                    label: "Cancel".to_string(),
                    danger: false,
                    action: modal::Action::Dismiss,
                },
                modal::Button {
                    label: "Save".to_string(),
                    danger: false,
                    action: modal::Action::NameWorld,
                },
            ],
        );
    }

    // Take the name typed into the prompt: on a usable one the session moves
    // onto `worlds/<name>.jsonl` and saves there, and on a rejected one the
    // prompt comes back carrying the reason.
    pub(super) fn name_untitled_world(&mut self, typed: &str) {
        let existing: Vec<String> = self.worlds_rows.iter().map(|r| r.name.clone()).collect();
        let name = match world_files::validate_name(typed, &existing) {
            Ok(name) => name,
            Err(reason) => return self.prompt_world_name(Some(reason)),
        };
        let Some(dir) = crate::project::worlds_dir() else {
            return self.prompt_world_name(Some("No project is open".to_string()));
        };
        // Claim the file before writing to it: the listing the name was checked
        // against is only as fresh as the last refresh, and `create` refuses a
        // path something already sits at rather than clobbering it.
        let path = match world_files::create(&dir, &name) {
            Ok(path) => path.to_string_lossy().into_owned(),
            Err(e) => return self.prompt_world_name(Some(format!("Create failed: {e}"))),
        };
        concinnity_engine::app::dev_flags::set_world_jsonl_path(Some(path.clone()));
        self.world_path = path;
        self.untitled = false;
        self.save();
        self.refresh_worlds();
    }

    // Ask before removing a world file: it is the authored source, and nothing
    // else in the project holds a copy of it.
    fn confirm_delete_world(&mut self, i: usize) {
        let Some(row) = self.worlds_rows.get(i) else {
            return;
        };
        let (name, path) = (row.name.clone(), row.path.clone());
        self.open_modal(
            &format!("Delete the world '{name}'? Its file is removed from disk."),
            vec![
                modal::Button {
                    label: "Cancel".to_string(),
                    danger: false,
                    action: modal::Action::Dismiss,
                },
                modal::Button {
                    label: "Delete".to_string(),
                    danger: true,
                    action: modal::Action::Worlds(WorldsConfirm::Delete(path)),
                },
            ],
        );
    }

    // Remove a world file and forget the session state kept under its name.
    // Deleting the world the session has open is allowed: the running session
    // keeps its entries, and since nothing of it is on disk any more it reads
    // as unsaved, so a later SAVE writes the file back.
    pub(super) fn delete_world(&mut self, path: &str) {
        if let Err(e) = world_files::delete(Path::new(path)) {
            self.worlds_status = Some(format!("Delete failed: {e}"));
            return;
        }
        let key = session_store::world_key(path);
        if let Some(store) = session_store::default_path()
            && let Err(e) = session_store::forget(&store, &key)
        {
            tracing::warn!("editor: could not clear the session entry for {key}: {e}");
        }
        // Where the selection lands once the row is gone.
        let was_at = self.worlds_rows.iter().position(|r| r.path == path);
        if !self.start_mode && path == self.world_path {
            self.saved = Vec::new();
            self.dirty = !self.entries.is_empty();
        }
        self.refresh_worlds();
        if self.start_mode {
            self.reselect_after_delete(path, was_at);
        }
        self.notifier.success(&format!("Deleted world '{key}'"));
    }

    // Point the whole session at another world: the path SAVE writes, the
    // working entries with a fresh history, the hot-reload watcher, and every
    // piece of session state that indexes the world left behind. The compiled
    // world follows on the next frame's rebuild, which swaps it in under the
    // live render backend rather than recreating the window.
    pub(super) fn retarget(&mut self, path: String, entries: Vec<serde_json::Value>, adopt: Adopt) {
        concinnity_engine::app::dev_flags::set_world_jsonl_path(Some(path.clone()));
        self.bookmarks = session_store::default_path()
            .and_then(|store| {
                session_store::load(&store)
                    .worlds
                    .get(&session_store::world_key(&path))
                    .map(|w| w.bookmarks)
            })
            .unwrap_or_default();
        self.world_path = path;
        self.history = History::default();
        self.baseline = entries.clone();
        self.saved = entries.clone();
        // The start screen's preview compiled these very entries and swapped
        // the result in, so opening adopts the world already showing instead of
        // building it a second time. Any other switch is showing some other
        // world, and a preview whose compile failed still owes one.
        let showing = adopt == Adopt::Showing && self.world_entries == entries;
        self.entries = entries;
        self.dirty = false;
        if !showing {
            self.world_shadows = None;
            self.require_rebuild();
        }

        // Everything below indexes, names, or was read out of the world that
        // is being left behind.
        self.tree_groups.clear();
        self.tree_unfolded.clear();
        self.tree_scroll = 0;
        self.tree_status = None;
        self.tree_stale = true;
        self.template_index = None;
        self.close_form();
        self.row_menu = None;
        self.picker_open = false;
        self.selection.clear();
        self.pick_last = None;
        self.marquee = None;
        self.gizmo_drag = None;
        self.shape_drag = None;
        self.shape_status = None;
        self.content_drag = None;
        self.create_menu = None;
        self.hidden_assets.clear();
        self.locked_assets.clear();
        self.isolate = None;
        self.behavior_index = 0;
        self.behavior_row = None;
        self.behavior_scroll = 0;
        self.behavior_status = None;
        self.variables_row = None;
        self.variables_scroll = 0;
        self.lighting_focus = None;
        self.lighting_status = None;
        self.story_lines = vec![String::new()];
        self.story_line = 0;
        self.story_scroll = 0;
        self.story_focus = false;
        self.story_path = String::new();
        self.story_status = None;
        self.form_touched = false;
        self.lighting_touched = false;
        self.story_touched = false;

        // The panel has done its job; it stays a registered panel, so the View
        // row reopens it as a switcher. The session owns a world now, so the
        // start screen is over for good and the rest of the editor comes back.
        self.leave_start_screen();
        self.worlds_open = false;
        self.worlds_menu = None;
        self.worlds_status = None;
        self.worlds_scroll = 0;
        self.untitled = false;
        self.refresh_worlds();
    }
}

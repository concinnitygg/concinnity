// src/editor/hook/worlds_edit.rs
//
// EditorHook: the Worlds panel's actions. Opening a world retargets the whole
// session -- the path SAVE writes, the working entries and their history, the
// hot-reload watcher, and the per-world session state -- then asks for the
// preview rebuild that swaps the compiled world under the live backend, the
// same machinery every other rebuild takes. Creating names the file first so
// it exists (and lists) before anything is authored into it; deleting removes
// the file and the session store's entry for it. Both switches run behind the
// confirmation dialog whenever the open world has unsaved edits.

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
        let max = self.worlds_rows.len().saturating_sub(worlds::POOL);
        self.worlds_scroll = self.worlds_scroll.min(max);
    }

    // Show the panel, with a fresh listing and an empty name field ready to
    // type into.
    pub(super) fn open_worlds_panel(&mut self, world: &mut World) {
        self.worlds_open = true;
        self.worlds_status = None;
        self.worlds_scroll = 0;
        self.worlds_focus = true;
        self.refresh_worlds();
        widget::seed_field(world, worlds::NAME_INPUT, "");
    }

    pub(super) fn make_worlds_view<'a>(&'a self, mouse: [f32; 2]) -> WorldsView<'a> {
        WorldsView {
            rows: &self.worlds_rows,
            scroll: self.worlds_scroll,
            // Focus is asserted only while frontmost, matching the other
            // panels' guard against fighting for typed keys.
            focus: self.worlds_focus && self.panel_order.last() == Some(&PanelKey::Worlds),
            status: self.worlds_status.as_deref(),
            mouse,
        }
    }

    pub(super) fn scroll_worlds(&mut self, delta: f32) {
        let max = self.worlds_rows.len().saturating_sub(worlds::POOL);
        self.worlds_scroll = scroll_step(self.worlds_scroll, delta, max);
    }

    // Enter in the name field creates, like pressing New.
    pub(super) fn worlds_keys(&mut self, world: &mut World, input: &FrameInput) {
        if self.worlds_focus && input.captured_key == Some(crate::components::InputKey::Enter) {
            self.new_world(world);
        }
    }

    // Route a resolved Worlds-panel click.
    pub(super) fn apply_worlds_action(&mut self, action: WorldsAction, world: &mut World) {
        match action {
            WorldsAction::FocusName => self.worlds_focus = true,
            WorldsAction::New => self.new_world(world),
            WorldsAction::Open(i) => {
                if let Some(row) = self.worlds_rows.get(i) {
                    let target = WorldTarget::Open(row.path.clone());
                    self.request_world(target, world);
                }
            }
            WorldsAction::Delete(i) => self.confirm_delete_world(i),
            // A click on panel chrome blurs the name field.
            WorldsAction::Consume => self.worlds_focus = false,
        }
    }

    // Take the typed name and, if it is usable, create the world and open it.
    fn new_world(&mut self, world: &mut World) {
        let typed = widget::field_text(world, worlds::NAME_INPUT);
        let existing: Vec<String> = self.worlds_rows.iter().map(|r| r.name.clone()).collect();
        match world_files::validate_name(&typed, &existing) {
            Ok(name) => {
                self.worlds_status = None;
                self.request_world(WorldTarget::Create(name), world);
            }
            Err(reason) => self.worlds_status = Some(reason),
        }
    }

    // Guard a switch behind the confirmation dialog while the open world has
    // edits that are not on disk; a clean world switches straight away.
    fn request_world(&mut self, target: WorldTarget, world: &mut World) {
        if !self.dirty {
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
        match target {
            WorldTarget::Open(path) => self.open_world(path),
            WorldTarget::Create(name) => self.create_world(&name, world),
        }
    }

    // Open an existing world file. A file that will not parse is reported on
    // the status line and nothing is retargeted, so the session keeps the world
    // it has.
    fn open_world(&mut self, path: String) {
        let entries = match std::fs::read_to_string(&path) {
            Ok(content) => match crate::world::parse_world_jsonl(&content) {
                Ok(entries) => entries,
                Err(e) => {
                    self.worlds_status = Some(format!("{e}"));
                    return;
                }
            },
            // A world file that vanished under the listing opens empty rather
            // than stranding the panel: the next SAVE writes it back.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => {
                self.worlds_status = Some(format!("Open failed: {e}"));
                return;
            }
        };
        self.retarget(path, entries);
    }

    // Create the named world under the project's `worlds/` and open it, so it
    // exists on disk (and lists) from the moment it is named.
    fn create_world(&mut self, name: &str, world: &mut World) {
        let Some(dir) = crate::project::worlds_dir() else {
            self.worlds_status = Some("No project is open".to_string());
            return;
        };
        match world_files::create(&dir, name) {
            Ok(path) => {
                widget::seed_field(world, worlds::NAME_INPUT, "");
                self.retarget(path.to_string_lossy().into_owned(), Vec::new());
                self.notifier.success(&format!("Created world '{name}'"));
            }
            Err(e) => self.worlds_status = Some(format!("Create failed: {e}")),
        }
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
        if path == self.world_path {
            self.saved = Vec::new();
            self.dirty = !self.entries.is_empty();
        }
        self.refresh_worlds();
        self.notifier.success(&format!("Deleted world '{key}'"));
    }

    // Point the whole session at another world: the path SAVE writes, the
    // working entries with a fresh history, the hot-reload watcher, and every
    // piece of session state that indexes the world left behind. The compiled
    // world follows on the next frame's rebuild, which swaps it in under the
    // live render backend rather than recreating the window.
    fn retarget(&mut self, path: String, entries: Vec<serde_json::Value>) {
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
        self.entries = entries;
        self.dirty = false;
        self.world_shadows = None;
        self.require_rebuild();

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
        // row reopens it as a switcher.
        self.worlds_open = false;
        self.worlds_focus = false;
        self.worlds_status = None;
        self.worlds_scroll = 0;
        self.refresh_worlds();
    }
}

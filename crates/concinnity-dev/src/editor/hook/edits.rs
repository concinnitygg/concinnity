//! EditorHook: unique-name generation and edit persistence (SAVE, the atomic
//! world.jsonl write, and the in-memory live-preview world rebuild).

use concinnity_cook::authoring::world::entry_handle;
use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;

use super::{EditorHook, EntryId, EntryList, FormTarget, declared_id};
use crate::editor::behavior;
use crate::editor::build_renderable;
use crate::editor::entry_list::build_text;
use crate::editor::live;
use crate::editor::modal;
use crate::editor::notify;
use crate::editor::panels::assets_panel;
use crate::editor::panels::form_panel;
use crate::editor::widget;

impl EditorHook {
    // Whether an entry already declares this `$id`.
    pub(super) fn name_taken(&self, n: &str) -> bool {
        self.entries.iter().any(|e| declared_id(e) == Some(n))
    }

    // Whether an entry other than `skip`'s already declares this `$id` (for
    // renames).
    pub(super) fn name_taken_except(&self, n: &str, skip: EntryId) -> bool {
        let skip = self.entries.index_of(skip);
        self.entries
            .iter()
            .enumerate()
            .any(|(i, e)| Some(i) != skip && declared_id(e) == Some(n))
    }

    // A world-unique name derived from the asset type: `editor_<kind>` plus a
    // numeric suffix bumped until it does not collide with an existing entry.
    pub(super) fn unique_name(&self, kind: &str) -> String {
        let base = format!("editor_{}", kind.to_ascii_lowercase());
        self.unique_from(&base)
    }

    // `base` if free, else `base_1`, `base_2`, ... until unused.
    pub(super) fn unique_from(&self, base: &str) -> String {
        if !self.name_taken(base) {
            return base.to_string();
        }
        let mut i = 1;
        loop {
            let candidate = format!("{base}_{i}");
            if !self.name_taken(&candidate) {
                return candidate;
            }
            i += 1;
        }
    }

    // The `$id` for a new-asset submission: the typed id (trimmed) made
    // unique, or `None` when the field was left blank and the asset is added
    // anonymous.
    pub(super) fn finalize_name(&self, typed: &str) -> Option<String> {
        let t = typed.trim();
        (!t.is_empty()).then(|| self.unique_from(t))
    }

    // The `$id` for a rename of `key`'s entry: the typed id (trimmed) made
    // unique against the *other* entries, or `None` when blank, which leaves
    // the entry anonymous.
    pub(super) fn finalize_rename(&self, typed: &str, key: EntryId) -> Option<String> {
        let base = typed.trim();
        if base.is_empty() {
            return None;
        }
        if !self.name_taken_except(base, key) {
            return Some(base.to_string());
        }
        let mut i = 1;
        loop {
            let candidate = format!("{base}_{i}");
            if !self.name_taken_except(&candidate, key) {
                return Some(candidate);
            }
            i += 1;
        }
    }

    // Drop the authored line at `idx` and record the edit. A form open on that
    // entry closes; one open on any other entry keeps its target, which is the
    // key rather than the position the removal shifted.
    pub(super) fn remove_entry_at(&mut self, idx: usize) {
        let Some(key) = self.entries.key_at(idx) else {
            return;
        };
        self.entries.remove(idx);
        self.mark_changed();
        if self.form.target == FormTarget::Entry(key) {
            self.form.close();
        }
    }

    // Record an authored-entry change: the live preview is out of date this
    // frame (`apply_world_swap` writes the change into the running world, or
    // reloads it when the change cannot be expressed there), and the change is
    // not yet on disk (SAVE clears `dirty`). The pre-edit list still sits in
    // `baseline` (only committed edits move it), so it becomes the undo
    // snapshot; a call that changed nothing records no step.
    //
    // An edit that reached an included entry or an `Include` line is undone
    // whole: those lines belong to another file, or decide what that file
    // brings in, so the editor never writes them.
    pub(super) fn mark_changed(&mut self) {
        if let Some(i) = self.entries.changed_read_only(&self.baseline) {
            self.refuse_read_only_edit(i);
            return;
        }
        if self.baseline != self.entries {
            let before = std::mem::replace(&mut self.baseline, self.entries.clone());
            self.history.record(before);
        }
        self.dirty = true;
        self.rebuild_preview = true;
        // The expansion follows the entries, so the Assets tree is now out of
        // date. Recomputed by the frame drive while the panel shows, so a burst
        // of edits costs one expansion rather than one per edit.
        self.tree_stale = true;
        // Template baselines follow the entries too; rebuilt on demand.
        self.template_index = None;
    }

    // `mark_changed`, answering whether the edit stood: one that reached a
    // read-only entry is undone, and says why.
    pub(super) fn commit(&mut self) -> bool {
        let refused = self.entries.changed_read_only(&self.baseline).is_some();
        self.mark_changed();
        !refused
    }

    // Put the entries back as they were before an edit that changed the
    // read-only baseline entry at `index`, and say where that entry lives. The
    // preview is rebuilt, since a drag may already have moved what it shows.
    fn refuse_read_only_edit(&mut self, index: usize) {
        let name = entry_handle(&self.baseline, index).unwrap_or_default();
        let message = match self.baseline.included_from(index) {
            Some(file) => format!(
                "'{name}' is included from {}; edit it there",
                file.display()
            ),
            None => format!(
                "'{name}' is an Include line; edit it in {}",
                self.world_path
            ),
        };
        self.entries = self.baseline.clone();
        self.require_rebuild();
        self.notifier.push(notify::Level::Error, &message);
    }

    // Ask for a full preview rebuild: the running world holds state no authored
    // diff describes (a simulation that ran, a story source re-read from disk),
    // so writing the entry diff into it would leave that state standing.
    pub(super) fn require_rebuild(&mut self) {
        self.rebuild_preview = true;
        self.rebuild_required = true;
    }

    // Step the entry list back / forward through the history stacks. No-ops at
    // either end of the history.
    pub(super) fn undo(&mut self, world: &mut World) {
        if let Some(snap) = self.history.undo(self.entries.clone()) {
            self.apply_history_jump(snap, world);
        }
    }

    pub(super) fn redo(&mut self, world: &mut World) {
        if let Some(snap) = self.history.redo(self.entries.clone()) {
            self.apply_history_jump(snap, world);
        }
    }

    pub(super) fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    pub(super) fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    // Install a history snapshot as the working entry list. A snapshot carries
    // the session keys, so the jump restores the very entries the selection
    // addresses and it survives; a member the restored list does not hold goes
    // undrawn rather than addressing something else. The open form's controls
    // hold text derived from the pre-jump args, and the pointer gestures were
    // sampled against the pre-jump list, so both are dropped. `dirty` is
    // recomputed against the on-disk state so unwinding back to the saved list
    // clears the Save chip.
    fn apply_history_jump(&mut self, snap: EntryList, world: &mut World) {
        self.entries = snap;
        self.baseline = self.entries.clone();
        self.dirty = self.entries != self.saved;
        self.rebuild_preview = true;
        self.tree_stale = true;
        self.template_index = None;
        self.form.close();
        self.row_menu = None;
        self.picker_open = false;
        self.pick_last = None;
        self.marquee = None;
        self.gizmo_drag = None;
        self.shape_drag = None;
        self.content_drag = None;
        self.create_menu = None;
        self.shaders.menu = None;
        self.follow_shader_source();
        // The Lighting panel's text controls hold committed values; re-seed so
        // they show the restored list, not the undone edit.
        if self.lighting_open {
            self.seed_lighting(world);
        }
    }

    // SAVE: persist the working entries to disk. world.jsonl is the source of
    // truth and the only thing a save writes -- the compiled blobs under the
    // build root belong to an explicit build, so a save never cooks and never
    // stalls behind one. The live preview is already up to date (every edit
    // refreshed it), so nothing is rebuilt or swapped here. On a write failure
    // the world stays dirty and the next SAVE retries.
    pub(super) fn save(&mut self) {
        // A world nobody has named has nowhere to go yet: ask first, and let
        // the dialog's Save come back through here once it has a path.
        if self.untitled {
            self.prompt_world_name(None);
            return;
        }
        let content = match self.entries.file_text() {
            Ok(c) => c,
            Err(e) => {
                self.save_failed(e);
                return;
            }
        };
        if let Err(e) = self.write_jsonl_content(&content) {
            self.save_failed(e);
            return;
        }
        self.dirty = false;
        self.saved = self.entries.clone();
        tracing::info!("editor: saved {}", self.world_path);
        self.notifier.success(&format!("Saved {}", self.world_path));
    }

    // Report a failed SAVE, leaving the world dirty for the next attempt.
    fn save_failed(&self, e: impl std::fmt::Display) {
        tracing::error!("editor: save failed: {e}");
        self.notifier
            .error_with(&format!("Save failed: {e}"), notify::Action::OpenConsole);
    }

    // Build a ready-to-run world from the in-memory entries, without touching disk
    // (SAVE owns persistence). The same compile the session booted through, so a
    // rebuild can only differ from boot by the edits since. The template
    // baselines the expansion merged authored patches over come back with it, so a
    // later edit can re-derive one asset's effective args without cooking again.
    pub(super) fn build_preview_world(&self) -> std::io::Result<(World, live::ShadowBaselines)> {
        let jsonl = build_text(&self.entries)?;
        let (world, shadowed) = build_renderable(&jsonl)?;
        let baselines = shadowed
            .into_iter()
            .map(|s| (s.name, s.args))
            .collect::<live::ShadowBaselines>();
        Ok((world, baselines))
    }

    // Snapshot the editor's text-field contents (the combo filter + the form's name
    // heading and arg inputs) by reserved id, so a live rebuild's fresh HUD
    // injection does not blank an open form.
    pub(super) fn field_snapshot(world: &World) -> Vec<(AssetId, String)> {
        assets_panel::all_field_ids()
            .into_iter()
            .chain(form_panel::all_field_ids())
            .chain(behavior::panel::all_field_ids())
            .chain(modal::all_field_ids())
            .map(|id| (id, widget::field_text(world, id)))
            .collect()
    }

    // Restore a `field_snapshot` into a freshly injected HUD.
    pub(super) fn restore_fields(world: &mut World, snapshot: &[(AssetId, String)]) {
        for (id, content) in snapshot {
            widget::seed_field(world, *id, content);
        }
    }

    // Write the working entries to world.jsonl atomically (temp file + rename),
    // so a crash mid-write cannot truncate the user's world. SAVE inlines the
    // serialization; this remains the test seam for the write itself.
    #[cfg(test)]
    pub(super) fn write_jsonl(&self) -> std::io::Result<()> {
        let out = self.entries.file_text()?;
        self.write_jsonl_content(&out)
    }

    // The file-write tail of `write_jsonl`, for a caller that already
    // serialized the entries. The directory is created first: a fresh project
    // saves its first world into a `worlds/` that does not exist yet.
    fn write_jsonl_content(&self, content: &str) -> std::io::Result<()> {
        if let Some(parent) = std::path::Path::new(&self.world_path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = format!("{}.tmp", self.world_path);
        std::fs::write(&tmp, content)?;
        std::fs::rename(&tmp, &self.world_path)
    }
}

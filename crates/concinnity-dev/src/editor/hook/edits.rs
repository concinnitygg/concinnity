// src/editor/hook/edits.rs
//
// EditorHook: unique-name generation and edit persistence (SAVE, the atomic
// world.jsonl write, and the in-memory live-preview world rebuild).

use super::*;

impl EditorHook {
    // Whether an entry with this name already exists.
    pub(super) fn name_taken(&self, n: &str) -> bool {
        self.entries.iter().any(|e| entry_name(e) == Some(n))
    }

    // Whether an entry other than `skip` already has this name (for renames).
    pub(super) fn name_taken_except(&self, n: &str, skip: usize) -> bool {
        self.entries
            .iter()
            .enumerate()
            .any(|(i, e)| i != skip && entry_name(e) == Some(n))
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

    // The final name for a new-asset submission: the typed name (trimmed) made
    // unique, or a generated unique name when the field was left blank.
    pub(super) fn finalize_name(&self, typed: &str, kind: &str) -> String {
        let t = typed.trim();
        if t.is_empty() {
            self.unique_name(kind)
        } else {
            self.unique_from(t)
        }
    }

    // The final name for a rename of entry `idx`: the typed name (trimmed), or a
    // generated one when blank, made unique against the *other* entries.
    pub(super) fn finalize_rename(&self, typed: &str, idx: usize, kind: &str) -> String {
        let t = typed.trim();
        let base = if t.is_empty() {
            format!("editor_{}", kind.to_ascii_lowercase())
        } else {
            t.to_string()
        };
        if !self.name_taken_except(&base, idx) {
            return base;
        }
        let mut i = 1;
        loop {
            let candidate = format!("{base}_{i}");
            if !self.name_taken_except(&candidate, idx) {
                return candidate;
            }
            i += 1;
        }
    }

    // Drop the authored line at `idx` and record the edit. The open form indexes
    // into `entries`, so removing the edited entry closes it and removing an
    // earlier one shifts it.
    pub(super) fn remove_entry_at(&mut self, idx: usize) {
        if idx >= self.entries.len() {
            return;
        }
        self.entries.remove(idx);
        self.mark_changed();
        match self.form_target {
            FormTarget::Entry(e) if e == idx => self.close_form(),
            FormTarget::Entry(e) if e > idx => self.form_target = FormTarget::Entry(e - 1),
            _ => {}
        }
    }

    // Record an authored-entry change: the live preview is out of date this
    // frame (`apply_world_swap` writes the change into the running world, or
    // reloads it when the change cannot be expressed there), and the change is
    // not yet on disk (SAVE clears `dirty`). The pre-edit list still sits in
    // `baseline` (only committed edits move it), so it becomes the undo
    // snapshot; a call that changed nothing records no step.
    pub(super) fn mark_changed(&mut self) {
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

    // Install a history snapshot as the working entry list. Everything indexing
    // into `entries` (the open form, the row menu) may point at removed or
    // shifted rows after a jump, so it is dropped; `dirty` is recomputed against
    // the on-disk state so unwinding back to the saved list clears the Save chip.
    fn apply_history_jump(&mut self, snap: Vec<serde_json::Value>, world: &mut World) {
        self.entries = snap;
        self.baseline = self.entries.clone();
        self.dirty = self.entries != self.saved;
        self.rebuild_preview = true;
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
        self.content_drag = None;
        self.create_menu = None;
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
        let content = match crate::world::write_world_jsonl(&self.entries) {
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
        let jsonl = crate::world::write_world_jsonl(&self.entries)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
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
        panel::all_field_ids()
            .into_iter()
            .chain(form_panel::all_field_ids())
            .chain(behavior_panel::all_field_ids())
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
        let out = crate::world::write_world_jsonl(&self.entries)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
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

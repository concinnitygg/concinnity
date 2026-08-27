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

    // Record an authored-entry change: the live preview needs a rebuild this frame
    // (`apply_world_swap` reloads the running world from the in-memory entries), and
    // the change is not yet on disk (SAVE clears `dirty`). The pre-edit list still
    // sits in `baseline` (only committed edits move it), so it becomes the undo
    // snapshot; a call that changed nothing records no step.
    pub(super) fn mark_changed(&mut self) {
        if self.baseline != self.entries {
            let before = std::mem::replace(&mut self.baseline, self.entries.clone());
            self.history.record(before);
        }
        self.dirty = true;
        self.rebuild_preview = true;
        // The rebuild discards a running simulation's state, so the transport
        // honestly drops to Stopped.
        self.sim.on_edit();
        // The expansion follows the entries, so the Assets tree is now out of
        // date. Recomputed by the frame drive while the panel shows, so a burst
        // of edits costs one expansion rather than one per edit.
        self.tree_stale = true;
        // Template baselines follow the entries too; rebuilt on demand.
        self.template_index = None;
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
        self.sim.on_edit();
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
    // truth, so its atomic write is the save -- `dirty` clears on it. The
    // compiled blobs are a derived cache (boot rebuilds them when missing), so
    // their recompile runs on the cook worker with a progress card instead of
    // stalling the frame loop; a blob failure toasts but the world stays
    // saved. The live preview is already up to date (every edit swaps it in),
    // so nothing is rebuilt or swapped here. On a write failure the world
    // stays dirty and the next SAVE retries.
    pub(super) fn save(&mut self) {
        // A cook is already writing the same blobs; stay dirty and let the
        // next SAVE retry instead of racing it. The swap also claims the
        // guard for this save's own blob cook.
        if self
            .console_build_running
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            tracing::warn!("editor: save deferred, a cook is writing blobs");
            self.notifier
                .warning("Save deferred: a cook is writing blobs");
            return;
        }
        let release = |hook: &Self| {
            hook.console_build_running
                .store(false, std::sync::atomic::Ordering::SeqCst);
        };
        let content = match crate::world::write_world_jsonl(&self.entries) {
            Ok(c) => c,
            Err(e) => {
                release(self);
                tracing::error!("editor: save failed: {e}");
                self.notifier
                    .error_with(&format!("Save failed: {e}"), notify::Action::OpenConsole);
                return;
            }
        };
        if let Err(e) = self.write_jsonl_content(&content) {
            release(self);
            tracing::error!("editor: save failed: {e}");
            self.notifier
                .error_with(&format!("Save failed: {e}"), notify::Action::OpenConsole);
            return;
        }
        self.dirty = false;
        self.saved = self.entries.clone();
        tracing::info!("editor: saved {}", self.world_path);
        self.notifier.success(&format!("Saved {}", self.world_path));
        // The worker releases the guard when it finishes. Success stays quiet
        // (the save already toasted); only a blob failure surfaces.
        let sink = self.console_sink.clone();
        let toasts = self.notifier.clone();
        self.spawn_cook_worker("Cooking blobs", content, move |outcome, _secs| {
            if let Err(e) = outcome {
                sink.error(&format!("save: blob cook failed: {e}"));
                toasts.error_with(
                    &format!("Blob cook failed: {e}"),
                    notify::Action::OpenConsole,
                );
            }
        });
    }

    // Build a ready-to-run world from the in-memory entries, without touching disk
    // (SAVE owns persistence). A GraphicsConfig is seeded when the authored entries
    // alone would not render, so the preview window never goes blank.
    pub(super) fn build_preview_world(&self) -> std::io::Result<World> {
        let jsonl = crate::world::write_world_jsonl(&self.entries)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        match crate::build_world_from_str(&jsonl) {
            Ok(world) if concinnity_engine::ecs::renders(&world) => Ok(world),
            _ => crate::build_world_from_str(&super::seeded_content(&jsonl)),
        }
    }

    // Snapshot the editor's text-field contents (the combo filter + the form's name
    // heading and arg inputs) by reserved id, so a live rebuild's fresh HUD
    // injection does not blank an open form.
    pub(super) fn field_snapshot(world: &World) -> Vec<(AssetId, String)> {
        panel::all_field_ids()
            .into_iter()
            .chain(form_panel::all_field_ids())
            .chain(behavior_panel::all_field_ids())
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
    // serialization (it reuses the string for the blob cook); this remains the
    // test seam for the write itself.
    #[cfg(test)]
    pub(super) fn write_jsonl(&self) -> std::io::Result<()> {
        let out = crate::world::write_world_jsonl(&self.entries)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        self.write_jsonl_content(&out)
    }

    // The file-write tail of `write_jsonl`, for a caller that already
    // serialized the entries (SAVE reuses the string for the blob cook).
    fn write_jsonl_content(&self, content: &str) -> std::io::Result<()> {
        let tmp = format!("{}.tmp", self.world_path);
        std::fs::write(&tmp, content)?;
        std::fs::rename(&tmp, &self.world_path)
    }
}

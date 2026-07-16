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

    // Record an authored-entry change: the live preview needs a rebuild this frame
    // (`apply_world_swap` reloads the running world from the in-memory entries), and
    // the change is not yet on disk (SAVE clears `dirty`).
    pub(super) fn mark_changed(&mut self) {
        self.dirty = true;
        self.rebuild_preview = true;
        // The expansion follows the entries, so the Expanded tab's model is now
        // out of date. Recomputed by the frame drive if that tab is showing, so
        // a burst of edits costs one expansion rather than one each.
        self.expanded_stale = true;
    }

    // SAVE: persist the working entries to disk (world.jsonl + recompiled blobs).
    // The live preview is already up to date (every edit swaps it in), so SAVE is
    // purely persistence -- it does not rebuild or swap the running world. On
    // success the world is clean again; on failure it stays dirty and the next
    // SAVE retries.
    pub(super) fn save(&mut self) {
        match self.persist() {
            Ok(()) => {
                self.dirty = false;
                tracing::info!("editor: saved {}", self.world_path);
            }
            Err(e) => tracing::error!("editor: save failed: {e}"),
        }
    }

    pub(super) fn persist(&self) -> std::io::Result<()> {
        self.write_jsonl()?;
        crate::build_world_to_disk(&self.world_path)?;
        Ok(())
    }

    // Build a ready-to-run world from the in-memory entries, without touching disk
    // (SAVE owns persistence). A GraphicsConfig is seeded when the authored entries
    // alone would not render, so the preview window never goes blank.
    pub(super) fn build_preview_world(&self) -> std::io::Result<World> {
        let jsonl = crate::world::write_world_jsonl(&self.entries)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        match crate::build_world_from_str(&jsonl) {
            Ok(world) if world.renders() => Ok(world),
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
    // so a crash mid-write cannot truncate the user's world. Split from `persist`
    // so the serialization is unit-testable without the compile step.
    pub(super) fn write_jsonl(&self) -> std::io::Result<()> {
        let out = crate::world::write_world_jsonl(&self.entries)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let tmp = format!("{}.tmp", self.world_path);
        std::fs::write(&tmp, out)?;
        std::fs::rename(&tmp, &self.world_path)
    }
}

//! `EditorHook`: unique-name generation and edit persistence (SAVE, the atomic
//! write of every changed world file, and the in-memory live-preview world
//! rebuild).

use std::path::{Path, PathBuf};

use concinnity_cook::authoring::refs::{count_references, rename_references};
use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_cook::authoring::world::{ID_KEY, entry_handle, set_entry_id};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use super::{EditorHook, EntryId, EntryList, FormTarget, declared_id, entry_type};
use crate::editor::entry_list::{Refusal, build_text, paths};
use crate::editor::panels::registry::PanelKey;
use crate::editor::panels::{assets_panel, form_panel};
use crate::editor::{behavior, build_renderable, live, modal, notify, widget};

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

    // Rename `key`'s entry to `typed` (made unique against the other entries,
    // or anonymous when blank) and point every reference to its old `$id` at
    // the new one. The caller commits, so the rename and the references it
    // moved are one undo step. An entry other entries reference keeps its
    // `$id` rather than going anonymous, since nothing can name an anonymous
    // entry.
    pub(super) fn rename_entry(&mut self, key: EntryId, typed: &str) -> Renamed {
        let mut name = self.finalize_rename(typed, key);
        let Some(idx) = self.entries.index_of(key) else {
            return Renamed {
                before: None,
                name,
                moved: 0,
            };
        };
        let before = declared_id(&self.entries[idx]).map(str::to_string);
        let target = entry_type(&self.entries[idx]).and_then(RegisteredType::parse);
        let args = self.entries[idx].get("args").cloned().unwrap_or_default();
        if let (None, Some(old), Some(target)) = (&name, &before, target) {
            let naming: usize = self
                .entries
                .iter()
                .map(|e| count_references(e, target, &args, old))
                .sum();
            if naming > 0 {
                self.notifier.push(
                    notify::Level::Error,
                    &format!("'{old}' keeps its `$id`: {}", naming_note(naming)),
                );
                name.clone_from(&before);
            }
        }
        match &name {
            Some(n) => set_entry_id(&mut self.entries[idx], n),
            None => {
                if let Some(args) = self.entries[idx]
                    .get_mut("args")
                    .and_then(|a| a.as_object_mut())
                {
                    args.remove(ID_KEY);
                }
            }
        }
        let moved = match (&before, &name, target) {
            (Some(old), Some(new), Some(target)) if old != new => {
                let args = self.entries[idx].get("args").cloned().unwrap_or_default();
                self.entries
                    .iter_mut()
                    .map(|e| rename_references(e, target, &args, old, new))
                    .sum()
            }
            _ => 0,
        };
        Renamed {
            before,
            name,
            moved,
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
    // An edit the entry list refuses (see `EntryList`) is undone whole.
    pub(super) fn mark_changed(&mut self) {
        if let Some(refusal) = self.refusal() {
            self.refuse_edit(refusal);
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
        self.assets.stale = true;
        // Template baselines follow the entries too; rebuilt on demand.
        self.template_index = None;
    }

    // `mark_changed`, answering whether the edit stood: one the entry list
    // refuses is undone, and says why.
    pub(super) fn commit(&mut self) -> bool {
        let refused = self.refusal().is_some();
        self.mark_changed();
        !refused
    }

    // Why the edit since `baseline` cannot stand, if it cannot.
    pub(super) fn refusal(&self) -> Option<Refusal> {
        self.entries
            .refusal(&self.baseline, Path::new(&self.world_path))
    }

    // Put the entries back as they were before a refused edit, and say why. The
    // preview is rebuilt, since a drag may already have moved what it shows.
    fn refuse_edit(&mut self, refusal: Refusal) {
        let handle = |list: &EntryList, i| entry_handle(list, i).unwrap_or_default();
        let file = |list: &EntryList, i| {
            list.included_from(i)
                .map_or_else(|| self.world_path.clone(), paths::shown)
        };
        let message = match refusal {
            Refusal::IncludeLine(i) => format!(
                "'{}' is an Include line; edit it in {}",
                handle(&self.baseline, i),
                file(&self.baseline, i)
            ),
            Refusal::Repeated(i) => format!(
                "'{}' comes from {}, which the world includes more than once",
                handle(&self.baseline, i),
                file(&self.baseline, i)
            ),
            Refusal::Moved(i) => format!(
                "'{}' belongs with the other entries of {}",
                handle(&self.entries, i),
                file(&self.entries, i)
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
        self.assets.stale = true;
        self.template_index = None;
        self.form.close();
        self.assets.row_menu = None;
        self.assets.picker_open = false;
        self.pick_last = None;
        self.marquee = None;
        self.gizmo_drag = None;
        self.shape.drag = None;
        self.content.drag = None;
        self.create_menu = None;
        self.shaders.menu = None;
        self.follow_shader_source();
        // The Lighting panel's text controls hold committed values; re-seed so
        // they show the restored list, not the undone edit.
        if self.open[PanelKey::Lighting] {
            self.seed_lighting(world);
        }
        if self.open[PanelKey::Planet] {
            self.seed_planet(world);
        }
    }

    // SAVE: persist the working entries to disk. The world files are the
    // source of truth and the only thing a save writes -- the compiled blobs
    // under the build root belong to an explicit build, so a save never cooks
    // and never stalls behind one. The live preview is already up to date
    // (every edit refreshed it), so nothing is rebuilt or swapped here. On a
    // write failure the world stays dirty and the next SAVE retries.
    pub(super) fn save(&mut self) {
        // A world nobody has named has nowhere to go yet: ask first, and let
        // the dialog's Save come back through here once it has a path.
        if self.untitled {
            self.prompt_world_name(None);
            return;
        }
        let written = match self.write_files() {
            Ok(written) => written,
            Err(e) => {
                self.save_failed(e);
                return;
            }
        };
        self.dirty = false;
        self.saved = self.entries.clone();
        for path in &written {
            tracing::info!("editor: saved {}", path.display());
        }
        let message = match written.len() {
            0 | 1 => format!("Saved {}", self.world_path),
            2 => format!("Saved {} and 1 included file", self.world_path),
            n => format!("Saved {} and {} included files", self.world_path, n - 1),
        };
        self.notifier.success(&message);
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
        [
            assets_panel::ids(),
            form_panel::ids(),
            behavior::panel::ids(),
            modal::ids(),
        ]
        .into_iter()
        .flat_map(super::super::hud_ids::HudIds::field_ids)
        .map(|id| (id, widget::field_text(world, id)))
        .collect()
    }

    // Restore a `field_snapshot` into a freshly injected HUD.
    pub(super) fn restore_fields(world: &mut World, snapshot: &[(AssetId, String)]) {
        for (id, content) in snapshot {
            widget::seed_field(world, *id, content);
        }
    }

    // Write the world file and every included file whose entries changed since
    // the last save, each atomically (temp file + rename) so a crash mid-write
    // cannot truncate it, returning the paths written. Included files go first,
    // so a world naming a new file is never on disk without it. Directories are
    // created as needed: a fresh project saves its first world into a
    // `worlds/` that does not exist yet.
    pub(super) fn write_files(&self) -> std::io::Result<Vec<PathBuf>> {
        let mut files = self.entries.unsaved_files(&self.saved);
        if !files.contains(&None) {
            files.push(None);
        }
        let mut written = Vec::with_capacity(files.len());
        for file in files {
            let text = self.entries.file_text(file.as_deref())?;
            let path = file.map_or_else(|| PathBuf::from(&self.world_path), |f| f.to_path_buf());
            concinnity_host::store::atomic::replace(&path, |out| {
                std::io::Write::write_all(out, text.as_bytes())
            })
            .map_err(|e| std::io::Error::new(e.kind(), format!("{}: {e}", paths::shown(&path))))?;
            written.push(path);
        }
        Ok(written)
    }
}

// What a rename did: the `$id` before and after, and how many references to
// the old one it moved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Renamed {
    pub(super) before: Option<String>,
    pub(super) name: Option<String>,
    pub(super) moved: usize,
}

impl Renamed {
    // The toast for a rename that moved references, else `None`.
    pub(super) fn message(&self, kind: &str) -> Option<String> {
        let (Some(before), Some(name)) = (&self.before, &self.name) else {
            return None;
        };
        if self.moved == 0 || before == name {
            return None;
        }
        Some(format!(
            "Renamed {kind} '{before}' to '{name}'{}",
            references_note(self.moved)
        ))
    }
}

// "N references name it", for an entry that cannot go anonymous.
fn naming_note(naming: usize) -> String {
    match naming {
        1 => "1 reference names it".to_string(),
        n => format!("{n} references name it"),
    }
}

// " and the N references to it", after a rename message.
pub(super) fn references_note(moved: usize) -> String {
    match moved {
        0 => String::new(),
        1 => " and the 1 reference to it".to_string(),
        n => format!(" and the {n} references to it"),
    }
}

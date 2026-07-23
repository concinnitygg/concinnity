// src/editor/hook/import_edit.rs
//
// EditorHook: the Import panel's actions. Adding resolves the typed path
// through `authoring::add::entry_from_path` -- the exact dispatch `cn add
// <file>` runs -- so a panel import and a CLI import produce identical
// entries; resolution failures (missing file, unknown extension, invalid
// content) land on the status line and nothing is committed. Browse fills the
// same field from a native picker, so both routes end at the same Add. Listed
// imports open in the standard edit form for full arg editing.

use super::*;
use std::path::Path;

impl EditorHook {
    // The world's file-backed entries, in entry order.
    pub(super) fn import_rows(&self) -> Vec<ImportRow> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                let ty = entry_type(e)?;
                if !import_panel::IMPORT_TYPES.contains(&ty) {
                    return None;
                }
                let name = entry_name(e).unwrap_or("?");
                let source = e
                    .get("args")
                    .map(import_panel::source_of)
                    .unwrap_or_default();
                Some(ImportRow {
                    entry: i,
                    caption: format!("{name}  ({ty})  {source}"),
                })
            })
            .collect()
    }

    pub(super) fn make_import_view<'a>(
        &'a self,
        rows: &'a [ImportRow],
        mouse: [f32; 2],
    ) -> ImportView<'a> {
        ImportView {
            rows,
            scroll: self.import_scroll,
            // Focus is asserted only while frontmost, matching the other
            // panels' guard against fighting for typed keys.
            focus: self.import_focus && self.panel_order.last() == Some(&PanelKey::Import),
            status: self.import_status.as_ref(),
            mouse,
        }
    }

    pub(super) fn scroll_imports(&mut self, delta: f32) {
        let rows_shown = import_panel::visible_rows(self.effective_size(PanelKey::Import)[1]);
        let max = self.import_rows().len().saturating_sub(rows_shown);
        self.import_scroll = scroll_step(self.import_scroll, delta, max);
    }

    // Route a resolved Import-panel click.
    pub(super) fn apply_import_action(&mut self, action: ImportAction, world: &mut World) {
        match action {
            ImportAction::FocusPath => self.import_focus = true,
            ImportAction::Browse => self.browse_import(world),
            ImportAction::Add => self.add_import(world),
            ImportAction::Open(i) => {
                if let Some(row) = self.import_rows().get(i) {
                    self.open_import(row.entry, world);
                }
            }
            // A click on panel chrome blurs the path field.
            ImportAction::Consume => self.import_focus = false,
        }
    }

    // Open the native file picker at the project directory and take what the
    // user picks. The dialog blocks the frame loop while it is up (it is a
    // modal OS window over the editor); cancelling leaves the field alone.
    // This wrapper is the only untestable step -- what the pick feeds is all in
    // `accept_browsed_path`.
    pub(super) fn browse_import(&mut self, world: &mut World) {
        let root = project_root();
        if let Some(picked) = super::super::file_dialog::pick_import_file(&root) {
            self.accept_browsed_path(world, &picked);
        }
    }

    // Put a picked file's project-relative path in the field, ready to Add. Not
    // added outright: the user still sees what resolved before committing.
    pub(super) fn accept_browsed_path(&mut self, world: &mut World, picked: &Path) {
        let text = super::super::file_dialog::project_path(picked, &project_root());
        widget::seed_field(world, import_panel::PATH_INPUT, &text);
        self.import_focus = true;
        self.import_status = None;
    }

    // Resolve the typed path and append its entries (renamed to stay unique),
    // exactly as `cn add <path>` would. Success clears the field; failure
    // shows on the status line and commits nothing.
    pub(super) fn add_import(&mut self, world: &mut World) {
        self.import_status = None;
        let path = widget::field_text(world, import_panel::PATH_INPUT)
            .trim()
            .to_string();
        if path.is_empty() {
            return;
        }
        // Scene entries defer reading their file to the cook, which would
        // surface a typo only as a preview-rebuild log; catch it here instead.
        if !std::path::Path::new(&path).is_file() {
            let msg = format!("{path}: no such file");
            self.console_sink.error(&format!("import: {msg}"));
            self.import_status = Some(ImportStatus::Error(short_status(&msg)));
            return;
        }
        let mut new_entries = match crate::authoring::entry_from_path(&path) {
            Ok(entries) => entries,
            Err(e) => {
                // The console keeps the full error; the status line clips it.
                self.console_sink.error(&format!("import: {e}"));
                self.import_status = Some(ImportStatus::Error(short_status(&e.to_string())));
                return;
            }
        };
        // An `.hdr` into a world that already has a lighting environment
        // retargets it rather than appending a second one the runtime would
        // ignore, matching `cn add <file>`.
        if let Some((name, source)) =
            crate::authoring::try_retarget_environment_map(&mut self.entries, &new_entries)
        {
            self.import_status = Some(ImportStatus::Notice(short_status(&format!(
                "{name} now uses {source}"
            ))));
            self.mark_changed();
            widget::seed_field(world, import_panel::PATH_INPUT, "");
            return;
        }
        for entry in &mut new_entries {
            let base = entry_name(entry).unwrap_or("import").to_string();
            let unique = self.unique_from(&base);
            entry["name"] = serde_json::Value::String(unique);
            self.entries.push(entry.clone());
        }
        self.mark_changed();
        widget::seed_field(world, import_panel::PATH_INPUT, "");
    }

    // Open import entry `idx` in the standard add / edit form (part of the
    // Assets UI, so the browse panel comes up alongside it).
    pub(super) fn open_import(&mut self, idx: usize, world: &mut World) {
        let Some(ty) = self.entries.get(idx).and_then(entry_type).map(String::from) else {
            return;
        };
        self.panel_open = true;
        self.open_form(world, ty, FormTarget::Entry(idx));
    }

    // Enter in the focused path field adds, like clicking the Add button.
    pub(super) fn import_keys(&mut self, world: &mut World, input: &FrameInput) {
        if self.import_focus && input.captured_key == Some(crate::assets::Key::Enter) {
            self.add_import(world);
        }
    }
}

// The directory imported paths are stored relative to, and where the picker
// opens: the process's working directory, which is the project root the cook
// resolves an entry's `source` against.
fn project_root() -> std::path::PathBuf {
    std::env::current_dir().unwrap_or_default()
}

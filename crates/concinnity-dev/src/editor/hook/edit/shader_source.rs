//! `EditorHook`: the Shader source panel's actions. The panel is a code text
//! area over one of a Shader's `.hlsl` files or an `SdfVolume`'s distance field.
//! Save (the button or the save shortcut) writes the file and nothing else:
//! the hot-reload watcher sees the write and recompiles the Shader, or every
//! volume reading the field, exactly as it does for a save from any other
//! editor, and the outcomes it publishes mark the lines their diagnostics name.
//! Leaving a file with unsaved edits asks first; a change on disk is followed
//! while the buffer is clean.

use concinnity_engine::ecs::World;
use concinnity_engine::render::shader::programs::vocabulary::{ENTRIES, Entry};

use super::shaders_state::{SourceState, vocabulary_of};
use crate::debug::hot_reload::ReloadSubject;
use crate::editor::hook::EditorHook;
use crate::editor::notify;
use crate::editor::panels::registry::PanelKey;
use crate::editor::panels::sdf;
use crate::editor::panels::shader;
use crate::editor::panels::shader::diagnostics::{Status, Tone};
use crate::editor::panels::shader::reference::{RefClick, RefRow};
use crate::editor::panels::shader::source::{Leave, SourceKey};
use crate::editor::panels::shader::source_panel::{SourceAction, SourceView};
use crate::editor::ui::BuiltPanel;

impl EditorHook {
    // Open `key`'s file in the source panel, asking first when that would
    // leave unsaved edits behind.
    pub(in crate::editor::hook) fn open_shader_file(&mut self, key: SourceKey) {
        self.leave_shader_source(Leave::Open(key));
    }

    // The panel's close button: the same question over unsaved edits.
    pub(in crate::editor::hook) fn close_shader_source(&mut self) {
        self.leave_shader_source(Leave::Close);
    }

    pub(in crate::editor::hook) fn leave_shader_source(&mut self, then: Leave) {
        if let Some(src) = &self.shaders.source
            && shader::source::must_ask(src.area.is_dirty(), &src.key, &then)
        {
            let prompt = shader::source::leave_prompt(src.file_name());
            self.open_modal(&prompt, shader::source::leave_buttons(then));
            return;
        }
        self.go_to_leave(then);
    }

    // The confirmation dialog's answer: write first when `save`, and leave
    // only when that write landed.
    pub(in crate::editor::hook) fn answer_leave_shader_source(&mut self, save: bool, then: Leave) {
        if save && !self.save_shader_source() {
            return;
        }
        let confirm = then == Leave::ConfirmForm;
        self.go_to_leave(then);
        if confirm {
            self.confirm_form();
        }
    }

    fn go_to_leave(&mut self, then: Leave) {
        match then {
            Leave::Close | Leave::ConfirmForm => self.shaders.source = None,
            Leave::Open(key) => self.load_shader_source(key),
            Leave::Edit(edit) => {
                self.shaders.source = None;
                self.apply_shader_edit(edit);
            }
        }
    }

    // Read `key`'s file into the panel and bring it to the front. Reopening
    // the open file keeps its buffer.
    fn load_shader_source(&mut self, key: SourceKey) {
        if self.shaders.source.as_ref().is_some_and(|s| s.key == key) {
            self.focus_panel(PanelKey::ShaderSource);
            return;
        }
        let path = match &key {
            SourceKey::Shader { name, stage } => self
                .declared_shaders()
                .into_iter()
                .find(|s| &s.name == name)
                .and_then(|s| s.file(*stage).map(|f| f.path.clone())),
            SourceKey::Field { path } => self
                .declared_fields()
                .into_iter()
                .find(|f| &f.path == path)
                .map(|f| f.path),
        };
        let Some(path) = path else {
            return;
        };
        let (text, status) = match std::fs::read_to_string(&path) {
            Ok(text) => (text, None),
            Err(e) => (
                String::new(),
                Some(Status::new(format!("{path}: {e}"), Tone::Error)),
            ),
        };
        let mut src = SourceState::new(key, path, text);
        src.status = status;
        src.focus = true;
        self.shaders.source = Some(src);
        self.focus_panel(PanelKey::ShaderSource);
    }

    // Write the buffer to its file. `false` when the write failed, which the
    // status line and a toast then say.
    pub(in crate::editor::hook) fn save_shader_source(&mut self) -> bool {
        let Some(src) = self.shaders.source.as_mut() else {
            return false;
        };
        let text = src.area.text();
        if let Err(e) = std::fs::write(&src.path, &text) {
            let message = format!("{}: {e}", src.path);
            src.status = Some(Status::new(message.clone(), Tone::Error));
            self.notifier.error_with(
                &format!("Save failed: {message}"),
                notify::Action::OpenConsole,
            );
            return false;
        }
        src.saved(text);
        // A file read by something outside the running world's catalog has no
        // recompile to wait on; the rebuild compiles it from what was written.
        if src.live == Some(false) {
            self.require_rebuild();
        }
        true
    }

    // What the open file's reports are published under: its Shader, or every
    // volume reading the field.
    fn source_subjects(&mut self) -> Vec<ReloadSubject> {
        match self.shaders.source.as_ref().map(|s| s.key.clone()) {
            Some(SourceKey::Shader { name, .. }) => vec![ReloadSubject::shader(&name)],
            Some(SourceKey::Field { path }) => self
                .declared_fields()
                .iter()
                .find(|f| f.path == path)
                .map_or_default(sdf::field_list::subjects),
            None => Vec::new(),
        }
    }

    // Once a frame: take in the latest reload outcomes, and follow the file on
    // disk.
    pub(in crate::editor::hook) fn drive_shader_source(&mut self) {
        let now = self.clock.elapsed().as_secs_f64();
        if self.shaders.source.is_none() {
            return;
        }
        let subjects = self.source_subjects();
        let Some(src) = self.shaders.source.as_mut() else {
            return;
        };
        if src.take_board(&self.shaders.reports.snapshot(), &subjects)
            && let Some((line, column)) = src.jump
        {
            src.area.go_to(line, column);
        }
        src.check_disk(now);
    }

    // Whether the text area holds the keyboard: focused by a press and the
    // panel frontmost.
    pub(in crate::editor::hook) fn shader_typing(&self) -> bool {
        self.shaders.source.as_ref().is_some_and(|s| s.focus)
            && self.panel_order.last() == Some(&PanelKey::ShaderSource)
    }

    // The status the panel shows: its own, or a note that what reads the file
    // is not in the running world yet.
    fn shader_status(src: &SourceState) -> Option<(&str, Tone)> {
        src.status
            .as_ref()
            .map(|s| (s.text.as_str(), s.tone))
            .or_else(|| {
                (src.live == Some(false)).then_some((
                    "Not in the running world yet; it joins when the world rebuilds",
                    Tone::Info,
                ))
            })
    }

    // The vocabulary the open file's reference column lists.
    fn reference_entries(&self) -> &'static [Entry] {
        self.shaders
            .source
            .as_ref()
            .map_or(ENTRIES, |s| vocabulary_of(&s.key))
    }

    // A key that changes whenever the reference column's rows do: the open
    // file's kind and the folded groups.
    fn reference_key(&self) -> i32 {
        let kind = match self.shaders.source.as_ref().map(|s| &s.key) {
            Some(SourceKey::Shader { stage, .. }) => *stage as i32 + 1,
            Some(SourceKey::Field { .. }) => 16,
            None => 0,
        };
        kind << 8 | i32::from(self.shaders.reference.folds())
    }

    // Show the panel, reading the reference column's rows only when they
    // changed.
    pub(in crate::editor::hook) fn sync_shader_source(
        &self,
        world: &mut World,
        built: &BuiltPanel,
    ) {
        let Some(src) = &self.shaders.source else {
            return;
        };
        let reference = &self.shaders.reference;
        let entries = self.reference_entries();
        // A hovered reference name says what it is on the status line.
        let hover = shader::source_panel::reference_hovered(world, built)
            .filter(|_| reference.open)
            .and_then(|i| match reference.row_at(entries, i)? {
                RefRow::Name(e) => Some([e.signature, e.summary]),
                RefRow::Heading { .. } => None,
            });
        let view = SourceView {
            title: src.title_parts(),
            path: &src.path,
            dirty: src.area.is_dirty(),
            focus: self.shader_typing(),
            hover,
            status: Self::shader_status(src),
            markers: &src.markers,
            reference: reference.open.then(|| self.reference_key()),
        };
        shader::source_panel::sync(&view, || reference.rows(entries), world, built);
    }

    // Take in a press that focused the text area, or left it.
    pub(in crate::editor::hook) fn observe_shader_source(
        &mut self,
        world: &World,
        built: &BuiltPanel,
    ) {
        let focused = shader::source_panel::area_focused(world, built);
        let shown = self.shader_typing();
        if focused != shown
            && let Some(src) = self.shaders.source.as_mut()
        {
            src.focus = focused;
        }
    }

    // Act on what the user asked of the panel.
    pub(in crate::editor::hook) fn apply_source_action(&mut self, action: SourceAction) {
        match action {
            SourceAction::Save => {
                self.save_shader_source();
            }
            SourceAction::Status => {
                if let Some(src) = self.shaders.source.as_mut()
                    && let Some((line, column)) = src.jump
                {
                    src.area.go_to(line, column);
                    src.focus = true;
                }
            }
            SourceAction::ToggleReference => {
                self.shaders.reference.open ^= true;
            }
            SourceAction::Reference(i) => self.press_reference(i),
        }
    }

    // A press on the reference column's row `i`: a heading folds its group, a
    // name goes in at the caret and the text takes the keyboard.
    fn press_reference(&mut self, i: usize) {
        let entries = self.reference_entries();
        let Some(row) = self.shaders.reference.row_at(entries, i) else {
            return;
        };
        match shader::reference::click(&row) {
            RefClick::Fold(group) => self.shaders.reference.toggle(group),
            RefClick::Insert(text) => {
                if let Some(src) = self.shaders.source.as_mut() {
                    src.area.insert(&text);
                    src.focus = true;
                }
            }
        }
    }
}

//! EditorHook: the Shader source panel's actions. The panel is a code text
//! area over one of a Shader's `.hlsl` files. Save (the button or the save
//! shortcut) writes the file and nothing else: the hot-reload watcher sees the
//! write and recompiles the Shader, exactly as it does for a save from any
//! other editor, and the outcome it publishes marks the lines its diagnostics
//! name. Leaving a file with unsaved edits asks first; a change on disk is
//! followed while the buffer is clean.

use concinnity_core::components::FrameInput;
use concinnity_core::ecs::World;

use super::shaders_state::SourceState;
use crate::editor::hook::EditorHook;
use crate::editor::notify;
use crate::editor::panels::registry::PanelKey;
use crate::editor::panels::shader_diagnostics::{Status, Tone};
use crate::editor::panels::shader_source::{self, Leave, SourceKey};
use crate::editor::panels::shader_source_panel::{self, SourceAction, SourceView};
use crate::editor::text_area::clipboard;
use crate::editor::text_area::keys::Platform;
use crate::editor::text_area::layout::{self, Geometry, Metrics, TextAreaView};

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
            && shader_source::must_ask(src.area.is_dirty(), &src.key, &then)
        {
            let prompt = shader_source::leave_prompt(src.file_name());
            self.open_modal(&prompt, shader_source::leave_buttons(then));
            return;
        }
        self.go_to_leave(then);
    }

    // The confirmation dialog's answer: write first when `save`, and leave
    // only when that write landed.
    pub(in crate::editor::hook) fn answer_leave_shader_source(
        &mut self,
        save: bool,
        then: Leave,
        world: &mut World,
    ) {
        if save && !self.save_shader_source() {
            return;
        }
        let confirm = then == Leave::ConfirmForm;
        self.go_to_leave(then);
        if confirm {
            self.confirm_form(world);
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
        let Some(file) = self
            .declared_shaders()
            .into_iter()
            .find(|s| s.name == key.shader)
            .and_then(|s| s.file(key.stage).cloned())
        else {
            return;
        };
        let (text, status) = match std::fs::read_to_string(&file.path) {
            Ok(text) => (text, None),
            Err(e) => (
                String::new(),
                Some(Status::new(format!("{}: {e}", file.path), Tone::Error)),
            ),
        };
        let mut src = SourceState::new(key, file.path, text);
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
                &format!("Shader save failed: {message}"),
                notify::Action::OpenConsole,
            );
            return false;
        }
        src.saved(text);
        // A Shader outside the running world's catalog has no recompile to
        // wait on; the rebuild compiles it from the file just written.
        if src.live == Some(false) {
            self.require_rebuild();
        }
        true
    }

    // Once a frame: take in the latest reload outcomes, and follow the file on
    // disk.
    pub(in crate::editor::hook) fn drive_shader_source(&mut self) {
        let now = self.clock.elapsed().as_secs_f64();
        let Some(src) = self.shaders.source.as_mut() else {
            return;
        };
        if src.take_board(&self.shaders.reports.snapshot())
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

    // The text area laid out in the panel as it stands, with the view it
    // shows pushed into the area so its scrolling keeps the caret on screen.
    fn shader_geometry(&mut self) -> Option<Geometry> {
        let vp = self.viewport;
        let o = self.origin(PanelKey::ShaderSource, vp);
        let s = self.effective_size(PanelKey::ShaderSource);
        let src = self.shaders.source.as_mut()?;
        let g = shader_source_panel::area_geometry(o, s, &src.area, Metrics::code());
        src.area.set_view(g.view());
        Some(g)
    }

    pub(in crate::editor::hook) fn scroll_shader_source(&mut self, delta: f32) {
        self.shader_geometry();
        let shift = self.shift_held;
        if let Some(src) = self.shaders.source.as_mut() {
            src.area.wheel(delta, shift);
        }
    }

    // The status the panel shows: its own, or a note that the Shader is not
    // in the running world yet.
    fn shader_status(src: &SourceState) -> Option<Status> {
        src.status.clone().or_else(|| {
            (src.live == Some(false)).then(|| {
                Status::new(
                    "Not in the running world yet; it joins when the world rebuilds",
                    Tone::Info,
                )
            })
        })
    }

    // The panel's view, with its title and status owned by the caller.
    pub(in crate::editor::hook) fn make_source_view<'a>(
        &'a self,
        src: &'a SourceState,
        title: &'a str,
        status: Option<&'a Status>,
        mouse: [f32; 2],
    ) -> SourceView<'a> {
        SourceView {
            title,
            path: &src.path,
            area: &src.area,
            focus: self.shader_typing(),
            status,
            markers: &src.markers,
            mouse,
        }
    }

    pub(in crate::editor::hook) fn draw_shader_source(
        &self,
        world: &mut World,
        o: [f32; 2],
        mouse: [f32; 2],
    ) {
        let Some(src) = &self.shaders.source else {
            shader_source_panel::hide_all(world);
            return;
        };
        let s = self.effective_size(PanelKey::ShaderSource);
        let title = src.title();
        let status = Self::shader_status(src);
        let view = self.make_source_view(src, &title, status.as_ref(), mouse);
        shader_source_panel::place(world, Some(&view), o, s, Metrics::code());
    }

    // Route a resolved source-panel click at `(mx, my)`. A press on a gutter
    // marker jumps to where its diagnostic points.
    pub(in crate::editor::hook) fn apply_source_action(
        &mut self,
        action: SourceAction,
        mx: f32,
        my: f32,
    ) {
        match action {
            SourceAction::Text => {
                let Some(g) = self.shader_geometry() else {
                    return;
                };
                let now = self.clock.elapsed().as_secs_f64();
                let shift = self.shift_held;
                let Some(src) = self.shaders.source.as_mut() else {
                    return;
                };
                let marker = {
                    let view = TextAreaView {
                        area: &src.area,
                        focused: true,
                        markers: &src.markers,
                        mouse: [mx, my],
                    };
                    layout::hovered_marker(&view, &g).map(|m| (m.line, m.column))
                };
                match marker {
                    Some((line, column)) => src.area.go_to(line, column),
                    None => {
                        src.area.press_at(&g, mx, my, shift, now);
                    }
                }
                src.focus = true;
            }
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
            // A click on panel chrome blurs the text area.
            SourceAction::Consume => {
                if let Some(src) = self.shaders.source.as_mut() {
                    src.focus = false;
                }
            }
        }
    }

    // The per-frame input while the panel is frontmost: a held press keeps
    // dragging, and while the area holds the keyboard every key event of the
    // frame edits it, in order. The save shortcut saves.
    pub(in crate::editor::hook) fn shader_source_keys(
        &mut self,
        world: &mut World,
        input: &FrameInput,
    ) {
        let Some(g) = self.shader_geometry() else {
            return;
        };
        let typing = self.shader_typing();
        let Some(src) = self.shaders.source.as_mut() else {
            return;
        };
        if src.area.pointer_busy() {
            src.area
                .pointer(&g, input.mouse_x, input.mouse_y, input.left_button_down);
        }
        if !typing || input.key_events.is_empty() {
            return;
        }
        let clipboard = clipboard::system_or(world, &mut self.text_clipboard);
        let response = src
            .area
            .handle_events(&input.key_events, Platform::current(), clipboard);
        if response.save {
            self.save_shader_source();
        }
    }
}

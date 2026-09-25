//! EditorHook: the Shaders panel's actions. The panel lists every Shader the
//! working entries declare; a file row opens the source panel on that file
//! (`shader_source.rs`), a Materials row selects the Materials naming the
//! Shader, and "+ New Shader" writes a starter fragment file and adds its
//! `Shader` entry like any other added asset.

use concinnity_core::components::ShaderStage;
use concinnity_core::ecs::World;
use std::path::{Path, PathBuf};

use super::shaders_state::RowsKey;
use crate::debug::hot_reload::ShaderReports;
use crate::editor::hook::EditorHook;
use crate::editor::notify;
use crate::editor::panels::registry::PanelKey;
use crate::editor::panels::shader_list::{self, Row, RowKind, ShaderDecl};
use crate::editor::panels::shader_list_panel::{self, ShadersAction, ShadersView};
use crate::editor::panels::shader_source::{self, SourceKey};

impl EditorHook {
    // A clone of the board the hot-reload driver publishes each Shader's
    // latest outcome to (see `run_editor`).
    pub(crate) fn shader_reports(&self) -> ShaderReports {
        self.shaders.reports.clone()
    }

    // Every Shader the working entries declare, each file resolved once and
    // kept (`ShadersState::resolved`).
    pub(in crate::editor::hook) fn declared_shaders(&mut self) -> Vec<ShaderDecl> {
        let dir = crate::project::assets_dir();
        let state = &mut self.shaders;
        shader_list::declared(&self.entries, |declared| {
            state.resolved(declared, dir.as_deref())
        })
    }

    // The list's rows, rebuilt only when the declared Shaders, the board, or
    // the open file changed since they were last built: a row's status
    // compares paths on disk, which is no work for every frame.
    pub(in crate::editor::hook) fn shader_rows(&mut self) -> &[Row] {
        let key = RowsKey {
            shaders: self.declared_shaders(),
            seq: self.shaders.reports.seq(),
            open: self.shaders.source.as_ref().map(|s| s.key.clone()),
        };
        let state = &mut self.shaders;
        if state.rows_key.as_ref() != Some(&key) {
            state.rows =
                shader_list::rows(&key.shaders, &state.reports.snapshot(), key.open.as_ref());
            state.rows_key = Some(key);
        }
        &state.rows
    }

    pub(in crate::editor::hook) fn make_shaders_view<'a>(
        &self,
        rows: &'a [Row],
        mouse: [f32; 2],
    ) -> ShadersView<'a> {
        ShadersView {
            rows,
            scroll: self.shaders.scroll,
            mouse,
        }
    }

    pub(in crate::editor::hook) fn scroll_shaders(&mut self, delta: f32) {
        let window = shader_list_panel::rows_for_height(self.effective_size(PanelKey::Shaders)[1]);
        let max = shader_list::row_count(&self.entries).saturating_sub(window);
        self.shaders.scroll = crate::editor::hook::scroll_step(self.shaders.scroll, delta, max);
    }

    // Route a resolved Shaders-panel click on `rows`.
    pub(in crate::editor::hook) fn apply_shaders_action(
        &mut self,
        action: ShadersAction,
        rows: &[Row],
        world: &mut World,
    ) {
        let ShadersAction::Row(i) = action else {
            return;
        };
        match rows.get(i).map(|r| &r.kind) {
            Some(RowKind::File(key)) => self.open_shader_file(key.clone()),
            Some(&RowKind::Materials(shader)) => self.select_shader_materials(shader, world),
            Some(RowKind::New) => self.create_shader(),
            Some(RowKind::Header | RowKind::Note) | None => {}
        }
    }

    // Select the Materials naming declared Shader `i`.
    fn select_shader_materials(&mut self, i: usize, world: &mut World) {
        let Some(shader) = self.declared_shaders().into_iter().nth(i) else {
            return;
        };
        let positions = shader.materials.iter().map(|&(m, _)| m).collect();
        let handles = self.entry_handles(positions);
        if handles.is_empty() {
            return;
        }
        self.selection.set(handles);
        self.follow_active(world);
    }

    // Write a starter fragment file under the project's assets and add a
    // `Shader` entry reading it, then open the file. The entry is a normal
    // world edit (dirty, undoable, previewed by a rebuild); the file is
    // written now, since the build reads Shader files from disk.
    pub(in crate::editor::hook) fn create_shader(&mut self) {
        if !shader_list::can_add_shader(shader_list::shader_count(&self.entries)) {
            self.notifier.push(
                notify::Level::Error,
                &format!("No Shader added: {}", shader_list::limit_reason()),
            );
            return;
        }
        let name = self.unique_name("shader");
        let dir = crate::project::assets_dir().unwrap_or_else(|| PathBuf::from("assets"));
        let path = shader_source::starter_path(&dir, &name, Path::exists);
        let written = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(&path, shader_source::STARTER_SHADER));
        if let Err(e) = written {
            self.notifier.error_with(
                &format!("Could not write {}: {e}", path.display()),
                notify::Action::OpenConsole,
            );
            return;
        }
        let root = std::env::current_dir().unwrap_or_default();
        let declared = shader_source::declared_form(&path, &root);
        self.entries.push(serde_json::json!({
            "type": "Shader", "args": { "$id": name, "fragment": declared },
        }));
        self.mark_changed();
        self.notifier
            .success(&format!("Added Shader '{name}' ({declared})"));
        self.open_shader_file(SourceKey {
            shader: name,
            stage: ShaderStage::Fragment,
        });
    }
}

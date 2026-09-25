//! EditorHook: the Shaders panel's actions. The panel lists every Shader the
//! working entries declare; a heading or its menu's Edit opens the Shader's
//! form and "+ New Shader" an empty one (`shader_form.rs`), a file row opens
//! the source panel on that file (`shader_source.rs`), a Materials row selects
//! the Materials naming the Shader, a heading's Duplicate copies it with its
//! files, and the rest (a delete, adding or removing a vertex file) are edits
//! in `shader_edits.rs`.

use concinnity_core::ecs::World;

use super::shaders_state::RowsKey;
use crate::debug::hot_reload::ShaderReports;
use crate::editor::asset_handle::AssetHandle;
use crate::editor::hook::{EditorHook, FormTarget};
use crate::editor::panels::registry::PanelKey;
use crate::editor::panels::shader_list::{self, MenuItem, Row, RowKind, ShaderDecl};
use crate::editor::panels::shader_list_panel::{self, ShadersAction, ShadersView};

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
        let menu = self.shaders.menu.as_ref();
        ShadersView {
            rows,
            scroll: self.shaders.scroll,
            mouse,
            menu: menu.and_then(|kind| rows.iter().position(|r| &r.kind == kind)),
        }
    }

    pub(in crate::editor::hook) fn scroll_shaders(&mut self, delta: f32) {
        let window = shader_list_panel::rows_for_height(self.effective_size(PanelKey::Shaders)[1]);
        let max = shader_list::row_count(&self.entries).saturating_sub(window);
        self.shaders.scroll = crate::editor::hook::scroll_step(self.shaders.scroll, delta, max);
        self.shaders.menu = None;
    }

    // Route a resolved Shaders-panel click on `rows`.
    pub(in crate::editor::hook) fn apply_shaders_action(
        &mut self,
        action: ShadersAction,
        rows: &[Row],
        world: &mut World,
    ) {
        let i = match action {
            ShadersAction::Row(i) => i,
            ShadersAction::OpenMenu(i) => {
                self.shaders.menu = rows.get(i).map(|r| r.kind.clone());
                return;
            }
            ShadersAction::Menu(item) => {
                if let Some(kind) = self.shaders.menu.take() {
                    self.apply_menu_item(item, kind, world);
                }
                return;
            }
            ShadersAction::CloseMenu => {
                self.shaders.menu = None;
                return;
            }
            ShadersAction::Consume => return,
        };
        match rows.get(i).map(|r| &r.kind) {
            Some(RowKind::File(key)) => self.open_shader_file(key.clone()),
            Some(&RowKind::Materials(shader)) => self.select_shader_materials(shader, world),
            Some(&RowKind::AddVertex(shader)) => self.add_vertex_file(shader),
            Some(RowKind::New) => self.open_shader_form(None, world),
            Some(&RowKind::Header(shader)) => self.open_shader_form(Some(shader), world),
            Some(RowKind::Note) | None => {}
        }
    }

    fn apply_menu_item(&mut self, item: MenuItem, kind: RowKind, world: &mut World) {
        match (item, kind) {
            (MenuItem::Edit, RowKind::Header(i)) => self.open_shader_form(Some(i), world),
            (MenuItem::Duplicate, RowKind::Header(i)) => self.duplicate_shader(i),
            (MenuItem::Delete, RowKind::Header(i)) => self.confirm_delete_shader(i),
            (MenuItem::RemoveVertex, RowKind::File(key)) => self.remove_vertex_file(&key.shader),
            _ => {}
        }
    }

    // Open the Shader form on declared Shader `i`, or empty for a new one. It
    // shows while this panel does.
    fn open_shader_form(&mut self, i: Option<usize>, world: &mut World) {
        let target = match i {
            None => FormTarget::New,
            Some(i) => {
                let Some(shader) = self.declared_shaders().into_iter().nth(i) else {
                    return;
                };
                let found =
                    concinnity_cook::authoring::world::find_entry(&self.entries, &shader.name);
                let Some(key) = found.and_then(|idx| self.entries.key_at(idx)) else {
                    return;
                };
                FormTarget::Entry(key)
            }
        };
        self.open_form(world, "Shader".to_string(), target);
        self.form.host = PanelKey::Shaders;
    }

    // Duplicate declared Shader `i` with its files, as Ctrl+D would.
    fn duplicate_shader(&mut self, i: usize) {
        let Some(shader) = self.declared_shaders().into_iter().nth(i) else {
            return;
        };
        let found = concinnity_cook::authoring::world::find_entry(&self.entries, &shader.name);
        if let Some(key) = found.and_then(|idx| self.entries.key_at(idx)) {
            self.duplicate(vec![AssetHandle::Entry(key)]);
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
}

//! The Import panel: a front end over the same file resolution `cn add <file>`
//! uses (`authoring/add.rs::entry_from_path`), so importing from the panel and
//! from the CLI produce identical entries. The header takes a source path, a
//! Browse button, and an Add button (Enter also adds); a status line shows
//! resolution errors; the body lists the world's existing file-backed entries
//! (scene / story imports, textures, audio, fonts, shaders, ...) and selecting
//! one opens it in the standard edit form. `hook/edit/import.rs` owns the
//! actions.

use concinnity_engine::components::{
    List, TextAlign, TextField, TextLabel, UiEvent, UiItem, UiValue,
};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use super::registry::PanelKey;
use crate::editor::theme;
use crate::editor::ui::build::BODY_PAD;
use crate::editor::ui::{BuiltPanel, PanelSpec, UiBuilder, panel_height, state};

// The file-backed asset types `entry_from_path` produces, in the order their
// entries list in the panel body. TextLabel is deliberately absent: a text
// import copies the file body into `content` and keeps no path to reopen.
pub(crate) const IMPORT_TYPES: &[&str] = &[
    "SceneImport",
    "StoryImport",
    "File",
    "EnvironmentMap",
    "AudioClip",
    "Font",
    "Shader",
    "LLM",
];

// The arg carrying an import entry's file path (the key varies by type).
pub(crate) fn source_of(args: &serde_json::Value) -> String {
    for key in ["source", "path", "model_path"] {
        if let Some(s) = args.get(key).and_then(|v| v.as_str())
            && !s.is_empty()
        {
            return s.to_string();
        }
    }
    String::new()
}

// The default (and minimum) panel width; the user can widen it past this.
pub(crate) const IMPORT_W: f32 = 520.0;
const HEADER_H: f32 = 32.0;
const LINE_H: f32 = 22.0;
const BTN_W: f32 = 70.0;
const BROWSE_W: f32 = 84.0;
const ROW_H: f32 = 26.0;
// Rows shown at the default height; a longer list scrolls.
const IMPORT_POOL: usize = 10;

const HINT_COLOR: [f32; 3] = [0.55, 0.58, 0.66];
const HEADER_LABEL: [f32; 3] = [0.70, 0.74, 0.84];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];
const NOTICE_LABEL: [f32; 3] = [0.55, 0.85, 0.65];

// The state the panel shows, and the events its controls send.
const ROWS: &str = "editor.import.rows";
const SHOWN: [&str; 2] = ["editor.import.shown.hi", "editor.import.shown.lo"];
const HEADING: &str = "editor.import.heading";
const STATUS: &str = "editor.import.status";
const PICK: &str = "editor.import.pick";
const BROWSE: &str = "editor.import.browse";
const ADD: &str = "editor.import.add";

// The panel's slots.
const PATH_SLOT: usize = 0;
const STATUS_SLOT: usize = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ImportStatus {
    Error(String),
    Notice(String),
}

impl ImportStatus {
    pub(crate) fn text(&self) -> &str {
        match self {
            Self::Error(s) | Self::Notice(s) => s,
        }
    }

    fn color(&self) -> [f32; 3] {
        match self {
            Self::Error(_) => ERROR_LABEL,
            Self::Notice(_) => NOTICE_LABEL,
        }
    }
}

// One file-backed entry of the world: where it is in the entries, and what its
// row reads.
pub(crate) struct ImportRow {
    pub entry: usize,
    pub caption: String,
}

// What the user asked of the panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImportAction {
    Browse,
    Add,
    // Row `i` of the list.
    Open(usize),
}

// The action an Import event asks for, if `event` is one.
pub(crate) fn action(event: &UiEvent) -> Option<ImportAction> {
    Some(match event.name.as_str() {
        BROWSE => ImportAction::Browse,
        ADD => ImportAction::Add,
        PICK => match event.value {
            Some(UiValue::Int(i)) => ImportAction::Open(usize::try_from(i).ok()?),
            _ => return None,
        },
        _ => return None,
    })
}

pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [(vw - IMPORT_W) * 0.5, crate::editor::hud::body_top() + 40.0]
}

pub(crate) fn size() -> [f32; 2] {
    [
        IMPORT_W,
        panel_height(HEADER_H + 3.0 * LINE_H + IMPORT_POOL as f32 * ROW_H),
    ]
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let path = ui.text_field(TextField {
        placeholder: "path/to/file.glb".to_string(),
        height: HEADER_H - 8.0,
        ..Default::default()
    })?;
    let path_id = path.item?.id();
    let browse = ui.button("Browse...", BROWSE, None, [BROWSE_W, HEADER_H - 8.0])?;
    let add = ui.button("Add", ADD, None, [BTN_W, HEADER_H - 8.0])?;
    let header = ui.row(vec![path, browse, add], HEADER_H)?;
    let hint = ui.clipped_label(
        "scenes (glb, fbx) - stories (md) - images - hdr - audio - fonts - shaders - text",
        HINT_COLOR,
        IMPORT_W - 2.0 * BODY_PAD,
    )?;
    let hint = ui.row(vec![hint], LINE_H)?;
    let status = ui.bound_label(STATUS, ERROR_LABEL, TextAlign::Left)?;
    let status_id = status.item?.id();
    let status = ui.row(vec![status], LINE_H)?;
    let heading = ui.bound_label(HEADING, HEADER_LABEL, TextAlign::Left)?;
    let heading = ui.row(vec![heading], LINE_H)?;
    let rows = ui.list(List {
        items: ROWS.to_string(),
        row_height: ROW_H,
        on_select: PICK.to_string(),
        ..Default::default()
    })?;
    let spec = PanelSpec {
        key: PanelKey::Import,
        title: "Import",
        title_state: None,
        size: size(),
        min_size: Some(size()),
    };
    let mut panel = ui.panel(&spec, vec![header, hint, status, heading, rows])?;
    panel.slots = vec![path_id, status_id];
    Some(panel)
}

// The path field's id.
pub(crate) fn path_field(built: &BuiltPanel) -> Option<AssetId> {
    built.slots.get(PATH_SLOT).copied()
}

// What the panel shows this frame.
pub(crate) struct ImportView<'a> {
    // Which version of the entries the rows are read from.
    pub revision: u64,
    pub focus: bool,
    pub status: Option<&'a ImportStatus>,
}

// Show the rows, rebuilt only when the entries changed, the status line, and
// the field's focus. `rows` reads them when they did.
pub(crate) fn sync(
    view: &ImportView<'_>,
    rows: impl FnOnce() -> Vec<ImportRow>,
    world: &mut World,
    built: &BuiltPanel,
) {
    if !state::is_shown(world, SHOWN, view.revision) {
        let rows = rows();
        let heading = if rows.is_empty() {
            "Imports (none yet)".to_string()
        } else {
            format!("Imports ({})", rows.len())
        };
        let items = rows.iter().map(|r| UiItem::new(&r.caption)).collect();
        state::with_state(world, |s| {
            s.set_list(ROWS, items);
            s.set(HEADING, UiValue::Text(heading));
            state::mark_shown(s, SHOWN, view.revision);
        });
    }
    state::set_text(
        world,
        STATUS,
        format_args!("{}", view.status.map_or("", ImportStatus::text)),
    );
    let slot = |i: usize| built.slots.get(i).copied();
    if let Some(l) = slot(STATUS_SLOT).and_then(|id| world.get_mut_by_id::<TextLabel>(id)) {
        l.color = view.status.map_or(theme::LABEL, ImportStatus::color);
    }
    if let Some(f) = slot(PATH_SLOT).and_then(|id| world.get_mut_by_id::<TextField>(id)) {
        f.focused = view.focus;
    }
}

// Whether the path field holds the keyboard, as the widget last showed it.
pub(crate) fn path_focused(world: &World, built: &BuiltPanel) -> bool {
    path_field(built)
        .and_then(|id| world.get_by_id::<TextField>(id))
        .is_some_and(|f| f.focused)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::UiState;

    fn built(world: &mut World) -> BuiltPanel {
        let mut ui = UiBuilder::new(world, AssetId(9), None);
        build(&mut ui).unwrap()
    }

    fn row(i: usize) -> ImportRow {
        ImportRow {
            entry: i,
            caption: format!("import {i}"),
        }
    }

    #[test]
    fn source_of_reads_whichever_path_arg_the_type_uses() {
        assert_eq!(source_of(&serde_json::json!({"source": "a.glb"})), "a.glb");
        assert_eq!(
            source_of(&serde_json::json!({"model_path": "m.gguf"})),
            "m.gguf"
        );
        assert_eq!(source_of(&serde_json::json!({})), "");
    }

    #[test]
    fn the_rows_are_read_once_per_version_of_the_entries() {
        let mut world = World::new();
        let panel = built(&mut world);
        let view = ImportView {
            revision: 3,
            focus: false,
            status: None,
        };
        sync(&view, || vec![row(0), row(1)], &mut world, &panel);
        let s = world.resource::<UiState>().unwrap();
        assert_eq!(s.list(ROWS).len(), 2);
        assert_eq!(s.get(HEADING), Some(&UiValue::Text("Imports (2)".into())));
        let mut read = false;
        sync(
            &view,
            || {
                read = true;
                Vec::new()
            },
            &mut world,
            &panel,
        );
        assert!(!read, "an unchanged world is not read again");
    }

    #[test]
    fn the_status_line_colors_by_outcome() {
        let mut world = World::new();
        let panel = built(&mut world);
        let status = ImportStatus::Notice("retargeted".into());
        let view = ImportView {
            revision: 1,
            focus: true,
            status: Some(&status),
        };
        sync(&view, Vec::new, &mut world, &panel);
        let label = world
            .get_by_id::<TextLabel>(panel.slots[STATUS_SLOT])
            .unwrap();
        assert_eq!(label.color, NOTICE_LABEL);
        let field = world
            .get_by_id::<TextField>(panel.slots[PATH_SLOT])
            .unwrap();
        assert!(field.focused);
    }
}

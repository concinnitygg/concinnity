//! The Shaders panel: a floating, scrolling list of the rows `list` builds.
//! Each Shader is a heading row (badged "default" on the first), the Materials
//! naming it, and a row per file with that file's last reload status beside
//! it, then "+ Add vertex file" for a Shader without one; the SDF fields follow
//! under their own heading, each file with the volumes reading it, and
//! "+ New Shader" ends the list. Selecting a file opens it in the Shader source
//! panel (`source_panel.rs`), a heading opens its Shader's form and a volume its
//! volume's form. A row's context menu edits, duplicates or deletes its Shader,
//! removes a vertex file, or opens a field, selects its volumes or deletes
//! them.

use concinnity_engine::components::{Binding, List, Menu, UiEvent, UiItem, UiState, UiValue};
use concinnity_engine::ecs::World;

use super::super::asset_list::ROW_H;
use super::super::registry::PanelKey;
use super::diagnostics::Tone;
use super::list::{MenuItem, Row, RowKind};
use crate::editor::theme;
use crate::editor::ui::{BuiltPanel, PanelSpec, UiBuilder, panel_height, state};

// The default (and minimum) width; the user can widen the panel past this.
const SHADERS_W: f32 = 400.0;
// Rows shown before the list scrolls, until the panel is resized taller.
const DEFAULT_ROWS: usize = 12;

const HEADING_COLOR: [f32; 3] = [0.58, 0.66, 0.80];
const ADD_COLOR: [f32; 3] = [0.55, 0.80, 0.60];

// The state the panel shows, and the events its list and menu send.
const ROWS: &str = "editor.shaders.rows";
const SHOWN: [&str; 2] = ["editor.shaders.shown.hi", "editor.shaders.shown.lo"];
const SELECTED: &str = "editor.shaders.selected";
const PICK: &str = "editor.shaders.pick";
const CONTEXT: &str = "editor.shaders.context";
const MENU_OPEN: &str = "editor.shaders.menu_open";
const MENU_ITEMS: &str = "editor.shaders.menu_items";
const MENU_PICK: &str = "editor.shaders.menu_pick";

pub(crate) fn tone_color(tone: Tone) -> [f32; 3] {
    match tone {
        Tone::Info => theme::LABEL_DIM,
        Tone::Warning => theme::LOG_WARN,
        Tone::Error => theme::LOG_ERROR,
    }
}

// What the user asked of the panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShadersAction {
    // Row `i` of the rows.
    Row(usize),
    // Row `i`'s context menu.
    Context(usize),
    // Item `k` of the open menu.
    Menu(usize),
}

// The action a Shaders event asks for, if `event` is one.
pub(crate) fn action(event: &UiEvent) -> Option<ShadersAction> {
    let index = match event.value {
        Some(UiValue::Int(i)) => usize::try_from(i).ok()?,
        _ => return None,
    };
    match event.name.as_str() {
        PICK => Some(ShadersAction::Row(index)),
        CONTEXT => Some(ShadersAction::Context(index)),
        MENU_PICK => Some(ShadersAction::Menu(index)),
        _ => None,
    }
}

// Where the panel sits until the user drags it: right of center, below the
// top bar.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [vw * 0.5 + 40.0, crate::editor::hud::body_top() + 20.0]
}

// The default footprint for `n_rows` rows: all of them, up to the default
// window.
pub(crate) fn size(n_rows: usize) -> [f32; 2] {
    [
        SHADERS_W,
        panel_height(n_rows.clamp(1, DEFAULT_ROWS) as f32 * ROW_H),
    ]
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let list = ui.list(List {
        items: ROWS.to_string(),
        selected: Some(Binding::State(SELECTED.to_string())),
        row_height: ROW_H,
        indent: 16.0,
        on_select: PICK.to_string(),
        on_activate: PICK.to_string(),
        on_context: CONTEXT.to_string(),
        ..Default::default()
    })?;
    ui.menu(Menu {
        open: Some(Binding::State(MENU_OPEN.to_string())),
        items: MENU_ITEMS.to_string(),
        on_pick: MENU_PICK.to_string(),
        at_cursor: true,
        ..Default::default()
    })?;
    let spec = PanelSpec {
        key: PanelKey::Shaders,
        title: "Shaders",
        title_state: None,
        size: size(DEFAULT_ROWS),
        min_size: Some(size(1)),
    };
    ui.panel(&spec, vec![list])
}

// What the panel shows this frame.
pub(crate) struct ShadersView<'a> {
    pub rows: &'a [Row],
    // Counts the times the rows were rebuilt, so the list is refilled only
    // when they change.
    pub rows_rev: u64,
    // The row whose menu is open.
    pub menu: Option<usize>,
}

// A row as a list item: a heading or "+" row in its own color, a file under
// its Shader, and the badge at the right end.
fn item(row: &Row) -> UiItem {
    let color = match &row.kind {
        RowKind::Header(_) | RowKind::Section => Some(HEADING_COLOR),
        RowKind::AddVertex(_) | RowKind::New => Some(ADD_COLOR),
        _ => row
            .badge
            .as_ref()
            .filter(|(_, tone)| *tone != Tone::Info)
            .map(|&(_, tone)| tone_color(tone)),
    };
    UiItem {
        detail: row
            .badge
            .as_ref()
            .map(|(b, _)| b.clone())
            .unwrap_or_default(),
        depth: u16::from(row.indent),
        dimmed: matches!(row.kind, RowKind::Note),
        color,
        ..UiItem::new(&row.text)
    }
}

// Show the rows, refilled only when they were rebuilt, the open file's row,
// and the open menu's items.
pub(crate) fn sync(view: &ShadersView<'_>, world: &mut World) {
    let refill = !state::is_shown(world, SHOWN, view.rows_rev);
    let selected = view
        .rows
        .iter()
        .position(|r| r.selected)
        .and_then(|i| i32::try_from(i).ok())
        .unwrap_or(-1);
    let menu: &[MenuItem] = view
        .menu
        .and_then(|i| view.rows.get(i))
        .map_or(&[], Row::menu);
    state::with_state(world, |s| {
        if refill {
            s.set_list(ROWS, view.rows.iter().map(item).collect());
            state::mark_shown(s, SHOWN, view.rows_rev);
        }
        s.set(SELECTED, UiValue::Int(selected));
        let color = |m: &MenuItem| m.danger().then_some(theme::LOG_ERROR);
        let listed = s.list(MENU_ITEMS);
        let same = listed.len() == menu.len()
            && listed
                .iter()
                .zip(menu)
                .all(|(a, b)| a.text == b.caption() && a.color == color(b));
        if !same {
            let items = menu.iter().map(|m| UiItem {
                color: color(m),
                ..UiItem::new(m.caption())
            });
            s.set_list(MENU_ITEMS, items.collect());
        }
        s.set(MENU_OPEN, UiValue::Bool(!menu.is_empty()));
    });
}

// Whether the menu the hook keeps open is still open: a press off it, or
// Escape, closes it.
pub(crate) fn menu_open(world: &World) -> bool {
    world
        .resource::<UiState>()
        .is_some_and(|s| s.get(MENU_OPEN) == Some(&UiValue::Bool(true)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(kind: RowKind, text: &str) -> Row {
        Row {
            kind,
            text: text.into(),
            badge: None,
            indent: false,
            selected: false,
        }
    }

    #[test]
    fn rows_show_their_kind_and_badge() {
        let header = Row {
            badge: Some(("default".into(), Tone::Info)),
            ..row(RowKind::Header(0), "lit")
        };
        let item = item(&header);
        assert_eq!(item.detail, "default");
        assert_eq!(item.color, Some(HEADING_COLOR));
        let broken = Row {
            badge: Some(("2 errors".into(), Tone::Error)),
            indent: true,
            ..row(RowKind::Note, "x")
        };
        let item = super::item(&broken);
        assert_eq!(item.color, Some(theme::LOG_ERROR));
        assert_eq!(item.depth, 1);
    }

    #[test]
    fn sync_refills_the_rows_only_when_rebuilt_and_lists_the_menu() {
        let mut world = World::new();
        let rows = [
            row(RowKind::Header(0), "lit"),
            row(RowKind::New, "+ New Shader"),
        ];
        let view = ShadersView {
            rows: &rows,
            rows_rev: 1,
            menu: Some(0),
        };
        sync(&view, &mut world);
        let revision = world.resource::<UiState>().unwrap().revision();
        sync(&view, &mut world);
        assert_eq!(world.resource::<UiState>().unwrap().revision(), revision);
        let s = world.resource::<UiState>().unwrap();
        let captions: Vec<&str> = s.list(MENU_ITEMS).iter().map(|i| i.text.as_str()).collect();
        assert_eq!(captions, ["Edit", "Duplicate", "Delete"]);
        assert!(menu_open(&world));
    }
}

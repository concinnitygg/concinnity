//! The Worlds panel's widgets: a `+` that starts an untitled world, a status
//! line, the listing's caption, and the worlds as a list with a context menu.
//! One build serves both presentations; each frame brings its surface, title
//! bar, padding, and text scale in line with the presentation that is up.

use concinnity_engine::components::{
    Binding, Button, LayoutSize, List, Menu, Sprite, TextAlign, TextLabel, UiEvent, UiItem,
    UiState, UiValue,
};
use concinnity_engine::ecs::World;

use super::geometry::{Layout, Mode};
use super::{WorldRow, WorldsAction};
use crate::editor::panels::registry::PanelKey;
use crate::editor::theme;
use crate::editor::ui::build::{emit, fill, fixed};
use crate::editor::ui::{BuiltPanel, PanelSpec, UiBuilder, state};

const NEW_SIDE: f32 = 28.0;
const HEADER_LABEL: [f32; 3] = [0.70, 0.74, 0.84];
const STATUS_COLOR: [f32; 3] = [0.95, 0.55, 0.55];
// Delete carries its warning in the text, not in a filled red button.
const DELETE_LABEL: [f32; 3] = [0.95, 0.60, 0.58];

// The state the panel shows, and the events its controls send.
const ROWS: &str = "editor.worlds.rows";
const SHOWN: [&str; 2] = ["editor.worlds.shown.hi", "editor.worlds.shown.lo"];
const STATUS: &str = "editor.worlds.status";
const HEADING: &str = "editor.worlds.heading";
const NEW: &str = "editor.worlds.new";
pub(crate) const ROW: &str = "editor.worlds.row";
pub(crate) const CONTEXT: &str = "editor.worlds.context";
const MENU_OPEN: &str = "editor.worlds.menu_open";
const MENU_ITEMS: &str = "editor.worlds.menu_items";
const MENU_PICK: &str = "editor.worlds.menu_pick";

// The panel's slots.
const STATUS_SLOT: usize = 0;
const HEADING_SLOT: usize = 1;
const LIST_SLOT: usize = 2;
const NEW_SLOT: usize = 3;
// The body's first child: the spacer that clears the window's chrome.
const SPACER_CHILD: usize = 0;

// What a row's menu offers, by presentation. The switcher opens on the row
// click itself, so there Open would say twice what a click already does.
fn menu_items(mode: Mode) -> &'static [MenuItem] {
    match mode {
        Mode::Start => &[MenuItem::Open, MenuItem::Delete],
        Mode::Session => &[MenuItem::Delete],
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MenuItem {
    Open,
    Delete,
}

impl MenuItem {
    fn caption(self) -> &'static str {
        match self {
            Self::Open => "Open",
            Self::Delete => "Delete",
        }
    }
}

// The action a Worlds event asks for, if `event` is one. A row click opens it
// in the switcher; the start screen previews it, and opens it only once it is
// the world showing behind the panel. A menu pick acts on `menu_row`.
pub(crate) fn action(
    event: &UiEvent,
    mode: Mode,
    previewing: Option<usize>,
    menu_row: Option<usize>,
) -> Option<WorldsAction> {
    if event.name == NEW {
        return Some(WorldsAction::New);
    }
    let i = match event.value {
        Some(UiValue::Int(i)) => usize::try_from(i).ok()?,
        _ => return None,
    };
    Some(match event.name.as_str() {
        ROW => match mode {
            Mode::Start if previewing != Some(i) => WorldsAction::Select(i),
            _ => WorldsAction::Open(i),
        },
        CONTEXT => WorldsAction::OpenMenu(i),
        MENU_PICK => {
            let row = menu_row?;
            match menu_items(mode).get(i)? {
                MenuItem::Open => WorldsAction::Open(row),
                MenuItem::Delete => WorldsAction::Delete(row),
            }
        }
        _ => return None,
    })
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let spacer = ui.sprite(Sprite {
        tint: [0.0; 4],
        ..Default::default()
    })?;
    let gap = ui.sprite(Sprite {
        tint: [0.0; 4],
        ..Default::default()
    })?;
    let new = ui.button_with(
        Button {
            label: "+".to_string(),
            action: Some(emit(NEW, None)),
            ..Default::default()
        },
        [NEW_SIDE, NEW_SIDE],
    )?;
    let new_id = new.item?.id();
    let header = ui.row(vec![fill(gap), new], NEW_SIDE + 4.0)?;
    let status = ui.bound_label(STATUS, STATUS_COLOR, TextAlign::Left)?;
    let status_id = status.item?.id();
    let status = ui.row(vec![status], 26.0)?;
    let heading = ui.bound_label(HEADING, HEADER_LABEL, TextAlign::Left)?;
    let heading_id = heading.item?.id();
    let heading = ui.row(vec![heading], 24.0)?;
    let list = ui.list(List {
        items: ROWS.to_string(),
        row_height: 26.0,
        on_select: ROW.to_string(),
        on_context: CONTEXT.to_string(),
        ..Default::default()
    })?;
    let list_id = list.item?.id();
    ui.menu(Menu {
        open: Some(Binding::State(MENU_OPEN.to_string())),
        items: MENU_ITEMS.to_string(),
        on_pick: MENU_PICK.to_string(),
        at_cursor: true,
        ..Default::default()
    })?;
    let layout = Layout::new(Mode::Session, [0.0; 2], 0.0);
    let spec = PanelSpec {
        key: PanelKey::Worlds,
        title: "Worlds",
        title_state: None,
        size: layout.size(),
        min_size: None,
    };
    let body = vec![fixed(spacer, [0.0, 0.0]), header, status, heading, list];
    let mut panel = ui.panel(&spec, body)?;
    panel.slots = vec![status_id, heading_id, list_id, new_id];
    Some(panel)
}

// The listing as list items: the start screen marks its selection, the
// switcher the world the session has open.
pub(crate) fn rows(rows: &[WorldRow], mode: Mode, selected: Option<usize>) -> Vec<UiItem> {
    rows.iter()
        .enumerate()
        .map(|(i, row)| UiItem {
            selected: match mode {
                Mode::Start => selected == Some(i),
                Mode::Session => row.open,
            },
            ..UiItem::new(&row.name)
        })
        .collect()
}

// What the panel shows this frame.
pub(crate) struct WorldsView<'a> {
    // Bumped whenever the items change.
    pub rows_rev: u64,
    pub rows: &'a [UiItem],
    pub layout: Layout,
    // Why the last preview or open failed, if it did.
    pub status: Option<&'a str>,
    // The row whose menu is open.
    pub menu: Option<usize>,
}

pub(crate) fn sync(view: &WorldsView<'_>, world: &mut World, built: &BuiltPanel) {
    let mode = view.layout.mode;
    if !state::is_shown(world, SHOWN, view.rows_rev) {
        let items = view.rows.to_vec();
        let heading = match (items.is_empty(), mode) {
            (true, Mode::Start) => "No worlds yet".to_string(),
            (true, Mode::Session) => "No worlds yet - press + to start one".to_string(),
            (false, _) => format!("Worlds ({})", items.len()),
        };
        state::with_state(world, |s| {
            s.set_list(ROWS, items);
            s.set(HEADING, UiValue::Text(heading));
            state::mark_shown(s, SHOWN, view.rows_rev);
        });
    }
    state::set_text(world, STATUS, format_args!("{}", view.status.unwrap_or("")));
    let menu = view.menu.map_or(&[][..], |_| menu_items(mode));
    state::with_state(world, |s| {
        let listed = s.list(MENU_ITEMS);
        let same = listed.len() == menu.len()
            && listed.iter().zip(menu).all(|(a, b)| a.text == b.caption());
        if !same {
            let items = menu.iter().map(|m| UiItem {
                color: (*m == MenuItem::Delete).then_some(DELETE_LABEL),
                ..UiItem::new(m.caption())
            });
            s.set_list(MENU_ITEMS, items.collect());
        }
        s.set(MENU_OPEN, UiValue::Bool(!menu.is_empty()));
    });
    present(&view.layout, world, built);
}

// Bring the panel's chrome in line with the presentation `layout` is in.
fn present(layout: &Layout, world: &mut World, built: &BuiltPanel) {
    let m = *layout.metrics();
    let slot = |i: usize| built.slots.get(i).copied();
    if let Some(root) = world.get_mut_by_id::<concinnity_engine::components::Layout>(built.root) {
        state::set_if_changed(&mut root.movable, !m.docked);
    }
    if let Some(bar) = world.get_mut_by_id::<concinnity_engine::components::Layout>(built.bar) {
        state::set_if_changed(&mut bar.visible, m.has_title_bar());
    }
    if let Some(surface) = world.get_mut_by_id::<Sprite>(built.surface) {
        state::set_if_changed(&mut surface.tint, m.panel_tint);
        let radius = if m.docked { 0.0 } else { theme::PANEL_RADIUS };
        state::set_if_changed(&mut surface.corner_radius, radius);
        let border = if m.docked {
            0.0
        } else {
            theme::PANEL_BORDER_WIDTH
        };
        state::set_if_changed(&mut surface.border_width, border);
    }
    if let Some(body) = world.get_mut_by_id::<concinnity_engine::components::Layout>(built.body) {
        state::set_if_changed(&mut body.padding, m.pad);
        if let Some(spacer) = body.children.get_mut(SPACER_CHILD) {
            state::set_if_changed(
                &mut spacer.height,
                LayoutSize::Fixed(layout.content_offset()),
            );
        }
    }
    if let Some(list) = slot(LIST_SLOT).and_then(|id| world.get_mut_by_id::<List>(id)) {
        state::set_if_changed(&mut list.row_height, m.row_h);
        state::set_if_changed(&mut list.style.scale, m.text_scale);
    }
    for label in [STATUS_SLOT, HEADING_SLOT] {
        if let Some(l) = slot(label).and_then(|id| world.get_mut_by_id::<TextLabel>(id)) {
            state::set_if_changed(&mut l.scale, m.text_scale);
            state::set_if_changed(&mut l.wrap_width, m.width - 2.0 * m.pad);
            state::set_if_changed(&mut l.max_lines, if label == STATUS_SLOT { 2 } else { 1 });
        }
    }
    if let Some(b) = slot(NEW_SLOT).and_then(|id| world.get_mut_by_id::<Button>(id)) {
        state::set_if_changed(&mut b.style.scale, m.text_scale);
    }
}

// Whether the row menu the hook keeps open is still open: a press off it, or
// Escape, closes it.
pub(crate) fn menu_open(world: &World) -> bool {
    world
        .resource::<UiState>()
        .is_some_and(|s| s.get(MENU_OPEN) == Some(&UiValue::Bool(true)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::Layout as UiLayout;
    use concinnity_engine::ecs::asset_id::AssetId;

    fn built(world: &mut World) -> BuiltPanel {
        let mut ui = UiBuilder::new(world, AssetId(9), None);
        build(&mut ui).unwrap()
    }

    fn event(name: &str, i: i32) -> UiEvent {
        UiEvent {
            name: name.into(),
            element: None,
            value: Some(UiValue::Int(i)),
        }
    }

    fn worlds(n: usize) -> Vec<WorldRow> {
        (0..n)
            .map(|i| WorldRow {
                name: format!("world{i}"),
                path: format!("/p/world{i}.jsonl"),
                open: i == 0,
            })
            .collect()
    }

    // The switcher opens on the click; the start screen previews on it and
    // opens only once that row is the one showing.
    #[test]
    fn a_row_click_opens_or_previews_by_presentation() {
        let row = event(ROW, 2);
        assert_eq!(
            action(&row, Mode::Session, None, None),
            Some(WorldsAction::Open(2))
        );
        assert_eq!(
            action(&row, Mode::Start, None, None),
            Some(WorldsAction::Select(2))
        );
        assert_eq!(
            action(&row, Mode::Start, Some(2), None),
            Some(WorldsAction::Open(2))
        );
        assert_eq!(
            action(&event(CONTEXT, 1), Mode::Session, None, None),
            Some(WorldsAction::OpenMenu(1))
        );
    }

    // The switcher's menu offers Delete alone; the start screen's offers Open
    // first.
    #[test]
    fn the_menu_acts_on_its_row() {
        let pick = |mode, k| action(&event(MENU_PICK, k), mode, None, Some(3));
        assert_eq!(pick(Mode::Session, 0), Some(WorldsAction::Delete(3)));
        assert_eq!(pick(Mode::Session, 1), None);
        assert_eq!(pick(Mode::Start, 0), Some(WorldsAction::Open(3)));
        assert_eq!(pick(Mode::Start, 1), Some(WorldsAction::Delete(3)));
        assert_eq!(action(&event(MENU_PICK, 0), Mode::Start, None, None), None);
    }

    #[test]
    fn rows_mark_the_selection_or_the_open_world() {
        let start = rows(&worlds(3), Mode::Start, Some(2));
        assert!(start[2].selected && !start[0].selected);
        let session = rows(&worlds(3), Mode::Session, Some(2));
        assert!(session[0].selected && !session[2].selected);
    }

    // The start screen's sidebar has no title bar, stays put, washes over the
    // preview, and drops its content below the window's chrome; the switcher
    // has all of its chrome back.
    #[test]
    fn each_presentation_shapes_the_chrome() {
        let mut world = World::new();
        let panel = built(&mut world);
        let items = rows(&worlds(2), Mode::Start, None);
        let start = Layout::new(Mode::Start, [1280.0, 720.0], 28.0);
        let view = WorldsView {
            rows_rev: 1,
            rows: &items,
            layout: start,
            status: None,
            menu: None,
        };
        sync(&view, &mut world, &panel);
        let bar = world.get_by_id::<UiLayout>(panel.bar).unwrap();
        assert!(!bar.visible);
        assert!(!world.get_by_id::<UiLayout>(panel.root).unwrap().movable);
        let body = world.get_by_id::<UiLayout>(panel.body).unwrap();
        assert_eq!(body.children[SPACER_CHILD].height, LayoutSize::Fixed(28.0));
        let list = world.get_by_id::<List>(panel.slots[LIST_SLOT]).unwrap();
        assert_eq!(list.row_height, 34.0);
        let s = world.resource::<UiState>().unwrap();
        assert_eq!(s.get(HEADING), Some(&UiValue::Text("Worlds (2)".into())));

        let session = WorldsView {
            layout: Layout::new(Mode::Session, [1280.0, 720.0], 28.0),
            menu: Some(0),
            ..view
        };
        sync(&session, &mut world, &panel);
        assert!(world.get_by_id::<UiLayout>(panel.bar).unwrap().visible);
        assert!(world.get_by_id::<UiLayout>(panel.root).unwrap().movable);
        let body = world.get_by_id::<UiLayout>(panel.body).unwrap();
        assert_eq!(body.children[SPACER_CHILD].height, LayoutSize::Fixed(0.0));
        let s = world.resource::<UiState>().unwrap();
        assert_eq!(s.list(MENU_ITEMS)[0].text, "Delete");
        assert!(menu_open(&world));
    }
}

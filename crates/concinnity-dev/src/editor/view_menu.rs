//! The Display menu: a compact dropdown under the top bar's Display chip
//! selecting the viewport view mode (radio entries) and the per-session show
//! flags (check entries), plus the editor-side billboard toggle and the extent
//! outline categories. An engine `Menu` that stays open as entries are picked;
//! the hook (`hook/drive/view_menu.rs`) owns the selections and applies each
//! pick.

use std::sync::LazyLock;

use concinnity_engine::components::{Binding, EntryMark, Menu, MenuEntry, UiEvent, UiValue};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;
pub(crate) use concinnity_engine::gfx::view_modes::{ShowFlags, ViewMode};
use concinnity_engine::protocol::ViewOverrides;

use super::outlines::{Category, CategorySet};
use super::ui::build::emit;
use super::ui::{UiBuilder, state};

const MENU_W: f32 = 190.0;
const ROW_H: f32 = 22.0;
const MARGIN: f32 = 8.0;

// The state the menu shows, and the event its picks send.
pub(crate) const OPEN: &str = "editor.display.open";
const PICK: &str = "editor.display.pick";

// One menu row: a view-mode radio, a section heading, one show-flag toggle,
// the billboard-icons toggle (editor overlay sprites, not a render pass), or
// one always-on extent-outline category.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum MenuRow {
    Mode(ViewMode),
    Heading(&'static str),
    Flag(ShowFlags, &'static str),
    Billboards,
    Extent(Category, &'static str),
}

pub(crate) fn rows() -> Vec<MenuRow> {
    std::iter::once(MenuRow::Heading("View mode"))
        .chain(ViewMode::ALL.into_iter().map(MenuRow::Mode))
        .chain(std::iter::once(MenuRow::Heading("Show")))
        .chain(
            ShowFlags::LABELED
                .into_iter()
                .map(|(f, label)| MenuRow::Flag(f, label)),
        )
        .chain(std::iter::once(MenuRow::Billboards))
        .chain(std::iter::once(MenuRow::Heading("Extents")))
        .chain(
            Category::LABELED
                .into_iter()
                .map(|(c, label)| MenuRow::Extent(c, label)),
        )
        .collect()
}

// The rows, and the state naming whether each is on, read once.
static ROWS: LazyLock<Vec<(MenuRow, String)>> = LazyLock::new(|| {
    rows()
        .into_iter()
        .enumerate()
        .map(|(i, row)| (row, format!("editor.display.on.{i}")))
        .collect()
});

// The row a pick event names, if `event` is one.
pub(crate) fn pick(event: &UiEvent) -> Option<MenuRow> {
    match (event.name.as_str(), &event.value) {
        (PICK, Some(UiValue::Int(i))) => usize::try_from(*i)
            .ok()
            .and_then(|i| ROWS.get(i))
            .map(|(row, _)| *row),
        _ => None,
    }
}

// The menu's left edge: right-aligned under the top bar, where the Display
// chip lives.
fn left(vw: f32) -> f32 {
    (vw - MENU_W - MARGIN).max(0.0)
}

fn entry(i: usize, row: MenuRow, on: &str) -> MenuEntry {
    let value = Some(UiValue::Int(i32::try_from(i).unwrap_or(i32::MAX)));
    let label = match row {
        MenuRow::Mode(m) => m.label(),
        MenuRow::Heading(label) | MenuRow::Flag(_, label) | MenuRow::Extent(_, label) => label,
        MenuRow::Billboards => "Billboards",
    };
    match row {
        MenuRow::Heading(_) => MenuEntry {
            label: label.to_string(),
            disabled: true,
            separator: i > 0,
            ..Default::default()
        },
        _ => MenuEntry {
            label: label.to_string(),
            action: Some(emit(PICK, value)),
            mark: Some(match row {
                MenuRow::Mode(_) => EntryMark::Radio(Binding::State(on.to_string())),
                _ => EntryMark::Check(Binding::State(on.to_string())),
            }),
            stay_open: true,
            ..Default::default()
        },
    }
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<AssetId> {
    let entries = ROWS
        .iter()
        .enumerate()
        .map(|(i, (row, on))| entry(i, *row, on))
        .collect();
    ui.menu(Menu {
        open: Some(Binding::State(OPEN.to_string())),
        entries,
        y: super::hud::BAR_H + 4.0,
        width: MENU_W,
        row_height: ROW_H,
        ..Default::default()
    })
}

// The menu's selection state this frame.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MenuState {
    pub open: bool,
    pub mode: ViewMode,
    pub show: ShowFlags,
    pub billboards: bool,
    pub extents: CategorySet,
}

impl MenuState {
    fn on(&self, row: MenuRow) -> bool {
        match row {
            MenuRow::Mode(m) => self.mode == m,
            MenuRow::Heading(_) => false,
            MenuRow::Flag(f, _) => self.show.contains(f),
            MenuRow::Billboards => self.billboards,
            MenuRow::Extent(c, _) => self.extents.contains(c),
        }
    }
}

// Bring the menu in line with `state`, under a window `vw` wide.
pub(crate) fn sync(world: &mut World, id: AssetId, state: &MenuState, vw: f32) {
    if let Some(menu) = world.get_mut_by_id::<Menu>(id) {
        state::set_if_changed(&mut menu.x, left(vw));
    }
    state::set_bool(world, OPEN, state.open);
    for (row, on) in ROWS.iter() {
        state::set_bool(world, on, state.on(*row));
    }
}

// Whether the menu is open in `world`.
pub(crate) fn is_open(world: &World) -> bool {
    state::get_bool(world, OPEN)
}

// The view state to publish this tick. One written from outside the editor
// since its last publish (the `view-set` and `show-set` debug verbs, or
// `--hide` at launch) wins, so the menu follows.
pub(crate) fn resolve_view(
    published: ViewOverrides,
    live: Option<ViewOverrides>,
    own: ViewOverrides,
) -> ViewOverrides {
    match live {
        Some(live) if live != published => live,
        _ => own,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::UiState;

    #[test]
    fn an_outside_view_change_wins_until_the_editor_changes_again() {
        let lit = ViewOverrides::default();
        let motion = ViewOverrides {
            mode: ViewMode::Motion,
            ..lit
        };
        let normals = ViewOverrides {
            mode: ViewMode::Normals,
            ..lit
        };
        // Nothing published yet, or the live state is the editor's own.
        assert_eq!(resolve_view(lit, None, normals), normals);
        assert_eq!(resolve_view(lit, Some(lit), normals), normals);
        // Written from outside since the last publish: adopted.
        assert_eq!(resolve_view(lit, Some(motion), lit), motion);
        // Once adopted and republished, the menu's next pick wins again.
        assert_eq!(resolve_view(motion, Some(motion), normals), normals);
    }

    fn state() -> MenuState {
        MenuState {
            open: true,
            mode: ViewMode::default(),
            show: ShowFlags::all(),
            billboards: true,
            extents: CategorySet::default(),
        }
    }

    // Every row but a heading picks; modes are radios and the rest checks.
    #[test]
    fn every_row_becomes_an_entry_that_stays_open() {
        let mut world = World::new();
        let id = build(&mut UiBuilder::new(&mut world, AssetId(9), None)).unwrap();
        let menu = world.get_by_id::<Menu>(id).unwrap();
        assert_eq!(menu.entries.len(), rows().len());
        for (e, row) in menu.entries.iter().zip(rows()) {
            match row {
                MenuRow::Heading(_) => assert!(e.disabled && e.mark.is_none()),
                MenuRow::Mode(_) => {
                    assert!(matches!(e.mark, Some(EntryMark::Radio(_))) && e.stay_open);
                }
                _ => assert!(matches!(e.mark, Some(EntryMark::Check(_))) && e.stay_open),
            }
        }
    }

    // The marks follow the selections, the menu sits at the window's right
    // edge, and its picks name their rows.
    #[test]
    fn sync_writes_the_marks_and_a_pick_names_its_row() {
        let mut world = World::new();
        let id = build(&mut UiBuilder::new(&mut world, AssetId(9), None)).unwrap();
        let mut s = state();
        s.show = ShowFlags::all().toggled(ShowFlags::FOG);
        sync(&mut world, id, &s, 1280.0);
        assert_eq!(
            world.get_by_id::<Menu>(id).unwrap().x,
            1280.0 - MENU_W - MARGIN
        );
        let ui = world.resource::<UiState>().unwrap();
        let fog = ROWS
            .iter()
            .position(|(r, _)| matches!(r, MenuRow::Flag(f, _) if *f == ShowFlags::FOG))
            .unwrap();
        assert_eq!(ui.get(&ROWS[fog].1), Some(&UiValue::Bool(false)));
        let lit = ROWS
            .iter()
            .position(|(r, _)| *r == MenuRow::Mode(ViewMode::default()))
            .unwrap();
        assert_eq!(ui.get(&ROWS[lit].1), Some(&UiValue::Bool(true)));
        assert!(is_open(&world));
        let event = UiEvent {
            name: PICK.into(),
            element: None,
            value: Some(UiValue::Int(i32::try_from(fog).unwrap())),
        };
        assert_eq!(pick(&event), Some(MenuRow::Flag(ShowFlags::FOG, "Fog")));
    }
}

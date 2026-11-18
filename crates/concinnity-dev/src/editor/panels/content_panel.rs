//! The Content panel: a thumbnail grid over the world's visual assets
//! (textures, materials, meshes, models, environment maps, prefabs). Cells show
//! the baked thumbnail when one exists (`editor/thumbs.rs`) and a typed icon
//! glyph otherwise; a search field ranks matches (`editor/filter.rs`) and a type
//! button cycles the kind filter. A cell dragged out of the grid places its
//! asset in the world. The item assembly and actions live in
//! `hook/edit/content.rs`.

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_engine::components::{
    Binding, Button, Grid, TextAlign, TextField, UiEvent, UiImage, UiItem, UiValue,
};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use super::registry::PanelKey;
use crate::editor::theme;
use crate::editor::thumbs::Thumb;
use crate::editor::ui::build::emit;
use crate::editor::ui::{BuiltPanel, PanelSpec, UiBuilder, panel_height, state};
use crate::editor::viewport::billboards;

const COLS: usize = 4;
const GRID_ROWS: usize = 4;
const CELL_W: f32 = 92.0;
const CELL_H: f32 = 112.0;
const GAP: f32 = 6.0;
// The search field + type button row.
const CTRL_H: f32 = 30.0;
const TYPE_W: f32 = 120.0;
const STATUS_H: f32 = 22.0;

// The state the panel shows, and the events its controls send.
const CELLS: &str = "editor.content.cells";
const SHOWN: [&str; 2] = ["editor.content.shown.hi", "editor.content.shown.lo"];
const TYPE: &str = "editor.content.type";
const STATUS: &str = "editor.content.status";
const CYCLE: &str = "editor.content.cycle";
const SELECT: &str = "editor.content.select";
const EXTEND: &str = "editor.content.extend";
const DRAG: &str = "editor.content.drag";

// The panel's slots.
const SEARCH_SLOT: usize = 0;

fn grid_size(cols: usize, rows: usize) -> [f32; 2] {
    [
        cols as f32 * CELL_W + (cols as f32 - 1.0) * GAP,
        rows as f32 * (CELL_H + GAP),
    ]
}

fn size_for(cols: usize, rows: usize) -> [f32; 2] {
    let grid = grid_size(cols, rows);
    [
        grid[0] + 2.0 * crate::editor::ui::build::BODY_PAD,
        panel_height(CTRL_H + grid[1] + STATUS_H),
    ]
}

// The size the panel opens at.
pub(crate) fn size() -> [f32; 2] {
    size_for(COLS, GRID_ROWS)
}

pub(crate) fn default_origin(vp: [f32; 2]) -> [f32; 2] {
    // Right side, clear of the Assets panel's left anchor.
    [
        (vp[0] - size()[0] - 16.0).max(0.0),
        crate::editor::hud::body_top(),
    ]
}

// What the user asked of the panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContentAction {
    CycleType,
    // Cell `i` of the shown items: select it.
    Select(usize),
    // Cell `i`, with a modifier held: add it to the selection or take it out.
    Extend(usize),
    // Cell `i`, dragged out of its place.
    Drag(usize),
}

// The action a Content event asks for, if `event` is one.
pub(crate) fn action(event: &UiEvent) -> Option<ContentAction> {
    if event.name == CYCLE {
        return Some(ContentAction::CycleType);
    }
    let i = match event.value {
        Some(UiValue::Int(i)) => usize::try_from(i).ok()?,
        _ => return None,
    };
    Some(match event.name.as_str() {
        SELECT => ContentAction::Select(i),
        EXTEND => ContentAction::Extend(i),
        DRAG => ContentAction::Drag(i),
        _ => return None,
    })
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let search = ui.text_field(TextField {
        placeholder: "search".to_string(),
        height: CTRL_H - 6.0,
        ..Default::default()
    })?;
    let search_id = search.item?.id();
    let kind = ui.button_with(
        Button {
            label: "All".to_string(),
            text_binding: Some(Binding::State(TYPE.to_string())),
            action: Some(emit(CYCLE, None)),
            ..Default::default()
        },
        [TYPE_W, CTRL_H - 6.0],
    )?;
    let header = ui.row(vec![search, kind], CTRL_H)?;
    let grid = ui.grid(Grid {
        items: CELLS.to_string(),
        cell_width: CELL_W,
        cell_height: CELL_H,
        gap: GAP,
        on_select: SELECT.to_string(),
        on_extend: EXTEND.to_string(),
        on_drag: DRAG.to_string(),
        ..Default::default()
    })?;
    let status = ui.bound_label(STATUS, theme::LABEL_DIM, TextAlign::Left)?;
    let status = ui.row(vec![status], STATUS_H)?;
    let spec = PanelSpec {
        key: PanelKey::Content,
        title: "Content",
        title_state: None,
        size: size(),
        min_size: Some(size_for(2, 1)),
    };
    let mut panel = ui.panel(&spec, vec![header, grid, status])?;
    panel.slots = vec![search_id];
    Some(panel)
}

// The search field's id.
pub(crate) fn search_field(built: &BuiltPanel) -> Option<AssetId> {
    built.slots.get(SEARCH_SLOT).copied()
}

// One visual asset as a grid cell: its thumbnail, or its type's glyph in the
// type's hue.
pub(crate) fn cell(name: &str, ty: RegisteredType, thumb: Option<Thumb>, selected: bool) -> UiItem {
    if let Some(t) = thumb {
        return UiItem {
            image: Some(UiImage {
                texture: t.handle,
                size: [t.width, t.height],
            }),
            selected,
            ..UiItem::new(name)
        };
    }
    let tint = billboards::tint(ty.as_str());
    UiItem {
        detail: billboards::glyph(ty.as_str()),
        detail_color: Some([tint[0], tint[1], tint[2]]),
        selected,
        ..UiItem::new(name)
    }
}

// What the panel shows this frame.
pub(crate) struct ContentView<'a> {
    // Bumped whenever the cells change.
    pub cells_rev: u64,
    pub cells: &'a [UiItem],
    pub type_caption: &'a str,
    pub search_focus: bool,
}

pub(crate) fn sync(view: &ContentView<'_>, world: &mut World, built: &BuiltPanel) {
    if !state::is_shown(world, SHOWN, view.cells_rev) {
        let cells = view.cells.to_vec();
        let count = cells.len();
        state::with_state(world, |s| {
            s.set_list(CELLS, cells);
            s.set(STATUS, UiValue::Text(format!("{count} asset(s)")));
            state::mark_shown(s, SHOWN, view.cells_rev);
        });
    }
    state::set_text(world, TYPE, format_args!("{}", view.type_caption));
    if let Some(f) = search_field(built).and_then(|id| world.get_mut_by_id::<TextField>(id)) {
        f.focused = view.search_focus;
    }
}

// Whether the search field holds the keyboard, as the widget last showed it.
pub(crate) fn search_focused(world: &World, built: &BuiltPanel) -> bool {
    search_field(built)
        .and_then(|id| world.get_by_id::<TextField>(id))
        .is_some_and(|f| f.focused)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::UiState;
    use concinnity_engine::ecs::TextureHandle;

    fn built(world: &mut World) -> BuiltPanel {
        let mut ui = UiBuilder::new(world, AssetId(9), None);
        build(&mut ui).unwrap()
    }

    fn event(name: &str, value: Option<UiValue>) -> UiEvent {
        UiEvent {
            name: name.into(),
            element: None,
            value,
        }
    }

    #[test]
    fn grid_events_name_their_cells() {
        let at = |i| Some(UiValue::Int(i));
        assert_eq!(action(&event(CYCLE, None)), Some(ContentAction::CycleType));
        assert_eq!(
            action(&event(SELECT, at(3))),
            Some(ContentAction::Select(3))
        );
        assert_eq!(
            action(&event(EXTEND, at(1))),
            Some(ContentAction::Extend(1))
        );
        assert_eq!(action(&event(DRAG, at(0))), Some(ContentAction::Drag(0)));
        assert_eq!(action(&event(DRAG, None)), None);
    }

    #[test]
    fn a_cell_shows_its_thumbnail_or_its_types_glyph() {
        let thumb = Thumb {
            handle: TextureHandle::new(4),
            width: 64,
            height: 32,
        };
        let shot = cell("ball", RegisteredType::Mesh, Some(thumb), true);
        assert_eq!(shot.image.map(|i| i.size), Some([64, 32]));
        assert!(shot.selected);
        let chip = cell("ball", RegisteredType::Mesh, None, false);
        assert!(chip.image.is_none());
        assert_eq!(chip.detail, billboards::glyph("Mesh"));
        assert!(chip.detail_color.is_some());
    }

    #[test]
    fn the_cells_are_refilled_only_when_they_change() {
        let mut world = World::new();
        let panel = built(&mut world);
        let cells = [UiItem::new("a"), UiItem::new("b")];
        let view = ContentView {
            cells_rev: 1,
            cells: &cells,
            type_caption: "Mesh",
            search_focus: false,
        };
        sync(&view, &mut world, &panel);
        let revision = world.resource::<UiState>().unwrap().revision();
        sync(&view, &mut world, &panel);
        let s = world.resource::<UiState>().unwrap();
        assert_eq!(s.revision(), revision);
        assert_eq!(s.list(CELLS).len(), 2);
        assert_eq!(s.get(STATUS), Some(&UiValue::Text("2 asset(s)".into())));
        assert_eq!(s.get(TYPE), Some(&UiValue::Text("Mesh".into())));
    }
}

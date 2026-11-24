//! The command palette's layout half: a centered floating panel with one text
//! field and a ranked list of results under it, each row a caption with its
//! hint and category tag at the right. The data model and ranking live in
//! `editor/palette/`; the drive in `hook/edit/palette.rs`, which owns the
//! query and the highlight.

use std::fmt::Write as _;

use concinnity_engine::components::{Binding, Layout, List, TextField, UiEvent, UiItem, UiValue};
use concinnity_engine::ecs::World;

use super::PaletteItem;
use crate::editor::panels::registry::PanelKey;
use crate::editor::ui::build::fill;
use crate::editor::ui::{BuiltPanel, PanelSpec, UiBuilder, panel_height, state};

pub(crate) const PALETTE_W: f32 = 560.0;
const INPUT_H: f32 = 28.0;
const ROW_H: f32 = 24.0;
// Rows shown before the list scrolls.
const ROWS_SHOWN: usize = 10;

// The state the panel shows, and the event a click on a row sends.
const ROWS: &str = "editor.palette.rows";
const PICKED: &str = "editor.palette.picked";
const PICK: &str = "editor.palette.pick";
// What a press outside the palette sends.
const DISMISS: &str = "editor.palette.dismiss";
// The matches' and the highlight's revisions last shown, so a steady frame
// rebuilds no list and scrolls nowhere.
const SHOWN: [&str; 2] = ["editor.palette.shown.hi", "editor.palette.shown.lo"];
const REVEALED: [&str; 2] = ["editor.palette.revealed.hi", "editor.palette.revealed.lo"];

// The panel's slots.
const FIELD_SLOT: usize = 0;
const LIST_SLOT: usize = 1;

// Whether `event` is a press outside the palette.
pub(crate) fn dismissed(event: &UiEvent) -> bool {
    event.name == DISMISS
}

// The match a click on a row picks, if `event` is one.
pub(crate) fn pick(event: &UiEvent) -> Option<usize> {
    match (event.name.as_str(), &event.value) {
        (PICK, Some(UiValue::Int(i))) => usize::try_from(*i).ok(),
        _ => None,
    }
}

// Centered near the top of the viewport, launcher-style.
pub(crate) fn default_origin(vp: [f32; 2]) -> [f32; 2] {
    let s = size();
    [
        ((vp[0] - s[0]) * 0.5).max(0.0),
        (vp[1] * 0.18).max(crate::editor::hud::body_top()),
    ]
}

pub(crate) fn size() -> [f32; 2] {
    [
        PALETTE_W,
        panel_height(INPUT_H + 8.0 + ROWS_SHOWN as f32 * ROW_H),
    ]
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let field = ui.text_field(TextField {
        height: INPUT_H,
        placeholder: "search".to_string(),
        ..Default::default()
    })?;
    let field_id = field.item?.id();
    let list = ui.list(List {
        items: ROWS.to_string(),
        selected: Some(Binding::State(PICKED.to_string())),
        on_select: PICK.to_string(),
        row_height: ROW_H,
        ..Default::default()
    })?;
    let list_id = list.item?.id();
    let spec = PanelSpec {
        key: PanelKey::Palette,
        title: "Palette",
        title_state: None,
        size: size(),
        min_size: None,
    };
    let mut panel = ui.panel(&spec, vec![field, fill(list_id)])?;
    panel.slots = vec![field_id, list_id];
    // The palette is a launcher, so a press away from it puts it away.
    if let Some(root) = ui.get_mut::<Layout>(panel.root) {
        root.on_dismiss = DISMISS.to_string();
    }
    Some(panel)
}

// What the panel shows this frame.
pub(crate) struct PaletteView<'a> {
    pub items: &'a [PaletteItem],
    // The items the query keeps, best first.
    pub matches: &'a [usize],
    pub query: &'a str,
    // The highlighted match.
    pub pick: usize,
    // Whether the field holds the keyboard this frame.
    pub focus: bool,
    // Bumped when the matches change, and when the highlight moves.
    pub revision: u64,
    pub moved: u64,
}

// The matches as list rows; with none, one dim row saying so instead of a
// silently blank list.
fn rows(view: &PaletteView<'_>) -> Vec<UiItem> {
    if view.matches.is_empty() {
        return vec![UiItem {
            dimmed: true,
            ..UiItem::new("no match")
        }];
    }
    view.matches
        .iter()
        .filter_map(|&at| view.items.get(at))
        .map(|item| {
            let mut detail = item.hint.clone();
            #[expect(
                clippy::let_underscore_must_use,
                reason = "writing to a String cannot fail"
            )]
            let _ = write!(detail, "   {}", item.category.tag());
            UiItem {
                detail,
                ..UiItem::new(&item.label)
            }
        })
        .collect()
}

pub(crate) fn sync(view: &PaletteView<'_>, world: &mut World, built: &BuiltPanel) {
    if let Some(f) = built
        .slots
        .get(FIELD_SLOT)
        .and_then(|&id| world.get_mut_by_id::<TextField>(id))
    {
        if f.content != view.query {
            f.content.clear();
            f.content.push_str(view.query);
            f.caret = view.query.chars().count();
        }
        state::set_if_changed(&mut f.focused, view.focus);
    }
    if !state::is_shown(world, SHOWN, view.revision) {
        let rows = rows(view);
        state::with_state(world, |s| {
            s.set_list(ROWS, rows);
            state::mark_shown(s, SHOWN, view.revision);
        });
    }
    let pick = i32::try_from(view.pick).unwrap_or(i32::MAX);
    state::with_state(world, |s| s.set(PICKED, UiValue::Int(pick)));
    if !state::is_shown(world, REVEALED, view.moved) {
        let count = view.matches.len();
        if let Some(list) = built
            .slots
            .get(LIST_SLOT)
            .and_then(|&id| world.get_mut_by_id::<List>(id))
        {
            list.reveal(view.pick, count);
        }
        state::with_state(world, |s| state::mark_shown(s, REVEALED, view.moved));
    }
}

// What is typed into the panel's field.
pub(crate) fn typed<'w>(world: &'w World, built: &BuiltPanel) -> Option<&'w str> {
    built
        .slots
        .get(FIELD_SLOT)
        .and_then(|&id| world.get_by_id::<TextField>(id))
        .map(|f| f.content.as_str())
}

#[cfg(test)]
mod tests;

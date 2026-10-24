//! The Planet panel's layout half: a floating panel over the world's `Planet`
//! (see `planet.rs` for its rows) with a status line and Apply button in its
//! header row and a scrolling window of rows below. A row is a caption plus a
//! text input, a reference picker cycled by clicking, or an add row, with up to
//! two buttons at its right end. Every number row keeps its own text input, so
//! a typed edit survives scrolling it out of view until Apply.

use concinnity_engine::components::TextAlign;
use concinnity_engine::ecs::World;

use super::form::FieldKind;
use super::planet::{self, Button, Row, RowKind};
use super::registry::PanelKey;
use crate::editor::hud_ids::{hud_ids, panel_base};
use crate::editor::theme;
use crate::editor::widget::{self, place_rounded, point_in};

/// Rows the window shows at once.
pub(crate) const VISIBLE_ROWS: usize = 22;

// Buttons a row carries at most.
const ROW_BUTTONS: usize = 2;

hud_ids! {
    base: panel_base(PanelKey::Planet);
    sprites: [
        pub(crate) PANEL_BG,
        pub(crate) CLOSE_BG,
        pub(crate) APPLY_BG,
        pub(crate) row_bg[VISIBLE_ROWS],
        pub(crate) picker_bg[VISIBLE_ROWS],
        pub(crate) button_bg[VISIBLE_ROWS * ROW_BUTTONS],
    ];
    labels: [
        pub(crate) TITLE_LABEL,
        pub(crate) CLOSE_LABEL,
        pub(crate) APPLY_LABEL,
        pub(crate) STATUS_LABEL,
        pub(crate) row_label[VISIBLE_ROWS],
        pub(crate) picker_label[VISIBLE_ROWS],
        pub(crate) button_label[VISIBLE_ROWS * ROW_BUTTONS],
    ];
    fields: [pub(crate) input[planet::MAX_ROWS] = ""];
}

// Geometry, in window pixels, every rect from the panel origin `o`.
pub(crate) const PLANET_W: f32 = 400.0;
const PAD: f32 = 10.0;
const HEADER_H: f32 = 32.0;
const APPLY_W: f32 = 70.0;
const ROW_H: f32 = 26.0;
const LABEL_COL: f32 = 150.0;
const BUTTON_W: f32 = 58.0;
const GAP: f32 = 4.0;
// Rows indented under a list entry's heading.
const INDENT: f32 = 14.0;

const SECTION_TINT: [f32; 4] = [0.16, 0.17, 0.22, 1.0];
const ITEM_TINT: [f32; 4] = [0.13, 0.14, 0.18, 1.0];
const ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const APPLY_TINT: [f32; 4] = [0.22, 0.40, 0.56, 1.0];
const APPLY_TINT_HOVER: [f32; 4] = [0.28, 0.48, 0.66, 1.0];
const SECTION_LABEL: [f32; 3] = [0.70, 0.74, 0.84];
const ADD_LABEL: [f32; 3] = [0.55, 0.80, 0.60];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];
const LABEL_TOP: f32 = ROW_H * 0.5 - theme::TEXT_HALF;

/// The per-frame view the hook assembles: the rows (their reference pickers'
/// options filled), the first row the window shows, the focused row's input,
/// and the status line from the last rejected edit.
pub(crate) struct PlanetView<'a> {
    pub rows: &'a [Row],
    pub scroll: usize,
    pub focus: Option<usize>,
    pub status: Option<&'a str>,
    /// Typed edits are waiting on Apply: the heading marks them.
    pub dirty: bool,
    pub mouse: [f32; 2],
}

/// A resolved Planet-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanetAction {
    /// Give row `r`'s text input keyboard focus.
    Focus(usize),
    /// Move row `r`'s reference picker to its next option.
    CycleRef(usize),
    /// A row's button.
    Button(Button),
    /// Add a Planet to a world without one.
    AddPlanet,
    /// Give the planet a sea.
    AddSea,
    /// Write every typed box into the planet.
    Apply,
    /// Elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

/// Where the panel sits until it is dragged: the window's right edge.
pub(crate) fn default_origin(vp: [f32; 2]) -> [f32; 2] {
    [(vp[0] - PLANET_W - 8.0).max(8.0), 48.0]
}

/// The panel footprint for `n_rows` rows, the window showing at most
/// [`VISIBLE_ROWS`] of them.
pub(crate) fn size(n_rows: usize) -> [f32; 2] {
    let shown = n_rows.min(VISIBLE_ROWS);
    [
        PLANET_W,
        widget::TITLE_H + HEADER_H + shown as f32 * ROW_H + PAD,
    ]
}

/// The furthest the window scrolls over `n_rows` rows.
pub(crate) fn max_scroll(n_rows: usize) -> usize {
    n_rows.saturating_sub(VISIBLE_ROWS)
}

fn apply_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    [
        o[0] + w - PAD - APPLY_W,
        o[1] + widget::TITLE_H + 4.0,
        APPLY_W,
        HEADER_H - 8.0,
    ]
}

// Where the rows start: under the header row.
fn body_top(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H + HEADER_H
}

/// The window's slot `j`.
fn slot_rect(o: [f32; 2], w: f32, j: usize) -> [f32; 4] {
    [o[0], body_top(o) + j as f32 * ROW_H, w, ROW_H]
}

/// The rows' scrolling region.
pub(crate) fn body_rect(o: [f32; 2], w: f32, n_rows: usize) -> [f32; 4] {
    [
        o[0],
        body_top(o),
        w,
        n_rows.min(VISIBLE_ROWS) as f32 * ROW_H,
    ]
}

// Button `k` of a row, counted from its right end.
fn button_rect(row: [f32; 4], k: usize) -> [f32; 4] {
    let right = row[0] + row[2] - PAD - k as f32 * (BUTTON_W + GAP);
    [right - BUTTON_W, row[1] + 3.0, BUTTON_W, row[3] - 6.0]
}

// The control on a field row, left of its buttons.
fn control_rect(row: [f32; 4], buttons: usize) -> [f32; 4] {
    let x = row[0] + LABEL_COL;
    let reserved = buttons as f32 * (BUTTON_W + GAP);
    let w = (row[0] + row[2] - PAD - reserved - x).max(0.0);
    [x, row[1] + 3.0, w, row[3] - 6.0]
}

// Where a label centered in `rect` is anchored.
fn center(rect: [f32; 4]) -> [f32; 2] {
    [
        rect[0] + rect[2] * 0.5,
        rect[1] + rect[3] * 0.5 - theme::TEXT_HALF,
    ]
}

fn button_count(row: &Row) -> usize {
    row.buttons.iter().flatten().count()
}

// Whether a row sits under a list entry's heading.
fn indented(row: &Row) -> bool {
    row.field().is_some_and(|f| {
        f.key.starts_with(planet::List::Covers.key())
            || f.key.starts_with(planet::List::Layers.key())
    })
}

/// Resolve a click at `(mx, my)` on the panel at origin `o`, size `s`; `None`
/// when it missed the panel.
pub(crate) fn hit_test(
    view: &PlanetView<'_>,
    mx: f32,
    my: f32,
    o: [f32; 2],
    s: [f32; 2],
) -> Option<PlanetAction> {
    let w = s[0];
    if point_in(mx, my, apply_rect(o, w)) {
        return Some(PlanetAction::Apply);
    }
    let window = view
        .rows
        .iter()
        .enumerate()
        .skip(view.scroll)
        .take(VISIBLE_ROWS);
    for (j, (r, row)) in window.enumerate() {
        let rect = slot_rect(o, w, j);
        if !point_in(mx, my, rect) {
            continue;
        }
        let pressed = row
            .buttons
            .iter()
            .flatten()
            .enumerate()
            .find(|(k, _)| point_in(mx, my, button_rect(rect, *k)));
        if let Some((_, b)) = pressed {
            return Some(PlanetAction::Button(*b));
        }
        return Some(match &row.kind {
            RowKind::AddPlanet => PlanetAction::AddPlanet,
            RowKind::AddSea => PlanetAction::AddSea,
            RowKind::Field { field, .. } if matches!(field.kind, FieldKind::Ref { .. }) => {
                PlanetAction::CycleRef(r)
            }
            RowKind::Field { field, .. } if field.kind.has_text_input() => PlanetAction::Focus(r),
            _ => PlanetAction::Consume,
        });
    }
    point_in(mx, my, widget::outer_rect(o, s)).then_some(PlanetAction::Consume)
}

/// Position and show the panel at size `s`.
pub(crate) fn place(world: &mut World, view: &PlanetView<'_>, o: [f32; 2], s: [f32; 2]) {
    let w = s[0];
    widget::place_panel(world, PANEL_BG, widget::outer_rect(o, s));
    let title = widget::title_rect(o, w);
    let heading = if view.dirty { "Planet *" } else { "Planet" };
    widget::place_heading(world, TITLE_LABEL, title, heading);
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);
    place_header(world, view, o, w);

    for r in 0..planet::MAX_ROWS {
        widget::hide_field(world, input(r));
    }
    for j in 0..VISIBLE_ROWS {
        match view.rows.get(view.scroll + j) {
            Some(row) => place_row(world, view, view.scroll + j, row, slot_rect(o, w, j)),
            None => hide_slot(world, j),
        }
    }
}

// The status line, shown after a rejected edit, beside the Apply button.
fn place_header(world: &mut World, view: &PlanetView<'_>, o: [f32; 2], w: f32) {
    let apply = apply_rect(o, w);
    let hover = point_in(view.mouse[0], view.mouse[1], apply);
    let tint = if hover { APPLY_TINT_HOVER } else { APPLY_TINT };
    place_rounded(world, APPLY_BG, apply, tint, theme::CONTROL_RADIUS, true);
    widget::place_center_label(
        world,
        APPLY_LABEL,
        center(apply),
        "Apply",
        [1.0, 1.0, 1.0],
        true,
    );
    let left = o[0] + PAD;
    widget::place_message(
        world,
        STATUS_LABEL,
        [
            left,
            o[1] + widget::TITLE_H + HEADER_H * 0.5 - theme::TEXT_HALF,
            (apply[0] - PAD - left).max(0.0),
            widget::LINE_H,
        ],
        view.status.unwrap_or(""),
        ERROR_LABEL,
        view.status.is_some(),
    );
}

// Row `r` in the window slot at `rect`.
fn place_row(world: &mut World, view: &PlanetView<'_>, r: usize, row: &Row, rect: [f32; 4]) {
    let j = r - view.scroll;
    let hovered = point_in(view.mouse[0], view.mouse[1], rect);
    let (tint, color) = match row.kind {
        RowKind::Header(_) => (SECTION_TINT, SECTION_LABEL),
        RowKind::List { .. } | RowKind::Item { .. } => (ITEM_TINT, theme::LABEL),
        RowKind::AddPlanet | RowKind::AddSea if hovered => (theme::HOVER_TINT, ADD_LABEL),
        RowKind::AddPlanet | RowKind::AddSea => (ROW_TINT, ADD_LABEL),
        RowKind::Field { .. } => (ROW_TINT, theme::LABEL),
    };
    place_rounded(
        world,
        row_bg(j),
        theme::highlight_rect(rect),
        tint,
        theme::CONTROL_RADIUS,
        true,
    );
    let indent = if indented(row) { INDENT } else { 0.0 };
    if let Some(l) = widget::label_mut(world, row_label(j)) {
        l.x = rect[0] + PAD + indent;
        l.y = rect[1] + LABEL_TOP;
        l.align = TextAlign::Left;
        l.color = color;
        l.visible = true;
        l.content.clone_from(&row.caption);
    }
    place_buttons(world, view, row, rect, j);

    let buttons = button_count(row);
    let picker = row
        .field()
        .filter(|f| matches!(f.kind, FieldKind::Ref { .. }));
    match (row.field(), picker) {
        (_, Some(field)) => {
            let control = control_rect(rect, buttons);
            let tint = if point_in(view.mouse[0], view.mouse[1], control) {
                theme::HOVER_TINT
            } else {
                theme::BUTTON_TINT
            };
            place_rounded(
                world,
                picker_bg(j),
                control,
                tint,
                theme::CONTROL_RADIUS,
                true,
            );
            let shown = field
                .variants
                .get(field.variant_idx)
                .map_or(field.initial.as_str(), String::as_str);
            let at = [
                control[0] + GAP * 2.0,
                control[1] + control[3] * 0.5 - theme::TEXT_HALF,
            ];
            widget::place_left_label(world, picker_label(j), at, shown, theme::LABEL, true);
        }
        (Some(field), None) if field.kind.has_text_input() => {
            widget::show_field(
                world,
                input(r),
                control_rect(rect, buttons),
                view.focus == Some(r),
            );
            hide_picker(world, j);
        }
        _ => hide_picker(world, j),
    }
}

fn place_buttons(world: &mut World, view: &PlanetView<'_>, row: &Row, rect: [f32; 4], j: usize) {
    let mut placed = 0;
    for (k, button) in row.buttons.iter().flatten().enumerate() {
        let b = button_rect(rect, k);
        let tint = if point_in(view.mouse[0], view.mouse[1], b) {
            theme::HOVER_TINT
        } else {
            theme::BUTTON_TINT
        };
        let id = j * ROW_BUTTONS + k;
        place_rounded(world, button_bg(id), b, tint, theme::CONTROL_RADIUS, true);
        let label = button.label();
        widget::place_center_label(
            world,
            button_label(id),
            center(b),
            label,
            theme::LABEL,
            true,
        );
        placed += 1;
    }
    for k in placed..ROW_BUTTONS {
        widget::set_sprite_visible(world, button_bg(j * ROW_BUTTONS + k), false);
        widget::set_label_visible(world, button_label(j * ROW_BUTTONS + k), false);
    }
}

fn hide_picker(world: &mut World, j: usize) {
    widget::set_sprite_visible(world, picker_bg(j), false);
    widget::set_label_visible(world, picker_label(j), false);
}

fn hide_slot(world: &mut World, j: usize) {
    widget::set_sprite_visible(world, row_bg(j), false);
    widget::set_label_visible(world, row_label(j), false);
    hide_picker(world, j);
    for k in 0..ROW_BUTTONS {
        widget::set_sprite_visible(world, button_bg(j * ROW_BUTTONS + k), false);
        widget::set_label_visible(world, button_label(j * ROW_BUTTONS + k), false);
    }
}

#[cfg(test)]
mod tests;

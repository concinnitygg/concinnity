// src/editor/variables_panel.rs
//
// The Variables panel's layout half: the world's variable table as one row per
// variable, name and type and starting value in their own columns. The header
// adds a variable and says whether the table is authoritative; the toolbar acts
// on the selected row (retype it, type its value, remove it), and a row for a
// name the behaviors use but the table leaves out offers to declare it instead.
//
// The table is one asset, so this panel edits one asset's args the way the
// Behavior panel edits one behavior's: directly, committing as it goes, with the
// build's own checker reporting on the result. `hook/variables_edit.rs` owns the
// actions and `editor/variables.rs` turns the args into these rows.

use super::registry::{self, PanelKey};
use super::theme;
use super::variables::Row;
use super::widget::{self, place_rounded, point_in};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::Variables);
pub(crate) const PANEL_BG: AssetId = AssetId(BASE);
pub(crate) const TITLE_LABEL: AssetId = AssetId(BASE + 1);
pub(crate) const CLOSE_BG: AssetId = AssetId(BASE + 2);
pub(crate) const CLOSE_LABEL: AssetId = AssetId(BASE + 3);
pub(crate) const NEW_BG: AssetId = AssetId(BASE + 4);
pub(crate) const NEW_LABEL: AssetId = AssetId(BASE + 5);
pub(crate) const MODE_LABEL: AssetId = AssetId(BASE + 6);
pub(crate) const TYPE_BG: AssetId = AssetId(BASE + 7);
pub(crate) const TYPE_LABEL: AssetId = AssetId(BASE + 8);
pub(crate) const DEL_BG: AssetId = AssetId(BASE + 9);
pub(crate) const DEL_LABEL: AssetId = AssetId(BASE + 10);
pub(crate) const HEAD_NAME: AssetId = AssetId(BASE + 11);
pub(crate) const HEAD_TYPE: AssetId = AssetId(BASE + 18);
pub(crate) const HEAD_VALUE: AssetId = AssetId(BASE + 19);
pub(crate) const STATUS_BG: AssetId = AssetId(BASE + 12);
pub(crate) const STATUS_LABEL: AssetId = AssetId(BASE + 13);
pub(crate) const LIST_TRACK: AssetId = AssetId(BASE + 14);
pub(crate) const LIST_THUMB: AssetId = AssetId(BASE + 15);
pub(crate) const NAME_INPUT: AssetId = AssetId(BASE + 16);
pub(crate) const VALUE_INPUT: AssetId = AssetId(BASE + 17);

pub(crate) fn row_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x40 + i as u32)
}
pub(crate) fn row_name(i: usize) -> AssetId {
    AssetId(BASE + 0x80 + i as u32)
}
pub(crate) fn row_type(i: usize) -> AssetId {
    AssetId(BASE + 0xC0 + i as u32)
}
pub(crate) fn row_value(i: usize) -> AssetId {
    AssetId(BASE + 0x100 + i as u32)
}

// Geometry, in window pixels, all derived from the panel origin `o`.
pub(crate) const VARIABLES_W: f32 = 460.0;
const PAD: f32 = 10.0;
const GAP: f32 = 6.0;
const HEADER_H: f32 = 32.0;
const TOOL_H: f32 = 30.0;
const HEAD_ROW_H: f32 = 20.0;
const ROW_H: f32 = 22.0;
const CTRL_H: f32 = 22.0;
const SCROLLBAR_W: f32 = 5.0;
const NEW_W: f32 = 58.0;
const BTN_W: f32 = 60.0;
const STATUS_H: f32 = 2.0 * widget::LINE_H + 6.0;
// The three columns, as offsets from a row's left edge.
const TYPE_X: f32 = 170.0;
const VALUE_X: f32 = 260.0;
// Visible rows at the default height; more appear as the panel grows.
pub(crate) const ROW_POOL: usize = 12;
pub(crate) const ROW_POOL_MAX: usize = 30;
const CHAR_W: f32 = 8.5;

const CHROME_H: f32 = widget::TITLE_H + HEADER_H + TOOL_H + HEAD_ROW_H;

const ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const NEW_TINT: [f32; 4] = [0.20, 0.44, 0.30, 1.0];
const DEL_TINT: [f32; 4] = [0.44, 0.22, 0.24, 1.0];
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
const WARN_TINT: [f32; 4] = [0.24, 0.18, 0.07, 1.0];
const WARN_BORDER: [f32; 4] = [0.55, 0.42, 0.18, 1.0];
// An undeclared name is what the table is missing, not what it holds.
const MISSING_LABEL: [f32; 3] = theme::LOG_WARN;

pub(crate) fn visible_rows(h: f32) -> usize {
    (((h - CHROME_H - PAD - STATUS_H) / ROW_H).floor() as usize).clamp(ROW_POOL, ROW_POOL_MAX)
}

pub(crate) fn size() -> [f32; 2] {
    [
        VARIABLES_W,
        CHROME_H + ROW_POOL as f32 * ROW_H + PAD + STATUS_H,
    ]
}

pub(crate) fn max_size() -> [f32; 2] {
    [
        VARIABLES_W * 2.0,
        CHROME_H + ROW_POOL_MAX as f32 * ROW_H + PAD + STATUS_H,
    ]
}

pub(crate) fn default_origin(vp_w: f32) -> [f32; 2] {
    [(vp_w - VARIABLES_W - 40.0).max(20.0), 140.0]
}

fn header_y(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H + (HEADER_H - CTRL_H) * 0.5
}

fn tool_y(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H + HEADER_H + (TOOL_H - CTRL_H) * 0.5
}

pub(crate) fn new_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    [o[0] + w - PAD - NEW_W, header_y(o), NEW_W, CTRL_H]
}

pub(crate) fn type_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0] + PAD, tool_y(o), BTN_W, CTRL_H]
}

pub(crate) fn delete_rect(o: [f32; 2]) -> [f32; 4] {
    let t = type_rect(o);
    [t[0] + BTN_W + GAP, t[1], BTN_W, CTRL_H]
}

pub(crate) fn value_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    let left = delete_rect(o)[0] + BTN_W + GAP;
    [left, tool_y(o), (o[0] + w - PAD - left).max(0.0), CTRL_H]
}

// The name field edits the selected variable's name, in the header where the
// panel's one editable identity belongs.
pub(crate) fn name_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    let right = new_rect(o, w)[0] - GAP;
    [
        o[0] + PAD,
        header_y(o),
        (right - o[0] - PAD).max(0.0),
        CTRL_H,
    ]
}

fn head_row_y(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H + HEADER_H + TOOL_H
}

fn body_top(o: [f32; 2]) -> f32 {
    head_row_y(o) + HEAD_ROW_H
}

pub(crate) fn row_rect(o: [f32; 2], s: [f32; 2], slot: usize) -> [f32; 4] {
    [
        o[0] + PAD,
        body_top(o) + slot as f32 * ROW_H,
        (s[0] - 2.0 * PAD).max(0.0),
        ROW_H,
    ]
}

pub(crate) fn status_rect(o: [f32; 2], s: [f32; 2]) -> [f32; 4] {
    let bottom = o[1] + s[1] - PAD;
    // Grown up to a row boundary, so the banner's edge never bisects the row it
    // floats over (the same rule the Behavior panel's banner follows).
    let rows = ((bottom - STATUS_H - body_top(o)) / ROW_H).floor().max(0.0);
    let top = body_top(o) + rows * ROW_H;
    [o[0] + PAD, top, (s[0] - 2.0 * PAD).max(0.0), bottom - top]
}

pub(crate) fn cursor_over_body(mx: f32, my: f32, o: [f32; 2], s: [f32; 2]) -> bool {
    let p = widget::outer_rect(o, s);
    mx >= p[0] && mx < p[0] + p[2] && my >= body_top(o) && my < p[1] + p[3]
}

// The per-frame view the hook assembles.
pub(crate) struct VariablesView<'a> {
    pub rows: &'a [Row],
    pub scroll: usize,
    pub selected: Option<usize>,
    // Whether the world declares a table at all, which is what decides between
    // calling the names implicit and holding the table to them.
    pub authoritative: bool,
    // Whether the name / value fields assert keyboard focus this frame.
    pub name_focus: bool,
    pub value_focus: bool,
    // What the panel has to say about the table, if anything.
    pub status: Option<&'a str>,
    // Whether the value column carries a live session's current values rather
    // than the declared starting values (retitles the heading).
    pub live: bool,
    pub mouse: [f32; 2],
}

impl VariablesView<'_> {
    fn row_at(&self, slot: usize) -> Option<&Row> {
        self.rows.get(self.scroll + slot)
    }

    fn selected_row(&self) -> Option<&Row> {
        self.selected.and_then(|i| self.rows.get(i))
    }

    // How many names the behaviors use that the table leaves out.
    fn missing(&self) -> usize {
        self.rows.iter().filter(|r| !r.declared()).count()
    }
}

// A resolved Variables-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VariablesAction {
    // Declare a new variable and select it.
    New,
    // Select row `i` (an absolute index).
    Select(usize),
    // Declare the selected undeclared name, keeping the name behaviors use.
    Declare,
    // Step the selected variable's type through the literal kinds.
    Retype,
    // Remove the selected declaration.
    Remove,
    // Give the name / value field keyboard focus.
    FocusName,
    FocusValue,
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

pub(crate) fn hit_test(
    view: &VariablesView,
    mx: f32,
    my: f32,
    o: [f32; 2],
    s: [f32; 2],
) -> Option<VariablesAction> {
    let w = s[0];
    if point_in(mx, my, new_rect(o, w)) {
        return Some(VariablesAction::New);
    }
    if let Some(row) = view.selected_row().filter(|r| !r.local) {
        if row.declared() && point_in(mx, my, name_rect(o, w)) {
            return Some(VariablesAction::FocusName);
        }
        // One chip declares an undeclared name and retypes a declared one: both
        // are "give this variable a type", which is the one thing a name the
        // table is missing needs. A live local is neither: it is not the
        // table's to declare, so the toolbar stands down on it (the filter
        // above).
        if point_in(mx, my, type_rect(o)) {
            return Some(match row.declared() {
                true => VariablesAction::Retype,
                false => VariablesAction::Declare,
            });
        }
        if row.declared() && point_in(mx, my, delete_rect(o)) {
            return Some(VariablesAction::Remove);
        }
        if row.declared() && point_in(mx, my, value_rect(o, w)) {
            return Some(VariablesAction::FocusValue);
        }
    }
    for slot in 0..visible_rows(s[1]) {
        if point_in(mx, my, row_rect(o, s, slot)) {
            return Some(match view.rows.get(view.scroll + slot) {
                Some(_) => VariablesAction::Select(view.scroll + slot),
                None => VariablesAction::Consume,
            });
        }
    }
    point_in(mx, my, widget::outer_rect(o, s)).then_some(VariablesAction::Consume)
}

// Position + show the panel (`Some(view)`) at effective size `s`, or blank every
// element (`None`).
pub(crate) fn apply(world: &mut World, view: Option<&VariablesView>, o: [f32; 2], s: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };
    let w = s[0];
    widget::place_panel(world, PANEL_BG, widget::outer_rect(o, s));
    let title = widget::title_rect(o, w);
    widget::place_heading(world, TITLE_LABEL, title, "Variables");
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);

    layout_header(world, view, o, w);
    layout_toolbar(world, view, o, w);
    layout_columns(world, view, o, w);
    layout_rows(world, view, o, s);
    layout_scrollbar(world, view, o, w, visible_rows(s[1]));
    layout_status(world, view, o, s);
}

fn layout_header(world: &mut World, view: &VariablesView, o: [f32; 2], w: f32) {
    let new = new_rect(o, w);
    let new_hover = point_in(view.mouse[0], view.mouse[1], new);
    place_rounded(
        world,
        NEW_BG,
        new,
        tint(NEW_TINT, true, new_hover),
        theme::CONTROL_RADIUS,
        true,
    );
    widget::place_left_label(
        world,
        NEW_LABEL,
        label_at(new, "New"),
        "New",
        theme::HEADING,
        true,
    );
    // The name field belongs to the selected declaration; with nothing selected
    // the header says what the table is instead.
    let field = name_rect(o, w);
    match view.selected_row().filter(|r| r.declared()) {
        Some(row) => {
            widget::set_label_visible(world, MODE_LABEL, false);
            if !view.name_focus {
                widget::seed_field(world, NAME_INPUT, &row.name);
            }
            widget::show_field(world, NAME_INPUT, field, view.name_focus);
        }
        None => {
            widget::hide_field(world, NAME_INPUT);
            let (text, color) = table_caption(view);
            widget::place_left_label(
                world,
                MODE_LABEL,
                [field[0], field[1] + CTRL_H * 0.5 - theme::TEXT_HALF],
                &widget::clip_text(&text, (field[2] / CHAR_W) as usize),
                color,
                true,
            );
        }
    }
}

// What the header says about the table as a whole. A declared table is held to
// every name its behaviors use, so what it is missing is the thing worth saying.
fn table_caption(view: &VariablesView) -> (String, [f32; 3]) {
    let missing = view.missing();
    match (view.authoritative, missing) {
        (false, 0) => (
            "no table: variables are implicit ints".to_string(),
            theme::LABEL_DIM,
        ),
        (false, n) => (
            format!("no table: {n} implicit int variable{}", plural(n)),
            theme::LABEL_DIM,
        ),
        (true, 0) => ("every variable is declared".to_string(), theme::LABEL_DIM),
        (true, n) => (
            format!("{n} variable{} used but not declared", plural(n)),
            MISSING_LABEL,
        ),
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn layout_toolbar(world: &mut World, view: &VariablesView, o: [f32; 2], w: f32) {
    let row = view.selected_row();
    let declared = row.is_some_and(Row::declared);
    let type_caption = match row {
        Some(r) if !r.declared() => "Declare",
        _ => "Type",
    };
    for (bg, label, rect, caption, base, live) in [
        (
            TYPE_BG,
            TYPE_LABEL,
            type_rect(o),
            type_caption,
            theme::BUTTON_TINT,
            row.is_some(),
        ),
        (DEL_BG, DEL_LABEL, delete_rect(o), "Del", DEL_TINT, declared),
    ] {
        let hover = live && point_in(view.mouse[0], view.mouse[1], rect);
        place_rounded(
            world,
            bg,
            rect,
            tint(base, live, hover),
            theme::CONTROL_RADIUS,
            true,
        );
        widget::place_left_label(
            world,
            label,
            label_at(rect, caption),
            caption,
            if live { theme::LABEL } else { theme::LABEL_DIM },
            true,
        );
    }
    match declared {
        true => widget::show_field(world, VALUE_INPUT, value_rect(o, w), view.value_focus),
        false => widget::hide_field(world, VALUE_INPUT),
    }
}

// The column headings. One label per column at the offset its rows use: the HUD
// font is proportional, so padding a single string out to the column would not
// line up with anything. While a session runs, the value column shows what
// each variable holds now rather than what it starts at.
fn layout_columns(world: &mut World, view: &VariablesView, o: [f32; 2], _w: f32) {
    let y = head_row_y(o) + HEAD_ROW_H * 0.5 - theme::TEXT_HALF;
    let left = o[0] + PAD;
    let value_caption = if view.live { "now" } else { "starts at" };
    for (id, x, caption) in [
        (HEAD_NAME, PAD, "name"),
        (HEAD_TYPE, TYPE_X, "type"),
        (HEAD_VALUE, VALUE_X, value_caption),
    ] {
        widget::place_left_label(world, id, [left + x, y], caption, theme::LABEL_DIM, true);
    }
}

fn layout_rows(world: &mut World, view: &VariablesView, o: [f32; 2], s: [f32; 2]) {
    let shown = visible_rows(s[1]);
    let budget = |from: f32, to: f32| (((to - from) / CHAR_W) as usize).max(4);
    for slot in 0..ROW_POOL_MAX {
        let Some(row) = view.row_at(slot).filter(|_| slot < shown) else {
            widget::set_sprite_visible(world, row_bg(slot), false);
            for id in [row_name(slot), row_type(slot), row_value(slot)] {
                widget::set_label_visible(world, id, false);
            }
            continue;
        };
        let r = row_rect(o, s, slot);
        let hovered = point_in(view.mouse[0], view.mouse[1], r);
        let selected = view.selected == Some(view.scroll + slot);
        place_rounded(
            world,
            row_bg(slot),
            theme::highlight_rect(r),
            if selected {
                theme::SELECTED_TINT
            } else if hovered {
                theme::HOVER_TINT
            } else {
                ROW_TINT
            },
            theme::CONTROL_RADIUS,
            true,
        );
        let y = r[1] + ROW_H * 0.5 - theme::TEXT_HALF;
        // An undeclared name reads in the warning colour and says so in the type
        // column: the row is a prompt, not a declaration. A live session's
        // behavior local reads in the command colour: inspectable, not part of
        // the table.
        let (color, ty) = if row.local {
            (theme::LOG_COMMAND, format!("{} local", row.ty))
        } else if row.declared() {
            (theme::LABEL, row.ty.clone())
        } else {
            (MISSING_LABEL, "undeclared".to_string())
        };
        widget::place_left_label(
            world,
            row_name(slot),
            [r[0] + PAD, y],
            &widget::clip_text(&row.name, budget(PAD, TYPE_X)),
            color,
            true,
        );
        widget::place_left_label(
            world,
            row_type(slot),
            [r[0] + TYPE_X, y],
            &widget::clip_text(&ty, budget(TYPE_X, VALUE_X)),
            color,
            true,
        );
        widget::place_left_label(
            world,
            row_value(slot),
            [r[0] + VALUE_X, y],
            &widget::clip_text(&row.value, budget(VALUE_X, r[2])),
            theme::LABEL,
            row.declared(),
        );
    }
}

fn layout_scrollbar(world: &mut World, view: &VariablesView, o: [f32; 2], w: f32, shown: usize) {
    let total = view.rows.len();
    if total <= shown {
        widget::set_sprite_visible(world, LIST_TRACK, false);
        widget::set_sprite_visible(world, LIST_THUMB, false);
        return;
    }
    let x = o[0] + w - SCROLLBAR_W;
    let top = body_top(o);
    let h = shown as f32 * ROW_H;
    place_rounded(
        world,
        LIST_TRACK,
        [x, top, SCROLLBAR_W, h],
        TRACK_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
    let thumb_h = (h * shown as f32 / total as f32).max(18.0);
    let off = (h - thumb_h) * (view.scroll.min(total - shown) as f32 / (total - shown) as f32);
    place_rounded(
        world,
        LIST_THUMB,
        [x, top + off, SCROLLBAR_W, thumb_h],
        THUMB_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
}

fn layout_status(world: &mut World, view: &VariablesView, o: [f32; 2], s: [f32; 2]) {
    let Some(text) = view.status else {
        widget::set_sprite_visible(world, STATUS_BG, false);
        widget::set_label_visible(world, STATUS_LABEL, false);
        return;
    };
    let b = status_rect(o, s);
    widget::place_bordered(world, STATUS_BG, b, WARN_TINT, WARN_BORDER, 1.0);
    widget::place_message(
        world,
        STATUS_LABEL,
        [b[0] + PAD, b[1] + 3.0, b[2] - 2.0 * PAD, b[3] - 6.0],
        text,
        theme::LOG_WARN,
        true,
    );
}

fn tint(base: [f32; 4], live: bool, hovered: bool) -> [f32; 4] {
    if !live {
        return theme::BUTTON_TINT;
    }
    if hovered { theme::HOVER_TINT } else { base }
}

// A chip's caption, centred in its rect.
fn label_at(rect: [f32; 4], caption: &str) -> [f32; 2] {
    let text_w = caption.chars().count() as f32 * CHAR_W;
    [
        rect[0] + (rect[2] - text_w) * 0.5,
        rect[1] + rect[3] * 0.5 - theme::TEXT_HALF,
    ]
}

pub(crate) fn hide_all(world: &mut World) {
    for id in all_sprite_ids() {
        widget::set_sprite_visible(world, id, false);
    }
    for id in all_label_ids() {
        widget::set_label_visible(world, id, false);
    }
    for id in all_field_ids() {
        widget::hide_field(world, id);
    }
}

pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![
        PANEL_BG, CLOSE_BG, NEW_BG, TYPE_BG, DEL_BG, STATUS_BG, LIST_TRACK, LIST_THUMB,
    ];
    ids.extend((0..ROW_POOL_MAX).map(row_bg));
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![
        TITLE_LABEL,
        CLOSE_LABEL,
        NEW_LABEL,
        MODE_LABEL,
        TYPE_LABEL,
        DEL_LABEL,
        HEAD_NAME,
        HEAD_TYPE,
        HEAD_VALUE,
        STATUS_LABEL,
    ];
    ids.extend((0..ROW_POOL_MAX).map(row_name));
    ids.extend((0..ROW_POOL_MAX).map(row_type));
    ids.extend((0..ROW_POOL_MAX).map(row_value));
    ids
}

pub(crate) fn all_field_ids() -> Vec<AssetId> {
    vec![NAME_INPUT, VALUE_INPUT]
}

pub(crate) fn status_ids() -> Vec<AssetId> {
    vec![STATUS_BG, STATUS_LABEL]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextInput, TextLabel};

    fn injected_world() -> World {
        let mut world = World::new_empty();
        for id in all_sprite_ids() {
            world.add_component(Sprite {
                asset_id: id,
                ..Default::default()
            });
        }
        for id in all_label_ids() {
            world.add_component(TextLabel {
                asset_id: id,
                ..Default::default()
            });
        }
        for id in all_field_ids() {
            world.add_component(TextInput {
                asset_id: id,
                ..Default::default()
            });
        }
        world
    }

    fn label(world: &World, id: AssetId) -> TextLabel {
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .cloned()
            .unwrap()
    }

    fn sprite(world: &World, id: AssetId) -> Sprite {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == id)
            .cloned()
            .unwrap()
    }

    fn field(world: &World, id: AssetId) -> TextInput {
        world
            .query::<TextInput>()
            .find(|t| t.asset_id == id)
            .cloned()
            .unwrap()
    }

    fn declared(name: &str, ty: &str, value: &str) -> Row {
        Row {
            name: name.to_string(),
            at: Some(0),
            ty: ty.to_string(),
            value: value.to_string(),
            local: false,
        }
    }

    fn undeclared(name: &str) -> Row {
        Row {
            name: name.to_string(),
            at: None,
            ty: String::new(),
            value: String::new(),
            local: false,
        }
    }

    fn view(rows: &[Row]) -> VariablesView<'_> {
        VariablesView {
            rows,
            scroll: 0,
            selected: None,
            authoritative: true,
            name_focus: false,
            value_focus: false,
            status: None,
            live: false,
            mouse: [-1.0, -1.0],
        }
    }

    #[test]
    fn a_declaration_draws_its_name_type_and_starting_value_in_columns() {
        let mut world = injected_world();
        let rows = vec![declared("visits", "int", "3")];
        let o = [20.0, 20.0];
        let s = size();
        apply(&mut world, Some(&view(&rows)), o, s);

        assert_eq!(label(&world, row_name(0)).content, "visits");
        assert_eq!(label(&world, row_type(0)).content, "int");
        assert_eq!(label(&world, row_value(0)).content, "3");
        // The columns line up left to right and stay inside the row.
        assert!(label(&world, row_name(0)).x < label(&world, row_type(0)).x);
        assert!(label(&world, row_type(0)).x < label(&world, row_value(0)).x);
        let r = row_rect(o, s, 0);
        assert!(label(&world, row_value(0)).x < r[0] + r[2]);
        // Each heading sits exactly over the column it names, which a single
        // space-padded label could not do in a proportional font.
        for (head, cell) in [
            (HEAD_NAME, row_name(0)),
            (HEAD_TYPE, row_type(0)),
            (HEAD_VALUE, row_value(0)),
        ] {
            assert_eq!(
                label(&world, head).x,
                label(&world, cell).x,
                "{head:?} does not line up with its column",
            );
        }
        assert_eq!(label(&world, HEAD_VALUE).content, "starts at");
    }

    // A name the behaviors use that the table leaves out is a prompt, not a
    // declaration: it says so in the type column and offers no value.
    #[test]
    fn an_undeclared_name_reads_as_missing_rather_than_as_a_declaration() {
        let mut world = injected_world();
        let rows = vec![declared("visits", "int", "3"), undeclared("score")];
        apply(&mut world, Some(&view(&rows)), [20.0, 20.0], size());

        assert_eq!(label(&world, row_name(1)).content, "score");
        assert_eq!(label(&world, row_type(1)).content, "undeclared");
        assert!(!label(&world, row_value(1)).visible, "it has no value yet");
        assert_ne!(
            label(&world, row_name(1)).color,
            label(&world, row_name(0)).color,
            "and it does not read like a declared one",
        );
    }

    // The header says what the table is when nothing is selected, because that is
    // what decides whether a missing name is a problem at all.
    #[test]
    fn the_header_says_whether_the_table_is_authoritative() {
        let mut world = injected_world();
        let rows = vec![declared("visits", "int", "3"), undeclared("score")];
        let o = [20.0, 20.0];
        apply(&mut world, Some(&view(&rows)), o, size());
        let held = label(&world, MODE_LABEL).content.clone();
        assert!(held.contains("1 variable used but not declared"), "{held}");

        let implicit = VariablesView {
            authoritative: false,
            ..view(&rows)
        };
        apply(&mut world, Some(&implicit), o, size());
        let loose = label(&world, MODE_LABEL).content.clone();
        assert!(loose.contains("no table"), "{loose}");
        assert!(!field(&world, NAME_INPUT).visible, "and no name to edit");
    }

    // Selecting a declaration hands the header over to its name field and lights
    // the toolbar that edits it.
    #[test]
    fn selecting_a_declaration_offers_its_name_type_value_and_removal() {
        let mut world = injected_world();
        let rows = vec![declared("visits", "int", "3")];
        let o = [20.0, 20.0];
        let s = size();
        let v = VariablesView {
            selected: Some(0),
            ..view(&rows)
        };
        apply(&mut world, Some(&v), o, s);

        assert!(field(&world, NAME_INPUT).visible);
        assert_eq!(field(&world, NAME_INPUT).content, "visits");
        assert!(field(&world, VALUE_INPUT).visible);
        assert_eq!(label(&world, TYPE_LABEL).content, "Type");
        assert_eq!(label(&world, DEL_LABEL).color, theme::LABEL);

        let n = name_rect(o, s[0]);
        assert_eq!(
            hit_test(&v, n[0] + 3.0, n[1] + 3.0, o, s),
            Some(VariablesAction::FocusName),
        );
        for (rect, want) in [
            (type_rect(o), VariablesAction::Retype),
            (delete_rect(o), VariablesAction::Remove),
            (value_rect(o, s[0]), VariablesAction::FocusValue),
        ] {
            assert_eq!(
                hit_test(&v, rect[0] + 3.0, rect[1] + 3.0, o, s),
                Some(want),
                "{rect:?}",
            );
        }
    }

    // On an undeclared name the one chip declares it instead of retyping it, and
    // there is nothing to remove or to type a value into.
    #[test]
    fn selecting_an_undeclared_name_offers_only_declaring_it() {
        let mut world = injected_world();
        let rows = vec![undeclared("score")];
        let o = [20.0, 20.0];
        let s = size();
        let v = VariablesView {
            selected: Some(0),
            ..view(&rows)
        };
        apply(&mut world, Some(&v), o, s);

        assert_eq!(label(&world, TYPE_LABEL).content, "Declare");
        assert_eq!(label(&world, DEL_LABEL).color, theme::LABEL_DIM);
        assert!(!field(&world, NAME_INPUT).visible, "it has no name to edit");
        assert!(!field(&world, VALUE_INPUT).visible);

        let t = type_rect(o);
        assert_eq!(
            hit_test(&v, t[0] + 3.0, t[1] + 3.0, o, s),
            Some(VariablesAction::Declare),
        );
        let d = delete_rect(o);
        assert_eq!(
            hit_test(&v, d[0] + 3.0, d[1] + 3.0, o, s),
            Some(VariablesAction::Consume),
            "the dim chip swallows rather than acting",
        );
    }

    #[test]
    fn a_row_press_selects_it_and_the_new_chip_declares_one() {
        let rows = vec![declared("visits", "int", "3"), undeclared("score")];
        let o = [20.0, 20.0];
        let s = size();
        let v = view(&rows);
        let r = row_rect(o, s, 1);
        assert_eq!(
            hit_test(&v, r[0] + 3.0, r[1] + 3.0, o, s),
            Some(VariablesAction::Select(1)),
        );
        let n = new_rect(o, s[0]);
        assert_eq!(
            hit_test(&v, n[0] + 3.0, n[1] + 3.0, o, s),
            Some(VariablesAction::New),
        );
        // A press past the last row lands on nothing to select.
        let empty = row_rect(o, s, 5);
        assert_eq!(
            hit_test(&v, empty[0] + 3.0, empty[1] + 3.0, o, s),
            Some(VariablesAction::Consume),
        );
        // And a press right outside the panel is not this panel's at all.
        let p = widget::outer_rect(o, s);
        assert_eq!(hit_test(&v, p[0] - 4.0, p[1] + 40.0, o, s), None);
    }

    // The scrolled window maps its slots to absolute rows, or a press would
    // select whatever happened to be drawn in that slot's place.
    #[test]
    fn a_scrolled_window_draws_and_selects_absolute_rows() {
        let mut world = injected_world();
        let rows: Vec<Row> = (0..20)
            .map(|i| declared(&format!("v{i}"), "int", "0"))
            .collect();
        let o = [20.0, 20.0];
        let s = size();
        let v = VariablesView {
            scroll: 4,
            ..view(&rows)
        };
        apply(&mut world, Some(&v), o, s);
        assert_eq!(label(&world, row_name(0)).content, "v4");
        assert!(sprite(&world, LIST_THUMB).visible, "20 rows overflow");

        let r = row_rect(o, s, 2);
        assert_eq!(
            hit_test(&v, r[0] + 3.0, r[1] + 3.0, o, s),
            Some(VariablesAction::Select(6)),
        );
    }

    // The banner is what tells the author a declared table is about to fail the
    // build, so it floats over the foot of the rows without bisecting one.
    #[test]
    fn the_status_banner_floats_over_a_row_boundary() {
        let mut world = injected_world();
        let rows = vec![declared("visits", "int", "3")];
        let o = [20.0, 20.0];
        let s = size();
        apply(&mut world, Some(&view(&rows)), o, s);
        assert!(!sprite(&world, STATUS_BG).visible, "nothing to warn about");

        let warned = VariablesView {
            status: Some("this table is authoritative"),
            ..view(&rows)
        };
        apply(&mut world, Some(&warned), o, s);
        assert!(sprite(&world, STATUS_BG).visible);
        let b = status_rect(o, s);
        assert!(b[1] + b[3] <= o[1] + s[1], "it stays inside the panel");
        let from_top = b[1] - row_rect(o, s, 0)[1];
        assert_eq!(from_top % ROW_H, 0.0, "its edge lands between rows");
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        let rows = vec![declared("visits", "int", "3")];
        let v = VariablesView {
            selected: Some(0),
            name_focus: true,
            status: Some("something"),
            ..view(&rows)
        };
        apply(&mut world, Some(&v), [20.0, 20.0], size());
        apply(&mut world, None, [0.0, 0.0], size());
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
        assert!(world.query::<TextInput>().all(|t| !t.visible && !t.focused));
    }
}

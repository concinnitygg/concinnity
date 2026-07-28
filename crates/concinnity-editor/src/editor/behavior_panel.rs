// src/editor/behavior_panel.rs
//
// The Behavior panel's layout half: a floating node-graph editor over one
// `Behavior` asset. The header cycles through the world's behaviors and adds
// new ones; the toolbar acts on the selected row (open its palette, delete it,
// move it, or type its value); the body is the whole asset as one indented
// outline of nodes and their expression trees (`behavior/outline.rs`), and the
// status line carries the world checker's message for the behavior as it
// stands. The palette opens as a floating list over the body, so picking a node
// or an expression never leaves the panel. `hook/behavior_edit.rs` owns the
// actions.

use super::behavior::edit::{self, Pick};
use super::behavior::graph::Chart;
use super::behavior::outline::{Kind, Row};
use super::behavior_chart;
use super::registry::{self, PanelKey};
use super::theme;
use super::widget::{self, place_rounded, point_in};
use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::Behavior);
pub(crate) const PANEL_BG: AssetId = AssetId(BASE);
pub(crate) const TITLE_LABEL: AssetId = AssetId(BASE + 1);
pub(crate) const CLOSE_BG: AssetId = AssetId(BASE + 2);
pub(crate) const CLOSE_LABEL: AssetId = AssetId(BASE + 3);
pub(crate) const PREV_BG: AssetId = AssetId(BASE + 4);
pub(crate) const PREV_LABEL: AssetId = AssetId(BASE + 5);
pub(crate) const NEXT_BG: AssetId = AssetId(BASE + 6);
pub(crate) const NEXT_LABEL: AssetId = AssetId(BASE + 7);
pub(crate) const NAME_LABEL: AssetId = AssetId(BASE + 8);
pub(crate) const NEW_BG: AssetId = AssetId(BASE + 9);
pub(crate) const NEW_LABEL: AssetId = AssetId(BASE + 10);
pub(crate) const PICK_BG: AssetId = AssetId(BASE + 11);
pub(crate) const PICK_LABEL: AssetId = AssetId(BASE + 12);
pub(crate) const DEL_BG: AssetId = AssetId(BASE + 13);
pub(crate) const DEL_LABEL: AssetId = AssetId(BASE + 14);
pub(crate) const UP_BG: AssetId = AssetId(BASE + 15);
pub(crate) const UP_LABEL: AssetId = AssetId(BASE + 16);
pub(crate) const DOWN_BG: AssetId = AssetId(BASE + 17);
pub(crate) const DOWN_LABEL: AssetId = AssetId(BASE + 18);
pub(crate) const STATUS_LABEL: AssetId = AssetId(BASE + 19);
pub(crate) const LIST_TRACK: AssetId = AssetId(BASE + 20);
pub(crate) const LIST_THUMB: AssetId = AssetId(BASE + 21);
pub(crate) const DROP_BG: AssetId = AssetId(BASE + 22);
pub(crate) const DROP_TRACK: AssetId = AssetId(BASE + 23);
pub(crate) const DROP_THUMB: AssetId = AssetId(BASE + 24);
pub(crate) const VALUE_INPUT: AssetId = AssetId(BASE + 25);
pub(crate) const VIEW_BG: AssetId = AssetId(BASE + 26);
pub(crate) const VIEW_LABEL: AssetId = AssetId(BASE + 27);

pub(crate) fn row_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x40 + i as u32)
}
pub(crate) fn row_label(i: usize) -> AssetId {
    AssetId(BASE + 0x80 + i as u32)
}
pub(crate) fn row_value(i: usize) -> AssetId {
    AssetId(BASE + 0xC0 + i as u32)
}
pub(crate) fn pick_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x100 + i as u32)
}
pub(crate) fn pick_label(i: usize) -> AssetId {
    AssetId(BASE + 0x120 + i as u32)
}
pub(crate) fn pick_hint(i: usize) -> AssetId {
    AssetId(BASE + 0x140 + i as u32)
}

// Geometry, in window pixels. Every rect derives from the panel origin `o` (the
// title bar's top-left), so dragging the title bar moves the whole panel. The
// default (and minimum) width; the user can widen the panel past this.
pub(crate) const BEHAVIOR_W: f32 = 620.0;
const PAD: f32 = 10.0;
const GAP: f32 = 6.0;
const HEADER_H: f32 = 32.0;
const TOOL_H: f32 = 30.0;
// Two lines: a checker message is a sentence and often needs the second, and a
// fixed band keeps the body from shifting as messages come and go.
const STATUS_H: f32 = 2.0 * widget::LINE_H + 6.0;
const ROW_H: f32 = 22.0;
const SCROLLBAR_W: f32 = 5.0;
const STEP_W: f32 = 26.0;
const NEW_W: f32 = 58.0;
const VIEW_W: f32 = 62.0;
const TOOL_BTN_W: f32 = 52.0;
const ARROW_W: f32 = 30.0;
// Visible outline rows at the default height; a longer outline scrolls.
// Resizing the panel taller reveals more, up to the injected pool.
pub(crate) const ROW_POOL: usize = 16;
pub(crate) const ROW_POOL_MAX: usize = 34;
// Visible palette options before the palette scrolls.
pub(crate) const PICK_POOL: usize = 10;
const PICK_ROW_H: f32 = 24.0;
// Value-column caption budget at the default width; widening the panel grows
// it. A row's label is budgeted by the column it stops at instead.
const MAX_VALUE_CHARS: usize = 34;
// Approximate body-font advance, for growing the caption budgets with the width.
pub(crate) const CHAR_W: f32 = 8.5;
// How far each outline level insets its caption.
const INDENT: f32 = 14.0;

// The chrome height above the outline (everything but the row band and pad).
const CHROME_H: f32 = widget::TITLE_H + HEADER_H + TOOL_H + STATUS_H;

pub(crate) fn visible_rows(h: f32) -> usize {
    (((h - CHROME_H - PAD) / ROW_H).floor() as usize).clamp(ROW_POOL, ROW_POOL_MAX)
}

fn max_value_chars(w: f32) -> usize {
    (MAX_VALUE_CHARS as f32 + (w - BEHAVIOR_W) / CHAR_W).max(8.0) as usize
}

// The column the value text starts at, so labels and values line up down the
// outline however deep the tree goes.
fn value_column(w: f32) -> f32 {
    (w * 0.38).max(150.0)
}

const ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const BTN_TINT: [f32; 4] = theme::BUTTON_TINT;
const NEW_TINT: [f32; 4] = [0.20, 0.44, 0.30, 1.0];
const DEL_TINT: [f32; 4] = [0.44, 0.22, 0.24, 1.0];
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
const DROP_TINT: [f32; 4] = [0.09, 0.09, 0.12, 1.0];
const OK_LABEL: [f32; 3] = [0.55, 0.85, 0.65];
const HINT_LABEL: [f32; 3] = theme::LABEL_DIM;

// Which way the body is shown. Both views read the same rows: the chart draws
// the control flow as cards, the outline every field of every node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ViewMode {
    #[default]
    Outline,
    Chart,
}

impl ViewMode {
    pub(crate) fn other(self) -> ViewMode {
        match self {
            Self::Outline => Self::Chart,
            Self::Chart => Self::Outline,
        }
    }

    // The toggle is captioned with the view it switches to, so the button says
    // what pressing it does rather than what is already on screen.
    fn caption(self) -> &'static str {
        match self.other() {
            Self::Outline => "Outline",
            Self::Chart => "Chart",
        }
    }
}

// The world checker's verdict on the open behavior, as the status line shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Status {
    Ok(String),
    Error(String),
}

impl Status {
    pub(crate) fn text(&self) -> &str {
        match self {
            Self::Ok(s) | Self::Error(s) => s,
        }
    }

    fn color(&self) -> [f32; 3] {
        match self {
            Self::Ok(_) => OK_LABEL,
            Self::Error(_) => theme::LOG_ERROR,
        }
    }
}

// The per-frame view the hook assembles.
pub(crate) struct BehaviorView<'a> {
    // The open behavior's name, its place among the world's behaviors, and how
    // many there are (zero shows the empty-world prompt instead of an outline).
    pub name: &'a str,
    pub index: usize,
    pub total: usize,
    // The open behavior as one outline, windowed by `scroll`.
    pub rows: &'a [Row],
    pub scroll: usize,
    pub selected: Option<usize>,
    // The same behavior laid out as a chart, and which view is showing.
    pub chart: &'a Chart,
    pub mode: ViewMode,
    pub pan: [f32; 2],
    // The palette for the selected row, shown while `picking`.
    pub picks: &'a [Pick],
    pub picking: bool,
    pub pick_scroll: usize,
    // Whether the selected row takes a typed value, and whether that field
    // asserts keyboard focus this frame.
    pub editable: bool,
    pub focus: bool,
    pub status: Option<&'a Status>,
    pub mouse: [f32; 2],
}

impl BehaviorView<'_> {
    // The palette rows shown this frame, after its scroll.
    fn picks_shown(&self) -> usize {
        self.picks
            .len()
            .saturating_sub(self.pick_scroll)
            .clamp(1, PICK_POOL)
    }
}

// A resolved Behavior-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BehaviorAction {
    // Open the previous / next behavior in the world.
    Step(i32),
    // Append a blank Behavior entry and open it.
    New,
    // Select outline row `i` (an absolute index).
    Select(usize),
    // Open (or close) the selected row's palette.
    Palette,
    // Insert palette option `i` (an absolute index).
    Choose(usize),
    // Close the open palette without picking.
    Dismiss,
    // Remove the selected list member.
    Delete,
    // Move the selected list member one place earlier / later.
    Move(i32),
    // Give the value field keyboard focus.
    FocusValue,
    // Switch between the chart and the outline.
    ToggleView,
    // Select the chart card at `i` (an index into the chart's cards).
    SelectCard(usize),
    // Press on empty canvas: start panning the chart.
    PanStart,
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Where the panel sits until the user drags it: left of centre below the top
// bar, clear of the Story and Import anchors.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [
        ((vw - BEHAVIOR_W) * 0.5 - 40.0).max(8.0),
        super::hud::body_top() + 24.0,
    ]
}

pub(crate) fn size() -> [f32; 2] {
    [BEHAVIOR_W, CHROME_H + ROW_POOL as f32 * ROW_H + PAD]
}

// The chart wants width where the outline wants rows: cards run left to right,
// so the panel opens wide enough to hold a few columns before any panning.
pub(crate) fn chart_size() -> [f32; 2] {
    [behavior_chart::default_width(), size()[1]]
}

// The tallest the panel resizes to: the injected row pool shown in full. Width
// is unbounded (capped only by the screen).
pub(crate) fn max_size() -> [f32; 2] {
    [
        f32::INFINITY,
        size()[1] + (ROW_POOL_MAX - ROW_POOL) as f32 * ROW_H,
    ]
}

fn header_y(o: [f32; 2]) -> f32 {
    widget::header_y(o, 4.0)
}

const CTRL_H: f32 = HEADER_H - 8.0;

pub(crate) fn prev_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0] + PAD, header_y(o), STEP_W, CTRL_H]
}

pub(crate) fn next_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0] + PAD + STEP_W + GAP, header_y(o), STEP_W, CTRL_H]
}

pub(crate) fn new_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    [o[0] + w - PAD - NEW_W, header_y(o), NEW_W, CTRL_H]
}

pub(crate) fn view_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    let n = new_rect(o, w);
    [n[0] - GAP - VIEW_W, n[1], VIEW_W, CTRL_H]
}

fn tool_y(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H + HEADER_H + 3.0
}

const TOOL_CTRL_H: f32 = TOOL_H - 6.0;

pub(crate) fn palette_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0] + PAD, tool_y(o), TOOL_BTN_W, TOOL_CTRL_H]
}

pub(crate) fn delete_rect(o: [f32; 2]) -> [f32; 4] {
    let p = palette_rect(o);
    [p[0] + TOOL_BTN_W + GAP, p[1], TOOL_BTN_W, TOOL_CTRL_H]
}

pub(crate) fn up_rect(o: [f32; 2]) -> [f32; 4] {
    let d = delete_rect(o);
    [d[0] + TOOL_BTN_W + GAP, d[1], ARROW_W, TOOL_CTRL_H]
}

pub(crate) fn down_rect(o: [f32; 2]) -> [f32; 4] {
    let u = up_rect(o);
    [u[0] + ARROW_W + GAP, u[1], ARROW_W, TOOL_CTRL_H]
}

pub(crate) fn value_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    let left = down_rect(o)[0] + ARROW_W + GAP;
    [
        left,
        tool_y(o),
        (o[0] + w - PAD - left).max(0.0),
        TOOL_CTRL_H,
    ]
}

fn body_top(o: [f32; 2]) -> f32 {
    o[1] + CHROME_H
}

// Visible outline row `slot` (0-based within the window) at effective width `w`.
pub(crate) fn row_rect(o: [f32; 2], w: f32, slot: usize) -> [f32; 4] {
    [
        o[0],
        body_top(o) + slot as f32 * ROW_H,
        w - SCROLLBAR_W - 2.0,
        ROW_H,
    ]
}

// The palette floats over the outline, anchored under the toolbar so it always
// stays inside the panel however far the body is scrolled.
fn palette_backing(o: [f32; 2], w: f32, shown: usize) -> [f32; 4] {
    // Rounded out to an outline-row boundary, so the palette's edge never
    // bisects the row beneath it and leaves it half drawn.
    let h = shown as f32 * PICK_ROW_H + 4.0;
    [
        o[0] + PAD,
        body_top(o),
        (w - 2.0 * PAD - SCROLLBAR_W).max(0.0),
        (h / ROW_H).ceil() * ROW_H,
    ]
}

pub(crate) fn pick_rect(o: [f32; 2], w: f32, slot: usize) -> [f32; 4] {
    let b = palette_backing(o, w, 1);
    [
        b[0] + 2.0,
        b[1] + 2.0 + slot as f32 * PICK_ROW_H,
        b[2] - 4.0,
        PICK_ROW_H,
    ]
}

// Whether the cursor is over the scrollable region (for wheel routing): the
// outline, or the palette while it is open.
pub(crate) fn cursor_over_body(mx: f32, my: f32, o: [f32; 2], s: [f32; 2]) -> bool {
    let p = widget::outer_rect(o, s);
    mx >= p[0] && mx < p[0] + p[2] && my >= body_top(o) && my < p[1] + p[3]
}

// Resolve a click at `(mx, my)` against the panel at origin `o`, size `s`.
// `None` means the click missed the panel. Title-bar presses never reach this:
// the shared routing intercepts them first.
pub(crate) fn hit_test(
    view: &BehaviorView,
    mx: f32,
    my: f32,
    o: [f32; 2],
    s: [f32; 2],
) -> Option<BehaviorAction> {
    let w = s[0];
    // An open palette is modal over the panel: its options pick, and anything
    // else on the panel closes it.
    if view.picking {
        for slot in 0..view.picks_shown() {
            if point_in(mx, my, pick_rect(o, w, slot)) {
                let i = view.pick_scroll + slot;
                return Some(if i < view.picks.len() {
                    BehaviorAction::Choose(i)
                } else {
                    BehaviorAction::Dismiss
                });
            }
        }
        return point_in(mx, my, widget::outer_rect(o, s)).then_some(BehaviorAction::Dismiss);
    }
    if point_in(mx, my, new_rect(o, w)) {
        return Some(BehaviorAction::New);
    }
    if view.total > 0 {
        if point_in(mx, my, view_rect(o, w)) {
            return Some(BehaviorAction::ToggleView);
        }
        if point_in(mx, my, prev_rect(o)) {
            return Some(BehaviorAction::Step(-1));
        }
        if point_in(mx, my, next_rect(o)) {
            return Some(BehaviorAction::Step(1));
        }
        for (rect, action) in [
            (palette_rect(o), BehaviorAction::Palette),
            (delete_rect(o), BehaviorAction::Delete),
            (up_rect(o), BehaviorAction::Move(-1)),
            (down_rect(o), BehaviorAction::Move(1)),
        ] {
            if point_in(mx, my, rect) {
                return Some(action);
            }
        }
        if view.editable && point_in(mx, my, value_rect(o, w)) {
            return Some(BehaviorAction::FocusValue);
        }
        if view.mode == ViewMode::Chart {
            let band = chart_band(o, s);
            let cv = chart_view(view);
            return Some(match behavior_chart::hit_card(&cv, mx, my, band) {
                Some(card) => BehaviorAction::SelectCard(card),
                None if point_in(mx, my, band) => BehaviorAction::PanStart,
                None => BehaviorAction::Consume,
            });
        }
        for slot in 0..visible_rows(s[1]) {
            if !point_in(mx, my, row_rect(o, w, slot)) {
                continue;
            }
            let i = view.scroll + slot;
            return Some(if i < view.rows.len() {
                BehaviorAction::Select(i)
            } else {
                BehaviorAction::Consume
            });
        }
    }
    point_in(mx, my, widget::outer_rect(o, s)).then_some(BehaviorAction::Consume)
}

// Position + show the panel (`Some(view)`) at effective size `s`, or blank every
// element (`None`).
pub(crate) fn apply(world: &mut World, view: Option<&BehaviorView>, o: [f32; 2], s: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };
    let w = s[0];
    widget::place_panel(world, PANEL_BG, widget::outer_rect(o, s));
    let title = widget::title_rect(o, w);
    widget::place_heading(world, TITLE_LABEL, title, "Behavior");
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);

    layout_header(world, view, o, w);
    layout_toolbar(world, view, o, w);
    layout_status(world, view, o, w);
    if view.mode == ViewMode::Chart && view.total > 0 {
        layout_rows(world, view, o, w, 0);
        widget::set_sprite_visible(world, LIST_TRACK, false);
        widget::set_sprite_visible(world, LIST_THUMB, false);
        behavior_chart::apply(world, &chart_view(view), chart_band(o, s));
    } else {
        behavior_chart::hide_all(world);
        layout_rows(world, view, o, w, visible_rows(s[1]));
        layout_scrollbar(world, view, o, w, visible_rows(s[1]));
    }
    layout_palette(world, view, o, w);
}

pub(crate) fn chart_band(o: [f32; 2], s: [f32; 2]) -> [f32; 4] {
    behavior_chart::band(o, s, body_top(o))
}

// The chart canvas's size at panel size `s`, which is all the pan maths needs.
pub(crate) fn chart_canvas(s: [f32; 2]) -> [f32; 2] {
    let b = chart_band([0.0, 0.0], s);
    [b[2], b[3]]
}

fn chart_view<'a>(view: &'a BehaviorView<'a>) -> behavior_chart::ChartView<'a> {
    behavior_chart::ChartView {
        chart: view.chart,
        selected: view
            .selected
            .and_then(|i| view.rows.get(i))
            .map(|r| &r.path),
        pan: view.pan,
        mouse: view.mouse,
    }
}

fn palette_open(view: &BehaviorView) -> bool {
    view.picking && !view.picks.is_empty()
}

fn layout_header(world: &mut World, view: &BehaviorView, o: [f32; 2], w: f32) {
    let has_any = view.total > 0;
    let new = new_rect(o, w);
    for chip in [
        Chip {
            bg: PREV_BG,
            label: PREV_LABEL,
            rect: prev_rect(o),
            caption: "<",
            tint: BTN_TINT,
            live: has_any,
        },
        Chip {
            bg: NEXT_BG,
            label: NEXT_LABEL,
            rect: next_rect(o),
            caption: ">",
            tint: BTN_TINT,
            live: has_any,
        },
        Chip {
            bg: VIEW_BG,
            label: VIEW_LABEL,
            rect: view_rect(o, w),
            caption: view.mode.caption(),
            tint: BTN_TINT,
            live: has_any,
        },
        Chip {
            bg: NEW_BG,
            label: NEW_LABEL,
            rect: new,
            caption: "New",
            tint: NEW_TINT,
            live: true,
        },
    ] {
        place_chip(world, &chip, view.mouse);
    }
    let left = next_rect(o)[0] + STEP_W + GAP + 4.0;
    let right = view_rect(o, w)[0];
    let caption = if has_any {
        format!("{} ({}/{})", view.name, view.index + 1, view.total)
    } else {
        "no behaviors yet -- press New".to_string()
    };
    let color = if has_any { theme::LABEL } else { HINT_LABEL };
    widget::place_left_label(
        world,
        NAME_LABEL,
        [left, header_y(o) + CTRL_H * 0.5 - theme::TEXT_HALF],
        &widget::clip_text(&caption, ((right - left) / CHAR_W) as usize),
        color,
        true,
    );
}

fn layout_toolbar(world: &mut World, view: &BehaviorView, o: [f32; 2], w: f32) {
    let on = view.total > 0;
    let selected = on && view.selected.is_some();
    let member = selected
        && view
            .selected
            .and_then(|i| view.rows.get(i))
            .is_some_and(|r| r.element.is_some());
    for chip in [
        Chip {
            bg: PICK_BG,
            label: PICK_LABEL,
            rect: palette_rect(o),
            caption: "Pick",
            tint: BTN_TINT,
            live: selected && !view.picks.is_empty(),
        },
        Chip {
            bg: DEL_BG,
            label: DEL_LABEL,
            rect: delete_rect(o),
            caption: "Del",
            tint: DEL_TINT,
            live: member,
        },
        Chip {
            bg: UP_BG,
            label: UP_LABEL,
            rect: up_rect(o),
            caption: "^",
            tint: BTN_TINT,
            live: member,
        },
        Chip {
            bg: DOWN_BG,
            label: DOWN_LABEL,
            rect: down_rect(o),
            caption: "v",
            tint: BTN_TINT,
            live: member,
        },
    ] {
        place_chip(world, &chip, view.mouse);
    }
    if on && view.editable {
        widget::show_field(world, VALUE_INPUT, value_rect(o, w), view.focus);
    } else {
        widget::hide_field(world, VALUE_INPUT);
    }
}

fn layout_status(world: &mut World, view: &BehaviorView, o: [f32; 2], w: f32) {
    widget::place_message(
        world,
        STATUS_LABEL,
        status_rect(o, w),
        view.status.map(Status::text).unwrap_or(""),
        view.status.map_or(HINT_LABEL, Status::color),
        view.status.is_some(),
    );
}

fn status_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    [
        o[0] + PAD,
        o[1] + widget::TITLE_H + HEADER_H + TOOL_H + 2.0,
        (w - 2.0 * PAD).max(0.0),
        STATUS_H - 4.0,
    ]
}

fn layout_rows(world: &mut World, view: &BehaviorView, o: [f32; 2], w: f32, shown: usize) {
    let value_x = value_column(w);
    let label_budget = ((value_x - PAD * 2.0) / CHAR_W) as usize;
    for slot in 0..ROW_POOL_MAX {
        let i = view.scroll + slot;
        let drawn = slot < shown && view.total > 0;
        let Some(row) = view.rows.get(i).filter(|_| drawn) else {
            widget::set_sprite_visible(world, row_bg(slot), false);
            widget::set_label_visible(world, row_label(slot), false);
            widget::set_label_visible(world, row_value(slot), false);
            continue;
        };
        let r = row_rect(o, w, slot);
        let hovered = point_in(view.mouse[0], view.mouse[1], r);
        let tint = if view.selected == Some(i) {
            theme::SELECTED_TINT
        } else if hovered {
            theme::HOVER_TINT
        } else {
            ROW_TINT
        };
        place_rounded(
            world,
            row_bg(slot),
            theme::highlight_rect(r),
            tint,
            theme::CONTROL_RADIUS,
            true,
        );
        let text_y = r[1] + ROW_H * 0.5 - theme::TEXT_HALF;
        widget::place_left_label(
            world,
            row_label(slot),
            [r[0] + PAD + row.depth as f32 * INDENT, text_y],
            &widget::clip_text(&row.label, label_budget.max(6)),
            label_color(&row.kind),
            true,
        );
        widget::place_left_label(
            world,
            row_value(slot),
            [r[0] + value_x, text_y],
            &widget::clip_text(&row.value, max_value_chars(w)),
            theme::LABEL,
            true,
        );
    }
}

// Structure reads dimmer than the things that carry a value, so the shape of a
// body is legible before any of its detail is.
fn label_color(kind: &Kind) -> [f32; 3] {
    match kind {
        Kind::Node | Kind::Source => theme::HEADING,
        Kind::List(_) => HINT_LABEL,
        _ => theme::LABEL,
    }
}

fn layout_scrollbar(world: &mut World, view: &BehaviorView, o: [f32; 2], w: f32, shown: usize) {
    let total = if view.total > 0 { view.rows.len() } else { 0 };
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

// The floating palette: an opaque backing over the outline, one row per offered
// verb with its hint, and its own scrollbar once the vocabulary overflows.
fn layout_palette(world: &mut World, view: &BehaviorView, o: [f32; 2], w: f32) {
    if !palette_open(view) {
        widget::set_sprite_visible(world, DROP_BG, false);
        widget::set_sprite_visible(world, DROP_TRACK, false);
        widget::set_sprite_visible(world, DROP_THUMB, false);
        for slot in 0..PICK_POOL {
            widget::set_sprite_visible(world, pick_bg(slot), false);
            widget::set_label_visible(world, pick_label(slot), false);
            widget::set_label_visible(world, pick_hint(slot), false);
        }
        return;
    }
    let shown = view.picks_shown();
    place_rounded(
        world,
        DROP_BG,
        palette_backing(o, w, shown),
        DROP_TINT,
        theme::CONTROL_RADIUS,
        true,
    );
    for slot in 0..PICK_POOL {
        let i = view.pick_scroll + slot;
        let Some(pick) = view.picks.get(i).filter(|_| slot < shown) else {
            widget::set_sprite_visible(world, pick_bg(slot), false);
            widget::set_label_visible(world, pick_label(slot), false);
            widget::set_label_visible(world, pick_hint(slot), false);
            continue;
        };
        let r = pick_rect(o, w, slot);
        let hovered = point_in(view.mouse[0], view.mouse[1], r);
        place_rounded(
            world,
            pick_bg(slot),
            r,
            if hovered { theme::HOVER_TINT } else { ROW_TINT },
            theme::CONTROL_RADIUS,
            true,
        );
        let text_y = r[1] + PICK_ROW_H * 0.5 - theme::TEXT_HALF;
        widget::place_left_label(
            world,
            pick_label(slot),
            [r[0] + PAD, text_y],
            edit::pick_caption(pick),
            theme::LABEL,
            true,
        );
        widget::place_left_label(
            world,
            pick_hint(slot),
            [r[0] + value_column(w), text_y],
            &widget::clip_text(&pick.hint, max_value_chars(w)),
            HINT_LABEL,
            !pick.hint.is_empty(),
        );
    }
    layout_palette_scrollbar(world, view, o, w, shown);
}

fn layout_palette_scrollbar(
    world: &mut World,
    view: &BehaviorView,
    o: [f32; 2],
    w: f32,
    shown: usize,
) {
    let total = view.picks.len();
    if total <= shown {
        widget::set_sprite_visible(world, DROP_TRACK, false);
        widget::set_sprite_visible(world, DROP_THUMB, false);
        return;
    }
    let b = palette_backing(o, w, shown);
    let x = b[0] + b[2] - SCROLLBAR_W - 2.0;
    let h = shown as f32 * PICK_ROW_H;
    place_rounded(
        world,
        DROP_TRACK,
        [x, b[1] + 2.0, SCROLLBAR_W, h],
        TRACK_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
    let thumb_h = (h * shown as f32 / total as f32).max(18.0);
    let max_scroll = (total - shown) as f32;
    let off = (h - thumb_h) * (view.pick_scroll.min(total - shown) as f32 / max_scroll);
    place_rounded(
        world,
        DROP_THUMB,
        [x, b[1] + 2.0 + off, SCROLLBAR_W, thumb_h],
        THUMB_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
}

// One labelled chip of the header or the toolbar: its two element ids, where it
// sits, what it says, its idle tint, and whether it has anything to act on.
struct Chip {
    bg: AssetId,
    label: AssetId,
    rect: [f32; 4],
    caption: &'static str,
    tint: [f32; 4],
    live: bool,
}

// A control with nothing to act on is drawn dimmed rather than hidden, so the
// toolbar keeps its shape as the selection moves.
fn place_chip(world: &mut World, chip: &Chip, mouse: [f32; 2]) {
    let hovered = chip.live && point_in(mouse[0], mouse[1], chip.rect);
    let t = chip.tint;
    let tint = if hovered {
        theme::HOVER_TINT
    } else if chip.live {
        t
    } else {
        [t[0] * 0.5, t[1] * 0.5, t[2] * 0.5, 0.7]
    };
    place_rounded(world, chip.bg, chip.rect, tint, theme::CONTROL_RADIUS, true);
    if let Some(l) = widget::label_mut(world, chip.label) {
        l.x = chip.rect[0] + chip.rect[2] * 0.5;
        l.y = chip.rect[1] + chip.rect[3] * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Center;
        l.color = if chip.live {
            [1.0, 1.0, 1.0]
        } else {
            HINT_LABEL
        };
        l.visible = true;
        l.content = chip.caption.to_string();
    }
}

// Hide every panel element, blurring the value field so a hidden field cannot
// keep keyboard focus.
pub(crate) fn hide_all(world: &mut World) {
    for id in all_sprite_ids() {
        widget::set_sprite_visible(world, id, false);
    }
    for id in all_label_ids() {
        widget::set_label_visible(world, id, false);
    }
    widget::hide_field(world, VALUE_INPUT);
}

// Every panel sprite id, in draw (insertion) order: chrome, then the outline
// rows, then the scrollbar and the palette floating above them.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![
        PANEL_BG, CLOSE_BG, PREV_BG, NEXT_BG, VIEW_BG, NEW_BG, PICK_BG, DEL_BG, UP_BG, DOWN_BG,
    ];
    ids.extend((0..ROW_POOL_MAX).map(row_bg));
    ids.extend(behavior_chart::all_sprite_ids());
    ids.extend([LIST_TRACK, LIST_THUMB, DROP_BG]);
    ids.extend((0..PICK_POOL).map(pick_bg));
    ids.extend([DROP_TRACK, DROP_THUMB]);
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![
        TITLE_LABEL,
        CLOSE_LABEL,
        PREV_LABEL,
        NEXT_LABEL,
        NAME_LABEL,
        VIEW_LABEL,
        NEW_LABEL,
        PICK_LABEL,
        DEL_LABEL,
        UP_LABEL,
        DOWN_LABEL,
        STATUS_LABEL,
    ];
    ids.extend((0..ROW_POOL_MAX).map(row_label));
    ids.extend((0..ROW_POOL_MAX).map(row_value));
    ids.extend(behavior_chart::all_label_ids());
    ids.extend((0..PICK_POOL).map(pick_label));
    ids.extend((0..PICK_POOL).map(pick_hint));
    ids
}

// The palette's own elements, which draw above the rest of the panel while it
// is open so its backing occludes whatever it floats over.
pub(crate) fn palette_ids() -> Vec<AssetId> {
    let mut ids = vec![DROP_BG, DROP_TRACK, DROP_THUMB];
    ids.extend((0..PICK_POOL).map(pick_bg));
    ids.extend((0..PICK_POOL).map(pick_label));
    ids.extend((0..PICK_POOL).map(pick_hint));
    ids
}

pub(crate) fn all_field_ids() -> Vec<AssetId> {
    vec![VALUE_INPUT]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextInput, TextLabel};
    use crate::editor::behavior::outline;

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

    fn sample_chart() -> &'static Chart {
        static CHART: std::sync::OnceLock<Chart> = std::sync::OnceLock::new();
        CHART.get_or_init(|| crate::editor::behavior::graph::chart(&sample_args()))
    }

    fn sample_args() -> serde_json::Value {
        serde_json::json!({"on": "tick", "scope": ["Prop"],
            "do": [{"hide": {"target": "self"}}]})
    }

    fn sample_rows() -> Vec<Row> {
        outline::rows(&sample_args())
    }

    fn view<'a>(rows: &'a [Row], picks: &'a [Pick]) -> BehaviorView<'a> {
        BehaviorView {
            name: "chase",
            index: 0,
            total: 1,
            rows,
            scroll: 0,
            selected: None,
            chart: sample_chart(),
            mode: ViewMode::Outline,
            pan: [0.0, 0.0],
            picks,
            picking: false,
            pick_scroll: 0,
            editable: false,
            focus: false,
            status: None,
            mouse: [0.0, 0.0],
        }
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

    #[test]
    fn header_and_toolbar_pack_without_overlap() {
        let o = [40.0, 40.0];
        let (p, n, new) = (prev_rect(o), next_rect(o), new_rect(o, BEHAVIOR_W));
        assert!(p[0] + p[2] < n[0], "the step buttons do not overlap");
        assert_eq!(new[0] + new[2], o[0] + BEHAVIOR_W - PAD, "New pinned right");
        let (pick, del, up, down, value) = (
            palette_rect(o),
            delete_rect(o),
            up_rect(o),
            down_rect(o),
            value_rect(o, BEHAVIOR_W),
        );
        let mut x = pick[0];
        for r in [del, up, down, value] {
            assert!(x < r[0], "the toolbar packs left to right");
            x = r[0];
        }
        assert_eq!(
            value[0] + value[2],
            o[0] + BEHAVIOR_W - PAD,
            "the value field fills the rest of the row"
        );
        assert!(value[2] > 180.0, "and stays usable: {}", value[2]);
        // The toolbar sits clear of the outline below it.
        assert!(pick[1] + pick[3] < row_rect(o, BEHAVIOR_W, 0)[1]);
    }

    #[test]
    fn hit_test_resolves_the_header_toolbar_and_rows() {
        let rows = sample_rows();
        let v = view(&rows, &[]);
        let o = [40.0, 40.0];
        let s = size();
        let hit = |r: [f32; 4]| hit_test(&v, r[0] + 3.0, r[1] + 3.0, o, s);
        assert_eq!(hit(prev_rect(o)), Some(BehaviorAction::Step(-1)));
        assert_eq!(hit(next_rect(o)), Some(BehaviorAction::Step(1)));
        assert_eq!(hit(new_rect(o, BEHAVIOR_W)), Some(BehaviorAction::New));
        assert_eq!(hit(palette_rect(o)), Some(BehaviorAction::Palette));
        assert_eq!(hit(delete_rect(o)), Some(BehaviorAction::Delete));
        assert_eq!(hit(up_rect(o)), Some(BehaviorAction::Move(-1)));
        assert_eq!(hit(down_rect(o)), Some(BehaviorAction::Move(1)));
        assert_eq!(
            hit(row_rect(o, BEHAVIOR_W, 2)),
            Some(BehaviorAction::Select(2))
        );
        // Past the last row the click is swallowed, and off-panel misses.
        let past = row_rect(o, BEHAVIOR_W, rows.len());
        assert_eq!(hit(past), Some(BehaviorAction::Consume));
        assert_eq!(hit_test(&v, 5000.0, 5000.0, o, s), None);
    }

    #[test]
    fn the_value_field_answers_clicks_only_when_the_row_takes_one() {
        let rows = sample_rows();
        let o = [40.0, 40.0];
        let s = size();
        let r = value_rect(o, BEHAVIOR_W);
        let idle = view(&rows, &[]);
        assert_eq!(
            hit_test(&idle, r[0] + 3.0, r[1] + 3.0, o, s),
            Some(BehaviorAction::Consume)
        );
        let typing = BehaviorView {
            editable: true,
            ..view(&rows, &[])
        };
        assert_eq!(
            hit_test(&typing, r[0] + 3.0, r[1] + 3.0, o, s),
            Some(BehaviorAction::FocusValue)
        );
    }

    // While the palette is open it is modal: its options pick and every other
    // click on the panel closes it, so a stray press cannot edit behind it.
    #[test]
    fn an_open_palette_is_modal_over_the_panel() {
        let rows = sample_rows();
        let picks = edit::picks(&Kind::List(outline::List::Nodes), &[]);
        let v = BehaviorView {
            picking: true,
            selected: Some(0),
            ..view(&rows, &picks)
        };
        let o = [40.0, 40.0];
        let s = size();
        let r = pick_rect(o, BEHAVIOR_W, 2);
        assert_eq!(
            hit_test(&v, r[0] + 3.0, r[1] + 3.0, o, s),
            Some(BehaviorAction::Choose(2))
        );
        let n = new_rect(o, BEHAVIOR_W);
        assert_eq!(
            hit_test(&v, n[0] + 3.0, n[1] + 3.0, o, s),
            Some(BehaviorAction::Dismiss),
            "a click elsewhere closes rather than acting"
        );
        // A scrolled palette maps its slots to absolute options.
        let scrolled = BehaviorView {
            pick_scroll: 5,
            ..v
        };
        assert_eq!(
            hit_test(&scrolled, r[0] + 3.0, r[1] + 3.0, o, s),
            Some(BehaviorAction::Choose(7))
        );
    }

    #[test]
    fn an_empty_world_offers_only_new() {
        let rows: Vec<Row> = Vec::new();
        let v = BehaviorView {
            total: 0,
            ..view(&rows, &[])
        };
        let o = [40.0, 40.0];
        let s = size();
        let p = prev_rect(o);
        assert_eq!(
            hit_test(&v, p[0] + 3.0, p[1] + 3.0, o, s),
            Some(BehaviorAction::Consume),
            "with nothing open the step buttons do nothing"
        );
        let n = new_rect(o, BEHAVIOR_W);
        assert_eq!(
            hit_test(&v, n[0] + 3.0, n[1] + 3.0, o, s),
            Some(BehaviorAction::New)
        );
        let mut world = injected_world();
        apply(&mut world, Some(&v), o, s);
        assert_eq!(
            label(&world, NAME_LABEL).content,
            "no behaviors yet -- press New"
        );
        assert!(!sprite(&world, row_bg(0)).visible, "no outline to draw");
    }

    #[test]
    fn apply_draws_the_outline_indented_with_a_value_column() {
        let mut world = injected_world();
        let rows = sample_rows();
        let v = BehaviorView {
            selected: Some(0),
            ..view(&rows, &[])
        };
        let o = [20.0, 20.0];
        apply(&mut world, Some(&v), o, size());
        assert_eq!(label(&world, TITLE_LABEL).content, "Behavior");
        assert_eq!(label(&world, NAME_LABEL).content, "chase (1/1)");
        let first = label(&world, row_label(0));
        assert!(first.visible && first.content == "on");
        assert_eq!(label(&world, row_value(0)).content, "tick");
        // The selected row is highlighted without a hover.
        assert_eq!(sprite(&world, row_bg(0)).tint, theme::SELECTED_TINT);
        // Nested rows inset, and every value shares one column.
        let scope_slot = rows.iter().position(|r| r.label == "component").unwrap();
        assert!(label(&world, row_label(scope_slot)).x > first.x);
        assert_eq!(
            label(&world, row_value(scope_slot)).x,
            label(&world, row_value(0)).x
        );
    }

    #[test]
    fn the_palette_draws_over_the_outline_and_stays_inside_the_panel() {
        let mut world = injected_world();
        let rows = sample_rows();
        let picks = edit::picks(&Kind::List(outline::List::Nodes), &[]);
        let v = BehaviorView {
            picking: true,
            selected: Some(0),
            ..view(&rows, &picks)
        };
        let o = [20.0, 20.0];
        let s = size();
        apply(&mut world, Some(&v), o, s);
        let backing = sprite(&world, DROP_BG);
        assert!(backing.visible);
        assert!(
            backing.y + backing.height <= o[1] + s[1],
            "the palette never spills past the panel"
        );
        assert!(label(&world, pick_label(0)).visible);
        assert_eq!(label(&world, pick_label(0)).content, "if");
        assert!(
            sprite(&world, DROP_THUMB).visible,
            "the node vocabulary overflows the palette window"
        );
        // Closing it blanks every option row.
        apply(&mut world, Some(&view(&rows, &picks)), o, s);
        assert!(!sprite(&world, DROP_BG).visible);
        assert!(!label(&world, pick_label(0)).visible);
    }

    // A panel's elements draw in injection order, so the palette's backing
    // cannot occlude what it floats over: the rows it covers are blanked, and
    // the ones below it keep drawing as context.
    // The palette floats over the outline on its own draw layer (see
    // `overlay_ids`), so the rows it covers keep drawing and its opaque backing
    // hides them. They must NOT be blanked: blanking is what the layer band
    // replaced, and it left the chart's cards showing through.
    #[test]
    fn the_palette_leaves_the_rows_it_covers_drawn() {
        let mut world = injected_world();
        let args = serde_json::json!({"do": (0..12).map(|_| serde_json::json!({"save": null}))
            .collect::<Vec<_>>()});
        let rows = outline::rows(&args);
        let picks = edit::picks(&Kind::List(outline::List::Nodes), &[]);
        let o = [20.0, 20.0];
        let s = size();
        let open = BehaviorView {
            picking: true,
            selected: Some(0),
            ..view(&rows, &picks)
        };
        apply(&mut world, Some(&open), o, s);
        assert!(sprite(&world, DROP_BG).visible, "the palette backing draws");
        for slot in 0..visible_rows(s[1]).min(rows.len()) {
            assert!(
                label(&world, row_label(slot)).visible,
                "row {slot} was blanked instead of being drawn under the palette"
            );
        }
        // Every element the palette draws is declared as an overlay, or the
        // layer bump would miss it and it would sink back into the panel.
        let declared = palette_ids();
        for id in [DROP_BG, DROP_TRACK, DROP_THUMB] {
            assert!(declared.contains(&id), "{id:?} is not declared an overlay");
        }
        for slot in 0..PICK_POOL {
            for id in [pick_bg(slot), pick_label(slot), pick_hint(slot)] {
                assert!(declared.contains(&id), "{id:?} is not declared an overlay");
            }
        }
    }

    // Same contract as the form's dropdown: the palette's edge lands between
    // outline rows, never through one.
    #[test]
    fn the_palette_backing_ends_on_an_outline_row_boundary() {
        let o = [20.0, 20.0];
        for shown in 1..=PICK_POOL {
            let b = palette_backing(o, BEHAVIOR_W, shown);
            let rows = b[3] / ROW_H;
            assert!(
                (rows - rows.round()).abs() < 0.01,
                "shown {shown}: height {} is {rows} rows",
                b[3],
            );
            assert!(b[3] >= shown as f32 * PICK_ROW_H, "{b:?}");
        }
    }

    #[test]
    fn status_colors_the_checker_verdict() {
        let mut world = injected_world();
        let rows = sample_rows();
        let error = Status::Error("Behavior 'x': reads unbound name 'q'".to_string());
        let v = BehaviorView {
            status: Some(&error),
            ..view(&rows, &[])
        };
        apply(&mut world, Some(&v), [20.0, 20.0], size());
        let status = label(&world, STATUS_LABEL);
        assert!(status.visible && status.color == theme::LOG_ERROR);
        let ok = Status::Ok("checks out".to_string());
        let v = BehaviorView {
            status: Some(&ok),
            ..view(&rows, &[])
        };
        apply(&mut world, Some(&v), [20.0, 20.0], size());
        assert_eq!(label(&world, STATUS_LABEL).color, OK_LABEL);
    }

    #[test]
    fn taller_size_reveals_more_rows_up_to_the_pool() {
        assert_eq!(visible_rows(size()[1]), ROW_POOL);
        assert_eq!(visible_rows(size()[1] + 5.0 * ROW_H), ROW_POOL + 5);
        assert_eq!(visible_rows(max_size()[1]), ROW_POOL_MAX);
        assert_eq!(visible_rows(10_000.0), ROW_POOL_MAX, "capped at the pool");
    }

    #[test]
    fn a_long_outline_scrolls_and_maps_slots_to_absolute_rows() {
        let mut world = injected_world();
        let args = serde_json::json!({"do": (0..40).map(|_| serde_json::json!({"save": null}))
            .collect::<Vec<_>>()});
        let rows = outline::rows(&args);
        let v = BehaviorView {
            scroll: 10,
            ..view(&rows, &[])
        };
        let o = [20.0, 20.0];
        apply(&mut world, Some(&v), o, size());
        assert!(sprite(&world, LIST_THUMB).visible);
        let r0 = row_rect(o, BEHAVIOR_W, 0);
        assert_eq!(
            hit_test(&v, r0[0] + 3.0, r0[1] + 3.0, o, size()),
            Some(BehaviorAction::Select(10))
        );
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        let rows = sample_rows();
        let picks = edit::picks(&Kind::Node, &[]);
        let v = BehaviorView {
            picking: true,
            editable: true,
            focus: true,
            selected: Some(0),
            ..view(&rows, &picks)
        };
        apply(&mut world, Some(&v), [20.0, 20.0], size());
        apply(&mut world, None, [0.0, 0.0], size());
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
        assert!(world.query::<TextInput>().all(|t| !t.visible && !t.focused));
    }
}

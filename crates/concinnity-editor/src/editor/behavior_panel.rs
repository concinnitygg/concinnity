// src/editor/behavior_panel.rs
//
// The Behavior panel's layout half: a floating node-graph editor over one
// `Behavior` asset. The header cycles through the world's behaviors and adds
// new ones; the toolbar acts on the selected row (open its palette, delete it,
// move it, or type its value). The body is the asset either way: the outline
// lists every field of every node (`behavior/outline.rs`), while the chart
// draws the control flow as cards with the selected node's own settings in the
// inspector column beside it (`behavior/fields.rs`).
//
// The palette opens as a floating list over the body, so picking a node or an
// expression never leaves the panel, and the checker's complaint floats over
// the body's foot rather than sitting in the chrome -- a behavior that checks
// out says nothing, and the body never moves under the user either way.
// `hook/behavior_edit.rs` owns the actions.

use super::behavior::edit::{self, Pick};
use super::behavior::fields;
use super::behavior::graph::{CardKind, Chart};
use super::behavior::outline::{Kind, Row};
use super::behavior::path::{Path, Step};
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
pub(crate) const STATUS_BG: AssetId = AssetId(BASE + 28);
pub(crate) const INSPECT_BG: AssetId = AssetId(BASE + 29);
pub(crate) const INSPECT_LABEL: AssetId = AssetId(BASE + 30);
pub(crate) const REMOVE_BG: AssetId = AssetId(BASE + 31);
pub(crate) const REMOVE_LABEL: AssetId = AssetId(BASE + 32);
pub(crate) const NAME_INPUT: AssetId = AssetId(BASE + 33);
pub(crate) const FILTER_INPUT: AssetId = AssetId(BASE + 34);
pub(crate) const DROP_FILTER_BG: AssetId = AssetId(BASE + 35);
pub(crate) const DUP_BG: AssetId = AssetId(BASE + 36);
pub(crate) const DUP_LABEL: AssetId = AssetId(BASE + 37);

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
// The error banner floats over the foot of the body rather than sitting in the
// chrome, so the body keeps the same geometry whether or not the open behavior
// checks out. Two lines, because a checker message is a sentence.
const STATUS_H: f32 = 2.0 * widget::LINE_H + 6.0;
const ROW_H: f32 = 22.0;
const SCROLLBAR_W: f32 = 5.0;
const STEP_W: f32 = 26.0;
const NEW_W: f32 = 58.0;
const VIEW_W: f32 = 62.0;
const REMOVE_W: f32 = 70.0;
// The name field, and the ordinal caption that follows it. The field is capped
// rather than filling, so the chart view's much wider header does not stretch a
// name across half the panel.
const NAME_W: f32 = 220.0;
const ORDINAL_W: f32 = 62.0;
const TOOL_BTN_W: f32 = 52.0;
const ARROW_W: f32 = 30.0;
// Visible outline rows at the default height; a longer outline scrolls.
// Resizing the panel taller reveals more, up to the injected pool.
pub(crate) const ROW_POOL: usize = 16;
pub(crate) const ROW_POOL_MAX: usize = 34;
// Visible palette options before the palette scrolls.
pub(crate) const PICK_POOL: usize = 10;
const PICK_ROW_H: f32 = 24.0;
// The palette's own filter field, above its options.
const FILTER_H: f32 = 26.0;
// Value-column caption budget at the default width; widening the panel grows
// it. A row's label is budgeted by the column it stops at instead.
const MAX_VALUE_CHARS: usize = 34;
// Approximate body-font advance, for growing the caption budgets with the width.
pub(crate) const CHAR_W: f32 = 8.5;
// How far each outline level insets its caption.
const INDENT: f32 = 14.0;

// The inspector: the column beside the chart listing the selected node's own
// settings, so a card is edited where it sits rather than by leaving for the
// outline. Its rows are narrow, so its value column is its own.
const INSPECT_W: f32 = 240.0;
const INSPECT_HEAD_H: f32 = 22.0;
const INSPECT_VALUE_X: f32 = 106.0;
const INSPECT_TINT: [f32; 4] = [0.10, 0.10, 0.13, 1.0];

// The chrome height above the outline (everything but the row band and pad).
const CHROME_H: f32 = widget::TITLE_H + HEADER_H + TOOL_H;

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
// The removal chip once it is armed: hotter than its idle tint, so a chip
// waiting on a second press does not read like one that has done nothing.
const CONFIRM_TINT: [f32; 4] = [0.68, 0.26, 0.28, 1.0];
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
const DROP_TINT: [f32; 4] = [0.09, 0.09, 0.12, 1.0];
const FILTER_TINT: [f32; 4] = [0.14, 0.14, 0.18, 1.0];
const ERROR_TINT: [f32; 4] = [0.20, 0.09, 0.11, 1.0];
const ERROR_BORDER: [f32; 4] = [0.52, 0.24, 0.27, 1.0];
const HINT_LABEL: [f32; 3] = theme::LABEL_DIM;

// Which way the panel shows its work. The outline and the chart are two views
// of the open behavior -- every field of every node, or the control flow as
// cards -- and the overview is the world's behaviors and how they reach each
// other (`behavior/relations.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ViewMode {
    #[default]
    Outline,
    Chart,
    Overview,
}

impl ViewMode {
    // The views cycle, so one button reaches all three.
    pub(crate) fn other(self) -> ViewMode {
        match self {
            Self::Outline => Self::Chart,
            Self::Chart => Self::Overview,
            Self::Overview => Self::Outline,
        }
    }

    // Whether this view draws a chart rather than a list of rows.
    pub(crate) fn drawn_as_chart(self) -> bool {
        !matches!(self, Self::Outline)
    }

    // The toggle is captioned with the view it switches to, so the button says
    // what pressing it does rather than what is already on screen.
    fn caption(self) -> &'static str {
        match self.other() {
            Self::Outline => "Outline",
            Self::Chart => "Chart",
            Self::Overview => "Overview",
        }
    }
}

// The world checker's verdict on the open behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Status {
    Ok,
    Error {
        message: String,
        // Where in the args the complaint is about, as the checker addressed it.
        // Empty when nothing narrower than the asset is to blame. Kept as the
        // path rather than a row, so it still resolves after an edit moves rows
        // around under it.
        at: Path,
    },
}

impl Status {
    // A complaint with nowhere in particular to point.
    pub(crate) fn message(text: impl Into<String>) -> Status {
        Status::Error {
            message: text.into(),
            at: Path::new(),
        }
    }

    // What the panel shows. A behavior that checks out says nothing, so the
    // banner appears only when something is wrong with it.
    pub(crate) fn error(&self) -> Option<&str> {
        match self {
            Self::Ok => None,
            Self::Error { message, .. } => Some(message),
        }
    }

    pub(crate) fn at(&self) -> &[Step] {
        match self {
            Self::Ok => &[],
            Self::Error { at, .. } => at,
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
    // The same behavior laid out as a chart, the world's behaviors laid out as
    // one map, and which view is showing.
    pub chart: &'a Chart,
    pub overview: &'a Chart,
    pub mode: ViewMode,
    pub pan: [f32; 2],
    // The card the selection belongs to, and the rows the inspector lists for
    // it. Empty until something on the chart is selected.
    pub card: Option<usize>,
    pub fields: &'a [usize],
    // The palette for the selected row, shown while `picking`, windowed by
    // `pick_scroll` with `pick` the option the keyboard is on.
    pub picks: &'a [Pick],
    // The options the typed filter keeps, best first, as indices into `picks`.
    // Every slot the palette draws and hit-tests goes through these, so `pick`
    // and the scroll count filtered places rather than absolute ones.
    pub matches: &'a [usize],
    pub picking: bool,
    pub pick_scroll: usize,
    pub pick: usize,
    // Whether the filter field asserts keyboard focus this frame.
    pub filter_focus: bool,
    // The overview's selected card, which is its own: the map's cards stand for
    // whole behaviors rather than for rows of the open one.
    pub overview_card: Option<usize>,
    // Whether the selected row takes a typed value, and whether that field
    // asserts keyboard focus this frame.
    pub editable: bool,
    pub focus: bool,
    // Whether the name field asserts keyboard focus this frame, and whether the
    // removal chip is waiting on the press that carries it out.
    pub name_focus: bool,
    pub remove_armed: bool,
    pub status: Option<&'a Status>,
    // The row the checker's complaint is about, resolved against the rows on
    // screen. Marked wherever it shows rather than jumped to, so a check running
    // after every edit never moves the body under the author.
    pub fault_row: Option<usize>,
    // Live-debug marks: executed cards / rows with each pulse's remaining
    // strength, and the cards holding a breakpoint. Empty outside a session.
    pub pulse_cards: &'a [(usize, f32)],
    pub pulse_rows: &'a [(usize, f32)],
    pub break_cards: &'a [usize],
    pub mouse: [f32; 2],
}

impl BehaviorView<'_> {
    // The palette rows shown this frame, after its scroll.
    fn picks_shown(&self) -> usize {
        self.matches
            .len()
            .saturating_sub(self.pick_scroll)
            .clamp(1, PICK_POOL)
    }

    // The option drawn in slot `slot`, as an index into `picks`.
    fn pick_at(&self, slot: usize) -> Option<usize> {
        self.matches.get(self.pick_scroll + slot).copied()
    }

    // The chart this view draws.
    pub(crate) fn shown_chart(&self) -> &Chart {
        match self.mode {
            ViewMode::Overview => self.overview,
            _ => self.chart,
        }
    }

    // The row body slot `slot` draws: the outline windowed by its scroll, or the
    // inspector's list of the selected node's fields.
    fn row_at(&self, slot: usize) -> Option<usize> {
        match self.mode {
            ViewMode::Chart => self.fields.get(slot).copied(),
            ViewMode::Overview => None,
            ViewMode::Outline => Some(self.scroll + slot),
        }
    }

    // The depth body rows indent from: the outline's own root, or the selected
    // node in the inspector.
    fn base_depth(&self) -> usize {
        match self.mode {
            ViewMode::Chart => self
                .fields
                .first()
                .and_then(|&i| self.rows.get(i))
                .map_or(0, |r| r.depth),
            _ => 0,
        }
    }
}

// A resolved Behavior-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BehaviorAction {
    // Open the previous / next behavior in the world.
    Step(i32),
    // Append a blank Behavior entry and open it.
    New,
    // Arm the open behavior's removal, or carry out an armed one.
    Remove,
    // Give the name field keyboard focus.
    FocusName,
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
    // Open the behavior the overview card at `i` stands for.
    OpenCard(usize),
    // Open the world's variable table on the variable card at `i`.
    OpenVariable(usize),
    // Press on empty canvas: start panning the chart.
    PanStart,
    // Select whatever the checker is complaining about.
    GoToFault,
    // Hold the selected list member for a later paste.
    Copy,
    // Put what is held into the list the selection addresses.
    Paste,
    // Put a copy of the selected member beside it, leaving what is held alone.
    Duplicate,
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
// so the panel opens wide enough to hold a trigger, a node, and the card that
// appends after it beside the inspector. A panel cannot size itself against the
// viewport (`Panel::size` never sees one, unlike `default_origin`), so the
// default has to fit a small window outright rather than be trimmed to it.
pub(crate) fn chart_size() -> [f32; 2] {
    [behavior_chart::width_for(3) + INSPECT_W, size()[1]]
}

// The overview has no inspector, so its columns take the whole width.
pub(crate) fn overview_size() -> [f32; 2] {
    [behavior_chart::width_for(4), size()[1]]
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

// Removing the open behavior sits beside adding one, since both act on the
// world's list rather than on anything inside a body. The toolbar's Del is a
// different action on a different subject: it takes out one node.
pub(crate) fn remove_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    let n = new_rect(o, w);
    [n[0] - GAP - REMOVE_W, n[1], REMOVE_W, CTRL_H]
}

pub(crate) fn view_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    let r = remove_rect(o, w);
    [r[0] - GAP - VIEW_W, r[1], VIEW_W, CTRL_H]
}

// Where the header's captions start: clear of the step buttons to their left.
fn caption_x(o: [f32; 2]) -> f32 {
    next_rect(o)[0] + STEP_W + GAP + 4.0
}

// The open behavior's name, typed in place. It gives up whatever the header
// cannot spare, so a narrow panel keeps the chips rather than overrunning them.
pub(crate) fn name_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    let left = caption_x(o);
    let room = (view_rect(o, w)[0] - GAP - ORDINAL_W - left).max(0.0);
    [left, header_y(o), room.min(NAME_W), CTRL_H]
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

pub(crate) fn dup_rect(o: [f32; 2]) -> [f32; 4] {
    let d = down_rect(o);
    [d[0] + ARROW_W + GAP, d[1], TOOL_BTN_W, TOOL_CTRL_H]
}

pub(crate) fn value_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    let left = dup_rect(o)[0] + TOOL_BTN_W + GAP;
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

// The inspector column, pinned to the panel's right edge so the chart keeps the
// rest of the body.
pub(crate) fn inspect_rect(o: [f32; 2], s: [f32; 2]) -> [f32; 4] {
    let top = body_top(o);
    [
        o[0] + s[0] - INSPECT_W - 1.0,
        top,
        INSPECT_W,
        (o[1] + s[1] - top - 2.0).max(0.0),
    ]
}

fn inspect_row_rect(o: [f32; 2], s: [f32; 2], slot: usize) -> [f32; 4] {
    let b = inspect_rect(o, s);
    [
        b[0] + 3.0,
        b[1] + INSPECT_HEAD_H + slot as f32 * ROW_H,
        b[2] - 6.0,
        ROW_H,
    ]
}

// Where body row `slot` sits: across the panel in outline view, in the
// inspector column beside the chart otherwise.
pub(crate) fn body_row_rect(o: [f32; 2], s: [f32; 2], mode: ViewMode, slot: usize) -> [f32; 4] {
    match mode {
        ViewMode::Outline => row_rect(o, s[0], slot),
        _ => inspect_row_rect(o, s, slot),
    }
}

// How many field rows the inspector has room for at panel size `s`.
pub(crate) fn inspect_rows(s: [f32; 2]) -> usize {
    let h = inspect_rect([0.0, 0.0], s)[3] - INSPECT_HEAD_H;
    ((h / ROW_H).floor().max(0.0) as usize).min(ROW_POOL_MAX)
}

// The palette floats over the outline, anchored under the toolbar so it always
// stays inside the panel however far the body is scrolled.
fn palette_backing(o: [f32; 2], w: f32, shown: usize) -> [f32; 4] {
    // Rounded out to an outline-row boundary, so the palette's edge never
    // bisects the row beneath it and leaves it half drawn.
    let h = FILTER_H + shown as f32 * PICK_ROW_H + 4.0;
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
        b[1] + 2.0 + FILTER_H + slot as f32 * PICK_ROW_H,
        b[2] - 4.0,
        PICK_ROW_H,
    ]
}

// The field the palette is narrowed by, along its top edge.
pub(crate) fn filter_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    let b = palette_backing(o, w, 1);
    [b[0] + 2.0, b[1] + 2.0, b[2] - 4.0, FILTER_H - 4.0]
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
        // Pressing the field is aimed at typing into it, so it must not read as
        // pressing away from the palette.
        if point_in(mx, my, filter_rect(o, w)) {
            return Some(BehaviorAction::Consume);
        }
        for slot in 0..view.picks_shown() {
            if point_in(mx, my, pick_rect(o, w, slot)) {
                return Some(match view.pick_at(slot) {
                    Some(i) => BehaviorAction::Choose(i),
                    None => BehaviorAction::Dismiss,
                });
            }
        }
        return point_in(mx, my, widget::outer_rect(o, s)).then_some(BehaviorAction::Dismiss);
    }
    if point_in(mx, my, new_rect(o, w)) {
        return Some(BehaviorAction::New);
    }
    if view.total > 0 {
        // The banner floats over the foot of the body, so it answers before
        // whatever it covers. Pressing it goes to what it is complaining about,
        // which is the only way to reach a fault that is off screen.
        if view.fault_row.is_some()
            && view.status.and_then(Status::error).is_some()
            && point_in(mx, my, status_rect(o, s))
        {
            return Some(BehaviorAction::GoToFault);
        }
        if point_in(mx, my, remove_rect(o, w)) {
            return Some(BehaviorAction::Remove);
        }
        if point_in(mx, my, name_rect(o, w)) {
            return Some(BehaviorAction::FocusName);
        }
        if point_in(mx, my, view_rect(o, w)) {
            return Some(BehaviorAction::ToggleView);
        }
        if point_in(mx, my, prev_rect(o)) {
            return Some(BehaviorAction::Step(-1));
        }
        if point_in(mx, my, next_rect(o)) {
            return Some(BehaviorAction::Step(1));
        }
        // The toolbar and the body rows act on the open behavior, which the
        // overview is not showing.
        if view.mode != ViewMode::Overview {
            for (rect, action) in [
                (palette_rect(o), BehaviorAction::Palette),
                (delete_rect(o), BehaviorAction::Delete),
                (up_rect(o), BehaviorAction::Move(-1)),
                (down_rect(o), BehaviorAction::Move(1)),
                (dup_rect(o), BehaviorAction::Duplicate),
            ] {
                if point_in(mx, my, rect) {
                    return Some(action);
                }
            }
            if view.editable && point_in(mx, my, value_rect(o, w)) {
                return Some(BehaviorAction::FocusValue);
            }
            let shown = match view.mode {
                ViewMode::Chart => view.fields.len().min(inspect_rows(s)),
                _ => visible_rows(s[1]),
            };
            for slot in 0..shown {
                if !point_in(mx, my, body_row_rect(o, s, view.mode, slot)) {
                    continue;
                }
                return Some(match view.row_at(slot) {
                    Some(i) if i < view.rows.len() => BehaviorAction::Select(i),
                    _ => BehaviorAction::Consume,
                });
            }
        }
        // Only the canvas answers here. A press that lands nowhere falls through
        // to the panel-wide check below, which is what lets a press that missed
        // the panel entirely reach the panels behind it.
        if view.mode.drawn_as_chart() {
            let band = chart_band(o, s, view.mode);
            let cv = chart_view(view);
            if let Some(i) = behavior_chart::hit_card(&cv, mx, my, band) {
                return Some(match view.mode {
                    // An overview card opens what it stands for: a behavior's
                    // body, or the table declaring a variable. A trigger or an
                    // asset card has neither.
                    ViewMode::Overview => match &view.overview.cards[i] {
                        c if c.behavior.is_some() => BehaviorAction::OpenCard(i),
                        c if c.kind == CardKind::Variable => BehaviorAction::OpenVariable(i),
                        _ => BehaviorAction::Consume,
                    },
                    _ => BehaviorAction::SelectCard(i),
                });
            }
            if point_in(mx, my, band) {
                return Some(BehaviorAction::PanStart);
            }
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
    layout_status(world, view, o, s);
    if view.mode.drawn_as_chart() && view.total > 0 {
        match view.mode {
            ViewMode::Chart => layout_inspector(world, view, o, s),
            _ => hide_inspector(world, view),
        }
        widget::set_sprite_visible(world, LIST_TRACK, false);
        widget::set_sprite_visible(world, LIST_THUMB, false);
        behavior_chart::apply(world, &chart_view(view), chart_band(o, s, view.mode));
    } else {
        behavior_chart::hide_all(world);
        hide_inspector(world, view);
        layout_rows(world, view, o, s, visible_rows(s[1]));
        layout_scrollbar(world, view, o, w, visible_rows(s[1]));
    }
    layout_palette(world, view, o, w);
}

fn hide_inspector(world: &mut World, view: &BehaviorView) {
    widget::set_sprite_visible(world, INSPECT_BG, false);
    widget::set_label_visible(world, INSPECT_LABEL, false);
    if view.mode != ViewMode::Outline {
        layout_rows(world, view, [0.0, 0.0], [0.0, 0.0], 0);
    }
}

// The chart gets the body, less the inspector column beside it. The overview
// has no inspector -- its cards are whole behaviors, not nodes -- so it takes
// the whole body.
pub(crate) fn chart_band(o: [f32; 2], s: [f32; 2], mode: ViewMode) -> [f32; 4] {
    let w = match mode {
        ViewMode::Chart => (s[0] - INSPECT_W).max(0.0),
        _ => s[0],
    };
    behavior_chart::band(o, [w, s[1]], body_top(o))
}

// The chart canvas's size at panel size `s`, which is all the pan maths needs.
pub(crate) fn chart_canvas(s: [f32; 2], mode: ViewMode) -> [f32; 2] {
    let b = chart_band([0.0, 0.0], s, mode);
    [b[2], b[3]]
}

fn chart_view<'a>(view: &'a BehaviorView<'a>) -> behavior_chart::ChartView<'a> {
    behavior_chart::ChartView {
        chart: view.shown_chart(),
        // The card the selection belongs to, so a node stays lit while its own
        // fields are picked through in the inspector. The overview keeps its
        // own, because its cards stand for behaviors rather than for rows.
        selected: match view.mode {
            ViewMode::Overview => view.overview_card,
            _ => view.card,
        },
        // The card that settles the faulting row, so a complaint about a field
        // lights the node carrying it. The overview draws whole behaviors, not
        // places inside one, so it has nothing to point at.
        faulted: match view.mode {
            ViewMode::Overview => None,
            _ => view
                .fault_row
                .and_then(|i| view.rows.get(i))
                .and_then(|r| fields::owning_card(&view.chart.cards, &r.path)),
        },
        // Pulses and breakpoints address nodes of the open body, so the
        // overview (whole behaviors) draws none.
        pulses: match view.mode {
            ViewMode::Overview => &[],
            _ => view.pulse_cards,
        },
        breakpoints: match view.mode {
            ViewMode::Overview => &[],
            _ => view.break_cards,
        },
        pan: view.pan,
        mouse: view.mouse,
        overflow: match view.mode {
            // The map keeps its middlemen, so what it runs out of room for is
            // behaviors -- and each of those is still its own chart.
            ViewMode::Overview => "cards -- open a behavior with < and > to see its own",
            _ => "nodes -- open the outline to reach them",
        },
    }
}

fn palette_open(view: &BehaviorView) -> bool {
    view.picking && !view.picks.is_empty()
}

fn layout_header(world: &mut World, view: &BehaviorView, o: [f32; 2], w: f32) {
    let has_any = view.total > 0;
    // The armed chip says what the next press does, so nothing is destroyed by
    // a press that only meant to reach for it.
    let (remove_caption, remove_tint) = match view.remove_armed && has_any {
        true => ("Confirm", CONFIRM_TINT),
        false => ("Remove", DEL_TINT),
    };
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
            bg: REMOVE_BG,
            label: REMOVE_LABEL,
            rect: remove_rect(o, w),
            caption: remove_caption,
            tint: remove_tint,
            live: has_any,
        },
        Chip {
            bg: NEW_BG,
            label: NEW_LABEL,
            rect: new_rect(o, w),
            caption: "New",
            tint: NEW_TINT,
            live: true,
        },
    ] {
        place_chip(world, &chip, view.mouse);
    }
    layout_name(world, view, o, w);
}

// The open behavior's name, typed in place, with its place in the world's
// behaviors beside it. An empty world has no name to type, so the field gives
// way to the prompt that says what to press.
fn layout_name(world: &mut World, view: &BehaviorView, o: [f32; 2], w: f32) {
    let text_y = header_y(o) + CTRL_H * 0.5 - theme::TEXT_HALF;
    if view.total == 0 {
        widget::hide_field(world, NAME_INPUT);
        let left = caption_x(o);
        let budget = ((view_rect(o, w)[0] - left) / CHAR_W) as usize;
        widget::place_left_label(
            world,
            NAME_LABEL,
            [left, text_y],
            &widget::clip_text("no behaviors yet -- press New", budget),
            HINT_LABEL,
            true,
        );
        return;
    }
    let field = name_rect(o, w);
    // An idle field mirrors the entry, so a rename from anywhere else (the
    // Assets form, an undo) shows here without the panel being reopened. While
    // it holds the keyboard what is typed stands, or it could never be edited.
    if !view.name_focus {
        widget::seed_field(world, NAME_INPUT, view.name);
    }
    widget::show_field(world, NAME_INPUT, field, view.name_focus);
    widget::place_left_label(
        world,
        NAME_LABEL,
        [field[0] + field[2] + GAP, text_y],
        &format!("{}/{}", view.index + 1, view.total),
        HINT_LABEL,
        true,
    );
}

fn layout_toolbar(world: &mut World, view: &BehaviorView, o: [f32; 2], w: f32) {
    // The toolbar acts on a row of the open behavior, so the overview leaves it
    // with nothing to act on.
    let on = view.total > 0 && view.mode != ViewMode::Overview;
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
        Chip {
            bg: DUP_BG,
            label: DUP_LABEL,
            rect: dup_rect(o),
            caption: "Dup",
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

// The checker's complaint, floating over the foot of the body. Nothing is drawn
// while the open behavior checks out, so a working panel is all body.
fn layout_status(world: &mut World, view: &BehaviorView, o: [f32; 2], s: [f32; 2]) {
    let Some(error) = view
        .status
        .and_then(Status::error)
        .filter(|_| view.total > 0)
    else {
        widget::set_sprite_visible(world, STATUS_BG, false);
        widget::set_label_visible(world, STATUS_LABEL, false);
        return;
    };
    let b = status_rect(o, s);
    widget::place_bordered(world, STATUS_BG, b, ERROR_TINT, ERROR_BORDER, 1.0);
    widget::place_message(
        world,
        STATUS_LABEL,
        [b[0] + PAD, b[1] + 3.0, b[2] - 2.0 * PAD, b[3] - 6.0],
        error,
        theme::LOG_ERROR,
        true,
    );
}

pub(crate) fn status_rect(o: [f32; 2], s: [f32; 2]) -> [f32; 4] {
    let bottom = o[1] + s[1] - PAD;
    // Grown up to an outline-row boundary, so the banner's edge never bisects
    // the row it floats over and leaves it half drawn.
    let rows = ((bottom - STATUS_H - body_top(o)) / ROW_H).floor().max(0.0);
    let top = body_top(o) + rows * ROW_H;
    [o[0] + PAD, top, (s[0] - 2.0 * PAD).max(0.0), bottom - top]
}

// The inspector: the selected node's own settings beside the chart. Every row
// is an ordinary body row, so the toolbar and the palette act on it exactly as
// they do in the outline.
fn layout_inspector(world: &mut World, view: &BehaviorView, o: [f32; 2], s: [f32; 2]) {
    let b = inspect_rect(o, s);
    place_rounded(
        world,
        INSPECT_BG,
        b,
        INSPECT_TINT,
        theme::CONTROL_RADIUS,
        true,
    );
    let room = inspect_rows(s);
    let caption = match (view.card.and_then(|i| view.chart.cards.get(i)), room) {
        (None, _) => "select a card".to_string(),
        // A node with more settings than the column has room for says so, the
        // way the chart does when a body outgrows it.
        (Some(c), room) if view.fields.len() > room => format!("{} (see outline)", c.title),
        (Some(c), _) => c.title.clone(),
    };
    widget::place_left_label(
        world,
        INSPECT_LABEL,
        [b[0] + PAD, b[1] + INSPECT_HEAD_H * 0.5 - theme::TEXT_HALF],
        &widget::clip_text(&caption, ((b[2] - 2.0 * PAD) / CHAR_W) as usize),
        if view.card.is_some() {
            theme::HEADING
        } else {
            HINT_LABEL
        },
        true,
    );
    layout_rows(world, view, o, s, view.fields.len().min(room));
}

fn layout_rows(world: &mut World, view: &BehaviorView, o: [f32; 2], s: [f32; 2], shown: usize) {
    let (value_x, value_budget) = match view.mode {
        ViewMode::Outline => (value_column(s[0]), max_value_chars(s[0])),
        _ => (
            INSPECT_VALUE_X,
            ((INSPECT_W - INSPECT_VALUE_X - PAD) / CHAR_W) as usize,
        ),
    };
    let label_budget = ((value_x - PAD * 2.0) / CHAR_W) as usize;
    // The inspector starts at the selected node, so its rows indent from there
    // rather than from however deep in the body that node sits.
    let base_depth = view.base_depth();
    for slot in 0..ROW_POOL_MAX {
        let drawn = slot < shown && view.total > 0;
        let Some((i, row)) = view
            .row_at(slot)
            .and_then(|i| view.rows.get(i).map(|r| (i, r)))
            .filter(|_| drawn)
        else {
            widget::set_sprite_visible(world, row_bg(slot), false);
            widget::set_label_visible(world, row_label(slot), false);
            widget::set_label_visible(world, row_value(slot), false);
            continue;
        };
        let depth = row.depth.saturating_sub(base_depth);
        let r = body_row_rect(o, s, view.mode, slot);
        let hovered = point_in(view.mouse[0], view.mouse[1], r);
        let tint = if view.selected == Some(i) {
            theme::SELECTED_TINT
        } else if hovered {
            theme::HOVER_TINT
        } else {
            ROW_TINT
        };
        // A row whose node just executed leans toward the pulse accent, the
        // same mark its chart card carries.
        let tint = match view.pulse_rows.iter().find(|(idx, _)| *idx == i) {
            Some((_, alpha)) => super::behavior::pulse::blend(tint, *alpha),
            None => tint,
        };
        // The fault is a border rather than a tint, so it says nothing about
        // whether the row is also the selected one.
        let faulted = view.fault_row == Some(i);
        widget::place_bordered(
            world,
            row_bg(slot),
            theme::highlight_rect(r),
            tint,
            ERROR_BORDER,
            if faulted { 1.0 } else { 0.0 },
        );
        let text_y = r[1] + ROW_H * 0.5 - theme::TEXT_HALF;
        widget::place_left_label(
            world,
            row_label(slot),
            [r[0] + PAD + depth as f32 * INDENT, text_y],
            &widget::clip_text(&row.label, label_budget.max(6)),
            label_color(&row.kind),
            true,
        );
        widget::place_left_label(
            world,
            row_value(slot),
            [r[0] + value_x, text_y],
            &widget::clip_text(&row.value, value_budget),
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

// The floating palette: an opaque backing over the outline, a field narrowing
// what it offers, one row per kept option with its hint, and its own scrollbar
// once what is kept overflows.
fn layout_palette(world: &mut World, view: &BehaviorView, o: [f32; 2], w: f32) {
    if !palette_open(view) {
        widget::set_sprite_visible(world, DROP_BG, false);
        widget::set_sprite_visible(world, DROP_TRACK, false);
        widget::set_sprite_visible(world, DROP_THUMB, false);
        widget::hide_field(world, FILTER_INPUT);
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
    layout_filter(world, view, o, w);
    for slot in 0..PICK_POOL {
        let Some(i) = view.pick_at(slot).filter(|_| slot < shown) else {
            // A query nothing answers keeps its one slot to say so, rather than
            // collapsing the palette out from under the field being typed into.
            if slot == 0 && view.matches.is_empty() {
                empty_palette_row(world, o, w);
                continue;
            }
            widget::set_sprite_visible(world, pick_bg(slot), false);
            widget::set_label_visible(world, pick_label(slot), false);
            widget::set_label_visible(world, pick_hint(slot), false);
            continue;
        };
        let Some(pick) = view.picks.get(i) else {
            widget::set_sprite_visible(world, pick_bg(slot), false);
            widget::set_label_visible(world, pick_label(slot), false);
            widget::set_label_visible(world, pick_hint(slot), false);
            continue;
        };
        let r = pick_rect(o, w, slot);
        let hovered = point_in(view.mouse[0], view.mouse[1], r);
        let tint = if view.pick_scroll + slot == view.pick {
            theme::SELECTED_TINT
        } else if hovered {
            theme::HOVER_TINT
        } else {
            ROW_TINT
        };
        place_rounded(world, pick_bg(slot), r, tint, theme::CONTROL_RADIUS, true);
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

// The field the palette is narrowed by. Its ghost says what typing does, so the
// affordance does not need a label of its own taking a row.
fn layout_filter(world: &mut World, view: &BehaviorView, o: [f32; 2], w: f32) {
    let r = filter_rect(o, w);
    place_rounded(
        world,
        DROP_FILTER_BG,
        r,
        FILTER_TINT,
        theme::CONTROL_RADIUS,
        true,
    );
    if let Some(t) = widget::input_mut(world, FILTER_INPUT) {
        t.ghost = "filter".to_string();
    }
    widget::show_field(
        world,
        FILTER_INPUT,
        [r[0] + 4.0, r[1] + 2.0, r[2] - 8.0, r[3] - 4.0],
        view.filter_focus,
    );
}

// What the palette says when the query answers nothing: the vocabulary is still
// there, so this is about what was typed rather than about the row.
fn empty_palette_row(world: &mut World, o: [f32; 2], w: f32) {
    let r = pick_rect(o, w, 0);
    widget::set_sprite_visible(world, pick_bg(0), false);
    widget::set_label_visible(world, pick_hint(0), false);
    widget::place_left_label(
        world,
        pick_label(0),
        [r[0] + PAD, r[1] + PICK_ROW_H * 0.5 - theme::TEXT_HALF],
        "no option matches",
        HINT_LABEL,
        true,
    );
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
    let top = b[1] + FILTER_H;
    let h = shown as f32 * PICK_ROW_H;
    place_rounded(
        world,
        DROP_TRACK,
        [x, top + 2.0, SCROLLBAR_W, h],
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
        [x, top + 2.0 + off, SCROLLBAR_W, thumb_h],
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
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &all_field_ids());
}

// Every panel sprite id, in draw (insertion) order: chrome, then the outline
// rows, then the scrollbar and the palette floating above them.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![
        PANEL_BG, CLOSE_BG, PREV_BG, NEXT_BG, VIEW_BG, REMOVE_BG, NEW_BG, PICK_BG, DEL_BG, UP_BG,
        DOWN_BG, DUP_BG,
    ];
    ids.push(INSPECT_BG);
    ids.extend((0..ROW_POOL_MAX).map(row_bg));
    ids.extend(behavior_chart::all_sprite_ids());
    ids.extend([STATUS_BG, LIST_TRACK, LIST_THUMB, DROP_BG, DROP_FILTER_BG]);
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
        REMOVE_LABEL,
        NEW_LABEL,
        PICK_LABEL,
        DEL_LABEL,
        UP_LABEL,
        DOWN_LABEL,
        DUP_LABEL,
        STATUS_LABEL,
        INSPECT_LABEL,
    ];
    ids.extend((0..ROW_POOL_MAX).map(row_label));
    ids.extend((0..ROW_POOL_MAX).map(row_value));
    ids.extend(behavior_chart::all_label_ids());
    ids.extend((0..PICK_POOL).map(pick_label));
    ids.extend((0..PICK_POOL).map(pick_hint));
    ids
}

// The error banner's elements. They float over the body, so they draw above it
// and its backing occludes the rows or cards beneath.
pub(crate) fn status_ids() -> Vec<AssetId> {
    vec![STATUS_BG, STATUS_LABEL]
}

// The palette's own elements, which draw above the rest of the panel while it
// is open so its backing occludes whatever it floats over.
pub(crate) fn palette_ids() -> Vec<AssetId> {
    // The filter field is one of these: a panel's fields otherwise draw at its
    // base layer, which is under the palette's own opaque backing.
    let mut ids = vec![
        DROP_BG,
        DROP_FILTER_BG,
        DROP_TRACK,
        DROP_THUMB,
        FILTER_INPUT,
    ];
    ids.extend((0..PICK_POOL).map(pick_bg));
    ids.extend((0..PICK_POOL).map(pick_label));
    ids.extend((0..PICK_POOL).map(pick_hint));
    ids
}

pub(crate) fn all_field_ids() -> Vec<AssetId> {
    vec![VALUE_INPUT, NAME_INPUT, FILTER_INPUT]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextInput, TextLabel};
    use crate::editor::behavior::outline;

    fn injected_world() -> World {
        let mut world = World::new();
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

    fn empty_chart() -> &'static Chart {
        static CHART: std::sync::OnceLock<Chart> = std::sync::OnceLock::new();
        CHART.get_or_init(Chart::default)
    }

    fn sample_args() -> serde_json::Value {
        serde_json::json!({"on": "tick", "scope": ["Prop"],
            "do": [{"hide": {"target": "self"}}]})
    }

    fn sample_rows() -> Vec<Row> {
        outline::rows(&sample_args())
    }

    // Every option kept, which is what an unfiltered palette shows.
    fn all(picks: &[Pick]) -> Vec<usize> {
        (0..picks.len()).collect()
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
            overview: empty_chart(),
            mode: ViewMode::Outline,
            pan: [0.0, 0.0],
            card: None,
            fields: &[],
            picks,
            matches: &[],
            picking: false,
            pick_scroll: 0,
            pick: 0,
            filter_focus: false,
            overview_card: None,
            editable: false,
            focus: false,
            name_focus: false,
            remove_armed: false,
            status: None,
            fault_row: None,
            pulse_cards: &[],
            pulse_rows: &[],
            break_cards: &[],
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

    fn field(world: &World, id: AssetId) -> TextInput {
        world
            .query::<TextInput>()
            .find(|t| t.asset_id == id)
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

    // The header packs left to right without a gap that lands on two controls
    // at once, at the default width and at the much wider chart view alike.
    #[test]
    fn the_header_packs_the_name_field_and_both_asset_chips() {
        let o = [40.0, 40.0];
        for w in [BEHAVIOR_W, chart_size()[0], overview_size()[0]] {
            let (name, view, remove, new) = (
                name_rect(o, w),
                view_rect(o, w),
                remove_rect(o, w),
                new_rect(o, w),
            );
            let mut x = next_rect(o)[0] + STEP_W;
            for r in [name, view, remove, new] {
                assert!(x <= r[0], "w {w}: the header packs left to right");
                x = r[0] + r[2];
            }
            assert_eq!(x, o[0] + w - PAD, "w {w}: New is still pinned right");
            assert!(
                name[2] >= 150.0,
                "w {w}: the name stays typable: {}",
                name[2]
            );
            assert!(name[2] <= NAME_W, "w {w}: and does not sprawl");
        }
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
        assert_eq!(
            hit(remove_rect(o, BEHAVIOR_W)),
            Some(BehaviorAction::Remove)
        );
        assert_eq!(
            hit(name_rect(o, BEHAVIOR_W)),
            Some(BehaviorAction::FocusName)
        );
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
        let kept = all(&picks);
        let v = BehaviorView {
            picking: true,
            matches: &kept,
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
        // Removing and renaming need something open, so both stand down too.
        for r in [remove_rect(o, BEHAVIOR_W), name_rect(o, BEHAVIOR_W)] {
            assert_eq!(
                hit_test(&v, r[0] + 3.0, r[1] + 3.0, o, s),
                Some(BehaviorAction::Consume)
            );
        }
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
        assert!(!field(&world, NAME_INPUT).visible, "and no name to type");
        assert!(!sprite(&world, row_bg(0)).visible, "no outline to draw");
    }

    // The name is a field, not a caption: it shows what the world holds while it
    // is idle, and stands still while it is being typed into.
    #[test]
    fn the_name_field_mirrors_the_open_behavior_unless_it_is_focused() {
        let mut world = injected_world();
        let rows = sample_rows();
        let (o, s) = ([20.0, 20.0], size());
        apply(&mut world, Some(&view(&rows, &[])), o, s);
        let f = field(&world, NAME_INPUT);
        assert!(f.visible && !f.focused);
        assert_eq!(f.content, "chase");
        assert_eq!(label(&world, NAME_LABEL).content, "1/1");

        widget::seed_field(&mut world, NAME_INPUT, "half typed");
        let typing = BehaviorView {
            name_focus: true,
            ..view(&rows, &[])
        };
        apply(&mut world, Some(&typing), o, s);
        let f = field(&world, NAME_INPUT);
        assert!(
            f.focused,
            "the field keeps the keyboard while it is typed into"
        );
        assert_eq!(f.content, "half typed", "and what is typed stands");

        // Losing focus puts the world's own name back in front of the user.
        apply(&mut world, Some(&view(&rows, &[])), o, s);
        assert_eq!(field(&world, NAME_INPUT).content, "chase");
    }

    // Removing a behavior destroys an authored asset, so the chip says what the
    // next press does before it does it.
    #[test]
    fn the_removal_chip_arms_before_it_commits() {
        let mut world = injected_world();
        let rows = sample_rows();
        let (o, s) = ([20.0, 20.0], size());
        apply(&mut world, Some(&view(&rows, &[])), o, s);
        assert_eq!(label(&world, REMOVE_LABEL).content, "Remove");
        let idle = sprite(&world, REMOVE_BG).tint;

        let armed = BehaviorView {
            remove_armed: true,
            ..view(&rows, &[])
        };
        apply(&mut world, Some(&armed), o, s);
        assert_eq!(label(&world, REMOVE_LABEL).content, "Confirm");
        assert_ne!(
            sprite(&world, REMOVE_BG).tint,
            idle,
            "an armed chip does not read like an idle one"
        );
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
        assert_eq!(label(&world, NAME_LABEL).content, "1/1");
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
        let kept = all(&picks);
        let v = BehaviorView {
            picking: true,
            matches: &kept,
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

    // A complaint the checker located is drawn where it is about: the outline row
    // takes a border (a border, so it says nothing about the selection) and the
    // chart lights the card that settles that row.
    #[test]
    fn a_located_fault_marks_the_row_and_the_card_it_is_about() {
        let mut world = injected_world();
        let rows = sample_rows();
        let target = rows
            .iter()
            .position(|r| r.label == "target")
            .expect("the hide node's target row");

        apply(
            &mut world,
            Some(&BehaviorView {
                fault_row: Some(target),
                ..view(&rows, &[])
            }),
            [20.0, 20.0],
            size(),
        );
        assert_eq!(sprite(&world, row_bg(target)).border_width, 1.0);
        assert_eq!(
            sprite(&world, row_bg(target)).border_color,
            ERROR_BORDER,
            "the faulting row is bordered, not tinted"
        );
        let other = (0..rows.len()).find(|i| *i != target).expect("another row");
        assert_eq!(
            sprite(&world, row_bg(other)).border_width,
            0.0,
            "a row with nothing wrong carries no border"
        );

        // In the chart the same fault lights the node that carries the field.
        apply(
            &mut world,
            Some(&BehaviorView {
                mode: ViewMode::Chart,
                fault_row: Some(target),
                ..view(&rows, &[])
            }),
            [20.0, 20.0],
            chart_size(),
        );
        let card =
            crate::editor::behavior::fields::owning_card(&sample_chart().cards, &rows[target].path)
                .expect("a card settles the target row");
        assert_eq!(
            sprite(&world, behavior_chart::card_bg(card)).border_width,
            2.0,
            "nothing is selected, so only the fault can have widened it"
        );
    }

    // The banner covers the foot of the body, so it has to answer before the rows
    // under it -- and only when there is somewhere to go.
    #[test]
    fn pressing_the_banner_goes_to_what_it_is_complaining_about() {
        let rows = sample_rows();
        let o = [20.0, 20.0];
        let s = size();
        let b = status_rect(o, s);
        let (mx, my) = (b[0] + 4.0, b[1] + 4.0);

        let complaining = BehaviorView {
            status: Some(&Status::message("Behavior 'x': reads unbound name 'q'")),
            fault_row: Some(2),
            ..view(&rows, &[])
        };
        assert_eq!(
            hit_test(&complaining, mx, my, o, s),
            Some(BehaviorAction::GoToFault),
        );
        // A complaint with nowhere to point leaves the press to the body.
        let unplaced = BehaviorView {
            status: Some(&Status::message("a behavior needs a name")),
            ..view(&rows, &[])
        };
        assert_ne!(
            hit_test(&unplaced, mx, my, o, s),
            Some(BehaviorAction::GoToFault),
        );
        // And so does a behavior that checks out.
        assert_ne!(
            hit_test(&view(&rows, &[]), mx, my, o, s),
            Some(BehaviorAction::GoToFault),
        );
    }

    // Duplicating is the discoverable half of the clipboard: the keys carry a
    // node between behaviors, but the common case is a copy beside the original,
    // so that one has a chip. It acts on a list member, like Del and the arrows.
    #[test]
    fn the_duplicate_chip_answers_only_for_a_list_member() {
        let mut world = injected_world();
        let rows = sample_rows();
        let o = [20.0, 20.0];
        let s = size();
        let member = rows
            .iter()
            .position(|r| r.element.is_some())
            .expect("the sample body has a node");
        let d = dup_rect(o);

        let on_member = BehaviorView {
            selected: Some(member),
            ..view(&rows, &[])
        };
        assert_eq!(
            hit_test(&on_member, d[0] + 3.0, d[1] + 3.0, o, s),
            Some(BehaviorAction::Duplicate),
        );
        apply(&mut world, Some(&on_member), o, s);
        let live = label(&world, DUP_LABEL).color;

        // A row that is not a member leaves the chip dim, like Del beside it.
        let not_member = rows
            .iter()
            .position(|r| r.element.is_none())
            .expect("the source row is not a member");
        apply(
            &mut world,
            Some(&BehaviorView {
                selected: Some(not_member),
                ..view(&rows, &[])
            }),
            o,
            s,
        );
        assert_ne!(label(&world, DUP_LABEL).color, live, "the chip went dim");
        assert_eq!(
            label(&world, DEL_LABEL).color,
            label(&world, DUP_LABEL).color,
            "and it reads the same as the other member-only chips",
        );
    }

    // The chip took width off the value field, which still has to hold a caption.
    #[test]
    fn the_toolbar_still_leaves_the_value_field_room() {
        let o = [0.0, 0.0];
        let v = value_rect(o, BEHAVIOR_W);
        let d = dup_rect(o);
        assert!(v[0] >= d[0] + d[2], "the field starts clear of the chip");
        assert!(
            v[2] >= MAX_VALUE_CHARS as f32 * CHAR_W,
            "the value column no longer fits its own budget: {v:?}",
        );
    }

    // The map keeps a selection of its own, because its cards stand for whole
    // behaviors rather than for rows of the open one. Without it a step through
    // the map would move nothing anyone can see.
    #[test]
    fn the_overview_lights_the_card_its_own_selection_is_on() {
        let mut world = injected_world();
        let rows = sample_rows();
        let v = BehaviorView {
            mode: ViewMode::Overview,
            overview: sample_chart(),
            overview_card: Some(1),
            // The outline selection the other views share reaches no card here.
            card: Some(0),
            ..view(&rows, &[])
        };
        apply(&mut world, Some(&v), [20.0, 20.0], overview_size());
        assert_eq!(sprite(&world, behavior_chart::card_bg(1)).border_width, 2.0);
        assert_eq!(sprite(&world, behavior_chart::card_bg(0)).border_width, 1.0);
    }

    // The field narrowing the palette sits inside it, so what is typed and what
    // it keeps are read in one place. Its slots address the kept options, which
    // is what lets a filtered palette be picked from at all.
    #[test]
    fn the_palette_carries_the_field_it_is_narrowed_by() {
        let mut world = injected_world();
        let rows = sample_rows();
        let picks = edit::picks(&Kind::List(outline::List::Nodes), &[]);
        let o = [20.0, 20.0];
        let s = size();
        // Two kept options, the second of which is option 5 of the vocabulary.
        let kept = vec![1usize, 5];
        let v = BehaviorView {
            picking: true,
            matches: &kept,
            filter_focus: true,
            ..view(&rows, &picks)
        };
        apply(&mut world, Some(&v), o, s);

        let f = field(&world, FILTER_INPUT);
        assert!(f.visible && f.focused, "the filter takes the keyboard");
        assert!(sprite(&world, DROP_FILTER_BG).visible);
        let first = pick_rect(o, BEHAVIOR_W, 0);
        assert!(
            f.y + f.height <= first[1] + 0.01,
            "the field sits above the options it narrows"
        );
        assert_eq!(label(&world, pick_label(0)).content, picks[1].verb);
        assert_eq!(label(&world, pick_label(1)).content, picks[5].verb);
        assert!(
            !label(&world, pick_label(2)).visible,
            "only what the query keeps is drawn"
        );
        // A slot addresses the option it draws, not its place in the palette.
        let r = pick_rect(o, BEHAVIOR_W, 1);
        assert_eq!(
            hit_test(&v, r[0] + 3.0, r[1] + 3.0, o, s),
            Some(BehaviorAction::Choose(5)),
        );
        // And pressing the field is aimed at typing, not at closing.
        let b = filter_rect(o, BEHAVIOR_W);
        assert_eq!(
            hit_test(&v, b[0] + 3.0, b[1] + 3.0, o, s),
            Some(BehaviorAction::Consume),
        );
    }

    // A query nothing answers keeps the palette open and says so: the field being
    // typed into is inside it, so collapsing would take away the fix.
    #[test]
    fn a_query_nothing_answers_still_draws_the_palette() {
        let mut world = injected_world();
        let rows = sample_rows();
        let picks = edit::picks(&Kind::List(outline::List::Nodes), &[]);
        let v = BehaviorView {
            picking: true,
            matches: &[],
            filter_focus: true,
            ..view(&rows, &picks)
        };
        apply(&mut world, Some(&v), [20.0, 20.0], size());
        assert!(sprite(&world, DROP_BG).visible, "the palette is still up");
        assert!(field(&world, FILTER_INPUT).visible, "and still typeable");
        assert_eq!(label(&world, pick_label(0)).content, "no option matches");
        assert!(!sprite(&world, pick_bg(0)).visible, "nothing to press");
    }

    // Where the keyboard is in the palette is drawn the way a selected outline
    // row is, so the highlight reads as the selection it is rather than as a
    // second kind of hover. It follows the scroll, because a slot stands for a
    // different option once the window has moved.
    #[test]
    fn the_palette_draws_the_option_the_keyboard_is_on_as_selected() {
        let mut world = injected_world();
        let rows = sample_rows();
        let picks = edit::picks(&Kind::List(outline::List::Nodes), &[]);
        let kept = all(&picks);
        let v = BehaviorView {
            picking: true,
            matches: &kept,
            pick: 2,
            ..view(&rows, &picks)
        };
        apply(&mut world, Some(&v), [20.0, 20.0], size());
        assert_eq!(sprite(&world, pick_bg(2)).tint, theme::SELECTED_TINT);
        assert_ne!(sprite(&world, pick_bg(1)).tint, theme::SELECTED_TINT);

        let scrolled = BehaviorView {
            picking: true,
            matches: &kept,
            pick: 2,
            pick_scroll: 2,
            ..view(&rows, &picks)
        };
        apply(&mut world, Some(&scrolled), [20.0, 20.0], size());
        assert_eq!(sprite(&world, pick_bg(0)).tint, theme::SELECTED_TINT);
        assert_ne!(sprite(&world, pick_bg(2)).tint, theme::SELECTED_TINT);
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
        let kept = all(&picks);
        let open = BehaviorView {
            picking: true,
            matches: &kept,
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
        for id in [
            DROP_BG,
            DROP_FILTER_BG,
            DROP_TRACK,
            DROP_THUMB,
            FILTER_INPUT,
        ] {
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

    // A behavior that checks out says nothing at all; only a complaint draws,
    // and it draws on an opaque backing because it floats over the body.
    #[test]
    fn only_a_failing_check_draws_a_banner() {
        let mut world = injected_world();
        let rows = sample_rows();
        let error = Status::message("Behavior 'x': reads unbound name 'q'");
        let v = BehaviorView {
            status: Some(&error),
            ..view(&rows, &[])
        };
        apply(&mut world, Some(&v), [20.0, 20.0], size());
        let status = label(&world, STATUS_LABEL);
        assert!(status.visible && status.color == theme::LOG_ERROR);
        let backing = sprite(&world, STATUS_BG);
        assert!(backing.visible && backing.tint[3] >= 1.0, "opaque backing");

        let v = BehaviorView {
            status: Some(&Status::Ok),
            ..view(&rows, &[])
        };
        apply(&mut world, Some(&v), [20.0, 20.0], size());
        assert!(!label(&world, STATUS_LABEL).visible);
        assert!(!sprite(&world, STATUS_BG).visible);
    }

    // A chart-view helper: the sample body's one node, selected, with the rows
    // the inspector would list for it.
    fn chart_view_of<'a>(rows: &'a [Row], fields: &'a [usize]) -> BehaviorView<'a> {
        BehaviorView {
            mode: ViewMode::Chart,
            card: Some(1),
            fields,
            selected: fields.first().copied(),
            ..view(rows, &[])
        }
    }

    fn node_fields(rows: &[Row]) -> Vec<usize> {
        let chart = sample_chart();
        crate::editor::behavior::fields::own_rows(rows, &chart.cards, 1)
    }

    // The inspector column is beside the chart, not over it, so selecting a card
    // never hides the body it belongs to.
    #[test]
    fn the_inspector_sits_clear_of_the_chart() {
        let (o, s) = ([20.0, 20.0], chart_size());
        let (band, pane) = (chart_band(o, s, ViewMode::Chart), inspect_rect(o, s));
        assert!(band[0] + band[2] <= pane[0], "the chart stops at the pane");
        assert!(
            pane[0] + pane[2] <= o[0] + s[0],
            "and the pane is on screen"
        );
        assert!(inspect_rows(s) >= 8, "with room for a node's settings");
        // The default width still leaves the chart its columns beside the pane.
        assert!(band[2] >= behavior_chart::width_for(3) - 2.0);
        // And the whole panel fits a small window, since it cannot trim itself
        // to one: a trigger, a node, and the card that appends after it.
        assert!(s[0] <= 1024.0, "chart view opens {} wide", s[0]);
        assert!(overview_size()[0] <= 1024.0);
    }

    #[test]
    fn chart_view_lists_the_selected_nodes_settings() {
        let mut world = injected_world();
        let rows = sample_rows();
        let fields = node_fields(&rows);
        assert!(fields.len() > 1, "the sample node has settings");
        let (o, s) = ([20.0, 20.0], chart_size());
        apply(&mut world, Some(&chart_view_of(&rows, &fields)), o, s);

        let head = label(&world, INSPECT_LABEL);
        assert!(head.visible && head.content == sample_chart().cards[1].title);
        let pane = inspect_rect(o, s);
        for slot in 0..fields.len() {
            let l = label(&world, row_label(slot));
            assert!(l.visible, "field {slot} is listed");
            assert!(l.x >= pane[0] && l.x < pane[0] + pane[2], "inside the pane");
        }
        assert!(
            !label(&world, row_label(fields.len())).visible,
            "and nothing past them"
        );
    }

    // A press that misses the panel has to miss in every view. `hit_test`
    // answering `Some` is what tells the router the press was swallowed, so a
    // chart that answered for points outside its own panel locked out every
    // panel behind it.
    #[test]
    fn a_press_outside_the_panel_misses_in_every_view() {
        let rows = sample_rows();
        let fields = node_fields(&rows);
        let o = [200.0, 150.0];
        for (mode, s) in [
            (ViewMode::Outline, size()),
            (ViewMode::Chart, chart_size()),
            (ViewMode::Overview, overview_size()),
        ] {
            let v = BehaviorView {
                mode,
                card: Some(1),
                fields: &fields,
                ..view(&rows, &[])
            };
            for at in [
                [o[0] - 40.0, o[1] + 60.0],
                [o[0] + s[0] + 40.0, o[1] + 60.0],
                [o[0] + 60.0, o[1] + s[1] + 40.0],
                [8.0, 8.0],
            ] {
                assert_eq!(
                    hit_test(&v, at[0], at[1], o, s),
                    None,
                    "{mode:?} swallowed a press at {at:?}"
                );
            }
            // A press on the panel is still the panel's, whatever it lands on.
            assert!(hit_test(&v, o[0] + 20.0, o[1] + s[1] - 20.0, o, s).is_some());
        }
    }

    // A click in the pane picks the field it lands on rather than falling
    // through to the canvas behind and starting a pan.
    #[test]
    fn clicking_an_inspector_row_selects_that_field() {
        let rows = sample_rows();
        let fields = node_fields(&rows);
        let (o, s) = ([20.0, 20.0], chart_size());
        let v = chart_view_of(&rows, &fields);
        let r = body_row_rect(o, s, ViewMode::Chart, 1);
        assert_eq!(
            hit_test(&v, r[0] + 4.0, r[1] + 4.0, o, s),
            Some(BehaviorAction::Select(fields[1]))
        );
    }

    // With nothing selected the pane says so instead of standing empty.
    #[test]
    fn an_empty_inspector_prompts_for_a_card() {
        let mut world = injected_world();
        let rows = sample_rows();
        let v = BehaviorView {
            mode: ViewMode::Chart,
            ..view(&rows, &[])
        };
        apply(&mut world, Some(&v), [20.0, 20.0], chart_size());
        assert_eq!(label(&world, INSPECT_LABEL).content, "select a card");
        assert!(!label(&world, row_label(0)).visible);
    }

    // The banner floats over the foot of the body instead of sitting in the
    // chrome, so the rows are in the same place whether or not it is showing.
    #[test]
    fn the_banner_does_not_move_the_body() {
        let o = [20.0, 20.0];
        let s = size();
        let first = row_rect(o, s[0], 0);
        let b = status_rect(o, s);
        assert!(b[1] > first[1], "the banner sits below the first row");
        assert!(b[1] + b[3] <= o[1] + s[1], "and inside the panel");
        assert_eq!(body_top(o), o[1] + widget::TITLE_H + HEADER_H + TOOL_H);
        // Its top lands on a row boundary, so no row is left half drawn, and it
        // is still tall enough for a two-line message.
        assert_eq!((b[1] - body_top(o)) % ROW_H, 0.0, "on a row boundary");
        assert!(b[3] >= STATUS_H, "and no shorter than the message needs");
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
        let kept = all(&picks);
        let v = BehaviorView {
            picking: true,
            matches: &kept,
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

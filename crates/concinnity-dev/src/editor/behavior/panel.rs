//! The Behavior panel: a floating node-graph editor over one `Behavior` asset,
//! built from engine widgets. The header cycles through the world's behaviors,
//! names the open one, and adds or removes one; the toolbar acts on the selected
//! row (open its palette, delete it, move it, or type its value). The body is the
//! asset either way: the outline lists every field of every node
//! (`behavior/outline.rs`), while the chart draws the control flow as cards on a
//! canvas with the selected node's own settings in the inspector beside it
//! (`behavior/fields.rs`), and the overview maps the world's behaviors.
//!
//! The palette floats over the top of the body, so picking a node or an
//! expression never leaves the panel, and a press away from it closes it. The
//! checker's complaint floats over the body's foot rather than sitting in the
//! chrome -- a behavior that checks out says nothing, and the body never moves
//! under the user either way.
//! `hook/edit/behavior.rs` owns the model and the actions.

use concinnity_engine::components::{
    Binding, Button, Canvas, Layout, LayoutAlign, LayoutChild, LayoutDirection, LayoutSize, List,
    Sprite, TextAlign, TextField, TextLabel, UiEvent, UiItem, UiValue, WidgetStyle,
};
use concinnity_engine::ecs::asset_id::AssetId;
use concinnity_engine::ecs::{Ref, World};

use super::chart;
use super::edit::{self, Pick};
use super::graph::{Card, CardKind, Chart};
use super::outline::{Kind, Row};
use super::path::{Path, Step};
use super::pulse;
use crate::editor::panels::registry::PanelKey;
use crate::editor::theme;
use crate::editor::theme::TITLE_H;
use crate::editor::ui::build::{BODY_PAD, emit, fill};
use crate::editor::ui::{BuiltPanel, PanelSpec, UiBuilder, panel_height, state};

// The default (and minimum) outline width; the user can widen the panel past it.
pub(crate) const BEHAVIOR_W: f32 = 620.0;
const GAP: f32 = 4.0;
const CTRL_H: f32 = 24.0;
const ROW_H: f32 = 22.0;
const STEP_W: f32 = 26.0;
const NEW_W: f32 = 58.0;
const VIEW_W: f32 = 74.0;
const REMOVE_W: f32 = 70.0;
// The name field is capped rather than filling, so the chart view's much wider
// header does not stretch a name across half the panel.
const NAME_W: f32 = 220.0;
const ORDINAL_W: f32 = 62.0;
const TOOL_BTN_W: f32 = 52.0;
const ARROW_W: f32 = 30.0;
// Outline rows the panel opens tall enough for.
const OUTLINE_ROWS: usize = 16;
// Palette options shown before its list scrolls, and its filter field.
const PICK_ROWS: usize = 10;
const PICK_ROW_H: f32 = 24.0;
const FILTER_H: f32 = 26.0;
// The checker's complaint, floated over the foot of the body.
const STATUS_H: f32 = 30.0;
// How far each outline level insets its caption.
const INDENT: f32 = 14.0;

// The inspector: the column beside the chart listing the selected node's own
// settings, so a card is edited where it sits rather than by leaving for the
// outline.
const INSPECT_W: f32 = 240.0;
const INSPECT_HEAD_H: f32 = 22.0;
const INSPECT_TINT: [f32; 4] = [0.10, 0.10, 0.13, 1.0];

const NEW_TINT: [f32; 4] = [0.20, 0.44, 0.30, 1.0];
const DEL_TINT: [f32; 4] = [0.44, 0.22, 0.24, 1.0];
// The removal chip once it is armed: hotter than its idle tint, so a chip
// waiting on a second press does not read like one that has done nothing.
const CONFIRM_TINT: [f32; 4] = [0.68, 0.26, 0.28, 1.0];
const DROP_TINT: [f32; 4] = [0.09, 0.09, 0.12, 1.0];
const ERROR_TINT: [f32; 4] = [0.20, 0.09, 0.11, 1.0];
const ERROR_BORDER: [f32; 4] = [0.52, 0.24, 0.27, 1.0];

// The chrome above the body: the title bar, the header, and the toolbar.
const CHROME_H: f32 = TITLE_H + 2.0 * BODY_PAD + 2.0 * CTRL_H + 2.0 * GAP;

// The lists, values, and canvas shapes the panel shows.
const ROWS: &str = "editor.behavior.rows";
const FIELDS: &str = "editor.behavior.fields";
const PICKS: &str = "editor.behavior.picks";
pub(crate) const SHAPES: &str = "editor.behavior.chart";
const PICKED: &str = "editor.behavior.picked";
const ORDINAL: &str = "editor.behavior.ordinal";
const VIEW_CAPTION: &str = "editor.behavior.view";
const REMOVE_CAPTION: &str = "editor.behavior.remove_caption";
const INSPECT_CAPTION: &str = "editor.behavior.inspect";
const STATUS: &str = "editor.behavior.status";
// What the lists last showed, so a steady frame rebuilds none.
const ROWS_SHOWN: [&str; 2] = ["editor.behavior.rows.hi", "editor.behavior.rows.lo"];
const FIELDS_SHOWN: [&str; 2] = ["editor.behavior.fields.hi", "editor.behavior.fields.lo"];
const PICKS_SHOWN: [&str; 2] = ["editor.behavior.picks.hi", "editor.behavior.picks.lo"];

// The events the panel's controls send.
const STEP: &str = "editor.behavior.step";
const NEW: &str = "editor.behavior.new";
const REMOVE: &str = "editor.behavior.remove";
const VIEW: &str = "editor.behavior.view_toggle";
const PICK: &str = "editor.behavior.pick";
const DELETE: &str = "editor.behavior.delete";
const MOVE: &str = "editor.behavior.move";
const DUPLICATE: &str = "editor.behavior.duplicate";
const SELECT: &str = "editor.behavior.select";
const SELECT_FIELD: &str = "editor.behavior.select_field";
const CARD: &str = "editor.behavior.card";
const CHOOSE: &str = "editor.behavior.choose";
const FAULT: &str = "editor.behavior.fault";
const DISMISS: &str = "editor.behavior.dismiss";

// The panel's slots, in the order the build fills them.
#[derive(Debug, Clone, Copy)]
enum Slot {
    Prev,
    Next,
    Name,
    Hint,
    Ordinal,
    View,
    Remove,
    Pick,
    Delete,
    Up,
    Down,
    Dup,
    Value,
    Outline,
    ChartRow,
    Canvas,
    Inspector,
    InspectHead,
    Fields,
    Status,
    Palette,
    PaletteBox,
    Filter,
    Picks,
}

const SLOTS: usize = Slot::Picks as usize + 1;

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
    pub(crate) fn other(self) -> Self {
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
    pub(crate) fn message(text: impl Into<String>) -> Self {
        Self::Error {
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

// A resolved Behavior-panel request: a press on one of its controls, or a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BehaviorAction {
    // Open the previous / next behavior in the world.
    Step(i32),
    // Append a blank Behavior entry and open it.
    New,
    // Arm the open behavior's removal, or carry out an armed one.
    Remove,
    // Select outline row `i` (an absolute index).
    Select(usize),
    // Select the inspector's row `i`: its `i`th field of the selected node.
    SelectField(usize),
    // Open (or close) the selected row's palette.
    Palette,
    // Insert the option the palette shows at `i` among those the filter keeps.
    Choose(usize),
    // Close the open palette without picking.
    Dismiss,
    // Remove the selected list member.
    Delete,
    // Move the selected list member one place earlier / later.
    Move(i32),
    // Step to the next view.
    ToggleView,
    // A press on the card at `i` of the chart showing.
    Card(usize),
    // Select the chart card at `i` (an index into the chart's cards).
    SelectCard(usize),
    // Open the behavior the overview card at `i` stands for.
    OpenCard(usize),
    // Open the world's variable table on the variable card at `i`.
    OpenVariable(usize),
    // Select the world asset the overview card at `i` stands for.
    SelectAsset(usize),
    // Select whatever the checker is complaining about.
    GoToFault,
    // Hold the selected list member for a later paste.
    Copy,
    // Put what is held into the list the selection addresses.
    Paste,
    // Put a copy of the selected member beside it, leaving what is held alone.
    Duplicate,
}

// The action a Behavior event asks for, if `event` is one.
pub(crate) fn action(event: &UiEvent) -> Option<BehaviorAction> {
    let index = || match event.value {
        Some(UiValue::Int(i)) => usize::try_from(i).ok(),
        _ => None,
    };
    let delta = || match event.value {
        Some(UiValue::Int(i)) => Some(i),
        _ => None,
    };
    Some(match event.name.as_str() {
        STEP => BehaviorAction::Step(delta()?),
        NEW => BehaviorAction::New,
        REMOVE => BehaviorAction::Remove,
        VIEW => BehaviorAction::ToggleView,
        PICK => BehaviorAction::Palette,
        DELETE => BehaviorAction::Delete,
        MOVE => BehaviorAction::Move(delta()?),
        DUPLICATE => BehaviorAction::Duplicate,
        SELECT => BehaviorAction::Select(index()?),
        SELECT_FIELD => BehaviorAction::SelectField(index()?),
        CARD => BehaviorAction::Card(index()?),
        CHOOSE => BehaviorAction::Choose(index()?),
        FAULT => BehaviorAction::GoToFault,
        DISMISS => BehaviorAction::Dismiss,
        _ => return None,
    })
}

// What reaching the overview card at `i` does: open the behavior it stands
// for, open the table declaring its variable, or select the world asset it
// addresses -- which is how the map of places follows it. A trigger and a name
// the world does not declare stand for nothing to reach.
pub(crate) fn overview_action(card: &Card, i: usize) -> Option<BehaviorAction> {
    Some(match card {
        c if c.behavior.is_some() => BehaviorAction::OpenCard(i),
        c if c.kind == CardKind::Variable => BehaviorAction::OpenVariable(i),
        c if c.handle.is_some() => BehaviorAction::SelectAsset(i),
        _ => return None,
    })
}

// Where the panel sits until the user drags it: left of center below the top
// bar, clear of the Story and Import anchors.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [
        ((vw - BEHAVIOR_W) * 0.5 - 40.0).max(8.0),
        crate::editor::hud::body_top() + 24.0,
    ]
}

pub(crate) fn size() -> [f32; 2] {
    [
        BEHAVIOR_W,
        panel_height(2.0 * CTRL_H + 2.0 * GAP + OUTLINE_ROWS as f32 * ROW_H),
    ]
}

// The chart wants width where the outline wants rows: cards run left to right,
// so the panel opens wide enough to hold a trigger, a node, and the card that
// appends after it beside the inspector. A panel cannot size itself against the
// viewport, so the default has to fit a small window outright.
pub(crate) fn chart_size() -> [f32; 2] {
    [
        chart::width_for(3) + INSPECT_W + GAP + 2.0 * BODY_PAD,
        size()[1],
    ]
}

// The overview has no inspector, so its columns take the whole width.
pub(crate) fn overview_size() -> [f32; 2] {
    [chart::width_for(4) + 2.0 * BODY_PAD, size()[1]]
}

// The chart canvas's size at panel size `s`, which is all the pan maths needs.
pub(crate) fn chart_canvas(s: [f32; 2], mode: ViewMode) -> [f32; 2] {
    let inspector = match mode {
        ViewMode::Chart => INSPECT_W + GAP,
        _ => 0.0,
    };
    [
        (s[0] - 2.0 * BODY_PAD - inspector).max(0.0),
        (s[1] - CHROME_H).max(0.0),
    ]
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let mut slots = [AssetId(0); SLOTS];
    let mut keep = |slot: Slot, child: &LayoutChild| -> Option<()> {
        slots[slot as usize] = child.item?.id();
        Some(())
    };
    let style = ui.style();
    let tinted = |tint: [f32; 4]| WidgetStyle {
        background: tint,
        ..style
    };

    let prev = ui.button("<", STEP, Some(UiValue::Int(-1)), [STEP_W, CTRL_H])?;
    keep(Slot::Prev, &prev)?;
    let next = ui.button(">", STEP, Some(UiValue::Int(1)), [STEP_W, CTRL_H])?;
    keep(Slot::Next, &next)?;
    let mut name = ui.text_field(TextField {
        placeholder: "name".to_string(),
        height: CTRL_H,
        ..Default::default()
    })?;
    name.width = LayoutSize::Fixed(NAME_W);
    keep(Slot::Name, &name)?;
    let hint = ui.label("no behaviors yet -- press New", theme::LABEL_DIM)?;
    keep(Slot::Hint, &hint)?;
    let mut ordinal = ui.bound_label(ORDINAL, theme::LABEL_DIM, TextAlign::Left)?;
    ordinal.width = LayoutSize::Fixed(ORDINAL_W);
    keep(Slot::Ordinal, &ordinal)?;
    let gap = ui.layout(Layout::default())?;
    let view = ui.button_with(
        Button {
            text_binding: Some(Binding::State(VIEW_CAPTION.to_string())),
            action: Some(emit(VIEW, None)),
            ..Default::default()
        },
        [VIEW_W, CTRL_H],
    )?;
    keep(Slot::View, &view)?;
    let remove = ui.button_with(
        Button {
            text_binding: Some(Binding::State(REMOVE_CAPTION.to_string())),
            action: Some(emit(REMOVE, None)),
            ..Default::default()
        },
        [REMOVE_W, CTRL_H],
    )?;
    keep(Slot::Remove, &remove)?;
    let new = ui.button("New", NEW, None, [NEW_W, CTRL_H])?;
    if let Some(b) = new.item.and_then(|r| ui.get_mut::<Button>(r.id())) {
        b.style = tinted(NEW_TINT);
    }
    let header = ui.row(
        vec![
            prev,
            next,
            name,
            hint,
            ordinal,
            fill(gap),
            view,
            remove,
            new,
        ],
        CTRL_H,
    )?;

    let tool = |ui: &mut UiBuilder<'_>, caption: &str, event: &str, value: Option<i32>, w: f32| {
        ui.button(caption, event, value.map(UiValue::Int), [w, CTRL_H])
    };
    let pick = tool(ui, "Pick", PICK, None, TOOL_BTN_W)?;
    keep(Slot::Pick, &pick)?;
    let delete = tool(ui, "Del", DELETE, None, TOOL_BTN_W)?;
    keep(Slot::Delete, &delete)?;
    if let Some(b) = delete.item.and_then(|r| ui.get_mut::<Button>(r.id())) {
        b.style = tinted(DEL_TINT);
    }
    let up = tool(ui, "^", MOVE, Some(-1), ARROW_W)?;
    keep(Slot::Up, &up)?;
    let down = tool(ui, "v", MOVE, Some(1), ARROW_W)?;
    keep(Slot::Down, &down)?;
    let dup = tool(ui, "Dup", DUPLICATE, None, TOOL_BTN_W)?;
    keep(Slot::Dup, &dup)?;
    let value = ui.text_field(TextField {
        placeholder: "value".to_string(),
        height: CTRL_H,
        ..Default::default()
    })?;
    keep(Slot::Value, &value)?;
    let toolbar = ui.row(vec![pick, delete, up, down, dup, value], CTRL_H)?;

    let outline = ui.list(List {
        items: ROWS.to_string(),
        row_height: ROW_H,
        indent: INDENT,
        on_select: SELECT.to_string(),
        ..Default::default()
    })?;
    keep(Slot::Outline, &outline)?;
    let chart_row = chart_row(ui, &mut keep)?;
    keep(Slot::ChartRow, &chart_row)?;
    let status = status(ui)?;
    keep(Slot::Status, &status)?;
    let palette = palette(ui, &mut keep)?;
    keep(Slot::Palette, &palette)?;
    let body = ui.layout(Layout {
        direction: LayoutDirection::Stack,
        children: vec![outline, chart_row, status, palette],
        ..Default::default()
    })?;

    let spec = PanelSpec {
        key: PanelKey::Behavior,
        title: "Behavior",
        title_state: None,
        size: size(),
        min_size: Some(size()),
    };
    let mut panel = ui.panel(&spec, vec![header, toolbar, fill(body)])?;
    panel.slots = slots.to_vec();
    Some(panel)
}

// The chart's canvas and the inspector beside it.
fn chart_row(
    ui: &mut UiBuilder<'_>,
    keep: &mut impl FnMut(Slot, &LayoutChild) -> Option<()>,
) -> Option<LayoutChild> {
    let canvas = ui.canvas(Canvas {
        shapes: SHAPES.to_string(),
        on_press: CARD.to_string(),
        pannable: true,
        ..Default::default()
    })?;
    keep(Slot::Canvas, &canvas)?;
    let mut head = ui.bound_label(INSPECT_CAPTION, theme::HEADING, TextAlign::Left)?;
    head.height = LayoutSize::Fixed(INSPECT_HEAD_H);
    keep(Slot::InspectHead, &head)?;
    let fields = ui.list(List {
        items: FIELDS.to_string(),
        row_height: ROW_H,
        indent: INDENT,
        on_select: SELECT_FIELD.to_string(),
        ..Default::default()
    })?;
    keep(Slot::Fields, &fields)?;
    let column = ui.layout(Layout {
        direction: LayoutDirection::Column,
        children: vec![head, fields],
        padding: GAP,
        ..Default::default()
    })?;
    let backing = ui.sprite(Sprite {
        tint: INSPECT_TINT,
        corner_radius: theme::CONTROL_RADIUS,
        ..Default::default()
    })?;
    let inspector = ui.layout(Layout {
        direction: LayoutDirection::Stack,
        children: vec![fill(backing), fill(column)],
        ..Default::default()
    })?;
    let inspector = LayoutChild {
        item: Some(Ref::new(inspector)),
        width: LayoutSize::Fixed(INSPECT_W),
        height: LayoutSize::Fill(1.0),
    };
    keep(Slot::Inspector, &inspector)?;
    let row = ui.layout(Layout {
        direction: LayoutDirection::Row,
        children: vec![canvas, inspector],
        gap: GAP,
        ..Default::default()
    })?;
    Some(fill(row))
}

// The checker's complaint, at the body's foot: pressing it goes to what it is
// about, which is the only way to reach a fault that is off screen.
fn status(ui: &mut UiBuilder<'_>) -> Option<LayoutChild> {
    let button = ui.button_with(
        Button {
            text_binding: Some(Binding::State(STATUS.to_string())),
            action: Some(emit(FAULT, None)),
            ..Default::default()
        },
        [0.0, STATUS_H],
    )?;
    if let Some(b) = button.item.and_then(|r| ui.get_mut::<Button>(r.id())) {
        b.style.background = ERROR_TINT;
        b.style.border = ERROR_BORDER;
        b.style.text = theme::LOG_ERROR;
    }
    let button = LayoutChild {
        width: LayoutSize::Fill(1.0),
        ..button
    };
    let column = ui.layout(Layout {
        direction: LayoutDirection::Column,
        children: vec![button],
        justify: LayoutAlign::End,
        ..Default::default()
    })?;
    Some(fill(column))
}

// The palette, floating over the top of the body: a field narrowing what it
// offers over one row per kept option with its hint.
fn palette(
    ui: &mut UiBuilder<'_>,
    keep: &mut impl FnMut(Slot, &LayoutChild) -> Option<()>,
) -> Option<LayoutChild> {
    let filter = ui.text_field(TextField {
        placeholder: "filter".to_string(),
        height: FILTER_H,
        ..Default::default()
    })?;
    keep(Slot::Filter, &filter)?;
    let picks = ui.list(List {
        items: PICKS.to_string(),
        selected: Some(Binding::State(PICKED.to_string())),
        row_height: PICK_ROW_H,
        on_select: CHOOSE.to_string(),
        ..Default::default()
    })?;
    keep(Slot::Picks, &picks)?;
    let column = ui.layout(Layout {
        direction: LayoutDirection::Column,
        children: vec![filter, picks],
        gap: GAP,
        padding: GAP,
        ..Default::default()
    })?;
    let backing = ui.sprite(Sprite {
        tint: DROP_TINT,
        corner_radius: theme::CONTROL_RADIUS,
        ..Default::default()
    })?;
    // A press away from the palette closes it, whatever it lands on.
    let boxed = ui.layout(Layout {
        direction: LayoutDirection::Stack,
        children: vec![fill(backing), fill(column)],
        on_dismiss: DISMISS.to_string(),
        ..Default::default()
    })?;
    let boxed = LayoutChild {
        item: Some(Ref::new(boxed)),
        width: LayoutSize::Fill(1.0),
        height: LayoutSize::Fixed(palette_height(PICK_ROWS)),
    };
    keep(Slot::PaletteBox, &boxed)?;
    let column = ui.layout(Layout {
        direction: LayoutDirection::Column,
        children: vec![boxed],
        ..Default::default()
    })?;
    Some(fill(column))
}

// How tall the palette is showing `rows` options.
fn palette_height(rows: usize) -> f32 {
    FILTER_H + rows as f32 * PICK_ROW_H + 3.0 * GAP
}

// The text fields' ids: the name, the value, and the palette's filter.
pub(crate) fn fields(built: &BuiltPanel) -> [Option<AssetId>; 3] {
    [Slot::Name, Slot::Value, Slot::Filter].map(|s| built.slots.get(s as usize).copied())
}

// The chart canvas's id.
pub(crate) fn canvas_of(built: &BuiltPanel) -> Option<AssetId> {
    built.slots.get(Slot::Canvas as usize).copied()
}

// Which of the three fields hold the keyboard, as `[name, value, filter]`.
pub(crate) fn focused(world: &World, built: &BuiltPanel) -> [bool; 3] {
    fields(built).map(|id| {
        id.and_then(|id| world.get_by_id::<TextField>(id))
            .is_some_and(|f| f.focused)
    })
}

// What the panel shows this frame.
#[expect(
    clippy::struct_excessive_bools,
    reason = "a per-frame snapshot of independent panel states"
)]
pub(crate) struct BehaviorView<'a> {
    // The open behavior's place among the world's behaviors, and how many
    // there are (zero shows the empty-world prompt instead of a body).
    pub index: usize,
    pub total: usize,
    // The open behavior as one outline, and the row selected in it.
    pub rows: &'a [Row],
    pub selected: Option<usize>,
    pub mode: ViewMode,
    // The open behavior's chart, the card the selection belongs to, and the
    // rows the inspector lists for it.
    pub chart: &'a Chart,
    pub card: Option<usize>,
    pub fields: &'a [usize],
    // The selected row's palette while `picking`, the options its filter keeps
    // (indices into `picks`, best first), and the one the keyboard is on.
    pub picks: &'a [Pick],
    pub matches: &'a [usize],
    pub picking: bool,
    pub pick: usize,
    // Whether the selected row takes a typed value, and which fields hold the
    // keyboard.
    pub editable: bool,
    pub focus: bool,
    pub name_focus: bool,
    pub filter_focus: bool,
    pub remove_armed: bool,
    pub status: Option<&'a Status>,
    // The row the checker's complaint is about, marked wherever it shows rather
    // than jumped to.
    pub fault_row: Option<usize>,
    // Rows whose node just executed. Empty outside a live session.
    pub pulse_rows: &'a [(usize, f32)],
    // Bumped whenever the rows, the inspector's fields, or the palette change.
    pub revision: u64,
    // The outline row to bring into view, and whether the palette's highlight
    // moved, this frame.
    pub reveal: Option<usize>,
    pub reveal_pick: bool,
    pub pan: [f32; 2],
}

impl BehaviorView<'_> {
    fn palette_open(&self) -> bool {
        self.picking && !self.picks.is_empty()
    }

    // What changes the outline's rows: their content, the selection, the
    // fault, and which rows pulse.
    fn rows_key(&self) -> u64 {
        let mut key = self.revision;
        let mut mix = |v: u64| key = key.rotate_left(7) ^ v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        mix(self.selected.map_or(0, |i| i as u64 + 1));
        mix(self.fault_row.map_or(0, |i| i as u64 + 1));
        for &(i, _) in self.pulse_rows {
            mix(i as u64 + 1);
        }
        key
    }
}

// Structure reads dimmer than the things that carry a value, so the shape of a
// body is legible before any of its detail is.
fn label_color(kind: &Kind) -> [f32; 3] {
    match kind {
        Kind::Node | Kind::Source => theme::HEADING,
        Kind::List(_) => theme::LABEL_DIM,
        _ => theme::LABEL,
    }
}

// Row `i` of the outline as a list item, indented from `base` depth: its label
// and value, colored for a fault or a pulse, and marked when it is selected.
fn row_item(view: &BehaviorView<'_>, i: usize, base: usize) -> Option<UiItem> {
    let row = view.rows.get(i)?;
    let color = if view.fault_row == Some(i) {
        theme::LOG_ERROR
    } else if view.pulse_rows.iter().any(|&(r, _)| r == i) {
        pulse::PULSE_TINT
    } else {
        label_color(&row.kind)
    };
    Some(UiItem {
        detail: row.value.clone(),
        depth: u16::try_from(row.depth.saturating_sub(base)).unwrap_or(u16::MAX),
        color: Some(color),
        selected: view.selected == Some(i),
        ..UiItem::new(&row.label)
    })
}

// The palette's kept options, or one row saying the query answers nothing.
fn pick_items(view: &BehaviorView<'_>) -> Vec<UiItem> {
    if view.matches.is_empty() {
        return vec![UiItem {
            dimmed: true,
            ..UiItem::new("no option matches")
        }];
    }
    view.matches
        .iter()
        .filter_map(|&i| view.picks.get(i))
        .map(|p: &Pick| UiItem {
            detail: p.hint.clone(),
            ..UiItem::new(edit::pick_caption(p))
        })
        .collect()
}

// Show the model: which parts show for the view, the lists rebuilt only when
// they changed, and the controls the selection allows.
pub(crate) fn sync(view: &BehaviorView<'_>, world: &mut World, built: &BuiltPanel) {
    let slot = |s: Slot| built.slots.get(s as usize).copied();
    let any = view.total > 0;
    let on = any && view.mode != ViewMode::Overview;
    let row = view.selected.and_then(|i| view.rows.get(i)).filter(|_| on);
    let member = row.is_some_and(|r| r.element.is_some());
    let disable = |world: &mut World, s: Slot, live: bool| {
        if let Some(b) = slot(s).and_then(|id| world.get_mut_by_id::<Button>(id)) {
            state::set_if_changed(&mut b.disabled, !live);
        }
    };
    for (s, live) in [
        (Slot::Prev, any),
        (Slot::Next, any),
        (Slot::View, any),
        (Slot::Remove, any),
        (Slot::Pick, row.is_some() && !view.picks.is_empty()),
        (Slot::Delete, member),
        (Slot::Up, member),
        (Slot::Down, member),
        (Slot::Dup, member),
    ] {
        disable(world, s, live);
    }
    let show = |world: &mut World, s: Slot, shown: bool| set_shown(world, slot(s), shown);
    show(world, Slot::Hint, !any);
    show(world, Slot::Ordinal, any);
    show(world, Slot::Outline, any && view.mode == ViewMode::Outline);
    show(world, Slot::ChartRow, any && view.mode.drawn_as_chart());
    show(world, Slot::Inspector, view.mode == ViewMode::Chart);
    let error = view.status.and_then(Status::error).filter(|_| any);
    show(world, Slot::Status, error.is_some());
    show(world, Slot::Palette, on && view.palette_open());

    state::set_text(
        world,
        ORDINAL,
        format_args!("{}/{}", view.index + 1, view.total),
    );
    state::set_text(world, VIEW_CAPTION, format_args!("{}", view.mode.caption()));
    let (caption, tint) = if view.remove_armed && any {
        ("Confirm", CONFIRM_TINT)
    } else {
        ("Remove", DEL_TINT)
    };
    state::set_text(world, REMOVE_CAPTION, format_args!("{caption}"));
    if let Some(b) = slot(Slot::Remove).and_then(|id| world.get_mut_by_id::<Button>(id)) {
        state::set_if_changed(&mut b.style.background, tint);
    }
    state::set_text(world, STATUS, format_args!("{}", error.unwrap_or("")));

    sync_fields(view, world, built, on);
    sync_rows(view, world, built);
    sync_inspector(view, world, built);
    sync_palette(view, world, built);
    if let Some(c) = slot(Slot::Canvas).and_then(|id| world.get_mut_by_id::<Canvas>(id)) {
        state::set_if_changed(&mut c.pan, view.pan);
    }
}

fn set_shown(world: &mut World, id: Option<AssetId>, shown: bool) {
    let Some(id) = id else {
        return;
    };
    if let Some(l) = world.get_mut_by_id::<Layout>(id) {
        state::set_if_changed(&mut l.visible, shown);
    } else if let Some(l) = world.get_mut_by_id::<List>(id) {
        state::set_if_changed(&mut l.visible, shown);
    } else if let Some(l) = world.get_mut_by_id::<TextLabel>(id) {
        state::set_if_changed(&mut l.visible, shown);
    }
}

// The name field shows while there is a behavior to name, the value field while
// the selected row takes one, and the filter while the palette is up.
fn sync_fields(view: &BehaviorView<'_>, world: &mut World, built: &BuiltPanel, on: bool) {
    let [name, value, filter] = fields(built);
    let any = view.total > 0;
    for (id, shown, focus) in [
        (name, any, view.name_focus),
        (value, on && view.editable, view.focus),
        (filter, view.palette_open(), view.filter_focus),
    ] {
        if let Some(f) = id.and_then(|id| world.get_mut_by_id::<TextField>(id)) {
            state::set_if_changed(&mut f.visible, shown);
            state::set_if_changed(&mut f.focused, shown && focus);
        }
    }
}

fn sync_rows(view: &BehaviorView<'_>, world: &mut World, built: &BuiltPanel) {
    let key = view.rows_key();
    if !state::is_shown(world, ROWS_SHOWN, key) {
        let items: Vec<UiItem> = (0..view.rows.len())
            .filter_map(|i| row_item(view, i, 0))
            .collect();
        state::with_state(world, |s| {
            s.set_list(ROWS, items);
            state::mark_shown(s, ROWS_SHOWN, key);
        });
    }
    if let Some(i) = view.reveal
        && let Some(list) = built
            .slots
            .get(Slot::Outline as usize)
            .and_then(|&id| world.get_mut_by_id::<List>(id))
    {
        list.reveal(i, view.rows.len());
    }
}

// The inspector: the selected node's own settings, indented from that node, so
// the toolbar and the palette act on them exactly as on the outline's rows.
fn sync_inspector(view: &BehaviorView<'_>, world: &mut World, built: &BuiltPanel) {
    let key = view.rows_key();
    if !state::is_shown(world, FIELDS_SHOWN, key) {
        let base = view
            .fields
            .first()
            .and_then(|&i| view.rows.get(i))
            .map_or(0, |r| r.depth);
        let items: Vec<UiItem> = view
            .fields
            .iter()
            .filter_map(|&i| row_item(view, i, base))
            .collect();
        state::with_state(world, |s| {
            s.set_list(FIELDS, items);
            state::mark_shown(s, FIELDS_SHOWN, key);
        });
    }
    let card = view.card.and_then(|i| view.chart.cards.get(i));
    match card {
        Some(c) => state::set_text(world, INSPECT_CAPTION, format_args!("{}", c.title)),
        None => state::set_text(world, INSPECT_CAPTION, format_args!("select a card")),
    }
    let color = if card.is_some() {
        theme::HEADING
    } else {
        theme::LABEL_DIM
    };
    if let Some(l) = built
        .slots
        .get(Slot::InspectHead as usize)
        .and_then(|&id| world.get_mut_by_id::<TextLabel>(id))
    {
        state::set_if_changed(&mut l.color, color);
    }
}

fn sync_palette(view: &BehaviorView<'_>, world: &mut World, built: &BuiltPanel) {
    if !view.palette_open() {
        return;
    }
    if !state::is_shown(world, PICKS_SHOWN, view.revision) {
        let items = pick_items(view);
        state::with_state(world, |s| {
            s.set_list(PICKS, items);
            state::mark_shown(s, PICKS_SHOWN, view.revision);
        });
    }
    let pick = i32::try_from(view.pick).unwrap_or(i32::MAX);
    state::with_state(world, |s| s.set(PICKED, UiValue::Int(pick)));
    let shown = view.matches.len().clamp(1, PICK_ROWS);
    set_palette_height(world, built, palette_height(shown));
    if view.reveal_pick
        && let Some(list) = built
            .slots
            .get(Slot::Picks as usize)
            .and_then(|&id| world.get_mut_by_id::<List>(id))
    {
        list.reveal(view.pick, view.matches.len());
    }
}

// Size the palette's box to the options it shows, through the child entry
// that places it.
fn set_palette_height(world: &mut World, built: &BuiltPanel, height: f32) {
    let (Some(column), Some(boxed)) = (
        built.slots.get(Slot::Palette as usize).copied(),
        built.slots.get(Slot::PaletteBox as usize).copied(),
    ) else {
        return;
    };
    if let Some(l) = world.get_mut_by_id::<Layout>(column)
        && let Some(child) = l
            .children
            .iter_mut()
            .find(|c| c.item.is_some_and(|r| r.id() == boxed))
    {
        state::set_if_changed(&mut child.height, LayoutSize::Fixed(height));
    }
}

// The card the checker's complaint sits in, so a complaint about a field lights
// the node carrying it.
pub(crate) fn fault_card(chart: &Chart, rows: &[Row], fault_row: Option<usize>) -> Option<usize> {
    fault_row
        .and_then(|i| rows.get(i))
        .and_then(|r| super::fields::owning_card(&chart.cards, &r.path))
}

#[cfg(test)]
mod tests {
    use concinnity_engine::components::UiState;
    use serde_json::json;

    use super::*;
    use crate::editor::behavior::{graph, outline};

    fn built(world: &mut World) -> BuiltPanel {
        build(&mut UiBuilder::new(world, AssetId(9), None)).expect("the panel builds")
    }

    fn body() -> serde_json::Value {
        json!({"on": "tick", "do": [{"hide": {"target": "self"}}, {"save": {}}]})
    }

    struct Model {
        rows: Vec<Row>,
        chart: Chart,
        picks: Vec<Pick>,
        matches: Vec<usize>,
        fields: Vec<usize>,
    }

    fn model() -> Model {
        let args = body();
        Model {
            rows: outline::rows(&args),
            chart: graph::chart(&args),
            picks: vec![
                Pick {
                    verb: "save",
                    hint: "store the world".into(),
                },
                Pick {
                    verb: "spawn",
                    hint: String::new(),
                },
            ],
            matches: vec![0, 1],
            fields: vec![1, 2],
        }
    }

    fn view(m: &Model, mode: ViewMode) -> BehaviorView<'_> {
        BehaviorView {
            index: 0,
            total: 1,
            rows: &m.rows,
            selected: None,
            mode,
            chart: &m.chart,
            card: None,
            fields: &m.fields,
            picks: &m.picks,
            matches: &m.matches,
            picking: false,
            pick: 0,
            editable: false,
            focus: false,
            name_focus: false,
            filter_focus: false,
            remove_armed: false,
            status: None,
            fault_row: None,
            pulse_rows: &[],
            revision: 1,
            reveal: None,
            reveal_pick: false,
            pan: [0.0, 0.0],
        }
    }

    fn slot(b: &BuiltPanel, s: Slot) -> AssetId {
        b.slots[s as usize]
    }

    fn layout_shown(world: &World, b: &BuiltPanel, s: Slot) -> bool {
        world.get_by_id::<Layout>(slot(b, s)).unwrap().visible
    }

    fn list_shown(world: &World, b: &BuiltPanel, s: Slot) -> bool {
        world.get_by_id::<List>(slot(b, s)).unwrap().visible
    }

    fn items<'w>(world: &'w World, name: &str) -> &'w [UiItem] {
        world.resource::<UiState>().unwrap().list(name)
    }

    #[test]
    fn every_control_sends_its_action() {
        let event = |name: &str, value: Option<i32>| UiEvent {
            name: name.into(),
            element: None,
            value: value.map(UiValue::Int),
        };
        for (e, want) in [
            (event(STEP, Some(-1)), BehaviorAction::Step(-1)),
            (event(NEW, None), BehaviorAction::New),
            (event(REMOVE, None), BehaviorAction::Remove),
            (event(VIEW, None), BehaviorAction::ToggleView),
            (event(PICK, None), BehaviorAction::Palette),
            (event(DELETE, None), BehaviorAction::Delete),
            (event(MOVE, Some(1)), BehaviorAction::Move(1)),
            (event(DUPLICATE, None), BehaviorAction::Duplicate),
            (event(SELECT, Some(3)), BehaviorAction::Select(3)),
            (event(SELECT_FIELD, Some(1)), BehaviorAction::SelectField(1)),
            (event(CARD, Some(2)), BehaviorAction::Card(2)),
            (event(CHOOSE, Some(0)), BehaviorAction::Choose(0)),
            (event(FAULT, None), BehaviorAction::GoToFault),
            (event(DISMISS, None), BehaviorAction::Dismiss),
        ] {
            assert_eq!(action(&e), Some(want), "{}", e.name);
        }
        assert_eq!(action(&event(SELECT, Some(-1))), None, "no row");
        assert_eq!(action(&event("editor.other", None)), None);
    }

    // The outline shows in the outline view, the canvas and the inspector in
    // the chart view, and the canvas alone in the overview.
    #[test]
    fn each_view_shows_its_own_body() {
        let mut world = World::new();
        let b = built(&mut world);
        let m = model();
        sync(&view(&m, ViewMode::Outline), &mut world, &b);
        assert!(list_shown(&world, &b, Slot::Outline));
        assert!(!layout_shown(&world, &b, Slot::ChartRow));

        sync(&view(&m, ViewMode::Chart), &mut world, &b);
        assert!(!list_shown(&world, &b, Slot::Outline));
        assert!(layout_shown(&world, &b, Slot::ChartRow));
        assert!(layout_shown(&world, &b, Slot::Inspector));

        sync(&view(&m, ViewMode::Overview), &mut world, &b);
        assert!(layout_shown(&world, &b, Slot::ChartRow));
        assert!(!layout_shown(&world, &b, Slot::Inspector));
    }

    // Each outline row is its label, indented by depth, with its value beside
    // it; the selection and a fault mark the rows they name.
    #[test]
    fn the_outline_rows_carry_their_values_and_marks() {
        let mut world = World::new();
        let b = built(&mut world);
        let m = model();
        let mut v = view(&m, ViewMode::Outline);
        v.selected = Some(1);
        v.fault_row = Some(2);
        sync(&v, &mut world, &b);
        let rows = items(&world, ROWS);
        assert_eq!(rows.len(), m.rows.len());
        assert_eq!(rows[1].text, m.rows[1].label);
        assert_eq!(rows[1].detail, m.rows[1].value);
        assert_eq!(usize::from(rows[1].depth), m.rows[1].depth);
        assert!(rows[1].selected && !rows[0].selected);
        assert_eq!(rows[2].color, Some(theme::LOG_ERROR));

        // A steady frame leaves the UI state alone; a new selection rebuilds
        // the rows.
        let revision = |world: &World| world.resource::<UiState>().unwrap().revision();
        let before = revision(&world);
        sync(&v, &mut world, &b);
        assert_eq!(revision(&world), before, "a steady frame changed nothing");
        v.selected = Some(0);
        sync(&v, &mut world, &b);
        assert!(items(&world, ROWS)[0].selected);
    }

    // The inspector lists the node's own rows, indented from the node.
    #[test]
    fn the_inspector_lists_the_cards_fields() {
        let mut world = World::new();
        let b = built(&mut world);
        let m = model();
        let mut v = view(&m, ViewMode::Chart);
        v.card = Some(1);
        sync(&v, &mut world, &b);
        let fields = items(&world, FIELDS);
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].depth, 0, "indented from the node itself");
        assert_eq!(fields[0].text, m.rows[1].label);
    }

    // The toolbar acts on the selected row: dimmed with nothing selected, and
    // Pick lit only for a row with a palette.
    #[test]
    fn the_toolbar_follows_the_selection() {
        let mut world = World::new();
        let b = built(&mut world);
        let m = model();
        let disabled =
            |world: &World, s: Slot| world.get_by_id::<Button>(slot(&b, s)).unwrap().disabled;
        sync(&view(&m, ViewMode::Outline), &mut world, &b);
        assert!(disabled(&world, Slot::Pick) && disabled(&world, Slot::Delete));
        let mut v = view(&m, ViewMode::Outline);
        v.selected = Some(1);
        sync(&v, &mut world, &b);
        assert!(!disabled(&world, Slot::Pick));
        assert!(
            !disabled(&world, Slot::Prev),
            "there is a behavior to step from"
        );
    }

    // An armed removal says what the next press does, in a hotter tint.
    #[test]
    fn an_armed_removal_says_confirm() {
        let mut world = World::new();
        let b = built(&mut world);
        let m = model();
        let mut v = view(&m, ViewMode::Outline);
        v.remove_armed = true;
        sync(&v, &mut world, &b);
        let caption = world.resource::<UiState>().unwrap().get(REMOVE_CAPTION);
        assert_eq!(caption, Some(&UiValue::Text("Confirm".into())));
        let remove = world.get_by_id::<Button>(slot(&b, Slot::Remove)).unwrap();
        assert_eq!(remove.style.background, CONFIRM_TINT);
    }

    // The palette shows the kept options with their hints, sized to them, and
    // a query keeping none still says so.
    #[test]
    fn the_palette_shows_what_the_filter_keeps() {
        let mut world = World::new();
        let b = built(&mut world);
        let m = model();
        let mut v = view(&m, ViewMode::Outline);
        v.selected = Some(1);
        v.picking = true;
        sync(&v, &mut world, &b);
        assert!(layout_shown(&world, &b, Slot::Palette));
        let picks = items(&world, PICKS);
        assert_eq!(picks[0].text, "save");
        assert_eq!(picks[0].detail, "store the world");

        let none = Model {
            matches: Vec::new(),
            ..model()
        };
        let mut v = view(&none, ViewMode::Outline);
        v.selected = Some(1);
        v.picking = true;
        v.revision = 2;
        sync(&v, &mut world, &b);
        let picks = items(&world, PICKS);
        assert_eq!(picks.len(), 1);
        assert!(picks[0].dimmed && picks[0].text == "no option matches");
    }

    // A world with no behaviors offers the prompt instead of a name to type.
    #[test]
    fn an_empty_world_offers_the_prompt() {
        let mut world = World::new();
        let b = built(&mut world);
        let m = model();
        let mut v = view(&m, ViewMode::Outline);
        v.total = 0;
        sync(&v, &mut world, &b);
        let hint = world.get_by_id::<TextLabel>(slot(&b, Slot::Hint)).unwrap();
        assert!(hint.visible);
        let [name, _, _] = fields(&b);
        assert!(!world.get_by_id::<TextField>(name.unwrap()).unwrap().visible);
        assert!(!list_shown(&world, &b, Slot::Outline));
    }
}

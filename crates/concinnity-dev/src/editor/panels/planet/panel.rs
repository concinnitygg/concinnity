//! The Planet panel: a status line and Apply button, the tool strip
//! (`tool_strip.rs`), and a scrolling column of the planet's rows (see
//! `mod.rs`). A row is a caption and a text field, a reference picker a click
//! cycles, or an add button, with up to two buttons at its right end. The
//! rows are built again only when what they lay out changes; the hook owns the
//! texts and focus, so a typed edit survives until Apply.

use concinnity_engine::components::{
    Binding, Button as UiButton, LayoutChild, LayoutSize, TextAlign, TextField, UiEvent, UiValue,
};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use super::super::form::FieldKind;
use super::super::registry::PanelKey;
use super::tool_strip::{self, StripAction, StripIds, StripView};
use super::{Button, Row, RowKind};
use crate::editor::theme;
use crate::editor::ui::build::{emit, fill};
use crate::editor::ui::{BuiltPanel, PanelSpec, UiBuilder, panel_height, state};

/// Rows the panel shows before its column scrolls, until it is resized taller.
pub(crate) const VISIBLE_ROWS: usize = 22;

// Buttons a row carries at most.
const ROW_BUTTONS: usize = 2;

pub(crate) const PLANET_W: f32 = 400.0;
const HEADER_H: f32 = 30.0;
const APPLY_W: f32 = 70.0;
const ROW_H: f32 = 26.0;
const ROW_GAP: f32 = 2.0;
const LABEL_COL: f32 = 150.0;
const BUTTON_W: f32 = 58.0;
// Captions indented under a list entry's heading.
const INDENT: f32 = 14.0;
// The tool strip's height with no tool picked.
const STRIP_H: f32 = 44.0;

const SECTION_LABEL: [f32; 3] = [0.70, 0.74, 0.84];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];

// The state the panel shows, and the events its controls send.
const TITLE: &str = "editor.planet.title";
const STATUS: &str = "editor.planet.status";
const APPLY: &str = "editor.planet.apply";
const ADD_PLANET: &str = "editor.planet.add_planet";
const ADD_SEA: &str = "editor.planet.add_sea";
// A reference picker's click, its row the value.
const CYCLE_REF: &str = "editor.planet.ref";
// A row's button, its row and slot the value.
const ROW_BUTTON: &str = "editor.planet.button";
// Each reference picker's shown name, its row after the prefix.
const REF_TEXT: &str = "editor.planet.ref_text.";

// The panel's slots.
const COLUMN_SLOT: usize = 0;
const BRUSH_SLOT: usize = 1;
const PICK_SLOT: usize = 2;

/// What the user asked of the panel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PlanetAction {
    /// Move row `r`'s reference picker to its next option.
    CycleRef(usize),
    /// Slot `slot` of row `row`'s buttons, counted from its right end.
    RowButton { row: usize, slot: usize },
    /// Add a Planet to a world without one.
    AddPlanet,
    /// Give the planet a sea.
    AddSea,
    /// Write every typed box into the planet.
    Apply,
    /// A press on the tool strip.
    Strip(StripAction),
}

fn index(value: Option<&UiValue>) -> Option<usize> {
    match value {
        Some(UiValue::Int(i)) => usize::try_from(*i).ok(),
        _ => None,
    }
}

fn int(i: usize) -> UiValue {
    UiValue::Int(i32::try_from(i).unwrap_or(i32::MAX))
}

/// The action a Planet event asks for, if `event` is one.
pub(crate) fn action(event: &UiEvent) -> Option<PlanetAction> {
    if let Some(a) = tool_strip::action(event) {
        return Some(PlanetAction::Strip(a));
    }
    match event.name.as_str() {
        APPLY => Some(PlanetAction::Apply),
        ADD_PLANET => Some(PlanetAction::AddPlanet),
        ADD_SEA => Some(PlanetAction::AddSea),
        CYCLE_REF => index(event.value.as_ref()).map(PlanetAction::CycleRef),
        ROW_BUTTON => index(event.value.as_ref()).map(|i| PlanetAction::RowButton {
            row: i / ROW_BUTTONS,
            slot: i % ROW_BUTTONS,
        }),
        _ => None,
    }
}

/// Where the panel sits until it is dragged: the window's right edge.
pub(crate) fn default_origin(vp: [f32; 2]) -> [f32; 2] {
    [(vp[0] - PLANET_W - 8.0).max(8.0), 48.0]
}

/// The panel's size showing `rows` rows under an idle tool strip.
pub(crate) fn size(rows: usize) -> [f32; 2] {
    let shown = rows.max(1) as f32;
    [
        PLANET_W,
        panel_height(HEADER_H + STRIP_H + shown * (ROW_H + ROW_GAP)),
    ]
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let status = ui.bound_label(STATUS, ERROR_LABEL, TextAlign::Left)?;
    let apply = ui.button("Apply", APPLY, None, [APPLY_W, HEADER_H - 6.0])?;
    let header = ui.row(vec![status, apply], HEADER_H)?;
    let (strip, strip_ids) = tool_strip::build(ui)?;
    let column = ui.scroll_column(Vec::new(), ROW_GAP)?;
    let mut children = vec![header];
    children.extend(strip);
    children.push(fill(column));
    let spec = PanelSpec {
        key: PanelKey::Planet,
        title: "Planet",
        title_state: Some(TITLE),
        size: size(VISIBLE_ROWS),
        min_size: Some(size(4)),
    };
    let mut panel = ui.panel(&spec, children)?;
    panel.slots = vec![column, strip_ids.brush, strip_ids.pick];
    Some(panel)
}

/// The control a row holds beside its caption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Control {
    None,
    Text,
    Ref,
    // An add button sending this event in place of a caption.
    Add(&'static str),
}

/// What one row lays out: when any row's changes, the rows are built again.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RowShape {
    caption: String,
    control: Control,
    heading: bool,
    indented: bool,
    buttons: [Option<Button>; ROW_BUTTONS],
}

fn control(row: &Row) -> Control {
    match &row.kind {
        RowKind::AddPlanet => Control::Add(ADD_PLANET),
        RowKind::AddSea => Control::Add(ADD_SEA),
        RowKind::Field { field, .. } if matches!(field.kind, FieldKind::Ref { .. }) => Control::Ref,
        RowKind::Field { field, .. } if field.kind.has_text_field() => Control::Text,
        _ => Control::None,
    }
}

// Whether a row sits under a list entry's heading.
fn indented(row: &Row) -> bool {
    row.field().is_some_and(|f| {
        f.key.starts_with(super::List::Covers.key()) || f.key.starts_with(super::List::Layers.key())
    })
}

fn shape(row: &Row) -> RowShape {
    RowShape {
        caption: row.caption.clone(),
        control: control(row),
        heading: matches!(row.kind, RowKind::Header(_)),
        indented: indented(row),
        buttons: row.buttons,
    }
}

/// Whether `rows` lay out as the rows `shapes` were built from.
pub(crate) fn same_layout(shapes: &[RowShape], rows: &[Row]) -> bool {
    shapes.len() == rows.len()
        && shapes.iter().zip(rows).all(|(s, r)| {
            s.caption == r.caption
                && s.control == control(r)
                && s.heading == matches!(r.kind, RowKind::Header(_))
                && s.indented == indented(r)
                && s.buttons == r.buttons
        })
}

/// The rows built into one world: what they lay out, and each row's text
/// field and reference picker state.
#[derive(Debug, Clone, Default)]
pub(crate) struct PlanetIds {
    pub shapes: Vec<RowShape>,
    texts: Vec<Option<AssetId>>,
    refs: Vec<Option<String>>,
}

impl PlanetIds {
    /// Whether the rows lay out as `rows` do.
    pub(crate) fn lays_out(&self, rows: &[Row]) -> bool {
        same_layout(&self.shapes, rows)
    }
}

// Row `r`'s caption and control, and its buttons at its right end.
fn row(
    ui: &mut UiBuilder<'_>,
    r: usize,
    shape: &RowShape,
    ids: &mut PlanetIds,
) -> Option<LayoutChild> {
    let mut children = Vec::new();
    if let Control::Add(event) = shape.control {
        let add = ui.button(&shape.caption, event, None, [0.0, ROW_H - 4.0])?;
        children.push(LayoutChild {
            width: LayoutSize::Fill(1.0),
            ..add
        });
        return ui.row(children, ROW_H);
    }
    if shape.indented {
        let pad = ui.label("", theme::LABEL)?;
        children.push(LayoutChild {
            width: LayoutSize::Fixed(INDENT),
            ..pad
        });
    }
    let color = if shape.heading {
        SECTION_LABEL
    } else {
        theme::LABEL
    };
    let caption_w = LABEL_COL - if shape.indented { INDENT } else { 0.0 };
    let caption = ui.clipped_label(&shape.caption, color, caption_w)?;
    let caption_width = if shape.control == Control::None {
        LayoutSize::Fill(1.0)
    } else {
        LayoutSize::Fixed(caption_w)
    };
    children.push(LayoutChild {
        width: caption_width,
        ..caption
    });
    match shape.control {
        Control::Text => {
            let text = ui.text_field(TextField {
                height: ROW_H - 6.0,
                ..Default::default()
            })?;
            ids.texts[r] = text.item.map(concinnity_engine::ecs::Ref::id);
            children.push(text);
        }
        Control::Ref => {
            let key = format!("{REF_TEXT}{r}");
            let picker = ui.button_with(
                UiButton {
                    text_binding: Some(Binding::State(key.clone())),
                    action: Some(emit(CYCLE_REF, Some(int(r)))),
                    ..Default::default()
                },
                [0.0, ROW_H - 6.0],
            )?;
            ids.refs[r] = Some(key);
            children.push(LayoutChild {
                width: LayoutSize::Fill(1.0),
                ..picker
            });
        }
        Control::None | Control::Add(_) => {}
    }
    for (slot, button) in shape.buttons.iter().enumerate().rev() {
        if let Some(b) = button {
            let pressed = ui.button(
                b.label(),
                ROW_BUTTON,
                Some(int(r * ROW_BUTTONS + slot)),
                [BUTTON_W, ROW_H - 6.0],
            )?;
            children.push(LayoutChild {
                width: LayoutSize::Fixed(BUTTON_W),
                ..pressed
            });
        }
    }
    ui.row(children, ROW_H)
}

/// Build `rows` into the panel's column, replacing what it held.
pub(crate) fn build_rows(world: &mut World, panel: &mut BuiltPanel, rows: &[Row]) -> PlanetIds {
    let mut ids = PlanetIds {
        shapes: rows.iter().map(shape).collect(),
        texts: vec![None; rows.len()],
        refs: vec![None; rows.len()],
    };
    let Some(column) = panel.slots.get(COLUMN_SLOT).copied() else {
        return ids;
    };
    let shapes = ids.shapes.clone();
    crate::editor::ui::rows::rebuild(world, panel, column, |ui| {
        let mut built = Vec::with_capacity(shapes.len());
        for (r, s) in shapes.iter().enumerate() {
            built.push(row(ui, r, s, &mut ids)?);
        }
        Some(built)
    });
    ids
}

/// What the panel shows this frame.
pub(crate) struct PlanetView<'a> {
    pub rows: &'a [Row],
    /// Each row's typed text, empty for one with no text field.
    pub texts: &'a [String],
    pub focus: Option<usize>,
    pub status: Option<&'a str>,
    /// Typed edits are waiting on Apply: the title marks them.
    pub dirty: bool,
    pub strip: StripView<'a>,
}

pub(crate) fn sync(view: &PlanetView<'_>, ids: &PlanetIds, panel: &BuiltPanel, world: &mut World) {
    let dirty = if view.dirty { " *" } else { "" };
    state::set_text(world, TITLE, format_args!("Planet{dirty}"));
    state::set_text(world, STATUS, format_args!("{}", view.status.unwrap_or("")));
    for (r, id) in ids.texts.iter().enumerate() {
        let Some(f) = id.and_then(|id| world.get_mut_by_id::<TextField>(id)) else {
            continue;
        };
        let text = view.texts.get(r).map_or("", String::as_str);
        if f.content != text {
            f.content.clear();
            f.content.push_str(text);
            f.caret = text.chars().count();
        }
        let focused = view.focus == Some(r);
        if f.focused != focused {
            f.focused = focused;
        }
    }
    for (r, key) in ids.refs.iter().enumerate() {
        let (Some(key), Some(field)) = (key, view.rows.get(r).and_then(Row::field)) else {
            continue;
        };
        let shown = field
            .variants
            .get(field.variant_idx)
            .map_or(field.initial.as_str(), String::as_str);
        state::set_text(world, key, format_args!("{shown}"));
    }
    if let (Some(&brush), Some(&pick)) = (panel.slots.get(BRUSH_SLOT), panel.slots.get(PICK_SLOT)) {
        tool_strip::sync(&view.strip, StripIds { brush, pick }, world);
    }
}

/// What was typed into the panel's text fields, and which of them holds the
/// keyboard.
pub(crate) fn observe(world: &World, ids: &PlanetIds, texts: &mut [String]) -> Option<usize> {
    let mut focus = None;
    for (r, id) in ids.texts.iter().enumerate() {
        let Some(f) = id.and_then(|id| world.get_by_id::<TextField>(id)) else {
            continue;
        };
        if let Some(text) = texts.get_mut(r)
            && f.content != *text
        {
            text.clone_from(&f.content);
        }
        if f.focused {
            focus = Some(r);
        }
    }
    focus
}

#[cfg(test)]
mod tests;

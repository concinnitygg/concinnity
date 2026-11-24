//! The `CharacterShape` panel: a status line and Reset / Randomize buttons over
//! a scrolling column of the shape's rows (see `character_shape.rs`): the
//! schema's presets, then each section's heading and a captioned slider per
//! row, or the add row alone while the selected mesh has no shape. The rows are
//! built again only when what they lay out changes; the hook owns the values,
//! the drag preview, and the commit path.

use concinnity_engine::components::{
    Binding, Layout, LayoutChild, LayoutSize, TextAlign, TextLabel, UiEvent, UiValue, ValueSlider,
};
use concinnity_engine::ecs::World;

use super::character_shape::{Row, SliderRow};
use super::registry::PanelKey;
use crate::editor::theme;
use crate::editor::ui::build::fill;
use crate::editor::ui::{BuiltPanel, PanelSpec, UiBuilder, panel_height, state};

// The default (and minimum) panel width; the user can widen it past this.
const SHAPE_W: f32 = 400.0;
// Rows shown before the column scrolls, until the panel is resized taller.
pub(crate) const DEFAULT_ROWS: usize = 18;
const HEADER_H: f32 = 30.0;
const BTN_W: f32 = 86.0;
const BTN_GAP: f32 = 4.0;
const ROW_H: f32 = 24.0;
const ROW_GAP: f32 = 2.0;
const LABEL_COL: f32 = 130.0;
const VALUE_W: f32 = 44.0;

const HEADER_LABEL: [f32; 3] = [0.70, 0.74, 0.84];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];

// The state the panel shows, and the events its controls send.
const TITLE: &str = "editor.shape.title";
const STATUS: &str = "editor.shape.status";
const RESET: &str = "editor.shape.reset";
const RANDOMIZE: &str = "editor.shape.randomize";
const ADD: &str = "editor.shape.add";
const PRESET: &str = "editor.shape.preset";
// Each slider's value, change and commit name carry its slider index after
// the prefix: a slider's events carry its number, not which slider it is.
const VALUE: &str = "editor.shape.value.";
const CHANGE: &str = "editor.shape.change.";
const COMMIT: &str = "editor.shape.commit.";

// The panel's slots.
const STATUS_SLOT: usize = 0;
const BUTTONS_SLOT: usize = 1;
const COLUMN_SLOT: usize = 2;

// What the user asked of the panel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ShapeAction {
    Reset,
    Randomize,
    // Create a shape targeting the selected mesh.
    Add,
    // Apply the schema preset at this index.
    Preset(usize),
    // Slider `slider` moved to `value` while held.
    Change { slider: usize, value: f32 },
    // Slider `slider` was let go at `value`.
    Commit { slider: usize, value: f32 },
}

fn slider_event(event: &UiEvent, prefix: &str) -> Option<(usize, f32)> {
    let slider = event.name.strip_prefix(prefix)?.parse().ok()?;
    match event.value {
        Some(UiValue::Float(value)) => Some((slider, value)),
        _ => None,
    }
}

// The action a Character Shape event asks for, if `event` is one.
pub(crate) fn action(event: &UiEvent) -> Option<ShapeAction> {
    if let Some((slider, value)) = slider_event(event, CHANGE) {
        return Some(ShapeAction::Change { slider, value });
    }
    if let Some((slider, value)) = slider_event(event, COMMIT) {
        return Some(ShapeAction::Commit { slider, value });
    }
    match event.name.as_str() {
        RESET => Some(ShapeAction::Reset),
        RANDOMIZE => Some(ShapeAction::Randomize),
        ADD => Some(ShapeAction::Add),
        PRESET => match event.value {
            Some(UiValue::Int(p)) => usize::try_from(p).ok().map(ShapeAction::Preset),
            _ => None,
        },
        _ => None,
    }
}

// Where the panel sits until the user drags it: below the Lighting panel's
// default anchor on the window's left edge.
pub(crate) fn default_origin() -> [f32; 2] {
    let light = super::lighting_panel::default_origin();
    [light[0], light[1] + 40.0]
}

// The panel's size showing `rows` rows.
pub(crate) fn size(rows: usize) -> [f32; 2] {
    let shown = rows.max(1) as f32;
    [SHAPE_W, panel_height(HEADER_H + shown * (ROW_H + ROW_GAP))]
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let status = ui.bound_label(STATUS, ERROR_LABEL, TextAlign::Left)?;
    let status_id = status.item?.id();
    let reset = ui.button("Reset", RESET, None, [BTN_W, HEADER_H - 6.0])?;
    let random = ui.button("Randomize", RANDOMIZE, None, [BTN_W, HEADER_H - 6.0])?;
    let buttons = ui.row(vec![reset, random], HEADER_H)?;
    let buttons_id = buttons.item?.id();
    let buttons = LayoutChild {
        width: LayoutSize::Fixed(2.0 * BTN_W + BTN_GAP),
        ..buttons
    };
    let header = ui.row(vec![status, buttons], HEADER_H)?;
    let column = ui.scroll_column(Vec::new(), ROW_GAP)?;
    let spec = PanelSpec {
        key: PanelKey::CharacterShape,
        title: "Character Shape",
        title_state: Some(TITLE),
        size: size(DEFAULT_ROWS),
        min_size: Some(size(4)),
    };
    let mut panel = ui.panel(&spec, vec![header, fill(column)])?;
    panel.slots = vec![status_id, buttons_id, column];
    Some(panel)
}

// What the panel's rows lay out: when this changes, the rows are built again.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ShapeLayout {
    pub rows: Vec<Row>,
    // Section captions `Row::Header` indexes.
    pub sections: Vec<String>,
    pub sliders: Vec<SliderRow>,
    // Preset names `Row::Preset` indexes.
    pub presets: Vec<String>,
}

// The rows built into one world: what they lay out, and each slider's value
// state.
#[derive(Debug, Clone, Default)]
pub(crate) struct ShapeIds {
    pub layout: ShapeLayout,
    values: Vec<String>,
}

fn slider_row(ui: &mut UiBuilder<'_>, i: usize, row: &SliderRow) -> Option<LayoutChild> {
    let caption = ui.clipped_label(&row.caption, theme::LABEL, LABEL_COL)?;
    let caption = LayoutChild {
        width: LayoutSize::Fixed(LABEL_COL),
        ..caption
    };
    let (min, max) = row.kind.range();
    let slider = ui.value_slider(
        ValueSlider {
            value: Some(Binding::State(format!("{VALUE}{i}"))),
            min,
            max,
            step: 0.01,
            decimals: 2,
            value_width: VALUE_W,
            on_change: format!("{CHANGE}{i}"),
            on_commit: format!("{COMMIT}{i}"),
            ..Default::default()
        },
        ROW_H,
    )?;
    ui.row(vec![caption, slider], ROW_H)
}

fn wide_button(
    ui: &mut UiBuilder<'_>,
    caption: &str,
    event: &str,
    value: Option<UiValue>,
) -> Option<LayoutChild> {
    let button = ui.button(caption, event, value, [0.0, ROW_H - 2.0])?;
    ui.row(
        vec![LayoutChild {
            width: LayoutSize::Fill(1.0),
            ..button
        }],
        ROW_H,
    )
}

// Build `layout`'s rows into the panel's column, replacing what it held.
pub(crate) fn build_rows(
    world: &mut World,
    panel: &mut BuiltPanel,
    layout: &ShapeLayout,
) -> ShapeIds {
    let ids = ShapeIds {
        layout: layout.clone(),
        values: (0..layout.sliders.len())
            .map(|i| format!("{VALUE}{i}"))
            .collect(),
    };
    let Some(column) = panel.slots.get(COLUMN_SLOT).copied() else {
        return ids;
    };
    crate::editor::ui::rows::rebuild(world, panel, column, |ui| {
        let mut rows = Vec::with_capacity(layout.rows.len());
        for row in &layout.rows {
            rows.push(match *row {
                Row::Header(s) => {
                    let caption = layout.sections.get(s).map_or("", String::as_str);
                    let heading = ui.label(caption, HEADER_LABEL)?;
                    ui.row(vec![heading], ROW_H)?
                }
                Row::PresetHeader => {
                    let heading = ui.label("Presets", HEADER_LABEL)?;
                    ui.row(vec![heading], ROW_H)?
                }
                Row::Preset(p) => {
                    let name = layout.presets.get(p).map_or("", String::as_str);
                    let value = Some(UiValue::Int(i32::try_from(p).unwrap_or(i32::MAX)));
                    wide_button(ui, &format!("apply {name}"), PRESET, value)?
                }
                Row::Add => wide_button(ui, "+ Add CharacterShape", ADD, None)?,
                Row::Slider(i) => slider_row(ui, i, layout.sliders.get(i)?)?,
            });
        }
        Some(rows)
    });
    ids
}

// What the panel shows this frame.
pub(crate) struct ShapeView<'a> {
    // The edited shape's name in the title; `None` while the selected mesh
    // has no shape, which also hides the buttons.
    pub shape: Option<&'a str>,
    pub status: Option<&'a str>,
    // One value per slider row.
    pub values: &'a [f32],
}

pub(crate) fn sync(view: &ShapeView<'_>, ids: &ShapeIds, panel: &BuiltPanel, world: &mut World) {
    match view.shape {
        Some(name) => state::set_text(world, TITLE, format_args!("Character Shape: {name}")),
        None => state::set_text(world, TITLE, format_args!("Character Shape")),
    }
    state::set_text(world, STATUS, format_args!("{}", view.status.unwrap_or("")));
    let has_shape = view.shape.is_some();
    let color = if has_shape {
        ERROR_LABEL
    } else {
        theme::LABEL_DIM
    };
    if let Some(label) = panel
        .slots
        .get(STATUS_SLOT)
        .and_then(|&id| world.get_mut_by_id::<TextLabel>(id))
    {
        state::set_if_changed(&mut label.color, color);
    }
    if let Some(buttons) = panel
        .slots
        .get(BUTTONS_SLOT)
        .and_then(|&id| world.get_mut_by_id::<Layout>(id))
        && buttons.visible != has_shape
    {
        buttons.visible = has_shape;
    }
    state::with_state(world, |s| {
        for (key, &value) in ids.values.iter().zip(view.values) {
            s.set(key, UiValue::Float(value));
        }
    });
}

#[cfg(test)]
mod tests;

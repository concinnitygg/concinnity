//! The Planet panel's tool strip, between its header and its rows: a row of
//! square icon buttons, one per tool, and while a tool is picked its brush
//! settings under them: a slider each for the radius, strength and falloff,
//! and for a paint tool the layer or cover it lays.

use concinnity_engine::components::{
    Binding, Button as UiButton, Layout, LayoutChild, LayoutDirection, LayoutSize, TextAlign,
    UiEvent, UiValue, ValueSlider,
};
use concinnity_engine::ecs::asset_id::AssetId;
use concinnity_engine::ecs::{Ref, World};

use super::List;
use crate::editor::icons;
use crate::editor::planet_tools::{Setting, Tool, Tools};
use crate::editor::theme;
use crate::editor::ui::build::emit;
use crate::editor::ui::{UiBuilder, state};

/// The tools the strip shows.
pub(crate) const TOOLS: usize = Tool::ALL.len();

/// The brush settings with a slider.
pub(crate) const SETTINGS: usize = Setting::ALL.len();

const BUTTON: f32 = 36.0;
const BUTTON_GAP: f32 = 6.0;
const ROW_H: f32 = 26.0;
const LABEL_COL: f32 = 110.0;
const VALUE_W: f32 = 56.0;
const HINT: &str = "[ ] size   Esc";

// The events the strip's controls send, and the state they show.
const TOOL: &str = "editor.planet.tool";
const PICK: &str = "editor.planet.pick";
const CAPTION: &str = "editor.planet.tool_caption";
const PICK_CAPTION: &str = "editor.planet.pick_caption";
const PICK_TEXT: &str = "editor.planet.pick_text";
// Whether each tool is picked, its button showing it pressed.
const PICKED: [&str; TOOLS] = [
    "editor.planet.picked.0",
    "editor.planet.picked.1",
    "editor.planet.picked.2",
    "editor.planet.picked.3",
    "editor.planet.picked.4",
    "editor.planet.picked.5",
    "editor.planet.picked.6",
    "editor.planet.picked.7",
];
// Each setting's slider position: its state and the event a drag sends.
const POSITION: [&str; SETTINGS] = [
    "editor.planet.brush.radius",
    "editor.planet.brush.strength",
    "editor.planet.brush.falloff",
];
// Each setting's value, written beside its slider.
const VALUE: [&str; SETTINGS] = [
    "editor.planet.brush_value.radius",
    "editor.planet.brush_value.strength",
    "editor.planet.brush_value.falloff",
];

/// What a press on the strip asked for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum StripAction {
    /// Pick a tool, or put it down when it is picked.
    Toggle(Tool),
    /// Move a setting's slider to position `t`.
    Slide(Setting, f32),
    /// Move a paint tool's pick to the next entry of its list.
    Cycle,
}

/// The strip's action `event` asks for, if it is one.
pub(crate) fn action(event: &UiEvent) -> Option<StripAction> {
    if let Some(s) = POSITION.iter().position(|p| *p == event.name) {
        return match event.value {
            Some(UiValue::Float(t)) => Some(StripAction::Slide(Setting::ALL[s], t)),
            _ => None,
        };
    }
    match (event.name.as_str(), &event.value) {
        (TOOL, Some(UiValue::Int(i))) => usize::try_from(*i)
            .ok()
            .and_then(|i| Tool::ALL.get(i))
            .map(|t| StripAction::Toggle(*t)),
        (PICK, _) => Some(StripAction::Cycle),
        _ => None,
    }
}

/// The ids of the strip's parts that show and hide.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StripIds {
    // The brush rows, shown while a tool is picked.
    pub brush: AssetId,
    // The paint tool's pick row.
    pub pick: AssetId,
}

fn tool_button(ui: &mut UiBuilder<'_>, i: usize, tool: Tool) -> Option<LayoutChild> {
    let value = Some(UiValue::Int(i32::try_from(i).unwrap_or(i32::MAX)));
    ui.button_with(
        UiButton {
            label: tool.name().to_string(),
            icon: icons::icon(tool.glyph()),
            action: Some(emit(TOOL, value)),
            active: Some(Binding::State(PICKED[i].to_string())),
            ..Default::default()
        },
        [BUTTON, BUTTON],
    )
}

fn fixed_width(mut child: LayoutChild, width: f32) -> LayoutChild {
    child.width = LayoutSize::Fixed(width);
    child
}

fn setting_row(ui: &mut UiBuilder<'_>, s: usize, setting: Setting) -> Option<LayoutChild> {
    let caption = ui.clipped_label(setting.caption(), theme::LABEL, LABEL_COL)?;
    let slider = ui.value_slider(
        ValueSlider {
            value: Some(Binding::State(POSITION[s].to_string())),
            min: 0.0,
            max: 1.0,
            value_width: 0.0,
            on_change: POSITION[s].to_string(),
            ..Default::default()
        },
        ROW_H - 4.0,
    )?;
    let value = ui.bound_label(VALUE[s], theme::LABEL, TextAlign::Right)?;
    ui.row(
        vec![
            fixed_width(caption, LABEL_COL),
            slider,
            fixed_width(value, VALUE_W),
        ],
        ROW_H,
    )
}

/// Build the strip: its buttons row and its brush rows, as children of the
/// panel's column.
pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<(Vec<LayoutChild>, StripIds)> {
    let mut buttons = Vec::with_capacity(TOOLS);
    for (i, tool) in Tool::ALL.into_iter().enumerate() {
        buttons.push(fixed_width(tool_button(ui, i, tool)?, BUTTON));
    }
    let id = ui.stack(LayoutDirection::Row, buttons, BUTTON_GAP, 0.0)?;
    let palette = LayoutChild {
        item: Some(Ref::new(id)),
        width: LayoutSize::Fill(1.0),
        height: LayoutSize::Fixed(BUTTON),
    };

    let caption = ui.bound_label(CAPTION, theme::HEADING, TextAlign::Left)?;
    let hint = ui.label(HINT, theme::LABEL_DIM)?;
    let mut rows = vec![ui.row(vec![caption, hint], ROW_H)?];
    for (s, setting) in Setting::ALL.into_iter().enumerate() {
        rows.push(setting_row(ui, s, setting)?);
    }
    let pick_caption = ui.bound_label(PICK_CAPTION, theme::LABEL, TextAlign::Left)?;
    let pick_button = ui.button_with(
        UiButton {
            label: String::new(),
            text_binding: Some(Binding::State(PICK_TEXT.to_string())),
            action: Some(emit(PICK, None)),
            ..Default::default()
        },
        [0.0, ROW_H - 4.0],
    )?;
    let pick = ui.row(
        vec![
            fixed_width(pick_caption, LABEL_COL),
            LayoutChild {
                width: LayoutSize::Fill(1.0),
                ..pick_button
            },
        ],
        ROW_H,
    )?;
    let pick_id = pick.item?.id();
    rows.push(pick);
    let brush = ui.stack(LayoutDirection::Column, rows, 0.0, 0.0)?;
    let brush_child = LayoutChild {
        item: Some(Ref::new(brush)),
        width: LayoutSize::Fill(1.0),
        height: LayoutSize::Fit,
    };
    Some((
        vec![palette, brush_child],
        StripIds {
            brush,
            pick: pick_id,
        },
    ))
}

/// What the strip shows this frame.
pub(crate) struct StripView<'a> {
    pub tools: &'a Tools,
    /// The paint tool's pick, named.
    pub pick: &'a str,
}

fn set_visible(world: &mut World, id: AssetId, visible: bool) {
    if let Some(l) = world.get_mut_by_id::<Layout>(id)
        && l.visible != visible
    {
        l.visible = visible;
    }
}

/// Bring the strip's controls up to date with `view`.
pub(crate) fn sync(view: &StripView<'_>, ids: StripIds, world: &mut World) {
    let active = view.tools.active;
    for (i, tool) in Tool::ALL.into_iter().enumerate() {
        state::set_bool(world, PICKED[i], active == Some(tool));
    }
    set_visible(world, ids.brush, active.is_some());
    let Some(tool) = active else {
        return;
    };
    let key = tool.key().display_name();
    state::set_text(world, CAPTION, format_args!("{} ({key})", tool.name()));
    let brush = &view.tools.brush;
    for (s, setting) in Setting::ALL.into_iter().enumerate() {
        let t = brush.position(setting);
        state::with_state(world, |st| st.set(POSITION[s], UiValue::Float(t)));
        let v = brush.value(setting);
        match setting {
            Setting::Radius => state::set_text(world, VALUE[s], format_args!("{v:.1} m")),
            Setting::Strength | Setting::Falloff => {
                state::set_text(world, VALUE[s], format_args!("{:.0}%", v * 100.0));
            }
        }
    }
    let list = tool.list();
    set_visible(world, ids.pick, list.is_some());
    if let Some(list) = list {
        let caption = match list {
            List::Layers => "layer",
            List::Covers => "cover",
        };
        state::set_text(world, PICK_CAPTION, format_args!("{caption}"));
        state::set_text(world, PICK_TEXT, format_args!("{}", view.pick));
    }
}

#[cfg(test)]
mod tests;

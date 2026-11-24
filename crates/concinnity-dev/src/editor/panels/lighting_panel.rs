//! The Lighting panel: a status line and Apply button over a scrolling column
//! of themed sections (see `lighting.rs` for the bindings). Each section is a
//! heading, then a row per binding -- a caption and a text field, a checkbox,
//! or a text field with a color swatch -- or, while the world has no asset for
//! the section, a button that adds one. Every row is built once; which of a
//! section's rows show follows the world. The hook owns the texts, focus, and
//! the commit path.

use concinnity_engine::components::{
    Binding, Checkbox, Layout, LayoutChild, LayoutSize, Sprite, TextAlign, TextField, UiEvent,
    UiValue,
};
use concinnity_engine::ecs::asset_id::AssetId;
use concinnity_engine::ecs::{Ref, World};

use super::form::{FieldKind, FormField};
use super::form_panel::swatch_rgb;
use super::lighting::{self, SECTIONS};
use super::registry::PanelKey;
use crate::editor::theme;
use crate::editor::ui::build::{fill, fixed};
use crate::editor::ui::{BuiltPanel, PanelSpec, UiBuilder, panel_height, state};

// The default (and minimum) panel width; the user can widen it past this.
const LIGHT_W: f32 = 380.0;
const HEADER_H: f32 = 30.0;
const BTN_W: f32 = 70.0;
const ROW_H: f32 = 28.0;
const ROW_GAP: f32 = 2.0;
const LABEL_COL: f32 = 150.0;
const SWATCH_W: f32 = 24.0;
// Rows shown before the column scrolls, until the panel is resized taller.
const DEFAULT_ROWS: usize = 14;

const HEADER_LABEL: [f32; 3] = [0.70, 0.74, 0.84];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];

// The state the panel shows, and the events its controls send.
const TITLE: &str = "editor.lighting.title";
const STATUS: &str = "editor.lighting.status";
const APPLY: &str = "editor.lighting.apply";
const ADD: &str = "editor.lighting.add";
// Each checkbox sends its own event, its binding after the prefix: a
// checkbox's event carries its new value, not its binding.
const TOGGLE: &str = "editor.lighting.toggle.";

// The panel's slots.
const COLUMN_SLOT: usize = 1;

// What the user asked of the panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LightingAction {
    // Flip binding `b`'s checkbox (committed immediately by the hook).
    Toggle(usize),
    // Add the missing singleton asset of section `s`.
    Add(usize),
    // Capture every text control, validate per asset, and commit.
    Apply,
}

// The action a Lighting event asks for, if `event` is one.
pub(crate) fn action(event: &UiEvent) -> Option<LightingAction> {
    if let Some(b) = event.name.strip_prefix(TOGGLE) {
        return b.parse().ok().map(LightingAction::Toggle);
    }
    match event.name.as_str() {
        APPLY => Some(LightingAction::Apply),
        ADD => match event.value {
            Some(UiValue::Int(s)) => usize::try_from(s).ok().map(LightingAction::Add),
            _ => None,
        },
        _ => None,
    }
}

// Where the panel sits until the user drags it: the window's left edge, below
// the View panel's default anchor.
pub(crate) fn default_origin() -> [f32; 2] {
    [16.0, crate::editor::hud::body_top() + 280.0]
}

// The size the panel opens at.
pub(crate) fn default_size() -> [f32; 2] {
    size(DEFAULT_ROWS)
}

// The panel's size for `rows` rows: all of them, up to the default window.
fn size(rows: usize) -> [f32; 2] {
    let shown = rows.clamp(1, DEFAULT_ROWS) as f32;
    [LIGHT_W, panel_height(HEADER_H + shown * (ROW_H + ROW_GAP))]
}

// The ids of one binding's row and its controls, and its checkbox's state.
#[derive(Debug, Clone, Default)]
struct BindingRow {
    row: Option<AssetId>,
    text: Option<AssetId>,
    swatch: Option<AssetId>,
    check: Option<String>,
}

// The ids of the rows built into one world, which the panel changes as it
// runs.
#[derive(Debug, Clone, Default)]
pub(crate) struct LightingIds {
    bindings: Vec<BindingRow>,
    // Each section's add button's row.
    adds: Vec<Option<AssetId>>,
}

fn binding_row(
    ui: &mut UiBuilder<'_>,
    b: usize,
    field: &FormField,
) -> Option<(LayoutChild, BindingRow)> {
    let (_, _, caption) = lighting::binding(b);
    let mut ids = BindingRow::default();
    let label = ui.clipped_label(caption, theme::LABEL, LABEL_COL)?;
    let mut children = vec![LayoutChild {
        width: LayoutSize::Fixed(LABEL_COL),
        ..label
    }];
    if field.kind == FieldKind::Bool {
        let event = format!("{TOGGLE}{b}");
        ids.check = Some(event.clone());
        children.push(ui.checkbox(
            Checkbox {
                checked: Some(Binding::State(event.clone())),
                on_change: event,
                width: 0.0,
                ..Default::default()
            },
            ROW_H - 4.0,
        )?);
    } else {
        let text = ui.text_field(TextField {
            height: ROW_H - 4.0,
            ..Default::default()
        })?;
        ids.text = text.item.map(Ref::id);
        children.push(text);
        if matches!(field.kind, FieldKind::Vec { color: true, .. }) {
            let swatch = ui.sprite(Sprite {
                corner_radius: 4.0,
                ..Default::default()
            })?;
            ids.swatch = Some(swatch);
            children.push(fixed(swatch, [SWATCH_W, ROW_H - 8.0]));
        }
    }
    let row = ui.row(children, ROW_H)?;
    ids.row = row.item.map(Ref::id);
    Some((row, ids))
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let status = ui.bound_label(STATUS, ERROR_LABEL, TextAlign::Left)?;
    let status_id = status.item?.id();
    let apply = ui.button("Apply", APPLY, None, [BTN_W, HEADER_H - 4.0])?;
    let header = ui.row(vec![status, apply], HEADER_H)?;
    let column = ui.scroll_column(Vec::new(), ROW_GAP)?;
    let spec = PanelSpec {
        key: PanelKey::Lighting,
        title: "Lighting",
        title_state: Some(TITLE),
        size: default_size(),
        min_size: Some(size(4)),
    };
    let mut panel = ui.panel(&spec, vec![header, fill(column)])?;
    panel.slots = vec![status_id, column];
    Some(panel)
}

// Build every section's rows into the panel's column, once per world.
pub(crate) fn build_rows(world: &mut World, panel: &mut BuiltPanel) -> LightingIds {
    let mut ids = LightingIds::default();
    let Some(column) = panel.slots.get(COLUMN_SLOT).copied() else {
        return ids;
    };
    crate::editor::ui::rows::rebuild(world, panel, column, |ui| {
        let mut rows = Vec::new();
        for (s, section) in SECTIONS.iter().enumerate() {
            let heading = ui.label(section.title, HEADER_LABEL)?;
            rows.push(ui.row(vec![heading], ROW_H)?);
            let fields = lighting::section_fields(section, None);
            let base = lighting::section_base(s);
            for (j, field) in fields.iter().enumerate() {
                let (row, b) = binding_row(ui, base + j, field)?;
                rows.push(row);
                ids.bindings.push(b);
            }
            let value = Some(UiValue::Int(i32::try_from(s).unwrap_or(i32::MAX)));
            let add = ui.button(
                &format!("+ Add {}", section.ty),
                ADD,
                value,
                [0.0, ROW_H - 4.0],
            )?;
            let add = ui.row(
                vec![LayoutChild {
                    width: LayoutSize::Fill(1.0),
                    ..add
                }],
                ROW_H,
            )?;
            ids.adds.push(add.item.map(Ref::id));
            rows.push(add);
        }
        Some(rows)
    });
    ids
}

// What the panel shows this frame.
pub(crate) struct LightingView<'a> {
    // Whether each section's asset is in the world.
    pub present: &'a [bool],
    // Each binding's derived field, `None` while its section's asset is absent.
    pub fields: &'a [Option<FormField>],
    // Each binding's text, empty for one with no text field.
    pub texts: &'a [String],
    pub focus: Option<usize>,
    pub status: Option<&'a str>,
    // Typed-but-not-applied edits exist: the title marks them.
    pub dirty: bool,
}

pub(crate) fn sync(view: &LightingView<'_>, ids: &LightingIds, world: &mut World) {
    let dirty = if view.dirty { " *" } else { "" };
    state::set_text(world, TITLE, format_args!("Lighting{dirty}"));
    state::set_text(world, STATUS, format_args!("{}", view.status.unwrap_or("")));
    let mut b = 0;
    for (s, section) in SECTIONS.iter().enumerate() {
        let present = view.present.get(s).copied().unwrap_or(false);
        if let Some(row) = ids
            .adds
            .get(s)
            .copied()
            .flatten()
            .and_then(|id| world.get_mut_by_id::<Layout>(id))
        {
            state::set_if_changed(&mut row.visible, !present);
        }
        for _ in section.fields {
            let Some(row) = ids.bindings.get(b) else {
                return;
            };
            if let Some(l) = row.row.and_then(|id| world.get_mut_by_id::<Layout>(id)) {
                state::set_if_changed(&mut l.visible, present);
            }
            let text = view.texts.get(b).map_or("", String::as_str);
            if let Some(f) = row.text.and_then(|id| world.get_mut_by_id::<TextField>(id)) {
                if f.content != text {
                    f.content.clear();
                    f.content.push_str(text);
                    f.caret = text.chars().count();
                }
                state::set_if_changed(&mut f.focused, view.focus == Some(b));
            }
            if let Some(sprite) = row.swatch.and_then(|id| world.get_mut_by_id::<Sprite>(id)) {
                state::set_if_changed(&mut sprite.tint, swatch_rgb(text));
            }
            if let (Some(key), Some(field)) =
                (&row.check, view.fields.get(b).and_then(Option::as_ref))
            {
                state::set_bool(world, key, field.boolval);
            }
            b += 1;
        }
    }
}

// Read what the user typed back from the panel's text fields, returning which
// of them holds the keyboard.
pub(crate) fn observe(world: &World, ids: &LightingIds, texts: &mut [String]) -> Option<usize> {
    let mut focus = None;
    for (b, row) in ids.bindings.iter().enumerate() {
        let Some(f) = row.text.and_then(|id| world.get_by_id::<TextField>(id)) else {
            continue;
        };
        if let Some(text) = texts.get_mut(b)
            && f.content != *text
        {
            text.clone_from(&f.content);
        }
        if f.focused {
            focus = Some(b);
        }
    }
    focus
}

#[cfg(test)]
mod tests;

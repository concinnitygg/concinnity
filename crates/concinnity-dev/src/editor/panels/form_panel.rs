//! The editor's add / edit form, in its own floating panel beside the panel it
//! opened from: the asset's editable name and the Apply button, the override
//! summary of a template-derived asset, a status line, and a scrolling column
//! of the type's arg fields and extra rows. The field model is `form.rs` and
//! the hook owns all state; this module builds the widgets, rebuilds the rows
//! when the fields change shape, and shows the hook's model in them.

use concinnity_engine::components::{
    Binding, Button, Checkbox, Layout, LayoutChild, LayoutDirection, LayoutSize, Menu, Sprite,
    TextAlign, TextField, TextLabel, UiEvent, UiItem, UiState, UiValue,
};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use super::form::{self, FieldKind, FormField};
use super::form_extras::{ExtraControl, ExtraRow};
use super::registry::PanelKey;
use crate::editor::overrides::FieldOrigin;
use crate::editor::theme;
use crate::editor::ui::build::{emit, fill, fit, fixed};
use crate::editor::ui::{BuiltPanel, PanelSpec, UiBuilder, panel_height, state};

// An enum / ref field with this many variants or fewer cycles in place on
// click; more opens a menu of them under the field. A reference field's count
// includes its `(none)` option.
pub(crate) const CYCLE_MAX: usize = 5;

// Which of the form's inputs holds keyboard focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FormFocus {
    Name,
    Field(usize),
}

pub(crate) const EDIT_W: f32 = 480.0;
const GAP: f32 = 8.0;
const NAME_H: f32 = 30.0;
const BTN_W: f32 = 84.0;
const STATUS_H: f32 = 22.0;
const FIELD_H: f32 = 32.0;
const LABEL_COL: f32 = 180.0;
// Left inset per nesting level for a flattened nested (dotted-path) leaf.
const NEST_INDENT: f32 = 10.0;
const BAR_W: f32 = 3.0;
const SMALL_BTN: f32 = 26.0;
const SWATCH_W: f32 = 26.0;
// Space between the rows of the scrolling column.
const ROW_GAP: f32 = 2.0;
// A heading or note row's caption runs the width of the column, short of the
// panel's padding and the scrollbar.
const EXTRA_LABEL_W: f32 = EDIT_W - 2.0 * crate::editor::ui::build::BODY_PAD - BAR_W - 16.0;

const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];
const OVERRIDE_LABEL: [f32; 3] = [0.62, 0.76, 1.0];
const OVERRIDE_TINT: [f32; 4] = [0.62, 0.76, 1.0, 1.0];
// Instance-only fields (no template counterpart) mark dimmer.
const INSTANCE_ONLY_TINT: [f32; 4] = [0.55, 0.58, 0.68, 1.0];
const NO_SWATCH: [f32; 4] = [0.15, 0.15, 0.18, 1.0];

// The state the panel shows, and the events its controls send.
const TITLE: &str = "editor.form.title";
const APPLY_TEXT: &str = "editor.form.apply_text";
const STATUS: &str = "editor.form.status";
const CHIP: &str = "editor.form.chip";
const MENU_OPEN: &str = "editor.form.menu_open";
const MENU_ITEMS: &str = "editor.form.menu_items";
const APPLY: &str = "editor.form.apply";
const JUMP: &str = "editor.form.jump";
const ENTITY: &str = "editor.form.entity";
const MENU_PICK: &str = "editor.form.menu_pick";
const CHOICE: &str = "editor.form.choice";
const ADD: &str = "editor.form.add";
const REMOVE: &str = "editor.form.remove";
const EXPAND: &str = "editor.form.expand";
const OVERRIDE: &str = "editor.form.override";
const EXTRA: &str = "editor.form.extra";
// Each checkbox sends its own event, the field's index or the extra row's id
// after the prefix: a checkbox's event carries its new value, not its row.
const CHECK: &str = "editor.form.check.";
const EXTRA_CHECK: &str = "editor.form.extra_check.";

// The panel's slots.
const NAME_SLOT: usize = 0;
const APPLY_SLOT: usize = 1;
const OVERRIDES_SLOT: usize = 2;
const STATUS_SLOT: usize = 3;
const ROWS_SLOT: usize = 4;
const MENU_SLOT: usize = 5;

// Where the panel sits until the user drags it: to the left of the Assets
// panel's default anchor, below the top bar.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [
        vw - super::assets_panel::PANEL_W - GAP - EDIT_W,
        crate::editor::hud::body_top(),
    ]
}

// The panel's size for a form of `rows` fields and extra rows: all of them, up
// to the default window; a longer form scrolls.
pub(crate) fn size(rows: usize) -> [f32; 2] {
    let shown = rows.clamp(1, form::FIELD_POOL);
    [
        EDIT_W,
        panel_height(NAME_H + GAP + STATUS_H + shown as f32 * FIELD_H),
    ]
}

// What the user asked of the form. Field actions carry the field's index,
// extra-row actions the row's id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FormAction {
    // Toggle the bool arg field `i`.
    ToggleField(usize),
    // Advance the enum / ref arg field `i` to its next variant (small sets).
    CycleField(usize),
    // Open (or, if already open, close) the value menu for field `i`.
    OpenFieldDropdown(usize),
    // Pick option `i` from the open field-value menu.
    PickFieldOption(usize),
    // Press the type's extra row with this id.
    PressExtra(usize),
    // Append / drop the last element of array arg field `i`.
    AddArrayElement(usize),
    RemoveArrayElement(usize),
    // Expand / collapse the per-element leaves of non-color vector field `i`.
    ToggleVecExpand(usize),
    // Open (or close) the override menu of the marked field `i`.
    OpenOverrideMenu(usize),
    // Pick option `i` from the open per-field override menu.
    PickOverrideOption(usize),
    // Open (or close) the entity-level override menu in the header.
    OpenEntityMenu,
    // Pick option `i` from the open entity-level menu.
    PickEntityOption(usize),
    // The override summary: scroll to the next overridden field.
    JumpOverride,
    // The Apply / Add button: validate + commit.
    Confirm,
    // Close the panel, discarding the form.
    Close,
}

// Which menu the form has open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FormMenu {
    // The value menu of field `i`.
    Field(usize),
    // The override menu of field `i`.
    Override(usize),
    // The entity-level override menu.
    Entity,
}

// The index after `prefix` in an event's name.
fn suffix(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?.parse().ok()
}

// The action a form event asks for, if `event` is one: a choice resolved
// against `fields`, a menu pick against the menu that is open.
pub(crate) fn action(
    event: &UiEvent,
    fields: &[FormField],
    menu: Option<FormMenu>,
) -> Option<FormAction> {
    let name = event.name.as_str();
    if let Some(j) = suffix(name, CHECK) {
        return Some(FormAction::ToggleField(j));
    }
    if let Some(id) = suffix(name, EXTRA_CHECK) {
        return Some(FormAction::PressExtra(id));
    }
    match name {
        APPLY => return Some(FormAction::Confirm),
        JUMP => return Some(FormAction::JumpOverride),
        ENTITY => return Some(FormAction::OpenEntityMenu),
        _ => {}
    }
    let i = match event.value {
        Some(UiValue::Int(i)) => usize::try_from(i).ok()?,
        _ => return None,
    };
    Some(match name {
        CHOICE => {
            let field = fields.get(i)?;
            if field.variants.len() <= CYCLE_MAX {
                FormAction::CycleField(i)
            } else {
                FormAction::OpenFieldDropdown(i)
            }
        }
        ADD => FormAction::AddArrayElement(i),
        REMOVE => FormAction::RemoveArrayElement(i),
        EXPAND => FormAction::ToggleVecExpand(i),
        OVERRIDE => FormAction::OpenOverrideMenu(i),
        EXTRA => FormAction::PressExtra(i),
        MENU_PICK => match menu? {
            FormMenu::Field(_) => FormAction::PickFieldOption(i),
            FormMenu::Override(_) => FormAction::PickOverrideOption(i),
            FormMenu::Entity => FormAction::PickEntityOption(i),
        },
        _ => return None,
    })
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltPanel> {
    let name = ui.text_field(TextField {
        placeholder: "name".to_string(),
        height: NAME_H,
        scale: 1.05 * theme::TEXT_SCALE,
        ..Default::default()
    })?;
    let name_id = name.item?.id();
    let apply = ui.button_with(
        Button {
            label: "Add".to_string(),
            text_binding: Some(Binding::State(APPLY_TEXT.to_string())),
            action: Some(emit(APPLY, None)),
            ..Default::default()
        },
        [BTN_W, NAME_H],
    )?;
    let apply_id = apply.item?.id();
    let header = ui.row(vec![name, apply], NAME_H)?;
    let chip = ui.button_with(
        Button {
            text_binding: Some(Binding::State(CHIP.to_string())),
            action: Some(emit(JUMP, None)),
            ..Default::default()
        },
        [0.0, SMALL_BTN],
    )?;
    let chip = LayoutChild {
        width: LayoutSize::Fill(1.0),
        ..chip
    };
    let entity = ui.button("...", ENTITY, None, [SMALL_BTN * 1.5, SMALL_BTN])?;
    let overrides = ui.stack(LayoutDirection::Row, vec![chip, entity], 4.0, 0.0)?;
    let overrides_row = LayoutChild {
        height: LayoutSize::Fixed(SMALL_BTN),
        ..fill(overrides)
    };
    let status = ui.bound_label(STATUS, ERROR_LABEL, TextAlign::Left)?;
    let status_id = status.item?.id();
    let status = ui.row(vec![status], STATUS_H)?;
    let rows = ui.scroll_column(ROW_GAP)?;
    let menu = ui.menu(Menu {
        open: Some(Binding::State(MENU_OPEN.to_string())),
        items: MENU_ITEMS.to_string(),
        on_pick: MENU_PICK.to_string(),
        width: 300.0,
        ..Default::default()
    })?;
    let spec = PanelSpec {
        key: PanelKey::Edit,
        title: "New asset",
        title_state: Some(TITLE),
        size: size(form::FIELD_POOL),
        min_size: Some(size(1)),
    };
    let body = vec![header, overrides_row, status, fill(rows)];
    let mut panel = ui.panel(&spec, body)?;
    panel.slots = vec![name_id, apply_id, overrides, status_id, rows, menu];
    Some(panel)
}

// The ids of what one field's row holds, for the parts the form changes as it
// runs.
#[derive(Debug, Clone, Default)]
struct FieldRow {
    text: Option<AssetId>,
    swatch: Option<AssetId>,
    choice: Option<String>,
    check: Option<String>,
    bar: Option<AssetId>,
    marker: Option<AssetId>,
    // The control a value menu opens under.
    anchor: Option<AssetId>,
}

// The ids and state names of an extra row's parts the form changes as it
// runs: its caption, its control, and its detail.
#[derive(Debug, Clone, Default)]
struct ExtraRowIds {
    caption: Option<AssetId>,
    check: Option<(String, AssetId)>,
    choice: Option<String>,
    detail: Option<AssetId>,
}

// The rows built for one shape of the form.
#[derive(Debug, Clone, Default)]
pub(crate) struct ShownForm {
    // The fields' derivation the rows were built for.
    pub(crate) rev: u64,
    // The form the rows were first built for, so rows rebuilt for the same
    // form keep their scroll.
    pub(crate) opened: u64,
    fields: Vec<FieldRow>,
    extras: Vec<ExtraRowIds>,
    // The extra rows' shape: each row's id, control kind, and whether it draws
    // a detail label.
    extra_shape: Vec<ExtraShape>,
}

type ExtraShape = (usize, std::mem::Discriminant<ExtraControl>, bool);

fn extra_shape(row: &ExtraRow) -> ExtraShape {
    (
        row.id,
        std::mem::discriminant(&row.control),
        // A check shows its detail as the checkbox's note, which takes new
        // text in place; the other rows draw it as a label of their own.
        row.detail.is_some() && !matches!(row.control, ExtraControl::Check { .. }),
    )
}

impl ShownForm {
    // Whether the extra rows still have the shape they were built for.
    pub(crate) fn extras_match(&self, extras: &[ExtraRow]) -> bool {
        self.extra_shape.len() == extras.len()
            && self
                .extra_shape
                .iter()
                .zip(extras)
                .all(|(shape, row)| *shape == extra_shape(row))
    }
}

// Whether the non-color vector at field `j` is disclosed: its element leaves
// (`key.0` ..) follow it directly.
fn vec_expanded(fields: &[FormField], j: usize) -> bool {
    let Some(field) = fields.get(j) else {
        return false;
    };
    fields.get(j + 1).is_some_and(|next| {
        next.key
            .strip_prefix(field.key.as_str())
            .is_some_and(|rest| rest.starts_with('.'))
    })
}

// The display caption for a field: a disclosed vector element (`position.0`)
// reads as its axis; every other field keeps its leaf name. An unset leaf of a
// sparse type is tagged, since the value it shows is its default.
fn field_caption(fields: &[FormField], field: &FormField) -> String {
    let caption = element_axis_label(fields, &field.key).unwrap_or_else(|| {
        field
            .key
            .rsplit('.')
            .next()
            .unwrap_or(&field.key)
            .to_string()
    });
    if field.unset {
        format!("{caption} {}", form::AUTO_LABEL)
    } else {
        caption
    }
}

// The axis label for `key` when it is an element leaf of a non-color vector.
fn element_axis_label(fields: &[FormField], key: &str) -> Option<String> {
    let (parent, last) = key.rsplit_once('.')?;
    let idx: usize = last.parse().ok()?;
    let len = fields.iter().find_map(|f| match f.kind {
        FieldKind::Vec { len, color: false } if f.key == parent => Some(len),
        _ => None,
    })?;
    Some(axis_letter(idx, len))
}

// A vector element's label: x / y / z / w in a vector of up to four, and its
// index in a longer one, whose elements are not axes.
fn axis_letter(i: usize, len: usize) -> String {
    match ["x", "y", "z", "w"].get(i) {
        Some(axis) if len <= 4 => (*axis).to_string(),
        _ => format!("[{i}]"),
    }
}

// The color a color field's text describes: its first three numbers, clamped,
// or a neutral placeholder while it holds fewer.
pub(crate) fn swatch_rgb(text: &str) -> [f32; 4] {
    let mut rgb = [0.0_f32; 3];
    let mut n = 0;
    for part in text
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
    {
        let Ok(v) = part.parse::<f32>() else {
            continue;
        };
        rgb[n] = v.clamp(0.0, 1.0);
        n += 1;
        if n == 3 {
            return [rgb[0], rgb[1], rgb[2], 1.0];
        }
    }
    NO_SWATCH
}

// One field's row: its override bar, caption, control, and override marker.
fn field_row(
    ui: &mut UiBuilder<'_>,
    fields: &[FormField],
    j: usize,
) -> Option<(LayoutChild, FieldRow)> {
    let field = &fields[j];
    let index = Some(UiValue::Int(i32::try_from(j).unwrap_or(i32::MAX)));
    let mut ids = FieldRow::default();
    let bar = ui.sprite(Sprite {
        tint: OVERRIDE_TINT,
        corner_radius: 1.5,
        visible: false,
        ..Default::default()
    })?;
    ids.bar = Some(bar);
    let depth = field.key.matches('.').count() as f32;
    let indent = ui.sprite(Sprite {
        tint: [0.0; 4],
        ..Default::default()
    })?;
    let caption = ui.clipped_label(
        &field_caption(fields, field),
        theme::LABEL,
        LABEL_COL - depth * NEST_INDENT,
    )?;
    let mut row = vec![
        fixed(bar, [BAR_W, FIELD_H - 8.0]),
        fixed(indent, [depth * NEST_INDENT, 1.0]),
        LayoutChild {
            width: LayoutSize::Fixed(LABEL_COL - depth * NEST_INDENT),
            ..caption
        },
    ];
    match field.kind {
        FieldKind::Bool => {
            let key = format!("{CHECK}{j}");
            let check = ui.checkbox(
                Checkbox {
                    checked: Some(Binding::State(key.clone())),
                    on_change: key.clone(),
                    width: 0.0,
                    ..Default::default()
                },
                FIELD_H - 6.0,
            )?;
            ids.check = Some(key);
            row.push(check);
        }
        FieldKind::Enum | FieldKind::Ref { .. } | FieldKind::Action { .. } => {
            let key = format!("editor.form.choice.{j}");
            let choice = ui.button_with(
                Button {
                    text_binding: Some(Binding::State(key.clone())),
                    action: Some(emit(CHOICE, index.clone())),
                    ..Default::default()
                },
                [0.0, FIELD_H - 6.0],
            )?;
            ids.anchor = choice.item.map(concinnity_engine::ecs::Ref::id);
            ids.choice = Some(key);
            row.push(LayoutChild {
                width: LayoutSize::Fill(1.0),
                ..choice
            });
        }
        FieldKind::Array => {
            let count = ui.label(&format!("({})", field.variant_idx), theme::LABEL_DIM)?;
            row.push(LayoutChild {
                width: LayoutSize::Fill(1.0),
                ..count
            });
            row.push(ui.button("+", ADD, index.clone(), [SMALL_BTN, SMALL_BTN])?);
            row.push(ui.button("-", REMOVE, index.clone(), [SMALL_BTN, SMALL_BTN])?);
        }
        FieldKind::Vec { color: false, len } => {
            // A disclosure: its elements edit one at a time once it opens.
            let open = vec_expanded(fields, j);
            let caption = format!("[{len}] {}", if open { "v" } else { ">" });
            let toggle = ui.button(&caption, EXPAND, index.clone(), [0.0, FIELD_H - 6.0])?;
            row.push(LayoutChild {
                width: LayoutSize::Fill(1.0),
                ..toggle
            });
        }
        kind => {
            let text = ui.text_field(TextField {
                height: FIELD_H - 6.0,
                ..Default::default()
            })?;
            ids.text = text.item.map(concinnity_engine::ecs::Ref::id);
            row.push(text);
            if let FieldKind::Vec { color: true, .. } = kind {
                let swatch = ui.sprite(Sprite {
                    tint: NO_SWATCH,
                    corner_radius: 4.0,
                    ..Default::default()
                })?;
                ids.swatch = Some(swatch);
                row.push(fixed(swatch, [SWATCH_W, FIELD_H - 10.0]));
            }
        }
    }
    let marker = ui.button("*", OVERRIDE, index, [SMALL_BTN, SMALL_BTN])?;
    ids.marker = marker.item.map(concinnity_engine::ecs::Ref::id);
    row.push(marker);
    Some((ui.row(row, FIELD_H)?, ids))
}

// One extra row: its caption, its control, and its detail.
fn extra_row(ui: &mut UiBuilder<'_>, row: &ExtraRow) -> Option<(LayoutChild, ExtraRowIds)> {
    let mut ids = ExtraRowIds::default();
    let indent = if row.indent { NEST_INDENT } else { 0.0 };
    let pad = ui.sprite(Sprite {
        tint: [0.0; 4],
        ..Default::default()
    })?;
    let mut children = vec![fixed(pad, [BAR_W + indent, 1.0])];
    match &row.control {
        ExtraControl::Label => {
            let color = if row.indent {
                theme::LABEL_DIM
            } else {
                theme::HEADING
            };
            let caption = ui.clipped_label(&row.caption, color, EXTRA_LABEL_W - indent)?;
            ids.caption = caption.item.map(concinnity_engine::ecs::Ref::id);
            children.push(caption);
        }
        ExtraControl::Check { .. } => {
            let key = format!("{EXTRA_CHECK}{}", row.id);
            let check = ui.checkbox(
                Checkbox {
                    label: row.caption.clone(),
                    note: row.detail.clone().unwrap_or_default(),
                    checked: Some(Binding::State(key.clone())),
                    on_change: key.clone(),
                    ..Default::default()
                },
                FIELD_H - 6.0,
            )?;
            let id = check.item?.id();
            ids.check = Some((key, id));
            children.push(check);
        }
        ExtraControl::Choice { .. } => {
            let caption = ui.clipped_label(&row.caption, theme::LABEL, LABEL_COL - indent)?;
            ids.caption = caption.item.map(concinnity_engine::ecs::Ref::id);
            children.push(LayoutChild {
                width: LayoutSize::Fixed(LABEL_COL - indent),
                ..caption
            });
            let key = format!("editor.form.extra_choice.{}", row.id);
            let id = Some(UiValue::Int(i32::try_from(row.id).unwrap_or(i32::MAX)));
            let choice = ui.button_with(
                Button {
                    text_binding: Some(Binding::State(key.clone())),
                    action: Some(emit(EXTRA, id)),
                    ..Default::default()
                },
                [0.0, FIELD_H - 6.0],
            )?;
            ids.choice = Some(key);
            children.push(LayoutChild {
                width: LayoutSize::Fill(1.0),
                ..choice
            });
        }
    }
    if let (Some(detail), false) = (
        &row.detail,
        matches!(row.control, ExtraControl::Check { .. }),
    ) {
        let detail = ui.label(detail, theme::LABEL_DIM)?.item?.id();
        ids.detail = Some(detail);
        children.push(fit(detail));
    }
    Some((ui.row(children, FIELD_H)?, ids))
}

// Build the rows of a form of `fields` and `extras` into the scrolling
// column of `panel`, despawning the rows it held: for the fields' derivation
// `rev` of the form `opened`, replacing the rows `was`.
pub(crate) fn rebuild_rows(
    world: &mut World,
    panel: &mut BuiltPanel,
    fields: &[FormField],
    extras: &[ExtraRow],
    (rev, opened): (u64, u64),
    was: Option<&ShownForm>,
) -> ShownForm {
    let mut shown = ShownForm {
        rev,
        opened,
        extra_shape: extras.iter().map(extra_shape).collect(),
        ..ShownForm::default()
    };
    let Some(host) = panel.slots.get(ROWS_SLOT).copied() else {
        return shown;
    };
    crate::editor::ui::rows::rebuild(world, panel, host, |ui| {
        let mut rows = Vec::with_capacity(fields.len() + extras.len());
        for j in 0..fields.len() {
            let (row, ids) = field_row(ui, fields, j)?;
            rows.push(row);
            shown.fields.push(ids);
        }
        for extra in extras {
            let (row, ids) = extra_row(ui, extra)?;
            rows.push(row);
            shown.extras.push(ids);
        }
        Some(rows)
    });
    // A form opened afresh starts at its top; the same form rebuilt keeps
    // its place.
    if was.is_none_or(|w| w.opened != opened)
        && let Some(layout) = world.get_mut_by_id::<Layout>(host)
    {
        layout.scrolled = 0.0;
    }
    shown
}

// What the form shows this frame.
pub(crate) struct FormView<'a> {
    pub title: &'a str,
    // Whether an existing entry is being edited (Apply) or a new asset (Add).
    pub editing: bool,
    pub fields: &'a [FormField],
    pub texts: &'a [String],
    pub name: &'a str,
    pub focus: Option<FormFocus>,
    pub error: Option<&'a str>,
    pub blocked: Option<&'a str>,
    // Each field's origin, and how many differ, for a template-derived asset.
    pub marks: Option<(&'a [FieldOrigin], usize)>,
    pub extras: &'a [ExtraRow],
    // The open menu and its options.
    pub menu: Option<(FormMenu, &'a [String])>,
    // A row to scroll into view.
    pub reveal: Option<usize>,
}

// Set `slot` to `value` when it differs.
fn set<T: PartialEq>(slot: &mut T, value: T) {
    if *slot != value {
        *slot = value;
    }
}

// Write `text` into the field `id` when it holds something else.
fn set_field_text(world: &mut World, id: Option<AssetId>, text: &str, focused: bool) {
    if let Some(f) = id.and_then(|id| world.get_mut_by_id::<TextField>(id)) {
        if f.content != text {
            f.content.clear();
            f.content.push_str(text);
            f.caret = text.chars().count();
        }
        set(&mut f.focused, focused);
    }
}

pub(crate) fn sync(view: &FormView<'_>, shown: &ShownForm, world: &mut World, built: &BuiltPanel) {
    let slot = |i: usize| built.slots.get(i).copied();
    state::set_text(world, TITLE, format_args!("{}", view.title));
    let apply = if view.editing { "Apply" } else { "Add" };
    state::set_text(world, APPLY_TEXT, format_args!("{apply}"));
    if let Some(b) = slot(APPLY_SLOT).and_then(|id| world.get_mut_by_id::<Button>(id)) {
        set(&mut b.disabled, view.blocked.is_some());
    }
    let (status, color) = match (view.error, view.blocked) {
        (Some(e), _) => (e, ERROR_LABEL),
        (None, Some(b)) => (b, theme::LABEL_DIM),
        (None, None) => ("", theme::LABEL_DIM),
    };
    state::set_text(world, STATUS, format_args!("{status}"));
    if let Some(l) = slot(STATUS_SLOT).and_then(|id| world.get_mut_by_id::<TextLabel>(id)) {
        set(&mut l.color, color);
    }
    if let Some(l) = slot(OVERRIDES_SLOT).and_then(|id| world.get_mut_by_id::<Layout>(id)) {
        set(&mut l.visible, view.marks.is_some());
    }
    if let Some((_, count)) = view.marks {
        state::set_text(world, CHIP, format_args!("{count} overridden"));
    }
    set_field_text(
        world,
        slot(NAME_SLOT),
        view.name,
        view.focus == Some(FormFocus::Name),
    );
    for (j, (field, row)) in view.fields.iter().zip(&shown.fields).enumerate() {
        let text = view.texts.get(j).map_or("", String::as_str);
        if row.text.is_some() {
            set_field_text(
                world,
                row.text,
                text,
                view.focus == Some(FormFocus::Field(j)),
            );
        }
        if let Some(sprite) = row.swatch.and_then(|id| world.get_mut_by_id::<Sprite>(id)) {
            set(&mut sprite.tint, swatch_rgb(text));
        }
        if let Some(key) = &row.check {
            state::set_bool(world, key, field.boolval);
        }
        if let Some(key) = &row.choice {
            let value = field
                .variants
                .get(field.variant_idx)
                .map_or("", String::as_str);
            state::set_text(world, key, format_args!("{value}"));
        }
        let mark = view
            .marks
            .and_then(|(marks, _)| marks.get(j))
            .copied()
            .unwrap_or(FieldOrigin::Inherited);
        let marked = mark != FieldOrigin::Inherited;
        if let Some(bar) = row.bar.and_then(|id| world.get_mut_by_id::<Sprite>(id)) {
            set(&mut bar.visible, marked);
            let tint = if mark == FieldOrigin::InstanceOnly {
                INSTANCE_ONLY_TINT
            } else {
                OVERRIDE_TINT
            };
            set(&mut bar.tint, tint);
        }
        if let Some(b) = row.marker.and_then(|id| world.get_mut_by_id::<Button>(id)) {
            set(&mut b.visible, marked);
            set(&mut b.style.text, OVERRIDE_LABEL);
        }
    }
    for (row, ids) in view.extras.iter().zip(&shown.extras) {
        for (id, text) in [
            (ids.caption, Some(&row.caption)),
            (ids.detail, row.detail.as_ref()),
        ] {
            if let (Some(text), Some(l)) =
                (text, id.and_then(|id| world.get_mut_by_id::<TextLabel>(id)))
                && l.content != *text
            {
                l.content.clone_from(text);
            }
        }
        match &row.control {
            ExtraControl::Check { on, enabled } => {
                if let Some((key, id)) = &ids.check {
                    state::set_bool(world, key, *on);
                    if let Some(c) = world.get_mut_by_id::<Checkbox>(*id) {
                        set(&mut c.disabled, !enabled);
                        if c.label != row.caption {
                            c.label.clone_from(&row.caption);
                        }
                        let note = row.detail.as_deref().unwrap_or("");
                        if c.note != note {
                            c.note.clear();
                            c.note.push_str(note);
                        }
                    }
                }
            }
            ExtraControl::Choice { options, selected } => {
                if let Some(key) = &ids.choice {
                    let value = options.get(*selected).map_or("", String::as_str);
                    state::set_text(world, key, format_args!("{value}"));
                }
            }
            ExtraControl::Label => {}
        }
    }
    sync_menu(view, shown, world, built);
    if let (Some(row), Some(host)) = (view.reveal, slot(ROWS_SLOT))
        && let Some(layout) = world.get_mut_by_id::<Layout>(host)
    {
        layout.scrolled = row as f32 * (FIELD_H + ROW_GAP);
    }
}

// The open menu's items, placed under the control it belongs to.
fn sync_menu(view: &FormView<'_>, shown: &ShownForm, world: &mut World, built: &BuiltPanel) {
    let options = view.menu.map_or(&[][..], |(_, options)| options);
    let anchor = view.menu.and_then(|(menu, _)| match menu {
        FormMenu::Field(j) => shown.fields.get(j).and_then(|r| r.anchor),
        FormMenu::Override(j) => shown.fields.get(j).and_then(|r| r.marker),
        FormMenu::Entity => built.slots.get(OVERRIDES_SLOT).copied(),
    });
    let at = anchor.and_then(|id| control_rect(world, id));
    state::with_state(world, |s| {
        let listed = s.list(MENU_ITEMS);
        let same =
            listed.len() == options.len() && listed.iter().zip(options).all(|(a, b)| a.text == *b);
        if !same {
            s.set_list(MENU_ITEMS, options.iter().map(|o| UiItem::new(o)).collect());
        }
        s.set(MENU_OPEN, UiValue::Bool(!options.is_empty()));
    });
    if let (Some(rect), Some(menu)) = (at, built.slots.get(MENU_SLOT).copied())
        && let Some(m) = world.get_mut_by_id::<Menu>(menu)
    {
        set(&mut m.x, rect[0]);
        set(&mut m.y, rect[1] + rect[3]);
    }
}

// Where the control `id` was last laid out.
fn control_rect(world: &World, id: AssetId) -> Option<[f32; 4]> {
    if let Some(b) = world.get_by_id::<Button>(id) {
        return Some([b.x, b.y, b.width, b.height]);
    }
    world.get_by_id::<Layout>(id).and_then(|l| l.rect)
}

// Whether the menu the hook keeps open is still open: a press off it, or
// Escape, closes it.
pub(crate) fn menu_open(world: &World) -> bool {
    world
        .resource::<UiState>()
        .is_some_and(|s| s.get(MENU_OPEN) == Some(&UiValue::Bool(true)))
}

// What the user typed, read back from the panel's fields: the name, each
// field's text, and which of them holds the keyboard.
pub(crate) fn observe(
    world: &World,
    built: &BuiltPanel,
    shown: &ShownForm,
    name: &mut String,
    texts: &mut [String],
) -> (bool, Option<FormFocus>) {
    let mut changed = false;
    let mut focus = None;
    let mut take = |id: Option<AssetId>, into: &mut String| -> bool {
        let Some(f) = id.and_then(|id| world.get_by_id::<TextField>(id)) else {
            return false;
        };
        if f.content != *into {
            into.clone_from(&f.content);
            changed = true;
        }
        f.focused
    };
    if take(built.slots.get(NAME_SLOT).copied(), name) {
        focus = Some(FormFocus::Name);
    }
    for (j, row) in shown.fields.iter().enumerate() {
        if row.text.is_some()
            && let Some(text) = texts.get_mut(j)
            && take(row.text, text)
        {
            focus = Some(FormFocus::Field(j));
        }
    }
    (changed, focus)
}

#[cfg(test)]
mod tests;

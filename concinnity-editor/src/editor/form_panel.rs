// src/editor/form_panel.rs
//
// The editor's add / edit form, in its own floating panel (separate from the
// Assets browse panel, which stays on its list while this is open). Spawned by
// clicking an asset in the browse list (or picking a type from the "+" picker):
// a draggable title bar ("Edit <type>" / "New <type>"), then a header row with
// the asset's editable NAME as a large heading and the Apply / Cancel buttons
// pinned to the panel's top-right, then the scrollable arg-field area. Like the
// rest of the editor HUD it is plain `Sprite` / `TextLabel` / `TextInput`
// components at reserved ids driven each frame by the hook; the field model
// itself lives in `form.rs` and the hook owns all state.
//
// The field area renders a scrolling window over ALL of the type's fields into
// a fixed pool of `form::FIELD_POOL` control slots: visible slot `r` shows
// logical field `form_scroll + r`, and hit-test actions carry the LOGICAL index.
// An enum / ref field with a large variant set opens a floating value dropdown
// below the field (its own row pool here, so it can coexist with the Assets
// panel's combo).

use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

use super::form::{self, FieldKind, FormField};
use super::widget::{self, place_sprite, point_in};

// An enum / ref field with this many variants or fewer cycles in place on click
// (fast for small sets); more than this opens a floating value dropdown anchored
// below the field (a long list is tedious to cycle through). A reference field's
// count includes its `(none)` option.
pub(crate) const CYCLE_MAX: usize = 5;

// Visible rows in an open field-value dropdown before it scrolls.
pub(crate) const MAX_DROP_ROWS: usize = 8;
// A field-value dropdown option row's height.
const DROP_ROW_H: f32 = 28.0;

// Reserved asset-id families, between the Assets panel's (up past
// `ID_BASE + 0x40`) and the Preview panel's (`ID_BASE + 0x400`).
const EDIT: u32 = 0x3000_0000 + 0x200;
pub(crate) const EDIT_BG: AssetId = AssetId(EDIT);
pub(crate) const TITLE_BG: AssetId = AssetId(EDIT + 1);
pub(crate) const TITLE_LABEL: AssetId = AssetId(EDIT + 2);
pub(crate) const APPLY_BG: AssetId = AssetId(EDIT + 3);
pub(crate) const APPLY_LABEL: AssetId = AssetId(EDIT + 4);
pub(crate) const CANCEL_BG: AssetId = AssetId(EDIT + 5);
pub(crate) const CANCEL_LABEL: AssetId = AssetId(EDIT + 6);
// The asset-name heading: a real editable TextInput drawn a step larger.
pub(crate) const NAME_INPUT: AssetId = AssetId(EDIT + 7);
pub(crate) const FORM_STATUS: AssetId = AssetId(EDIT + 8);
pub(crate) const FORM_TRACK: AssetId = AssetId(EDIT + 9);
pub(crate) const FORM_THUMB: AssetId = AssetId(EDIT + 10);
pub(crate) const DROP_BG: AssetId = AssetId(EDIT + 11);
pub(crate) const DROP_TRACK: AssetId = AssetId(EDIT + 12);
pub(crate) const DROP_THUMB: AssetId = AssetId(EDIT + 13);

// The per-slot control pool (`form::FIELD_POOL` of each). `form_row_label(r)` is
// the caption of visible slot `r`; `form_input(r)` its text control;
// `form_toggle_bg(r)` its checkbox / cycle-button / array-add background;
// `form_swatch(r)` its colour swatch / array-remove button;
// `form_enum_label(r)` its enum-value / array-count caption.
pub(crate) fn form_row_label(r: usize) -> AssetId {
    AssetId(EDIT + 0x20 + r as u32)
}
pub(crate) fn form_input(r: usize) -> AssetId {
    AssetId(EDIT + 0x40 + r as u32)
}
pub(crate) fn form_toggle_bg(r: usize) -> AssetId {
    AssetId(EDIT + 0x60 + r as u32)
}
pub(crate) fn form_swatch(r: usize) -> AssetId {
    AssetId(EDIT + 0x80 + r as u32)
}
pub(crate) fn form_enum_label(r: usize) -> AssetId {
    AssetId(EDIT + 0xA0 + r as u32)
}
// The value dropdown's own option rows (the Assets panel's combo pool can be in
// use at the same time now that the two are separate panels).
pub(crate) fn drop_row_bg(r: usize) -> AssetId {
    AssetId(EDIT + 0xC0 + r as u32)
}
pub(crate) fn drop_row_label(r: usize) -> AssetId {
    AssetId(EDIT + 0xE0 + r as u32)
}

// Which of the form's inputs holds keyboard focus (re-asserted each frame so the
// opening click cannot blur it; only the focused input re-asserts, so N text
// fields do not fight).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FormFocus {
    Name,
    Field(usize),
}

// Geometry, in window pixels. A wider column than the Assets panel so field
// names and values read comfortably; every rect derives from the panel origin
// `o` (the title bar's top-left), so dragging the title bar moves the panel.
pub(crate) const EDIT_W: f32 = 480.0;
const PAD: f32 = 10.0;
const GAP: f32 = 8.0;
// The header row: the name heading and the Apply / Cancel buttons.
const NAME_H: f32 = 34.0;
const BTN_W: f32 = 84.0;
// The validation status line reserved under the header row.
const STATUS_H: f32 = 22.0;
// One arg-field row; the caption column on its left.
const FIELD_H: f32 = 34.0;
const LABEL_COL: f32 = 190.0;
// Left inset per nesting level for a flattened nested (dotted-path) leaf caption.
const NEST_INDENT: f32 = 10.0;
const SCROLLBAR_W: f32 = 5.0;
// The name heading draws larger than the body text.
const NAME_SCALE: f32 = 1.2;

const PANEL_TINT: [f32; 4] = [0.09, 0.09, 0.12, 0.97];
const BTN_TINT: [f32; 4] = [0.22, 0.40, 0.56, 1.0];
const BTN_TINT_HOVER: [f32; 4] = [0.24, 0.28, 0.40, 1.0];
const CANCEL_TINT: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const CYCLE_TINT: [f32; 4] = [0.18, 0.20, 0.28, 1.0];
const OPTION_TINT: [f32; 4] = [0.16, 0.16, 0.20, 1.0];
const OPTION_TINT_HOVER: [f32; 4] = [0.24, 0.28, 0.40, 1.0];
const OPTION_TINT_SELECTED: [f32; 4] = [0.16, 0.22, 0.34, 1.0];
const DROP_BG_TINT: [f32; 4] = [0.10, 0.10, 0.13, 1.0];
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
const CHECK_ON: [f32; 4] = [0.30, 0.66, 0.34, 1.0];
const CHECK_OFF: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const ADD_BTN_TINT: [f32; 4] = [0.24, 0.52, 0.32, 1.0];
const REMOVE_BTN_TINT: [f32; 4] = [0.52, 0.26, 0.26, 1.0];
const LABEL: [f32; 3] = [0.90, 0.90, 0.92];
const LABEL_DIM: [f32; 3] = [0.60, 0.60, 0.66];
const LABEL_WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];

// Where the panel sits until the user drags it: to the left of the Assets
// panel's default anchor, below the top bar.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [
        vw - super::panel::PANEL_W - GAP - EDIT_W,
        super::hud::body_top(),
    ]
}

// The panel footprint for `n_fields` total fields (the field area is capped at
// the control pool; a wider form scrolls). Origin-independent; the hook's drag
// clamp uses this.
pub(crate) fn size(n_fields: usize) -> [f32; 2] {
    let visible = n_fields.clamp(1, form::FIELD_POOL);
    [
        EDIT_W,
        widget::TITLE_H + PAD + NAME_H + 4.0 + STATUS_H + visible as f32 * FIELD_H + PAD,
    ]
}

// The panel outer rect for a view (its height tracks the visible field count).
fn panel_rect(view: &FormView, o: [f32; 2]) -> [f32; 4] {
    let s = size(view.form_fields.len());
    [o[0], o[1], s[0], s[1]]
}

// The draggable title bar across the panel top.
pub(crate) fn title_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0], o[1], EDIT_W, widget::TITLE_H]
}

// The header row under the title bar: name heading + the two pinned buttons.
fn header_y(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H + PAD
}
pub(crate) fn name_rect(o: [f32; 2]) -> [f32; 4] {
    [
        o[0] + PAD,
        header_y(o),
        EDIT_W - 2.0 * PAD - 2.0 * (BTN_W + GAP),
        NAME_H,
    ]
}
pub(crate) fn apply_rect(o: [f32; 2]) -> [f32; 4] {
    [
        o[0] + EDIT_W - PAD - 2.0 * BTN_W - GAP,
        header_y(o),
        BTN_W,
        NAME_H,
    ]
}
pub(crate) fn cancel_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0] + EDIT_W - PAD - BTN_W, header_y(o), BTN_W, NAME_H]
}

// Where the scrollable field area begins (below the header + status line).
fn fields_top(o: [f32; 2]) -> f32 {
    header_y(o) + NAME_H + 4.0 + STATUS_H
}

// The full-width rect of visible field slot `r`.
fn field_row_rect(o: [f32; 2], r: usize) -> [f32; 4] {
    [o[0], fields_top(o) + r as f32 * FIELD_H, EDIT_W, FIELD_H]
}
// The control (text field / checkbox area) on the right of slot `r`.
pub(crate) fn form_control_rect(o: [f32; 2], r: usize) -> [f32; 4] {
    let row = field_row_rect(o, r);
    [
        row[0] + LABEL_COL,
        row[1] + 3.0,
        EDIT_W - LABEL_COL - PAD - 2.0,
        FIELD_H - 8.0,
    ]
}
// The checkbox box for a bool field on slot `r`.
pub(crate) fn form_toggle_rect(o: [f32; 2], r: usize) -> [f32; 4] {
    let c = form_control_rect(o, r);
    let s = FIELD_H - 12.0;
    [c[0], c[1], s, s]
}
// The `[+]` add and `[-]` remove buttons of an array header on slot `r`.
fn array_add_rect(o: [f32; 2], r: usize) -> [f32; 4] {
    let c = form_control_rect(o, r);
    let s = FIELD_H - 12.0;
    [c[0] + c[2] - s, c[1], s, s]
}
fn array_remove_rect(o: [f32; 2], r: usize) -> [f32; 4] {
    let a = array_add_rect(o, r);
    [a[0] - a[2] - 4.0, a[1], a[2], a[3]]
}

// An open value dropdown for the enum / ref field on slot `r`: it floats
// directly below that field's control, aligned to it.
fn field_option_rect(o: [f32; 2], slot: usize, r: usize) -> [f32; 4] {
    let c = form_control_rect(o, slot);
    [c[0], c[1] + c[3] + r as f32 * DROP_ROW_H, c[2], DROP_ROW_H]
}
fn field_dropdown_backing(o: [f32; 2], slot: usize, shown: usize) -> [f32; 4] {
    let c = form_control_rect(o, slot);
    [c[0], c[1] + c[3], c[2], shown as f32 * DROP_ROW_H + 4.0]
}

// Whether two rects overlap (touching edges do not count).
fn rects_intersect(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3]
}

// How many field slots are on screen: the field count capped at the pool.
fn visible_field_count(view: &FormView) -> usize {
    view.form_fields.len().min(form::FIELD_POOL)
}

// The slot showing logical field `j`, if it is inside the scroll window.
fn field_slot(view: &FormView, j: usize) -> Option<usize> {
    let scroll = view.form_scroll;
    (j >= scroll && j < scroll + form::FIELD_POOL).then(|| j - scroll)
}

// A resolved form-panel click. Field actions carry the LOGICAL field index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FormAction {
    // Focus the name heading, or arg text field `i`.
    FocusName,
    FocusField(usize),
    // Toggle the bool arg field `i`.
    ToggleField(usize),
    // Advance the enum / ref arg field `i` to its next variant (small sets).
    CycleField(usize),
    // Open (or, if already open, close) the value dropdown for field `i`.
    OpenFieldDropdown(usize),
    // Pick option `i` from the open field-value dropdown.
    PickFieldOption(usize),
    // Append / drop the last element of array arg field `i`.
    AddArrayElement(usize),
    RemoveArrayElement(usize),
    // The Apply / Add button: validate + commit.
    Confirm,
    // The Cancel button: close the panel, discarding the form.
    Close,
    // Dismiss the open value dropdown without picking.
    CloseOverlays,
    // A click inside the panel that hits no control (swallowed).
    Consume,
}

// The per-frame data the hook hands to `apply` / `hit_test`.
pub(crate) struct FormView<'a> {
    // The title-bar heading ("Edit Camera3D" / "New PointLight").
    pub title: &'a str,
    // Whether an existing entry is being edited (confirm says "Apply", and the
    // hook highlights the entry's browse row) vs a new asset ("Add").
    pub editing: bool,
    // The editable arg fields; the panel renders a `form::FIELD_POOL` window.
    pub form_fields: &'a [FormField],
    // First visible field of the scroll window.
    pub form_scroll: usize,
    // Which input holds keyboard focus.
    pub form_focus: FormFocus,
    // The field whose value dropdown is open, and its scroll offset.
    pub field_dropdown: Option<usize>,
    pub field_dropdown_scroll: usize,
    // A validation error from the last rejected commit.
    pub form_error: Option<&'a str>,
    pub mouse: [f32; 2],
}

// Whether the cursor is over the panel (for wheel-scrolling the field window).
pub(crate) fn cursor_over(view: &FormView, mx: f32, my: f32, o: [f32; 2]) -> bool {
    point_in(mx, my, panel_rect(view, o))
}

// Resolve a click against the open form panel at origin `o`. `None` means the
// click missed the panel (the caller lets it fall through). Title-bar presses
// never reach this: the hook intercepts them first to start a drag.
pub(crate) fn hit_test(view: &FormView, mx: f32, my: f32, o: [f32; 2]) -> Option<FormAction> {
    // An open value dropdown is modal over the panel: its option rows pick,
    // anything else dismisses it.
    if let Some(open) = view.field_dropdown
        && let Some(field) = view.form_fields.get(open)
        && let Some(slot) = field_slot(view, open)
    {
        let total = field.variants.len();
        let scroll = view.field_dropdown_scroll.min(total.saturating_sub(1));
        for r in 0..MAX_DROP_ROWS {
            let idx = scroll + r;
            if idx >= total {
                break;
            }
            if point_in(mx, my, field_option_rect(o, slot, r)) {
                return Some(FormAction::PickFieldOption(idx));
            }
        }
        return Some(FormAction::CloseOverlays);
    }

    if !point_in(mx, my, panel_rect(view, o)) {
        return None;
    }
    if point_in(mx, my, apply_rect(o)) {
        return Some(FormAction::Confirm);
    }
    if point_in(mx, my, cancel_rect(o)) {
        return Some(FormAction::Close);
    }
    if point_in(mx, my, name_rect(o)) {
        return Some(FormAction::FocusName);
    }

    // Arg field controls over the visible window: slot `r` is logical field
    // `form_scroll + r`. A bool toggles, an enum cycles / opens a dropdown, an
    // array grows / shrinks, a text field takes focus.
    let scroll = view.form_scroll;
    for r in 0..visible_field_count(view) {
        let j = scroll + r;
        let Some(f) = view.form_fields.get(j) else {
            break;
        };
        match f.kind {
            FieldKind::Bool => {
                if point_in(mx, my, form_toggle_rect(o, r)) {
                    return Some(FormAction::ToggleField(j));
                }
            }
            FieldKind::Enum | FieldKind::Ref { .. } => {
                if point_in(mx, my, form_control_rect(o, r)) {
                    // A small variant set cycles in place; a large one opens a
                    // floating dropdown instead of forcing many clicks.
                    if f.variants.len() > CYCLE_MAX {
                        return Some(FormAction::OpenFieldDropdown(j));
                    }
                    return Some(FormAction::CycleField(j));
                }
            }
            FieldKind::Array => {
                if point_in(mx, my, array_add_rect(o, r)) {
                    return Some(FormAction::AddArrayElement(j));
                }
                if point_in(mx, my, array_remove_rect(o, r)) {
                    return Some(FormAction::RemoveArrayElement(j));
                }
            }
            _ => {
                if point_in(mx, my, form_control_rect(o, r)) {
                    return Some(FormAction::FocusField(j));
                }
            }
        }
    }
    // Empty panel space: swallow so it does not fall through to the world.
    Some(FormAction::Consume)
}

// Position + show the panel's elements for this frame at origin `o`, or hide
// them all when the form is closed (`view` is `None`).
pub(crate) fn apply(world: &mut World, view: Option<&FormView>, o: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };

    // Blank everything, then re-show what this frame needs.
    hide_all(world);

    place_sprite(world, EDIT_BG, panel_rect(view, o), PANEL_TINT, true);
    widget::place_title(world, TITLE_BG, TITLE_LABEL, title_rect(o), view.title);

    // The header row: the asset-name heading (drawn larger) and the pinned
    // Apply / Cancel buttons at the panel's top right.
    show_name_heading(world, o, view.form_focus == FormFocus::Name);
    let apply_btn = apply_rect(o);
    let hover = point_in(view.mouse[0], view.mouse[1], apply_btn);
    place_sprite(
        world,
        APPLY_BG,
        apply_btn,
        if hover { BTN_TINT_HOVER } else { BTN_TINT },
        true,
    );
    let confirm = if view.editing { "Apply" } else { "Add" };
    place_center_label(world, APPLY_LABEL, apply_btn, confirm, LABEL_WHITE, true);
    let cancel = cancel_rect(o);
    place_sprite(world, CANCEL_BG, cancel, CANCEL_TINT, true);
    place_center_label(world, CANCEL_LABEL, cancel, "Cancel", LABEL, true);

    // A validation error, if the last commit was rejected (a reserved line, so
    // showing it never shifts the fields below).
    if let Some(err) = view.form_error {
        place_left_label(
            world,
            FORM_STATUS,
            [o[0] + PAD, header_y(o) + NAME_H + 6.0],
            err,
            ERROR_LABEL,
            true,
        );
    }

    // An open value dropdown floats over the slots below its field. Those rows
    // are left fully hidden this frame (both the control and its caption): a
    // covered VISIBLE `TextInput` would synth an opaque background box over the
    // dropdown (it is synthesised only for visible inputs, see
    // graphics_system/frame.rs), and a long caption could overflow across it.
    let drop_backing = view.field_dropdown.and_then(|open| {
        let field = view.form_fields.get(open)?;
        let slot = field_slot(view, open)?;
        let total = field.variants.len();
        let scroll = view.field_dropdown_scroll.min(total.saturating_sub(1));
        let shown = total.saturating_sub(scroll).clamp(1, MAX_DROP_ROWS);
        Some(field_dropdown_backing(o, slot, shown))
    });

    // The arg fields over the visible window, drawn into the slot-indexed pool.
    let scroll = view.form_scroll;
    for r in 0..visible_field_count(view) {
        let j = scroll + r;
        let Some(field) = view.form_fields.get(j) else {
            break;
        };
        let row = field_row_rect(o, r);
        if drop_backing.is_some_and(|d| rects_intersect(form_control_rect(o, r), d)) {
            continue;
        }
        // A nested (dotted-path) field shows its indented leaf name.
        let depth = field.key.matches('.').count();
        let leaf = field.key.rsplit('.').next().unwrap_or(field.key.as_str());
        place_left_label(
            world,
            form_row_label(r),
            [
                row[0] + PAD + depth as f32 * NEST_INDENT,
                row[1] + FIELD_H * 0.5 - 10.0,
            ],
            leaf,
            LABEL,
            true,
        );
        match field.kind {
            FieldKind::Bool => {
                let t = form_toggle_rect(o, r);
                let tint = if field.boolval { CHECK_ON } else { CHECK_OFF };
                place_sprite(world, form_toggle_bg(r), t, tint, true);
            }
            FieldKind::Enum | FieldKind::Ref { .. } => {
                // A cycling button spanning the control, captioned with the
                // current selection.
                let c = form_control_rect(o, r);
                let hover = point_in(view.mouse[0], view.mouse[1], c);
                let tint = if hover { OPTION_TINT_HOVER } else { CYCLE_TINT };
                place_sprite(world, form_toggle_bg(r), c, tint, true);
                let value = field
                    .variants
                    .get(field.variant_idx)
                    .map(String::as_str)
                    .unwrap_or("");
                place_center_label(world, form_enum_label(r), c, value, LABEL, true);
            }
            FieldKind::Array => {
                // A header for a variable-length array: its element count and a
                // red `[-]` remove + green `[+]` add button.
                let c = form_control_rect(o, r);
                place_left_label(
                    world,
                    form_enum_label(r),
                    [c[0], c[1] + c[3] * 0.5 - 10.0],
                    &format!("({})", field.variant_idx),
                    LABEL_DIM,
                    true,
                );
                place_sprite(
                    world,
                    form_swatch(r),
                    array_remove_rect(o, r),
                    REMOVE_BTN_TINT,
                    true,
                );
                place_sprite(
                    world,
                    form_toggle_bg(r),
                    array_add_rect(o, r),
                    ADD_BTN_TINT,
                    true,
                );
            }
            kind => {
                let control = form_control_rect(o, r);
                // A colour vector reserves a right-hand strip for a live preview
                // swatch and narrows its field to clear it. The swatch must sit
                // OUTSIDE the field rect: a `TextInput`'s opaque background box
                // is synthesised after every authored sprite (the swatch
                // included), so a swatch drawn under the field would be painted
                // over.
                let swatch = matches!(kind, FieldKind::Vec { color: true, .. }).then(|| {
                    let s = FIELD_H - 14.0;
                    [
                        control[0] + control[2] - s,
                        control[1] + (control[3] - s) * 0.5,
                        s,
                        s,
                    ]
                });
                let rect = match swatch {
                    Some(sw) => [control[0], control[1], sw[0] - control[0] - 4.0, control[3]],
                    None => control,
                };
                widget::show_field(
                    world,
                    form_input(r),
                    rect,
                    view.form_focus == FormFocus::Field(j),
                );
                if let Some(sw) = swatch {
                    let rgb = swatch_rgb(&widget::field_text(world, form_input(r)));
                    place_sprite(world, form_swatch(r), sw, rgb, true);
                }
            }
        }
    }

    // A scrollbar down the field region's right edge when the form overflows the
    // pool.
    layout_form_scrollbar(world, view.form_fields.len(), scroll, o);

    // The open value dropdown draws last so it floats over the slots below it.
    if let Some(open) = view.field_dropdown {
        layout_field_dropdown(world, view, o, open);
    }
}

// The name heading: the editable TextInput under the title bar, drawn larger
// than the body text.
fn show_name_heading(world: &mut World, o: [f32; 2], focused: bool) {
    widget::show_field(world, NAME_INPUT, name_rect(o), focused);
    if let Some(t) = widget::input_mut(world, NAME_INPUT) {
        t.scale = NAME_SCALE;
    }
}

// The form's scrollbar thumb, sizing the visible window against the total field
// count. Shown only when the form overflows the pool.
fn layout_form_scrollbar(world: &mut World, total: usize, scroll: usize, o: [f32; 2]) {
    if total <= form::FIELD_POOL {
        return;
    }
    let region_top = fields_top(o);
    let track_h = form::FIELD_POOL as f32 * FIELD_H;
    let track = [
        o[0] + EDIT_W - SCROLLBAR_W,
        region_top,
        SCROLLBAR_W,
        track_h,
    ];
    place_sprite(world, FORM_TRACK, track, TRACK_TINT, true);
    let frac_visible = form::FIELD_POOL as f32 / total as f32;
    let thumb_h = (track_h * frac_visible).max(20.0);
    let max_scroll = (total - form::FIELD_POOL) as f32;
    let t = if max_scroll > 0.0 {
        scroll.min(total - form::FIELD_POOL) as f32 / max_scroll
    } else {
        0.0
    };
    let thumb_y = region_top + t * (track_h - thumb_h);
    place_sprite(
        world,
        FORM_THUMB,
        [track[0], thumb_y, SCROLLBAR_W, thumb_h],
        THUMB_TINT,
        true,
    );
}

// The floating value dropdown for an open enum / ref field: an opaque backing
// plus its option rows (the current selection highlighted) and, when it
// overflows, its own thin scrollbar.
fn layout_field_dropdown(world: &mut World, view: &FormView, o: [f32; 2], open: usize) {
    let Some(field) = view.form_fields.get(open) else {
        return;
    };
    let Some(slot) = field_slot(view, open) else {
        return;
    };
    let total = field.variants.len();
    let scroll = view.field_dropdown_scroll.min(total.saturating_sub(1));
    let shown = total.saturating_sub(scroll).clamp(1, MAX_DROP_ROWS);
    place_sprite(
        world,
        DROP_BG,
        field_dropdown_backing(o, slot, shown),
        DROP_BG_TINT,
        true,
    );
    for r in 0..MAX_DROP_ROWS {
        let idx = scroll + r;
        if idx >= total {
            break;
        }
        let rect = field_option_rect(o, slot, r);
        let hovered = point_in(view.mouse[0], view.mouse[1], rect);
        let tint = if hovered {
            OPTION_TINT_HOVER
        } else if idx == field.variant_idx {
            OPTION_TINT_SELECTED
        } else {
            OPTION_TINT
        };
        place_sprite(world, drop_row_bg(r), rect, tint, true);
        place_left_label(
            world,
            drop_row_label(r),
            [rect[0] + PAD, rect[1] + DROP_ROW_H * 0.5 - 10.0],
            &field.variants[idx],
            LABEL,
            true,
        );
    }
    if total > MAX_DROP_ROWS {
        let back = field_dropdown_backing(o, slot, shown);
        let track = [
            back[0] + back[2] - SCROLLBAR_W,
            back[1],
            SCROLLBAR_W,
            back[3],
        ];
        place_sprite(world, DROP_TRACK, track, TRACK_TINT, true);
        let frac = MAX_DROP_ROWS as f32 / total as f32;
        let thumb_h = (back[3] * frac).max(20.0);
        let max_scroll = (total - MAX_DROP_ROWS) as f32;
        let t = if max_scroll > 0.0 {
            scroll as f32 / max_scroll
        } else {
            0.0
        };
        let thumb_y = back[1] + t * (back[3] - thumb_h);
        place_sprite(
            world,
            DROP_THUMB,
            [track[0], thumb_y, SCROLLBAR_W, thumb_h],
            THUMB_TINT,
            true,
        );
    }
}

// Parse a colour field's live text ("r, g, b" / "r, g, b, a") into an opaque RGB
// tint for its preview swatch, clamped to the displayable 0..=1 range. Falls
// back to a neutral dark swatch until three components parse.
fn swatch_rgb(text: &str) -> [f32; 4] {
    let nums: Vec<f32> = text
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<f32>().ok())
        .take(3)
        .collect();
    if nums.len() < 3 {
        return [0.15, 0.15, 0.18, 1.0];
    }
    [
        nums[0].clamp(0.0, 1.0),
        nums[1].clamp(0.0, 1.0),
        nums[2].clamp(0.0, 1.0),
        1.0,
    ]
}

fn place_center_label(
    world: &mut World,
    id: AssetId,
    rect: [f32; 4],
    content: &str,
    color: [f32; 3],
    visible: bool,
) {
    if let Some(l) = widget::label_mut(world, id) {
        l.x = rect[0] + rect[2] * 0.5;
        l.y = rect[1] + rect[3] * 0.5 - 10.0;
        l.align = TextAlign::Center;
        l.color = color;
        l.visible = visible;
        l.content = content.to_string();
    }
}

fn place_left_label(
    world: &mut World,
    id: AssetId,
    pos: [f32; 2],
    content: &str,
    color: [f32; 3],
    visible: bool,
) {
    if let Some(l) = widget::label_mut(world, id) {
        l.x = pos[0];
        l.y = pos[1];
        l.align = TextAlign::Left;
        l.color = color;
        l.visible = visible;
        l.content = content.to_string();
    }
}

// Hide every panel element, including the typed fields (and blur them so a
// hidden field cannot keep keyboard focus).
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

// Every panel sprite id. THE ORDER OF THIS VEC IS THE DRAW ORDER (inject.rs
// inserts in this sequence; the overlay draws in insertion order): background,
// title, buttons, per-slot chrome, then the floating overlays (scrollbar, value
// dropdown) which must sit above the slot chrome.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![EDIT_BG, TITLE_BG, APPLY_BG, CANCEL_BG];
    ids.extend((0..form::FIELD_POOL).map(form_toggle_bg));
    ids.extend((0..form::FIELD_POOL).map(form_swatch));
    ids.extend([FORM_TRACK, FORM_THUMB, DROP_BG]);
    ids.extend((0..MAX_DROP_ROWS).map(drop_row_bg));
    ids.extend([DROP_TRACK, DROP_THUMB]);
    ids
}
// Same draw-order contract; the dropdown's option captions come last so they
// draw above the slot captions the dropdown floats over.
pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![TITLE_LABEL, APPLY_LABEL, CANCEL_LABEL, FORM_STATUS];
    ids.extend((0..form::FIELD_POOL).map(form_row_label));
    ids.extend((0..form::FIELD_POOL).map(form_enum_label));
    ids.extend((0..MAX_DROP_ROWS).map(drop_row_label));
    ids
}
// Every typed field: the name heading and the arg-field text inputs.
pub(crate) fn all_field_ids() -> Vec<AssetId> {
    let mut ids = vec![NAME_INPUT];
    ids.extend((0..form::FIELD_POOL).map(form_input));
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextInput, TextLabel};

    fn test_origin() -> [f32; 2] {
        default_origin(1280.0)
    }

    fn field(key: &str, kind: FieldKind) -> FormField {
        FormField {
            key: key.into(),
            kind,
            initial: "0".into(),
            boolval: false,
            variants: Vec::new(),
            variant_idx: 0,
        }
    }

    fn float_fields(n: usize) -> Vec<FormField> {
        (0..n)
            .map(|i| field(&format!("f{i}"), FieldKind::Float))
            .collect()
    }

    // A ref field with `n` options past `(none)`. With `n > CYCLE_MAX` it opens
    // a dropdown rather than cycling.
    fn ref_field(n: usize) -> FormField {
        let mut variants = vec![form::NONE_LABEL.to_string()];
        variants.extend((0..n).map(|i| format!("tex_{i}")));
        FormField {
            key: "texture".into(),
            kind: FieldKind::Ref { target: "Texture" },
            initial: String::new(),
            boolval: false,
            variants,
            variant_idx: 0,
        }
    }

    fn view<'a>(fields: &'a [FormField]) -> FormView<'a> {
        FormView {
            title: "New PointLight",
            editing: false,
            form_fields: fields,
            form_scroll: 0,
            form_focus: FormFocus::Name,
            field_dropdown: None,
            field_dropdown_scroll: 0,
            form_error: None,
            mouse: [0.0, 0.0],
        }
    }

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

    fn sprite(world: &World, id: AssetId) -> Sprite {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == id)
            .cloned()
            .expect("sprite present")
    }
    fn sprite_visible(world: &World, id: AssetId) -> bool {
        sprite(world, id).visible
    }
    fn label(world: &World, id: AssetId) -> TextLabel {
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .cloned()
            .expect("label present")
    }
    fn input(world: &World, id: AssetId) -> TextInput {
        world
            .query::<TextInput>()
            .find(|t| t.asset_id == id)
            .cloned()
            .expect("input present")
    }

    // The header pins the name heading and the Apply / Cancel buttons under the
    // title bar, buttons flush to the panel's top-right corner.
    #[test]
    fn header_pins_name_and_buttons_under_the_title_bar() {
        let o = test_origin();
        let title = title_rect(o);
        assert_eq!(title, [o[0], o[1], EDIT_W, widget::TITLE_H]);
        let name = name_rect(o);
        let apply_btn = apply_rect(o);
        let cancel = cancel_rect(o);
        assert_eq!(name[1], o[1] + widget::TITLE_H + PAD, "under the title bar");
        assert_eq!(name[1], apply_btn[1], "buttons share the header row");
        assert!(
            name[0] + name[2] <= apply_btn[0],
            "the name heading ends left of Apply"
        );
        assert!(
            apply_btn[0] + apply_btn[2] <= cancel[0],
            "Apply left of Cancel"
        );
        assert_eq!(
            cancel[0] + cancel[2],
            o[0] + EDIT_W - PAD,
            "Cancel flush to the panel's right pad"
        );
        // The whole panel follows its origin.
        let v = view(&[]);
        let moved = panel_rect(&v, [40.0, 60.0]);
        assert_eq!((moved[0], moved[1]), (40.0, 60.0));
    }

    // The panel height tracks the field count up to the pool cap; `size` is what
    // the hook's drag clamp uses.
    #[test]
    fn panel_height_tracks_fields_and_caps_at_the_pool() {
        let short = size(2);
        let tall = size(form::FIELD_POOL + 20);
        assert!(short[1] < tall[1]);
        assert_eq!(
            tall[1],
            size(form::FIELD_POOL)[1],
            "a form past the pool scrolls instead of growing"
        );
        let fields = float_fields(2);
        let v = view(&fields);
        assert_eq!(panel_rect(&v, test_origin())[3], short[1]);
    }

    #[test]
    fn buttons_and_name_hit_test() {
        let fields = float_fields(1);
        let v = view(&fields);
        let o = test_origin();
        let a = apply_rect(o);
        let c = cancel_rect(o);
        let n = name_rect(o);
        assert_eq!(
            hit_test(&v, a[0] + 5.0, a[1] + 5.0, o),
            Some(FormAction::Confirm)
        );
        assert_eq!(
            hit_test(&v, c[0] + 5.0, c[1] + 5.0, o),
            Some(FormAction::Close)
        );
        assert_eq!(
            hit_test(&v, n[0] + 5.0, n[1] + 5.0, o),
            Some(FormAction::FocusName)
        );
        // Title clicks are consumed (the hook grabs drags before this).
        let t = title_rect(o);
        assert_eq!(
            hit_test(&v, t[0] + 5.0, t[1] + 5.0, o),
            Some(FormAction::Consume)
        );
        // A miss falls through.
        assert_eq!(hit_test(&v, 10.0, 700.0, o), None);
    }

    // The title bar carries the heading, the confirm button is captioned "Apply"
    // while editing and "Add" for a new asset, and the name heading draws larger.
    #[test]
    fn apply_shows_title_confirm_caption_and_scaled_name() {
        let fields = float_fields(1);
        let mut world = injected_world();
        let o = test_origin();
        apply(&mut world, Some(&view(&fields)), o);
        assert_eq!(label(&world, TITLE_LABEL).content, "New PointLight");
        assert_eq!(label(&world, APPLY_LABEL).content, "Add");
        assert_eq!(label(&world, CANCEL_LABEL).content, "Cancel");
        let name = input(&world, NAME_INPUT);
        assert!(name.visible);
        assert!(name.scale > 1.0, "the name heading draws larger");
        let mut editing = view(&fields);
        editing.editing = true;
        editing.title = "Edit Camera3D";
        apply(&mut world, Some(&editing), o);
        assert_eq!(label(&world, TITLE_LABEL).content, "Edit Camera3D");
        assert_eq!(label(&world, APPLY_LABEL).content, "Apply");
    }

    // Field slots map logical fields through the scroll window; a form past the
    // pool shows its scrollbar and keeps the buttons pinned in the header.
    #[test]
    fn a_form_past_the_pool_scrolls_a_window() {
        let o = test_origin();
        let fields = float_fields(form::FIELD_POOL + 4);
        let mut world = injected_world();
        let v = view(&fields);
        apply(&mut world, Some(&v), o);
        assert!(
            sprite_visible(&world, FORM_THUMB),
            "the form scrollbar shows"
        );
        // Slot 0 resolves to logical field 0.
        let c0 = form_control_rect(o, 0);
        assert_eq!(
            hit_test(&v, c0[0] + 5.0, c0[1] + 5.0, o),
            Some(FormAction::FocusField(0))
        );
        // Scrolled by three: slot 0 shows logical field 3; the last slot shows
        // the window's last field.
        let mut vs = view(&fields);
        vs.form_scroll = 3;
        assert_eq!(
            hit_test(&vs, c0[0] + 5.0, c0[1] + 5.0, o),
            Some(FormAction::FocusField(3))
        );
        let last = form_control_rect(o, form::FIELD_POOL - 1);
        assert_eq!(
            hit_test(&vs, last[0] + 5.0, last[1] + 5.0, o),
            Some(FormAction::FocusField(3 + form::FIELD_POOL - 1))
        );
        // The Apply button stays pinned in the header regardless.
        let a = apply_rect(o);
        assert_eq!(
            hit_test(&vs, a[0] + 5.0, a[1] + 5.0, o),
            Some(FormAction::Confirm)
        );
    }

    #[test]
    fn a_form_within_the_pool_has_no_scrollbar() {
        let fields = float_fields(1);
        let mut world = injected_world();
        apply(&mut world, Some(&view(&fields)), test_origin());
        assert!(!sprite_visible(&world, FORM_THUMB));
    }

    // A colour-vector arg field renders its text control plus a live preview
    // swatch tinted from the field's current text; the swatch sits outside the
    // field rect (a TextInput's opaque background box is synthesised after every
    // authored sprite, so a swatch under the field would be painted over).
    #[test]
    fn colour_vector_field_shows_a_live_swatch() {
        let mut world = injected_world();
        for t in world.query_mut::<TextInput>() {
            if t.asset_id == form_input(0) {
                t.content = "1, 0, 0".into();
            }
        }
        let fields = [FormField {
            key: "color".into(),
            kind: FieldKind::Vec {
                len: 3,
                color: true,
            },
            initial: "1, 0, 0".into(),
            boolval: false,
            variants: Vec::new(),
            variant_idx: 0,
        }];
        apply(&mut world, Some(&view(&fields)), test_origin());
        let sw = sprite(&world, form_swatch(0));
        assert!(sw.visible, "the colour field draws a swatch");
        assert_eq!(sw.tint, [1.0, 0.0, 0.0, 1.0], "swatch tint is the RGB text");
        let ti = input(&world, form_input(0));
        assert!(ti.visible, "the editable text field still shows");
        assert!(
            sw.x >= ti.x + ti.width,
            "the swatch sits outside the field's opaque background box"
        );
    }

    #[test]
    fn plain_vector_field_has_no_swatch() {
        let mut world = injected_world();
        let fields = [FormField {
            key: "position".into(),
            kind: FieldKind::Vec {
                len: 3,
                color: false,
            },
            initial: "0, 0, 0".into(),
            boolval: false,
            variants: Vec::new(),
            variant_idx: 0,
        }];
        apply(&mut world, Some(&view(&fields)), test_origin());
        assert!(!sprite_visible(&world, form_swatch(0)));
        assert!(input(&world, form_input(0)).visible);
    }

    // An enum / ref field renders as a cycling button captioned with the current
    // variant and hit-tests to a cycle; a large variant set opens a dropdown.
    #[test]
    fn choice_fields_cycle_or_open_a_dropdown() {
        let o = test_origin();
        let mut world = injected_world();
        let small = [FormField {
            key: "align".into(),
            kind: FieldKind::Enum,
            initial: String::new(),
            boolval: false,
            variants: vec!["left".into(), "center".into(), "right".into()],
            variant_idx: 1,
        }];
        let v = view(&small);
        apply(&mut world, Some(&v), o);
        assert!(
            sprite_visible(&world, form_toggle_bg(0)),
            "cycle button shows"
        );
        let cap = label(&world, form_enum_label(0));
        assert!(cap.visible && cap.content == "center");
        assert!(
            !input(&world, form_input(0)).visible,
            "an enum has no editable text field"
        );
        let c = form_control_rect(o, 0);
        assert_eq!(
            hit_test(&v, c[0] + 5.0, c[1] + 5.0, o),
            Some(FormAction::CycleField(0))
        );
        let big = [ref_field(CYCLE_MAX + 1)];
        let vb = view(&big);
        assert_eq!(
            hit_test(&vb, c[0] + 5.0, c[1] + 5.0, o),
            Some(FormAction::OpenFieldDropdown(0)),
            "a large variant set opens a dropdown"
        );
    }

    // An open value dropdown is modal: its option rows pick (through the scroll
    // offset), anything else closes it.
    #[test]
    fn open_field_dropdown_picks_options_and_is_modal() {
        let o = test_origin();
        let fields = [ref_field(CYCLE_MAX + 3)];
        let mut v = view(&fields);
        v.field_dropdown = Some(0);
        let opt = field_option_rect(o, 0, 2);
        assert_eq!(
            hit_test(&v, opt[0] + 5.0, opt[1] + 5.0, o),
            Some(FormAction::PickFieldOption(2))
        );
        let n = name_rect(o);
        assert_eq!(
            hit_test(&v, n[0] + 5.0, n[1] + 5.0, o),
            Some(FormAction::CloseOverlays),
            "a click off the option list dismisses it"
        );
        v.field_dropdown_scroll = 3;
        let r0 = field_option_rect(o, 0, 0);
        assert_eq!(
            hit_test(&v, r0[0] + 5.0, r0[1] + 5.0, o),
            Some(FormAction::PickFieldOption(3)),
            "visible row 0 is option `scroll + 0`"
        );
    }

    // Laying out an open dropdown shows its backing + option rows and hides the
    // slots it floats over (their opaque TextInput boxes would paint over it).
    #[test]
    fn open_dropdown_hides_covered_slots_and_shows_options() {
        let mut world = injected_world();
        let fields = [
            ref_field(CYCLE_MAX + 2),
            field("intensity", FieldKind::Float),
        ];
        let mut v = view(&fields);
        v.field_dropdown = Some(0);
        apply(&mut world, Some(&v), test_origin());
        assert!(sprite_visible(&world, DROP_BG), "dropdown backing shows");
        assert!(sprite_visible(&world, drop_row_bg(0)), "option row shows");
        let opt0 = label(&world, drop_row_label(0));
        assert!(opt0.visible && opt0.content == form::NONE_LABEL);
        assert!(
            sprite_visible(&world, form_toggle_bg(0)),
            "the open field's button stays"
        );
        assert!(
            !input(&world, form_input(1)).visible,
            "the covered text field is hidden while the dropdown is open"
        );
        assert!(
            !label(&world, form_row_label(1)).visible,
            "the covered slot's caption is hidden too"
        );
    }

    // An array header renders its element count + red remove / green add buttons,
    // and the buttons hit-test to the add / remove actions.
    #[test]
    fn array_header_renders_count_and_buttons_and_hit_tests() {
        let o = test_origin();
        let mut world = injected_world();
        let fields = [FormField {
            key: "waves".into(),
            kind: FieldKind::Array,
            initial: String::new(),
            boolval: false,
            variants: Vec::new(),
            variant_idx: 3,
        }];
        let v = view(&fields);
        apply(&mut world, Some(&v), o);
        assert!(
            sprite_visible(&world, form_toggle_bg(0)),
            "add button shows"
        );
        assert!(
            sprite_visible(&world, form_swatch(0)),
            "remove button shows"
        );
        let cap = label(&world, form_enum_label(0));
        assert!(cap.visible && cap.content == "(3)");
        assert!(!input(&world, form_input(0)).visible);
        let add = array_add_rect(o, 0);
        let rem = array_remove_rect(o, 0);
        assert_eq!(
            hit_test(&v, add[0] + 3.0, add[1] + 3.0, o),
            Some(FormAction::AddArrayElement(0))
        );
        assert_eq!(
            hit_test(&v, rem[0] + 3.0, rem[1] + 3.0, o),
            Some(FormAction::RemoveArrayElement(0))
        );
    }

    // A validation error shows on the reserved status line; hiding the panel
    // blanks everything including the typed fields.
    #[test]
    fn error_line_shows_and_hide_all_blanks() {
        let fields = float_fields(1);
        let mut world = injected_world();
        let mut v = view(&fields);
        v.form_error = Some("Invalid argument");
        apply(&mut world, Some(&v), test_origin());
        let status = label(&world, FORM_STATUS);
        assert!(status.visible);
        assert_eq!(status.content, "Invalid argument");
        apply(&mut world, None, [0.0, 0.0]);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
        assert!(world.query::<TextInput>().all(|t| !t.visible));
    }

    // The floating value dropdown draws above the slot chrome (insertion order).
    #[test]
    fn dropdown_is_injected_after_the_slot_chrome() {
        let sprites = all_sprite_ids();
        let spos = |id: AssetId| sprites.iter().position(|&x| x == id).unwrap();
        let last_slot = (0..form::FIELD_POOL)
            .map(form_toggle_bg)
            .chain((0..form::FIELD_POOL).map(form_swatch))
            .map(spos)
            .max()
            .unwrap();
        assert!(
            spos(DROP_BG) > last_slot,
            "dropdown backing above the slots"
        );
        let first_drop_row = (0..MAX_DROP_ROWS).map(drop_row_bg).map(spos).min().unwrap();
        assert!(spos(DROP_BG) < first_drop_row, "backing under its rows");
        let labels = all_label_ids();
        let lpos = |id: AssetId| labels.iter().position(|&x| x == id).unwrap();
        let last_slot_label = (0..form::FIELD_POOL)
            .map(form_row_label)
            .chain((0..form::FIELD_POOL).map(form_enum_label))
            .map(lpos)
            .max()
            .unwrap();
        for r in 0..MAX_DROP_ROWS {
            assert!(lpos(drop_row_label(r)) > last_slot_label);
        }
    }
}

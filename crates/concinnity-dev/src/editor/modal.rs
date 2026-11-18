//! The confirmation dialog: a centered modal panel with a wrapped message, an
//! optional name field, an optional checkbox, and a row of two or three
//! buttons, built from engine widgets on a modal root that dims the screen
//! and takes all of its input while open. The open / press / close flow lives
//! in `hook/drive/modal.rs`. Not a registered panel: the dialog has no title
//! bar, drag, focus rank, or View toggle. A click away is not a cancel, since
//! the dialog guards destructive actions.

use concinnity_engine::components::{
    Binding, Button as UiButton, Checkbox, Layout, LayoutAlign, LayoutAnchor, LayoutChild,
    LayoutDirection, LayoutSize, Sprite, TextField, TextLabel, UiEvent, UiValue,
};
use concinnity_engine::ecs::asset_id::AssetId;
use concinnity_engine::ecs::{Ref, World};

use super::theme;
use super::ui::build::fill;
use super::ui::{UiBuilder, state};
use super::widget;

pub(crate) const MAX_BUTTONS: usize = 3;

const PANEL_W: f32 = 380.0;
const PAD: f32 = 14.0;
// The message box's height in wrapped lines; longer text ends in an ellipsis.
const MSG_LINES: u32 = 3;
const BTN_W: f32 = 100.0;
const BTN_H: f32 = 28.0;
const BTN_GAP: f32 = 10.0;
const FIELD_H: f32 = 28.0;
const CHECK_H: f32 = 42.0;

// The destructive-action red, matching the editor's other confirm chips.
const DANGER_TINT: [f32; 4] = [0.68, 0.26, 0.28, 1.0];
const DANGER_HOVER_TINT: [f32; 4] = [0.80, 0.32, 0.34, 1.0];

// The state the dialog shows, and the events its controls send.
const MESSAGE: &str = "editor.dialog.message";
const CHECKED: &str = "editor.dialog.checked";
const CHECK: &str = "editor.dialog.check";
const PRESS: &str = "editor.dialog.button";

// One dialog button. `danger` draws it in the destructive red so the intent
// is unmistakable; `action` is what pressing it does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Button {
    pub(crate) label: String,
    pub(crate) danger: bool,
    pub(crate) action: Action,
}

// What pressing a dialog button does; the hook runs it as the dialog closes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Action {
    // Close without acting (a Cancel / No button).
    Dismiss,
    // A Worlds-panel decision the dialog was guarding.
    Worlds(super::worlds::WorldsConfirm),
    // Name the untitled world the session is on, with whatever the dialog's
    // field holds, and save it there.
    NameWorld,
    // Leave the Shader source panel's file for `then`, writing it first when
    // `save`.
    LeaveShaderSource {
        save: bool,
        then: super::panels::shader::source::Leave,
    },
    // Delete the Shader of this name, and its files when the dialog's box is
    // checked.
    DeleteShader(String),
    // Delete the volumes reading the distance field at this path, and the
    // file when the dialog's box is checked.
    DeleteSdfField(String),
}

// The dialog's checkbox: its caption, whether it is on, whether it can be
// changed, and a note saying why not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Check {
    pub(crate) caption: String,
    pub(crate) on: bool,
    pub(crate) enabled: bool,
    pub(crate) note: Option<String>,
}

impl Check {
    // Flip the state; a disabled checkbox stays as it is.
    pub(crate) fn toggle(&mut self) {
        if self.enabled {
            self.on = !self.on;
        }
    }
}

// An open dialog: its message, its buttons left to right, whether it carries
// a name field (a prompt) and what is typed there, its checkbox, if any, and
// whether the world's dialog shows it yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Dialog {
    pub(crate) message: String,
    pub(crate) buttons: Vec<Button>,
    pub(crate) field: bool,
    pub(crate) text: String,
    pub(crate) check: Option<Check>,
    pub(crate) shown: bool,
}

// What the user did in the dialog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DialogEvent {
    // Pressed button `i`.
    Press(usize),
    // Flipped the checkbox.
    Check,
}

// What a dialog event asks for, if `event` is one.
pub(crate) fn event(event: &UiEvent) -> Option<DialogEvent> {
    match (event.name.as_str(), &event.value) {
        (PRESS, Some(UiValue::Int(i))) => usize::try_from(*i).ok().map(DialogEvent::Press),
        (CHECK, _) => Some(DialogEvent::Check),
        _ => None,
    }
}

// The dialog's height for its controls: a prompt is taller by its field, and
// a dialog with a checkbox by that.
fn height(field: bool, check: bool) -> f32 {
    let row = |on: bool, h: f32| if on { h + PAD } else { 0.0 };
    PAD + MSG_LINES as f32 * widget::LINE_H
        + PAD
        + row(field, FIELD_H)
        + row(check, CHECK_H)
        + BTN_H
        + PAD
}

// The dialog built into one world.
#[derive(Debug, Clone, Default)]
pub(crate) struct BuiltDialog {
    pub(crate) root: AssetId,
    field_row: AssetId,
    field: AssetId,
    check_row: AssetId,
    check: AssetId,
    buttons: Vec<AssetId>,
    pub(crate) elements: Vec<AssetId>,
}

fn row(ui: &mut UiBuilder<'_>, child: LayoutChild, height: f32) -> Option<(LayoutChild, AssetId)> {
    let row = ui.row(vec![child], height)?;
    let id = row.item?.id();
    Some((row, id))
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<BuiltDialog> {
    ui.take_pushed();
    let surface = ui.sprite(Sprite {
        tint: theme::CHROME_TINT,
        corner_radius: theme::PANEL_RADIUS,
        border_width: theme::PANEL_BORDER_WIDTH,
        border_color: theme::PANEL_BORDER_TINT,
        ..Default::default()
    })?;
    let message = ui.bound_label(
        MESSAGE,
        theme::LABEL,
        concinnity_engine::components::TextAlign::Left,
    )?;
    let message_id = message.item?.id();
    if let Some(label) = ui.get_mut::<TextLabel>(message_id) {
        label.wrap_width = PANEL_W - 2.0 * PAD;
        label.max_lines = MSG_LINES;
    }
    let message = LayoutChild {
        height: LayoutSize::Fixed(MSG_LINES as f32 * widget::LINE_H),
        ..message
    };
    let field = ui.text_field(TextField {
        height: FIELD_H,
        placeholder: "world name".to_string(),
        ..Default::default()
    })?;
    let field_id = field.item?.id();
    let (field_row, field_row_id) = row(ui, field, FIELD_H)?;
    let check = ui.checkbox(
        Checkbox {
            checked: Some(Binding::State(CHECKED.to_string())),
            on_change: CHECK.to_string(),
            ..Default::default()
        },
        CHECK_H,
    )?;
    let check_id = check.item?.id();
    let (check_row, check_row_id) = row(ui, check, CHECK_H)?;
    let mut buttons = Vec::with_capacity(MAX_BUTTONS);
    let mut button_children = Vec::with_capacity(MAX_BUTTONS);
    for i in 0..MAX_BUTTONS {
        let value = Some(UiValue::Int(i32::try_from(i).unwrap_or(i32::MAX)));
        let button = ui.button("", PRESS, value, [BTN_W, BTN_H])?;
        buttons.push(button.item?.id());
        button_children.push(button);
    }
    let button_row = ui.layout(Layout {
        direction: LayoutDirection::Row,
        children: button_children,
        gap: BTN_GAP,
        justify: LayoutAlign::End,
        ..Default::default()
    })?;
    let button_row = LayoutChild {
        item: Some(Ref::new(button_row)),
        width: LayoutSize::Fill(1.0),
        height: LayoutSize::Fixed(BTN_H),
    };
    let column = ui.layout(Layout {
        direction: LayoutDirection::Column,
        children: vec![message, field_row, check_row, button_row],
        gap: PAD,
        padding: PAD,
        ..Default::default()
    })?;
    let root = ui.layout(Layout {
        direction: LayoutDirection::Stack,
        children: vec![fill(surface), fill(column)],
        width: LayoutSize::Fixed(PANEL_W),
        height: LayoutSize::Fixed(height(false, false)),
        anchor: LayoutAnchor::Center,
        modal: true,
        visible: false,
        ..Default::default()
    })?;
    Some(BuiltDialog {
        root,
        field_row: field_row_id,
        field: field_id,
        check_row: check_row_id,
        check: check_id,
        buttons,
        elements: ui.take_pushed(),
    })
}

fn set_visible(world: &mut World, id: AssetId, visible: bool) {
    if let Some(layout) = world.get_mut_by_id::<Layout>(id) {
        state::set_if_changed(&mut layout.visible, visible);
    }
}

// Bring the dialog in line with `dialog`, or hide it.
pub(crate) fn sync(world: &mut World, built: &BuiltDialog, dialog: Option<&Dialog>) {
    let Some(d) = dialog else {
        set_visible(world, built.root, false);
        return;
    };
    if let Some(root) = world.get_mut_by_id::<Layout>(built.root) {
        state::set_if_changed(&mut root.visible, true);
        state::set_if_changed(
            &mut root.height,
            LayoutSize::Fixed(height(d.field, d.check.is_some())),
        );
    }
    state::set_text(world, MESSAGE, format_args!("{}", d.message));
    set_visible(world, built.field_row, d.field);
    if let Some(f) = world.get_mut_by_id::<TextField>(built.field) {
        if f.content != d.text {
            f.content.clone_from(&d.text);
            f.caret = d.text.chars().count();
        }
        // A prompt's field always owns the keyboard: the dialog takes all
        // input, so nothing else can be typing into anything.
        state::set_if_changed(&mut f.focused, d.field);
    }
    set_visible(world, built.check_row, d.check.is_some());
    if let Some(check) = &d.check {
        state::set_bool(world, CHECKED, check.on);
        if let Some(c) = world.get_mut_by_id::<Checkbox>(built.check) {
            if c.label != check.caption {
                c.label.clone_from(&check.caption);
            }
            let note = check.note.as_deref().unwrap_or("");
            if c.note != note {
                c.note = note.to_string();
            }
            state::set_if_changed(&mut c.disabled, !check.enabled);
        }
    }
    for (i, &id) in built.buttons.iter().enumerate() {
        let Some(b) = world.get_mut_by_id::<UiButton>(id) else {
            continue;
        };
        let shown = d.buttons.get(i);
        state::set_if_changed(&mut b.visible, shown.is_some());
        if let Some(button) = shown {
            if b.label != button.label {
                b.label.clone_from(&button.label);
            }
            let (fill, hover) = if button.danger {
                (DANGER_TINT, DANGER_HOVER_TINT)
            } else {
                (theme::BUTTON_TINT, theme::HOVER_TINT)
            };
            state::set_if_changed(&mut b.style.background, fill);
            state::set_if_changed(&mut b.style.hover, hover);
        }
    }
}

// What is typed into the dialog's name field.
pub(crate) fn typed<'w>(world: &'w World, built: &BuiltDialog) -> Option<&'w str> {
    world
        .get_by_id::<TextField>(built.field)
        .map(|f| f.content.as_str())
}

#[cfg(test)]
mod tests;

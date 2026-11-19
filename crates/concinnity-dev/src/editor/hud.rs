//! The editor HUD's top bar: a slim full-width strip in the shared chrome tint
//! (matching the floating panels, like a modern macOS toolbar) holding the
//! simulation transport (Play / Pause, Step, Stop -- `editor/sim.rs`) centered
//! and the Undo / Redo, Display, View and Save chips at its right end. This
//! lives in the editor crate (not in a client ECS system) so no editor code is
//! compiled into the shipped runtime: the bar is engine widgets the hook keeps
//! in step from its `FrameHook` tick, which runs only under `cn edit`. On macOS
//! the window's traffic-light buttons float over the bar's left end (the editor
//! drops the OS title bar there), so the bar doubles as their backdrop. The
//! whole HUD toggles with F1 (see `hook.rs`).

use concinnity_engine::components::{
    Binding, Button, Layout, LayoutAlign, LayoutAnchor, LayoutChild, LayoutDirection, LayoutSize,
    Sprite, UiEvent,
};
use concinnity_engine::ecs::asset_id::AssetId;
use concinnity_engine::ecs::{Ref, World};

use super::sim::SimState;
use super::theme;
use super::ui::build::emit;
use super::ui::{UiBuilder, state};

pub(crate) const BAR_H: f32 = 30.0;
pub(crate) const BTN_H: f32 = 22.0;
const SAVE_W: f32 = 64.0;
const VIEW_W: f32 = 72.0;
const DISPLAY_W: f32 = 72.0;
const HISTORY_W: f32 = 56.0;
const SIM_W: f32 = 56.0;
const GAP: f32 = 8.0;
const MARGIN: f32 = 8.0;

const SAVE_TINT_ACTIVE: [f32; 4] = [0.72, 0.18, 0.22, 1.0];
const LABEL_ACTIVE: [f32; 3] = [1.0, 1.0, 1.0];
// The armed Stop chip: a run's state is there to discard.
const STOP_TINT_ARMED: [f32; 4] = [0.44, 0.22, 0.24, 1.0];

// The state the chips show, and the events they send.
const VIEW_OPEN: &str = "editor.bar.view_open";
const DISPLAY_OPEN: &str = "editor.bar.display_open";
const PLAYING: &str = "editor.bar.playing";
const PLAY_LABEL: &str = "editor.bar.play_label";
const SAVE: &str = "editor.bar.save";
const UNDO: &str = "editor.bar.undo";
const REDO: &str = "editor.bar.redo";
const VIEW: &str = "editor.bar.view";
const DISPLAY: &str = "editor.bar.display";
const PLAY: &str = "editor.bar.play";
const STEP: &str = "editor.bar.step";
const STOP: &str = "editor.bar.stop";

// Per-frame top-bar state the hook hands to `sync`.
#[derive(Debug, Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "a per-frame snapshot of independent top-bar chip states"
)]
pub(crate) struct HudState {
    // Are there unsaved edits (Save active)?
    pub dirty: bool,
    // Is there a history step to unwind / replay (Undo / Redo active)?
    pub undo: bool,
    pub redo: bool,
    // Is the View panel open (accents the View chip)?
    pub view_open: bool,
    // Is the Display menu open (accents the Display chip)?
    pub display_open: bool,
    // Where the simulation transport stands (drives the Play / Stop chips).
    pub sim: SimState,
    // Is the whole HUD shown (F1 toggle)?
    pub visible: bool,
}

// A click the top bar resolved to one of its controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HudAction {
    // The Save button, while there are edits to persist.
    Save,
    // The Undo / Redo buttons, while their history stack has a step.
    Undo,
    Redo,
    // The View button: open / close the View panel.
    ToggleView,
    // The Display button: open / close the Display menu (view mode + show
    // flags).
    ToggleDisplay,
    // The transport: Play / Pause, advance one frame, and (while a run's
    // state is there to discard) restore the authored state.
    PlayPause,
    Step,
    Stop,
}

// The action a top-bar event asks for, if `event` is one.
pub(crate) fn action(event: &UiEvent) -> Option<HudAction> {
    Some(match event.name.as_str() {
        SAVE => HudAction::Save,
        UNDO => HudAction::Undo,
        REDO => HudAction::Redo,
        VIEW => HudAction::ToggleView,
        DISPLAY => HudAction::ToggleDisplay,
        PLAY => HudAction::PlayPause,
        STEP => HudAction::Step,
        STOP => HudAction::Stop,
        _ => return None,
    })
}

// The y where the body region (the floating panels' default anchors) begins:
// just below the bar.
pub(crate) fn body_top() -> f32 {
    BAR_H + 8.0
}

// The bar built into one world: its root, the chips whose look follows the
// hook, and every element it holds.
#[derive(Debug, Clone, Default)]
pub(crate) struct TopBar {
    pub(crate) root: AssetId,
    save: AssetId,
    undo: AssetId,
    redo: AssetId,
    stop: AssetId,
}

// A chip captioned `label` sending `event`, shown pressed while `active`
// holds.
fn chip(
    ui: &mut UiBuilder<'_>,
    label: &str,
    event: &str,
    width: f32,
    active: Option<&str>,
) -> Option<(LayoutChild, AssetId)> {
    let button = Button {
        label: label.to_string(),
        action: Some(emit(event, None)),
        active: active.map(|a| Binding::State(a.to_string())),
        ..Default::default()
    };
    chip_with(ui, button, width)
}

// `button` as a chip, pressed in the accent.
fn chip_with(ui: &mut UiBuilder<'_>, button: Button, width: f32) -> Option<(LayoutChild, AssetId)> {
    let child = ui.button_with(button, [width, BTN_H])?;
    let id = child.item?.id();
    if let Some(b) = ui.get_mut::<Button>(id) {
        b.style.selected = theme::ACCENT_TINT;
    }
    Some((child, id))
}

// A row of chips across the bar, placed by `justify`.
fn strip(
    ui: &mut UiBuilder<'_>,
    children: Vec<LayoutChild>,
    justify: LayoutAlign,
) -> Option<LayoutChild> {
    let id = ui.layout(Layout {
        direction: LayoutDirection::Row,
        children,
        gap: GAP,
        padding: MARGIN,
        justify,
        align: LayoutAlign::Center,
        ..Default::default()
    })?;
    Some(LayoutChild {
        item: Some(Ref::new(id)),
        width: LayoutSize::Fill(1.0),
        height: LayoutSize::Fill(1.0),
    })
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<TopBar> {
    ui.take_pushed();
    let backdrop = ui.sprite(Sprite {
        tint: theme::CHROME_TINT,
        ..Default::default()
    })?;
    let play = Button {
        text_binding: Some(Binding::State(PLAY_LABEL.to_string())),
        action: Some(emit(PLAY, None)),
        active: Some(Binding::State(PLAYING.to_string())),
        ..Default::default()
    };
    let (play, _) = chip_with(ui, play, SIM_W)?;
    let (step, _) = chip(ui, "Step", STEP, SIM_W, None)?;
    let (stop, stop_id) = chip(ui, "Stop", STOP, SIM_W, None)?;
    let transport = strip(ui, vec![play, step, stop], LayoutAlign::Center)?;
    let (undo, undo_id) = chip(ui, "Undo", UNDO, HISTORY_W, None)?;
    let (redo, redo_id) = chip(ui, "Redo", REDO, HISTORY_W, None)?;
    let (display, _) = chip(ui, "Display", DISPLAY, DISPLAY_W, Some(DISPLAY_OPEN))?;
    let (view, _) = chip(ui, "View", VIEW, VIEW_W, Some(VIEW_OPEN))?;
    let (save, save_id) = chip(ui, "Save", SAVE, SAVE_W, None)?;
    let document = strip(ui, vec![undo, redo, display, view, save], LayoutAlign::End)?;
    let root = ui.layout(Layout {
        direction: LayoutDirection::Stack,
        children: vec![super::ui::build::fill(backdrop), transport, document],
        width: LayoutSize::Fill(1.0),
        height: LayoutSize::Fixed(BAR_H),
        anchor: LayoutAnchor::Top,
        front: true,
        ..Default::default()
    })?;
    // What the bar pushed is its own, not the next panel's.
    ui.take_pushed();
    Some(TopBar {
        root,
        save: save_id,
        undo: undo_id,
        redo: redo_id,
        stop: stop_id,
    })
}

// A chip's fill and whether it takes clicks.
fn set_chip(world: &mut World, id: AssetId, background: [f32; 4], text: [f32; 3], live: bool) {
    if let Some(b) = world.get_mut_by_id::<Button>(id) {
        state::set_if_changed(&mut b.style.background, background);
        state::set_if_changed(&mut b.style.text, text);
        state::set_if_changed(&mut b.disabled, !live);
    }
}

// Bring the bar in line with `hud`: the chips' looks, and the whole bar's
// visibility (F1).
pub(crate) fn sync(world: &mut World, bar: &TopBar, hud: HudState) {
    if let Some(root) = world.get_mut_by_id::<Layout>(bar.root) {
        state::set_if_changed(&mut root.visible, hud.visible);
    }
    if !hud.visible {
        return;
    }
    let playing = hud.sim == SimState::Playing;
    state::set_bool(world, VIEW_OPEN, hud.view_open);
    state::set_bool(world, DISPLAY_OPEN, hud.display_open);
    state::set_bool(world, PLAYING, playing);
    let play = if playing { "Pause" } else { "Play" };
    state::set_text(world, PLAY_LABEL, format_args!("{play}"));
    let (save_tint, save_text) = if hud.dirty {
        (SAVE_TINT_ACTIVE, LABEL_ACTIVE)
    } else {
        (theme::BUTTON_TINT, theme::LABEL)
    };
    set_chip(world, bar.save, save_tint, save_text, hud.dirty);
    set_chip(world, bar.undo, theme::BUTTON_TINT, theme::LABEL, hud.undo);
    set_chip(world, bar.redo, theme::BUTTON_TINT, theme::LABEL, hud.redo);
    let armed = hud.sim != SimState::Stopped;
    let stop_tint = if armed {
        STOP_TINT_ARMED
    } else {
        theme::BUTTON_TINT
    };
    set_chip(world, bar.stop, stop_tint, theme::LABEL, armed);
}

#[cfg(test)]
mod tests;

// src/editor/hud.rs
//
// The editor HUD's top bar: a slim full-width strip in the shared chrome tint
// (matching the floating panels, like a modern macOS toolbar) holding the Save
// and View buttons as compact rounded chips at its right end. This lives in the
// editor crate (not in a client ECS system) so no editor code is compiled into
// the shipped runtime: the HUD is driven from the editor's `DebugHook` tick,
// which runs only under `cn editor`.
//
// These are plain `Sprite` + `TextLabel` components (injected by `inject.rs` at
// reserved ids). Each frame the hook re-anchors the bar to the window width
// from the live viewport and hit-tests clicks. Save persists + live-swaps the
// world; View opens / closes the View panel (`view.rs`), which in turn toggles
// the Assets, Preview, and Templates panels. The simulation transport
// (Play / Pause, Step, Stop -- `editor/sim.rs`) sits centered in the bar.
// Running in the tick (before the world step) means the layout applies the
// same frame GraphicsSystem draws it. The whole HUD toggles with F1 (see
// `hook.rs`).

use super::registry::ID_BASE;
use super::sim::SimState;
use super::theme;
use super::widget::{self, place_rounded, place_sprite, point_in};
use crate::assets::{FrameInput, TextAlign};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;
pub(crate) const SAVE_BUTTON: AssetId = AssetId(ID_BASE);
pub(crate) const SAVE_LABEL: AssetId = AssetId(ID_BASE + 1);
pub(crate) const VIEW_BUTTON: AssetId = AssetId(ID_BASE + 2);
pub(crate) const VIEW_LABEL: AssetId = AssetId(ID_BASE + 3);
pub(crate) const BAR_BG: AssetId = AssetId(ID_BASE + 4);
pub(crate) const UNDO_BUTTON: AssetId = AssetId(ID_BASE + 5);
pub(crate) const UNDO_LABEL: AssetId = AssetId(ID_BASE + 6);
pub(crate) const REDO_BUTTON: AssetId = AssetId(ID_BASE + 7);
pub(crate) const REDO_LABEL: AssetId = AssetId(ID_BASE + 8);
pub(crate) const PLAY_BUTTON: AssetId = AssetId(ID_BASE + 9);
pub(crate) const PLAY_LABEL: AssetId = AssetId(ID_BASE + 10);
pub(crate) const STEP_BUTTON: AssetId = AssetId(ID_BASE + 11);
pub(crate) const STEP_LABEL: AssetId = AssetId(ID_BASE + 12);
pub(crate) const STOP_BUTTON: AssetId = AssetId(ID_BASE + 13);
pub(crate) const STOP_LABEL: AssetId = AssetId(ID_BASE + 14);
pub(crate) const DISPLAY_BUTTON: AssetId = AssetId(ID_BASE + 15);
pub(crate) const DISPLAY_LABEL: AssetId = AssetId(ID_BASE + 16);

// Bar + button geometry, in window pixels. The bar spans the window top edge;
// the chips sit vertically centered at its right end. On macOS the window's
// traffic-light buttons float over the bar's left end (the editor drops the OS
// title bar there), so the bar doubles as their backdrop.
pub(crate) const BAR_H: f32 = 30.0;
pub(crate) const BTN_H: f32 = 22.0;
const SAVE_W: f32 = 64.0;
const VIEW_W: f32 = 72.0;
const DISPLAY_W: f32 = 72.0;
const HISTORY_W: f32 = 56.0;
const SIM_W: f32 = 56.0;
const GAP: f32 = 8.0;
const MARGIN: f32 = 8.0;

// Vertical offset of a chip's label from the chip top, centering the scaled
// editor text without measuring glyphs here (the font metrics live on
// GraphicsSystem).
pub(crate) const LABEL_TOP: f32 = BTN_H * 0.5 - theme::TEXT_HALF;

const SAVE_TINT_ACTIVE: [f32; 4] = [0.72, 0.18, 0.22, 1.0];
const LABEL_ACTIVE: [f32; 3] = [1.0, 1.0, 1.0];
// The armed Stop chip: a run's state is there to discard.
const STOP_TINT_ARMED: [f32; 4] = [0.44, 0.22, 0.24, 1.0];

// Per-frame top-bar state the hook hands to `apply_layout` and `hit_test`.
#[derive(Debug, Clone, Copy)]
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
    // A click on the bar that hits no chip: swallowed so it cannot reach the
    // world behind the bar.
    Consume,
}

// The chip rects (`[x, y, w, h]`, window pixels) for a `vw`-wide window: the
// document chips packed right-to-left from the bar's right end, the transport
// centered (clear of the macOS traffic lights at the left end). Pure: the
// layout pass and the hit test both derive from it.
pub(crate) struct BarLayout {
    pub save: [f32; 4],
    pub view: [f32; 4],
    pub display: [f32; 4],
    pub redo: [f32; 4],
    pub undo: [f32; 4],
    pub play: [f32; 4],
    pub step: [f32; 4],
    pub stop: [f32; 4],
}

pub(crate) fn layout(vw: f32) -> BarLayout {
    let y = (BAR_H - BTN_H) * 0.5;
    let save = [vw - MARGIN - SAVE_W, y, SAVE_W, BTN_H];
    let view = [save[0] - GAP - VIEW_W, y, VIEW_W, BTN_H];
    let display = [view[0] - GAP - DISPLAY_W, y, DISPLAY_W, BTN_H];
    let redo = [display[0] - GAP * 2.0 - HISTORY_W, y, HISTORY_W, BTN_H];
    let undo = [redo[0] - GAP - HISTORY_W, y, HISTORY_W, BTN_H];
    let play = [(vw - SIM_W * 3.0 - GAP * 2.0) * 0.5, y, SIM_W, BTN_H];
    let step = [play[0] + SIM_W + GAP, y, SIM_W, BTN_H];
    let stop = [step[0] + SIM_W + GAP, y, SIM_W, BTN_H];
    BarLayout {
        save,
        view,
        display,
        redo,
        undo,
        play,
        step,
        stop,
    }
}

// The y where the body region (the floating panels' default anchors) begins:
// just below the bar.
pub(crate) fn body_top() -> f32 {
    BAR_H + 8.0
}

// Resolve a top-bar click at `(mx, my)` for a `vw`-wide window. Pure -- the hook
// maps the action to a method and updates its own flags. Returns `None` for a
// click the top bar does not own (the hook then offers it to the panels). An
// inert chip (clean Save, empty history stack) swallows its click like the bar.
pub(crate) fn hit_test(
    mx: f32,
    my: f32,
    clicked: bool,
    state: HudState,
    vw: f32,
) -> Option<HudAction> {
    if !clicked || vw <= 0.0 {
        return None;
    }
    let bar = layout(vw);
    if state.dirty && point_in(mx, my, bar.save) {
        Some(HudAction::Save)
    } else if state.undo && point_in(mx, my, bar.undo) {
        Some(HudAction::Undo)
    } else if state.redo && point_in(mx, my, bar.redo) {
        Some(HudAction::Redo)
    } else if point_in(mx, my, bar.view) {
        Some(HudAction::ToggleView)
    } else if point_in(mx, my, bar.display) {
        Some(HudAction::ToggleDisplay)
    } else if point_in(mx, my, bar.play) {
        Some(HudAction::PlayPause)
    } else if point_in(mx, my, bar.step) {
        Some(HudAction::Step)
    } else if state.sim != SimState::Stopped && point_in(mx, my, bar.stop) {
        Some(HudAction::Stop)
    } else if my < BAR_H && mx >= 0.0 && mx < vw {
        // The bar itself: swallow the click so it cannot fall through to the
        // world (and the hook dismisses any open overlays).
        Some(HudAction::Consume)
    } else {
        None
    }
}

// Re-anchor the bar to the window width from the live viewport and colour the
// Save + View chips by state. Hides the entire HUD when `state.visible` is
// false (F1). A no-op until a `FrameInput` exists (frame 0) or a zero-width
// window.
pub(crate) fn apply_layout(world: &mut World, state: HudState) {
    if !state.visible {
        hide_all(world);
        return;
    }
    let Some(input) = world.query::<FrameInput>().last().cloned() else {
        return;
    };
    let vw = input.viewport[0];
    if vw <= 0.0 {
        return;
    }
    let bar = layout(vw);
    let save_tint = if state.dirty {
        SAVE_TINT_ACTIVE
    } else {
        theme::BUTTON_TINT
    };
    let save_color = if state.dirty {
        LABEL_ACTIVE
    } else {
        theme::LABEL_DIM
    };
    let view_tint = if state.view_open {
        theme::ACCENT_TINT
    } else {
        theme::BUTTON_TINT
    };
    let display_tint = if state.display_open {
        theme::ACCENT_TINT
    } else {
        theme::BUTTON_TINT
    };
    let step_color = |avail: bool| {
        if avail {
            LABEL_ACTIVE
        } else {
            theme::LABEL_DIM
        }
    };
    // The transport: the Play chip carries the accent (and reads "Pause")
    // while the world runs; the Stop chip arms red while a run's state is
    // there to discard, dimmed-not-hidden otherwise so the bar keeps its
    // shape.
    let playing = state.sim == SimState::Playing;
    let stop_armed = state.sim != SimState::Stopped;
    let play_tint = if playing {
        theme::ACCENT_TINT
    } else {
        theme::BUTTON_TINT
    };
    let stop_tint = if stop_armed {
        STOP_TINT_ARMED
    } else {
        theme::BUTTON_TINT
    };

    place_sprite(
        world,
        BAR_BG,
        [0.0, 0.0, vw, BAR_H],
        theme::CHROME_TINT,
        true,
    );
    place_rounded(
        world,
        SAVE_BUTTON,
        bar.save,
        save_tint,
        theme::CONTROL_RADIUS,
        true,
    );
    place_rounded(
        world,
        VIEW_BUTTON,
        bar.view,
        view_tint,
        theme::CONTROL_RADIUS,
        true,
    );
    place_rounded(
        world,
        DISPLAY_BUTTON,
        bar.display,
        display_tint,
        theme::CONTROL_RADIUS,
        true,
    );
    place_rounded(
        world,
        UNDO_BUTTON,
        bar.undo,
        theme::BUTTON_TINT,
        theme::CONTROL_RADIUS,
        true,
    );
    place_rounded(
        world,
        REDO_BUTTON,
        bar.redo,
        theme::BUTTON_TINT,
        theme::CONTROL_RADIUS,
        true,
    );
    place_rounded(
        world,
        PLAY_BUTTON,
        bar.play,
        play_tint,
        theme::CONTROL_RADIUS,
        true,
    );
    place_rounded(
        world,
        STEP_BUTTON,
        bar.step,
        theme::BUTTON_TINT,
        theme::CONTROL_RADIUS,
        true,
    );
    place_rounded(
        world,
        STOP_BUTTON,
        bar.stop,
        stop_tint,
        theme::CONTROL_RADIUS,
        true,
    );
    place_caption(
        world,
        PLAY_LABEL,
        centered(bar.play),
        if playing { "Pause" } else { "Play" },
        LABEL_ACTIVE,
    );
    place_caption(world, STEP_LABEL, centered(bar.step), "Step", LABEL_ACTIVE);
    place_caption(
        world,
        STOP_LABEL,
        centered(bar.stop),
        "Stop",
        step_color(stop_armed),
    );
    place_label(
        world,
        SAVE_LABEL,
        centered(bar.save),
        save_color,
        TextAlign::Center,
        true,
    );
    place_label(
        world,
        VIEW_LABEL,
        centered(bar.view),
        LABEL_ACTIVE,
        TextAlign::Center,
        true,
    );
    place_label(
        world,
        DISPLAY_LABEL,
        centered(bar.display),
        LABEL_ACTIVE,
        TextAlign::Center,
        true,
    );
    place_label(
        world,
        UNDO_LABEL,
        centered(bar.undo),
        step_color(state.undo),
        TextAlign::Center,
        true,
    );
    place_label(
        world,
        REDO_LABEL,
        centered(bar.redo),
        step_color(state.redo),
        TextAlign::Center,
        true,
    );
}

// Every injected top-bar sprite / label id, so the F1-hidden pass can blank it.
fn all_sprite_ids() -> Vec<AssetId> {
    vec![
        BAR_BG,
        SAVE_BUTTON,
        VIEW_BUTTON,
        DISPLAY_BUTTON,
        UNDO_BUTTON,
        REDO_BUTTON,
        PLAY_BUTTON,
        STEP_BUTTON,
        STOP_BUTTON,
    ]
}
fn all_label_ids() -> Vec<AssetId> {
    vec![
        SAVE_LABEL,
        VIEW_LABEL,
        DISPLAY_LABEL,
        UNDO_LABEL,
        REDO_LABEL,
        PLAY_LABEL,
        STEP_LABEL,
        STOP_LABEL,
    ]
}

// Every top-bar element id (sprites + labels), so the hook can pin the whole bar
// to the top draw layer above the floating panels.
pub(crate) fn all_ids() -> Vec<AssetId> {
    all_sprite_ids()
        .into_iter()
        .chain(all_label_ids())
        .collect()
}

fn hide_all(world: &mut World) {
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &[]);
}

fn centered(rect: [f32; 4]) -> [f32; 2] {
    [rect[0] + rect[2] * 0.5, rect[1] + LABEL_TOP]
}

// Position + colour + retitle a transport chip's label (the Play chip reads
// "Pause" while the world runs).
fn place_caption(world: &mut World, id: AssetId, pos: [f32; 2], content: &str, color: [f32; 3]) {
    if let Some(l) = widget::label_mut(world, id) {
        if l.content != content {
            l.content = content.to_string();
        }
        l.x = pos[0];
        l.y = pos[1];
        l.align = TextAlign::Center;
        l.color = color;
        l.visible = true;
    }
}

// Position + colour + show/hide a fixed-content label (a top-bar chip).
fn place_label(
    world: &mut World,
    id: AssetId,
    pos: [f32; 2],
    color: [f32; 3],
    align: TextAlign,
    visible: bool,
) {
    if let Some(l) = widget::label_mut(world, id) {
        l.x = pos[0];
        l.y = pos[1];
        l.align = align;
        l.color = color;
        l.visible = visible;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextLabel};

    fn state(dirty: bool, view: bool, visible: bool) -> HudState {
        HudState {
            dirty,
            undo: false,
            redo: false,
            view_open: view,
            display_open: false,
            sim: SimState::Stopped,
            visible,
        }
    }

    fn hud_world(vw: f32, mouse: (f32, f32)) -> World {
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
        world.add_component(FrameInput {
            viewport: [vw, 720.0],
            mouse_x: mouse.0,
            mouse_y: mouse.1,
            ..Default::default()
        });
        world
    }

    fn sprite(world: &World, id: AssetId) -> Sprite {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == id)
            .cloned()
            .expect("sprite present")
    }

    // The chips pack right-to-left inside the bar without overlapping.
    #[test]
    fn layout_packs_the_chips_inside_the_bar() {
        let bar = layout(1280.0);
        assert_eq!(
            bar.save[0] + bar.save[2],
            1280.0 - MARGIN,
            "Save inset from right"
        );
        assert_eq!(
            bar.view[0] + bar.view[2],
            bar.save[0] - GAP,
            "View left of Save"
        );
        assert!(
            bar.redo[0] + bar.redo[2] < bar.view[0],
            "Redo left of View with a wider break"
        );
        assert_eq!(
            bar.undo[0] + bar.undo[2],
            bar.redo[0] - GAP,
            "Undo left of Redo"
        );
        assert!(
            bar.save[1] > 0.0 && bar.save[1] + bar.save[3] < BAR_H,
            "centered in bar"
        );
        assert!(body_top() > BAR_H, "panels anchor below the bar");
    }

    fn mid(r: [f32; 4]) -> (f32, f32) {
        (r[0] + r[2] * 0.5, r[1] + r[3] * 0.5)
    }

    #[test]
    fn hit_test_resolves_each_control() {
        let bar = layout(1280.0);
        let (sx, sy) = mid(bar.save);
        assert_eq!(
            hit_test(sx, sy, true, state(true, false, true), 1280.0),
            Some(HudAction::Save)
        );
        assert_eq!(
            hit_test(sx, sy, true, state(false, false, true), 1280.0),
            Some(HudAction::Consume),
            "a clean Save chip is inert; the bar still swallows the click"
        );
        let (vx, vy) = mid(bar.view);
        assert_eq!(
            hit_test(vx, vy, true, state(false, false, true), 1280.0),
            Some(HudAction::ToggleView)
        );
        // Empty bar area: swallowed, never reaching the world behind the bar.
        assert_eq!(
            hit_test(100.0, BAR_H * 0.5, true, state(false, false, true), 1280.0),
            Some(HudAction::Consume)
        );
        // Below the bar is not top-bar territory: the click falls through.
        assert_eq!(
            hit_test(
                1180.0,
                BAR_H + 10.0,
                true,
                state(false, false, true),
                1280.0
            ),
            None
        );
    }

    #[test]
    fn hit_test_resolves_undo_redo_only_while_armed() {
        let bar = layout(1280.0);
        let armed = HudState {
            undo: true,
            redo: true,
            ..state(false, false, true)
        };
        let (ux, uy) = mid(bar.undo);
        let (rx, ry) = mid(bar.redo);
        assert_eq!(hit_test(ux, uy, true, armed, 1280.0), Some(HudAction::Undo));
        assert_eq!(hit_test(rx, ry, true, armed, 1280.0), Some(HudAction::Redo));
        // Empty stacks: the chips are inert, the bar swallows the clicks.
        let inert = state(false, false, true);
        assert_eq!(
            hit_test(ux, uy, true, inert, 1280.0),
            Some(HudAction::Consume)
        );
        assert_eq!(
            hit_test(rx, ry, true, inert, 1280.0),
            Some(HudAction::Consume)
        );
    }

    // The transport packs centered in the bar without touching the
    // right-packed document chips.
    #[test]
    fn layout_centers_the_transport() {
        let bar = layout(1280.0);
        let cluster = [bar.play[0], bar.stop[0] + bar.stop[2]];
        let mid = (cluster[0] + cluster[1]) * 0.5;
        assert!((mid - 640.0).abs() < 0.5, "centered: {cluster:?}");
        assert_eq!(bar.step[0], bar.play[0] + bar.play[2] + GAP);
        assert_eq!(bar.stop[0], bar.step[0] + bar.step[2] + GAP);
        assert!(
            bar.stop[0] + bar.stop[2] < bar.undo[0],
            "clear of the history chips"
        );
    }

    #[test]
    fn hit_test_resolves_the_transport() {
        let bar = layout(1280.0);
        let (px, py) = mid(bar.play);
        let (tx, ty) = mid(bar.step);
        let (sx, sy) = mid(bar.stop);
        let stopped = state(false, false, true);
        assert_eq!(
            hit_test(px, py, true, stopped, 1280.0),
            Some(HudAction::PlayPause)
        );
        assert_eq!(
            hit_test(tx, ty, true, stopped, 1280.0),
            Some(HudAction::Step)
        );
        assert_eq!(
            hit_test(sx, sy, true, stopped, 1280.0),
            Some(HudAction::Consume),
            "Stop is inert with nothing to discard"
        );
        let paused = HudState {
            sim: SimState::Paused,
            ..stopped
        };
        assert_eq!(
            hit_test(sx, sy, true, paused, 1280.0),
            Some(HudAction::Stop)
        );
    }

    // The Play chip reads Pause and takes the accent while the world runs;
    // the Stop chip arms red.
    #[test]
    fn apply_layout_marks_a_running_transport() {
        let mut world = hud_world(1280.0, (0.0, 0.0));
        apply_layout(&mut world, state(false, false, true));
        let label = |world: &World, id: AssetId| {
            world
                .query::<TextLabel>()
                .find(|l| l.asset_id == id)
                .cloned()
                .expect("label present")
        };
        assert_eq!(label(&world, PLAY_LABEL).content, "Play");
        assert_eq!(sprite(&world, PLAY_BUTTON).tint, theme::BUTTON_TINT);
        assert_eq!(sprite(&world, STOP_BUTTON).tint, theme::BUTTON_TINT);

        let playing = HudState {
            sim: SimState::Playing,
            ..state(false, false, true)
        };
        apply_layout(&mut world, playing);
        assert_eq!(label(&world, PLAY_LABEL).content, "Pause");
        assert_eq!(sprite(&world, PLAY_BUTTON).tint, theme::ACCENT_TINT);
        assert_eq!(sprite(&world, STOP_BUTTON).tint, STOP_TINT_ARMED);
    }

    #[test]
    fn hit_test_ignores_non_clicks_and_zero_width() {
        assert_eq!(
            hit_test(1240.0, 20.0, false, state(true, false, true), 1280.0),
            None
        );
        assert_eq!(
            hit_test(0.0, 0.0, true, state(true, false, true), 0.0),
            None
        );
    }

    // The bar spans the window and the chips are shown, Save coloured by dirty.
    #[test]
    fn apply_layout_shows_bar_and_chips() {
        let mut world = hud_world(1024.0, (0.0, 0.0));
        apply_layout(&mut world, state(true, false, true));
        let bar = sprite(&world, BAR_BG);
        assert!(bar.visible);
        assert_eq!((bar.x, bar.width, bar.height), (0.0, 1024.0, BAR_H));
        assert_eq!(bar.tint, theme::CHROME_TINT, "bar matches the panel chrome");
        assert_eq!(sprite(&world, SAVE_BUTTON).tint, SAVE_TINT_ACTIVE);
        assert!(
            sprite(&world, SAVE_BUTTON).corner_radius > 0.0,
            "chips are rounded"
        );
        for id in [BAR_BG, SAVE_BUTTON, VIEW_BUTTON] {
            assert!(sprite(&world, id).visible, "{id:?} shown");
        }
    }

    // A clean Save chip recedes to the neutral chip tint.
    #[test]
    fn apply_layout_dims_a_clean_save() {
        let mut world = hud_world(1280.0, (0.0, 0.0));
        apply_layout(&mut world, state(false, false, true));
        assert_eq!(sprite(&world, SAVE_BUTTON).tint, theme::BUTTON_TINT);
    }

    // The View chip takes the accent while the View panel is open.
    #[test]
    fn apply_layout_marks_open_view() {
        let mut world = hud_world(1280.0, (0.0, 0.0));
        apply_layout(&mut world, state(false, true, true));
        assert_eq!(sprite(&world, VIEW_BUTTON).tint, theme::ACCENT_TINT);
    }

    // F1 hidden: every top-bar element is blanked.
    #[test]
    fn apply_layout_hidden_blanks_everything() {
        let mut world = hud_world(1280.0, (0.0, 0.0));
        apply_layout(&mut world, state(true, true, true));
        apply_layout(&mut world, state(true, true, false));
        for id in all_sprite_ids() {
            assert!(!sprite(&world, id).visible, "sprite {id:?} hidden");
        }
    }
}

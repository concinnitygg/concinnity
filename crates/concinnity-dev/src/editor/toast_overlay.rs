// src/editor/toast_overlay.rs
//
// The toast stack's card geometry and draw: transient message cards anchored
// above the viewport's bottom-right corner, newest nearest the corner, with a
// "+N more" row above the stack when the queue outgrows the visible cap. Pure
// placement over the queue state in `editor/notify.rs`; the per-frame drive
// and click routing live in `hook/notify_drive.rs`. Not a registered panel:
// toasts have no title bar, drag, focus rank, or View toggle, and they draw
// above all of that chrome.

use super::notify::{Level, Stack};
use super::registry::ID_BASE;
use super::widget::{self, point_in};
use super::{hud, theme};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

// Reserved id family: the block above the palette's (0x8000), leaving 0x9000
// free for the next panel.
const BASE: u32 = ID_BASE + 0xA000;

const fn card_bg(slot: usize) -> AssetId {
    AssetId(BASE + slot as u32)
}
const fn card_accent(slot: usize) -> AssetId {
    AssetId(BASE + 0x10 + slot as u32)
}
const fn card_msg(slot: usize) -> AssetId {
    AssetId(BASE + 0x20 + slot as u32)
}
const OVERFLOW_BG: AssetId = AssetId(BASE + 0x30);
const OVERFLOW_LABEL: AssetId = AssetId(BASE + 0x31);

// Operation cards: label plus a progress bar. Concurrent long operations are
// rare (the cook guard serializes cooks), so the pool is small.
pub(crate) const MAX_OPS: usize = 2;
const fn op_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x40 + i as u32)
}
const fn op_label(i: usize) -> AssetId {
    AssetId(BASE + 0x44 + i as u32)
}
const fn op_track(i: usize) -> AssetId {
    AssetId(BASE + 0x48 + i as u32)
}
const fn op_fill(i: usize) -> AssetId {
    AssetId(BASE + 0x4C + i as u32)
}

pub(crate) const CARD_W: f32 = 320.0;
// Two wrapped message lines plus the vertical padding.
const CARD_H: f32 = PAD * 2.0 + 2.0 * widget::LINE_H;
const PAD: f32 = 8.0;
const GAP: f32 = 8.0;
const MARGIN: f32 = 14.0;
const ACCENT_W: f32 = 4.0;
const OVERFLOW_H: f32 = 22.0;
// The operation card's progress-bar height.
const BAR_H: f32 = 6.0;

// Stack row `row`'s rect: row 0 hugs the bottom-right corner, higher rows
// stack upward. Clamped below the top bar for tiny viewports.
fn row_rect(vp: [f32; 2], row: usize) -> [f32; 4] {
    let x = (vp[0] - CARD_W - MARGIN).max(0.0);
    let y = vp[1] - MARGIN - CARD_H - row as f32 * (CARD_H + GAP);
    [x, y.max(hud::BAR_H), CARD_W, CARD_H]
}

// Operation cards fill the lowest rows; message cards stack above them.
pub(crate) fn op_rect(vp: [f32; 2], i: usize) -> [f32; 4] {
    row_rect(vp, i)
}

pub(crate) fn card_rect(vp: [f32; 2], ops: usize, slot: usize) -> [f32; 4] {
    row_rect(vp, ops + slot)
}

// The "+N more" row above the top-most card.
pub(crate) fn overflow_rect(vp: [f32; 2], rows: usize) -> [f32; 4] {
    let top = row_rect(vp, rows.saturating_sub(1));
    [
        top[0],
        (top[1] - GAP - OVERFLOW_H).max(hud::BAR_H),
        CARD_W,
        OVERFLOW_H,
    ]
}

fn shown_ops(stack: &Stack) -> usize {
    stack.ops.len().min(MAX_OPS)
}

fn accent_tint(level: Level) -> [f32; 4] {
    match level {
        Level::Info => theme::NOTIFY_INFO,
        Level::Success => theme::NOTIFY_SUCCESS,
        Level::Warning => theme::NOTIFY_WARNING,
        Level::Error => theme::NOTIFY_ERROR,
    }
}

fn faded(tint: [f32; 4], alpha: f32) -> [f32; 4] {
    [tint[0], tint[1], tint[2], tint[3] * alpha]
}

// Labels carry no alpha channel, so a fading toast dims its text toward the
// dark card behind it instead.
fn faded_text(color: [f32; 3], alpha: f32) -> [f32; 3] {
    [color[0] * alpha, color[1] * alpha, color[2] * alpha]
}

// What a press over the stack landed on. An operation card consumes its press
// (nothing to click through to) but carries no action.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Hit {
    Op,
    Card(usize),
    Overflow,
}

pub(crate) fn hit(mx: f32, my: f32, vp: [f32; 2], stack: &Stack) -> Option<Hit> {
    let ops = shown_ops(stack);
    for i in 0..ops {
        if point_in(mx, my, op_rect(vp, i)) {
            return Some(Hit::Op);
        }
    }
    for slot in 0..stack.cards.len() {
        if point_in(mx, my, card_rect(vp, ops, slot)) {
            return Some(Hit::Card(slot));
        }
    }
    if stack.overflow > 0 && point_in(mx, my, overflow_rect(vp, ops + stack.cards.len())) {
        return Some(Hit::Overflow);
    }
    None
}

// The indeterminate sweep: a short fill segment gliding back and forth along
// the track, positioned purely from the operation's age.
fn sweep_offset(phase: f32, span: f32) -> f32 {
    let t = (phase / 1.2).fract();
    let pos = if t < 0.5 { 2.0 * t } else { 2.0 - 2.0 * t };
    pos * span
}

pub(crate) fn apply(world: &mut World, stack: &Stack, vp: [f32; 2], mouse: [f32; 2]) {
    let ops = shown_ops(stack);
    for i in 0..MAX_OPS {
        match (i < ops).then(|| &stack.ops[i]) {
            Some(op) => {
                let r = op_rect(vp, i);
                widget::place_bordered(
                    world,
                    op_bg(i),
                    r,
                    theme::CHROME_TINT,
                    theme::PANEL_BORDER_TINT,
                    theme::PANEL_BORDER_WIDTH,
                );
                widget::place_left_label(
                    world,
                    op_label(i),
                    [r[0] + PAD, r[1] + PAD * 0.5],
                    &widget::clip_text(&op.label, 34),
                    theme::LABEL,
                    true,
                );
                let track = [
                    r[0] + PAD,
                    r[1] + r[3] - PAD - BAR_H,
                    r[2] - 2.0 * PAD,
                    BAR_H,
                ];
                widget::place_rounded(
                    world,
                    op_track(i),
                    track,
                    theme::BUTTON_TINT,
                    theme::CONTROL_RADIUS,
                    true,
                );
                let fill = match op.fraction {
                    Some(f) => [track[0], track[1], track[2] * f.clamp(0.0, 1.0), BAR_H],
                    None => {
                        let w = track[2] * 0.3;
                        [
                            track[0] + sweep_offset(op.phase, track[2] - w),
                            track[1],
                            w,
                            BAR_H,
                        ]
                    }
                };
                widget::place_rounded(
                    world,
                    op_fill(i),
                    fill,
                    theme::ACCENT_TINT,
                    theme::CONTROL_RADIUS,
                    true,
                );
            }
            None => hide_op(world, i),
        }
    }
    for slot in 0..super::notify::MAX_VISIBLE {
        match stack.cards.get(slot) {
            Some(card) => {
                let r = card_rect(vp, ops, slot);
                let hovered = point_in(mouse[0], mouse[1], r);
                let border = if hovered {
                    accent_tint(card.level)
                } else {
                    theme::PANEL_BORDER_TINT
                };
                widget::place_bordered(
                    world,
                    card_bg(slot),
                    r,
                    faded(theme::CHROME_TINT, card.alpha),
                    faded(border, card.alpha),
                    theme::PANEL_BORDER_WIDTH,
                );
                widget::place_rounded(
                    world,
                    card_accent(slot),
                    [r[0], r[1], ACCENT_W, r[3]],
                    faded(accent_tint(card.level), card.alpha),
                    theme::CONTROL_RADIUS,
                    true,
                );
                widget::place_message(
                    world,
                    card_msg(slot),
                    [
                        r[0] + ACCENT_W + PAD,
                        r[1] + PAD,
                        r[2] - ACCENT_W - 2.0 * PAD,
                        2.0 * widget::LINE_H,
                    ],
                    &card.message,
                    faded_text(theme::LABEL, card.alpha),
                    true,
                );
            }
            None => hide_card(world, slot),
        }
    }
    if stack.overflow > 0 {
        let r = overflow_rect(vp, ops + stack.cards.len());
        widget::place_rounded(
            world,
            OVERFLOW_BG,
            r,
            theme::BUTTON_TINT,
            theme::CONTROL_RADIUS,
            true,
        );
        widget::place_left_label(
            world,
            OVERFLOW_LABEL,
            [r[0] + PAD, r[1] + r[3] * 0.5 - theme::TEXT_HALF],
            &format!("+{} more", stack.overflow),
            theme::LABEL_DIM,
            true,
        );
    } else {
        widget::set_sprite_visible(world, OVERFLOW_BG, false);
        widget::set_label_visible(world, OVERFLOW_LABEL, false);
    }
}

fn hide_card(world: &mut World, slot: usize) {
    widget::set_sprite_visible(world, card_bg(slot), false);
    widget::set_sprite_visible(world, card_accent(slot), false);
    widget::set_label_visible(world, card_msg(slot), false);
}

fn hide_op(world: &mut World, i: usize) {
    widget::set_sprite_visible(world, op_bg(i), false);
    widget::set_sprite_visible(world, op_track(i), false);
    widget::set_sprite_visible(world, op_fill(i), false);
    widget::set_label_visible(world, op_label(i), false);
}

pub(crate) fn hide(world: &mut World) {
    for slot in 0..super::notify::MAX_VISIBLE {
        hide_card(world, slot);
    }
    for i in 0..MAX_OPS {
        hide_op(world, i);
    }
    widget::set_sprite_visible(world, OVERFLOW_BG, false);
    widget::set_label_visible(world, OVERFLOW_LABEL, false);
}

pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    (0..super::notify::MAX_VISIBLE)
        .flat_map(|s| [card_bg(s), card_accent(s)])
        .chain((0..MAX_OPS).flat_map(|i| [op_bg(i), op_track(i), op_fill(i)]))
        .chain([OVERFLOW_BG])
        .collect()
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    (0..super::notify::MAX_VISIBLE)
        .map(card_msg)
        .chain((0..MAX_OPS).map(op_label))
        .chain([OVERFLOW_LABEL])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::notify::{Card, Level, OpCard, Stack};
    use super::*;

    fn stack_of(n: usize, overflow: usize) -> Stack {
        Stack {
            ops: Vec::new(),
            cards: (0..n)
                .map(|i| Card {
                    level: Level::Info,
                    message: format!("m{i}"),
                    alpha: 1.0,
                })
                .collect(),
            overflow,
        }
    }

    #[test]
    fn cards_stack_upward_from_the_corner() {
        let vp = [1280.0, 720.0];
        let r0 = card_rect(vp, 0, 0);
        let r1 = card_rect(vp, 0, 1);
        assert!(r0[0] + r0[2] <= vp[0], "inside the right edge");
        assert!(r0[1] + r0[3] <= vp[1], "inside the bottom edge");
        assert!(r1[1] < r0[1], "slot 1 sits above slot 0");
        assert_eq!(r0[0], r1[0], "one column");
        // An op card takes the bottom row and pushes the messages up.
        assert_eq!(op_rect(vp, 0), r0);
        assert_eq!(card_rect(vp, 1, 0), r1);
    }

    #[test]
    fn overflow_row_sits_above_the_top_card() {
        let vp = [1280.0, 720.0];
        let top = card_rect(vp, 0, 2);
        let o = overflow_rect(vp, 3);
        assert!(o[1] + o[3] <= top[1], "above the highest card");
    }

    #[test]
    fn a_tiny_viewport_clamps_below_the_top_bar() {
        let vp = [400.0, 120.0];
        for slot in 0..4 {
            assert!(card_rect(vp, 0, slot)[1] >= hud::BAR_H);
        }
        assert!(overflow_rect(vp, 4)[1] >= hud::BAR_H);
    }

    #[test]
    fn hit_resolves_ops_then_cards_then_overflow_then_misses() {
        let vp = [1280.0, 720.0];
        let stack = stack_of(2, 3);
        let r0 = card_rect(vp, 0, 0);
        assert_eq!(
            hit(r0[0] + 2.0, r0[1] + 2.0, vp, &stack),
            Some(Hit::Card(0))
        );
        let o = overflow_rect(vp, 2);
        assert_eq!(hit(o[0] + 2.0, o[1] + 2.0, vp, &stack), Some(Hit::Overflow));
        assert_eq!(hit(5.0, 5.0, vp, &stack), None);
        // Without overflow the row is inert even at its would-be rect.
        let flat = stack_of(2, 0);
        assert_eq!(hit(o[0] + 2.0, o[1] + 2.0, vp, &flat), None);
        // A running op takes the bottom row: it consumes its press, and the
        // cards answer one row higher.
        let mut with_op = stack_of(2, 0);
        with_op.ops.push(OpCard {
            label: "Cooking".to_string(),
            fraction: Some(0.5),
            phase: 0.0,
        });
        assert_eq!(hit(r0[0] + 2.0, r0[1] + 2.0, vp, &with_op), Some(Hit::Op));
        let c0 = card_rect(vp, 1, 0);
        assert_eq!(
            hit(c0[0] + 2.0, c0[1] + 2.0, vp, &with_op),
            Some(Hit::Card(0))
        );
    }

    #[test]
    fn the_indeterminate_sweep_stays_on_the_track() {
        for phase in [0.0, 0.3, 0.6, 0.9, 1.2, 5.7] {
            let off = sweep_offset(phase, 100.0);
            assert!((0.0..=100.0).contains(&off), "phase {phase}: {off}");
        }
        assert_eq!(sweep_offset(0.0, 100.0), 0.0);
        assert_eq!(
            sweep_offset(0.6, 100.0),
            100.0,
            "half period reaches the far end"
        );
    }

    #[test]
    fn id_lists_cover_every_slot_without_repeats() {
        let sprites = all_sprite_ids();
        let labels = all_label_ids();
        let mut all: Vec<AssetId> = sprites.iter().chain(labels.iter()).copied().collect();
        let n = all.len();
        all.sort_by_key(|id| id.0);
        all.dedup();
        assert_eq!(all.len(), n, "no duplicate reserved ids");
        assert_eq!(
            sprites.len(),
            super::super::notify::MAX_VISIBLE * 2 + MAX_OPS * 3 + 1
        );
        assert_eq!(
            labels.len(),
            super::super::notify::MAX_VISIBLE + MAX_OPS + 1
        );
    }
}

// src/editor/behavior_chart.rs
//
// The Behavior panel's chart view: the open behavior's body as cards flowing
// left to right, wired parent to child. `behavior/graph.rs` decides where each
// card sits; this places them and resolves a click back to the card under the
// cursor. The panel's toolbar, palette, and value field are unchanged, because
// a card stands for the same path its outline row does.
//
// Nothing scissors an editor element (clip bands are captured once at init from
// the authored ScrollPanels), so a card panned past the canvas edge is drawn as
// the part still inside it and drops its text once too little is left to read.

use super::behavior::graph::{Card, CardKind, Chart};
use super::behavior::pulse;
use super::behavior_panel::{CHAR_W, LIST_THUMB, LIST_TRACK};
use super::registry::{self, PanelKey};
use super::theme;
use super::widget::{self, place_rounded, point_in};
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::Behavior);
pub(crate) const HINT_LABEL: AssetId = AssetId(BASE + 0x1F0);

// A body wider than this shows what it can and says so; the outline still
// reaches every node.
pub(crate) const CARD_POOL: usize = 32;
// Three segments per elbow, one wire per card past the first.
const SEG_POOL: usize = 96;
const WIRE_LABEL_POOL: usize = 32;

pub(crate) fn card_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x200 + i as u32)
}
pub(crate) fn card_title(i: usize) -> AssetId {
    AssetId(BASE + 0x240 + i as u32)
}
pub(crate) fn card_detail(i: usize) -> AssetId {
    AssetId(BASE + 0x280 + i as u32)
}
pub(crate) fn wire_label(i: usize) -> AssetId {
    AssetId(BASE + 0x2C0 + i as u32)
}
pub(crate) fn segment(i: usize) -> AssetId {
    AssetId(BASE + 0x300 + i as u32)
}
pub(crate) fn card_break(i: usize) -> AssetId {
    AssetId(BASE + 0x360 + i as u32)
}

// Narrow enough that a trigger, a node, and the card that appends after it all
// fit the default width beside the inspector on a small window: adding a node
// is the reason the chart is editable at all, so its card should not start off
// screen.
const CARD_W: f32 = 184.0;
const CARD_H: f32 = 46.0;
// Wide enough for a wire's label to sit in the gap without reaching either card
// it joins. The overview's labels name what carries a relation ("spawns",
// "reads"), so the gap is sized for a word rather than for a branch's "then".
const COL_GAP: f32 = 72.0;
const ROW_GAP: f32 = 14.0;
const PITCH_X: f32 = CARD_W + COL_GAP;
const PITCH_Y: f32 = CARD_H + ROW_GAP;
const MARGIN: f32 = 12.0;
const CARD_PAD: f32 = 8.0;
const WIRE_W: f32 = 2.0;
// How far past the parent a wire turns. Turning early leaves the rest of the
// gap as a clear horizontal run for the wire's label.
const ELBOW_IN: f32 = 12.0;
// That run, measured from where the label starts to the card it points at. A
// label is clipped to it, so a long one can never reach that card however far
// apart the two are.
const LABEL_IN: f32 = ELBOW_IN + 6.0;
const LABEL_RUN: f32 = COL_GAP - LABEL_IN;
// Wider than `CHAR_W`, because here the estimate has to come out long: a label
// measured short is a label drawn over the card it points at.
const LABEL_CHAR_W: f32 = 10.0;
// The longest label the gap holds. A longer one is drawn clipped, so what the
// overview puts on a wire is checked against this (`behavior/relations.rs`)
// rather than guessed.
pub(crate) const LABEL_CHARS: usize = (LABEL_RUN / LABEL_CHAR_W) as usize;
const INDICATOR_H: f32 = 4.0;

const NODE_TINT: [f32; 4] = [0.17, 0.18, 0.23, 1.0];
const TRIGGER_TINT: [f32; 4] = [0.16, 0.30, 0.34, 1.0];
const ADD_TINT: [f32; 4] = [0.13, 0.13, 0.16, 0.85];
const VARIABLE_TINT: [f32; 4] = [0.28, 0.24, 0.16, 1.0];
const ASSET_TINT: [f32; 4] = [0.22, 0.19, 0.36, 1.0];
// A name the world does not answer, tinted like the checker's own banner.
const MISSING_TINT: [f32; 4] = [0.32, 0.14, 0.17, 1.0];
const BORDER_TINT: [f32; 4] = [0.30, 0.32, 0.40, 1.0];
const SELECTED_BORDER: [f32; 4] = [0.55, 0.70, 0.98, 1.0];
const FAULT_BORDER: [f32; 4] = [0.85, 0.42, 0.45, 1.0];
const HOVER_BORDER: [f32; 4] = [0.44, 0.48, 0.60, 1.0];
const WIRE_TINT: [f32; 4] = [0.38, 0.41, 0.52, 1.0];
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
// The breakpoint dot in a card's corner.
const BREAK_TINT: [f32; 4] = [0.85, 0.30, 0.32, 1.0];
const BREAK_R: f32 = 3.5;

pub(crate) struct ChartView<'a> {
    pub chart: &'a Chart,
    // The card the panel's selection lights up, if any.
    pub selected: Option<usize>,
    // The card the checker's complaint is about, if any.
    pub faulted: Option<usize>,
    // Cards whose node just executed, with each pulse's remaining strength;
    // and cards holding a breakpoint. Both empty outside a live-debug session.
    pub pulses: &'a [(usize, f32)],
    pub breakpoints: &'a [usize],
    pub pan: [f32; 2],
    pub mouse: [f32; 2],
    // What to do about the cards that did not fit, which depends on what the
    // chart is of.
    pub overflow: &'static str,
}

// How far the chart reaches, so panning stops at its edge rather than running
// off into empty canvas.
pub(crate) fn extent(chart: &Chart) -> [f32; 2] {
    [
        chart.columns as f32 * PITCH_X - COL_GAP + MARGIN * 2.0,
        chart.rows as f32 * PITCH_Y - ROW_GAP + MARGIN * 2.0,
    ]
}

// Panning depends only on how big the canvas is, not where it sits, so these
// take its size and the placement functions take the band.
pub(crate) fn max_pan(chart: &Chart, canvas: [f32; 2]) -> [f32; 2] {
    let e = extent(chart);
    [(e[0] - canvas[0]).max(0.0), (e[1] - canvas[1]).max(0.0)]
}

pub(crate) fn clamp_pan(pan: [f32; 2], chart: &Chart, canvas: [f32; 2]) -> [f32; 2] {
    let m = max_pan(chart, canvas);
    [pan[0].clamp(0.0, m[0]), pan[1].clamp(0.0, m[1])]
}

// Where a card sits in window pixels, before clamping to the band.
pub(crate) fn card_rect(card: &Card, band: [f32; 4], pan: [f32; 2]) -> [f32; 4] {
    [
        band[0] + MARGIN + card.column as f32 * PITCH_X - pan[0],
        band[1] + MARGIN + card.row as f32 * PITCH_Y - pan[1],
        CARD_W,
        CARD_H,
    ]
}

// The pan that brings `card` fully inside the band, moving no further than it
// has to so selecting a neighbour does not re-centre the whole chart.
pub(crate) fn pan_to(card: &Card, canvas: [f32; 2], pan: [f32; 2], chart: &Chart) -> [f32; 2] {
    let x = MARGIN + card.column as f32 * PITCH_X;
    let y = MARGIN + card.row as f32 * PITCH_Y;
    let want = [
        pan[0].clamp(x + CARD_W + MARGIN - canvas[0], x - MARGIN),
        pan[1].clamp(y + CARD_H + MARGIN - canvas[1], y - MARGIN),
    ];
    clamp_pan(want, chart, canvas)
}

// How wide a canvas showing `columns` columns of cards is, which is how the
// panel picks the width it opens at.
pub(crate) fn width_for(columns: usize) -> f32 {
    columns as f32 * PITCH_X - COL_GAP + MARGIN * 2.0 + 2.0
}

// The canvas: the panel's body band, inset off the border.
pub(crate) fn band(o: [f32; 2], s: [f32; 2], body_top: f32) -> [f32; 4] {
    [
        o[0] + 1.0,
        body_top,
        (s[0] - 2.0).max(0.0),
        (o[1] + s[1] - body_top - 2.0).max(0.0),
    ]
}

// The part of `rect` inside `band`, or `None` when none of it is.
fn visible_part(rect: [f32; 4], band: [f32; 4]) -> Option<[f32; 4]> {
    let x0 = rect[0].max(band[0]);
    let y0 = rect[1].max(band[1]);
    let x1 = (rect[0] + rect[2]).min(band[0] + band[2]);
    let y1 = (rect[1] + rect[3]).min(band[1] + band[3]);
    (x1 > x0 && y1 > y0).then_some([x0, y0, x1 - x0, y1 - y0])
}

fn overlaps(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3]
}

// The card under the cursor, if the cursor is over the canvas at all.
pub(crate) fn hit_card(view: &ChartView, mx: f32, my: f32, band: [f32; 4]) -> Option<usize> {
    if !point_in(mx, my, band) {
        return None;
    }
    view.chart
        .cards
        .iter()
        .take(CARD_POOL)
        .position(|c| point_in(mx, my, card_rect(c, band, view.pan)))
}

pub(crate) fn apply(world: &mut World, view: &ChartView, band: [f32; 4]) {
    layout_wires(world, view, band);
    layout_cards(world, view, band);
    layout_hint(world, view, band);
    layout_indicator(world, view, band);
}

fn layout_cards(world: &mut World, view: &ChartView, band: [f32; 4]) {
    for slot in 0..CARD_POOL {
        let Some(card) = view.chart.cards.get(slot) else {
            hide_card(world, slot);
            continue;
        };
        let rect = card_rect(card, band, view.pan);
        let Some(part) = visible_part(rect, band) else {
            hide_card(world, slot);
            continue;
        };
        let selected = view.selected == Some(slot);
        let faulted = view.faulted == Some(slot);
        let hovered = point_in(view.mouse[0], view.mouse[1], rect);
        // What is wrong outranks what is selected: the selection is legible from
        // the inspector beside the chart, a complaint is only legible here.
        let border = if faulted {
            FAULT_BORDER
        } else if selected {
            SELECTED_BORDER
        } else if hovered {
            HOVER_BORDER
        } else {
            BORDER_TINT
        };
        let pulse_alpha = view
            .pulses
            .iter()
            .find(|(s, _)| *s == slot)
            .map(|(_, a)| *a)
            .unwrap_or(0.0);
        widget::place_bordered(
            world,
            card_bg(slot),
            part,
            pulse::blend(fill(card.kind), pulse_alpha),
            border,
            if selected || faulted { 2.0 } else { 1.0 },
        );
        // The breakpoint dot sits inside the card's top-right corner.
        place_rounded(
            world,
            card_break(slot),
            [
                part[0] + part[2] - BREAK_R * 2.0 - 5.0,
                part[1] + 5.0,
                BREAK_R * 2.0,
                BREAK_R * 2.0,
            ],
            BREAK_TINT,
            BREAK_R,
            view.breakpoints.contains(&slot),
        );
        // Text is placed only in a card still showing its left edge and its full
        // height, so a half-scrolled card truncates cleanly instead of drawing
        // its label over the panel chrome.
        let readable = part[0] <= rect[0] + 0.5 && part[3] >= CARD_H - 0.5;
        let budget = ((part[2] - CARD_PAD * 2.0) / CHAR_W).max(0.0) as usize;
        if !readable || budget < 4 {
            widget::set_label_visible(world, card_title(slot), false);
            widget::set_label_visible(world, card_detail(slot), false);
            continue;
        }
        widget::place_left_label(
            world,
            card_title(slot),
            [part[0] + CARD_PAD, part[1] + 9.0],
            &widget::clip_text(&card.title, budget),
            title_color(card.kind),
            true,
        );
        widget::place_left_label(
            world,
            card_detail(slot),
            [part[0] + CARD_PAD, part[1] + 27.0],
            &widget::clip_text(&card.detail, budget),
            theme::LABEL_DIM,
            !card.detail.is_empty(),
        );
    }
}

fn fill(kind: CardKind) -> [f32; 4] {
    match kind {
        CardKind::Trigger => TRIGGER_TINT,
        CardKind::Node | CardKind::Behavior => NODE_TINT,
        CardKind::Add => ADD_TINT,
        CardKind::Variable => VARIABLE_TINT,
        CardKind::Asset => ASSET_TINT,
        CardKind::Missing => MISSING_TINT,
    }
}

fn title_color(kind: CardKind) -> [f32; 3] {
    match kind {
        CardKind::Add => theme::LABEL_DIM,
        CardKind::Missing => theme::LOG_ERROR,
        _ => theme::HEADING,
    }
}

fn hide_card(world: &mut World, slot: usize) {
    widget::set_sprite_visible(world, card_bg(slot), false);
    widget::set_sprite_visible(world, card_break(slot), false);
    widget::set_label_visible(world, card_title(slot), false);
    widget::set_label_visible(world, card_detail(slot), false);
}

// Each wire is an elbow of axis-aligned segments: out of the parent's right
// edge, across to the child's row, then into its left edge. A child on the
// parent's own row needs only the one run.
fn layout_wires(world: &mut World, view: &ChartView, band: [f32; 4]) {
    let mut seg = 0;
    let mut label = 0;
    // What the labels drawn so far cover, so two wires cannot stack their words
    // in one place: a card in the overview reaches several others, and the ones
    // it reaches on the same row would otherwise share a label position.
    let mut taken: Vec<[f32; 4]> = Vec::new();
    for wire in &view.chart.wires {
        let (Some(from), Some(to)) = (
            view.chart
                .cards
                .get(wire.from)
                .filter(|_| wire.from < CARD_POOL),
            view.chart
                .cards
                .get(wire.to)
                .filter(|_| wire.to < CARD_POOL),
        ) else {
            continue;
        };
        let a = card_rect(from, band, view.pan);
        let b = card_rect(to, band, view.pan);
        let (ax, ay) = (a[0] + a[2], a[1] + a[3] * 0.5);
        let (bx, by) = (b[0], b[1] + b[3] * 0.5);
        let mid = ax + ELBOW_IN;
        for rect in elbow(ax, ay, bx, by, mid) {
            if seg >= SEG_POOL {
                break;
            }
            if let Some(part) = visible_part(rect, band) {
                place_rounded(world, segment(seg), part, WIRE_TINT, 0.0, true);
                seg += 1;
            }
        }
        if let Some(text) = wire.label.as_deref().filter(|_| label < WIRE_LABEL_POOL) {
            // Above the run that enters the card it points at, in the gap just
            // before it: past the turn for a card one column along, and along
            // the run for one further off, which is what keeps two words leaving
            // the same card apart. Either way it crosses neither the wire nor
            // either card.
            let text = widget::clip_text(text, LABEL_CHARS);
            let x = (ax + LABEL_IN).max(bx - LABEL_RUN);
            let w = text.chars().count() as f32 * LABEL_CHAR_W;
            // Above the run, or below it when something is there already: two
            // behaviors reaching the same card enter it along the same run, so
            // one side of it is not room enough for both their words.
            let free = [by - theme::TEXT_HALF - 9.0, by + 9.0]
                .into_iter()
                .map(|y| [x, y, w, theme::TEXT_HALF * 2.0])
                .find(|over| !taken.iter().any(|r| overlaps(*r, *over)));
            if let Some(over) = free.filter(|over| point_in(over[0], over[1], band)) {
                widget::place_left_label(
                    world,
                    wire_label(label),
                    [over[0], over[1]],
                    &text,
                    theme::LABEL_DIM,
                    true,
                );
                taken.push(over);
                label += 1;
            }
        }
    }
    for i in seg..SEG_POOL {
        widget::set_sprite_visible(world, segment(i), false);
    }
    for i in label..WIRE_LABEL_POOL {
        widget::set_label_visible(world, wire_label(i), false);
    }
}

fn elbow(ax: f32, ay: f32, bx: f32, by: f32, mid: f32) -> Vec<[f32; 4]> {
    let half = WIRE_W * 0.5;
    if (by - ay).abs() <= 1.0 {
        return vec![[ax, ay - half, (bx - ax).max(0.0), WIRE_W]];
    }
    vec![
        [ax, ay - half, (mid - ax).max(0.0), WIRE_W],
        [
            mid - half,
            ay.min(by) - half,
            WIRE_W,
            (by - ay).abs() + WIRE_W,
        ],
        [mid - half, by - half, (bx - mid).max(0.0) + half, WIRE_W],
    ]
}

// A body past the card pool says how much is not drawn rather than trailing off
// as though the chart were complete.
fn layout_hint(world: &mut World, view: &ChartView, band: [f32; 4]) {
    let total = view.chart.cards.len();
    let hidden = total.saturating_sub(CARD_POOL);
    widget::place_message(
        world,
        HINT_LABEL,
        [
            band[0] + MARGIN,
            band[1] + band[3] - 18.0,
            (band[2] - MARGIN * 2.0).max(0.0),
            widget::LINE_H,
        ],
        &format!("{hidden} more {}", view.overflow),
        theme::LOG_WARN,
        hidden > 0,
    );
}

// How much of the chart's width is on screen, so a body reaching past the panel
// is visibly wider than the canvas.
fn layout_indicator(world: &mut World, view: &ChartView, band: [f32; 4]) {
    let e = extent(view.chart);
    if e[0] <= band[2] {
        widget::set_sprite_visible(world, LIST_TRACK, false);
        widget::set_sprite_visible(world, LIST_THUMB, false);
        return;
    }
    let y = band[1] + band[3] - INDICATOR_H - 2.0;
    place_rounded(
        world,
        LIST_TRACK,
        [band[0] + MARGIN, y, band[2] - MARGIN * 2.0, INDICATOR_H],
        TRACK_TINT,
        INDICATOR_H * 0.5,
        true,
    );
    let track_w = band[2] - MARGIN * 2.0;
    let thumb_w = (track_w * band[2] / e[0]).max(20.0);
    let max = max_pan(view.chart, [band[2], band[3]])[0].max(1.0);
    let off = (track_w - thumb_w) * (view.pan[0] / max).clamp(0.0, 1.0);
    place_rounded(
        world,
        LIST_THUMB,
        [band[0] + MARGIN + off, y, thumb_w, INDICATOR_H],
        THUMB_TINT,
        INDICATOR_H * 0.5,
        true,
    );
}

pub(crate) fn hide_all(world: &mut World) {
    for id in all_sprite_ids() {
        widget::set_sprite_visible(world, id, false);
    }
    for id in all_label_ids() {
        widget::set_label_visible(world, id, false);
    }
}

pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids: Vec<AssetId> = (0..SEG_POOL).map(segment).collect();
    ids.extend((0..CARD_POOL).map(card_bg));
    ids.extend((0..CARD_POOL).map(card_break));
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![HINT_LABEL];
    ids.extend((0..WIRE_LABEL_POOL).map(wire_label));
    ids.extend((0..CARD_POOL).map(card_title));
    ids.extend((0..CARD_POOL).map(card_detail));
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextLabel};
    use crate::editor::behavior::graph;
    use serde_json::json;

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
        world
    }

    fn branching() -> Chart {
        graph::chart(&json!({
            "on": "tick",
            "do": [
                {"if": {
                    "cond": {"bool": true},
                    "then": [{"show": {"target": "self"}}],
                    "else": [{"hide": {"target": "self"}}],
                }},
                {"save": {}},
            ],
        }))
    }

    fn view<'a>(chart: &'a Chart, pan: [f32; 2]) -> ChartView<'a> {
        ChartView {
            overflow: "nodes -- open the outline to reach them",
            chart,
            selected: None,
            faulted: None,
            pulses: &[],
            breakpoints: &[],
            pan,
            mouse: [-1.0, -1.0],
        }
    }

    const BAND: [f32; 4] = [100.0, 200.0, 600.0, 300.0];

    fn label(world: &World, id: AssetId) -> TextLabel {
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .cloned()
            .unwrap()
    }

    fn sprite(world: &World, id: AssetId) -> Sprite {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == id)
            .cloned()
            .unwrap()
    }

    #[test]
    fn cards_lay_out_left_to_right_at_the_pitch() {
        let chart = branching();
        let trigger = card_rect(&chart.cards[0], BAND, [0.0, 0.0]);
        let branch = card_rect(&chart.cards[1], BAND, [0.0, 0.0]);
        assert_eq!(branch[0] - trigger[0], PITCH_X);
        assert_eq!(branch[1], trigger[1]);
        // The `else` chain sits one row below the branch it leaves.
        let otherwise = chart.cards.iter().find(|c| c.title == "hide").unwrap();
        assert_eq!(
            card_rect(otherwise, BAND, [0.0, 0.0])[1] - branch[1],
            PITCH_Y
        );
    }

    #[test]
    fn panning_moves_the_cards_and_stops_at_the_chart_edge() {
        let chart = branching();
        let at_zero = card_rect(&chart.cards[0], BAND, [0.0, 0.0]);
        let panned = card_rect(&chart.cards[0], BAND, [40.0, 0.0]);
        assert_eq!(at_zero[0] - panned[0], 40.0);
        // The chart is taller than it is wide against this band, so only the
        // horizontal pan is free.
        let e = extent(&chart);
        assert!(e[0] > BAND[2], "{e:?}");
        let canvas = [BAND[2], BAND[3]];
        assert_eq!(clamp_pan([-30.0, -30.0], &chart, canvas), [0.0, 0.0]);
        assert_eq!(
            clamp_pan([9_000.0, 9_000.0], &chart, canvas),
            max_pan(&chart, canvas),
        );
    }

    #[test]
    fn a_card_off_the_band_keeps_its_visible_part_and_drops_its_text() {
        let chart = branching();
        let mut world = injected_world();
        // Pan so the trigger card is cut by the band's left edge.
        apply(&mut world, &view(&chart, [MARGIN + 60.0, 0.0]), BAND);
        let bg = sprite(&world, card_bg(0));
        assert!(bg.visible);
        assert_eq!(bg.x, BAND[0]);
        assert_eq!(bg.width, CARD_W - 60.0);
        assert!(!label(&world, card_title(0)).visible, "text left the card");
        // A card entirely off the canvas is gone, not clamped to the edge.
        apply(
            &mut world,
            &view(&chart, [MARGIN + CARD_W + 40.0, 0.0]),
            BAND,
        );
        assert!(!sprite(&world, card_bg(0)).visible);
    }

    // A wire label drawn over the card it points at or over its own wire is
    // unreadable, which is exactly what a wide card and a narrow gap produce if
    // the elbow turns at the midpoint. `long` relabels every wire with the
    // longest word the overview puts on one, so the gap is checked against what
    // it actually has to hold rather than against the shorter "then".
    #[test]
    fn a_pin_label_clears_both_the_cards_and_its_own_wire() {
        check_wire_labels(branching());
    }

    #[test]
    fn a_long_wire_label_is_kept_inside_the_gap() {
        let mut chart = branching();
        for wire in &mut chart.wires {
            wire.label = Some("reads".to_string());
        }
        check_wire_labels(chart);
    }

    fn check_wire_labels(chart: Chart) {
        let mut world = injected_world();
        let band = [100.0, 200.0, 2_000.0, 600.0];
        apply(&mut world, &view(&chart, [0.0, 0.0]), band);

        let cards: Vec<[f32; 4]> = chart
            .cards
            .iter()
            .map(|c| card_rect(c, band, [0.0, 0.0]))
            .collect();
        let verticals: Vec<[f32; 4]> = (0..SEG_POOL)
            .map(|i| sprite(&world, segment(i)))
            .filter(|s| s.visible && s.width <= WIRE_W + 0.01)
            .map(|s| [s.x, s.y, s.width, s.height])
            .collect();
        assert!(!verticals.is_empty(), "the else branch drops a run");

        let mut checked = 0;
        for i in 0..WIRE_LABEL_POOL {
            let l = label(&world, wire_label(i));
            if !l.visible {
                continue;
            }
            checked += 1;
            let rect = [
                l.x,
                l.y,
                l.content.chars().count() as f32 * CHAR_W,
                theme::TEXT_HALF * 2.0,
            ];
            for card in &cards {
                assert!(
                    !overlaps(rect, *card),
                    "'{}' over a card {card:?}",
                    l.content
                );
            }
            for run in &verticals {
                assert!(!overlaps(rect, *run), "'{}' over a wire {run:?}", l.content);
            }
        }
        assert!(checked >= 2, "both branches are labelled");
    }

    // A map of cards at the given places, wired `from -> to` and labelled: the
    // shapes the overview makes that a body never does.
    fn wired(places: &[(usize, usize)], wires: &[(usize, usize, &str)]) -> Chart {
        Chart {
            cards: places
                .iter()
                .map(|&(column, row)| Card {
                    column,
                    row,
                    title: "card".to_string(),
                    detail: String::new(),
                    kind: CardKind::Behavior,
                    path: Vec::new(),
                    settles: Vec::new(),
                    behavior: None,
                })
                .collect(),
            wires: wires
                .iter()
                .map(|&(from, to, label)| graph::Wire {
                    from,
                    to,
                    label: Some(label.to_string()),
                })
                .collect(),
            columns: places.iter().map(|p| p.0 + 1).max().unwrap_or(0),
            rows: places.iter().map(|p| p.1 + 1).max().unwrap_or(0),
        }
    }

    const WIDE: [f32; 4] = [100.0, 200.0, 2_000.0, 600.0];

    // Every wire label drawn, and the space each takes at the width one is
    // measured at.
    fn label_rects(world: &World) -> Vec<[f32; 4]> {
        (0..WIRE_LABEL_POOL)
            .map(|i| label(world, wire_label(i)))
            .filter(|l| l.visible)
            .map(|l| {
                [
                    l.x,
                    l.y,
                    l.content.chars().count() as f32 * LABEL_CHAR_W,
                    theme::TEXT_HALF * 2.0,
                ]
            })
            .collect()
    }

    // A card in the overview reaches several others at once, and two of them on
    // the same row put their labels in the same place if a label only ever sits
    // beside the card it leaves: what is drawn then is one pile of overprinted
    // words rather than two labels.
    #[test]
    fn two_labels_leaving_one_card_land_apart() {
        let chart = wired(
            &[(0, 0), (1, 0), (3, 0)],
            &[(0, 1, "sets"), (0, 2, "hides")],
        );
        let mut world = injected_world();
        apply(&mut world, &view(&chart, [0.0, 0.0]), WIDE);

        let drawn = label_rects(&world);
        assert_eq!(drawn.len(), 2, "both wires are labelled");
        assert!(!overlaps(drawn[0], drawn[1]), "{drawn:?} share a place");
        // And each still sits in the gap before the card it points at.
        for (r, card) in drawn.iter().zip([&chart.cards[1], &chart.cards[2]]) {
            let card = card_rect(card, WIDE, [0.0, 0.0]);
            assert!(r[0] + r[2] <= card[0], "{r:?} reaches {card:?}");
        }
    }

    // Two behaviors reaching the same asset enter its card along the one run,
    // so their labels take opposite sides of it rather than one being dropped:
    // which of the two does what is the relation the map is drawn for.
    #[test]
    fn two_labels_reaching_one_card_take_both_sides_of_its_run() {
        let chart = wired(
            &[(0, 0), (0, 1), (1, 0)],
            &[(0, 2, "hides"), (1, 2, "shows")],
        );
        let mut world = injected_world();
        apply(&mut world, &view(&chart, [0.0, 0.0]), WIDE);

        let drawn = label_rects(&world);
        assert_eq!(drawn.len(), 2, "neither wire loses its word");
        assert!(!overlaps(drawn[0], drawn[1]), "{drawn:?} share a place");
        // The run they both enter by sits between them.
        let run = card_rect(&chart.cards[2], WIDE, [0.0, 0.0])[1] + CARD_H * 0.5;
        assert!(drawn[0][1] + drawn[0][3] <= run, "{drawn:?}");
        assert!(drawn[1][1] >= run, "{drawn:?}");
    }

    #[test]
    fn a_branch_wire_elbows_and_carries_its_pin_label() {
        let chart = branching();
        let mut world = injected_world();
        apply(&mut world, &view(&chart, [0.0, 0.0]), BAND);
        let labels: Vec<String> = (0..WIRE_LABEL_POOL)
            .map(|i| label(&world, wire_label(i)))
            .filter(|l| l.visible)
            .map(|l| l.content)
            .collect();
        assert!(labels.contains(&"then".to_string()), "{labels:?}");
        assert!(labels.contains(&"else".to_string()), "{labels:?}");
        // Every drawn segment is axis-aligned: one of its sides is the wire's
        // own thickness.
        let drawn: Vec<Sprite> = (0..SEG_POOL)
            .map(|i| sprite(&world, segment(i)))
            .filter(|s| s.visible)
            .collect();
        // A wire between rows turns, and a turn means a vertical run.
        assert!(
            drawn.iter().any(|s| (s.width - WIRE_W).abs() < 0.01),
            "an elbow is more than one run"
        );
        for s in &drawn {
            assert!(
                (s.height - WIRE_W).abs() < 0.01 || (s.width - WIRE_W).abs() < 0.01,
                "{s:?}",
            );
        }
    }

    #[test]
    fn clicking_lands_on_the_card_under_the_cursor() {
        let chart = branching();
        let v = view(&chart, [0.0, 0.0]);
        let target = card_rect(&chart.cards[1], BAND, [0.0, 0.0]);
        let hit = hit_card(&v, target[0] + 4.0, target[1] + 4.0, BAND);
        assert_eq!(hit, Some(1));
        // The gap between two columns is canvas, not a card.
        assert_eq!(
            hit_card(&v, target[0] - COL_GAP * 0.5, target[1] + 4.0, BAND),
            None
        );
        assert_eq!(hit_card(&v, BAND[0] - 5.0, BAND[1] + 5.0, BAND), None);
    }

    #[test]
    fn selecting_a_card_pans_only_as_far_as_it_must() {
        let chart = branching();
        let narrow = [100.0, 200.0, 300.0, 300.0];
        // A card already on screen leaves the pan where it is.
        let canvas = [narrow[2], narrow[3]];
        assert_eq!(
            pan_to(&chart.cards[0], canvas, [0.0, 0.0], &chart),
            [0.0, 0.0]
        );
        // One off the right edge is brought just inside it.
        let far = chart.cards.iter().find(|c| c.title == "save").unwrap();
        let pan = pan_to(far, canvas, [0.0, 0.0], &chart);
        let rect = card_rect(far, narrow, pan);
        assert!(rect[0] >= narrow[0], "{rect:?}");
        assert!(
            rect[0] + rect[2] <= narrow[0] + narrow[2] + 0.01,
            "{rect:?}"
        );
    }

    #[test]
    fn a_body_past_the_pool_says_how_much_is_missing() {
        let body: Vec<serde_json::Value> =
            (0..CARD_POOL + 5).map(|_| json!({"save": {}})).collect();
        let chart = graph::chart(&json!({"on": "start", "do": body}));
        let mut world = injected_world();
        apply(&mut world, &view(&chart, [0.0, 0.0]), BAND);
        let hint = label(&world, HINT_LABEL);
        assert!(hint.visible);
        assert!(hint.content.starts_with("7 more nodes"), "{}", hint.content);
    }
}

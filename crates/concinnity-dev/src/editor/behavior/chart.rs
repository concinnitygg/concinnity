//! A chart view: a `Chart`'s cards flowing left to right, wired parent to
//! child, written as a canvas's shapes. Whoever builds the chart decides where
//! each card sits; this lays the cards and wires out in the chart's own space,
//! tags each card with its index so a press on it reports which, and works out
//! the pans that keep a card in view. The Behavior panel draws one behavior's
//! body and the world's behaviors this way, and the Map panel the world's
//! places.

use concinnity_engine::components::UiShape;

use super::graph::{Card, CardKind, Chart};
use super::pulse;
use crate::editor::theme;
use crate::editor::ui::shapes::{Shape, Shapes};

// Narrow enough that a trigger, a node, and the card that appends after it all
// fit the default width beside the inspector on a small window: adding a node
// is the reason the chart is editable at all, so its card should not start off
// screen.
const CARD_W: f32 = 184.0;
const CARD_H: f32 = 46.0;
// Wide enough for a wire's label to sit in the gap without reaching either card
// it joins. A label names what carries a relation ("reads") or what kind of
// move a wire is ("toggle"), so the gap is sized for an ordinary word rather
// than for a branch's "then".
const COL_GAP: f32 = 78.0;
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
// A generous advance per character, because the estimate has to come out long:
// a label measured short is a label drawn over the card it points at.
const LABEL_CHAR_W: f32 = 10.0;
// The longest label the gap holds. A longer one is drawn clipped, so what the
// overview puts on a wire is checked against this (`behavior/relations.rs`)
// rather than guessed.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a character count from small positive layout constants"
)]
pub(crate) const LABEL_CHARS: usize = (LABEL_RUN / LABEL_CHAR_W) as usize;

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
// The breakpoint dot in a card's corner.
const BREAK_TINT: [f32; 4] = [0.85, 0.30, 0.32, 1.0];
const BREAK_R: f32 = 3.5;
// How much further in a card's line box starts, so the canvas's text inset puts
// its text where the card pads it.
const TEXT_NUDGE: f32 = CARD_PAD - UiShape::TEXT_INSET;
const LINE_BOX: f32 = 18.0;
const TITLE_TOP: f32 = 5.0;
const DETAIL_TOP: f32 = 23.0;

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

// Where a card sits in the chart's own space, before a canvas pans it.
pub(crate) fn card_rect(card: &Card) -> [f32; 4] {
    [
        MARGIN + card.column as f32 * PITCH_X,
        MARGIN + card.row as f32 * PITCH_Y,
        CARD_W,
        CARD_H,
    ]
}

// The pan that brings `card` fully inside a canvas `canvas` big, moving no
// further than it has to so selecting a neighbor does not re-center the chart.
pub(crate) fn pan_to(card: &Card, canvas: [f32; 2], pan: [f32; 2], chart: &Chart) -> [f32; 2] {
    let [x, y, ..] = card_rect(card);
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

// The same for the height a canvas of `rows` stacked cards takes.
pub(crate) fn height_for(rows: usize) -> f32 {
    rows as f32 * PITCH_Y - ROW_GAP + MARGIN * 2.0 + 2.0
}

fn overlaps(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3]
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

// What a chart on a canvas lights up beside its cards.
pub(crate) struct ChartMarks<'a> {
    pub chart: &'a Chart,
    pub selected: Option<usize>,
    pub faulted: Option<usize>,
    // The card under the cursor.
    pub hovered: Option<usize>,
    pub pulses: &'a [(usize, f32)],
    pub breakpoints: &'a [usize],
}

// The chart as canvas shapes, in the chart's own space: a clear frame as big as
// the chart so the canvas pans to its margin, the wires and their words, then
// each card tagged with its index, its breakpoint dot, and its two lines.
// `taken` is scratch for where the wire words went.
pub(crate) fn push(shapes: &mut Shapes<'_>, marks: &ChartMarks<'_>, taken: &mut Vec<[f32; 4]>) {
    let e = extent(marks.chart);
    shapes.push(Shape {
        rect: [0.0, 0.0, e[0], e[1]],
        ..Shape::default()
    });
    push_wires(shapes, marks.chart, taken);
    for (index, card) in marks.chart.cards.iter().enumerate() {
        push_card(shapes, marks, index, card);
    }
}

fn push_card(shapes: &mut Shapes<'_>, marks: &ChartMarks<'_>, index: usize, card: &Card) {
    let rect = card_rect(card);
    let selected = marks.selected == Some(index);
    let faulted = marks.faulted == Some(index);
    // What is wrong outranks what is selected: the selection is legible from
    // the inspector beside the chart, a complaint is only legible here.
    let border = if faulted {
        FAULT_BORDER
    } else if selected {
        SELECTED_BORDER
    } else if marks.hovered == Some(index) {
        HOVER_BORDER
    } else {
        BORDER_TINT
    };
    let pulse_alpha = marks
        .pulses
        .iter()
        .find(|(c, _)| *c == index)
        .map_or(0.0, |(_, a)| *a);
    shapes.push(Shape {
        rect,
        fill: pulse::blend(fill(card.kind), pulse_alpha),
        border,
        border_width: if selected || faulted { 2.0 } else { 1.0 },
        tag: u32::try_from(index).ok(),
        ..Shape::default()
    });
    if marks.breakpoints.contains(&index) {
        shapes.push(Shape {
            rect: [
                rect[0] + rect[2] - BREAK_R * 2.0 - 5.0,
                rect[1] + 5.0,
                BREAK_R * 2.0,
                BREAK_R * 2.0,
            ],
            fill: BREAK_TINT,
            radius: BREAK_R,
            ..Shape::default()
        });
    }
    let line = |top: f32| {
        [
            rect[0] + TEXT_NUDGE,
            rect[1] + top,
            rect[2] - TEXT_NUDGE * 2.0,
            LINE_BOX,
        ]
    };
    shapes.push(Shape {
        rect: line(TITLE_TOP),
        text: &card.title,
        text_color: title_color(card.kind),
        ..Shape::default()
    });
    shapes.push(Shape {
        rect: line(DETAIL_TOP),
        text: &card.detail,
        text_color: theme::LABEL_DIM,
        ..Shape::default()
    });
}

// Each wire is an elbow of axis-aligned runs, with its words above the run
// that enters the card it points at, or below when that is taken.
fn push_wires(shapes: &mut Shapes<'_>, chart: &Chart, taken: &mut Vec<[f32; 4]>) {
    taken.clear();
    for wire in &chart.wires {
        let (Some(from), Some(to)) = (chart.cards.get(wire.from), chart.cards.get(wire.to)) else {
            continue;
        };
        let a = card_rect(from);
        let b = card_rect(to);
        let (ax, ay) = (a[0] + a[2], a[1] + a[3] * 0.5);
        let (bx, by) = (b[0], b[1] + b[3] * 0.5);
        let (runs, count) = elbow_runs(ax, ay, bx, by, ax + ELBOW_IN);
        for rect in &runs[..count] {
            shapes.push(Shape {
                rect: *rect,
                fill: WIRE_TINT,
                ..Shape::default()
            });
        }
        let Some(text) = wire.label.as_deref() else {
            continue;
        };
        let x = (ax + LABEL_IN).max(bx - LABEL_RUN);
        let free = [by - theme::TEXT_HALF - 9.0, by + 9.0]
            .into_iter()
            .map(|y| [x, y, LABEL_RUN, theme::TEXT_HALF * 2.0])
            .find(|over| !taken.iter().any(|r| overlaps(*r, *over)));
        if let Some(over) = free {
            taken.push(over);
            shapes.push(Shape {
                rect: [
                    over[0] - UiShape::TEXT_INSET,
                    over[1],
                    over[2] + UiShape::TEXT_INSET * 2.0,
                    over[3],
                ],
                text,
                text_color: theme::LABEL_DIM,
                ..Shape::default()
            });
        }
    }
}

// The runs of an elbow from `(ax, ay)` to `(bx, by)` turning at `mid`, and how
// many there are: one along a row, three across rows.
fn elbow_runs(ax: f32, ay: f32, bx: f32, by: f32, mid: f32) -> ([[f32; 4]; 3], usize) {
    let half = WIRE_W * 0.5;
    if (by - ay).abs() <= 1.0 {
        return (
            [
                [ax, ay - half, (bx - ax).max(0.0), WIRE_W],
                [0.0; 4],
                [0.0; 4],
            ],
            1,
        );
    }
    (
        [
            [ax, ay - half, (mid - ax).max(0.0), WIRE_W],
            [
                mid - half,
                ay.min(by) - half,
                WIRE_W,
                (by - ay).abs() + WIRE_W,
            ],
            [mid - half, by - half, (bx - mid).max(0.0) + half, WIRE_W],
        ],
        3,
    )
}

#[cfg(test)]
mod tests {
    use super::super::graph;
    use super::*;
    use serde_json::json;

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

    fn plain(chart: &Chart) -> ChartMarks<'_> {
        ChartMarks {
            chart,
            selected: None,
            faulted: None,
            hovered: None,
            pulses: &[],
            breakpoints: &[],
        }
    }

    // The shapes `chart` is written as, and where its wire words went.
    fn drawn(marks: &ChartMarks<'_>) -> (Vec<UiShape>, Vec<[f32; 4]>) {
        let mut list = Vec::new();
        let mut taken = Vec::new();
        let mut shapes = Shapes::new(&mut list);
        push(&mut shapes, marks, &mut taken);
        shapes.finish();
        (list, taken)
    }

    #[test]
    fn cards_lay_out_left_to_right_at_the_pitch() {
        let chart = branching();
        let trigger = card_rect(&chart.cards[0]);
        let branch = card_rect(&chart.cards[1]);
        assert_eq!(branch[0] - trigger[0], PITCH_X);
        assert_eq!(branch[1], trigger[1]);
        // The `else` chain sits one row below the branch it leaves.
        let otherwise = chart.cards.iter().find(|c| c.title == "hide").unwrap();
        assert_eq!(card_rect(otherwise)[1] - branch[1], PITCH_Y);
    }

    #[test]
    fn the_pan_stops_at_the_chart_edge() {
        let chart = branching();
        let canvas = [600.0, 300.0];
        let e = extent(&chart);
        assert!(e[0] > canvas[0], "{e:?}");
        assert_eq!(clamp_pan([-30.0, -30.0], &chart, canvas), [0.0, 0.0]);
        assert_eq!(
            clamp_pan([9_000.0, 9_000.0], &chart, canvas),
            max_pan(&chart, canvas),
        );
    }

    // Every card is a pressable shape tagged with its index, drawn over the
    // wires, and its two lines inside it; the frame first spans the chart.
    #[test]
    fn each_card_is_a_tagged_shape_with_its_lines() {
        let chart = branching();
        let (shapes, _) = drawn(&plain(&chart));
        assert_eq!(&shapes[0].rect[2..], &extent(&chart)[..]);
        let tags: Vec<u32> = shapes.iter().filter_map(|s| s.tag).collect();
        let want: Vec<u32> = (0..chart.cards.len())
            .map(|i| u32::try_from(i).unwrap())
            .collect();
        assert_eq!(tags, want);
        let first_card = shapes.iter().position(|s| s.tag.is_some()).unwrap();
        assert!(
            shapes[..first_card]
                .iter()
                .all(|s| s.text.is_empty() || s.fill[3] == 0.0),
            "the wires draw under the cards"
        );
        assert!(shapes.iter().any(|s| s.text == chart.cards[1].title));
    }

    // What the selection, a fault, and a breakpoint look like on a card.
    #[test]
    fn a_card_shows_its_selection_fault_and_breakpoint() {
        let chart = branching();
        let card =
            |shapes: &[UiShape], i: u32| shapes.iter().find(|s| s.tag == Some(i)).cloned().unwrap();
        let marks = ChartMarks {
            selected: Some(1),
            faulted: Some(2),
            breakpoints: &[1],
            ..plain(&chart)
        };
        let (shapes, _) = drawn(&marks);
        assert_eq!(card(&shapes, 1).border, SELECTED_BORDER);
        assert_eq!(card(&shapes, 2).border, FAULT_BORDER);
        assert_eq!(card(&shapes, 0).border_width, 1.0);
        assert!(shapes.iter().any(|s| s.fill == BREAK_TINT));
    }

    // A wire label drawn over the card it points at or over its own wire is
    // unreadable, which is exactly what a wide card and a narrow gap produce if
    // the elbow turns at the midpoint. The long label is the longest word the
    // overview puts on one, so the gap is checked against what it has to hold.
    #[test]
    fn a_pin_label_clears_both_the_cards_and_its_own_wire() {
        check_wire_labels(&branching());
    }

    #[test]
    fn a_long_wire_label_is_kept_inside_the_gap() {
        let mut chart = branching();
        for wire in &mut chart.wires {
            wire.label = Some("reads".to_string());
        }
        check_wire_labels(&chart);
    }

    fn check_wire_labels(chart: &Chart) {
        let (shapes, labels) = drawn(&plain(chart));
        let cards: Vec<[f32; 4]> = chart.cards.iter().map(card_rect).collect();
        let verticals: Vec<[f32; 4]> = shapes
            .iter()
            .filter(|s| s.fill == WIRE_TINT && s.rect[2] <= WIRE_W + 0.01)
            .map(|s| s.rect)
            .collect();
        assert!(!verticals.is_empty(), "the else branch drops a run");
        assert!(labels.len() >= 2, "both branches are labeled");
        for rect in &labels {
            for card in &cards {
                assert!(!overlaps(*rect, *card), "{rect:?} over a card {card:?}");
            }
            for run in &verticals {
                assert!(!overlaps(*rect, *run), "{rect:?} over a wire {run:?}");
            }
        }
    }

    // A map of cards at the given places, wired `from -> to` and labeled: the
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
                    handle: None,
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

    // A card in the overview reaches several others at once, and two of them on
    // the same row put their labels in the same place if a label only ever sits
    // beside the card it leaves.
    #[test]
    fn two_labels_leaving_one_card_land_apart() {
        let chart = wired(
            &[(0, 0), (1, 0), (3, 0)],
            &[(0, 1, "sets"), (0, 2, "hides")],
        );
        let (_, drawn) = drawn(&plain(&chart));
        assert_eq!(drawn.len(), 2, "both wires are labeled");
        assert!(!overlaps(drawn[0], drawn[1]), "{drawn:?} share a place");
        // And each still sits in the gap before the card it points at.
        for (r, card) in drawn.iter().zip([&chart.cards[1], &chart.cards[2]]) {
            let card = card_rect(card);
            assert!(r[0] + r[2] <= card[0] + 0.01, "{r:?} reaches {card:?}");
        }
    }

    // Two behaviors reaching the same asset enter its card along the one run,
    // so their labels take opposite sides of it rather than one being dropped.
    #[test]
    fn two_labels_reaching_one_card_take_both_sides_of_its_run() {
        let chart = wired(
            &[(0, 0), (0, 1), (1, 0)],
            &[(0, 2, "hides"), (1, 2, "shows")],
        );
        let (_, drawn) = drawn(&plain(&chart));
        assert_eq!(drawn.len(), 2, "neither wire loses its word");
        assert!(!overlaps(drawn[0], drawn[1]), "{drawn:?} share a place");
        let run = card_rect(&chart.cards[2])[1] + CARD_H * 0.5;
        assert!(drawn[0][1] + drawn[0][3] <= run, "{drawn:?}");
        assert!(drawn[1][1] >= run, "{drawn:?}");
    }

    // A wire between rows turns, and every run of it is axis-aligned.
    #[test]
    fn a_branch_wire_elbows_and_carries_its_pin_label() {
        let chart = branching();
        let (shapes, _) = drawn(&plain(&chart));
        let words: Vec<&str> = shapes.iter().map(|s| s.text.as_str()).collect();
        assert!(
            words.contains(&"then") && words.contains(&"else"),
            "{words:?}"
        );
        let runs: Vec<&UiShape> = shapes.iter().filter(|s| s.fill == WIRE_TINT).collect();
        assert!(
            runs.iter().any(|s| (s.rect[2] - WIRE_W).abs() < 0.01),
            "an elbow is more than one run"
        );
        for s in runs {
            assert!(
                (s.rect[3] - WIRE_W).abs() < 0.01 || (s.rect[2] - WIRE_W).abs() < 0.01,
                "{s:?}",
            );
        }
    }

    #[test]
    fn selecting_a_card_pans_only_as_far_as_it_must() {
        let chart = branching();
        let canvas = [300.0, 300.0];
        // A card already on screen leaves the pan where it is.
        assert_eq!(
            pan_to(&chart.cards[0], canvas, [0.0, 0.0], &chart),
            [0.0, 0.0]
        );
        // One off the right edge is brought just inside it.
        let far = chart.cards.iter().find(|c| c.title == "save").unwrap();
        let pan = pan_to(far, canvas, [0.0, 0.0], &chart);
        let at = card_rect(far);
        let rect = [at[0] - pan[0], at[1] - pan[1], at[2], at[3]];
        assert!(rect[0] >= 0.0, "{rect:?}");
        assert!(rect[0] + rect[2] <= canvas[0] + 0.01, "{rect:?}");
    }
}

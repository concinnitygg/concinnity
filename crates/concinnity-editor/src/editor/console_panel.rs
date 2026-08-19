// src/editor/console_panel.rs
//
// The Console panel's layout half: a terminal-style floating panel with a
// scrollable log window filling the body and a single command line pinned to
// the bottom (the panel's one TextInput, which also carries the /del name
// autocomplete as ghost text). Log lines colour by severity (`theme::LOG_*`).
// The log model and command dispatch live in `console.rs` and
// `hook/console_edit.rs`.

use super::console::{ConsoleLine, Severity};
use super::registry::{self, PanelKey};
use super::theme;
use super::widget::{self, place_rounded, point_in};
use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::Console);
pub(crate) const PANEL_BG: AssetId = AssetId(BASE);
pub(crate) const TITLE_LABEL: AssetId = AssetId(BASE + 2);
pub(crate) const CLOSE_BG: AssetId = AssetId(BASE + 3);
pub(crate) const CLOSE_LABEL: AssetId = AssetId(BASE + 4);
pub(crate) const LOG_TRACK: AssetId = AssetId(BASE + 5);
pub(crate) const LOG_THUMB: AssetId = AssetId(BASE + 6);
pub(crate) const INPUT: AssetId = AssetId(BASE + 7);

pub(crate) fn row_label(i: usize) -> AssetId {
    AssetId(BASE + 0x40 + i as u32)
}

// Geometry, in window pixels. Every rect derives from the panel origin `o`
// (the title bar's top-left), so dragging the title bar moves the whole panel.
// The default (and minimum) width; the user can widen the panel past this.
pub(crate) const CONSOLE_W: f32 = 560.0;
const PAD: f32 = 10.0;
const LINE_H: f32 = 22.0;
const INPUT_H: f32 = 28.0;
const SCROLLBAR_W: f32 = 5.0;
// Visible log rows at the default height; a longer log scrolls. Resizing the
// panel taller reveals more, up to the injected pool `LINE_POOL_MAX`.
pub(crate) const LINE_POOL: usize = 14;
pub(crate) const LINE_POOL_MAX: usize = 28;
// A log line's character budget at the default width; widening the panel grows it.
const MAX_LINE_CHARS: usize = 64;
// Approximate body-font advance, for growing the line budget with the width.
const CHAR_W: f32 = 8.5;

// The chrome bracketing the log window: the title bar above and the command line
// (plus its padding) below.
const CHROME_H: f32 = widget::TITLE_H + INPUT_H + PAD;

// The visible log-row count for panel height `h`, clamped to the injected pool.
// At the default height this is exactly `LINE_POOL`.
pub(crate) fn visible_lines(h: f32) -> usize {
    (((h - CHROME_H) / LINE_H).floor() as usize).clamp(LINE_POOL, LINE_POOL_MAX)
}

// The log-line character budget at width `w` (grows past the default with width).
fn max_line_chars(w: f32) -> usize {
    (MAX_LINE_CHARS as f32 + (w - CONSOLE_W) / CHAR_W).max(8.0) as usize
}

const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];

fn severity_color(severity: Severity) -> [f32; 3] {
    match severity {
        Severity::Info => theme::LOG_INFO,
        Severity::Warn => theme::LOG_WARN,
        Severity::Error => theme::LOG_ERROR,
        Severity::Command => theme::LOG_COMMAND,
    }
}

// The per-frame view the hook assembles. `lines` is the visible window
// (already cut by `first`); `total` sizes the scrollbar.
pub(crate) struct ConsoleView<'a> {
    pub lines: &'a [ConsoleLine],
    pub total: usize,
    pub first: usize,
    // Whether the command line asserts keyboard focus this frame.
    pub focus: bool,
    // The autocomplete's inline ghost suffix for the command line.
    pub ghost: &'a str,
    pub mouse: [f32; 2],
}

// A resolved Console-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConsoleAction {
    // Give the command line keyboard focus.
    FocusInput,
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Where the panel sits until the user drags it: bottom-left, terminal-style.
pub(crate) fn default_origin(vp: [f32; 2]) -> [f32; 2] {
    let s = size();
    [8.0, (vp[1] - s[1] - 8.0).max(super::hud::body_top())]
}

// The panel's default (and minimum) footprint.
pub(crate) fn size() -> [f32; 2] {
    [
        CONSOLE_W,
        widget::TITLE_H + LINE_POOL as f32 * LINE_H + INPUT_H + 2.0 * PAD,
    ]
}

// The tallest the panel resizes to: the injected log pool shown in full. Width
// is unbounded (capped only by the screen).
pub(crate) fn max_size() -> [f32; 2] {
    [
        f32::INFINITY,
        size()[1] + (LINE_POOL_MAX - LINE_POOL) as f32 * LINE_H,
    ]
}

fn log_top(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H + PAD * 0.5
}

// Visible log row `slot` (0-based within the window) at the effective width `w`.
pub(crate) fn line_rect(o: [f32; 2], w: f32, slot: usize) -> [f32; 4] {
    [
        o[0],
        log_top(o) + slot as f32 * LINE_H,
        w - SCROLLBAR_W - 2.0,
        LINE_H,
    ]
}

// The command line, pinned to the panel's bottom edge.
pub(crate) fn input_rect(o: [f32; 2], s: [f32; 2]) -> [f32; 4] {
    [
        o[0] + PAD,
        o[1] + s[1] - INPUT_H - PAD * 0.5,
        s[0] - 2.0 * PAD,
        INPUT_H,
    ]
}

// Whether the cursor is over the scrollable log window (for wheel routing).
pub(crate) fn cursor_over_log(mx: f32, my: f32, o: [f32; 2], s: [f32; 2]) -> bool {
    let top = log_top(o);
    let shown = visible_lines(s[1]);
    mx >= o[0] && mx < o[0] + s[0] && my >= top && my < top + shown as f32 * LINE_H
}

// Resolve a click at `(mx, my)` against the panel at origin `o`, size `s`.
// `None` means the click missed the panel. Title-bar presses never reach this:
// the shared routing intercepts them first.
pub(crate) fn hit_test(mx: f32, my: f32, o: [f32; 2], s: [f32; 2]) -> Option<ConsoleAction> {
    if point_in(mx, my, input_rect(o, s)) {
        return Some(ConsoleAction::FocusInput);
    }
    point_in(mx, my, widget::outer_rect(o, s)).then_some(ConsoleAction::Consume)
}

// Position + show the panel (`Some(view)`) at effective size `s`, or blank every
// element (`None`).
pub(crate) fn apply(world: &mut World, view: Option<&ConsoleView>, o: [f32; 2], s: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };
    let w = s[0];
    let shown = visible_lines(s[1]);
    widget::place_panel(world, PANEL_BG, widget::outer_rect(o, s));
    let title = widget::title_rect(o, w);
    widget::place_heading(world, TITLE_LABEL, title, "Console");
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);

    for slot in 0..LINE_POOL_MAX {
        let Some(line) = view.lines.get(slot).filter(|_| slot < shown) else {
            widget::set_label_visible(world, row_label(slot), false);
            continue;
        };
        let r = line_rect(o, w, slot);
        if let Some(l) = widget::label_mut(world, row_label(slot)) {
            l.x = r[0] + PAD;
            l.y = r[1] + LINE_H * 0.5 - theme::TEXT_HALF;
            l.align = TextAlign::Left;
            l.color = severity_color(line.severity);
            l.visible = true;
            l.content = widget::clip_text(&line.text, max_line_chars(w));
        }
    }

    widget::show_field(world, INPUT, input_rect(o, s), view.focus);
    if let Some(t) = widget::input_mut(world, INPUT) {
        t.ghost = view.ghost.to_string();
    }

    layout_scrollbar(world, view, o, w, shown);
}

// The log window's scrollbar: shown only when the log overflows the window.
fn layout_scrollbar(world: &mut World, view: &ConsoleView, o: [f32; 2], w: f32, shown: usize) {
    let total = view.total;
    if total <= shown {
        widget::set_sprite_visible(world, LOG_TRACK, false);
        widget::set_sprite_visible(world, LOG_THUMB, false);
        return;
    }
    let x = o[0] + w - SCROLLBAR_W;
    let top = log_top(o);
    let h = shown as f32 * LINE_H;
    place_rounded(
        world,
        LOG_TRACK,
        [x, top, SCROLLBAR_W, h],
        TRACK_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
    let thumb_h = (h * shown as f32 / total as f32).max(18.0);
    let max_first = (total - shown) as f32;
    let off = (h - thumb_h) * (view.first.min(total - shown) as f32 / max_first);
    place_rounded(
        world,
        LOG_THUMB,
        [x, top + off, SCROLLBAR_W, thumb_h],
        THUMB_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
}

// Hide every panel element, blurring the command line so a hidden field cannot
// keep keyboard focus.
pub(crate) fn hide_all(world: &mut World) {
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &all_field_ids());
    if let Some(t) = widget::input_mut(world, INPUT) {
        t.ghost.clear();
    }
}

pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    vec![PANEL_BG, CLOSE_BG, LOG_TRACK, LOG_THUMB]
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![TITLE_LABEL, CLOSE_LABEL];
    ids.extend((0..LINE_POOL_MAX).map(row_label));
    ids
}

pub(crate) fn all_field_ids() -> Vec<AssetId> {
    vec![INPUT]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextInput, TextLabel};

    fn injected_world() -> World {
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
        for id in all_field_ids() {
            world.add_component(TextInput {
                asset_id: id,
                ..Default::default()
            });
        }
        world
    }

    fn lines(n: usize) -> Vec<ConsoleLine> {
        (0..n)
            .map(|i| ConsoleLine {
                severity: Severity::Info,
                text: format!("line {i}"),
            })
            .collect()
    }

    fn view<'a>(lines: &'a [ConsoleLine], total: usize, first: usize) -> ConsoleView<'a> {
        ConsoleView {
            lines,
            total,
            first,
            focus: true,
            ghost: "",
            mouse: [0.0, 0.0],
        }
    }

    #[test]
    fn hit_test_resolves_the_input_or_swallows() {
        let o = [40.0, 40.0];
        let s = size();
        let i = input_rect(o, s);
        assert_eq!(
            hit_test(i[0] + 5.0, i[1] + 5.0, o, s),
            Some(ConsoleAction::FocusInput)
        );
        let r0 = line_rect(o, s[0], 0);
        assert_eq!(
            hit_test(r0[0] + 5.0, r0[1] + 5.0, o, s),
            Some(ConsoleAction::Consume)
        );
        assert_eq!(hit_test(5000.0, 5000.0, o, s), None);
    }

    #[test]
    fn input_pins_to_the_bottom_and_log_fills_the_body() {
        let o = [0.0, 0.0];
        let s = size();
        let p = widget::outer_rect(o, s);
        let i = input_rect(o, s);
        assert!(i[1] + i[3] < p[1] + p[3], "input inside the panel");
        let last = line_rect(o, s[0], LINE_POOL - 1);
        assert!(
            last[1] + last[3] <= i[1],
            "the last log row sits above the input"
        );
        assert!(cursor_over_log(10.0, last[1] + 5.0, o, s));
        assert!(!cursor_over_log(10.0, i[1] + 5.0, o, s));
    }

    #[test]
    fn taller_size_reveals_more_lines_up_to_the_pool() {
        // The default height shows exactly LINE_POOL lines; taller reveals more,
        // capped at the injected pool.
        assert_eq!(visible_lines(size()[1]), LINE_POOL);
        assert_eq!(visible_lines(size()[1] + 4.0 * LINE_H), LINE_POOL + 4);
        assert_eq!(visible_lines(max_size()[1]), LINE_POOL_MAX);
        assert_eq!(visible_lines(10_000.0), LINE_POOL_MAX, "capped at the pool");
    }

    #[test]
    fn wider_width_grows_the_line_budget() {
        assert_eq!(max_line_chars(CONSOLE_W), MAX_LINE_CHARS);
        assert!(max_line_chars(CONSOLE_W + 200.0) > MAX_LINE_CHARS);
    }

    // A taller panel draws log rows past the default pool that a default one
    // would never show, and the command line stays pinned inside the panel.
    #[test]
    fn taller_panel_draws_lines_past_the_default_pool() {
        let mut world = injected_world();
        let l = lines(LINE_POOL_MAX);
        let tall = [CONSOLE_W, max_size()[1]];
        apply(
            &mut world,
            Some(&view(&l, LINE_POOL_MAX, 0)),
            [20.0, 20.0],
            tall,
        );
        assert!(
            world
                .query::<TextLabel>()
                .find(|l| l.asset_id == row_label(LINE_POOL))
                .unwrap()
                .visible
        );
        let p = widget::outer_rect([20.0, 20.0], tall);
        let i = input_rect([20.0, 20.0], tall);
        assert!(
            i[1] + i[3] < p[1] + p[3],
            "input stays inside the taller panel"
        );
    }

    #[test]
    fn apply_colors_lines_by_severity_and_seeds_the_ghost() {
        let mut world = injected_world();
        let l = vec![
            ConsoleLine {
                severity: Severity::Error,
                text: "boom".to_string(),
            },
            ConsoleLine {
                severity: Severity::Command,
                text: "> /help".to_string(),
            },
        ];
        let v = ConsoleView {
            ghost: "cube",
            ..view(&l, 2, 0)
        };
        apply(&mut world, Some(&v), [20.0, 20.0], size());
        let first = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == row_label(0))
            .unwrap();
        assert!(first.visible && first.content == "boom");
        assert_eq!(first.color, theme::LOG_ERROR);
        let second = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == row_label(1))
            .unwrap();
        assert_eq!(second.color, theme::LOG_COMMAND);
        assert!(
            !world
                .query::<TextLabel>()
                .find(|l| l.asset_id == row_label(2))
                .unwrap()
                .visible
        );
        let input = world
            .query::<TextInput>()
            .find(|t| t.asset_id == INPUT)
            .unwrap();
        assert!(input.visible && input.focused);
        assert_eq!(input.ghost, "cube");
        // A short log shows no scrollbar.
        assert!(
            !world
                .query::<Sprite>()
                .find(|s| s.asset_id == LOG_THUMB)
                .unwrap()
                .visible
        );
    }

    #[test]
    fn long_log_shows_the_scrollbar() {
        let mut world = injected_world();
        let l = lines(LINE_POOL);
        apply(&mut world, Some(&view(&l, 40, 10)), [20.0, 20.0], size());
        assert!(
            world
                .query::<Sprite>()
                .find(|s| s.asset_id == LOG_THUMB)
                .unwrap()
                .visible
        );
    }

    #[test]
    fn hide_all_blanks_every_element_and_clears_the_ghost() {
        let mut world = injected_world();
        let l = lines(3);
        let v = ConsoleView {
            ghost: "cube",
            ..view(&l, 3, 0)
        };
        apply(&mut world, Some(&v), [20.0, 20.0], size());
        apply(&mut world, None, [0.0, 0.0], size());
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
        let input = world.query::<TextInput>().next().unwrap();
        assert!(!input.visible && !input.focused && input.ghost.is_empty());
    }
}

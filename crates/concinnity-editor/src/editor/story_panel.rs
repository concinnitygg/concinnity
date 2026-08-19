// src/editor/story_panel.rs
//
// The Story panel's layout half: a floating line editor over the Markdown
// source of the world's `StoryImport` (see `story.rs` for the text model and
// `hook/story_edit.rs` for the actions). The header carries the source path
// and the Apply button, a reserved status line shows parse / IO errors, and
// the body is a fixed window of line rows: the current line is the panel's
// single real `TextInput` (the engine's existing primitive -- no multiline
// runtime asset), every other visible line a clipped `TextLabel`. A world
// with no `StoryImport` shows a single "+ Create story" row instead.

use super::registry::{self, PanelKey};
use super::theme;
use super::widget::{self, place_rounded, point_in};
use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::Story);
pub(crate) const PANEL_BG: AssetId = AssetId(BASE);
pub(crate) const TITLE_LABEL: AssetId = AssetId(BASE + 2);
pub(crate) const CLOSE_BG: AssetId = AssetId(BASE + 3);
pub(crate) const CLOSE_LABEL: AssetId = AssetId(BASE + 4);
pub(crate) const APPLY_BG: AssetId = AssetId(BASE + 5);
pub(crate) const APPLY_LABEL: AssetId = AssetId(BASE + 6);
pub(crate) const PATH_LABEL: AssetId = AssetId(BASE + 7);
pub(crate) const STATUS_LABEL: AssetId = AssetId(BASE + 8);
pub(crate) const LINE_TRACK: AssetId = AssetId(BASE + 9);
pub(crate) const LINE_THUMB: AssetId = AssetId(BASE + 10);
// The single edit control; it renders at the current line's row.
pub(crate) const LINE_INPUT: AssetId = AssetId(BASE + 11);

pub(crate) fn row_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x20 + i as u32)
}
pub(crate) fn row_label(i: usize) -> AssetId {
    AssetId(BASE + 0x40 + i as u32)
}

// Geometry, in window pixels. Every rect derives from the panel origin `o`
// (the title bar's top-left), so dragging the title bar moves the whole panel.
// The default (and minimum) width; the user can widen the panel past this.
pub(crate) const STORY_W: f32 = 560.0;
const PAD: f32 = 10.0;
const HEADER_H: f32 = 32.0;
// Two lines: a parse or IO message carries a path and rarely fits one.
const STATUS_H: f32 = 2.0 * widget::LINE_H + 6.0;
const BTN_W: f32 = 70.0;
const LINE_H: f32 = 24.0;
const SCROLLBAR_W: f32 = 5.0;
// Visible line rows at the default height; a longer story scrolls. Resizing the
// panel taller reveals more, up to the injected pool `LINE_POOL_MAX`.
pub(crate) const LINE_POOL: usize = 16;
pub(crate) const LINE_POOL_MAX: usize = 30;
// A label line's character budget before it clips at the default width (the edit
// line's TextInput scrolls instead of clipping); widening grows the budget.
const MAX_LINE_CHARS: usize = 62;
// Approximate body-font advance, for growing the caption budget with the width.
const CHAR_W: f32 = 8.5;

// The chrome height above the line window (everything but the line band + pad).
const CHROME_H: f32 = widget::TITLE_H + HEADER_H + STATUS_H;

// The visible line count for panel height `h`, clamped to the injected pool. At
// the default height this is exactly `LINE_POOL`.
pub(crate) fn visible_rows(h: f32) -> usize {
    (((h - CHROME_H - PAD) / LINE_H).floor() as usize).clamp(LINE_POOL, LINE_POOL_MAX)
}

// The line character budget at width `w` (grows past the default with width).
fn max_line_chars(w: f32) -> usize {
    (MAX_LINE_CHARS as f32 + (w - STORY_W) / CHAR_W).max(8.0) as usize
}

// The tallest the panel resizes to: the injected line pool shown in full.
pub(crate) fn max_size() -> [f32; 2] {
    [
        f32::INFINITY,
        size()[1] + (LINE_POOL_MAX - LINE_POOL) as f32 * LINE_H,
    ]
}

const ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const ROW_TINT_CURRENT: [f32; 4] = theme::SELECTED_TINT;
const ROW_TINT_HOVER: [f32; 4] = theme::HOVER_TINT;
const BTN_TINT: [f32; 4] = [0.22, 0.40, 0.56, 1.0];
const BTN_TINT_HOVER: [f32; 4] = [0.28, 0.48, 0.66, 1.0];
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
const LABEL: [f32; 3] = [0.90, 0.90, 0.92];
const PATH_COLOR: [f32; 3] = [0.62, 0.66, 0.76];
const ADD_LABEL: [f32; 3] = [0.55, 0.80, 0.60];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];

// The per-frame view the hook assembles.
pub(crate) struct StoryView<'a> {
    // Every story line; the panel windows them by `scroll`.
    pub lines: &'a [String],
    pub scroll: usize,
    // The line being edited (rendered as the TextInput when visible).
    pub current: usize,
    // Whether the edit line asserts keyboard focus this frame.
    pub focus: bool,
    // The story source path shown in the header; empty in create mode.
    pub path: &'a str,
    // Parse / IO error from the last Apply (or load).
    pub status: Option<&'a str>,
    // No StoryImport in the world: show the create row instead of lines.
    pub create: bool,
    // Line edits not yet Applied to the source file: the heading marks them.
    pub dirty: bool,
    pub mouse: [f32; 2],
}

// A resolved Story-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StoryAction {
    // Move the edit line to absolute line index `i`.
    Line(usize),
    // Write the starter story and add its StoryImport entry.
    Create,
    // Validate, write the source file, and refresh the live preview.
    Apply,
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Where the panel sits until the user drags it: centred below the top bar
// (clear of the left default stack and the right Assets anchor).
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [(vw - STORY_W) * 0.5, super::hud::body_top() + 8.0]
}

// The panel's fixed footprint (the line window does not grow with the story).
pub(crate) fn size() -> [f32; 2] {
    [
        STORY_W,
        widget::TITLE_H + HEADER_H + STATUS_H + LINE_POOL as f32 * LINE_H + PAD,
    ]
}

// The Apply button, pinned to the header row's right end.
pub(crate) fn apply_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    [
        o[0] + w - PAD - BTN_W,
        o[1] + widget::TITLE_H + 4.0,
        BTN_W,
        HEADER_H - 8.0,
    ]
}

fn body_top(o: [f32; 2]) -> f32 {
    o[1] + CHROME_H
}

// Visible line row `slot` (0-based within the window) at the effective width `w`.
pub(crate) fn line_rect(o: [f32; 2], w: f32, slot: usize) -> [f32; 4] {
    [
        o[0],
        body_top(o) + slot as f32 * LINE_H,
        w - SCROLLBAR_W - 2.0,
        LINE_H,
    ]
}

// Whether the cursor is over the scrollable line window (for wheel routing).
pub(crate) fn cursor_over_lines(mx: f32, my: f32, o: [f32; 2], s: [f32; 2]) -> bool {
    let p = widget::outer_rect(o, s);
    mx >= p[0] && mx < p[0] + p[2] && my >= body_top(o) && my < p[1] + p[3]
}

// Resolve a click at `(mx, my)` against the panel at origin `o`, size `s`. `None`
// means the click missed the panel. Title-bar presses never reach this: the
// shared routing intercepts them first.
pub(crate) fn hit_test(
    view: &StoryView,
    mx: f32,
    my: f32,
    o: [f32; 2],
    s: [f32; 2],
) -> Option<StoryAction> {
    let w = s[0];
    if point_in(mx, my, apply_rect(o, w)) && !view.create {
        return Some(StoryAction::Apply);
    }
    for slot in 0..visible_rows(s[1]) {
        if !point_in(mx, my, line_rect(o, w, slot)) {
            continue;
        }
        if view.create {
            return Some(if slot == 0 {
                StoryAction::Create
            } else {
                StoryAction::Consume
            });
        }
        let i = view.scroll + slot;
        return Some(if i < view.lines.len() {
            StoryAction::Line(i)
        } else {
            StoryAction::Consume
        });
    }
    point_in(mx, my, widget::outer_rect(o, s)).then_some(StoryAction::Consume)
}

// Position + show the panel (`Some(view)`) at effective size `s`, or blank every
// element (`None`).
pub(crate) fn apply(world: &mut World, view: Option<&StoryView>, o: [f32; 2], s: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };
    let w = s[0];
    let rows_shown = visible_rows(s[1]);
    widget::place_panel(world, PANEL_BG, widget::outer_rect(o, s));
    let title = widget::title_rect(o, w);
    let heading = if view.dirty { "Story *" } else { "Story" };
    widget::place_heading(world, TITLE_LABEL, title, heading);
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);

    // Header: the source path on the left, Apply on the right (hidden in
    // create mode, where there is nothing to write yet).
    if let Some(l) = widget::label_mut(world, PATH_LABEL) {
        l.x = o[0] + PAD;
        l.y = o[1] + widget::TITLE_H + HEADER_H * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Left;
        l.color = PATH_COLOR;
        l.visible = !view.create;
        l.content = widget::clip_text(view.path, 52);
    }
    let apply_btn = apply_rect(o, w);
    let hover = point_in(view.mouse[0], view.mouse[1], apply_btn);
    let tint = if hover { BTN_TINT_HOVER } else { BTN_TINT };
    place_rounded(
        world,
        APPLY_BG,
        apply_btn,
        tint,
        theme::CONTROL_RADIUS,
        !view.create,
    );
    if let Some(l) = widget::label_mut(world, APPLY_LABEL) {
        l.x = apply_btn[0] + apply_btn[2] * 0.5;
        l.y = apply_btn[1] + apply_btn[3] * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Center;
        l.color = [1.0, 1.0, 1.0];
        l.visible = !view.create;
        l.content = "Apply".to_string();
    }
    widget::place_message(
        world,
        STATUS_LABEL,
        [
            o[0] + PAD,
            o[1] + widget::TITLE_H + HEADER_H + 1.0,
            (w - 2.0 * PAD).max(0.0),
            STATUS_H - 4.0,
        ],
        view.status.unwrap_or(""),
        ERROR_LABEL,
        view.status.is_some(),
    );

    widget::hide_field(world, LINE_INPUT);
    for slot in 0..LINE_POOL_MAX {
        if slot >= rows_shown {
            widget::set_sprite_visible(world, row_bg(slot), false);
            widget::set_label_visible(world, row_label(slot), false);
            continue;
        }
        let r = line_rect(o, w, slot);
        if view.create {
            // One actionable create row; the rest of the window is blank.
            let on = slot == 0;
            let hovered = on && point_in(view.mouse[0], view.mouse[1], r);
            let tint = if hovered { ROW_TINT_HOVER } else { ROW_TINT };
            place_rounded(
                world,
                row_bg(slot),
                theme::highlight_rect(r),
                tint,
                theme::CONTROL_RADIUS,
                on,
            );
            if let Some(l) = widget::label_mut(world, row_label(slot)) {
                l.x = r[0] + PAD;
                l.y = r[1] + LINE_H * 0.5 - theme::TEXT_HALF;
                l.align = TextAlign::Left;
                l.color = ADD_LABEL;
                l.visible = on;
                l.content = "+ Create story".to_string();
            }
            continue;
        }
        let i = view.scroll + slot;
        if i >= view.lines.len() {
            widget::set_sprite_visible(world, row_bg(slot), false);
            widget::set_label_visible(world, row_label(slot), false);
            continue;
        }
        let current = i == view.current;
        let tint = if current { ROW_TINT_CURRENT } else { ROW_TINT };
        place_rounded(
            world,
            row_bg(slot),
            theme::highlight_rect(r),
            tint,
            theme::CONTROL_RADIUS,
            true,
        );
        if current {
            // The edit line: the panel's one TextInput, focused per the view.
            widget::set_label_visible(world, row_label(slot), false);
            widget::show_field(
                world,
                LINE_INPUT,
                [r[0] + 2.0, r[1] + 1.0, r[2] - 4.0, r[3] - 2.0],
                view.focus,
            );
        } else if let Some(l) = widget::label_mut(world, row_label(slot)) {
            l.x = r[0] + PAD;
            l.y = r[1] + LINE_H * 0.5 - theme::TEXT_HALF;
            l.align = TextAlign::Left;
            l.color = LABEL;
            l.visible = true;
            l.content = widget::clip_text(&view.lines[i], max_line_chars(w));
        }
    }

    layout_scrollbar(world, view, o, w, rows_shown);
}

// The line window's scrollbar: shown only when the story overflows the pool.
fn layout_scrollbar(world: &mut World, view: &StoryView, o: [f32; 2], w: f32, rows_shown: usize) {
    let total = view.lines.len();
    if view.create || total <= rows_shown {
        widget::set_sprite_visible(world, LINE_TRACK, false);
        widget::set_sprite_visible(world, LINE_THUMB, false);
        return;
    }
    let x = o[0] + w - SCROLLBAR_W;
    let top = body_top(o);
    let h = rows_shown as f32 * LINE_H;
    place_rounded(
        world,
        LINE_TRACK,
        [x, top, SCROLLBAR_W, h],
        TRACK_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
    let frac = rows_shown as f32 / total as f32;
    let thumb_h = (h * frac).max(18.0);
    let max_scroll = (total - rows_shown) as f32;
    let off = (h - thumb_h) * (view.scroll as f32 / max_scroll);
    place_rounded(
        world,
        LINE_THUMB,
        [x, top + off, SCROLLBAR_W, thumb_h],
        THUMB_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
}

// Hide every panel element, blurring the edit field so a hidden field cannot
// keep keyboard focus.
pub(crate) fn hide_all(world: &mut World) {
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &all_field_ids());
}

// Every panel sprite id, in draw (insertion) order: chrome, then the line
// rows, then the scrollbar floating above them.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG, APPLY_BG];
    ids.extend((0..LINE_POOL_MAX).map(row_bg));
    ids.extend([LINE_TRACK, LINE_THUMB]);
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![
        TITLE_LABEL,
        CLOSE_LABEL,
        APPLY_LABEL,
        PATH_LABEL,
        STATUS_LABEL,
    ];
    ids.extend((0..LINE_POOL_MAX).map(row_label));
    ids
}

pub(crate) fn all_field_ids() -> Vec<AssetId> {
    vec![LINE_INPUT]
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

    fn lines(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("line {i}")).collect()
    }

    fn view<'a>(lines: &'a [String], scroll: usize, current: usize) -> StoryView<'a> {
        StoryView {
            lines,
            scroll,
            current,
            focus: true,
            path: "story.md",
            status: None,
            create: false,
            dirty: false,
            mouse: [0.0, 0.0],
        }
    }

    #[test]
    fn hit_test_resolves_lines_apply_and_create() {
        let l = lines(4);
        let v = view(&l, 0, 1);
        let o = [40.0, 40.0];
        let s = size();
        let a = apply_rect(o, STORY_W);
        assert_eq!(
            hit_test(&v, a[0] + 5.0, a[1] + 5.0, o, s),
            Some(StoryAction::Apply)
        );
        let r2 = line_rect(o, STORY_W, 2);
        assert_eq!(
            hit_test(&v, r2[0] + 5.0, r2[1] + 5.0, o, s),
            Some(StoryAction::Line(2))
        );
        // Past the last line the click is swallowed, not a line action.
        let r5 = line_rect(o, STORY_W, 5);
        assert_eq!(
            hit_test(&v, r5[0] + 5.0, r5[1] + 5.0, o, s),
            Some(StoryAction::Consume)
        );
        assert_eq!(hit_test(&v, 5000.0, 5000.0, o, s), None);
        // Create mode: row 0 creates, Apply is absent.
        let create = StoryView {
            create: true,
            ..view(&l, 0, 0)
        };
        let r0 = line_rect(o, STORY_W, 0);
        assert_eq!(
            hit_test(&create, r0[0] + 5.0, r0[1] + 5.0, o, s),
            Some(StoryAction::Create)
        );
        assert_eq!(
            hit_test(&create, a[0] + 5.0, a[1] + 5.0, o, s),
            Some(StoryAction::Consume),
            "no Apply in create mode"
        );
    }

    #[test]
    fn scrolled_window_maps_slots_to_absolute_lines() {
        let l = lines(40);
        let v = view(&l, 10, 12);
        let o = [0.0, 0.0];
        let r0 = line_rect(o, STORY_W, 0);
        assert_eq!(
            hit_test(&v, r0[0] + 5.0, r0[1] + 5.0, o, size()),
            Some(StoryAction::Line(10))
        );
    }

    #[test]
    fn apply_draws_the_edit_line_as_the_input_and_others_as_labels() {
        let mut world = injected_world();
        let l = lines(4);
        apply(&mut world, Some(&view(&l, 0, 1)), [20.0, 20.0], size());
        let input = world
            .query::<TextInput>()
            .find(|t| t.asset_id == LINE_INPUT)
            .unwrap();
        assert!(
            input.visible && input.focused,
            "edit line is the live input"
        );
        let slot1 = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == row_label(1))
            .unwrap();
        assert!(!slot1.visible, "the edit slot's label yields to the input");
        let slot2 = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == row_label(2))
            .unwrap();
        assert!(slot2.visible);
        assert_eq!(slot2.content, "line 2");
        // Rows past the story are blank; short story shows no scrollbar.
        let slot5 = world
            .query::<Sprite>()
            .find(|s| s.asset_id == row_bg(5))
            .unwrap();
        assert!(!slot5.visible);
        assert!(
            !world
                .query::<Sprite>()
                .find(|s| s.asset_id == LINE_THUMB)
                .unwrap()
                .visible
        );
    }

    #[test]
    fn edit_line_outside_the_window_hides_the_input() {
        let mut world = injected_world();
        let l = lines(40);
        apply(&mut world, Some(&view(&l, 20, 3)), [20.0, 20.0], size());
        let input = world
            .query::<TextInput>()
            .find(|t| t.asset_id == LINE_INPUT)
            .unwrap();
        assert!(!input.visible, "a scrolled-away edit line has no control");
        assert!(
            world
                .query::<Sprite>()
                .find(|s| s.asset_id == LINE_THUMB)
                .unwrap()
                .visible,
            "a long story shows the scrollbar"
        );
    }

    #[test]
    fn create_mode_shows_the_create_row_only() {
        let mut world = injected_world();
        let l = lines(1);
        let v = StoryView {
            create: true,
            ..view(&l, 0, 0)
        };
        apply(&mut world, Some(&v), [20.0, 20.0], size());
        let r0 = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == row_label(0))
            .unwrap();
        assert!(r0.visible && r0.content == "+ Create story");
        assert!(
            !world
                .query::<Sprite>()
                .find(|s| s.asset_id == APPLY_BG)
                .unwrap()
                .visible,
            "Apply hidden in create mode"
        );
        assert!(
            !world
                .query::<TextInput>()
                .find(|t| t.asset_id == LINE_INPUT)
                .unwrap()
                .visible
        );
    }

    #[test]
    fn taller_size_reveals_more_lines_up_to_the_pool() {
        assert_eq!(visible_rows(size()[1]), LINE_POOL);
        assert_eq!(visible_rows(size()[1] + 4.0 * LINE_H), LINE_POOL + 4);
        assert_eq!(visible_rows(max_size()[1]), LINE_POOL_MAX);
        assert_eq!(visible_rows(10_000.0), LINE_POOL_MAX, "capped at the pool");
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        let l = lines(4);
        apply(&mut world, Some(&view(&l, 0, 1)), [20.0, 20.0], size());
        apply(&mut world, None, [0.0, 0.0], size());
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
        assert!(world.query::<TextInput>().all(|t| !t.visible));
    }
}

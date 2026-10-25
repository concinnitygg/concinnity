//! The Planet panel's tool strip, between its header and its rows: a row of
//! square icon buttons, one per tool, and while a tool is picked its brush
//! settings under them: a slider each for the radius, strength and falloff,
//! and for a paint tool the layer or cover it lays.

use std::fmt::Write as _;

use concinnity_engine::components::TextAlign;
use concinnity_engine::ecs::World;

use super::planet::List;
use super::planet_panel::{
    PICK_BG, PICK_CAPTION, PICK_LABEL, TOOL_CAPTION, TOOL_HINT, setting_label, setting_value,
    slider_fill, slider_handle, slider_track, tool_bg, tool_part,
};
use crate::editor::icons::{self, MAX_PARTS};
use crate::editor::planet_tools::{Setting, Tool, Tools};
use crate::editor::theme;
use crate::editor::widget::{self, place_rounded, point_in};
use crate::editor::widget_slider::{self, SliderIds};

/// The tools the strip shows.
pub(crate) const TOOLS: usize = Tool::ALL.len();

/// The brush settings with a slider.
pub(crate) const SETTINGS: usize = Setting::ALL.len();

// Geometry, in window pixels.
const PAD: f32 = 10.0;
const BUTTON: f32 = 36.0;
const BUTTON_GAP: f32 = 6.0;
const ICON_INSET: f32 = 7.0;
const PALETTE_H: f32 = BUTTON + 8.0;
const ROW_H: f32 = 26.0;
const LABEL_COL: f32 = 110.0;
const BRUSH_GAP: f32 = 6.0;

const ACTIVE_ICON: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const ICON: [f32; 4] = [0.82, 0.84, 0.90, 1.0];
const HINT: &str = "[ ] size   Esc";

/// What the strip shows this frame.
pub(crate) struct StripView<'a> {
    pub tools: &'a Tools,
    /// The paint tool's pick, named.
    pub pick: &'a str,
    /// The setting whose slider is being dragged.
    pub sliding: Option<Setting>,
    pub mouse: [f32; 2],
}

/// A resolved press on the strip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum StripAction {
    /// Pick a tool, or put it down when it is picked.
    Toggle(Tool),
    /// Grab a setting's slider, its position moved to `t` under the press.
    Slide(Setting, f32),
    /// Move a paint tool's pick to the next entry of its list.
    Cycle(List),
    /// Elsewhere on the strip.
    Consume,
}

// The brush rows a picked tool shows: its caption, the sliders and a paint
// tool's pick.
fn brush_rows(tools: &Tools) -> usize {
    tools
        .active
        .map_or(0, |t| 1 + SETTINGS + usize::from(t.list().is_some()))
}

/// The strip's height for `tools`.
pub(crate) fn height(tools: &Tools) -> f32 {
    let rows = brush_rows(tools);
    let brush = if rows == 0 {
        0.0
    } else {
        rows as f32 * ROW_H + BRUSH_GAP
    };
    PALETTE_H + brush
}

/// Tool `i`'s button in the strip starting at `top`.
pub(crate) fn button_rect(o: [f32; 2], top: f32, i: usize) -> [f32; 4] {
    let x = o[0] + PAD + i as f32 * (BUTTON + BUTTON_GAP);
    [x, top + (PALETTE_H - BUTTON) * 0.5, BUTTON, BUTTON]
}

// Brush row `k` across the panel's width `w`.
fn row_rect(o: [f32; 2], top: f32, w: f32, k: usize) -> [f32; 4] {
    [o[0], top + PALETTE_H + k as f32 * ROW_H, w, ROW_H]
}

// The control on a brush row, right of its caption.
fn control_rect(row: [f32; 4]) -> [f32; 4] {
    let x = row[0] + LABEL_COL;
    [x, row[1] + 3.0, row[0] + row[2] - PAD - x, row[3] - 6.0]
}

/// Setting `s`'s slider.
pub(crate) fn slider_rect(o: [f32; 2], top: f32, w: f32, s: Setting) -> [f32; 4] {
    control_rect(row_rect(o, top, w, 1 + s as usize))
}

fn pick_rect(o: [f32; 2], top: f32, w: f32) -> [f32; 4] {
    control_rect(row_rect(o, top, w, 1 + SETTINGS))
}

fn slider_ids(s: Setting) -> SliderIds {
    let i = s as usize;
    SliderIds {
        track: slider_track(i),
        fill: slider_fill(i),
        handle: slider_handle(i),
        value: setting_value(i),
    }
}

fn part_ids(i: usize) -> [concinnity_engine::ecs::asset_id::AssetId; MAX_PARTS] {
    core::array::from_fn(|k| tool_part(i * MAX_PARTS + k))
}

/// The slider position of the cursor at `mx` over setting `s`'s slider.
pub(crate) fn slide_to(o: [f32; 2], top: f32, w: f32, s: Setting, mx: f32) -> f32 {
    widget_slider::value_at(slider_rect(o, top, w, s), mx, (0.0, 1.0))
}

/// Resolve a press at `(mx, my)` on the strip starting at `top`, `None` when
/// it missed the strip.
pub(crate) fn hit_test(
    view: &StripView<'_>,
    mx: f32,
    my: f32,
    o: [f32; 2],
    top: f32,
    w: f32,
) -> Option<StripAction> {
    if !point_in(mx, my, [o[0], top, w, height(view.tools)]) {
        return None;
    }
    if let Some(tool) = Tool::ALL
        .iter()
        .enumerate()
        .find(|(i, _)| point_in(mx, my, button_rect(o, top, *i)))
        .map(|(_, t)| *t)
    {
        return Some(StripAction::Toggle(tool));
    }
    let Some(active) = view.tools.active else {
        return Some(StripAction::Consume);
    };
    for s in Setting::ALL {
        if widget_slider::hit(slider_rect(o, top, w, s), mx, my) {
            return Some(StripAction::Slide(s, slide_to(o, top, w, s, mx)));
        }
    }
    if let Some(list) = active.list()
        && point_in(mx, my, pick_rect(o, top, w))
    {
        return Some(StripAction::Cycle(list));
    }
    Some(StripAction::Consume)
}

/// Position and show the strip starting at `top`.
pub(crate) fn place(world: &mut World, view: &StripView<'_>, o: [f32; 2], top: f32, w: f32) {
    let [mx, my] = view.mouse;
    for (i, tool) in Tool::ALL.into_iter().enumerate() {
        let b = button_rect(o, top, i);
        let active = view.tools.active == Some(tool);
        let tint = if active {
            theme::ACCENT_TINT
        } else if point_in(mx, my, b) {
            theme::HOVER_TINT
        } else {
            theme::BUTTON_TINT
        };
        place_rounded(world, tool_bg(i), b, tint, theme::CONTROL_RADIUS, true);
        let cell = [
            b[0] + ICON_INSET,
            b[1] + ICON_INSET,
            BUTTON - 2.0 * ICON_INSET,
        ];
        let color = if active { ACTIVE_ICON } else { ICON };
        icons::place(world, &part_ids(i), cell, tool.glyph(), color);
    }
    let Some(tool) = view.tools.active else {
        hide_brush(world);
        return;
    };
    place_caption(world, tool, row_rect(o, top, w, 0));
    for s in Setting::ALL {
        let row = row_rect(o, top, w, 1 + s as usize);
        place_left(world, setting_label(s as usize), row, s.caption());
        let rect = control_rect(row);
        let brush = &view.tools.brush;
        let hot = view.sliding == Some(s) || widget_slider::hit(rect, mx, my);
        widget_slider::place(
            world,
            slider_ids(s),
            rect,
            brush.position(s),
            (0.0, 1.0),
            hot,
        );
        if let Some(l) = widget::label_mut(world, setting_value(s as usize)) {
            let v = brush.value(s);
            match s {
                Setting::Radius => set_text(&mut l.content, format_args!("{v:.1} m")),
                Setting::Strength | Setting::Falloff => {
                    set_text(&mut l.content, format_args!("{:.0}%", v * 100.0));
                }
            }
        }
    }
    match tool.list() {
        Some(list) => {
            let row = row_rect(o, top, w, 1 + SETTINGS);
            let caption = match list {
                List::Layers => "layer",
                List::Covers => "cover",
            };
            place_left(world, PICK_CAPTION, row, caption);
            let rect = control_rect(row);
            let tint = if point_in(mx, my, rect) {
                theme::HOVER_TINT
            } else {
                theme::BUTTON_TINT
            };
            place_rounded(world, PICK_BG, rect, tint, theme::CONTROL_RADIUS, true);
            let at = [rect[0] + 8.0, rect[1] + rect[3] * 0.5 - theme::TEXT_HALF];
            widget::place_left_label(world, PICK_LABEL, at, view.pick, theme::LABEL, true);
        }
        None => hide_pick(world),
    }
}

// The picked tool's name, and the keys that work it at the row's right end.
fn place_caption(world: &mut World, tool: Tool, row: [f32; 4]) {
    if let Some(l) = widget::label_mut(world, TOOL_CAPTION) {
        l.x = row[0] + PAD;
        l.y = row[1] + row[3] * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Left;
        l.color = theme::HEADING;
        l.visible = true;
        let key = tool.key().display_name();
        set_text(&mut l.content, format_args!("{} ({key})", tool.name()));
    }
    if let Some(l) = widget::label_mut(world, TOOL_HINT) {
        l.x = row[0] + row[2] - PAD;
        l.y = row[1] + row[3] * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Right;
        l.color = theme::LABEL_DIM;
        l.visible = true;
        if l.content != HINT {
            l.content.clear();
            l.content.push_str(HINT);
        }
    }
}

// Write `args` over `text`, keeping its storage.
fn set_text(text: &mut String, args: core::fmt::Arguments<'_>) {
    text.clear();
    if text.write_fmt(args).is_err() {
        text.clear();
    }
}

fn place_left(
    world: &mut World,
    id: concinnity_engine::ecs::asset_id::AssetId,
    row: [f32; 4],
    text: &str,
) {
    let at = [row[0] + PAD, row[1] + row[3] * 0.5 - theme::TEXT_HALF];
    widget::place_left_label(world, id, at, text, theme::LABEL, true);
}

fn hide_pick(world: &mut World) {
    widget::set_sprite_visible(world, PICK_BG, false);
    widget::set_label_visible(world, PICK_CAPTION, false);
    widget::set_label_visible(world, PICK_LABEL, false);
}

fn hide_brush(world: &mut World) {
    widget::set_label_visible(world, TOOL_CAPTION, false);
    widget::set_label_visible(world, TOOL_HINT, false);
    for s in Setting::ALL {
        widget::set_label_visible(world, setting_label(s as usize), false);
        widget_slider::hide(world, slider_ids(s));
    }
    hide_pick(world);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::planet_tools::Brush;

    fn view(tools: &Tools) -> StripView<'_> {
        StripView {
            tools,
            pick: "meadow",
            sliding: None,
            mouse: [0.0, 0.0],
        }
    }

    const O: [f32; 2] = [20.0, 40.0];
    const TOP: f32 = 120.0;
    const W: f32 = 400.0;

    fn center(r: [f32; 4]) -> (f32, f32) {
        (r[0] + r[2] * 0.5, r[1] + r[3] * 0.5)
    }

    #[test]
    fn every_button_fits_and_picks_its_tool() {
        let tools = Tools::default();
        let v = view(&tools);
        for (i, tool) in Tool::ALL.into_iter().enumerate() {
            let b = button_rect(O, TOP, i);
            assert!(b[0] + b[2] <= O[0] + W - PAD);
            assert_eq!(b[2], b[3], "square");
            let (x, y) = center(b);
            assert_eq!(
                hit_test(&v, x, y, O, TOP, W),
                Some(StripAction::Toggle(tool))
            );
        }
    }

    // The brush rows show only under a picked tool, a paint tool adding its
    // pick, and the strip grows to hold them.
    #[test]
    fn the_strip_grows_with_the_brush_rows() {
        let mut tools = Tools::default();
        let bare = height(&tools);
        tools.toggle(Tool::Raise);
        let brush = height(&tools);
        tools.toggle(Tool::PaintCover);
        let paint = height(&tools);
        assert!(bare < brush && brush < paint);
        assert_eq!(paint - brush, ROW_H);
    }

    #[test]
    fn sliders_and_the_pick_resolve_under_a_tool() {
        let mut tools = Tools {
            brush: Brush::default(),
            ..Tools::default()
        };
        let r = slider_rect(O, TOP, W, Setting::Strength);
        let (x, y) = (r[0] + 5.0, r[1] + r[3] * 0.5);
        // Without a tool the strip ends under its buttons.
        assert_eq!(hit_test(&view(&tools), x, y, O, TOP, W), None);
        tools.toggle(Tool::PaintGrass);
        let Some(StripAction::Slide(Setting::Strength, t)) =
            hit_test(&view(&tools), x, y, O, TOP, W)
        else {
            panic!("the strength slider takes the press");
        };
        assert!(t < 0.05, "{t}");
        let (x, y) = center(pick_rect(O, TOP, W));
        assert_eq!(
            hit_test(&view(&tools), x, y, O, TOP, W),
            Some(StripAction::Cycle(List::Layers))
        );
        assert_eq!(hit_test(&view(&tools), x, TOP - 1.0, O, TOP, W), None);
    }
}

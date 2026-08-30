// src/editor/character_shape_panel.rs
//
// The CharacterShape panel's layout half: a floating panel of the schema's
// sections (see `character_shape.rs` for the rows), preset rows, Reset /
// Randomize buttons and a status line in its header row. Each slider row is a caption plus one drag
// slider (`widget_slider.rs`) backed by a per-visible-row control pool at
// reserved ids; the row list scrolls through a window sized by the panel
// height. Plain `Sprite` / `TextLabel` components driven each frame by the
// editor hook, which owns the drag, the commit path, and the selection.

use super::character_shape::{Row, SliderRow};
use super::registry::{self, PanelKey};
use super::theme;
use super::widget::{self, place_rounded, point_in};
use super::widget_slider::{self, SliderIds};
use crate::components::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::CharacterShape);
pub(crate) const PANEL_BG: AssetId = AssetId(BASE);
pub(crate) const TITLE_LABEL: AssetId = AssetId(BASE + 2);
pub(crate) const CLOSE_BG: AssetId = AssetId(BASE + 3);
pub(crate) const CLOSE_LABEL: AssetId = AssetId(BASE + 4);
pub(crate) const RESET_BG: AssetId = AssetId(BASE + 5);
pub(crate) const RESET_LABEL: AssetId = AssetId(BASE + 6);
pub(crate) const RANDOM_BG: AssetId = AssetId(BASE + 7);
pub(crate) const RANDOM_LABEL: AssetId = AssetId(BASE + 8);
pub(crate) const STATUS_LABEL: AssetId = AssetId(BASE + 9);

// The visible-row pool: the most rows the window can show at once. A mesh
// with more rows than fit scrolls.
pub(crate) const MAX_ROWS: usize = 40;

// Per-visible-row chrome and slider controls (pool index).
pub(crate) fn row_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x20 + i as u32)
}
pub(crate) fn row_label(i: usize) -> AssetId {
    AssetId(BASE + 0x60 + i as u32)
}
pub(crate) fn slider_ids(i: usize) -> SliderIds {
    SliderIds {
        track: AssetId(BASE + 0xA0 + i as u32),
        fill: AssetId(BASE + 0xE0 + i as u32),
        handle: AssetId(BASE + 0x120 + i as u32),
        value: AssetId(BASE + 0x160 + i as u32),
    }
}

// Geometry, in window pixels. Every rect derives from the panel origin `o`.
pub(crate) const SHAPE_W: f32 = 400.0;
// Rows shown before the user resizes the panel taller.
pub(crate) const DEFAULT_ROWS: usize = 18;
const PAD: f32 = 10.0;
const HEADER_H: f32 = 32.0;
const BTN_W: f32 = 86.0;
const BTN_GAP: f32 = 6.0;
pub(crate) const ROW_H: f32 = 24.0;
const LABEL_COL: f32 = 130.0;

const HEADER_ROW_TINT: [f32; 4] = [0.16, 0.17, 0.22, 1.0];
const ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const ROW_TINT_HOVER: [f32; 4] = theme::HOVER_TINT;
const BTN_TINT: [f32; 4] = [0.22, 0.40, 0.56, 1.0];
const BTN_TINT_HOVER: [f32; 4] = [0.28, 0.48, 0.66, 1.0];
const HEADER_LABEL: [f32; 3] = [0.70, 0.74, 0.84];
const ADD_LABEL: [f32; 3] = [0.55, 0.80, 0.60];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];
const LABEL_TOP: f32 = ROW_H * 0.5 - theme::TEXT_HALF;

// The per-frame view the hook assembles.
pub(crate) struct ShapeView<'a> {
    pub rows: &'a [Row],
    // Section captions `Row::Header` indexes.
    pub sections: &'a [String],
    pub sliders: &'a [SliderRow],
    // Preset names `Row::Preset` indexes.
    pub presets: &'a [String],
    // One current value per slider row.
    pub values: &'a [f32],
    // First visible row.
    pub scroll: usize,
    // The slider row being dragged, if any.
    pub dragging: Option<usize>,
    // The edited shape's name in the heading; `None` while nothing is
    // selected (the status line then says what to select).
    pub shape: Option<&'a str>,
    pub status: Option<&'a str>,
    pub mouse: [f32; 2],
}

// A resolved panel click.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ShapeAction {
    Reset,
    Randomize,
    // Create a shape targeting the selected mesh.
    Add,
    // Apply the schema preset at this index.
    Preset(usize),
    // Start dragging slider row `slider` on the slider drawn at `rect`.
    Drag { slider: usize, rect: [f32; 4] },
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Where the panel sits until the user drags it: below the Lighting panel's
// default anchor on the window's left edge.
pub(crate) fn default_origin() -> [f32; 2] {
    let light = super::lighting_panel::default_origin();
    [light[0], light[1] + 40.0]
}

// The panel footprint for `n_rows` visible rows.
pub(crate) fn size(n_rows: usize) -> [f32; 2] {
    [
        SHAPE_W,
        widget::TITLE_H + HEADER_H + n_rows as f32 * ROW_H + PAD,
    ]
}

// How many rows a panel `h` tall shows, capped by the pool.
pub(crate) fn window(h: f32) -> usize {
    (((h - widget::TITLE_H - HEADER_H - PAD) / ROW_H)
        .floor()
        .max(0.0) as usize)
        .min(MAX_ROWS)
}

// The header buttons, pinned to the header row's right end: Randomize at the
// edge, Reset beside it.
pub(crate) fn random_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    [
        o[0] + w - PAD - BTN_W,
        o[1] + widget::TITLE_H + 4.0,
        BTN_W,
        HEADER_H - 8.0,
    ]
}
pub(crate) fn reset_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    let r = random_rect(o, w);
    [r[0] - BTN_GAP - BTN_W, r[1], BTN_W, r[3]]
}

// Visible row `i`, stacked below the header row.
pub(crate) fn row_rect(o: [f32; 2], w: f32, i: usize) -> [f32; 4] {
    [
        o[0],
        o[1] + widget::TITLE_H + HEADER_H + i as f32 * ROW_H,
        w,
        ROW_H,
    ]
}

// The slider on a row's right side.
pub(crate) fn slider_rect(row: [f32; 4]) -> [f32; 4] {
    let x = row[0] + LABEL_COL;
    [x, row[1], (row[0] + row[2] - PAD - x).max(1.0), row[3]]
}

// Whether the cursor is over the scrolling row area.
pub(crate) fn cursor_over_rows(mx: f32, my: f32, o: [f32; 2], s: [f32; 2]) -> bool {
    let top = o[1] + widget::TITLE_H + HEADER_H;
    point_in(mx, my, [o[0], top, s[0], (o[1] + s[1] - top).max(0.0)])
}

// Resolve a click at `(mx, my)` against the panel at origin `o`, effective
// size `s`. `None` means the click missed the panel. Title-bar presses never
// reach this: the shared routing intercepts them first.
pub(crate) fn hit_test(
    view: &ShapeView,
    mx: f32,
    my: f32,
    o: [f32; 2],
    s: [f32; 2],
) -> Option<ShapeAction> {
    let w = s[0];
    if view.shape.is_some() {
        if point_in(mx, my, reset_rect(o, w)) {
            return Some(ShapeAction::Reset);
        }
        if point_in(mx, my, random_rect(o, w)) {
            return Some(ShapeAction::Randomize);
        }
    }
    let window = window(s[1]);
    for (i, row) in view.rows.iter().skip(view.scroll).take(window).enumerate() {
        let r = row_rect(o, w, i);
        if !point_in(mx, my, r) {
            continue;
        }
        return Some(match *row {
            Row::Header(_) | Row::PresetHeader => ShapeAction::Consume,
            Row::Add => ShapeAction::Add,
            Row::Preset(i) => ShapeAction::Preset(i),
            Row::Slider(slider) => {
                let rect = slider_rect(r);
                if widget_slider::hit(rect, mx, my) {
                    ShapeAction::Drag { slider, rect }
                } else {
                    ShapeAction::Consume
                }
            }
        });
    }
    point_in(mx, my, widget::outer_rect(o, s)).then_some(ShapeAction::Consume)
}

fn place_button(
    world: &mut World,
    bg: AssetId,
    label: AssetId,
    rect: [f32; 4],
    caption: &str,
    hovered: bool,
    shown: bool,
) {
    let tint = if hovered { BTN_TINT_HOVER } else { BTN_TINT };
    place_rounded(world, bg, rect, tint, theme::CONTROL_RADIUS, shown);
    if let Some(l) = widget::label_mut(world, label) {
        l.x = rect[0] + rect[2] * 0.5;
        l.y = rect[1] + rect[3] * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Center;
        l.color = [1.0, 1.0, 1.0];
        l.visible = shown;
        l.content = caption.to_string();
    }
}

// Position + show the panel (`Some(view)`) at effective size `s`, or blank
// every element (`None`).
pub(crate) fn apply(world: &mut World, view: Option<&ShapeView>, o: [f32; 2], s: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };
    let w = s[0];
    widget::place_panel(world, PANEL_BG, widget::outer_rect(o, s));
    let title = widget::title_rect(o, w);
    let heading = match view.shape {
        Some(name) => format!("Character Shape: {name}"),
        None => "Character Shape".to_string(),
    };
    widget::place_heading(world, TITLE_LABEL, title, &heading);
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);

    let has_shape = view.shape.is_some();
    let reset = reset_rect(o, w);
    let random = random_rect(o, w);
    let (mx, my) = (view.mouse[0], view.mouse[1]);
    place_button(
        world,
        RESET_BG,
        RESET_LABEL,
        reset,
        "Reset",
        point_in(mx, my, reset),
        has_shape,
    );
    place_button(
        world,
        RANDOM_BG,
        RANDOM_LABEL,
        random,
        "Randomize",
        point_in(mx, my, random),
        has_shape,
    );
    let left = o[0] + PAD;
    let status_right = if has_shape { reset[0] } else { o[0] + w };
    widget::place_message(
        world,
        STATUS_LABEL,
        [
            left,
            o[1] + widget::TITLE_H + HEADER_H * 0.5 - theme::TEXT_HALF,
            (status_right - PAD - left).max(0.0),
            widget::LINE_H,
        ],
        view.status.unwrap_or(""),
        if has_shape {
            ERROR_LABEL
        } else {
            theme::LABEL_DIM
        },
        view.status.is_some(),
    );

    let window = window(s[1]);
    let mut shown = 0;
    for (i, row) in view.rows.iter().skip(view.scroll).take(window).enumerate() {
        shown = i + 1;
        let r = row_rect(o, w, i);
        let hovered = point_in(mx, my, r);
        let click_tint = if hovered { ROW_TINT_HOVER } else { ROW_TINT };
        let (bg, caption, color) = match *row {
            Row::Header(section) => (
                HEADER_ROW_TINT,
                view.sections.get(section).cloned().unwrap_or_default(),
                HEADER_LABEL,
            ),
            Row::PresetHeader => (HEADER_ROW_TINT, "Presets".to_string(), HEADER_LABEL),
            Row::Preset(pi) => (
                click_tint,
                format!("apply {}", view.presets.get(pi).map_or("", String::as_str)),
                ADD_LABEL,
            ),
            Row::Add => (click_tint, "+ Add CharacterShape".to_string(), ADD_LABEL),
            Row::Slider(si) => (ROW_TINT, view.sliders[si].caption.clone(), theme::LABEL),
        };
        place_rounded(
            world,
            row_bg(i),
            theme::highlight_rect(r),
            bg,
            theme::CONTROL_RADIUS,
            true,
        );
        if let Some(l) = widget::label_mut(world, row_label(i)) {
            l.x = r[0] + PAD;
            l.y = r[1] + LABEL_TOP;
            l.align = TextAlign::Left;
            l.color = color;
            l.visible = true;
            l.content = caption;
        }
        match *row {
            Row::Slider(si) => {
                let rect = slider_rect(r);
                let hot = view.dragging == Some(si)
                    || (view.dragging.is_none() && widget_slider::hit(rect, mx, my));
                widget_slider::place(
                    world,
                    slider_ids(i),
                    rect,
                    view.values.get(si).copied().unwrap_or(0.0),
                    view.sliders[si].kind.range(),
                    hot,
                );
            }
            _ => widget_slider::hide(world, slider_ids(i)),
        }
    }
    for i in shown..MAX_ROWS {
        widget::set_sprite_visible(world, row_bg(i), false);
        widget::set_label_visible(world, row_label(i), false);
        widget_slider::hide(world, slider_ids(i));
    }
}

pub(crate) fn hide_all(world: &mut World) {
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &[]);
}

// Every panel sprite id, in draw (insertion) order: chrome, row backgrounds,
// then the slider parts above them.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG, RESET_BG, RANDOM_BG];
    ids.extend((0..MAX_ROWS).map(row_bg));
    ids.extend((0..MAX_ROWS).map(|i| slider_ids(i).track));
    ids.extend((0..MAX_ROWS).map(|i| slider_ids(i).fill));
    ids.extend((0..MAX_ROWS).map(|i| slider_ids(i).handle));
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![
        TITLE_LABEL,
        CLOSE_LABEL,
        RESET_LABEL,
        RANDOM_LABEL,
        STATUS_LABEL,
    ];
    ids.extend((0..MAX_ROWS).map(row_label));
    ids.extend((0..MAX_ROWS).map(|i| slider_ids(i).value));
    ids
}

#[cfg(test)]
mod tests {
    use super::super::character_shape::{self, Rows};
    use super::*;
    use crate::components::{Sprite, TextLabel};
    use concinnity_cook::compile::character::builtin_schema;

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
        world
    }

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn fixture() -> (Rows, Vec<Row>, Vec<f32>) {
        let derived = character_shape::derive_rows(
            builtin_schema::humanoid(),
            &names(&["jaw+", "jaw-", "muscle"]),
            &names(&["thigh_l", "thigh_r"]),
        );
        let rows = character_shape::rows(&derived, 0, true);
        let values = vec![0.25, 0.5, -0.5];
        (derived, rows, values)
    }

    fn view<'a>(
        rows: &'a [Row],
        derived: &'a Rows,
        values: &'a [f32],
        shape: Option<&'a str>,
    ) -> ShapeView<'a> {
        ShapeView {
            rows,
            sections: &derived.sections,
            sliders: &derived.sliders,
            presets: &[],
            values,
            scroll: 0,
            dragging: None,
            shape,
            status: None,
            mouse: [0.0, 0.0],
        }
    }

    #[test]
    fn hit_test_resolves_buttons_rows_and_sliders() {
        let (derived, rows, values) = fixture();
        let v = view(&rows, &derived, &values, Some("body_shape"));
        let o = [40.0, 40.0];
        let s = size(rows.len());
        let r = reset_rect(o, SHAPE_W);
        assert_eq!(
            hit_test(&v, r[0] + 5.0, r[1] + 5.0, o, s),
            Some(ShapeAction::Reset)
        );
        let r = random_rect(o, SHAPE_W);
        assert_eq!(
            hit_test(&v, r[0] + 5.0, r[1] + 5.0, o, s),
            Some(ShapeAction::Randomize)
        );
        // Row 0 is the Face header; row 1 the jaw slider.
        let r0 = row_rect(o, SHAPE_W, 0);
        assert_eq!(
            hit_test(&v, r0[0] + 5.0, r0[1] + 5.0, o, s),
            Some(ShapeAction::Consume)
        );
        let r1 = row_rect(o, SHAPE_W, 1);
        let rect = slider_rect(r1);
        assert_eq!(
            hit_test(&v, rect[0] + 20.0, r1[1] + 5.0, o, s),
            Some(ShapeAction::Drag { slider: 0, rect })
        );
        // The caption column does not start a drag.
        assert_eq!(
            hit_test(&v, r1[0] + 5.0, r1[1] + 5.0, o, s),
            Some(ShapeAction::Consume)
        );
        assert_eq!(hit_test(&v, 5000.0, 5000.0, o, s), None);
    }

    #[test]
    fn hit_test_without_a_shape_offers_only_the_add_row() {
        let derived = Rows::default();
        let rows = character_shape::rows(&derived, 0, false);
        let v = view(&rows, &derived, &[], None);
        let o = [0.0, 0.0];
        let s = size(rows.len());
        let r = reset_rect(o, SHAPE_W);
        assert_eq!(
            hit_test(&v, r[0] + 5.0, r[1] + 5.0, o, s),
            Some(ShapeAction::Consume),
            "hidden buttons do not act"
        );
        let r0 = row_rect(o, SHAPE_W, 0);
        assert_eq!(
            hit_test(&v, r0[0] + 5.0, r0[1] + 5.0, o, s),
            Some(ShapeAction::Add)
        );
    }

    #[test]
    fn scrolled_rows_shift_the_pool() {
        let (derived, rows, values) = fixture();
        let mut v = view(&rows, &derived, &values, Some("body_shape"));
        v.scroll = 1;
        let o = [0.0, 0.0];
        // Two visible rows: the jaw slider and the Torso header.
        let s = size(2);
        assert_eq!(window(s[1]), 2);
        let r0 = row_rect(o, SHAPE_W, 0);
        let rect = slider_rect(r0);
        assert_eq!(
            hit_test(&v, rect[0] + 20.0, r0[1] + 5.0, o, s),
            Some(ShapeAction::Drag { slider: 0, rect })
        );
        // The row past the window is not drawn: its slider cannot be grabbed.
        let r2 = row_rect(o, SHAPE_W, 2);
        let rect = slider_rect(r2);
        assert_eq!(
            hit_test(&v, rect[0] + 20.0, r2[1] + 5.0, o, s),
            Some(ShapeAction::Consume)
        );
    }

    #[test]
    fn apply_draws_sections_sliders_and_values() {
        let mut world = injected_world();
        let (derived, rows, values) = fixture();
        let v = view(&rows, &derived, &values, Some("body_shape"));
        apply(&mut world, Some(&v), [20.0, 20.0], size(rows.len()));
        let label = |world: &World, id: AssetId| {
            world
                .query::<TextLabel>()
                .find(|l| l.asset_id == id)
                .unwrap()
                .clone()
        };
        let sprite_visible = |world: &World, id: AssetId| {
            world
                .query::<Sprite>()
                .find(|s| s.asset_id == id)
                .unwrap()
                .visible
        };
        assert_eq!(
            label(&world, TITLE_LABEL).content,
            "Character Shape: body_shape"
        );
        assert_eq!(label(&world, row_label(0)).content, "Face");
        assert_eq!(label(&world, row_label(1)).content, "jaw");
        assert_eq!(label(&world, slider_ids(1).value).content, "+0.25");
        assert!(sprite_visible(&world, slider_ids(1).handle));
        assert!(
            !sprite_visible(&world, slider_ids(0).handle),
            "a header row has no slider"
        );
        assert!(sprite_visible(&world, RESET_BG));
        // Without a shape the buttons hide and the status shows.
        let none = Rows::default();
        let no_rows = character_shape::rows(&none, 0, false);
        let mut nv = view(&no_rows, &none, &[], None);
        nv.status = Some("Select a SkinnedMesh");
        apply(&mut world, Some(&nv), [20.0, 20.0], size(1));
        assert!(!sprite_visible(&world, RESET_BG));
        assert!(label(&world, STATUS_LABEL).visible);
        assert!(
            !sprite_visible(&world, slider_ids(1).handle),
            "stale slider rows blank"
        );
    }

    #[test]
    fn preset_rows_resolve_to_their_index_and_show_their_name() {
        let mut world = injected_world();
        let (derived, _, values) = fixture();
        let rows = character_shape::rows(&derived, 2, true);
        let presets = names(&["slim", "heavy"]);
        let mut v = view(&rows, &derived, &values, Some("body_shape"));
        v.presets = &presets;
        let o = [0.0, 0.0];
        let s = size(rows.len());
        let r1 = row_rect(o, SHAPE_W, 1);
        assert_eq!(
            hit_test(&v, r1[0] + 5.0, r1[1] + 5.0, o, s),
            Some(ShapeAction::Preset(0))
        );
        let r2 = row_rect(o, SHAPE_W, 2);
        assert_eq!(
            hit_test(&v, r2[0] + 5.0, r2[1] + 5.0, o, s),
            Some(ShapeAction::Preset(1))
        );
        apply(&mut world, Some(&v), o, s);
        let label = |id: AssetId| {
            world
                .query::<TextLabel>()
                .find(|l| l.asset_id == id)
                .unwrap()
                .content
                .clone()
        };
        assert_eq!(label(row_label(0)), "Presets");
        assert_eq!(label(row_label(2)), "apply heavy");
        assert_eq!(label(row_label(3)), "Face");
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        let (derived, rows, values) = fixture();
        let v = view(&rows, &derived, &values, Some("body_shape"));
        apply(&mut world, Some(&v), [20.0, 20.0], size(rows.len()));
        apply(&mut world, None, [0.0, 0.0], size(0));
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}

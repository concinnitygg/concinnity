//! The Shaders panel's layout half: a floating, scrollable list of the rows
//! `shader_list` builds. Each Shader is a heading row (badged "default" on the
//! first), the Materials naming it, and a row per file with that file's last
//! reload status on the right, then "+ Add vertex file" for a Shader without
//! one; "+ New Shader" ends the list. Clicking a file opens it in the Shader
//! source panel (`shader_source_panel.rs`). A heading's "..." menu renames or
//! deletes its Shader, and a vertex file's removes that file.

use concinnity_core::components::TextAlign;
use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;

use super::asset_list::{self, ROW_H};
use super::registry::{self, PanelKey};
use super::shader_diagnostics::Tone;
use super::shader_list::{MenuItem, Row, RowKind};
use crate::editor::theme;
use crate::editor::widget::{self, place_rounded, point_in};
use crate::editor::widget_menu::{self, MenuIds};

const BASE: u32 = registry::base(PanelKey::Shaders);
pub(crate) const PANEL_BG: AssetId = AssetId(BASE);
const TITLE_LABEL: AssetId = AssetId(BASE + 1);
const CLOSE_BG: AssetId = AssetId(BASE + 2);
const CLOSE_LABEL: AssetId = AssetId(BASE + 3);
const LIST_TRACK: AssetId = AssetId(BASE + 4);
const LIST_THUMB: AssetId = AssetId(BASE + 5);
const MENU: MenuIds = MenuIds {
    dot_bg: AssetId(BASE + 6),
    dots: [AssetId(BASE + 7), AssetId(BASE + 8), AssetId(BASE + 9)],
    bg: AssetId(BASE + 10),
    item_bgs: [AssetId(BASE + 11), AssetId(BASE + 12)],
    item_labels: [AssetId(BASE + 13), AssetId(BASE + 14)],
};

// The row pools sit above the chrome ids, one sub-range per element.
const POOL_MAX: usize = 32;
fn row_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x40 + i as u32)
}
fn row_label(i: usize) -> AssetId {
    AssetId(BASE + 0x80 + i as u32)
}
fn badge_label(i: usize) -> AssetId {
    AssetId(BASE + 0xC0 + i as u32)
}

// The default (and minimum) width; the user can widen the panel past this.
const SHADERS_W: f32 = 400.0;
// Rows shown before the list scrolls, until the panel is resized taller.
const DEFAULT_ROWS: usize = 12;
const PAD: f32 = 8.0;
const BOTTOM_PAD: f32 = 6.0;
// Room kept at a row's right end for its badge, left of the dots' slot.
const BADGE_W: f32 = 120.0;
const DOT_GAP: f32 = 4.0;
// The dots' slot, held on every row so a badge never moves to make room.
const DOT_INSET: f32 = PAD + asset_list::SCROLLBAR_W;
const CHROME_H: f32 = widget::TITLE_H + BOTTOM_PAD;

const HEADING_COLOR: [f32; 3] = [0.58, 0.66, 0.80];
const ADD_COLOR: [f32; 3] = [0.55, 0.80, 0.60];

pub(crate) fn tone_color(tone: Tone) -> [f32; 3] {
    match tone {
        Tone::Info => theme::LABEL_DIM,
        Tone::Warning => theme::LOG_WARN,
        Tone::Error => theme::LOG_ERROR,
    }
}

// The per-frame view the hook assembles.
pub(crate) struct ShadersView<'a> {
    pub rows: &'a [Row],
    // First visible row of the scroll window.
    pub scroll: usize,
    pub mouse: [f32; 2],
    // The row whose menu is open.
    pub menu: Option<usize>,
}

// A resolved Shaders-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShadersAction {
    // Row `i` of the view's rows (a clickable one).
    Row(usize),
    // Row `i`'s dots: open its menu.
    OpenMenu(usize),
    // An item of the open menu.
    Menu(MenuItem),
    // A click off the open menu: close it.
    CloseMenu,
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Where the panel sits until the user drags it: right of center, below the
// top bar.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [vw * 0.5 + 40.0, crate::editor::hud::body_top() + 20.0]
}

fn panel_height(rows: usize) -> f32 {
    CHROME_H + rows as f32 * ROW_H
}

// The default footprint for `n_rows` rows: all of them, up to the default
// window.
pub(crate) fn size(n_rows: usize) -> [f32; 2] {
    [SHADERS_W, panel_height(n_rows.clamp(1, DEFAULT_ROWS))]
}

// The tallest the panel resizes to: every row shown, up to the pool.
pub(crate) fn max_size(n_rows: usize) -> [f32; 2] {
    [f32::INFINITY, panel_height(n_rows.clamp(1, POOL_MAX))]
}

// The rows that fit a panel `h` tall, capped at the pool.
pub(crate) fn rows_for_height(h: f32) -> usize {
    (((h - CHROME_H) / ROW_H).floor() as usize).clamp(1, POOL_MAX)
}

fn body_top(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H
}

fn row_rect(o: [f32; 2], w: f32, r: usize) -> [f32; 4] {
    [o[0], body_top(o) + r as f32 * ROW_H, w, ROW_H]
}

fn dot_rect(o: [f32; 2], w: f32, r: usize) -> [f32; 4] {
    widget_menu::dot_rect(row_rect(o, w, r), DOT_INSET)
}

fn first_shown(view: &ShadersView, window: usize) -> usize {
    view.scroll.min(view.rows.len().saturating_sub(window))
}

// The open menu's backing and item rects, while its row is on screen.
fn open_menu_rects(
    view: &ShadersView,
    o: [f32; 2],
    s: [f32; 2],
) -> Option<(usize, [f32; 4], Vec<[f32; 4]>)> {
    let i = view.menu?;
    let window = rows_for_height(s[1]);
    let slot = i.checked_sub(first_shown(view, window))?;
    let count = view.rows.get(i)?.menu().len();
    if slot >= window || count == 0 {
        return None;
    }
    let right = o[0] + s[0] - DOT_INSET;
    let (bg, items) = widget_menu::menu_rects(row_rect(o, s[0], slot), right, count, o[1] + s[1]);
    Some((slot, bg, items))
}

pub(crate) fn cursor_over(mx: f32, my: f32, o: [f32; 2], s: [f32; 2]) -> bool {
    point_in(mx, my, widget::outer_rect(o, s))
}

// Resolve a click against the panel at origin `o`, size `s`. `None` means the
// click missed the panel.
pub(crate) fn hit_test(
    view: &ShadersView,
    mx: f32,
    my: f32,
    o: [f32; 2],
    s: [f32; 2],
) -> Option<ShadersAction> {
    if !point_in(mx, my, widget::outer_rect(o, s)) {
        return None;
    }
    // An open menu is modal over the panel: its items pick, anything else
    // closes it.
    if view.menu.is_some() {
        let item = open_menu_rects(view, o, s).and_then(|(_, _, rects)| {
            let menu = view.rows[view.menu?].menu();
            widget_menu::hit_item(mx, my, &rects).map(|k| menu[k])
        });
        return Some(item.map_or(ShadersAction::CloseMenu, ShadersAction::Menu));
    }
    let window = rows_for_height(s[1]);
    let scroll = first_shown(view, window);
    let Some(i) = (0..window)
        .map(|r| scroll + r)
        .take_while(|&i| i < view.rows.len())
        .find(|&i| point_in(mx, my, row_rect(o, s[0], i - scroll)))
    else {
        return Some(ShadersAction::Consume);
    };
    let row = &view.rows[i];
    Some(
        if !row.menu().is_empty() && point_in(mx, my, dot_rect(o, s[0], i - scroll)) {
            ShadersAction::OpenMenu(i)
        } else if row.clickable() {
            ShadersAction::Row(i)
        } else {
            ShadersAction::Consume
        },
    )
}

// Position + show the panel at origin `o`, effective size `s`, or hide it all.
pub(crate) fn place(world: &mut World, view: Option<&ShadersView>, o: [f32; 2], s: [f32; 2]) {
    hide_all(world);
    let Some(view) = view else {
        return;
    };
    let w = s[0];
    widget::place_panel(world, PANEL_BG, widget::outer_rect(o, s));
    let title = widget::title_rect(o, w);
    widget::place_heading(world, TITLE_LABEL, title, "Shaders");
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);

    let window = rows_for_height(s[1]);
    let total = view.rows.len();
    let scroll = first_shown(view, window);
    for r in 0..window {
        let Some(row) = view.rows.get(scroll + r) else {
            break;
        };
        place_row(world, r, row, row_rect(o, w, r), view.mouse);
    }
    place_menu(world, view, o, s, scroll, window);
    asset_list::layout_scrollbar(
        world,
        (LIST_TRACK, LIST_THUMB),
        total,
        scroll,
        window,
        o[0] + w,
        body_top(o),
    );
}

// The dots on the row whose menu is open, else on the hovered row with a
// menu, and the open menu over the rows below it.
fn place_menu(
    world: &mut World,
    view: &ShadersView,
    o: [f32; 2],
    s: [f32; 2],
    scroll: usize,
    window: usize,
) {
    let hovered = (0..window)
        .filter(|&r| {
            view.rows
                .get(scroll + r)
                .is_some_and(|row| !row.menu().is_empty())
        })
        .find(|&r| point_in(view.mouse[0], view.mouse[1], row_rect(o, s[0], r)));
    let open = open_menu_rects(view, o, s);
    match open.as_ref().map(|(slot, _, _)| *slot).or(hovered) {
        Some(slot) => {
            let d = dot_rect(o, s[0], slot);
            let boxed = open.is_some() || point_in(view.mouse[0], view.mouse[1], d);
            widget_menu::place_dots(world, &MENU, d, boxed);
        }
        None => widget_menu::hide_dots(world, &MENU),
    }
    let Some((slot, bg, rects)) = open else {
        widget_menu::hide_menu(world, &MENU);
        return;
    };
    // Every TextLabel draws after every Sprite, so the menu's backing cannot
    // cover the captions under it; they are blanked instead.
    for covered in (0..window).filter(|&r| r != slot) {
        let r = row_rect(o, s[0], covered);
        if r[1] < bg[1] + bg[3] && r[1] + r[3] > bg[1] {
            widget::set_label_visible(world, row_label(covered), false);
            widget::set_label_visible(world, badge_label(covered), false);
        }
    }
    let items: Vec<widget_menu::Item> = view
        .menu
        .map(|i| view.rows[i].menu())
        .unwrap_or_default()
        .iter()
        .map(|m| widget_menu::Item {
            caption: m.caption(),
            danger: m.danger(),
        })
        .collect();
    widget_menu::place_menu(world, &MENU, (&bg, &rects), &items, view.mouse);
}

fn place_row(world: &mut World, slot: usize, row: &Row, rect: [f32; 4], mouse: [f32; 2]) {
    let hovered = row.clickable() && point_in(mouse[0], mouse[1], rect);
    let tint = if hovered {
        theme::HOVER_TINT
    } else if row.selected {
        theme::SELECTED_TINT
    } else {
        asset_list::ROW_TINT
    };
    place_rounded(
        world,
        row_bg(slot),
        theme::highlight_rect(rect),
        tint,
        theme::CONTROL_RADIUS,
        true,
    );
    let color = match row.kind {
        RowKind::New | RowKind::AddVertex(_) => ADD_COLOR,
        RowKind::Header(_) => HEADING_COLOR,
        RowKind::Note => theme::LABEL_DIM,
        RowKind::Materials(_) | RowKind::File(_) => theme::LABEL,
    };
    let x = rect[0] + PAD + if row.indent { asset_list::INDENT } else { 0.0 };
    let dots = DOT_INSET + widget_menu::DOT_SZ + DOT_GAP;
    let right = rect[0] + rect[2] - dots - if row.badge.is_some() { BADGE_W } else { 0.0 };
    widget::place_message(
        world,
        row_label(slot),
        [
            x,
            rect[1] + asset_list::ROW_LABEL_TOP,
            (right - x).max(0.0),
            widget::LINE_H,
        ],
        &row.text,
        color,
        true,
    );
    let Some((badge, tone)) = &row.badge else {
        return;
    };
    if let Some(l) = widget::label_mut(world, badge_label(slot)) {
        l.x = rect[0] + rect[2] - dots;
        l.y = rect[1] + asset_list::ROW_LABEL_TOP;
        l.align = TextAlign::Right;
        l.color = tone_color(*tone);
        l.visible = true;
        l.wrap_width = BADGE_W;
        l.max_lines = 1;
        l.content = badge.clone();
    }
}

pub(crate) fn hide_all(world: &mut World) {
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &[]);
}

// Every sprite id in draw order: the panel, the row highlights, the scrollbar
// over them, then a row's dots and its menu over everything.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG];
    ids.extend((0..POOL_MAX).map(row_bg));
    ids.extend([LIST_TRACK, LIST_THUMB]);
    ids.extend(MENU.sprites());
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![TITLE_LABEL, CLOSE_LABEL];
    ids.extend((0..POOL_MAX).map(row_label));
    ids.extend((0..POOL_MAX).map(badge_label));
    ids.extend(MENU.labels());
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::panels::shader_source::SourceKey;
    use concinnity_core::components::{ShaderStage, Sprite, TextLabel};

    fn injected_world() -> World {
        crate::test_support::injected_world(&all_sprite_ids(), &all_label_ids(), &[])
    }

    fn row(kind: RowKind, text: &str, badge: Option<&str>) -> Row {
        Row {
            kind,
            text: text.to_string(),
            badge: badge.map(|b| (b.to_string(), Tone::Error)),
            indent: false,
            selected: false,
        }
    }

    fn rows() -> Vec<Row> {
        vec![
            row(RowKind::Header(0), "lit", Some("default")),
            row(RowKind::Materials(0), "used by a", None),
            row(
                RowKind::File(SourceKey {
                    shader: "lit".to_string(),
                    stage: ShaderStage::Fragment,
                }),
                "fragment  lit.hlsl",
                Some("failed"),
            ),
            row(RowKind::New, "+ New Shader", None),
            row(
                RowKind::File(SourceKey {
                    shader: "lit".to_string(),
                    stage: ShaderStage::Vertex,
                }),
                "vertex  sway.hlsl",
                None,
            ),
        ]
    }

    fn label(world: &World, id: AssetId) -> TextLabel {
        world.get_by_id::<TextLabel>(id).cloned().unwrap()
    }

    #[test]
    fn a_click_resolves_clickable_rows_only() {
        let rows = rows();
        let view = ShadersView {
            rows: &rows,
            scroll: 0,
            mouse: [0.0, 0.0],
            menu: None,
        };
        let o = [40.0, 40.0];
        let s = size(rows.len());
        let at = |r: usize| {
            let rect = row_rect(o, s[0], r);
            (rect[0] + 20.0, rect[1] + 5.0)
        };
        let (x, y) = at(2);
        assert_eq!(hit_test(&view, x, y, o, s), Some(ShadersAction::Row(2)));
        let (x, y) = at(0);
        assert_eq!(
            hit_test(&view, x, y, o, s),
            Some(ShadersAction::Consume),
            "a heading does nothing"
        );
        assert_eq!(hit_test(&view, 5000.0, 5000.0, o, s), None);
        let scrolled = ShadersView { scroll: 1, ..view };
        let (x, y) = at(0);
        assert_eq!(
            hit_test(&scrolled, x, y, o, [s[0], panel_height(2)]),
            Some(ShadersAction::Row(1))
        );
    }

    #[test]
    fn place_draws_rows_and_badges() {
        let rows = rows();
        let mut world = injected_world();
        let view = ShadersView {
            rows: &rows,
            scroll: 0,
            mouse: [0.0, 0.0],
            menu: None,
        };
        place(&mut world, Some(&view), [20.0, 20.0], size(rows.len()));
        assert_eq!(label(&world, TITLE_LABEL).content, "Shaders");
        assert_eq!(label(&world, row_label(0)).content, "lit");
        assert_eq!(label(&world, badge_label(0)).content, "default");
        let badge = label(&world, badge_label(2));
        assert!(badge.visible && badge.content == "failed");
        assert_eq!(badge.color, theme::LOG_ERROR);
        assert!(!label(&world, badge_label(1)).visible);
        assert_eq!(label(&world, row_label(3)).content, "+ New Shader");
    }

    // The dots open a row's menu; with it open its items pick and any other
    // click on the panel closes it.
    #[test]
    fn the_dots_open_a_menu_whose_items_pick() {
        let rows = rows();
        let o = [40.0, 40.0];
        let s = size(rows.len());
        let view = ShadersView {
            rows: &rows,
            scroll: 0,
            mouse: [0.0, 0.0],
            menu: None,
        };
        let d = dot_rect(o, s[0], 0);
        let (x, y) = (d[0] + 2.0, d[1] + 2.0);
        assert_eq!(
            hit_test(&view, x, y, o, s),
            Some(ShadersAction::OpenMenu(0))
        );
        let d = dot_rect(o, s[0], 2);
        assert_eq!(
            hit_test(&view, d[0] + 2.0, d[1] + 2.0, o, s),
            Some(ShadersAction::Row(2)),
            "the fragment row has no menu"
        );

        let open = ShadersView {
            menu: Some(0),
            ..view
        };
        let (_, _, items) = open_menu_rects(&open, o, s).unwrap();
        let at = |r: [f32; 4]| (r[0] + 2.0, r[1] + 2.0);
        let (x, y) = at(items[1]);
        assert_eq!(
            hit_test(&open, x, y, o, s),
            Some(ShadersAction::Menu(MenuItem::Delete))
        );
        let (x, y) = at(row_rect(o, s[0], 3));
        assert_eq!(
            hit_test(&open, x + 10.0, y, o, s),
            Some(ShadersAction::CloseMenu)
        );
        assert_eq!(hit_test(&open, 5000.0, 5000.0, o, s), None);

        let vertex = ShadersView {
            menu: Some(4),
            ..open
        };
        let (_, _, items) = open_menu_rects(&vertex, o, s).unwrap();
        let (x, y) = at(items[0]);
        assert_eq!(
            hit_test(&vertex, x, y, o, s),
            Some(ShadersAction::Menu(MenuItem::RemoveVertex))
        );
    }

    #[test]
    fn an_open_menu_draws_over_the_rows_it_covers() {
        let rows = rows();
        let mut world = injected_world();
        let o = [20.0, 20.0];
        let s = size(rows.len());
        let view = ShadersView {
            rows: &rows,
            scroll: 0,
            mouse: [0.0, 0.0],
            menu: Some(0),
        };
        place(&mut world, Some(&view), o, s);
        assert!(label(&world, MENU.item_labels[0]).visible);
        assert_eq!(label(&world, MENU.item_labels[1]).content, "Delete");
        assert!(
            !label(&world, badge_label(2)).visible,
            "covered by the menu"
        );
        assert!(label(&world, row_label(0)).visible, "its own row stays");
        let closed = ShadersView { menu: None, ..view };
        place(&mut world, Some(&closed), o, s);
        assert!(!label(&world, MENU.item_labels[0]).visible);
    }

    #[test]
    fn the_panel_grows_with_its_rows_up_to_the_pool() {
        assert!(size(3)[1] < size(DEFAULT_ROWS + 5)[1]);
        assert_eq!(size(DEFAULT_ROWS + 5), size(DEFAULT_ROWS));
        assert_eq!(rows_for_height(max_size(POOL_MAX + 10)[1]), POOL_MAX);
        assert_eq!(rows_for_height(size(4)[1]), 4);
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let rows = rows();
        let mut world = injected_world();
        let view = ShadersView {
            rows: &rows,
            scroll: 0,
            mouse: [0.0, 0.0],
            menu: None,
        };
        place(&mut world, Some(&view), [20.0, 20.0], size(rows.len()));
        place(&mut world, None, [0.0, 0.0], size(1));
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}

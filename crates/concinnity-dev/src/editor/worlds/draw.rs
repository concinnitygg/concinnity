// src/editor/worlds/draw.rs
//
// The Worlds panel's per-frame layout. One pass drives both presentations: the
// rects come from `geometry.rs` at the view's metrics, and every caption is
// drawn at that presentation's text scale, so switching from the start screen
// to the switcher resizes the panel's text with it.

use super::geometry::{Metrics, Mode};
use super::*;
use crate::components::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;
use crate::editor::theme;
use crate::editor::widget::{self, place_rounded, point_in};

const ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
const DOT_BG_TINT: [f32; 4] = [0.30, 0.34, 0.46, 0.95];
const DOT_TINT: [f32; 4] = [0.90, 0.92, 0.96, 1.0];
// Lifted clear of the panel behind it and framed by the shared hairline, as the
// Assets panel's row menu is, so it reads as a floating surface.
const MENU_BG_TINT: [f32; 4] = [0.22, 0.23, 0.29, 1.0];
const MENU_ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const HEADER_LABEL: [f32; 3] = [0.70, 0.74, 0.84];
const STATUS_LABEL_COLOR: [f32; 3] = [0.95, 0.55, 0.55];
// Delete carries its warning in the text, not in a filled red button.
const DELETE_LABEL: [f32; 3] = [0.95, 0.60, 0.58];

// Position + show the panel (`Some(view)`) at origin `o`, or blank every
// element (`None`).
pub(crate) fn apply(world: &mut World, view: Option<&WorldsView>, o: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };
    let m = *view.layout.metrics();
    widget::place_panel_tinted(world, PANEL_BG, view.layout.panel_rect(o), m.panel_tint);
    header(world, view, &m, o);
    list_header(world, view, &m, o);
    for slot in 0..POOL {
        row(world, view, &m, o, slot);
    }
    scrollbar(world, view, o);
    row_affordances(world, view, &m, o);
}

// The title bar (where there is one), the name field, and New.
fn header(world: &mut World, view: &WorldsView, m: &Metrics, o: [f32; 2]) {
    let l = &view.layout;
    if m.has_title_bar() {
        let title = l.title_rect(o);
        widget::place_heading(world, TITLE_LABEL, title, "Worlds");
        scale_label(world, TITLE_LABEL, m.text_scale);
        let hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
        widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, hover);
        scale_label(world, CLOSE_LABEL, m.text_scale);
    } else {
        widget::set_label_visible(world, TITLE_LABEL, false);
        widget::set_sprite_visible(world, CLOSE_BG, false);
        widget::set_label_visible(world, CLOSE_LABEL, false);
    }

    let new = l.new_rect(o);
    let hover = point_in(view.mouse[0], view.mouse[1], new);
    let tint = if hover { theme::HOVER_TINT } else { ROW_TINT };
    place_rounded(world, NEW_BG, new, tint, theme::CONTROL_RADIUS, true);
    chip_label(world, NEW_LABEL, new, "+", m);

    let status = l.status_rect(o);
    place_wrapped(
        world,
        STATUS_LABEL,
        status,
        view.status.unwrap_or(""),
        STATUS_LABEL_COLOR,
        view.status.is_some(),
        m,
    );
}

// The caption above the rows: the listing's size, or what to do about an empty
// one. The sidebar's column is narrow, so it says the same thing in fewer
// words.
fn list_header(world: &mut World, view: &WorldsView, m: &Metrics, o: [f32; 2]) {
    let content = match (view.rows.is_empty(), view.layout.mode) {
        (true, Mode::Start) => "No worlds yet".to_string(),
        (true, Mode::Session) => "No worlds yet - press + to start one".to_string(),
        (false, _) => format!("Worlds ({})", view.rows.len()),
    };
    let pos = view.layout.list_header_pos(o);
    widget::place_left_label(world, LIST_HEADER, pos, &content, HEADER_LABEL, true);
    scale_label(world, LIST_HEADER, m.text_scale);
}

// One row slot: the world's name over its highlight. Slots past the listing
// draw nothing. The triple-dot and its menu float over the rows and are laid
// out afterwards (`row_affordances`), so they cannot be covered.
fn row(world: &mut World, view: &WorldsView, m: &Metrics, o: [f32; 2], slot: usize) {
    let l = &view.layout;
    let i = view.scroll + slot;
    let listed = (slot < l.rows()).then(|| view.rows.get(i)).flatten();
    let Some(entry) = listed else {
        widget::set_sprite_visible(world, row_bg(slot), false);
        widget::set_label_visible(world, row_label(slot), false);
        return;
    };
    let r = l.row_rect(o, slot);
    // The start screen marks its selection; the switcher marks the world the
    // session has open.
    let picked = match l.mode {
        Mode::Start => view.selected == Some(i),
        Mode::Session => entry.open,
    };
    let tint = if view.menu == Some(i) || point_in(view.mouse[0], view.mouse[1], r) {
        theme::HOVER_TINT
    } else if picked {
        theme::SELECTED_TINT
    } else {
        ROW_TINT
    };
    place_rounded(
        world,
        row_bg(slot),
        theme::highlight_rect(r),
        tint,
        theme::CONTROL_RADIUS,
        true,
    );
    if let Some(label) = widget::label_mut(world, row_label(slot)) {
        label.x = r[0] + m.pad;
        label.y = r[1] + m.row_h * 0.5 - m.text_half();
        label.align = TextAlign::Left;
        label.color = theme::LABEL;
        label.scale = m.text_scale;
        label.visible = true;
        label.content = widget::clip_text(&entry.name, m.max_row_chars);
    }
}

// The triple-dot on the row offering it, and the menu it opened. The dots go to
// the row whose menu is up, or to the hovered one; nothing else carries them.
fn row_affordances(world: &mut World, view: &WorldsView, m: &Metrics, o: [f32; 2]) {
    let l = &view.layout;
    let hovered = (0..l.rows())
        .find(|&slot| {
            view.rows.len() > view.scroll + slot
                && point_in(view.mouse[0], view.mouse[1], l.row_rect(o, slot))
        })
        .map(|slot| view.scroll + slot);
    let active = view.menu.or(hovered);
    match active.and_then(|i| view.slot_of(i)) {
        Some(slot) => {
            let d = l.dot_rect(o, slot);
            let boxed = view.menu.is_some() || point_in(view.mouse[0], view.mouse[1], d);
            place_dot(world, d, boxed);
        }
        None => {
            for id in [DOT_BG, DOT1, DOT2, DOT3] {
                widget::set_sprite_visible(world, id, false);
            }
        }
    }
    match view.menu.and_then(|i| view.slot_of(i)) {
        Some(slot) => row_menu(world, view, m, o, slot),
        None => hide_row_menu(world),
    }
}

// The three stacked dots. The backing box shows only while the dots are hovered
// or their menu is open; a plain row-hover shows the bare dots.
fn place_dot(world: &mut World, d: [f32; 4], boxed: bool) {
    match boxed {
        true => place_rounded(world, DOT_BG, d, DOT_BG_TINT, theme::CONTROL_RADIUS, true),
        false => widget::set_sprite_visible(world, DOT_BG, false),
    }
    let (cx, cy) = (d[0] + d[2] * 0.5, d[1] + d[3] * 0.5);
    let (s, gap) = (3.5, 3.5);
    for (id, dy) in [(DOT1, -gap - s), (DOT2, -s * 0.5), (DOT3, gap)] {
        widget::place_sprite(world, id, [cx - s * 0.5, cy + dy, s, s], DOT_TINT, true);
    }
}

fn row_menu(world: &mut World, view: &WorldsView, m: &Metrics, o: [f32; 2], slot: usize) {
    let (bg, open, delete) = view.layout.menu_rects(o, slot);
    // Blank the captions the menu floats over: every TextLabel draws after
    // every Sprite, so its opaque backing cannot cover them. The menu's own row
    // is laid out clear of it, above or below, and keeps its name.
    for covered in 0..view.layout.rows() {
        let r = view.layout.row_rect(o, covered);
        if covered != slot && r[1] < bg[1] + bg[3] && r[1] + r[3] > bg[1] {
            widget::set_label_visible(world, row_label(covered), false);
        }
    }
    if let Some(sprite) = widget::sprite_mut(world, MENU_BG) {
        sprite.x = bg[0];
        sprite.y = bg[1];
        sprite.width = bg[2];
        sprite.height = bg[3];
        sprite.tint = MENU_BG_TINT;
        sprite.corner_radius = theme::CONTROL_RADIUS;
        sprite.border_width = theme::PANEL_BORDER_WIDTH;
        sprite.border_color = theme::PANEL_BORDER_TINT;
        sprite.visible = true;
    }
    if m.menu_opens {
        menu_row(
            world,
            (MENU_OPEN_BG, MENU_OPEN_LABEL),
            open,
            view,
            "Open",
            theme::LABEL,
        );
    } else {
        widget::set_sprite_visible(world, MENU_OPEN_BG, false);
        widget::set_label_visible(world, MENU_OPEN_LABEL, false);
    }
    menu_row(
        world,
        (MENU_DELETE_BG, MENU_DELETE_LABEL),
        delete,
        view,
        "Delete",
        DELETE_LABEL,
    );
}

fn menu_row(
    world: &mut World,
    ids: (AssetId, AssetId),
    rect: [f32; 4],
    view: &WorldsView,
    caption: &str,
    color: [f32; 3],
) {
    let hovered = point_in(view.mouse[0], view.mouse[1], rect);
    place_rounded(
        world,
        ids.0,
        rect,
        if hovered {
            theme::HOVER_TINT
        } else {
            MENU_ROW_TINT
        },
        theme::CONTROL_RADIUS,
        true,
    );
    widget::place_left_label(
        world,
        ids.1,
        [rect[0] + 10.0, rect[1] + rect[3] * 0.5 - theme::TEXT_HALF],
        caption,
        color,
        true,
    );
}

fn hide_row_menu(world: &mut World) {
    for id in [MENU_BG, MENU_OPEN_BG, MENU_DELETE_BG] {
        widget::set_sprite_visible(world, id, false);
    }
    for id in [MENU_OPEN_LABEL, MENU_DELETE_LABEL] {
        widget::set_label_visible(world, id, false);
    }
}

fn scrollbar(world: &mut World, view: &WorldsView, o: [f32; 2]) {
    let Some(bar) = view.layout.scrollbar(o, view.rows.len(), view.scroll) else {
        widget::set_sprite_visible(world, LIST_TRACK, false);
        widget::set_sprite_visible(world, LIST_THUMB, false);
        return;
    };
    place_rounded(world, LIST_TRACK, bar.track, TRACK_TINT, bar.radius, true);
    place_rounded(world, LIST_THUMB, bar.thumb, THUMB_TINT, bar.radius, true);
}

// A button's centered caption.
fn chip_label(world: &mut World, id: AssetId, rect: [f32; 4], text: &str, m: &Metrics) {
    if let Some(l) = widget::label_mut(world, id) {
        l.x = rect[0] + rect[2] * 0.5;
        l.y = rect[1] + rect[3] * 0.5 - m.text_half();
        l.align = TextAlign::Center;
        l.color = theme::LABEL;
        l.scale = m.text_scale;
        l.visible = true;
        l.content = text.to_string();
    }
}

// A message bounded by the box that holds it, at the panel's text scale, so a
// long rejection wraps and clips inside the panel rather than running past it.
fn place_wrapped(
    world: &mut World,
    id: AssetId,
    rect: [f32; 4],
    content: &str,
    color: [f32; 3],
    visible: bool,
    m: &Metrics,
) {
    let lines = m.lines_in(rect[3]);
    if let Some(l) = widget::label_mut(world, id) {
        l.x = rect[0];
        l.y = rect[1];
        l.align = TextAlign::Left;
        l.color = color;
        l.scale = m.text_scale;
        l.visible = visible;
        l.wrap_width = rect[2].max(0.0);
        l.max_lines = lines;
        l.content = content.to_string();
    }
}

fn scale_label(world: &mut World, id: AssetId, scale: f32) {
    if let Some(l) = widget::label_mut(world, id) {
        l.scale = scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Sprite, TextLabel};
    use crate::editor::worlds::Layout;

    const VP: [f32; 2] = [1280.0, 720.0];
    // A macOS title bar's worth of chrome floating over the top of the frame.
    const INSET: f32 = 28.0;

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

    fn rows(n: usize) -> Vec<WorldRow> {
        (0..n)
            .map(|i| WorldRow {
                name: format!("world{i}"),
                path: format!("/p/worlds/world{i}.jsonl"),
                open: i == 0,
            })
            .collect()
    }

    fn layout(mode: Mode) -> Layout {
        Layout::new(mode, VP, 0.0)
    }

    fn view<'a>(l: Layout, rows: &'a [WorldRow], mouse: [f32; 2]) -> WorldsView<'a> {
        WorldsView {
            rows,
            scroll: 0,
            layout: l,
            selected: None,
            previewing: None,
            menu: None,
            status: None,
            mouse,
        }
    }

    fn sprite(world: &World, id: AssetId) -> Sprite {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == id)
            .cloned()
            .unwrap()
    }

    fn label(world: &World, id: AssetId) -> TextLabel {
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .cloned()
            .unwrap()
    }

    #[test]
    fn apply_labels_rows_and_marks_the_open_world() {
        let mut world = injected_world();
        let l = layout(Mode::Session);
        let o = l.default_origin();
        let rows = rows(2);
        apply(&mut world, Some(&view(l, &rows, [0.0, 0.0])), o);
        assert_eq!(label(&world, TITLE_LABEL).content, "Worlds");
        assert_eq!(label(&world, LIST_HEADER).content, "Worlds (2)");
        assert_eq!(label(&world, row_label(0)).content, "world0");
        assert_eq!(
            sprite(&world, row_bg(0)).tint,
            theme::SELECTED_TINT,
            "the open world is highlighted without a hover"
        );
        assert_eq!(sprite(&world, row_bg(1)).tint[3], 0.0);
        assert!(!sprite(&world, row_bg(2)).visible, "slots past the listing");
        // Nothing is hovered and no menu is up, so no row offers its dots.
        assert!(!sprite(&world, DOT1).visible);
        assert!(!sprite(&world, MENU_BG).visible);
    }

    // The `+` is a bare button: no fill at rest, and the same hover the rows
    // below it take.
    #[test]
    fn the_plus_carries_no_fill_until_it_is_hovered() {
        let mut world = injected_world();
        let l = layout(Mode::Start);
        let o = l.default_origin();
        let rows = rows(2);
        apply(&mut world, Some(&view(l, &rows, [0.0, 0.0])), o);
        assert_eq!(label(&world, NEW_LABEL).content, "+");
        assert_eq!(sprite(&world, NEW_BG).tint[3], 0.0, "no fill at rest");

        let n = l.new_rect(o);
        apply(
            &mut world,
            Some(&view(l, &rows, [n[0] + 2.0, n[1] + 2.0])),
            o,
        );
        assert_eq!(sprite(&world, NEW_BG).tint, theme::HOVER_TINT);
    }

    // The sidebar drops the heading entirely and names its listing in the short
    // form the narrow column has room for.
    #[test]
    fn the_sidebar_drops_the_heading_and_marks_its_selection() {
        let mut world = injected_world();
        let l = layout(Mode::Start);
        let o = l.default_origin();
        let rows = rows(3);
        let v = WorldsView {
            selected: Some(1),
            previewing: Some(1),
            ..view(l, &rows, [0.0, 0.0])
        };
        apply(&mut world, Some(&v), o);

        assert!(!label(&world, TITLE_LABEL).visible, "no heading at all");
        assert_eq!(label(&world, LIST_HEADER).content, "Worlds (3)");
        assert_eq!(sprite(&world, row_bg(1)).tint, theme::SELECTED_TINT);
        assert_eq!(
            sprite(&world, row_bg(0)).tint[3],
            0.0,
            "the world the session holds is not a start-screen selection"
        );
    }

    // The triple-dot belongs to the hovered row, and to the row whose menu is
    // open. No other row carries it, and the caption never moves to make room.
    #[test]
    fn the_triple_dot_follows_the_hovered_row() {
        let mut world = injected_world();
        let l = layout(Mode::Start);
        let o = l.default_origin();
        let rows = rows(3);
        let hovered = l.row_rect(o, 2);
        apply(
            &mut world,
            Some(&view(l, &rows, [hovered[0] + 4.0, hovered[1] + 4.0])),
            o,
        );
        let dots = sprite(&world, DOT1);
        assert!(dots.visible);
        let d = l.dot_rect(o, 2);
        assert!(dots.y > d[1] && dots.y < d[1] + d[3], "on the hovered row");
        assert!(
            !sprite(&world, DOT_BG).visible,
            "a row hover shows the bare dots"
        );
        let caption = label(&world, row_label(2)).x;

        // Off every row: nothing carries the dots.
        apply(&mut world, Some(&view(l, &rows, [0.0, 0.0])), o);
        assert!(!sprite(&world, DOT1).visible);
        assert_eq!(
            label(&world, row_label(2)).x,
            caption,
            "the caption never moves to make room"
        );

        // An open menu keeps the dots on its own row, boxed.
        let v = WorldsView {
            menu: Some(0),
            ..view(l, &rows, [0.0, 0.0])
        };
        apply(&mut world, Some(&v), o);
        assert!(sprite(&world, DOT_BG).visible);
        let d = l.dot_rect(o, 0);
        assert!(sprite(&world, DOT1).y > d[1]);
    }

    // The menu carries Delete as red text on the same clear backing every other
    // row has, never as a filled red button.
    #[test]
    fn the_menu_puts_delete_in_red_text_not_a_red_button() {
        let mut world = injected_world();
        let l = layout(Mode::Start);
        let o = l.default_origin();
        let rows = rows(3);
        let v = WorldsView {
            menu: Some(1),
            ..view(l, &rows, [0.0, 0.0])
        };
        apply(&mut world, Some(&v), o);

        assert!(sprite(&world, MENU_BG).visible);
        assert_eq!(label(&world, MENU_DELETE_LABEL).content, "Delete");
        assert_eq!(label(&world, MENU_DELETE_LABEL).color, DELETE_LABEL);
        assert!(
            label(&world, MENU_DELETE_LABEL).color[0] > label(&world, MENU_OPEN_LABEL).color[0],
            "Delete reads warmer than the plain rows"
        );
        assert_eq!(
            sprite(&world, MENU_DELETE_BG).tint[3],
            0.0,
            "no fill behind it"
        );
        assert_eq!(label(&world, MENU_OPEN_LABEL).content, "Open");
        assert_eq!(label(&world, MENU_OPEN_LABEL).color, theme::LABEL);

        // The captions the menu floats over are blanked: every label draws
        // after every sprite, so the backing alone cannot cover them.
        let (bg, ..) = l.menu_rects(o, 1);
        for slot in 0..rows.len() {
            let r = l.row_rect(o, slot);
            if slot != 1 && r[1] < bg[1] + bg[3] && r[1] + r[3] > bg[1] {
                assert!(
                    !label(&world, row_label(slot)).visible,
                    "slot {slot} sits under the menu"
                );
            }
        }
        assert!(
            label(&world, row_label(1)).visible,
            "the menu's own row still reads"
        );

        // Hovering a menu row takes the shared highlight, red text included.
        let (_, _, delete) = l.menu_rects(o, 1);
        let v = WorldsView {
            menu: Some(1),
            ..view(l, &rows, [delete[0] + 4.0, delete[1] + 4.0])
        };
        apply(&mut world, Some(&v), o);
        assert_eq!(sprite(&world, MENU_DELETE_BG).tint, theme::HOVER_TINT);
        assert_eq!(label(&world, MENU_DELETE_LABEL).color, DELETE_LABEL);
    }

    // The switcher's menu offers Delete alone: a row click already opens there.
    #[test]
    fn the_switchers_menu_hides_its_open_row() {
        let mut world = injected_world();
        let l = layout(Mode::Session);
        let o = l.default_origin();
        let rows = rows(3);
        let v = WorldsView {
            menu: Some(1),
            ..view(l, &rows, [0.0, 0.0])
        };
        apply(&mut world, Some(&v), o);
        assert!(sprite(&world, MENU_DELETE_BG).visible);
        assert!(!sprite(&world, MENU_OPEN_BG).visible);
        assert!(!label(&world, MENU_OPEN_LABEL).visible);
    }

    // The `+` and the first row are drawn below whatever chrome floats over the
    // top of the frame, while the sidebar's own background still runs to the
    // very top of the window.
    #[test]
    fn a_top_inset_pushes_the_sidebar_content_below_the_window_chrome() {
        let flush = Layout::new(Mode::Start, VP, 0.0);
        let inset = Layout::new(Mode::Start, VP, INSET);
        let rows = rows(2);
        let mut a = injected_world();
        apply(&mut a, Some(&view(flush, &rows, [0.0, 0.0])), [0.0, 0.0]);
        let mut b = injected_world();
        apply(&mut b, Some(&view(inset, &rows, [0.0, 0.0])), [0.0, 0.0]);

        assert_eq!(sprite(&b, NEW_BG).y - sprite(&a, NEW_BG).y, INSET);
        assert_eq!(sprite(&b, row_bg(0)).y - sprite(&a, row_bg(0)).y, INSET);
        assert!(sprite(&b, NEW_BG).y > INSET, "clear of the window buttons");
        // The background is unmoved: it fills under the chrome on purpose.
        assert_eq!(sprite(&b, PANEL_BG).y, sprite(&a, PANEL_BG).y);
        assert!(sprite(&b, PANEL_BG).y <= 0.0);
    }

    // The sidebar stands over the world it previews, so its surface is a wash
    // the render reads through, while the floating switcher keeps the opaque
    // chrome every other panel sits on.
    #[test]
    fn the_sidebar_is_a_wash_and_the_switcher_is_not() {
        let rows = rows(3);
        let mut sidebar = injected_world();
        let l = layout(Mode::Start);
        apply(
            &mut sidebar,
            Some(&view(l, &rows, [0.0, 0.0])),
            l.default_origin(),
        );
        let bg = sprite(&sidebar, PANEL_BG);
        let wash = bg.tint;
        assert_eq!(wash, theme::SIDEBAR_TINT);
        assert!(
            (0.3..0.75).contains(&wash[3]),
            "seen through, but still a surface: {wash:?}"
        );
        // The frame is part of that surface, so it is let down with it rather
        // than cutting a solid hairline into the world behind.
        assert_eq!(bg.border_color, theme::panel_border(wash[3]));
        assert_eq!(bg.border_color[3], wash[3]);

        let mut switcher = injected_world();
        let l = layout(Mode::Session);
        apply(
            &mut switcher,
            Some(&view(l, &rows, [0.0, 0.0])),
            l.default_origin(),
        );
        let chrome = sprite(&switcher, PANEL_BG);
        assert_eq!(chrome.tint, theme::CHROME_TINT);
        assert!(
            chrome.border_color[3] > sprite(&sidebar, PANEL_BG).border_color[3],
            "and a chrome panel keeps its solid frame"
        );
    }

    // The picked row is the one the preview behind the screen is showing, so it
    // reads solid against the wash the rest of the column is drawn on.
    #[test]
    fn the_picked_row_reads_solid_against_the_wash() {
        let mut world = injected_world();
        let l = layout(Mode::Start);
        let o = l.default_origin();
        let rows = rows(3);
        let v = WorldsView {
            selected: Some(1),
            previewing: Some(1),
            ..view(l, &rows, [0.0, 0.0])
        };
        apply(&mut world, Some(&v), o);

        let picked = sprite(&world, row_bg(1)).tint;
        assert_eq!(picked, theme::SELECTED_TINT);
        assert_eq!(picked[3], 1.0, "the world on screen is opaque");
        assert!(
            picked[3] > sprite(&world, PANEL_BG).tint[3],
            "and stands out of the column it sits in"
        );
        // Every other row leaves the wash showing.
        assert_eq!(sprite(&world, row_bg(0)).tint[3], 0.0);
        assert_eq!(sprite(&world, row_bg(2)).tint[3], 0.0);
    }

    // The start screen has no title bar and so no close button (closing it
    // would strand the session), and its captions are larger than the
    // switcher's.
    #[test]
    fn the_start_screen_drops_the_close_button_and_scales_its_text_up() {
        let mut world = injected_world();
        let rows = rows(2);
        let start = layout(Mode::Start);
        apply(
            &mut world,
            Some(&view(start, &rows, [0.0, 0.0])),
            start.default_origin(),
        );
        assert!(!sprite(&world, CLOSE_BG).visible);
        assert!(!label(&world, CLOSE_LABEL).visible);
        let big = label(&world, row_label(0)).scale;

        // Switching back to the switcher restores the close button and the
        // smaller text, so no element keeps the start screen's presentation.
        let session = layout(Mode::Session);
        apply(
            &mut world,
            Some(&view(session, &rows, [0.0, 0.0])),
            session.default_origin(),
        );
        assert!(label(&world, CLOSE_LABEL).visible);
        assert!(label(&world, TITLE_LABEL).visible);
        let small = label(&world, row_label(0)).scale;
        assert!(big > small, "{big} is larger than {small}");
        assert_eq!(small, theme::TEXT_SCALE);
    }

    // A row slot the presentation's shorter window does not offer stays blank,
    // even when the listing has a world for it.
    #[test]
    fn slots_past_the_visible_window_stay_blank() {
        let mut world = injected_world();
        let l = layout(Mode::Session);
        let o = l.default_origin();
        let rows = rows(POOL);
        apply(&mut world, Some(&view(l, &rows, [0.0, 0.0])), o);
        let shown = l.rows();
        assert!(shown < POOL, "the switcher shows fewer rows than the pool");
        assert!(sprite(&world, row_bg(shown - 1)).visible);
        assert!(!sprite(&world, row_bg(shown)).visible);
    }

    #[test]
    fn an_empty_listing_says_so_and_a_failure_shows_on_the_status_line() {
        let mut world = injected_world();
        let l = layout(Mode::Session);
        let o = l.default_origin();
        apply(&mut world, Some(&view(l, &[], [0.0, 0.0])), o);
        assert!(label(&world, LIST_HEADER).content.contains("No worlds"));
        assert!(!label(&world, STATUS_LABEL).visible);

        let v = WorldsView {
            status: Some("Open failed: 'arena' is not valid JSON"),
            ..view(l, &[], [0.0, 0.0])
        };
        apply(&mut world, Some(&v), o);
        let status = label(&world, STATUS_LABEL);
        assert!(status.visible && status.content.contains("arena"));
        assert!(status.wrap_width > 0.0 && status.max_lines >= 1);
    }

    #[test]
    fn the_scrollbar_shows_only_when_the_listing_overflows() {
        let mut world = injected_world();
        let l = layout(Mode::Session);
        let o = l.default_origin();
        let short = rows(l.rows());
        apply(&mut world, Some(&view(l, &short, [0.0, 0.0])), o);
        assert!(!sprite(&world, LIST_TRACK).visible);

        let long = rows(l.rows() + 4);
        apply(&mut world, Some(&view(l, &long, [0.0, 0.0])), o);
        assert!(sprite(&world, LIST_TRACK).visible);
        assert!(sprite(&world, LIST_THUMB).visible);
    }

    #[test]
    fn hovering_highlights_the_row_under_the_cursor() {
        let mut world = injected_world();
        let l = layout(Mode::Session);
        let o = l.default_origin();
        let rows = rows(2);
        let r = l.row_rect(o, 1);
        apply(
            &mut world,
            Some(&view(l, &rows, [r[0] + 2.0, r[1] + 2.0])),
            o,
        );
        assert_eq!(sprite(&world, row_bg(1)).tint, theme::HOVER_TINT);
        assert_eq!(sprite(&world, row_bg(0)).tint, theme::SELECTED_TINT);
    }

    #[test]
    fn hide_blanks_every_element() {
        let mut world = injected_world();
        let rows = rows(3);
        let l = layout(Mode::Start);
        let v = WorldsView {
            selected: Some(0),
            menu: Some(0),
            ..view(l, &rows, [0.0, 0.0])
        };
        apply(&mut world, Some(&v), l.default_origin());
        apply(&mut world, None, [0.0, 0.0]);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}

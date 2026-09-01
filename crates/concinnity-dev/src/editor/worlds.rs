// src/editor/worlds.rs
//
// The Worlds panel: the project's worlds, most recently edited first, with a
// name field that creates a new one and a per-row delete chip. Clicking a row
// opens that world in the running session. It is the first thing `cn editor`
// shows when no world was named on the command line, and stays a registered
// panel afterwards, so it doubles as the switcher for a session already
// running. Layout half only; `hook/worlds_edit.rs` owns the actions.

use super::registry::{self, PanelKey};
use super::theme;
use super::widget::{self, place_rounded, point_in};
use crate::components::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::Worlds);
pub(crate) const PANEL_BG: AssetId = AssetId(BASE);
pub(crate) const TITLE_LABEL: AssetId = AssetId(BASE + 1);
pub(crate) const CLOSE_BG: AssetId = AssetId(BASE + 2);
pub(crate) const CLOSE_LABEL: AssetId = AssetId(BASE + 3);
pub(crate) const NEW_BG: AssetId = AssetId(BASE + 4);
pub(crate) const NEW_LABEL: AssetId = AssetId(BASE + 5);
pub(crate) const STATUS_LABEL: AssetId = AssetId(BASE + 6);
pub(crate) const LIST_HEADER: AssetId = AssetId(BASE + 7);
pub(crate) const LIST_TRACK: AssetId = AssetId(BASE + 8);
pub(crate) const LIST_THUMB: AssetId = AssetId(BASE + 9);
pub(crate) const NAME_INPUT: AssetId = AssetId(BASE + 0x10);

pub(crate) fn row_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x20 + i as u32)
}
pub(crate) fn row_label(i: usize) -> AssetId {
    AssetId(BASE + 0x40 + i as u32)
}
pub(crate) fn del_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x60 + i as u32)
}
pub(crate) fn del_label(i: usize) -> AssetId {
    AssetId(BASE + 0x80 + i as u32)
}

// Geometry, in window pixels. Every rect derives from the panel origin `o`, so
// dragging the title bar moves the whole panel.
pub(crate) const WORLDS_W: f32 = 340.0;
const PAD: f32 = 10.0;
const HEADER_H: f32 = 32.0;
const STATUS_H: f32 = widget::LINE_H + 6.0;
const LIST_HEADER_H: f32 = 24.0;
const ROW_H: f32 = 26.0;
const NEW_W: f32 = 70.0;
const DEL_W: f32 = 44.0;
const GAP: f32 = 6.0;
const SCROLLBAR_W: f32 = 5.0;
// Visible rows; a longer listing scrolls.
pub(crate) const POOL: usize = 12;
// Row caption budget, leaving the delete chip its strip.
const MAX_ROW_CHARS: usize = 32;

const CHROME_H: f32 = widget::TITLE_H + HEADER_H + STATUS_H + LIST_HEADER_H;

const ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const NEW_TINT: [f32; 4] = [0.20, 0.44, 0.30, 1.0];
const NEW_TINT_HOVER: [f32; 4] = [0.28, 0.56, 0.38, 1.0];
// The destructive chip, matching the Variables panel's Del.
const DEL_TINT: [f32; 4] = [0.44, 0.22, 0.24, 1.0];
const DEL_TINT_HOVER: [f32; 4] = [0.60, 0.28, 0.30, 1.0];
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
const HEADER_LABEL: [f32; 3] = [0.70, 0.74, 0.84];
const STATUS_LABEL_COLOR: [f32; 3] = [0.95, 0.55, 0.55];

// One listed world, as the panel draws it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorldRow {
    pub name: String,
    pub path: String,
    // Whether this is the world the session currently has open.
    pub open: bool,
}

// What switching to another world does once the unsaved-changes guard clears.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WorldTarget {
    // Open the world file at this path.
    Open(String),
    // Create a world under this (already validated) name, then open it.
    Create(String),
}

// A Worlds-panel decision the confirmation dialog is holding: the dialog's
// button hands one of these back when it is pressed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WorldsConfirm {
    // Delete the world file at this path.
    Delete(String),
    // Save the open world, then go to the target.
    Save(WorldTarget),
    // Go to the target, dropping the open world's unsaved edits.
    Discard(WorldTarget),
}

// The per-frame view the hook assembles.
pub(crate) struct WorldsView<'a> {
    pub rows: &'a [WorldRow],
    pub scroll: usize,
    // Whether the name field asserts keyboard focus this frame.
    pub focus: bool,
    // Why the last New was rejected, if it was.
    pub status: Option<&'a str>,
    pub mouse: [f32; 2],
}

// A resolved Worlds-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorldsAction {
    // Give the name field keyboard focus.
    FocusName,
    // Create a world under the typed name and open it.
    New,
    // Open listed world `i` (an index into the view's rows).
    Open(usize),
    // Delete listed world `i`, behind the confirmation dialog.
    Delete(usize),
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Where the panel sits until the user drags it: centred horizontally below the
// top bar, since a session with no named world opens on it.
pub(crate) fn default_origin(vp: [f32; 2]) -> [f32; 2] {
    [(vp[0] - WORLDS_W) * 0.5, super::hud::body_top() + 24.0]
}

// The panel's fixed footprint.
pub(crate) fn size() -> [f32; 2] {
    [WORLDS_W, CHROME_H + POOL as f32 * ROW_H + PAD]
}

// The header's controls: New pinned to the right end, the name field filling
// the rest.
const CTRL_H: f32 = HEADER_H - 8.0;

pub(crate) fn new_rect(o: [f32; 2]) -> [f32; 4] {
    [
        o[0] + WORLDS_W - PAD - NEW_W,
        widget::header_y(o, 4.0),
        NEW_W,
        CTRL_H,
    ]
}

pub(crate) fn name_rect(o: [f32; 2]) -> [f32; 4] {
    let left = o[0] + PAD;
    [
        left,
        widget::header_y(o, 4.0),
        (new_rect(o)[0] - GAP - left).max(0.0),
        CTRL_H,
    ]
}

fn list_top(o: [f32; 2]) -> f32 {
    o[1] + CHROME_H
}

// Visible row `slot` (0-based within the window).
pub(crate) fn row_rect(o: [f32; 2], slot: usize) -> [f32; 4] {
    [
        o[0],
        list_top(o) + slot as f32 * ROW_H,
        WORLDS_W - SCROLLBAR_W - 2.0,
        ROW_H,
    ]
}

// The delete chip at row `slot`'s right end.
pub(crate) fn del_rect(o: [f32; 2], slot: usize) -> [f32; 4] {
    let r = row_rect(o, slot);
    [
        r[0] + r[2] - PAD - DEL_W,
        r[1] + (ROW_H - CTRL_H) * 0.5,
        DEL_W,
        CTRL_H,
    ]
}

// Whether the cursor is over the scrollable listing (for wheel routing).
pub(crate) fn cursor_over_list(mx: f32, my: f32, o: [f32; 2]) -> bool {
    let p = widget::outer_rect(o, size());
    mx >= p[0] && mx < p[0] + p[2] && my >= list_top(o) && my < p[1] + p[3]
}

// Resolve a click at `(mx, my)` against the panel at origin `o`. `None` means
// the click missed the panel entirely and belongs to whatever is behind it.
// Title-bar presses never reach this: the shared routing intercepts them first.
pub(crate) fn hit_test(view: &WorldsView, mx: f32, my: f32, o: [f32; 2]) -> Option<WorldsAction> {
    if point_in(mx, my, new_rect(o)) {
        return Some(WorldsAction::New);
    }
    if point_in(mx, my, name_rect(o)) {
        return Some(WorldsAction::FocusName);
    }
    for slot in 0..POOL {
        if !point_in(mx, my, row_rect(o, slot)) {
            continue;
        }
        let i = view.scroll + slot;
        if i >= view.rows.len() {
            return Some(WorldsAction::Consume);
        }
        // The chip is checked before the rest of the row so deleting never
        // opens the world it is removing.
        return Some(if point_in(mx, my, del_rect(o, slot)) {
            WorldsAction::Delete(i)
        } else {
            WorldsAction::Open(i)
        });
    }
    point_in(mx, my, widget::outer_rect(o, size())).then_some(WorldsAction::Consume)
}

// Position + show the panel (`Some(view)`) at origin `o`, or blank every
// element (`None`).
pub(crate) fn apply(world: &mut World, view: Option<&WorldsView>, o: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };
    widget::place_panel(world, PANEL_BG, widget::outer_rect(o, size()));
    let title = widget::title_rect(o, WORLDS_W);
    widget::place_heading(world, TITLE_LABEL, title, "Worlds");
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);

    widget::show_field(world, NAME_INPUT, name_rect(o), view.focus);
    let new = new_rect(o);
    let hover = point_in(view.mouse[0], view.mouse[1], new);
    let tint = if hover { NEW_TINT_HOVER } else { NEW_TINT };
    place_rounded(world, NEW_BG, new, tint, theme::CONTROL_RADIUS, true);
    chip_label(world, NEW_LABEL, new, "New", theme::LABEL);

    widget::place_message(
        world,
        STATUS_LABEL,
        [
            o[0] + PAD,
            o[1] + widget::TITLE_H + HEADER_H,
            (WORLDS_W - 2.0 * PAD).max(0.0),
            STATUS_H - 4.0,
        ],
        view.status.unwrap_or(""),
        STATUS_LABEL_COLOR,
        view.status.is_some(),
    );
    if let Some(l) = widget::label_mut(world, LIST_HEADER) {
        l.x = o[0] + PAD;
        l.y = o[1] + widget::TITLE_H + HEADER_H + STATUS_H + 2.0;
        l.align = TextAlign::Left;
        l.color = HEADER_LABEL;
        l.visible = true;
        l.content = if view.rows.is_empty() {
            "No worlds yet - name one and press New".to_string()
        } else {
            format!("Worlds ({})", view.rows.len())
        };
    }

    for slot in 0..POOL {
        let i = view.scroll + slot;
        let Some(row) = view.rows.get(i) else {
            widget::set_sprite_visible(world, row_bg(slot), false);
            widget::set_label_visible(world, row_label(slot), false);
            widget::set_sprite_visible(world, del_bg(slot), false);
            widget::set_label_visible(world, del_label(slot), false);
            continue;
        };
        let r = row_rect(o, slot);
        let del = del_rect(o, slot);
        let over_del = point_in(view.mouse[0], view.mouse[1], del);
        let tint = if point_in(view.mouse[0], view.mouse[1], r) {
            theme::HOVER_TINT
        } else if row.open {
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
        if let Some(l) = widget::label_mut(world, row_label(slot)) {
            l.x = r[0] + PAD;
            l.y = r[1] + ROW_H * 0.5 - theme::TEXT_HALF;
            l.align = TextAlign::Left;
            l.color = theme::LABEL;
            l.visible = true;
            l.content = widget::clip_text(&row.name, MAX_ROW_CHARS);
        }
        place_rounded(
            world,
            del_bg(slot),
            del,
            if over_del { DEL_TINT_HOVER } else { DEL_TINT },
            theme::CONTROL_RADIUS,
            true,
        );
        chip_label(world, del_label(slot), del, "Del", theme::LABEL);
    }

    layout_scrollbar(world, view, o);
}

// A button's centered caption.
fn chip_label(world: &mut World, id: AssetId, rect: [f32; 4], text: &str, color: [f32; 3]) {
    if let Some(l) = widget::label_mut(world, id) {
        l.x = rect[0] + rect[2] * 0.5;
        l.y = rect[1] + rect[3] * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Center;
        l.color = color;
        l.visible = true;
        l.content = text.to_string();
    }
}

fn layout_scrollbar(world: &mut World, view: &WorldsView, o: [f32; 2]) {
    let total = view.rows.len();
    if total <= POOL {
        widget::set_sprite_visible(world, LIST_TRACK, false);
        widget::set_sprite_visible(world, LIST_THUMB, false);
        return;
    }
    let x = o[0] + WORLDS_W - SCROLLBAR_W;
    let top = list_top(o);
    let h = POOL as f32 * ROW_H;
    place_rounded(
        world,
        LIST_TRACK,
        [x, top, SCROLLBAR_W, h],
        TRACK_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
    let thumb_h = (h * POOL as f32 / total as f32).max(18.0);
    let max_scroll = (total - POOL) as f32;
    let off = (h - thumb_h) * (view.scroll as f32 / max_scroll);
    place_rounded(
        world,
        LIST_THUMB,
        [x, top + off, SCROLLBAR_W, thumb_h],
        THUMB_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
}

// Hide every panel element, blurring the name field so a hidden field cannot
// keep keyboard focus.
pub(crate) fn hide_all(world: &mut World) {
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &all_field_ids());
}

// Every panel sprite id, in draw (insertion) order: chrome, then the rows, the
// delete chips over them, and the scrollbar above both.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG, NEW_BG];
    ids.extend((0..POOL).map(row_bg));
    ids.extend((0..POOL).map(del_bg));
    ids.extend([LIST_TRACK, LIST_THUMB]);
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![
        TITLE_LABEL,
        CLOSE_LABEL,
        NEW_LABEL,
        STATUS_LABEL,
        LIST_HEADER,
    ];
    ids.extend((0..POOL).map(row_label));
    ids.extend((0..POOL).map(del_label));
    ids
}

pub(crate) fn all_field_ids() -> Vec<AssetId> {
    vec![NAME_INPUT]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Sprite, TextInput, TextLabel};

    const VP: [f32; 2] = [1280.0, 720.0];

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

    fn rows(n: usize) -> Vec<WorldRow> {
        (0..n)
            .map(|i| WorldRow {
                name: format!("world{i}"),
                path: format!("/p/worlds/world{i}.jsonl"),
                open: i == 0,
            })
            .collect()
    }

    fn view<'a>(rows: &'a [WorldRow], mouse: [f32; 2]) -> WorldsView<'a> {
        WorldsView {
            rows,
            scroll: 0,
            focus: false,
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
    fn hit_test_resolves_the_header_controls() {
        let o = default_origin(VP);
        let rows = rows(2);
        let v = view(&rows, [0.0, 0.0]);
        let n = new_rect(o);
        assert_eq!(
            hit_test(&v, n[0] + 2.0, n[1] + 2.0, o),
            Some(WorldsAction::New)
        );
        let f = name_rect(o);
        assert_eq!(
            hit_test(&v, f[0] + 2.0, f[1] + 2.0, o),
            Some(WorldsAction::FocusName)
        );
    }

    // The chip wins over the row it sits in, so a delete never opens instead.
    #[test]
    fn a_row_opens_and_its_chip_deletes() {
        let o = default_origin(VP);
        let rows = rows(3);
        let v = view(&rows, [0.0, 0.0]);
        let r = row_rect(o, 1);
        assert_eq!(
            hit_test(&v, r[0] + 4.0, r[1] + 4.0, o),
            Some(WorldsAction::Open(1))
        );
        let d = del_rect(o, 1);
        assert_eq!(
            hit_test(&v, d[0] + 2.0, d[1] + 2.0, o),
            Some(WorldsAction::Delete(1))
        );
        assert!(
            d[0] > r[0] && d[0] + d[2] <= r[0] + r[2],
            "the chip sits inside its row"
        );
    }

    // A scrolled listing resolves to the world under the cursor, not the slot.
    #[test]
    fn a_scrolled_row_resolves_to_its_world() {
        let o = default_origin(VP);
        let rows = rows(POOL + 5);
        let v = WorldsView {
            scroll: 3,
            ..view(&rows, [0.0, 0.0])
        };
        let r = row_rect(o, 0);
        assert_eq!(
            hit_test(&v, r[0] + 4.0, r[1] + 4.0, o),
            Some(WorldsAction::Open(3))
        );
    }

    // Every press inside the panel is swallowed, and every press outside misses
    // entirely, so the panels behind stay reachable.
    #[test]
    fn presses_are_rect_guarded() {
        let o = default_origin(VP);
        let rows = rows(1);
        let v = view(&rows, [0.0, 0.0]);
        // An empty row slot, the title bar, and the padding below the rows.
        let empty = row_rect(o, 5);
        assert_eq!(
            hit_test(&v, empty[0] + 4.0, empty[1] + 4.0, o),
            Some(WorldsAction::Consume)
        );
        let t = widget::title_rect(o, WORLDS_W);
        assert_eq!(
            hit_test(&v, t[0] + 4.0, t[1] + 4.0, o),
            Some(WorldsAction::Consume)
        );
        for (x, y) in [
            (o[0] - 2.0, o[1] + 20.0),
            (o[0] + WORLDS_W + 2.0, o[1] + 20.0),
            (o[0] + 20.0, o[1] - 2.0),
            (o[0] + 20.0, o[1] + size()[1] + 2.0),
        ] {
            assert_eq!(hit_test(&v, x, y, o), None, "({x}, {y}) is off the panel");
        }
    }

    #[test]
    fn apply_labels_rows_and_marks_the_open_world() {
        let mut world = injected_world();
        let o = default_origin(VP);
        let rows = rows(2);
        apply(&mut world, Some(&view(&rows, [0.0, 0.0])), o);
        assert_eq!(label(&world, TITLE_LABEL).content, "Worlds");
        assert_eq!(label(&world, LIST_HEADER).content, "Worlds (2)");
        assert_eq!(label(&world, row_label(0)).content, "world0");
        assert_eq!(label(&world, del_label(0)).content, "Del");
        assert_eq!(
            sprite(&world, row_bg(0)).tint,
            theme::SELECTED_TINT,
            "the open world is highlighted without a hover"
        );
        assert_eq!(sprite(&world, row_bg(1)).tint[3], 0.0);
        // Slots past the listing draw nothing.
        assert!(!sprite(&world, row_bg(2)).visible);
        assert!(!sprite(&world, del_bg(2)).visible);
    }

    #[test]
    fn an_empty_listing_says_so_and_a_rejection_shows_on_the_status_line() {
        let mut world = injected_world();
        let o = default_origin(VP);
        apply(&mut world, Some(&view(&[], [0.0, 0.0])), o);
        assert!(label(&world, LIST_HEADER).content.contains("No worlds"));
        assert!(!label(&world, STATUS_LABEL).visible);

        let v = WorldsView {
            status: Some("A world named 'arena' already exists"),
            ..view(&[], [0.0, 0.0])
        };
        apply(&mut world, Some(&v), o);
        let status = label(&world, STATUS_LABEL);
        assert!(status.visible && status.content.contains("arena"));
    }

    #[test]
    fn the_scrollbar_shows_only_when_the_listing_overflows() {
        let mut world = injected_world();
        let o = default_origin(VP);
        let short = rows(POOL);
        apply(&mut world, Some(&view(&short, [0.0, 0.0])), o);
        assert!(!sprite(&world, LIST_TRACK).visible);

        let long = rows(POOL + 4);
        apply(&mut world, Some(&view(&long, [0.0, 0.0])), o);
        assert!(sprite(&world, LIST_TRACK).visible);
        assert!(sprite(&world, LIST_THUMB).visible);
    }

    #[test]
    fn hovering_highlights_the_row_and_its_chip() {
        let mut world = injected_world();
        let o = default_origin(VP);
        let rows = rows(2);
        let d = del_rect(o, 1);
        apply(&mut world, Some(&view(&rows, [d[0] + 2.0, d[1] + 2.0])), o);
        assert_eq!(sprite(&world, row_bg(1)).tint, theme::HOVER_TINT);
        assert_eq!(sprite(&world, del_bg(1)).tint, DEL_TINT_HOVER);
        assert_eq!(sprite(&world, del_bg(0)).tint, DEL_TINT);
    }

    #[test]
    fn hide_blanks_every_element_and_blurs_the_field() {
        let mut world = injected_world();
        let rows = rows(3);
        apply(
            &mut world,
            Some(&view(&rows, [0.0, 0.0])),
            default_origin(VP),
        );
        apply(&mut world, None, default_origin(VP));
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
        assert!(world.query::<TextInput>().all(|t| !t.visible && !t.focused));
    }

    #[test]
    fn the_panel_anchors_below_the_top_bar_and_the_wheel_region_covers_the_list() {
        let o = default_origin(VP);
        assert!(o[1] >= super::super::hud::body_top());
        let r = row_rect(o, 0);
        assert!(cursor_over_list(r[0] + 4.0, r[1] + 4.0, o));
        let t = widget::title_rect(o, WORLDS_W);
        assert!(!cursor_over_list(t[0] + 4.0, t[1] + 4.0, o));
    }

    #[test]
    fn id_lists_cover_every_slot_without_repeats() {
        let mut all: Vec<AssetId> = all_sprite_ids()
            .into_iter()
            .chain(all_label_ids())
            .chain(all_field_ids())
            .collect();
        let n = all.len();
        all.sort_by_key(|id| id.0);
        all.dedup();
        assert_eq!(all.len(), n, "no duplicate reserved ids");
    }
}

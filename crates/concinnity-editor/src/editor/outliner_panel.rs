// src/editor/outliner_panel.rs
//
// The Outliner panel: the expanded world as a collapsible tree (built by
// `outliner.rs`), a search field over it, a provenance badge per asset, and
// the per-row editor-session hide / lock toggles. Selection is two-way: rows
// reflect the viewport selection set and clicking one drives it. Layout half
// only; `hook/outliner_edit.rs` owns the actions and the session state.

use super::outliner::{Badge, OutlinerRow};
use super::registry::{self, PanelKey};
use super::selection::Selection;
use super::theme;
use super::widget::{self, place_rounded, point_in};
use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;
use std::collections::BTreeSet;

const BASE: u32 = registry::base(PanelKey::Outliner);
pub(crate) const PANEL_BG: AssetId = AssetId(BASE);
pub(crate) const TITLE_LABEL: AssetId = AssetId(BASE + 1);
pub(crate) const CLOSE_BG: AssetId = AssetId(BASE + 2);
pub(crate) const CLOSE_LABEL: AssetId = AssetId(BASE + 3);
pub(crate) const STATUS_LABEL: AssetId = AssetId(BASE + 4);
pub(crate) const LIST_TRACK: AssetId = AssetId(BASE + 5);
pub(crate) const LIST_THUMB: AssetId = AssetId(BASE + 6);
pub(crate) const FILTER_INPUT: AssetId = AssetId(BASE + 7);

pub(crate) fn row_bg(slot: usize) -> AssetId {
    AssetId(BASE + 0x10 + slot as u32)
}
pub(crate) fn name_label(slot: usize) -> AssetId {
    AssetId(BASE + 0x20 + slot as u32)
}
pub(crate) fn badge_label(slot: usize) -> AssetId {
    AssetId(BASE + 0x30 + slot as u32)
}
pub(crate) fn eye_box(slot: usize) -> AssetId {
    AssetId(BASE + 0x40 + slot as u32)
}
pub(crate) fn lock_box(slot: usize) -> AssetId {
    AssetId(BASE + 0x50 + slot as u32)
}

// Geometry, in window pixels. Every rect derives from the panel origin `o`.
pub(crate) const OUTLINER_W: f32 = 460.0;
const PAD: f32 = 10.0;
const HEADER_H: f32 = 32.0;
const STATUS_H: f32 = 20.0;
const ROW_H: f32 = 26.0;
const SCROLLBAR_W: f32 = 5.0;
const BOX_SIZE: f32 = 14.0;
const GAP: f32 = 6.0;
// An asset row's name is inset under its group header.
const ASSET_INDENT: f32 = 24.0;
// Visible rows; a longer tree scrolls.
pub(crate) const ROW_POOL: usize = 14;
const MAX_NAME_CHARS: usize = 30;

const ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
// The eye box: green while the asset renders, dim once hidden.
const EYE_TINT_ON: [f32; 4] = [0.30, 0.66, 0.34, 1.0];
const EYE_TINT_OFF: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
// The lock box: amber while locked, dim while pickable.
const LOCK_TINT_ON: [f32; 4] = [0.78, 0.56, 0.22, 1.0];
const LOCK_TINT_OFF: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];

// Per-badge label colours, so provenance reads at a glance.
fn badge_color(badge: Badge) -> [f32; 3] {
    match badge {
        Badge::Authored => theme::LABEL_DIM,
        Badge::Imported => [0.45, 0.72, 0.62],
        Badge::Injected => [0.70, 0.58, 0.88],
    }
}

// The per-frame view the hook assembles: the flattened tree, the scroll
// window, the filter field's focus, the selection set the rows mirror, the
// session hide / lock sets, and the cook status (an error replaces the count).
pub(crate) struct OutlinerView<'a> {
    pub rows: &'a [OutlinerRow],
    pub scroll: usize,
    pub focus: bool,
    pub selection: &'a Selection,
    pub hidden: &'a BTreeSet<String>,
    pub locked: &'a BTreeSet<String>,
    pub status: Option<&'a str>,
    pub total: usize,
    pub mouse: [f32; 2],
}

// A resolved Outliner click.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum OutlinerAction {
    // Give the search field keyboard focus.
    FocusFilter,
    // Fold / unfold a group (an index into the grouped model).
    ToggleGroup(usize),
    // Select the named asset (the hook applies plain / shift semantics).
    Select(String),
    // Flip the named asset's editor-session hide.
    ToggleHide(String),
    // Flip the named asset's pick lock.
    ToggleLock(String),
    // A click elsewhere on the panel: swallowed, and the filter blurs.
    Consume,
}

// Where the panel sits until the user drags it: the left edge, below the View
// panel's default anchor so the launch layout does not overlap.
pub(crate) fn default_origin(_vp: [f32; 2]) -> [f32; 2] {
    let view = super::view::default_origin();
    [8.0, view[1] + super::view::size()[1] + 8.0]
}

// The panel's fixed footprint.
pub(crate) fn size() -> [f32; 2] {
    [
        OUTLINER_W,
        widget::TITLE_H + HEADER_H + STATUS_H + ROW_POOL as f32 * ROW_H + PAD,
    ]
}

fn panel_rect(o: [f32; 2]) -> [f32; 4] {
    let s = size();
    [o[0], o[1], s[0], s[1]]
}

fn title_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0], o[1], OUTLINER_W, widget::TITLE_H]
}

// The search field fills the header row.
pub(crate) fn filter_rect(o: [f32; 2]) -> [f32; 4] {
    [
        o[0] + PAD,
        o[1] + widget::TITLE_H + 4.0,
        OUTLINER_W - 2.0 * PAD,
        HEADER_H - 8.0,
    ]
}

fn list_top(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H + HEADER_H + STATUS_H
}

// Visible row `slot` (0-based within the scroll window).
pub(crate) fn row_rect(o: [f32; 2], slot: usize) -> [f32; 4] {
    [
        o[0],
        list_top(o) + slot as f32 * ROW_H,
        OUTLINER_W - SCROLLBAR_W - 2.0,
        ROW_H,
    ]
}

// The hide toggle at an asset row's right end.
pub(crate) fn eye_rect(o: [f32; 2], slot: usize) -> [f32; 4] {
    let r = row_rect(o, slot);
    [
        lock_rect(o, slot)[0] - GAP - BOX_SIZE,
        r[1] + (ROW_H - BOX_SIZE) * 0.5,
        BOX_SIZE,
        BOX_SIZE,
    ]
}

// The pick-lock toggle, outermost.
pub(crate) fn lock_rect(o: [f32; 2], slot: usize) -> [f32; 4] {
    let r = row_rect(o, slot);
    [
        r[0] + r[2] - PAD - BOX_SIZE,
        r[1] + (ROW_H - BOX_SIZE) * 0.5,
        BOX_SIZE,
        BOX_SIZE,
    ]
}

// Whether the cursor is over the scrollable tree (for wheel routing).
pub(crate) fn cursor_over_list(mx: f32, my: f32, o: [f32; 2]) -> bool {
    let p = panel_rect(o);
    mx >= p[0] && mx < p[0] + p[2] && my >= list_top(o) && my < p[1] + p[3]
}

// Resolve a click at `(mx, my)` against the panel at origin `o`. `None` means
// the click missed the panel. Title-bar presses never reach this: the shared
// routing intercepts them first.
pub(crate) fn hit_test(
    view: &OutlinerView,
    mx: f32,
    my: f32,
    o: [f32; 2],
) -> Option<OutlinerAction> {
    if point_in(mx, my, filter_rect(o)) {
        return Some(OutlinerAction::FocusFilter);
    }
    for slot in 0..ROW_POOL {
        if !point_in(mx, my, row_rect(o, slot)) {
            continue;
        }
        let Some(row) = view.rows.get(view.scroll + slot) else {
            return Some(OutlinerAction::Consume);
        };
        return Some(match row {
            OutlinerRow::Header { group, .. } => OutlinerAction::ToggleGroup(*group),
            OutlinerRow::Asset { name, .. } => {
                if point_in(mx, my, eye_rect(o, slot)) {
                    OutlinerAction::ToggleHide(name.clone())
                } else if point_in(mx, my, lock_rect(o, slot)) {
                    OutlinerAction::ToggleLock(name.clone())
                } else {
                    OutlinerAction::Select(name.clone())
                }
            }
        });
    }
    point_in(mx, my, panel_rect(o)).then_some(OutlinerAction::Consume)
}

// Position + show the panel (`Some(view)`), or blank every element (`None`).
pub(crate) fn apply(world: &mut World, view: Option<&OutlinerView>, o: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };
    widget::place_panel(world, PANEL_BG, panel_rect(o));
    let title = title_rect(o);
    widget::place_heading(world, TITLE_LABEL, title, "Outliner");
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);
    widget::show_field(world, FILTER_INPUT, filter_rect(o), view.focus);

    if let Some(l) = widget::label_mut(world, STATUS_LABEL) {
        l.x = o[0] + PAD;
        l.y = o[1] + widget::TITLE_H + HEADER_H;
        l.align = TextAlign::Left;
        l.visible = true;
        match view.status {
            Some(e) => {
                l.color = ERROR_LABEL;
                l.content = e.to_string();
            }
            None => {
                l.color = theme::LABEL_DIM;
                l.content = format!("Assets ({})", view.total);
            }
        }
    }

    for slot in 0..ROW_POOL {
        let r = row_rect(o, slot);
        let Some(row) = view.rows.get(view.scroll + slot) else {
            hide_row(world, slot);
            continue;
        };
        let hovered = point_in(view.mouse[0], view.mouse[1], r);
        match row {
            OutlinerRow::Header {
                label, count, open, ..
            } => {
                let tint = if hovered { theme::HOVER_TINT } else { ROW_TINT };
                place_rounded(
                    world,
                    row_bg(slot),
                    theme::highlight_rect(r),
                    tint,
                    theme::CONTROL_RADIUS,
                    true,
                );
                if let Some(l) = widget::label_mut(world, name_label(slot)) {
                    let marker = if *open { "-" } else { "+" };
                    l.x = r[0] + PAD;
                    l.y = r[1] + ROW_H * 0.5 - theme::TEXT_HALF;
                    l.align = TextAlign::Left;
                    l.color = theme::HEADING;
                    l.visible = true;
                    l.content = format!("{marker} {label} ({count})");
                }
                widget::set_label_visible(world, badge_label(slot), false);
                widget::set_sprite_visible(world, eye_box(slot), false);
                widget::set_sprite_visible(world, lock_box(slot), false);
            }
            OutlinerRow::Asset { name, badge, .. } => {
                let selected = view.selection.contains(name);
                let tint = if hovered {
                    theme::HOVER_TINT
                } else if selected {
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
                if let Some(l) = widget::label_mut(world, name_label(slot)) {
                    l.x = r[0] + ASSET_INDENT;
                    l.y = r[1] + ROW_H * 0.5 - theme::TEXT_HALF;
                    l.align = TextAlign::Left;
                    // The hidden state dims the name too, so a collapsed
                    // object reads as absent at a glance.
                    l.color = if view.hidden.contains(name) {
                        theme::LABEL_DIM
                    } else {
                        theme::LABEL
                    };
                    l.visible = true;
                    l.content = widget::clip_text(name, MAX_NAME_CHARS);
                }
                if let Some(l) = widget::label_mut(world, badge_label(slot)) {
                    l.x = eye_rect(o, slot)[0] - GAP;
                    l.y = r[1] + ROW_H * 0.5 - theme::TEXT_HALF;
                    l.align = TextAlign::Right;
                    l.color = badge_color(*badge);
                    l.visible = true;
                    l.content = badge.caption().to_string();
                }
                let eye = if view.hidden.contains(name) {
                    EYE_TINT_OFF
                } else {
                    EYE_TINT_ON
                };
                place_rounded(world, eye_box(slot), eye_rect(o, slot), eye, 4.0, true);
                let lock = if view.locked.contains(name) {
                    LOCK_TINT_ON
                } else {
                    LOCK_TINT_OFF
                };
                place_rounded(world, lock_box(slot), lock_rect(o, slot), lock, 4.0, true);
            }
        }
    }

    layout_scrollbar(world, view, o);
}

fn hide_row(world: &mut World, slot: usize) {
    widget::set_sprite_visible(world, row_bg(slot), false);
    widget::set_sprite_visible(world, eye_box(slot), false);
    widget::set_sprite_visible(world, lock_box(slot), false);
    widget::set_label_visible(world, name_label(slot), false);
    widget::set_label_visible(world, badge_label(slot), false);
}

fn layout_scrollbar(world: &mut World, view: &OutlinerView, o: [f32; 2]) {
    let total = view.rows.len();
    if total <= ROW_POOL {
        widget::set_sprite_visible(world, LIST_TRACK, false);
        widget::set_sprite_visible(world, LIST_THUMB, false);
        return;
    }
    let x = o[0] + OUTLINER_W - SCROLLBAR_W;
    let top = list_top(o);
    let h = ROW_POOL as f32 * ROW_H;
    place_rounded(
        world,
        LIST_TRACK,
        [x, top, SCROLLBAR_W, h],
        TRACK_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
    let thumb_h = (h * ROW_POOL as f32 / total as f32).max(18.0);
    let max_scroll = (total - ROW_POOL) as f32;
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

// Hide every panel element, blurring the search field so a hidden field
// cannot keep keyboard focus.
pub(crate) fn hide_all(world: &mut World) {
    for id in all_sprite_ids() {
        widget::set_sprite_visible(world, id, false);
    }
    for id in all_label_ids() {
        widget::set_label_visible(world, id, false);
    }
    widget::hide_field(world, FILTER_INPUT);
}

// Every panel sprite id, in draw (insertion) order: chrome, then the rows'
// backgrounds and toggles, then the scrollbar floating above them.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG];
    ids.extend((0..ROW_POOL).map(row_bg));
    ids.extend((0..ROW_POOL).map(eye_box));
    ids.extend((0..ROW_POOL).map(lock_box));
    ids.extend([LIST_TRACK, LIST_THUMB]);
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![TITLE_LABEL, CLOSE_LABEL, STATUS_LABEL];
    ids.extend((0..ROW_POOL).map(name_label));
    ids.extend((0..ROW_POOL).map(badge_label));
    ids
}

pub(crate) fn all_field_ids() -> Vec<AssetId> {
    vec![FILTER_INPUT]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextInput, TextLabel};

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
        for id in all_field_ids() {
            world.add_component(TextInput {
                asset_id: id,
                ..Default::default()
            });
        }
        world
    }

    fn tree() -> Vec<OutlinerRow> {
        vec![
            OutlinerRow::Header {
                group: 0,
                label: "World".to_string(),
                count: 2,
                open: true,
            },
            OutlinerRow::Asset {
                name: "cam".to_string(),
                asset_type: "Camera3D".to_string(),
                badge: Badge::Authored,
            },
            OutlinerRow::Asset {
                name: "fox".to_string(),
                asset_type: "SceneImport".to_string(),
                badge: Badge::Authored,
            },
            OutlinerRow::Header {
                group: 1,
                label: "fox".to_string(),
                count: 1,
                open: false,
            },
        ]
    }

    struct Fixture {
        rows: Vec<OutlinerRow>,
        selection: Selection,
        hidden: BTreeSet<String>,
        locked: BTreeSet<String>,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                rows: tree(),
                selection: Selection::default(),
                hidden: BTreeSet::new(),
                locked: BTreeSet::new(),
            }
        }

        fn view(&self, scroll: usize) -> OutlinerView<'_> {
            OutlinerView {
                rows: &self.rows,
                scroll,
                focus: true,
                selection: &self.selection,
                hidden: &self.hidden,
                locked: &self.locked,
                status: None,
                total: 3,
                mouse: [0.0, 0.0],
            }
        }
    }

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
    fn geometry_stacks_and_toggles_stay_inside_the_row() {
        let o = [40.0, 60.0];
        assert_eq!(filter_rect(o)[1], 60.0 + widget::TITLE_H + 4.0);
        assert_eq!(row_rect(o, 0)[1], list_top(o));
        assert_eq!(row_rect(o, 1)[1], list_top(o) + ROW_H);
        let r = row_rect(o, 0);
        let (e, l) = (eye_rect(o, 0), lock_rect(o, 0));
        assert!(e[0] + e[2] + GAP <= l[0] + 0.01, "eye sits left of lock");
        assert!(l[0] + l[2] <= r[0] + r[2], "lock stays inside the row");
        assert!(e[1] >= r[1] && e[1] + e[3] <= r[1] + r[3]);
        assert_eq!(size()[0], OUTLINER_W);
    }

    #[test]
    fn hit_test_resolves_filter_headers_assets_and_toggles() {
        let f = Fixture::new();
        let v = f.view(0);
        let o = [40.0, 40.0];
        let fr = filter_rect(o);
        assert_eq!(
            hit_test(&v, fr[0] + 5.0, fr[1] + 5.0, o),
            Some(OutlinerAction::FocusFilter)
        );
        let r0 = row_rect(o, 0);
        assert_eq!(
            hit_test(&v, r0[0] + 5.0, r0[1] + 5.0, o),
            Some(OutlinerAction::ToggleGroup(0))
        );
        let r1 = row_rect(o, 1);
        assert_eq!(
            hit_test(&v, r1[0] + 5.0, r1[1] + 5.0, o),
            Some(OutlinerAction::Select("cam".to_string()))
        );
        let e = eye_rect(o, 1);
        assert_eq!(
            hit_test(&v, e[0] + 2.0, e[1] + 2.0, o),
            Some(OutlinerAction::ToggleHide("cam".to_string()))
        );
        let l = lock_rect(o, 1);
        assert_eq!(
            hit_test(&v, l[0] + 2.0, l[1] + 2.0, o),
            Some(OutlinerAction::ToggleLock("cam".to_string()))
        );
        // Past the last row the click is swallowed; off the panel it misses.
        let r9 = row_rect(o, 9);
        assert_eq!(
            hit_test(&v, r9[0] + 5.0, r9[1] + 5.0, o),
            Some(OutlinerAction::Consume)
        );
        assert_eq!(hit_test(&v, 5000.0, 5000.0, o), None);
    }

    #[test]
    fn scrolled_rows_resolve_through_the_window_offset() {
        let f = Fixture::new();
        let v = f.view(2);
        let o = [40.0, 40.0];
        let r0 = row_rect(o, 0);
        assert_eq!(
            hit_test(&v, r0[0] + 5.0, r0[1] + 5.0, o),
            Some(OutlinerAction::Select("fox".to_string())),
            "slot 0 shows row 2 under scroll 2"
        );
    }

    #[test]
    fn apply_draws_headers_badges_and_toggle_states() {
        let mut world = injected_world();
        let mut f = Fixture::new();
        f.selection.replace("cam".to_string());
        f.hidden.insert("fox".to_string());
        f.locked.insert("cam".to_string());
        let o = [20.0, 20.0];
        apply(&mut world, Some(&f.view(0)), o);

        let title = label(&world, TITLE_LABEL);
        assert!(title.visible && title.content == "Outliner");
        assert_eq!(label(&world, STATUS_LABEL).content, "Assets (3)");
        assert_eq!(label(&world, name_label(0)).content, "- World (2)");
        assert!(
            !sprite(&world, eye_box(0)).visible,
            "header rows draw no toggles"
        );
        // The collapsed group header shows its fold marker.
        assert_eq!(label(&world, name_label(3)).content, "+ fox (1)");

        // cam: selected, locked, visible.
        assert_eq!(sprite(&world, row_bg(1)).tint, theme::SELECTED_TINT);
        assert_eq!(sprite(&world, eye_box(1)).tint, EYE_TINT_ON);
        assert_eq!(sprite(&world, lock_box(1)).tint, LOCK_TINT_ON);
        assert_eq!(label(&world, badge_label(1)).content, "authored");

        // fox: hidden dims the name and flips the eye.
        assert_eq!(sprite(&world, eye_box(2)).tint, EYE_TINT_OFF);
        assert_eq!(label(&world, name_label(2)).color, theme::LABEL_DIM);
        assert_eq!(sprite(&world, lock_box(2)).tint, LOCK_TINT_OFF);

        // The filter field is shown and asserts focus each frame.
        let input = world
            .query::<TextInput>()
            .find(|t| t.asset_id == FILTER_INPUT)
            .unwrap();
        assert!(input.visible && input.focused);
        // Empty slots past the tree stay blank.
        assert!(!sprite(&world, row_bg(5)).visible);
    }

    #[test]
    fn a_cook_error_replaces_the_count_line() {
        let mut world = injected_world();
        let f = Fixture::new();
        let mut v = f.view(0);
        v.status = Some("the world does not build");
        apply(&mut world, Some(&v), [20.0, 20.0]);
        let status = label(&world, STATUS_LABEL);
        assert_eq!(status.content, "the world does not build");
        assert_eq!(status.color, ERROR_LABEL);
    }

    #[test]
    fn long_tree_shows_the_scrollbar() {
        let mut world = injected_world();
        let mut f = Fixture::new();
        for i in 0..20 {
            f.rows.push(OutlinerRow::Asset {
                name: format!("a{i}"),
                asset_type: "Material".to_string(),
                badge: Badge::Imported,
            });
        }
        apply(&mut world, Some(&f.view(3)), [20.0, 20.0]);
        assert!(sprite(&world, LIST_THUMB).visible);
        // A short tree hides it again.
        let short = Fixture::new();
        apply(&mut world, Some(&short.view(0)), [20.0, 20.0]);
        assert!(!sprite(&world, LIST_THUMB).visible);
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        let f = Fixture::new();
        apply(&mut world, Some(&f.view(0)), [20.0, 20.0]);
        apply(&mut world, None, [0.0, 0.0]);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
        assert!(world.query::<TextInput>().all(|t| !t.visible));
    }
}

// src/editor/template_panel.rs
//
// The editor's Template detail panel: a floating preview of one engine-owned
// world template, spawned by clicking a row in the Templates list (`templates.rs`).
// It mirrors the edit-form panel's shape -- a draggable title bar ("Template
// <name>") with an "X" close button, a header row carrying the template's
// description on the left and an "Apply" button pinned to the top-right, then a
// scrollable list of the template's assets underneath. Applying layers the
// template's assets into the world (the hook owns that, idempotently).
//
// The asset list is the shared grouped list (`asset_list.rs`): assets grouped
// under a type sub-header, names indented and alphabetised -- identical to the
// Assets browse panel's list. Like the rest of the editor HUD the panel is plain
// `Sprite` / `TextLabel` components at reserved ids driven each frame by the hook;
// it is read-only (no per-row interaction), so it needs no typed fields.

use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

use super::asset_list::{self, ListRow, MAX_ROWS, ROW_H};
use super::registry::{self, PanelKey};
use super::theme;
use super::widget::{self, place_rounded, point_in};

// The row pools sit at `+0x20` / `+0x40`; the chrome ids below stay under
// `+0x20`, so the sub-ranges never overlap.
const TPL: u32 = registry::base(PanelKey::TemplateDetail);
pub(crate) const PANEL_BG: AssetId = AssetId(TPL);
pub(crate) const TITLE_LABEL: AssetId = AssetId(TPL + 2);
pub(crate) const CLOSE_BG: AssetId = AssetId(TPL + 3);
pub(crate) const CLOSE_LABEL: AssetId = AssetId(TPL + 4);
pub(crate) const APPLY_BG: AssetId = AssetId(TPL + 5);
pub(crate) const APPLY_LABEL: AssetId = AssetId(TPL + 6);
pub(crate) const DESC_LABEL: AssetId = AssetId(TPL + 7);
pub(crate) const LIST_TRACK: AssetId = AssetId(TPL + 8);
pub(crate) const LIST_THUMB: AssetId = AssetId(TPL + 9);

pub(crate) fn row_bg(i: usize) -> AssetId {
    AssetId(TPL + 0x20 + i as u32)
}
pub(crate) fn row_label(i: usize) -> AssetId {
    AssetId(TPL + 0x40 + i as u32)
}

// Geometry, in window pixels. Every rect derives from the panel origin `o` (the
// title bar's top-left), so dragging the title bar moves the whole panel.
pub(crate) const TPL_W: f32 = 420.0;
const PAD: f32 = 10.0;
const GAP: f32 = 8.0;
// The header row: the description and the Apply button.
const HEADER_H: f32 = 34.0;
const BTN_W: f32 = 84.0;
// The description draws a touch smaller than the body text so a full one-line
// summary clears the Apply button.
const DESC_SCALE: f32 = 0.8;

const BTN_TINT: [f32; 4] = [0.22, 0.40, 0.56, 1.0];
const BTN_TINT_HOVER: [f32; 4] = [0.28, 0.48, 0.66, 1.0];
const DESC_COLOR: [f32; 3] = [0.72, 0.76, 0.84];
const LABEL_WHITE: [f32; 3] = [1.0, 1.0, 1.0];

// A resolved Template-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TemplateAction {
    // The Apply button: layer the template's assets into the world.
    Apply,
    // The title-bar "X": close the panel.
    Close,
    // A click inside the panel that hits no control (swallowed).
    Consume,
}

// The per-frame data the hook hands to `apply` / `hit_test`.
pub(crate) struct TemplateView<'a> {
    // The title-bar heading, already prefixed ("Template Minimal 3D World").
    pub title: &'a str,
    // The template's one-line description.
    pub description: &'a str,
    // The template's assets as the shared grouped rows (read-only here).
    pub rows: &'a [ListRow],
    // First visible row of the scroll window.
    pub scroll: usize,
    pub mouse: [f32; 2],
}

// How many list rows are on screen: the row count capped at the window.
fn visible_rows(n_rows: usize) -> usize {
    n_rows.clamp(1, MAX_ROWS)
}

// Where the panel sits until the user drags it: centred, just below the top bar.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [(vw - TPL_W) * 0.5, super::hud::body_top() + 30.0]
}

// The panel footprint for a template of `n_rows` grouped rows (the list area
// tracks the row count, capped at the window). Origin-independent; the hook's
// drag clamp uses this.
pub(crate) fn size(n_rows: usize) -> [f32; 2] {
    [
        TPL_W,
        widget::TITLE_H + PAD + HEADER_H + PAD + visible_rows(n_rows) as f32 * ROW_H + PAD,
    ]
}

// The panel outer rect for a view (its height tracks the visible row count).
fn panel_rect(view: &TemplateView, o: [f32; 2]) -> [f32; 4] {
    let s = size(view.rows.len());
    [o[0], o[1], s[0], s[1]]
}

// The draggable title bar across the panel top.
pub(crate) fn title_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0], o[1], TPL_W, widget::TITLE_H]
}

// The "X" close button, flush in the title bar's top-right corner. The hook
// checks this before the title-bar drag, so clicking the X closes the panel.
pub(crate) fn close_rect(o: [f32; 2]) -> [f32; 4] {
    widget::close_rect(title_rect(o))
}

// The header row under the title bar: description on the left, Apply on the right.
fn header_y(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H + PAD
}
pub(crate) fn apply_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0] + TPL_W - PAD - BTN_W, header_y(o), BTN_W, HEADER_H]
}
// The description area, filling the header row left of the Apply button.
fn desc_rect(o: [f32; 2]) -> [f32; 4] {
    [
        o[0] + PAD,
        header_y(o),
        TPL_W - 2.0 * PAD - (BTN_W + GAP),
        HEADER_H,
    ]
}

// Where the scrollable asset list begins (below the header row).
fn body_top(o: [f32; 2]) -> f32 {
    header_y(o) + HEADER_H + PAD
}

// Full-width rect of visible list row `r`.
fn list_row_rect(o: [f32; 2], r: usize) -> [f32; 4] {
    [o[0], body_top(o) + r as f32 * ROW_H, TPL_W, ROW_H]
}

// Whether the cursor is over the panel (for wheel-scrolling the asset list).
pub(crate) fn cursor_over(view: &TemplateView, mx: f32, my: f32, o: [f32; 2]) -> bool {
    point_in(mx, my, panel_rect(view, o))
}

// Resolve a click against the open panel at origin `o`. `None` means the click
// missed the panel. Title-bar presses never reach this: the hook intercepts them
// first to start a drag (and to catch the "X").
pub(crate) fn hit_test(
    view: &TemplateView,
    mx: f32,
    my: f32,
    o: [f32; 2],
) -> Option<TemplateAction> {
    if !point_in(mx, my, panel_rect(view, o)) {
        return None;
    }
    if point_in(mx, my, close_rect(o)) {
        return Some(TemplateAction::Close);
    }
    if point_in(mx, my, apply_rect(o)) {
        return Some(TemplateAction::Apply);
    }
    // The asset list is read-only; every other click inside the panel is
    // swallowed so it cannot fall through to the world.
    Some(TemplateAction::Consume)
}

// Position + show the panel's elements for this frame at origin `o`, or hide them
// all when the panel is closed (`view` is `None`).
pub(crate) fn apply(world: &mut World, view: Option<&TemplateView>, o: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };

    // Blank everything, then re-show what this frame needs.
    hide_all(world);

    widget::place_panel(world, PANEL_BG, panel_rect(view, o));
    widget::place_heading(world, TITLE_LABEL, title_rect(o), view.title);
    let close_hover = point_in(view.mouse[0], view.mouse[1], close_rect(o));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title_rect(o), close_hover);

    // The header row: the description (left) and the pinned Apply button (right).
    let desc = desc_rect(o);
    if let Some(l) = widget::label_mut(world, DESC_LABEL) {
        l.x = desc[0];
        l.y = desc[1] + HEADER_H * 0.5 - 10.0 * DESC_SCALE;
        l.align = TextAlign::Left;
        l.color = DESC_COLOR;
        l.scale = DESC_SCALE;
        l.visible = true;
        l.content = view.description.to_string();
    }
    let apply_btn = apply_rect(o);
    let hover = point_in(view.mouse[0], view.mouse[1], apply_btn);
    place_rounded(
        world,
        APPLY_BG,
        apply_btn,
        if hover { BTN_TINT_HOVER } else { BTN_TINT },
        theme::CONTROL_RADIUS,
        true,
    );
    if let Some(l) = widget::label_mut(world, APPLY_LABEL) {
        l.x = apply_btn[0] + apply_btn[2] * 0.5;
        l.y = apply_btn[1] + apply_btn[3] * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Center;
        l.color = LABEL_WHITE;
        l.visible = true;
        l.content = "Apply".to_string();
    }

    // The template's assets: the shared grouped list, a scrolling window over the
    // row pool, plus a scrollbar when it overflows.
    let total = view.rows.len();
    let scroll = view.scroll.min(total.saturating_sub(1));
    for r in 0..MAX_ROWS {
        let idx = scroll + r;
        if idx >= total {
            break;
        }
        asset_list::place_row(
            world,
            row_bg(r),
            row_label(r),
            &view.rows[idx],
            list_row_rect(o, r),
            asset_list::ROW_TINT,
        );
    }
    asset_list::layout_scrollbar(
        world,
        LIST_TRACK,
        LIST_THUMB,
        total,
        scroll,
        o[0] + TPL_W,
        body_top(o),
    );
}

// Hide every panel element (the closed / F1-hidden pass).
pub(crate) fn hide_all(world: &mut World) {
    for id in all_sprite_ids() {
        widget::set_sprite_visible(world, id, false);
    }
    for id in all_label_ids() {
        widget::set_label_visible(world, id, false);
    }
}

// Every panel sprite id, for injection and the hidden pass. THE ORDER IS THE DRAW
// ORDER (inject.rs inserts in this sequence, the overlay draws in insertion
// order): background + chrome, the row backgrounds, then the scrollbar above them.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG, APPLY_BG];
    ids.extend((0..MAX_ROWS).map(row_bg));
    ids.extend([LIST_TRACK, LIST_THUMB]);
    ids
}
pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![TITLE_LABEL, CLOSE_LABEL, APPLY_LABEL, DESC_LABEL];
    ids.extend((0..MAX_ROWS).map(row_label));
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Sprite, TextLabel};

    fn test_origin() -> [f32; 2] {
        default_origin(1280.0)
    }

    fn rows() -> Vec<ListRow> {
        vec![
            ListRow {
                is_header: true,
                text: "Camera3D".into(),
                entry: None,
            },
            ListRow {
                is_header: false,
                text: "world_camera".into(),
                entry: Some(0),
            },
            ListRow {
                is_header: true,
                text: "Room".into(),
                entry: None,
            },
            ListRow {
                is_header: false,
                text: "world_room".into(),
                entry: Some(1),
            },
        ]
    }

    fn view(rows: &[ListRow]) -> TemplateView<'_> {
        TemplateView {
            title: "Template Minimal 3D World",
            description: "A lit room with a camera and sky.",
            rows,
            scroll: 0,
            mouse: [0.0, 0.0],
        }
    }

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

    fn sprite(world: &World, id: AssetId) -> Sprite {
        world
            .query::<Sprite>()
            .find(|s| s.asset_id == id)
            .cloned()
            .expect("sprite present")
    }
    fn label(world: &World, id: AssetId) -> TextLabel {
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .cloned()
            .expect("label present")
    }

    // The header pins the description left of the Apply button, both under the
    // title bar; the X sits in the title bar's right corner.
    #[test]
    fn header_pins_description_and_apply_under_the_title_bar() {
        let o = test_origin();
        let title = title_rect(o);
        assert_eq!(title, [o[0], o[1], TPL_W, widget::TITLE_H]);
        let close = close_rect(o);
        assert_eq!(close[1], o[1], "the X sits in the title bar");
        assert_eq!(
            close[0] + close[2],
            o[0] + TPL_W,
            "flush to the panel right"
        );
        let desc = desc_rect(o);
        let apply_btn = apply_rect(o);
        assert_eq!(desc[1], o[1] + widget::TITLE_H + PAD, "under the title bar");
        assert_eq!(
            desc[1], apply_btn[1],
            "description shares Apply's header row"
        );
        assert!(
            desc[0] + desc[2] <= apply_btn[0],
            "description ends left of Apply"
        );
        assert_eq!(
            apply_btn[0] + apply_btn[2],
            o[0] + TPL_W - PAD,
            "Apply flush to the panel's right pad"
        );
        // The whole panel follows its origin.
        let v = view(&[]);
        let moved = panel_rect(&v, [40.0, 60.0]);
        assert_eq!((moved[0], moved[1]), (40.0, 60.0));
        assert_eq!(
            size(0),
            [moved[2], moved[3]],
            "drag-clamp footprint matches"
        );
    }

    // The panel height tracks the row count up to the window cap.
    #[test]
    fn panel_height_tracks_rows_and_caps_at_the_window() {
        let short = size(3);
        let tall = size(MAX_ROWS + 10);
        assert!(short[1] < tall[1]);
        assert_eq!(
            tall[1],
            size(MAX_ROWS)[1],
            "a list past the window scrolls instead of growing"
        );
    }

    #[test]
    fn apply_and_close_hit_test() {
        let rs = rows();
        let v = view(&rs);
        let o = test_origin();
        let a = apply_rect(o);
        let x = close_rect(o);
        assert_eq!(
            hit_test(&v, a[0] + 5.0, a[1] + 5.0, o),
            Some(TemplateAction::Apply)
        );
        assert_eq!(
            hit_test(&v, x[0] + 5.0, x[1] + 5.0, o),
            Some(TemplateAction::Close)
        );
        // A title-bar click off the X is consumed (the hook grabs drags first).
        let t = title_rect(o);
        assert_eq!(
            hit_test(&v, t[0] + 5.0, t[1] + 5.0, o),
            Some(TemplateAction::Consume)
        );
        // A body click is consumed; a miss falls through.
        let row = list_row_rect(o, 1);
        assert_eq!(
            hit_test(&v, row[0] + 5.0, row[1] + 5.0, o),
            Some(TemplateAction::Consume)
        );
        assert_eq!(hit_test(&v, 5.0, 5.0, o), None);
    }

    // Applying lays out the title, description, Apply caption, and the grouped
    // asset rows (headers + indented names).
    #[test]
    fn apply_renders_title_description_and_rows() {
        let rs = rows();
        let mut world = injected_world();
        let o = test_origin();
        apply(&mut world, Some(&view(&rs)), o);
        assert_eq!(
            label(&world, TITLE_LABEL).content,
            "Template Minimal 3D World"
        );
        let desc = label(&world, DESC_LABEL);
        assert!(desc.visible && desc.content == "A lit room with a camera and sky.");
        assert!(desc.scale < 1.0, "description draws a touch smaller");
        assert_eq!(label(&world, APPLY_LABEL).content, "Apply");
        // Row 0 is the "Camera3D" header; row 1 is the indented "world_camera".
        assert_eq!(label(&world, row_label(0)).content, "Camera3D");
        let name = label(&world, row_label(1));
        assert_eq!(name.content, "world_camera");
        assert!(
            name.x > label(&world, row_label(0)).x,
            "the name is indented under its type header"
        );
        assert!(sprite(&world, PANEL_BG).visible);
    }

    // A short list draws no scrollbar; a long one does.
    #[test]
    fn scrollbar_only_when_the_list_overflows() {
        let short = rows();
        let mut world = injected_world();
        apply(&mut world, Some(&view(&short)), test_origin());
        assert!(!sprite(&world, LIST_THUMB).visible, "short list, no bar");

        let long: Vec<ListRow> = (0..MAX_ROWS + 6)
            .map(|i| ListRow {
                is_header: false,
                text: format!("asset_{i}"),
                entry: Some(i),
            })
            .collect();
        apply(&mut world, Some(&view(&long)), test_origin());
        assert!(sprite(&world, LIST_THUMB).visible, "overflow shows the bar");
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let rs = rows();
        let mut world = injected_world();
        apply(&mut world, Some(&view(&rs)), test_origin());
        hide_all(&mut world);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}

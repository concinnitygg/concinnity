//! The Story panel's layout half: an editor over the Markdown source of the
//! world's `StoryImport` (see `story.rs` for the starter text and
//! `hook/edit/story.rs` for the actions). The header carries the source path
//! and the Apply button, a reserved status line shows parse / IO errors, and
//! the body is a code text area. A world with no `StoryImport` shows a single
//! "+ Create story" row instead.

use concinnity_core::components::TextAlign;
use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;

use super::registry::{self, PanelKey};
use crate::editor::text_area::TextArea;
use crate::editor::text_area::layout::{self, Geometry, Metrics, TextAreaIds, TextAreaView};
use crate::editor::text_area::markers::GutterMarker;
use crate::editor::theme;
use crate::editor::widget::{self, place_rounded, point_in};

const BASE: u32 = registry::base(PanelKey::Story);
pub(crate) const PANEL_BG: AssetId = AssetId(BASE);
pub(crate) const TITLE_LABEL: AssetId = AssetId(BASE + 2);
pub(crate) const CLOSE_BG: AssetId = AssetId(BASE + 3);
pub(crate) const CLOSE_LABEL: AssetId = AssetId(BASE + 4);
pub(crate) const APPLY_BG: AssetId = AssetId(BASE + 5);
pub(crate) const APPLY_LABEL: AssetId = AssetId(BASE + 6);
pub(crate) const PATH_LABEL: AssetId = AssetId(BASE + 7);
pub(crate) const STATUS_LABEL: AssetId = AssetId(BASE + 8);
pub(crate) const CREATE_BG: AssetId = AssetId(BASE + 9);
pub(crate) const CREATE_LABEL: AssetId = AssetId(BASE + 10);
pub(crate) const AREA: TextAreaIds = TextAreaIds::new(BASE + 0x100);

// Geometry, in window pixels. Every rect derives from the panel origin `o`
// (the title bar's top-left), so dragging the title bar moves the whole panel.
// The default (and minimum) width; the user can widen the panel past this.
pub(crate) const STORY_W: f32 = 620.0;
const PAD: f32 = 10.0;
const HEADER_H: f32 = 32.0;
// Two lines: a parse or IO message carries a path and rarely fits one.
const STATUS_H: f32 = 2.0 * widget::LINE_H + 6.0;
const BTN_W: f32 = 70.0;
const CREATE_H: f32 = 24.0;
// Rows the text area shows at the default height.
const DEFAULT_ROWS: usize = 20;

// The chrome height above the text area.
const CHROME_H: f32 = widget::TITLE_H + HEADER_H + STATUS_H;

const BTN_TINT: [f32; 4] = [0.22, 0.40, 0.56, 1.0];
const BTN_TINT_HOVER: [f32; 4] = [0.28, 0.48, 0.66, 1.0];
const PATH_COLOR: [f32; 3] = [0.62, 0.66, 0.76];
const ADD_LABEL: [f32; 3] = [0.55, 0.80, 0.60];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];

// The per-frame view the hook assembles.
pub(crate) struct StoryView<'a> {
    pub area: &'a TextArea,
    // Whether the text area holds the keyboard this frame.
    pub focus: bool,
    // The story source path shown in the header; empty in create mode.
    pub path: &'a str,
    // Parse / IO error from the last Apply (or load).
    pub status: Option<&'a str>,
    // Lines the last Apply's parse error names.
    pub markers: &'a [GutterMarker],
    // No StoryImport in the world: show the create row instead of the text.
    pub create: bool,
    pub mouse: [f32; 2],
}

// A resolved Story-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StoryAction {
    // The press landed on the text area; the hook hands it to the area.
    Text,
    // Write the starter story and add its StoryImport entry.
    Create,
    // Validate, write the source file, and refresh the live preview.
    Apply,
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Where the panel sits until the user drags it: centered below the top bar
// (clear of the left default stack and the right Assets anchor).
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [(vw - STORY_W) * 0.5, crate::editor::hud::body_top() + 8.0]
}

// The panel's default (and minimum) footprint.
pub(crate) fn size() -> [f32; 2] {
    [
        STORY_W,
        CHROME_H + layout::height_for_rows(DEFAULT_ROWS, Metrics::code().line_h) + PAD,
    ]
}

// The tallest the panel resizes to: the text area's whole row pool shown.
pub(crate) fn max_size() -> [f32; 2] {
    let rows = layout::height_for_rows(layout::ROW_POOL, Metrics::code().line_h);
    [f32::INFINITY, CHROME_H + rows + PAD]
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

// The text area's rect for the panel at origin `o`, size `s`.
pub(crate) fn area_rect(o: [f32; 2], s: [f32; 2]) -> [f32; 4] {
    let top = body_top(o);
    [
        o[0] + PAD,
        top,
        (s[0] - 2.0 * PAD).max(0.0),
        (o[1] + s[1] - PAD - top).max(0.0),
    ]
}

// The text area laid out for the panel at origin `o`, size `s`.
pub(crate) fn area_geometry(o: [f32; 2], s: [f32; 2], area: &TextArea, m: Metrics) -> Geometry {
    layout::geometry(area_rect(o, s), m, area.line_count())
}

// The "+ Create story" row shown in create mode.
fn create_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    [o[0] + PAD, body_top(o), w - 2.0 * PAD, CREATE_H]
}

// Whether the cursor is over the text area (for wheel routing).
pub(crate) fn cursor_over_area(mx: f32, my: f32, o: [f32; 2], s: [f32; 2]) -> bool {
    point_in(mx, my, area_rect(o, s))
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
    if view.create {
        if point_in(mx, my, create_rect(o, s[0])) {
            return Some(StoryAction::Create);
        }
    } else if point_in(mx, my, apply_rect(o, s[0])) {
        return Some(StoryAction::Apply);
    } else if point_in(mx, my, area_rect(o, s)) {
        return Some(StoryAction::Text);
    }
    point_in(mx, my, widget::outer_rect(o, s)).then_some(StoryAction::Consume)
}

// Position + show the panel (`Some(view)`) at effective size `s`, or blank every
// element (`None`).
pub(crate) fn place(
    world: &mut World,
    view: Option<&StoryView>,
    o: [f32; 2],
    s: [f32; 2],
    m: Metrics,
) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };
    let w = s[0];
    widget::place_panel(world, PANEL_BG, widget::outer_rect(o, s));
    let title = widget::title_rect(o, w);
    let dirty = !view.create && view.area.is_dirty();
    let heading = if dirty { "Story *" } else { "Story" };
    widget::place_heading(world, TITLE_LABEL, title, heading);
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);
    place_header(world, view, o, w);

    let g = area_geometry(o, s, view.area, m);
    let area_view = TextAreaView {
        area: view.area,
        focused: view.focus,
        markers: view.markers,
        mouse: view.mouse,
    };
    let status = layout::hovered_marker(&area_view, &g)
        .map(|mk| mk.message.as_str())
        .or(view.status);
    widget::place_message(
        world,
        STATUS_LABEL,
        [
            o[0] + PAD,
            o[1] + widget::TITLE_H + HEADER_H + 1.0,
            (w - 2.0 * PAD).max(0.0),
            STATUS_H - 4.0,
        ],
        status.unwrap_or(""),
        ERROR_LABEL,
        status.is_some(),
    );

    if view.create {
        layout::hide(world, AREA);
        let r = create_rect(o, w);
        let hovered = point_in(view.mouse[0], view.mouse[1], r);
        let tint = if hovered { theme::HOVER_TINT } else { [0.0; 4] };
        place_rounded(world, CREATE_BG, r, tint, theme::CONTROL_RADIUS, true);
        widget::place_left_label(
            world,
            CREATE_LABEL,
            [r[0] + PAD, r[1] + CREATE_H * 0.5 - theme::TEXT_HALF],
            "+ Create story",
            ADD_LABEL,
            true,
        );
    } else {
        widget::set_sprite_visible(world, CREATE_BG, false);
        widget::set_label_visible(world, CREATE_LABEL, false);
        layout::place(world, AREA, Some(&area_view), &g);
    }
}

// The source path on the left and Apply on the right, both hidden in create
// mode, where there is nothing to write yet.
fn place_header(world: &mut World, view: &StoryView, o: [f32; 2], w: f32) {
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
}

// Hide every panel element.
pub(crate) fn hide_all(world: &mut World) {
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &[]);
    layout::hide(world, AREA);
}

// Every panel sprite id, in draw (insertion) order: chrome, then the create
// row, then the text area over the panel.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG, APPLY_BG, CREATE_BG];
    ids.extend(AREA.sprite_ids());
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    vec![
        TITLE_LABEL,
        CLOSE_LABEL,
        APPLY_LABEL,
        PATH_LABEL,
        STATUS_LABEL,
        CREATE_LABEL,
    ]
}

// The labels drawn in the code face: the text area's lines and line numbers.
pub(crate) fn code_label_ids() -> Vec<AssetId> {
    AREA.code_label_ids()
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::{Sprite, TextLabel};

    const M: Metrics = Metrics {
        line_h: 20.0,
        advance: 8.0,
        text_px: 14.0,
        label_scale: 0.7,
    };

    fn injected_world() -> World {
        let mut labels = all_label_ids();
        labels.extend(code_label_ids());
        crate::test_support::injected_world(&all_sprite_ids(), &labels, &[])
    }

    fn view(area: &TextArea) -> StoryView<'_> {
        StoryView {
            area,
            focus: true,
            path: "story.md",
            status: None,
            markers: &[],
            create: false,
            mouse: [0.0, 0.0],
        }
    }

    fn text(n: usize) -> TextArea {
        let lines: Vec<String> = (0..n).map(|i| format!("line {i}")).collect();
        TextArea::from_text(&lines.join("\n"))
    }

    fn label(world: &World, id: AssetId) -> TextLabel {
        world.get_by_id::<TextLabel>(id).unwrap().clone()
    }

    #[test]
    fn hit_test_resolves_apply_text_and_create() {
        let area = text(4);
        let v = view(&area);
        let o = [40.0, 40.0];
        let s = [STORY_W, 600.0];
        let a = apply_rect(o, STORY_W);
        assert_eq!(
            hit_test(&v, a[0] + 5.0, a[1] + 5.0, o, s),
            Some(StoryAction::Apply)
        );
        let r = area_rect(o, s);
        assert_eq!(
            hit_test(&v, r[0] + 50.0, r[1] + 50.0, o, s),
            Some(StoryAction::Text)
        );
        assert_eq!(
            hit_test(&v, o[0] + 2.0, o[1] + 40.0, o, s),
            Some(StoryAction::Consume),
            "the header outside Apply is swallowed"
        );
        assert_eq!(hit_test(&v, 5000.0, 5000.0, o, s), None);
        let create = StoryView {
            create: true,
            ..view(&area)
        };
        let c = create_rect(o, STORY_W);
        assert_eq!(
            hit_test(&create, c[0] + 5.0, c[1] + 5.0, o, s),
            Some(StoryAction::Create)
        );
        assert_eq!(
            hit_test(&create, a[0] + 5.0, a[1] + 5.0, o, s),
            Some(StoryAction::Consume),
            "no Apply in create mode"
        );
        assert_eq!(
            hit_test(&create, r[0] + 50.0, r[1] + 200.0, o, s),
            Some(StoryAction::Consume),
            "no text area in create mode"
        );
    }

    #[test]
    fn place_draws_the_text_area_and_marks_unapplied_edits() {
        let mut world = injected_world();
        let mut area = text(4);
        place(&mut world, Some(&view(&area)), [20.0, 20.0], size(), M);
        assert_eq!(label(&world, TITLE_LABEL).content, "Story");
        let lines: Vec<String> = AREA
            .code_label_ids()
            .into_iter()
            .map(|id| label(&world, id))
            .filter(|l| l.visible)
            .map(|l| l.content)
            .collect();
        assert!(lines.contains(&"line 2".to_string()), "{lines:?}");
        assert!(
            lines.contains(&"4".to_string()),
            "the gutter numbers from 1"
        );
        assert!(!world.get_by_id::<Sprite>(CREATE_BG).unwrap().visible);

        area.type_char('x');
        place(&mut world, Some(&view(&area)), [20.0, 20.0], size(), M);
        assert_eq!(label(&world, TITLE_LABEL).content, "Story *");
    }

    #[test]
    fn create_mode_shows_the_create_row_only() {
        let mut world = injected_world();
        let area = text(1);
        let v = StoryView {
            create: true,
            ..view(&area)
        };
        place(&mut world, Some(&v), [20.0, 20.0], size(), M);
        let row = label(&world, CREATE_LABEL);
        assert!(row.visible && row.content == "+ Create story");
        assert!(!world.get_by_id::<Sprite>(APPLY_BG).unwrap().visible);
        assert!(
            AREA.code_label_ids()
                .into_iter()
                .all(|id| !label(&world, id).visible),
            "no text area in create mode"
        );
    }

    #[test]
    fn the_status_line_shows_the_last_error() {
        let mut world = injected_world();
        let area = text(1);
        let v = StoryView {
            status: Some("bad heading"),
            ..view(&area)
        };
        place(&mut world, Some(&v), [20.0, 20.0], size(), M);
        let status = label(&world, STATUS_LABEL);
        assert!(status.visible && status.content.contains("bad heading"));
    }

    #[test]
    fn taller_panels_grow_the_text_area_up_to_its_pool() {
        let area = text(200);
        let default_rows = area_geometry([0.0, 0.0], size(), &area, Metrics::code())
            .view()
            .rows;
        assert_eq!(default_rows, DEFAULT_ROWS);
        let max = area_geometry([0.0, 0.0], max_size(), &area, Metrics::code())
            .view()
            .rows;
        assert_eq!(max, layout::ROW_POOL);
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        let area = text(4);
        place(&mut world, Some(&view(&area)), [20.0, 20.0], size(), M);
        place(&mut world, None, [0.0, 0.0], size(), M);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }
}

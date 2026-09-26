//! The Shader source panel's layout half: one of a Shader's `.hlsl` files in a
//! code text area (see `shader_source.rs` and `shader_diagnostics.rs` for the
//! data, `hook/edit/shader_source.rs` for the actions). The title names the
//! Shader and the file's stage, the header carries the path, an unsaved-edits
//! mark, the reference toggle and Save, a status line reports the last compile,
//! and the gutter marks the lines its diagnostics name. The reference column
//! (`shader_reference_panel.rs`) sits at the right, beside the text.

use concinnity_core::components::TextAlign;
use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;

use super::registry::{self, PanelKey};
use super::shader_diagnostics::{Status, Tone};
use super::shader_list_panel::tone_color;
use super::shader_reference_panel::{self as reference, COLUMN_W, RefView};
use crate::editor::text_area::TextArea;
use crate::editor::text_area::layout::{self, Geometry, Metrics, TextAreaIds, TextAreaView};
use crate::editor::text_area::markers::{GutterMarker, Severity};
use crate::editor::theme;
use crate::editor::widget::{self, place_rounded, point_in};

const BASE: u32 = registry::base(PanelKey::ShaderSource);
pub(crate) const PANEL_BG: AssetId = AssetId(BASE);
const TITLE_LABEL: AssetId = AssetId(BASE + 2);
const CLOSE_BG: AssetId = AssetId(BASE + 3);
const CLOSE_LABEL: AssetId = AssetId(BASE + 4);
const SAVE_BG: AssetId = AssetId(BASE + 5);
const SAVE_LABEL: AssetId = AssetId(BASE + 6);
const PATH_LABEL: AssetId = AssetId(BASE + 7);
const STATUS_LABEL: AssetId = AssetId(BASE + 8);
const DIRTY_LABEL: AssetId = AssetId(BASE + 9);
const REF_BG: AssetId = AssetId(BASE + 10);
const REF_LABEL: AssetId = AssetId(BASE + 11);
pub(crate) const AREA: TextAreaIds = TextAreaIds::new(BASE + 0x100);

// The default (and minimum) width; the user can widen the panel past this.
const SOURCE_W: f32 = 660.0;
const PAD: f32 = 10.0;
const HEADER_H: f32 = 32.0;
// Two lines: a compiler message quoting a path rarely fits one.
const STATUS_H: f32 = 2.0 * widget::LINE_H + 6.0;
const BTN_W: f32 = 70.0;
const REF_BTN_W: f32 = 90.0;
const DIRTY_W: f32 = 70.0;
const DEFAULT_ROWS: usize = 24;
const CHROME_H: f32 = widget::TITLE_H + HEADER_H + STATUS_H;

const BTN_TINT: [f32; 4] = [0.22, 0.40, 0.56, 1.0];
const BTN_TINT_HOVER: [f32; 4] = [0.28, 0.48, 0.66, 1.0];
const PATH_COLOR: [f32; 3] = [0.62, 0.66, 0.76];

// The per-frame view the hook assembles.
pub(crate) struct SourceView<'a> {
    // "water fragment".
    pub title: &'a str,
    pub path: &'a str,
    pub area: &'a TextArea,
    // Whether the text area holds the keyboard this frame.
    pub focus: bool,
    pub status: Option<&'a Status>,
    pub markers: &'a [GutterMarker],
    pub mouse: [f32; 2],
    // The reference column, while it is shown.
    pub reference: Option<RefView<'a>>,
}

// A resolved source-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceAction {
    // The press landed on the text area; the hook hands it to the area.
    Text,
    // Write the file; the hot-reload watcher recompiles it.
    Save,
    // The status line: jump to the error it reports.
    Status,
    // Show or hide the reference column.
    ToggleReference,
    // A row of the reference column, counted from its first shown row.
    Reference(usize),
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Where the panel sits until the user drags it: left of center, below the
// top bar, so it opens beside the Shaders list rather than over it.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [
        (vw * 0.5 - SOURCE_W).max(8.0),
        crate::editor::hud::body_top() + 8.0,
    ]
}

// The room the reference column takes from the text, when shown.
fn reference_w(reference: bool) -> f32 {
    if reference { COLUMN_W + PAD } else { 0.0 }
}

pub(crate) fn size(reference: bool) -> [f32; 2] {
    [
        SOURCE_W + reference_w(reference),
        CHROME_H + layout::height_for_rows(DEFAULT_ROWS, Metrics::code().line_h) + PAD,
    ]
}

// The tallest the panel resizes to: the text area's whole row pool shown.
pub(crate) fn max_size() -> [f32; 2] {
    let rows = layout::height_for_rows(layout::ROW_POOL, Metrics::code().line_h);
    [f32::INFINITY, CHROME_H + rows + PAD]
}

fn save_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    [
        o[0] + w - PAD - BTN_W,
        o[1] + widget::TITLE_H + 4.0,
        BTN_W,
        HEADER_H - 8.0,
    ]
}

fn reference_button_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    let save = save_rect(o, w);
    [save[0] - PAD - REF_BTN_W, save[1], REF_BTN_W, save[3]]
}

fn status_rect(o: [f32; 2], w: f32) -> [f32; 4] {
    [
        o[0] + PAD,
        o[1] + widget::TITLE_H + HEADER_H + 1.0,
        (w - 2.0 * PAD).max(0.0),
        STATUS_H - 4.0,
    ]
}

pub(crate) fn area_rect(o: [f32; 2], s: [f32; 2], reference: bool) -> [f32; 4] {
    let top = o[1] + CHROME_H;
    [
        o[0] + PAD,
        top,
        (s[0] - 2.0 * PAD - reference_w(reference)).max(0.0),
        (o[1] + s[1] - PAD - top).max(0.0),
    ]
}

// The reference column: the text's height, at the panel's right.
pub(crate) fn reference_rect(o: [f32; 2], s: [f32; 2]) -> [f32; 4] {
    let text = area_rect(o, s, true);
    [o[0] + s[0] - PAD - COLUMN_W, text[1], COLUMN_W, text[3]]
}

pub(crate) fn area_geometry(
    o: [f32; 2],
    s: [f32; 2],
    reference: bool,
    area: &TextArea,
    m: Metrics,
) -> Geometry {
    layout::geometry(area_rect(o, s, reference), m, area.line_count())
}

pub(crate) fn cursor_over_area(
    mx: f32,
    my: f32,
    o: [f32; 2],
    s: [f32; 2],
    reference: bool,
) -> bool {
    point_in(mx, my, area_rect(o, s, reference))
}

pub(crate) fn cursor_over_reference(mx: f32, my: f32, o: [f32; 2], s: [f32; 2]) -> bool {
    point_in(mx, my, reference_rect(o, s))
}

// Resolve a click at `(mx, my)` against the panel at origin `o`, size `s`,
// with the reference column shown or not. `None` means the click missed the
// panel.
pub(crate) fn hit_test(
    mx: f32,
    my: f32,
    o: [f32; 2],
    s: [f32; 2],
    reference: bool,
) -> Option<SourceAction> {
    if point_in(mx, my, save_rect(o, s[0])) {
        return Some(SourceAction::Save);
    }
    if point_in(mx, my, reference_button_rect(o, s[0])) {
        return Some(SourceAction::ToggleReference);
    }
    if point_in(mx, my, status_rect(o, s[0])) {
        return Some(SourceAction::Status);
    }
    if point_in(mx, my, area_rect(o, s, reference)) {
        return Some(SourceAction::Text);
    }
    if reference && let Some(slot) = reference::slot_at(reference_rect(o, s), mx, my) {
        return Some(SourceAction::Reference(slot));
    }
    point_in(mx, my, widget::outer_rect(o, s)).then_some(SourceAction::Consume)
}

// The status line's text and tone: the gutter marker or reference name under
// the cursor while one is hovered, else the panel's own status.
fn status_line(view: &SourceView, g: &Geometry, reference: [f32; 4]) -> Option<Status> {
    let area_view = TextAreaView {
        area: view.area,
        focused: view.focus,
        markers: view.markers,
        mouse: view.mouse,
    };
    match layout::hovered_marker(&area_view, g) {
        Some(m) => {
            let tone = match m.severity {
                Severity::Error => Tone::Error,
                Severity::Warning => Tone::Warning,
            };
            Some(Status::new(
                format!("line {}: {}", m.line + 1, m.message),
                tone,
            ))
        }
        None => view
            .reference
            .as_ref()
            .and_then(|r| reference::hovered(r, reference))
            .and_then(|row| row.describe())
            .map(|text| Status::new(text, Tone::Info))
            .or_else(|| view.status.cloned()),
    }
}

// Position + show the panel (`Some(view)`) at effective size `s`, or blank every
// element (`None`).
pub(crate) fn place(
    world: &mut World,
    view: Option<&SourceView>,
    o: [f32; 2],
    s: [f32; 2],
    m: Metrics,
) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };
    let w = s[0];
    let dirty = view.area.is_dirty();
    widget::place_panel(world, PANEL_BG, widget::outer_rect(o, s));
    let title = widget::title_rect(o, w);
    let heading = match dirty {
        true => format!("{} *", view.title),
        false => view.title.to_string(),
    };
    widget::place_heading(world, TITLE_LABEL, title, &heading);
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);
    place_header(world, view, o, w, dirty);

    let shows_reference = view.reference.is_some();
    let g = area_geometry(o, s, shows_reference, view.area, m);
    let reference_at = reference_rect(o, s);
    let status = status_line(view, &g, reference_at);
    let (text, tone) = status
        .as_ref()
        .map_or(("", Tone::Info), |s| (s.text.as_str(), s.tone));
    widget::place_message(
        world,
        STATUS_LABEL,
        status_rect(o, w),
        text,
        tone_color(tone),
        status.is_some(),
    );
    let area_view = TextAreaView {
        area: view.area,
        focused: view.focus,
        markers: view.markers,
        mouse: view.mouse,
    };
    layout::place(world, AREA, Some(&area_view), &g);
    reference::place(world, view.reference.as_ref(), reference_at);
}

// The path on the left, then the unsaved-edits mark and Save on the right.
fn place_header(world: &mut World, view: &SourceView, o: [f32; 2], w: f32, dirty: bool) {
    let y = o[1] + widget::TITLE_H + HEADER_H * 0.5 - theme::TEXT_HALF;
    let save = save_rect(o, w);
    let toggle = reference_button_rect(o, w);
    let path_w = (toggle[0] - DIRTY_W - PAD - (o[0] + PAD)).max(0.0);
    widget::place_message(
        world,
        PATH_LABEL,
        [o[0] + PAD, y, path_w, widget::LINE_H],
        view.path,
        PATH_COLOR,
        true,
    );
    if let Some(l) = widget::label_mut(world, DIRTY_LABEL) {
        l.x = toggle[0] - PAD;
        l.y = y;
        l.align = TextAlign::Right;
        l.color = theme::LOG_WARN;
        l.visible = dirty;
        l.content = "unsaved".to_string();
    }
    let open = view.reference.is_some();
    let toggle_tint = match (open, point_in(view.mouse[0], view.mouse[1], toggle)) {
        (true, _) => theme::ACCENT_TINT,
        (false, true) => theme::HOVER_TINT,
        (false, false) => theme::BUTTON_TINT,
    };
    place_rounded(
        world,
        REF_BG,
        toggle,
        toggle_tint,
        theme::CONTROL_RADIUS,
        true,
    );
    widget::place_center_label(
        world,
        REF_LABEL,
        [
            toggle[0] + toggle[2] * 0.5,
            toggle[1] + toggle[3] * 0.5 - theme::TEXT_HALF,
        ],
        "Reference",
        [1.0, 1.0, 1.0],
        true,
    );
    let hover = point_in(view.mouse[0], view.mouse[1], save);
    let tint = if hover { BTN_TINT_HOVER } else { BTN_TINT };
    place_rounded(world, SAVE_BG, save, tint, theme::CONTROL_RADIUS, true);
    widget::place_center_label(
        world,
        SAVE_LABEL,
        [
            save[0] + save[2] * 0.5,
            save[1] + save[3] * 0.5 - theme::TEXT_HALF,
        ],
        "Save",
        [1.0, 1.0, 1.0],
        true,
    );
}

pub(crate) fn hide_all(world: &mut World) {
    widget::hide_all(world, &all_sprite_ids(), &all_label_ids(), &[]);
    layout::hide(world, AREA);
    reference::hide_all(world);
}

// Every sprite id in draw order: the chrome, then the text area and the
// reference column over it.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG, SAVE_BG, REF_BG];
    ids.extend(AREA.sprite_ids());
    ids.extend(reference::sprite_ids());
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    vec![
        TITLE_LABEL,
        CLOSE_LABEL,
        SAVE_LABEL,
        PATH_LABEL,
        STATUS_LABEL,
        DIRTY_LABEL,
        REF_LABEL,
    ]
    .into_iter()
    .chain(reference::label_ids())
    .collect()
}

// The labels drawn in the code face: the text area's lines and line numbers,
// and the reference column's names.
pub(crate) fn code_label_ids() -> Vec<AssetId> {
    let mut ids = AREA.code_label_ids();
    ids.extend(reference::code_label_ids());
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::panels::shader_reference::{RefRow, Reference};
    use concinnity_core::components::{Sprite, TextLabel};
    use concinnity_core::render::shader_programs::vocabulary::ENTRIES;

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

    fn view<'a>(
        area: &'a TextArea,
        status: Option<&'a Status>,
        markers: &'a [GutterMarker],
    ) -> SourceView<'a> {
        SourceView {
            title: "water fragment",
            path: "shaders/water.hlsl",
            area,
            focus: true,
            status,
            markers,
            mouse: [0.0, 0.0],
            reference: None,
        }
    }

    fn label(world: &World, id: AssetId) -> TextLabel {
        world.get_by_id::<TextLabel>(id).cloned().unwrap()
    }

    #[test]
    fn hit_test_resolves_save_status_and_text() {
        let o = [40.0, 40.0];
        let s = size(false);
        let at = |r: [f32; 4]| (r[0] + 5.0, r[1] + 5.0);
        let (x, y) = at(save_rect(o, s[0]));
        assert_eq!(hit_test(x, y, o, s, false), Some(SourceAction::Save));
        let (x, y) = at(status_rect(o, s[0]));
        assert_eq!(hit_test(x, y, o, s, false), Some(SourceAction::Status));
        let (x, y) = at(area_rect(o, s, false));
        assert_eq!(hit_test(x, y, o, s, false), Some(SourceAction::Text));
        assert_eq!(
            hit_test(o[0] + 2.0, o[1] + widget::TITLE_H + 4.0, o, s, false),
            Some(SourceAction::Consume)
        );
        assert_eq!(hit_test(5000.0, 5000.0, o, s, false), None);
    }

    #[test]
    fn unsaved_edits_mark_the_title_and_header() {
        let mut world = injected_world();
        let mut area = TextArea::from_text("float4 shade() {}");
        place(
            &mut world,
            Some(&view(&area, None, &[])),
            [20.0, 20.0],
            size(false),
            M,
        );
        assert_eq!(label(&world, TITLE_LABEL).content, "water fragment");
        assert!(!label(&world, DIRTY_LABEL).visible);
        area.type_char('x');
        place(
            &mut world,
            Some(&view(&area, None, &[])),
            [20.0, 20.0],
            size(false),
            M,
        );
        assert_eq!(label(&world, TITLE_LABEL).content, "water fragment *");
        assert!(label(&world, DIRTY_LABEL).visible);
    }

    #[test]
    fn the_status_line_takes_its_tone() {
        let mut world = injected_world();
        let area = TextArea::from_text("a");
        let status = Status::new("line 1: bad", Tone::Error);
        place(
            &mut world,
            Some(&view(&area, Some(&status), &[])),
            [20.0, 20.0],
            size(false),
            M,
        );
        let l = label(&world, STATUS_LABEL);
        assert!(l.visible && l.content == "line 1: bad");
        assert_eq!(l.color, theme::LOG_ERROR);
    }

    // Hovering a marker in the gutter reads it out on the status line.
    #[test]
    fn a_hovered_marker_reads_out_on_the_status_line() {
        let mut world = injected_world();
        let area = TextArea::from_text("a\nb\nc");
        let markers = [GutterMarker {
            line: 1,
            column: 0,
            severity: Severity::Warning,
            message: "unused".to_string(),
        }];
        let o = [20.0, 20.0];
        let r = area_rect(o, size(false), false);
        let mut v = view(&area, None, &markers);
        v.mouse = [r[0] + 2.0, r[1] + M.line_h * 1.5];
        place(&mut world, Some(&v), o, size(false), M);
        let l = label(&world, STATUS_LABEL);
        assert_eq!(l.content, "line 2: unused");
        assert_eq!(l.color, theme::LOG_WARN);
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        let area = TextArea::from_text("a");
        place(
            &mut world,
            Some(&view(&area, None, &[])),
            [20.0, 20.0],
            size(false),
            M,
        );
        place(&mut world, None, [0.0, 0.0], size(false), M);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
    }

    fn reference_rows() -> Vec<RefRow<'static>> {
        Reference::default().rows(ENTRIES)
    }

    #[test]
    fn the_reference_column_widens_the_panel_beside_the_text() {
        let o = [20.0, 20.0];
        let s = size(true);
        assert_eq!(s[0], size(false)[0] + COLUMN_W + PAD);
        let text = area_rect(o, s, true);
        let column = reference_rect(o, s);
        assert_eq!(
            text[2],
            area_rect(o, size(false), false)[2],
            "the text keeps its width"
        );
        assert!(
            text[0] + text[2] <= column[0],
            "the column sits right of the text"
        );
        assert_eq!((column[1], column[3]), (text[1], text[3]));
    }

    #[test]
    fn hit_test_resolves_the_toggle_and_reference_rows() {
        let o = [40.0, 40.0];
        let s = size(true);
        let t = reference_button_rect(o, s[0]);
        assert_eq!(
            hit_test(t[0] + 2.0, t[1] + 2.0, o, s, true),
            Some(SourceAction::ToggleReference)
        );
        let c = reference_rect(o, s);
        assert_eq!(
            hit_test(c[0] + 5.0, c[1] + 30.0, o, s, true),
            Some(SourceAction::Reference(1))
        );
        assert_eq!(
            hit_test(c[0] + 5.0, c[1] + 30.0, o, s, false),
            Some(SourceAction::Text),
            "a hidden column's room is the text's"
        );
    }

    #[test]
    fn a_hovered_reference_name_reads_out_on_the_status_line() {
        let mut world = injected_world();
        let area = TextArea::from_text("a");
        let rows = reference_rows();
        let o = [20.0, 20.0];
        let s = size(true);
        let c = reference_rect(o, s);
        let status = Status::new("compiled", Tone::Info);
        let mut v = view(&area, Some(&status), &[]);
        let mouse = [c[0] + 5.0, c[1] + 30.0];
        v.mouse = mouse;
        v.reference = Some(RefView {
            rows: &rows,
            scroll: 0,
            mouse,
        });
        place(&mut world, Some(&v), o, s, M);
        let l = label(&world, STATUS_LABEL);
        assert!(
            l.content
                .starts_with("float4 shade_surface(VertexOut v, GpuObjectData od): "),
            "{}",
            l.content
        );
        assert!(label(&world, REF_LABEL).visible);
    }
}

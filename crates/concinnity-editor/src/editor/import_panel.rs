// src/editor/import_panel.rs
//
// The Import panel: a front end over the same file resolution `cn add <file>`
// uses (`authoring/add.rs::entry_from_path`), so importing from the panel and
// from the CLI produce identical entries. The header takes a source path and
// an Add button (Enter also adds); a status line shows resolution errors; the
// body lists the world's existing file-backed entries (scene / story imports,
// textures, audio, fonts, shaders, ...) and clicking one opens it in the
// standard edit form. Layout half only; `hook/import_edit.rs` owns the actions.

use super::registry::{self, PanelKey};
use super::theme;
use super::widget::{self, place_rounded, point_in};
use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::Import);
pub(crate) const PANEL_BG: AssetId = AssetId(BASE);
pub(crate) const TITLE_LABEL: AssetId = AssetId(BASE + 2);
pub(crate) const CLOSE_BG: AssetId = AssetId(BASE + 3);
pub(crate) const CLOSE_LABEL: AssetId = AssetId(BASE + 4);
pub(crate) const ADD_BG: AssetId = AssetId(BASE + 5);
pub(crate) const ADD_LABEL: AssetId = AssetId(BASE + 6);
pub(crate) const HINT_LABEL: AssetId = AssetId(BASE + 7);
pub(crate) const STATUS_LABEL: AssetId = AssetId(BASE + 8);
pub(crate) const LIST_HEADER: AssetId = AssetId(BASE + 9);
pub(crate) const LIST_TRACK: AssetId = AssetId(BASE + 10);
pub(crate) const LIST_THUMB: AssetId = AssetId(BASE + 11);
pub(crate) const PATH_INPUT: AssetId = AssetId(BASE + 12);
pub(crate) const BROWSE_BG: AssetId = AssetId(BASE + 13);
pub(crate) const BROWSE_LABEL: AssetId = AssetId(BASE + 14);

pub(crate) fn row_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x20 + i as u32)
}
pub(crate) fn row_label(i: usize) -> AssetId {
    AssetId(BASE + 0x40 + i as u32)
}

// The file-backed asset types `entry_from_path` produces, in the order their
// entries list in the panel body. TextLabel is deliberately absent: a text
// import copies the file body into `content` and keeps no path to reopen.
pub(crate) const IMPORT_TYPES: &[&str] = &[
    "SceneImport",
    "StoryImport",
    "File",
    "EnvironmentMap",
    "AudioClip",
    "Font",
    "ShaderStage",
    "LLM",
];

// The arg carrying an import entry's file path (the key varies by type).
pub(crate) fn source_of(args: &serde_json::Value) -> String {
    for key in ["source", "path", "model_path"] {
        if let Some(s) = args.get(key).and_then(|v| v.as_str())
            && !s.is_empty()
        {
            return s.to_string();
        }
    }
    String::new()
}

// Geometry, in window pixels. Every rect derives from the panel origin `o`
// (the title bar's top-left), so dragging the title bar moves the whole panel.
pub(crate) const IMPORT_W: f32 = 520.0;
const PAD: f32 = 10.0;
const HEADER_H: f32 = 32.0;
const HINT_H: f32 = 20.0;
const STATUS_H: f32 = 22.0;
const LIST_HEADER_H: f32 = 24.0;
const BTN_W: f32 = 70.0;
const BROWSE_W: f32 = 84.0;
const GAP: f32 = 6.0;
const ROW_H: f32 = 26.0;
const SCROLLBAR_W: f32 = 5.0;
// Visible import rows; a longer list scrolls.
pub(crate) const IMPORT_POOL: usize = 10;
const MAX_ROW_CHARS: usize = 56;

const ROW_TINT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const ROW_TINT_HOVER: [f32; 4] = theme::HOVER_TINT;
const BTN_TINT: [f32; 4] = [0.20, 0.44, 0.30, 1.0];
const BTN_TINT_HOVER: [f32; 4] = [0.28, 0.56, 0.38, 1.0];
// The Browse button reads as secondary next to the green Add.
const BROWSE_TINT: [f32; 4] = theme::BUTTON_TINT;
const TRACK_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.9];
const THUMB_TINT: [f32; 4] = [0.40, 0.44, 0.56, 0.95];
const LABEL: [f32; 3] = [0.90, 0.90, 0.92];
const HINT_COLOR: [f32; 3] = [0.55, 0.58, 0.66];
const HEADER_LABEL: [f32; 3] = [0.70, 0.74, 0.84];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];
const NOTICE_LABEL: [f32; 3] = [0.55, 0.85, 0.65];

// The outcome of the last Add, shown on the status line: a resolution failure,
// or a note about what the Add did when it appended no new row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ImportStatus {
    Error(String),
    Notice(String),
}

impl ImportStatus {
    pub(crate) fn text(&self) -> &str {
        match self {
            Self::Error(s) | Self::Notice(s) => s,
        }
    }

    fn color(&self) -> [f32; 3] {
        match self {
            Self::Error(_) => ERROR_LABEL,
            Self::Notice(_) => NOTICE_LABEL,
        }
    }
}

// One listed import: its entry index and the "name (Type) source" caption.
pub(crate) struct ImportRow {
    pub entry: usize,
    pub caption: String,
}

// The per-frame view the hook assembles.
pub(crate) struct ImportView<'a> {
    pub rows: &'a [ImportRow],
    pub scroll: usize,
    // Whether the path field asserts keyboard focus this frame.
    pub focus: bool,
    // Outcome of the last Add.
    pub status: Option<&'a ImportStatus>,
    pub mouse: [f32; 2],
}

// A resolved Import-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImportAction {
    // Give the path field keyboard focus.
    FocusPath,
    // Open the native file picker and put the result in the path field.
    Browse,
    // Resolve the typed path and add its entries.
    Add,
    // Open listed import `i` (a row index into the view) in the edit form.
    Open(usize),
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Where the panel sits until the user drags it: centred below the top bar,
// offset from the Story panel's anchor so the two do not stack exactly.
pub(crate) fn default_origin(vw: f32) -> [f32; 2] {
    [(vw - IMPORT_W) * 0.5, super::hud::body_top() + 40.0]
}

// The panel's fixed footprint.
pub(crate) fn size() -> [f32; 2] {
    [
        IMPORT_W,
        widget::TITLE_H
            + HEADER_H
            + HINT_H
            + STATUS_H
            + LIST_HEADER_H
            + IMPORT_POOL as f32 * ROW_H
            + PAD,
    ]
}

fn panel_rect(o: [f32; 2]) -> [f32; 4] {
    let s = size();
    [o[0], o[1], s[0], s[1]]
}

fn title_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0], o[1], IMPORT_W, widget::TITLE_H]
}

// The header row's controls, right to left: Add pinned to the right end, the
// Browse button beside it, and the path field filling the rest.
fn header_y(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H + 4.0
}
const CTRL_H: f32 = HEADER_H - 8.0;

pub(crate) fn add_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0] + IMPORT_W - PAD - BTN_W, header_y(o), BTN_W, CTRL_H]
}

// The "Browse..." button, opening the native file picker.
pub(crate) fn browse_rect(o: [f32; 2]) -> [f32; 4] {
    [
        add_rect(o)[0] - GAP - BROWSE_W,
        header_y(o),
        BROWSE_W,
        CTRL_H,
    ]
}

pub(crate) fn path_rect(o: [f32; 2]) -> [f32; 4] {
    let left = o[0] + PAD;
    [
        left,
        header_y(o),
        (browse_rect(o)[0] - GAP - left).max(0.0),
        CTRL_H,
    ]
}

fn list_top(o: [f32; 2]) -> f32 {
    o[1] + widget::TITLE_H + HEADER_H + HINT_H + STATUS_H + LIST_HEADER_H
}

// Visible import row `slot` (0-based within the window).
pub(crate) fn row_rect(o: [f32; 2], slot: usize) -> [f32; 4] {
    [
        o[0],
        list_top(o) + slot as f32 * ROW_H,
        IMPORT_W - SCROLLBAR_W - 2.0,
        ROW_H,
    ]
}

// Whether the cursor is over the scrollable import list (for wheel routing).
pub(crate) fn cursor_over_list(mx: f32, my: f32, o: [f32; 2]) -> bool {
    let p = panel_rect(o);
    mx >= p[0] && mx < p[0] + p[2] && my >= list_top(o) && my < p[1] + p[3]
}

// Resolve a click at `(mx, my)` against the panel at origin `o`. `None` means
// the click missed the panel. Title-bar presses never reach this: the shared
// routing intercepts them first.
pub(crate) fn hit_test(view: &ImportView, mx: f32, my: f32, o: [f32; 2]) -> Option<ImportAction> {
    if point_in(mx, my, add_rect(o)) {
        return Some(ImportAction::Add);
    }
    if point_in(mx, my, browse_rect(o)) {
        return Some(ImportAction::Browse);
    }
    if point_in(mx, my, path_rect(o)) {
        return Some(ImportAction::FocusPath);
    }
    for slot in 0..IMPORT_POOL {
        if !point_in(mx, my, row_rect(o, slot)) {
            continue;
        }
        let i = view.scroll + slot;
        return Some(if i < view.rows.len() {
            ImportAction::Open(i)
        } else {
            ImportAction::Consume
        });
    }
    point_in(mx, my, panel_rect(o)).then_some(ImportAction::Consume)
}

// Position + show the panel (`Some(view)`), or blank every element (`None`).
pub(crate) fn apply(world: &mut World, view: Option<&ImportView>, o: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };
    widget::place_panel(world, PANEL_BG, panel_rect(o));
    let title = title_rect(o);
    widget::place_heading(world, TITLE_LABEL, title, "Import");
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);

    widget::show_field(world, PATH_INPUT, path_rect(o), view.focus);
    let browse = browse_rect(o);
    let hover = point_in(view.mouse[0], view.mouse[1], browse);
    let tint = if hover {
        theme::HOVER_TINT
    } else {
        BROWSE_TINT
    };
    place_rounded(world, BROWSE_BG, browse, tint, theme::CONTROL_RADIUS, true);
    if let Some(l) = widget::label_mut(world, BROWSE_LABEL) {
        l.x = browse[0] + browse[2] * 0.5;
        l.y = browse[1] + browse[3] * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Center;
        l.color = [1.0, 1.0, 1.0];
        l.visible = true;
        l.content = "Browse...".to_string();
    }
    let add = add_rect(o);
    let hover = point_in(view.mouse[0], view.mouse[1], add);
    let tint = if hover { BTN_TINT_HOVER } else { BTN_TINT };
    place_rounded(world, ADD_BG, add, tint, theme::CONTROL_RADIUS, true);
    if let Some(l) = widget::label_mut(world, ADD_LABEL) {
        l.x = add[0] + add[2] * 0.5;
        l.y = add[1] + add[3] * 0.5 - theme::TEXT_HALF;
        l.align = TextAlign::Center;
        l.color = [1.0, 1.0, 1.0];
        l.visible = true;
        l.content = "Add".to_string();
    }
    if let Some(l) = widget::label_mut(world, HINT_LABEL) {
        l.x = o[0] + PAD;
        l.y = o[1] + widget::TITLE_H + HEADER_H;
        l.align = TextAlign::Left;
        l.color = HINT_COLOR;
        l.visible = true;
        l.content =
            "scenes (glb, fbx) - stories (md) - images - hdr - audio - fonts - shaders - text"
                .to_string();
    }
    if let Some(l) = widget::label_mut(world, STATUS_LABEL) {
        l.x = o[0] + PAD;
        l.y = o[1] + widget::TITLE_H + HEADER_H + HINT_H;
        l.align = TextAlign::Left;
        l.color = view.status.map_or(ERROR_LABEL, ImportStatus::color);
        l.visible = view.status.is_some();
        l.content = view
            .status
            .map(ImportStatus::text)
            .unwrap_or("")
            .to_string();
    }
    if let Some(l) = widget::label_mut(world, LIST_HEADER) {
        l.x = o[0] + PAD;
        l.y = o[1] + widget::TITLE_H + HEADER_H + HINT_H + STATUS_H + 2.0;
        l.align = TextAlign::Left;
        l.color = HEADER_LABEL;
        l.visible = true;
        l.content = if view.rows.is_empty() {
            "Imports (none yet)".to_string()
        } else {
            format!("Imports ({})", view.rows.len())
        };
    }

    for slot in 0..IMPORT_POOL {
        let i = view.scroll + slot;
        let r = row_rect(o, slot);
        if i >= view.rows.len() {
            widget::set_sprite_visible(world, row_bg(slot), false);
            widget::set_label_visible(world, row_label(slot), false);
            continue;
        }
        let hovered = point_in(view.mouse[0], view.mouse[1], r);
        let tint = if hovered { ROW_TINT_HOVER } else { ROW_TINT };
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
            l.color = LABEL;
            l.visible = true;
            l.content = widget::clip_text(&view.rows[i].caption, MAX_ROW_CHARS);
        }
    }

    layout_scrollbar(world, view, o);
}

fn layout_scrollbar(world: &mut World, view: &ImportView, o: [f32; 2]) {
    let total = view.rows.len();
    if total <= IMPORT_POOL {
        widget::set_sprite_visible(world, LIST_TRACK, false);
        widget::set_sprite_visible(world, LIST_THUMB, false);
        return;
    }
    let x = o[0] + IMPORT_W - SCROLLBAR_W;
    let top = list_top(o);
    let h = IMPORT_POOL as f32 * ROW_H;
    place_rounded(
        world,
        LIST_TRACK,
        [x, top, SCROLLBAR_W, h],
        TRACK_TINT,
        SCROLLBAR_W * 0.5,
        true,
    );
    let thumb_h = (h * IMPORT_POOL as f32 / total as f32).max(18.0);
    let max_scroll = (total - IMPORT_POOL) as f32;
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

// Hide every panel element, blurring the path field so a hidden field cannot
// keep keyboard focus.
pub(crate) fn hide_all(world: &mut World) {
    for id in all_sprite_ids() {
        widget::set_sprite_visible(world, id, false);
    }
    for id in all_label_ids() {
        widget::set_label_visible(world, id, false);
    }
    widget::hide_field(world, PATH_INPUT);
}

// Every panel sprite id, in draw (insertion) order: chrome, then the list
// rows, then the scrollbar floating above them.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, CLOSE_BG, BROWSE_BG, ADD_BG];
    ids.extend((0..IMPORT_POOL).map(row_bg));
    ids.extend([LIST_TRACK, LIST_THUMB]);
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![
        TITLE_LABEL,
        CLOSE_LABEL,
        BROWSE_LABEL,
        ADD_LABEL,
        HINT_LABEL,
        STATUS_LABEL,
        LIST_HEADER,
    ];
    ids.extend((0..IMPORT_POOL).map(row_label));
    ids
}

pub(crate) fn all_field_ids() -> Vec<AssetId> {
    vec![PATH_INPUT]
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

    fn rows(n: usize) -> Vec<ImportRow> {
        (0..n)
            .map(|i| ImportRow {
                entry: i,
                caption: format!("import {i}"),
            })
            .collect()
    }

    fn view<'a>(rows: &'a [ImportRow], scroll: usize) -> ImportView<'a> {
        ImportView {
            rows,
            scroll,
            focus: true,
            status: None,
            mouse: [0.0, 0.0],
        }
    }

    #[test]
    fn source_of_reads_the_per_type_path_key() {
        assert_eq!(source_of(&serde_json::json!({"source": "a.glb"})), "a.glb");
        assert_eq!(source_of(&serde_json::json!({"path": "f.ttf"})), "f.ttf");
        assert_eq!(
            source_of(&serde_json::json!({"model_path": "m.gguf"})),
            "m.gguf"
        );
        assert_eq!(source_of(&serde_json::json!({"other": 1})), "");
    }

    // The header packs right to left without overlapping, and the path field
    // keeps a usable width beside the two buttons.
    #[test]
    fn header_controls_pack_without_overlap() {
        let o = [40.0, 40.0];
        let (p, b, a) = (path_rect(o), browse_rect(o), add_rect(o));
        assert_eq!(a[0] + a[2], o[0] + IMPORT_W - PAD, "Add pinned right");
        assert_eq!(b[0] + b[2] + GAP, a[0], "Browse sits left of Add");
        assert_eq!(p[0] + p[2] + GAP, b[0], "the field fills up to Browse");
        assert!(p[2] > 200.0, "the path field stays usable: {}", p[2]);
        for r in [p, b, a] {
            assert_eq!(r[1], header_y(o));
            assert_eq!(r[3], CTRL_H);
        }
    }

    #[test]
    fn hit_test_resolves_path_browse_add_and_rows() {
        let r = rows(3);
        let v = view(&r, 0);
        let o = [40.0, 40.0];
        let a = add_rect(o);
        assert_eq!(
            hit_test(&v, a[0] + 5.0, a[1] + 5.0, o),
            Some(ImportAction::Add)
        );
        let b = browse_rect(o);
        assert_eq!(
            hit_test(&v, b[0] + 5.0, b[1] + 5.0, o),
            Some(ImportAction::Browse)
        );
        let p = path_rect(o);
        assert_eq!(
            hit_test(&v, p[0] + 5.0, p[1] + 5.0, o),
            Some(ImportAction::FocusPath)
        );
        let r1 = row_rect(o, 1);
        assert_eq!(
            hit_test(&v, r1[0] + 5.0, r1[1] + 5.0, o),
            Some(ImportAction::Open(1))
        );
        let r5 = row_rect(o, 5);
        assert_eq!(
            hit_test(&v, r5[0] + 5.0, r5[1] + 5.0, o),
            Some(ImportAction::Consume),
            "past the last row the click is swallowed"
        );
        assert_eq!(hit_test(&v, 5000.0, 5000.0, o), None);
    }

    #[test]
    fn apply_draws_header_rows_and_empty_list_note() {
        let mut world = injected_world();
        let r = rows(2);
        apply(&mut world, Some(&view(&r, 0)), [20.0, 20.0]);
        let title = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == TITLE_LABEL)
            .unwrap();
        assert!(title.visible && title.content == "Import");
        let header = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == LIST_HEADER)
            .unwrap();
        assert_eq!(header.content, "Imports (2)");
        let first = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == row_label(0))
            .unwrap();
        assert!(first.visible && first.content == "import 0");
        let input = world
            .query::<TextInput>()
            .find(|t| t.asset_id == PATH_INPUT)
            .unwrap();
        assert!(input.visible && input.focused);
        let browse = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == BROWSE_LABEL)
            .unwrap();
        assert!(browse.visible && browse.content == "Browse...");
        // An empty list keeps the header but no rows.
        apply(&mut world, Some(&view(&[], 0)), [20.0, 20.0]);
        let header = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == LIST_HEADER)
            .unwrap();
        assert_eq!(header.content, "Imports (none yet)");
        assert!(
            !world
                .query::<Sprite>()
                .find(|s| s.asset_id == row_bg(0))
                .unwrap()
                .visible
        );
    }

    #[test]
    fn long_list_shows_the_scrollbar_and_maps_scrolled_rows() {
        let mut world = injected_world();
        let r = rows(30);
        let v = view(&r, 10);
        apply(&mut world, Some(&v), [20.0, 20.0]);
        assert!(
            world
                .query::<Sprite>()
                .find(|s| s.asset_id == LIST_THUMB)
                .unwrap()
                .visible
        );
        let o = [20.0, 20.0];
        let r0 = row_rect(o, 0);
        assert_eq!(
            hit_test(&v, r0[0] + 5.0, r0[1] + 5.0, o),
            Some(ImportAction::Open(10))
        );
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        let r = rows(2);
        apply(&mut world, Some(&view(&r, 0)), [20.0, 20.0]);
        apply(&mut world, None, [0.0, 0.0]);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
        assert!(world.query::<TextInput>().all(|t| !t.visible));
    }
}

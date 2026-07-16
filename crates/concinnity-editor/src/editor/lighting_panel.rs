// src/editor/lighting_panel.rs
//
// The Lighting panel's layout half: a floating panel of themed sections (see
// `lighting.rs` for the bindings) with a status line + Apply button in its
// header row. Each field row is a caption plus one control -- a text input, a
// checkbox, or a text input with a colour swatch -- backed by a per-binding
// control pool at reserved ids. Like the rest of the editor HUD it is plain
// `Sprite` / `TextLabel` / `TextInput` components driven each frame by the
// editor hook; the hook owns seeding, focus, and the commit path.

use super::form::{FieldKind, FormField};
use super::lighting::{self, Row};
use super::registry::{self, PanelKey};
use super::widget::{self, place_sprite, point_in};
use crate::assets::TextAlign;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;

const BASE: u32 = registry::base(PanelKey::Lighting);
pub(crate) const PANEL_BG: AssetId = AssetId(BASE);
pub(crate) const TITLE_BG: AssetId = AssetId(BASE + 1);
pub(crate) const TITLE_LABEL: AssetId = AssetId(BASE + 2);
pub(crate) const CLOSE_BG: AssetId = AssetId(BASE + 3);
pub(crate) const CLOSE_LABEL: AssetId = AssetId(BASE + 4);
pub(crate) const APPLY_BG: AssetId = AssetId(BASE + 5);
pub(crate) const APPLY_LABEL: AssetId = AssetId(BASE + 6);
pub(crate) const STATUS_LABEL: AssetId = AssetId(BASE + 7);

// Per-row chrome (row index) and per-binding controls (global binding index);
// the sub-ranges stay disjoint for any plausible row / binding count.
pub(crate) fn row_bg(i: usize) -> AssetId {
    AssetId(BASE + 0x20 + i as u32)
}
pub(crate) fn row_label(i: usize) -> AssetId {
    AssetId(BASE + 0x40 + i as u32)
}
pub(crate) fn input(b: usize) -> AssetId {
    AssetId(BASE + 0x60 + b as u32)
}
pub(crate) fn check_bg(b: usize) -> AssetId {
    AssetId(BASE + 0x80 + b as u32)
}
pub(crate) fn swatch(b: usize) -> AssetId {
    AssetId(BASE + 0xA0 + b as u32)
}

// Geometry, in window pixels. Every rect derives from the panel origin `o` (the
// title bar's top-left), so dragging the title bar moves the whole panel.
pub(crate) const LIGHT_W: f32 = 380.0;
const PAD: f32 = 10.0;
// The header row under the title bar: the status line and the Apply button.
const HEADER_H: f32 = 32.0;
const BTN_W: f32 = 70.0;
const ROW_H: f32 = 28.0;
const LABEL_COL: f32 = 160.0;
const CHECK_SIZE: f32 = 18.0;
const SWATCH_W: f32 = 24.0;
const GAP: f32 = 6.0;

const PANEL_TINT: [f32; 4] = [0.09, 0.09, 0.12, 0.97];
const HEADER_ROW_TINT: [f32; 4] = [0.14, 0.16, 0.22, 1.0];
const ROW_TINT: [f32; 4] = [0.12, 0.12, 0.15, 0.92];
const ROW_TINT_HOVER: [f32; 4] = [0.22, 0.26, 0.36, 0.98];
const BTN_TINT: [f32; 4] = [0.22, 0.40, 0.56, 1.0];
const BTN_TINT_HOVER: [f32; 4] = [0.24, 0.28, 0.40, 1.0];
const CHECK_ON: [f32; 4] = [0.30, 0.66, 0.34, 1.0];
const CHECK_OFF: [f32; 4] = [0.30, 0.30, 0.34, 1.0];
const LABEL: [f32; 3] = [0.90, 0.90, 0.92];
const HEADER_LABEL: [f32; 3] = [0.70, 0.74, 0.84];
const ADD_LABEL: [f32; 3] = [0.55, 0.80, 0.60];
const ERROR_LABEL: [f32; 3] = [0.95, 0.55, 0.55];
const LABEL_TOP: f32 = ROW_H * 0.5 - 10.0;

// The per-frame view the hook assembles: the row list, the per-binding derived
// fields (`None` while a section's asset is absent), the focused binding, and
// the status line from the last rejected Apply.
pub(crate) struct LightingView<'a> {
    pub rows: &'a [Row],
    pub fields: &'a [Option<FormField>],
    pub focus: Option<usize>,
    pub status: Option<&'a str>,
    pub mouse: [f32; 2],
}

// A resolved Lighting-panel click.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LightingAction {
    // Give binding `b`'s text input keyboard focus.
    Focus(usize),
    // Flip binding `b`'s checkbox (committed immediately by the hook).
    Toggle(usize),
    // Add the missing singleton asset of section `s`.
    Add(usize),
    // Capture every text control, validate per asset, and commit.
    Apply,
    // A click elsewhere on the panel: swallowed so it cannot reach the world.
    Consume,
}

// Where the panel sits until the user drags it: the window's left edge, below
// the View panel's default anchor.
pub(crate) fn default_origin() -> [f32; 2] {
    let view = super::view::default_origin();
    [8.0, view[1] + super::view::size()[1] + 8.0]
}

// The panel footprint for `n_rows` rows, for the hook's drag clamp.
pub(crate) fn size(n_rows: usize) -> [f32; 2] {
    [
        LIGHT_W,
        widget::TITLE_H + HEADER_H + n_rows as f32 * ROW_H + PAD,
    ]
}

fn panel_rect(o: [f32; 2], n_rows: usize) -> [f32; 4] {
    let s = size(n_rows);
    [o[0], o[1], s[0], s[1]]
}

fn title_rect(o: [f32; 2]) -> [f32; 4] {
    [o[0], o[1], LIGHT_W, widget::TITLE_H]
}

// The Apply button, pinned to the header row's right end.
pub(crate) fn apply_rect(o: [f32; 2]) -> [f32; 4] {
    [
        o[0] + LIGHT_W - PAD - BTN_W,
        o[1] + widget::TITLE_H + 4.0,
        BTN_W,
        HEADER_H - 8.0,
    ]
}

// Row `i`, stacked below the header row.
pub(crate) fn row_rect(o: [f32; 2], i: usize) -> [f32; 4] {
    [
        o[0],
        o[1] + widget::TITLE_H + HEADER_H + i as f32 * ROW_H,
        LIGHT_W,
        ROW_H,
    ]
}

// The control area on a field row's right side; a colour field's input is
// narrowed to leave room for its swatch.
fn control_rect(row: [f32; 4], color: bool) -> [f32; 4] {
    let x = row[0] + LABEL_COL;
    let w = row[0] + row[2] - PAD - x - if color { SWATCH_W + GAP } else { 0.0 };
    [x, row[1] + 3.0, w, row[3] - 6.0]
}

fn swatch_rect(row: [f32; 4]) -> [f32; 4] {
    [
        row[0] + row[2] - PAD - SWATCH_W,
        row[1] + 4.0,
        SWATCH_W,
        row[3] - 8.0,
    ]
}

fn check_rect(row: [f32; 4]) -> [f32; 4] {
    [
        row[0] + LABEL_COL,
        row[1] + (ROW_H - CHECK_SIZE) * 0.5,
        CHECK_SIZE,
        CHECK_SIZE,
    ]
}

// A colour field's swatch tint, parsed from its comma-separated text (3 or 4
// components); an unparseable value shows white rather than stale colour.
fn parse_color(text: &str) -> [f32; 4] {
    let parts: Vec<f32> = text
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect();
    match parts.len() {
        3 => [parts[0], parts[1], parts[2], 1.0],
        4 => [parts[0], parts[1], parts[2], parts[3]],
        _ => [1.0, 1.0, 1.0, 1.0],
    }
}

// Resolve a click at `(mx, my)` against the panel at origin `o`. `None` means
// the click missed the panel. Title-bar presses never reach this: the shared
// routing intercepts them first.
pub(crate) fn hit_test(
    view: &LightingView,
    mx: f32,
    my: f32,
    o: [f32; 2],
) -> Option<LightingAction> {
    if point_in(mx, my, apply_rect(o)) {
        return Some(LightingAction::Apply);
    }
    for (i, row) in view.rows.iter().enumerate() {
        if !point_in(mx, my, row_rect(o, i)) {
            continue;
        }
        return Some(match *row {
            Row::Header(_) => LightingAction::Consume,
            Row::Add(s) => LightingAction::Add(s),
            Row::Field(b) => match view.fields.get(b).and_then(|f| f.as_ref()) {
                Some(f) if f.kind == FieldKind::Bool => LightingAction::Toggle(b),
                Some(_) => LightingAction::Focus(b),
                None => LightingAction::Consume,
            },
        });
    }
    point_in(mx, my, panel_rect(o, view.rows.len())).then_some(LightingAction::Consume)
}

// Position + show the panel (`Some(view)`), or blank every element (`None`).
pub(crate) fn apply(world: &mut World, view: Option<&LightingView>, o: [f32; 2]) {
    let Some(view) = view else {
        hide_all(world);
        return;
    };
    let n = view.rows.len();
    place_sprite(world, PANEL_BG, panel_rect(o, n), PANEL_TINT, true);
    let title = title_rect(o);
    widget::place_title(world, TITLE_BG, TITLE_LABEL, title, "Lighting");
    let close_hover = point_in(view.mouse[0], view.mouse[1], widget::close_rect(title));
    widget::place_close(world, CLOSE_BG, CLOSE_LABEL, title, close_hover);

    // Header row: the status line (only after a rejected Apply) + the button.
    let apply = apply_rect(o);
    let apply_hover = point_in(view.mouse[0], view.mouse[1], apply);
    let tint = if apply_hover {
        BTN_TINT_HOVER
    } else {
        BTN_TINT
    };
    place_sprite(world, APPLY_BG, apply, tint, true);
    if let Some(l) = widget::label_mut(world, APPLY_LABEL) {
        l.x = apply[0] + apply[2] * 0.5;
        l.y = apply[1] + apply[3] * 0.5 - 10.0;
        l.align = TextAlign::Center;
        l.color = [1.0, 1.0, 1.0];
        l.visible = true;
        l.content = "Apply".to_string();
    }
    if let Some(l) = widget::label_mut(world, STATUS_LABEL) {
        l.x = o[0] + PAD;
        l.y = o[1] + widget::TITLE_H + HEADER_H * 0.5 - 10.0;
        l.align = TextAlign::Left;
        l.color = ERROR_LABEL;
        l.visible = view.status.is_some();
        l.content = view.status.unwrap_or("").to_string();
    }

    // Every pooled control starts hidden; the row pass below shows what this
    // frame's rows actually use (sections toggle between fields and an add row).
    for b in 0..lighting::binding_count() {
        widget::hide_field(world, input(b));
        widget::set_sprite_visible(world, check_bg(b), false);
        widget::set_sprite_visible(world, swatch(b), false);
    }

    for (i, row) in view.rows.iter().enumerate() {
        let r = row_rect(o, i);
        let hovered = point_in(view.mouse[0], view.mouse[1], r);
        let (bg, caption, color) = match *row {
            Row::Header(s) => (
                HEADER_ROW_TINT,
                lighting::SECTIONS[s].title.to_string(),
                HEADER_LABEL,
            ),
            Row::Add(s) => (
                if hovered { ROW_TINT_HOVER } else { ROW_TINT },
                format!("+ Add {}", lighting::SECTIONS[s].ty),
                ADD_LABEL,
            ),
            Row::Field(b) => {
                let caption = lighting::binding(b).2.to_string();
                (ROW_TINT, caption, LABEL)
            }
        };
        place_sprite(world, row_bg(i), r, bg, true);
        if let Some(l) = widget::label_mut(world, row_label(i)) {
            l.x = r[0] + PAD;
            l.y = r[1] + LABEL_TOP;
            l.align = TextAlign::Left;
            l.color = color;
            l.visible = true;
            l.content = caption;
        }
        let Row::Field(b) = *row else { continue };
        let Some(field) = view.fields.get(b).and_then(|f| f.as_ref()) else {
            continue;
        };
        match field.kind {
            FieldKind::Bool => {
                let tint = if field.boolval { CHECK_ON } else { CHECK_OFF };
                place_sprite(world, check_bg(b), check_rect(r), tint, true);
            }
            FieldKind::Vec { color: true, .. } => {
                widget::show_field(
                    world,
                    input(b),
                    control_rect(r, true),
                    view.focus == Some(b),
                );
                place_sprite(
                    world,
                    swatch(b),
                    swatch_rect(r),
                    parse_color(&field.initial),
                    true,
                );
            }
            _ => {
                widget::show_field(
                    world,
                    input(b),
                    control_rect(r, false),
                    view.focus == Some(b),
                );
            }
        }
    }

    // Blank any pooled row chrome past this frame's row count.
    for i in n..max_rows() {
        widget::set_sprite_visible(world, row_bg(i), false);
        widget::set_label_visible(world, row_label(i), false);
    }
}

// The largest row count the panel can show: every section header plus every
// binding (an absent section's add row never exceeds its field rows).
pub(crate) fn max_rows() -> usize {
    lighting::SECTIONS.len() + lighting::binding_count()
}

// Hide every panel element, blurring the typed fields so a hidden field cannot
// keep keyboard focus.
pub(crate) fn hide_all(world: &mut World) {
    for id in all_sprite_ids() {
        widget::set_sprite_visible(world, id, false);
    }
    for id in all_label_ids() {
        widget::set_label_visible(world, id, false);
    }
    for id in all_field_ids() {
        widget::hide_field(world, id);
    }
}

// Every panel sprite id, in draw (insertion) order: chrome first, then the row
// backgrounds, then the floating per-binding controls above them.
pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut ids = vec![PANEL_BG, TITLE_BG, CLOSE_BG, APPLY_BG];
    ids.extend((0..max_rows()).map(row_bg));
    ids.extend((0..lighting::binding_count()).map(check_bg));
    ids.extend((0..lighting::binding_count()).map(swatch));
    ids
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    let mut ids = vec![TITLE_LABEL, CLOSE_LABEL, APPLY_LABEL, STATUS_LABEL];
    ids.extend((0..max_rows()).map(row_label));
    ids
}

pub(crate) fn all_field_ids() -> Vec<AssetId> {
    (0..lighting::binding_count()).map(input).collect()
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

    fn all_present_view<'a>(
        rows: &'a [Row],
        fields: &'a [Option<super::FormField>],
    ) -> LightingView<'a> {
        LightingView {
            rows,
            fields,
            focus: None,
            status: None,
            mouse: [0.0, 0.0],
        }
    }

    fn derived_fields() -> Vec<Option<super::FormField>> {
        lighting::SECTIONS
            .iter()
            .flat_map(|s| lighting::section_fields(s, None).into_iter().map(Some))
            .collect()
    }

    #[test]
    fn hit_test_resolves_each_control() {
        let rows = lighting::rows(&[true, true, true, true]);
        let fields = derived_fields();
        let view = all_present_view(&rows, &fields);
        let o = [40.0, 40.0];
        let a = apply_rect(o);
        assert_eq!(
            hit_test(&view, a[0] + 5.0, a[1] + 5.0, o),
            Some(LightingAction::Apply)
        );
        // Row 0 is the Sun header; row 1 its first (float) field.
        let r0 = row_rect(o, 0);
        assert_eq!(
            hit_test(&view, r0[0] + 5.0, r0[1] + 5.0, o),
            Some(LightingAction::Consume)
        );
        let r1 = row_rect(o, 1);
        assert_eq!(
            hit_test(&view, r1[0] + 5.0, r1[1] + 5.0, o),
            Some(LightingAction::Focus(0))
        );
        // The fog `enabled` row toggles rather than focusing.
        let fog_enabled = lighting::section_base(1);
        let i = rows
            .iter()
            .position(|r| *r == Row::Field(fog_enabled))
            .unwrap();
        let rf = row_rect(o, i);
        assert_eq!(
            hit_test(&view, rf[0] + 5.0, rf[1] + 5.0, o),
            Some(LightingAction::Toggle(fog_enabled))
        );
        assert_eq!(hit_test(&view, 5000.0, 5000.0, o), None);
    }

    #[test]
    fn hit_test_add_row_targets_the_missing_section() {
        let rows = lighting::rows(&[true, false, true, true]);
        let fields = derived_fields();
        let view = all_present_view(&rows, &fields);
        let o = [0.0, 0.0];
        let i = rows.iter().position(|r| *r == Row::Add(1)).unwrap();
        let r = row_rect(o, i);
        assert_eq!(
            hit_test(&view, r[0] + 5.0, r[1] + 5.0, o),
            Some(LightingAction::Add(1))
        );
    }

    #[test]
    fn apply_draws_sections_controls_and_swatch() {
        let mut world = injected_world();
        let rows = lighting::rows(&[true, true, true, true]);
        let fields = derived_fields();
        let view = all_present_view(&rows, &fields);
        let o = [20.0, 20.0];
        apply(&mut world, Some(&view), o);
        let title = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == TITLE_LABEL)
            .unwrap();
        assert!(title.visible && title.content == "Lighting");
        let header = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == row_label(0))
            .unwrap();
        assert_eq!(header.content, "Sun");
        // The sun colour binding shows a swatch tinted by its current value; the
        // fog toggle shows a checkbox; the intensity field shows a text input.
        let sun_color = 3;
        assert!(
            world
                .query::<Sprite>()
                .find(|s| s.asset_id == swatch(sun_color))
                .unwrap()
                .visible
        );
        let fog_enabled = lighting::section_base(1);
        assert!(
            world
                .query::<Sprite>()
                .find(|s| s.asset_id == check_bg(fog_enabled))
                .unwrap()
                .visible
        );
        assert!(
            world
                .query::<TextInput>()
                .find(|t| t.asset_id == input(4))
                .unwrap()
                .visible,
            "sun intensity input shown"
        );
        assert!(
            !world
                .query::<TextInput>()
                .find(|t| t.asset_id == input(fog_enabled))
                .unwrap()
                .visible,
            "a bool binding has no text input"
        );
        // The status line is hidden until an Apply is rejected.
        assert!(
            !world
                .query::<TextLabel>()
                .find(|l| l.asset_id == STATUS_LABEL)
                .unwrap()
                .visible
        );
    }

    #[test]
    fn absent_section_hides_its_controls_and_shows_the_add_row() {
        let mut world = injected_world();
        let rows_all = lighting::rows(&[true, true, true, true]);
        let fields_all = derived_fields();
        apply(
            &mut world,
            Some(&all_present_view(&rows_all, &fields_all)),
            [20.0, 20.0],
        );
        // Re-draw with fog absent: its controls blank, its add row appears.
        let rows = lighting::rows(&[true, false, true, true]);
        let mut fields = derived_fields();
        let base = lighting::section_base(1);
        for b in base..base + lighting::SECTIONS[1].fields.len() {
            fields[b] = None;
        }
        apply(
            &mut world,
            Some(&all_present_view(&rows, &fields)),
            [20.0, 20.0],
        );
        let fog_enabled = lighting::section_base(1);
        assert!(
            !world
                .query::<Sprite>()
                .find(|s| s.asset_id == check_bg(fog_enabled))
                .unwrap()
                .visible
        );
        let i = rows.iter().position(|r| *r == Row::Add(1)).unwrap();
        let add = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == row_label(i))
            .unwrap();
        assert_eq!(add.content, "+ Add VolumetricFog");
    }

    #[test]
    fn hide_all_blanks_every_element() {
        let mut world = injected_world();
        let rows = lighting::rows(&[true, true, true, true]);
        let fields = derived_fields();
        apply(
            &mut world,
            Some(&all_present_view(&rows, &fields)),
            [20.0, 20.0],
        );
        apply(&mut world, None, [0.0, 0.0]);
        assert!(world.query::<Sprite>().all(|s| !s.visible));
        assert!(world.query::<TextLabel>().all(|l| !l.visible));
        assert!(world.query::<TextInput>().all(|t| !t.visible));
    }

    #[test]
    fn parse_color_reads_rgb_and_rgba() {
        assert_eq!(parse_color("1, 0.5, 0"), [1.0, 0.5, 0.0, 1.0]);
        assert_eq!(parse_color("0 0 0 0.5"), [0.0, 0.0, 0.0, 0.5]);
        assert_eq!(parse_color("nonsense"), [1.0, 1.0, 1.0, 1.0]);
    }
}

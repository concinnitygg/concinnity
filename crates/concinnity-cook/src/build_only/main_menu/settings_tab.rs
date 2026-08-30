// The generated settings sub-screen: a tab bar, a scrollable band of rows, and
// a Back button, emitted as one Screen per tab.

use super::super::ui_spec::{centered_label, label_value, sprite};
use super::rows::{
    BodyRow, SettingsRow, option_select_row, settings_body_rows, settings_tabs, slider_row,
};
use super::{TOP_MARGIN_FRAC, cursor_sprite, opaque};
use crate::authoring::registry::build_only::MainMenu;
use crate::authoring::spec::{asset, spec_to_value};

// Average glyph advance as a fraction of the font pixel size, used to estimate
// a label's width when laying out the settings tab bar (the menu items and
// headings center with real font metrics via `align`; this only spaces the
// tab row). The built-in font is proportional, so this is an approximation.
const AVG_ADVANCE_RATIO: f32 = 0.5;

// A non-centered settings tab sizes its rows from the menu button width; a
// centered tab spans most of the window instead (see `SETTINGS_SIDE_MARGIN`).
const SETTINGS_ROW_WIDTH_MULT: f32 = 1.85;
// Settings rows use a smaller text scale than the menu buttons so the longer
// option names fit beside their controls.
const SETTINGS_ROW_SCALE: f32 = 0.6;
// A centered settings tab spans the window minus this side margin on each edge
// (plus the scrollbar gutter), in reference pixels, so the rows use most of the
// screen width instead of a narrow central column.
const SETTINGS_SIDE_MARGIN: f32 = 90.0;
// The interactive control (the value + steppers, or the slider track) sits in a
// fixed-width column anchored to the right of each row, so cycle, slider, and
// key-reference rows line up like a table. Mirrors `CONTROL_FRAC` /
// `MAX_CONTROL_WIDTH` in `crate::build_only::option_select` + `crate::build_only::slider`; keep the
// values in sync so all three column kinds align.
const SETTINGS_CONTROL_FRAC: f32 = 0.42;
const SETTINGS_CONTROL_WIDTH: f32 = 360.0;
// Per-row card backgrounds, drawn behind each settings row so the rows read as a
// table: a semi-transparent dark blue for normal rows and a slightly stronger
// fill behind group headers.
const SETTINGS_ROW_BG: [f32; 4] = [0.10, 0.13, 0.24, 0.55];
const SETTINGS_HEADER_BG: [f32; 4] = [0.16, 0.20, 0.34, 0.70];
// Horizontal padding inside each row card: the row content (name on the left,
// control on the right) is inset by this much from both card edges, so the text
// does not touch the edges and the left/right gaps match.
const SETTINGS_ROW_PAD: f32 = 18.0;

// How many setting rows the settings scroll band shows at once; a tab with more
// body rows than this scrolls (mouse wheel or scrollbar thumb). Sized so the
// band plus the tab bar and Back button all fit the reference canvas.
const VISIBLE_SETTINGS_ROWS: usize = 6;
// Scrollbar gutter width and its gap from the content band, in reference pixels.
const SCROLLBAR_GAP: f32 = 12.0;
const SCROLLBAR_WIDTH: f32 = 8.0;
// Gap between the scroll band's bottom and the Back button (fixed chrome below
// the band), in reference pixels.
const BACK_GAP: f32 = 24.0;

// Emit one settings tab as its own Screen: a "Settings" heading, the tab bar
// (this tab highlighted, the others clickable), the tab's setting rows, an
// optional read-only key reference, and a Back button. Each tab is a separate
// Screen so the active-tab highlight is baked in; switching tabs is a screen:show.
pub(super) fn emit_settings_tab(
    menu_name: &str,
    active: &str,
    style: &MainMenu,
    font: &str,
    win_w: f32,
    win_h: f32,
    font_px: f32,
) -> Vec<serde_json::Value> {
    let screen = format!("{}_settings_{}", menu_name, active);
    let mut out = Vec::new();

    out.push(spec_to_value(&asset::screen(&screen, false)));

    if style.dim[3] > 0.0 {
        out.push(sprite(
            &format!("{}_dim", screen),
            0.0,
            0.0,
            win_w,
            win_h,
            style.dim,
        ));
    }

    // Stacked from a fixed top margin: the tab bar, a scrollable body band, then
    // the Back button below the band. Top-aligned (not vertically centered) so
    // the tab bar holds position across tabs.
    let pitch = style.button_height + style.row_gap;
    let center_x = if style.centered { win_w / 2.0 } else { style.x };
    let start_y = if style.centered {
        win_h * TOP_MARGIN_FRAC
    } else {
        style.y
    };
    let row_y = |i: usize| start_y + i as f32 * pitch;
    let text_y = |i: usize, scale: f32| row_y(i) + (style.button_height - font_px * scale) / 2.0;

    let row_scale = style.text_scale * SETTINGS_ROW_SCALE;
    // A centered tab spans most of the width (leaving a side margin and room for
    // the scrollbar gutter); a non-centered menu keeps the narrower column form.
    let (row_x, row_width) = if style.centered {
        let w = win_w - 2.0 * SETTINGS_SIDE_MARGIN - SCROLLBAR_GAP - SCROLLBAR_WIDTH;
        (SETTINGS_SIDE_MARGIN, w)
    } else {
        let w = (style.button_width * SETTINGS_ROW_WIDTH_MULT).min(win_w - 80.0);
        (center_x - w / 2.0, w)
    };
    // The row content sits inside the card, inset by a uniform padding on both
    // sides so the name does not touch the left edge and the left/right gaps
    // match. The card background still spans the full row width.
    let content_x = row_x + SETTINGS_ROW_PAD;
    let content_w = (row_width - 2.0 * SETTINGS_ROW_PAD).max(0.0);
    // Left edge of the interactive control column, shared by the cycle, slider,
    // and key-reference rows so their controls line up. A fixed-width column
    // anchored to the right of the content area, falling back to a fraction of
    // the width if the row is too narrow (matches `crate::build_only::option_select` +
    // `crate::build_only::slider`).
    let control_x = (content_x + content_w * SETTINGS_CONTROL_FRAC)
        .max(content_x + content_w - SETTINGS_CONTROL_WIDTH);

    // Row 0: tab bar, laid out as a centered horizontal row. The active tab is
    // accent-colored with an underline marker and has no button (you are
    // already here); every other tab is a button that switches to its screen.
    // There is no "Settings" heading -- the tabs already name the screen, and
    // dropping it gives the body band an extra row of space.
    let tabs = settings_tabs(style.settings_profile);
    let tab_scale = style.text_scale * 1.1;
    let tab_text_y = text_y(0, tab_scale);
    let tab_gap = font_px * AVG_ADVANCE_RATIO * tab_scale * 1.2;
    let tab_widths: Vec<f32> = tabs
        .iter()
        .map(|&(_, label)| text_width(label, font_px, tab_scale))
        .collect();
    let tabs_total: f32 = tab_widths.iter().sum::<f32>() + tab_gap * (tabs.len() as f32 - 1.0);
    let mut tab_x = center_x - tabs_total / 2.0;
    for (&(suffix, label), w) in tabs.iter().zip(&tab_widths) {
        let is_active = suffix == active;
        let color = if is_active {
            style.hover_color
        } else {
            style.text_color
        };
        let label_name = format!("{}_tab_{}", screen, suffix);
        out.push(label_value(
            &label_name,
            label,
            font,
            tab_x,
            tab_text_y,
            color,
            tab_scale,
        ));
        if is_active {
            // Underline marker just below the active tab's text.
            let mark_h = (font_px * tab_scale * 0.08).max(2.0);
            out.push(sprite(
                &format!("{}_tabmark", screen),
                tab_x,
                tab_text_y + font_px * tab_scale + mark_h,
                *w,
                mark_h,
                opaque(style.hover_color),
            ));
        } else {
            out.push(spec_to_value(
                &asset::hit_region(
                    format!("{}_tabbtn_{}", screen, suffix),
                    [tab_x, row_y(0), *w, style.button_height],
                    format!("screen:show:{}_settings_{}", menu_name, suffix),
                )
                .set("label", label_name)
                .set("hover_color", style.hover_color)
                .set("hover_scale", tab_scale * style.hover_scale),
            ));
        }
        tab_x += w + tab_gap;
    }

    // Body band: the rows live in a fixed window starting just below the tab
    // bar. Rows past `VISIBLE_SETTINGS_ROWS` (or revealed by expanding a group)
    // overflow the band and scroll. `text_y_at` centers row text on an absolute
    // row top (the body rows are placed by base_y, not by chrome row index).
    let band_top = row_y(1);
    let band_h = VISIBLE_SETTINGS_ROWS as f32 * pitch;
    let text_y_at = |y: f32, scale: f32| y + (style.button_height - font_px * scale) / 2.0;

    let (body, groups) = settings_body_rows(active, style.settings_profile);

    // Emit each body row at its band-relative position, collecting a ScrollRow
    // (the row's reflowed/clipped element ids, its height, and its group).
    let mut scroll_rows: Vec<serde_json::Value> = Vec::new();
    for (j, row) in body.iter().enumerate() {
        let base_y = band_top + j as f32 * pitch;
        // A card background behind the row. Pushed (and so drawn) before the
        // row's content and listed first in the row's elements, so it sits behind
        // the row and reflows / clips / hides with it when scrolled or collapsed.
        let is_header = matches!(*row, BodyRow::GroupHeader(..));
        let bg_name = format!("{}_bg_{}", screen, j);
        out.push(sprite(
            &bg_name,
            row_x,
            base_y,
            row_width,
            style.button_height,
            if is_header {
                SETTINGS_HEADER_BG
            } else {
                SETTINGS_ROW_BG
            },
        ));
        let (mut elements, group): (Vec<String>, i32) = match *row {
            BodyRow::Option(setting, label, group) => {
                let name = format!("{}_opt_{}", screen, setting);
                out.push(option_select_row(&SettingsRow {
                    name: &name,
                    setting,
                    label,
                    font,
                    x: content_x,
                    y: base_y,
                    width: content_w,
                    scale: row_scale,
                    style,
                }));
                (
                    super::super::option_select::element_names(&name, setting),
                    group,
                )
            }
            BodyRow::Slider(setting, label, group) => {
                let name = format!("{}_sld_{}", screen, setting);
                out.push(slider_row(&SettingsRow {
                    name: &name,
                    setting,
                    label,
                    font,
                    x: content_x,
                    y: base_y,
                    width: content_w,
                    scale: row_scale,
                    style,
                }));
                (super::super::slider::element_names(&name), group)
            }
            BodyRow::Key(action_label, key, idx, group) => {
                let name = format!("{}_keyname_{}", screen, idx);
                let val = format!("{}_keyval_{}", screen, idx);
                out.push(label_value(
                    &name,
                    action_label,
                    font,
                    content_x,
                    text_y_at(base_y, row_scale),
                    style.text_color,
                    row_scale,
                ));
                out.push(label_value(
                    &val,
                    key,
                    font,
                    control_x,
                    text_y_at(base_y, row_scale),
                    style.text_color,
                    row_scale,
                ));
                (vec![name, val], group)
            }
            BodyRow::Rebind(action_label, setting, idx, group) => {
                let name = format!("{}_rebind_name_{}", screen, idx);
                let val = format!("{}_rebind_val_{}", screen, idx);
                out.push(label_value(
                    &name,
                    action_label,
                    font,
                    content_x,
                    text_y_at(base_y, row_scale),
                    style.text_color,
                    row_scale,
                ));
                // The value (the bound key) is a placeholder until the client
                // syncs it to the live key map at init.
                out.push(label_value(
                    &val,
                    "--",
                    font,
                    control_x,
                    text_y_at(base_y, row_scale),
                    style.text_color,
                    row_scale,
                ));
                // A HitRegion over the control column captures a new key on
                // click. Its `label` points at the value label so the client can
                // refresh it; the `setting:<key>:rebind` action is a scroll
                // content region, so it reflows / clips / gates with its row.
                let ctrl_w = (content_x + content_w - control_x).max(0.0);
                out.push(spec_to_value(
                    &asset::hit_region(
                        format!("{}_rebind_btn_{}", screen, idx),
                        [control_x, base_y, ctrl_w, style.button_height],
                        format!("setting:{}:rebind", setting),
                    )
                    .set("label", val.clone())
                    .set("hover_color", style.hover_color)
                    .set("hover_scale", row_scale * style.hover_scale),
                ));
                (vec![name, val], group)
            }
            BodyRow::GroupHeader(gid, title) => {
                let collapsed = groups.iter().any(|g| g.gid == gid && g.collapsed);
                let header = format!("{}_grphdr_{}", screen, gid);
                let header_scale = row_scale * 1.05;
                out.push(label_value(
                    &header,
                    &format!("{} {}", if collapsed { "+" } else { "-" }, title),
                    font,
                    content_x,
                    text_y_at(base_y, header_scale),
                    style.hover_color,
                    header_scale,
                ));
                out.push(spec_to_value(
                    &asset::hit_region(
                        format!("{}_grpbtn_{}", screen, gid),
                        [row_x, base_y, row_width, style.button_height],
                        format!("group:toggle:{}", gid),
                    )
                    .set("label", header.clone())
                    .set("hover_color", style.hover_color)
                    .set("hover_scale", header_scale * style.hover_scale),
                ));
                (vec![header], -1)
            }
        };
        // The card sits first so it reflows + clips + hides with the row.
        elements.insert(0, bg_name);
        scroll_rows.push(serde_json::json!({
            "elements": elements,
            "base_y": base_y,
            "height": pitch,
            "group": group,
        }));
    }

    // Scrollbar gutter (track + thumb) to the right of the band. The runtime
    // sizes + moves the thumb and hides both when the content fits.
    let track_x = row_x + row_width + SCROLLBAR_GAP;
    let track_tint = [
        style.text_color[0],
        style.text_color[1],
        style.text_color[2],
        0.25,
    ];
    out.push(sprite(
        &format!("{}_scrolltrack", screen),
        track_x,
        band_top,
        SCROLLBAR_WIDTH,
        band_h,
        track_tint,
    ));
    out.push(sprite(
        &format!("{}_scrollthumb", screen),
        track_x,
        band_top,
        SCROLLBAR_WIDTH,
        band_h * 0.4,
        opaque(style.hover_color),
    ));

    let scroll_groups: Vec<serde_json::Value> = groups
        .iter()
        .map(|g| {
            serde_json::json!({
                "collapsed": g.collapsed,
                "header": format!("{}_grphdr_{}", screen, g.gid),
                "title": g.title,
            })
        })
        .collect();
    out.push(serde_json::json!({
        "name": format!("{}_scroll", screen),
        "type": "ScrollPanel",
        "args": {
            "x": row_x, "y": band_top, "width": row_width, "height": band_h,
            "rows": scroll_rows,
            "groups": scroll_groups,
            "thumb": format!("{}_scrollthumb", screen),
            "track": format!("{}_scrolltrack", screen),
            "track_x": track_x, "track_y": band_top,
            "track_w": SCROLLBAR_WIDTH, "track_h": band_h,
        }
    }));

    // Back button: fixed chrome below the band. Returns to the menu screen by
    // default, or fires the menu's Back-action override (so a caller that owns
    // the settings navigation, e.g. a story, can route it).
    let back_y = band_top + band_h + BACK_GAP;
    let back_label = format!("{}_label_back", screen);
    let back_action = if style.settings_back_action.is_empty() {
        format!("screen:show:{}", menu_name)
    } else {
        style.settings_back_action.clone()
    };
    out.push(centered_label(
        &back_label,
        "Back",
        font,
        center_x,
        text_y_at(back_y, style.text_scale),
        style.text_color,
        style.text_scale,
    ));
    out.push(spec_to_value(
        &asset::hit_region(
            format!("{}_btn_back", screen),
            [
                center_x - style.button_width / 2.0,
                back_y,
                style.button_width,
                style.button_height,
            ],
            back_action,
        )
        .set("label", back_label)
        .set("hover_color", style.hover_color)
        .set("hover_scale", style.text_scale * style.hover_scale),
    ));

    if style.cursor {
        out.push(cursor_sprite(&format!("{}_cursor", screen), style));
    }

    out
}

// Estimated rendered width of `text`, from the average glyph advance. The
// built-in font is proportional, so this is an approximation good enough for
// centering and tab layout.
fn text_width(text: &str, font_px: f32, scale: f32) -> f32 {
    text.chars().count() as f32 * font_px * AVG_ADVANCE_RATIO * scale
}
